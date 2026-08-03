"""
Serves the sentence generator LM (Qwen3-30B-A3B-Instruct-2507) via a vLLM
OpenAI-compatible API, on a GPU box you already have instead of a Modal container.

This is the endpoint `generate_definitions_dspy.py` calls to write the stimulus
sentences. It is not one of the models under study, those are GPT-2 and Qwen3-1.7B and
they run locally through `run_expertise_pipeline.py`.

Run it directly on the GPU machine, over SSH inside tmux or screen, or as a systemd
unit. It is a plain foreground process, not a cloud function.

Prerequisites:
    pip install "vllm>=0.6.0" huggingface_hub pydantic
    export HF_TOKEN=<your huggingface token>          # gated or rate-limited models

Usage:
    python serve_generator_vllm.py                    # binds 0.0.0.0:8000, full bf16 weights
    python serve_generator_vllm.py --port 8001 --gpu-memory-utilization 0.9

    # GPU too small for full bf16 (needs about 60 GB)? Use a quantized checkpoint,
    # keeping the API-facing model name unchanged so dataset_config*.json files do not
    # need to change:
    python serve_generator_vllm.py \
        --model stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ \
        --quantization awq \
        --served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507

Then point a dataset_config_*.json's "base_url" at this machine:
    - Same network, port open in the firewall: "http://<remote-ip>:8000/v1"
    - Otherwise tunnel rather than exposing the port publicly:
        ssh -N -L 8000:localhost:8000 <user>@<remote-host>
      and use "http://localhost:8000/v1" in the config.

The server has no authentication by default, matching the Modal deployment, whose
client sends a placeholder key. If the port is reachable from the public internet, pass
--api-key so random requests cannot consume your GPU, and set the same value in
VLLM_API_KEY when running the generation scripts.

To deploy the same model on Modal instead, use `deploy_generator_modal.py`, which holds
the same vLLM settings as module constants.
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PORT = 8000

# vLLM serving knobs, kept identical to the constants in deploy_generator_modal.py so the
# two deployments produce comparable generations.
DTYPE = "bfloat16"
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.95
MAX_NUM_BATCHED_TOKENS = 8192
MAX_NUM_SEQS = 256

STARTUP_TIMEOUT = 1200              # seconds to wait for vLLM to accept connections
SHUTDOWN_GRACE = 30                 # seconds to wait after SIGTERM before killing


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model id")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind")
    parser.add_argument("--port", type=int, default=PORT, help="Port to serve on")
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-num-batched-tokens", type=int, default=MAX_NUM_BATCHED_TOKENS)
    parser.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    parser.add_argument("--quantization", default=None,
                        help="e.g. 'awq' when --model points at a pre-quantized checkpoint "
                             "(vLLM often auto-detects this from the checkpoint's config, "
                             "but passing it explicitly is safer)")
    parser.add_argument("--served-model-name", default=None,
                        help="API-facing model name clients must send. Set this to the "
                             "original repo id (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507) when "
                             "--model points at a different (e.g. quantized) repo, so existing "
                             "dataset_config*.json files do not need to change.")
    parser.add_argument("--api-key", default=None, help="Require this key via Authorization header")
    parser.add_argument("--hf-home", default=os.path.expanduser("~/.cache/huggingface"),
                        help="Where to cache downloaded model weights")
    parser.add_argument("--vllm-cache-root", default=os.path.expanduser("~/.cache/vllm"),
                        help="Where vLLM caches compiled kernels and graphs")
    parser.add_argument("--startup-timeout", type=int, default=STARTUP_TIMEOUT,
                        help="Seconds to wait for the server to come up before giving up")
    return parser.parse_args()


def build_vllm_command(args) -> list:
    """The vLLM server command. Kept in sync with deploy_generator_modal.py, which builds
    the same list from module constants."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", args.host,
        "--port", str(args.port),
        "--model", args.model,
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--generation-config", "vllm",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-num-seqs", str(args.max_num_seqs),
        "--enable-chunked-prefill",
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    # Flags with no counterpart in the Modal deployment, which always serves the full
    # bf16 checkpoint under its own name and without authentication.
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.served_model_name:
        cmd += ["--served-model-name", args.served_model_name]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    return cmd


def wait_for_server(process: subprocess.Popen, host: str, port: int, timeout: int) -> None:
    """Blocks until vLLM accepts connections, and fails loudly rather than hanging when
    the process dies during weight loading or the load simply takes too long."""
    probe_host = "127.0.0.1" if host == "0.0.0.0" else host
    deadline = time.time() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM process terminated unexpectedly (return code: {process.returncode})"
            )
        try:
            with socket.create_connection((probe_host, port), timeout=1):
                print("|-|-| SERVER IS UP |-|-|")
                return
        except (OSError, ConnectionRefusedError):
            if time.time() > deadline:
                process.terminate()
                raise TimeoutError(f"vLLM server did not come up within {timeout}s")
            time.sleep(5)


def install_shutdown_handlers(process: subprocess.Popen) -> None:
    """Ctrl-C and SIGTERM must take the vLLM child down with us, otherwise it keeps the
    GPU allocated after this wrapper exits."""
    def _shutdown(signum, frame):
        print("\nShutting down vLLM server...")
        process.terminate()
        try:
            process.wait(timeout=SHUTDOWN_GRACE)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


def main():
    args = parse_args()

    os.makedirs(args.hf_home, exist_ok=True)
    os.makedirs(args.vllm_cache_root, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # faster downloads, HF_HUB_ENABLE_HF_TRANSFER is deprecated
    env["HF_HOME"] = args.hf_home
    env["VLLM_CACHE_ROOT"] = args.vllm_cache_root
    env.setdefault("VLLM_SKIP_P2P_CHECK", "1")

    print(f"Starting vLLM server for {args.model} on {args.host}:{args.port} ...")
    process = subprocess.Popen(build_vllm_command(args), env=env)
    install_shutdown_handlers(process)
    wait_for_server(process, args.host, args.port, args.startup_timeout)

    # Stay in the foreground, streaming vLLM's own output, until killed.
    process.wait()
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
