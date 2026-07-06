"""
Serves Qwen3-30B-A3B-Instruct-2507 via a vLLM OpenAI-compatible API, for a GPU box
you already have access to (rented or on-prem) instead of a Modal serverless container.

Run this directly on the remote GPU machine (over SSH, inside tmux/screen, or as a
systemd unit) -- it is a plain foreground process, not a cloud function.

Prerequisites on the remote machine:
    pip install "vllm>=0.6.0" huggingface_hub pydantic
    export HF_TOKEN=<your huggingface token>          # gated/rate-limited models

Usage:
    python serve_qwen_remote.py                       # binds 0.0.0.0:8000, full bf16 weights
    python serve_qwen_remote.py --port 8001 --gpu-memory-utilization 0.9

    # GPU too small for full bf16 (needs ~60GB)? Use a quantized checkpoint instead,
    # keeping the API-facing model name unchanged so dataset_config*.json files don't
    # need to change:
    python serve_qwen_remote.py \
        --model stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ \
        --quantization awq \
        --served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507

Then point a dataset_config_*.json's "base_url" at this machine, e.g.:
    - Same network / port opened in the firewall: "http://<remote-ip>:8000/v1"
    - Otherwise, tunnel instead of exposing the port publicly:
        ssh -N -L 8000:localhost:8000 <user>@<remote-host>
      and use "http://localhost:8000/v1" in the config.

This server has no authentication by default (matching the prior Modal endpoint's
client usage, which sends no API key). If the port is reachable from the public
internet, pass --api-key so random requests can't consume your GPU; then set the
same value in OPENAI_API_KEY when running scripts/generate_assets_openai.py.
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--quantization", default=None,
                         help="e.g. 'awq' when --model points at a pre-quantized checkpoint "
                              "(vLLM often auto-detects this from the checkpoint's config, "
                              "but passing it explicitly is safer)")
    parser.add_argument("--served-model-name", default=None,
                         help="API-facing model name clients must send. Set this to the "
                              "original repo id (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507) when "
                              "--model points at a different (e.g. quantized) repo, so existing "
                              "dataset_config*.json files don't need to change.")
    parser.add_argument("--api-key", default=None, help="Require this key via Authorization header")
    parser.add_argument("--hf-home", default=os.path.expanduser("~/.cache/huggingface"),
                         help="Where to cache downloaded model weights")
    parser.add_argument("--vllm-cache-root", default=os.path.expanduser("~/.cache/vllm"),
                         help="Where vLLM caches compiled kernels/graphs")
    parser.add_argument("--startup-timeout", type=int, default=1200,
                         help="Seconds to wait for the server to come up before giving up")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.hf_home, exist_ok=True)
    os.makedirs(args.vllm_cache_root, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # faster downloads (HF_HUB_ENABLE_HF_TRANSFER is deprecated)
    env["HF_HOME"] = args.hf_home
    env["VLLM_CACHE_ROOT"] = args.vllm_cache_root
    env.setdefault("VLLM_SKIP_P2P_CHECK", "1")

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
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.served_model_name:
        cmd += ["--served-model-name", args.served_model_name]
    if args.api_key:
        cmd += ["--api-key", args.api_key]

    print(f"Starting vLLM server for {args.model} on {args.host}:{args.port} ...")
    process = subprocess.Popen(cmd, env=env)

    def _shutdown(signum, frame):
        print("\nShutting down vLLM server...")
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Wait for the server to come up by polling the port, same behavior as the
    # Modal version's startup check.
    deadline = time.time() + args.startup_timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM process terminated unexpectedly (return code: {process.returncode})")
        try:
            with socket.create_connection((args.host if args.host != "0.0.0.0" else "127.0.0.1", args.port), timeout=1):
                print("|-|-| SERVER IS UP |-|-|")
                break
        except (OSError, ConnectionRefusedError):
            if time.time() > deadline:
                process.terminate()
                raise TimeoutError(f"vLLM server did not come up within {args.startup_timeout}s")
            time.sleep(5)

    # Keep running in the foreground until killed, streaming vLLM's own output.
    process.wait()
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
