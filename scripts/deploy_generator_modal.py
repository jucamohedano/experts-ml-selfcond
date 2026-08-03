"""
Deploys the sentence generator LM (Qwen3-30B-A3B-Instruct-2507) on Modal, behind a
vLLM OpenAI-compatible API.

This is the endpoint `generate_definitions_dspy.py` calls to write the stimulus
sentences, and the only paid component of the pipeline. It is not one of the models
under study, those are GPT-2 and Qwen3-1.7B and they run locally through
`run_expertise_pipeline.py`.

Usage:
    modal deploy scripts/deploy_generator_modal.py     # persistent, survives disconnect
    modal serve scripts/deploy_generator_modal.py      # ephemeral, dies with the terminal

The public URL is built by Modal from the app name and the function name, as
`https://<workspace>--<app>-<function>.modal.run`, giving

    https://<workspace>--qwen3-30b-cot-serve.modal.run

so renaming `app` or `serve` below changes the URL and breaks the `base_url` field of
every `dataset_config_*.json`. A 404 reading `modal-http: invalid function call` at that
URL means the app is simply not deployed.

Prerequisites:
    pip install modal && modal setup
    modal secret create hf-secret HF_TOKEN=<your huggingface token>

The container serves without authentication, matching the client, which sends the
placeholder key. Access control comes from the URL being unguessable rather than from a
key, so treat the URL as the credential.

To serve the same model on hardware you already have, use `serve_generator_vllm.py`,
which takes the same vLLM settings as command line flags.
"""
import os
import socket
import subprocess
import time

import modal

APP_NAME = "qwen3-30b-cot"          # part of the public URL, see the note above
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PORT = 8000

# vLLM serving knobs, kept identical to the defaults in serve_generator_vllm.py so the
# two deployments produce comparable generations.
DTYPE = "bfloat16"
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.95
MAX_NUM_BATCHED_TOKENS = 8192
MAX_NUM_SEQS = 256

GPU = "A100-80GB"
STARTUP_TIMEOUT = 1200              # seconds to wait for vLLM to accept connections
SCALEDOWN_WINDOW = 1200             # stay warm 20 minutes, the maximum Modal allows, so
                                    # back-to-back generation runs a few minutes apart do
                                    # not each pay the roughly 2 minute cold start
FUNCTION_TIMEOUT = 3600
MAX_CONCURRENT_REQUESTS = 100

app = modal.App(APP_NAME)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",          # inference engine
        "hf-transfer",          # fast weight downloads
        "huggingface_hub",
        "pydantic",
    )
)

# Persistent across container restarts, so the 60 GB of weights is downloaded once and
# vLLM's compiled kernels survive a cold start.
model_volume = modal.Volume.from_name("my-huggingface-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("my-vllm-cache", create_if_missing=True)


def build_vllm_command(host: str, port: int) -> list:
    """The vLLM server command. Kept in sync with serve_generator_vllm.py, which builds
    the same list from command line flags."""
    return [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", host,
        "--port", str(port),
        "--model", MODEL_NAME,
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--generation-config", "vllm",
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--enable-chunked-prefill",
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]


def wait_for_server(process: subprocess.Popen, port: int, timeout: int) -> None:
    """Blocks until vLLM accepts connections, and fails loudly rather than hanging when
    the process dies during weight loading or the load simply takes too long."""
    deadline = time.time() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM process terminated unexpectedly (return code: {process.returncode})"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                print("|-|-| SERVER IS UP |-|-|")
                return
        except (OSError, ConnectionRefusedError):
            if time.time() > deadline:
                process.terminate()
                raise TimeoutError(f"vLLM server did not come up within {timeout}s")
            time.sleep(5)


@app.function(
    image=vllm_image,
    gpu=GPU,
    timeout=FUNCTION_TIMEOUT,
    volumes={
        "/root/.cache/huggingface": model_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    max_containers=1,               # one instance, so all requests share a warm model
    scaledown_window=SCALEDOWN_WINDOW,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_REQUESTS)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT)
def serve():                        # part of the public URL, see the note above
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"   # matches the hf-transfer install above
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    os.environ["VLLM_CACHE_ROOT"] = "/root/.cache/vllm"
    os.environ["VLLM_SKIP_P2P_CHECK"] = "1"

    print(f"Starting vLLM server for {MODEL_NAME} on 0.0.0.0:{PORT} ...")
    process = subprocess.Popen(build_vllm_command("0.0.0.0", PORT))
    wait_for_server(process, PORT, STARTUP_TIMEOUT)
