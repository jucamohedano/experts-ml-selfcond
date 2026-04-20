import os
import modal
import subprocess
import socket
import time

app = modal.App("qwen3-30b-cot")

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "hf-transfer",
        "huggingface_hub",
        "pydantic",
    )
)

model_volume = modal.Volume.from_name(
    "my-huggingface-cache",
    create_if_missing=True,
)
vllm_cache_volume = modal.Volume.from_name(
    "my-vllm-cache",
    create_if_missing=True,
)

@app.function(
    image=vllm_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={
        "/root/.cache/huggingface": model_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8000, startup_timeout=1200)

def serve():

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    os.environ["VLLM_SKIP_P2P_CHECK"] = "1"
    os.environ["VLLM_CACHE_ROOT"] = "/root/.cache/vllm"

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--model", MODEL_NAME,
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--generation-config", "vllm",
        
        "--gpu-memory-utilization", "0.95", 

        "--max-num-batched-tokens", "8192",
        "--max-num-seqs", "256",
        "--enable-chunked-prefill",

        "--trust-remote-code",

        "--no-enable-log-requests",
    ]
    
    process = subprocess.Popen(cmd)
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"Warning: vLLM process terminated unexpectedly (return code: {process.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                print("|-|-| SERVER IS UP |-|-|")
                break
        except (OSError, ConnectionRefusedError):
            time.sleep(5)