# Modal app for serving Qwen3-30B-A3B model via vLLM OpenAI-compatible API
import os
import modal
import subprocess
import socket
import time

# Create the Modal application
app = modal.App("qwen3-30b-cot")

# Model identifier on HuggingFace Hub
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Define container image with vLLM and dependencies for efficient inference
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",  # High-performance LLM inference engine
        "hf-transfer",  # Fast HuggingFace downloads
        "huggingface_hub",
        "pydantic",
    )
)

# Persistent volumes for model and vLLM caches
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
    gpu="A100-80GB",  # Use high-memory GPU for large model
    timeout=3600,  # 1 hour timeout
    volumes={
        "/root/.cache/huggingface": model_volume,  # HuggingFace model cache
        "/root/.cache/vllm": vllm_cache_volume,  # vLLM internal cache
    },
    secrets=[modal.Secret.from_name("hf-secret")],  # HuggingFace authentication
    max_containers=1,  # Ensure single instance for consistent state
)
@modal.concurrent(max_inputs=100)  # Handle up to 100 concurrent requests
@modal.web_server(port=8000, startup_timeout=1200)  # Expose as web service

def serve():
    # Configure environment variables for optimal performance
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable fast downloads
    os.environ["HF_HOME"] = "/root/.cache/huggingface"  # Set cache location
    os.environ["VLLM_SKIP_P2P_CHECK"] = "1"  # Skip P2P validation
    os.environ["VLLM_CACHE_ROOT"] = "/root/.cache/vllm"  # vLLM cache location

    # Build vLLM server command with optimized settings
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0",  # Listen on all interfaces
        "--port", "8000",  # Standard OpenAI API port
        "--model", MODEL_NAME,
        "--dtype", "bfloat16",  # Use bfloat16 for efficiency
        "--max-model-len", "4096",  # Limit context length
        "--generation-config", "vllm",  # Use vLLM generation config
        
        "--gpu-memory-utilization", "0.95",  # Use 95% of GPU memory

        "--max-num-batched-tokens", "8192",  # Batch size optimization
        "--max-num-seqs", "256",  # Max concurrent sequences
        "--enable-chunked-prefill",  # Enable chunked prefill for long prompts

        "--trust-remote-code",  # Allow custom model code

        "--no-enable-log-requests",  # Reduce log verbosity
    ]
    
    # Start the vLLM server as a background process
    process = subprocess.Popen(cmd)
    
    # Wait for server to be ready by polling the port
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"Warning: vLLM process terminated unexpectedly (return code: {process.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                print("|-|-| SERVER IS UP |-|-|")
                break
        except (OSError, ConnectionRefusedError):
            time.sleep(5)