# Modal app for running GPT-2 expertise computation pipeline on cloud GPUs
import modal
import subprocess
import os

# Create the Modal application with a unique name
app = modal.App("gpt2-expertise-pipeline")

# Define persistent volumes for dataset, outputs, and HuggingFace cache
# These volumes persist data across container restarts
dataset_volume = modal.Volume.from_name("gpt2-dataset", create_if_missing=True)
output_volume = modal.Volume.from_name("gpt2-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("my-huggingface-cache", create_if_missing=True)

# Define the container image with required dependencies
gpt2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "pandas", "numpy", 
        "matplotlib", "scipy", "scikit-learn", "tqdm"
    )
    .add_local_dir(".", remote_path="/root")  # Mount current directory
    .add_local_dir("../selfcond", remote_path="/root/selfcond")  # Mount selfcond module
)

@app.function(
    image=gpt2_image,
    gpu="A100-80GB",  # Use NVIDIA A100 80GB GPU for large model inference
    timeout=72000,  # 20 hour timeout for long-running tasks
    volumes={
        "/root/assets": dataset_volume,  # Dataset files
        "/root/responses": output_volume,  # Generated responses output
        "/root/.cache/huggingface": hf_cache_volume,  # Model cache
    },
    secrets=[modal.Secret.from_name("hf-secret")],  # HuggingFace API token
)
def run_pipeline():
    # Execute the response generation script first
    print(">>> Starting compute_responses.py...")
    subprocess.run([
        "python", "/root/compute_responses.py",
        "--model-name-or-path", "gpt2",
        "--data-path", "assets/Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot",
        "--responses-path", "responses/GPT2_abstractiveness_Richie_HSJ_responses",

    ], check=True)

    # Then compute expertise metrics from the generated responses
    print("\n>>> Starting compute_expertise.py...")
    subprocess.run([
        "python", "/root/compute_expertise.py",
        "--root-dir", "responses/GPT2_abstractiveness_Richie_HSJ_responses",
        "--model-name", "gpt2",
        "--concepts", "assets/Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot/concept_list.csv",
    ], check=True)

    print("\n--- PIPELINE COMPLETED ---")

# Local entry point to trigger remote execution
@app.local_entrypoint()
def main():
    run_pipeline.remote()