"""
Run the Qwen3-1.7B expertise pipeline (compute_responses → compute_expertise) directly
on a remote GPU server — no Modal, no cloud function boilerplate.

All parameters are taken from the defaults set in compute_responses.py and
compute_expertise.py. Run from anywhere on the server:

    python scripts/serve_qwen_pipeline_remote.py

Prerequisites on the remote machine:
    pip install torch transformers pandas numpy matplotlib scipy scikit-learn tqdm
    export HF_TOKEN=<your huggingface token>  # only needed for gated models
"""
import pathlib
import runpy
import sys

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

# Make selfcond importable from anywhere.
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(">>> Step 1/2 — compute_responses")
    print(f"{'='*60}")
    runpy.run_path(str(SCRIPTS_DIR / "compute_responses.py"), run_name="__main__")

    print(f"\n{'='*60}")
    print(">>> Step 2/2 — compute_expertise")
    print(f"{'='*60}")
    runpy.run_path(str(SCRIPTS_DIR / "compute_expertise.py"), run_name="__main__")

    print("\n--- PIPELINE COMPLETED ---")
