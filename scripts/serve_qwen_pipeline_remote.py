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


DATA_PATH      = str(REPO_ROOT / "assets"    / "Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot")
RESPONSES_PATH = str(REPO_ROOT / "responses" / "Qwen3_1.7B_abstractiveness_Richie_HSJ_responses")
CONCEPTS_PATH  = str(REPO_ROOT / "assets"    / "Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot" / "concept_list.csv")
MODEL          = "Qwen/Qwen3-1.7B"

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(">>> Step 1/2 — compute_responses")
    print(f"{'='*60}")
    sys.argv = [
        "compute_responses.py",
        "--model-name-or-path", MODEL,
        "--data-path",          DATA_PATH,
        "--responses-path",     RESPONSES_PATH,
    ]
    runpy.run_path(str(SCRIPTS_DIR / "compute_responses.py"), run_name="__main__")

    print(f"\n{'='*60}")
    print(">>> Step 2/2 — compute_expertise")
    print(f"{'='*60}")
    sys.argv = [
        "compute_expertise.py",
        "--model-name", MODEL,
        "--root-dir",   RESPONSES_PATH,
        "--concepts",   CONCEPTS_PATH,
    ]
    runpy.run_path(str(SCRIPTS_DIR / "compute_expertise.py"), run_name="__main__")

    print("\n--- PIPELINE COMPLETED ---")
