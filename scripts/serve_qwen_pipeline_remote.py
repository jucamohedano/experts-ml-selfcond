"""
Run the expertise pipeline (compute_responses → compute_expertise) for the selected
model directly on a remote GPU server — no Modal, no cloud function boilerplate.

Select the model by commenting/uncommenting exactly one MODEL block below; the
dataset is the same Richie-HSJ set for every model. Only the essential path/model
arguments are passed, so everything else (batch size, device, sequence length, ...)
comes from the defaults set in compute_responses.py and compute_expertise.py.
Run from anywhere on the server:

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


# --- Model selection: keep exactly one block uncommented. -------------------

# Qwen3-1.7B
MODEL          = "Qwen/Qwen3-1.7B"
RESPONSES_PATH = str(REPO_ROOT / "abstractiveness" / "responses" / "Qwen3_1.7B_abstractiveness_Richie_HSJ_responses")

# GPT-2
# MODEL          = "gpt2"
# RESPONSES_PATH = str(REPO_ROOT / "abstractiveness" / "responses" / "GPT2_abstractiveness_Richie_HSJ_responses")

# --- Dataset (shared by all models). -----------------------------------------
DATA_PATH      = str(REPO_ROOT / "abstractiveness" / "assets" / "Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot")
CONCEPTS_PATH  = str(pathlib.Path(DATA_PATH) / "concept_list.csv")

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
