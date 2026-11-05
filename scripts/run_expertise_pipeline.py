"""
Run the expertise pipeline (compute_responses → compute_expertise) for the selected
model directly on a remote GPU server — no Modal, no cloud function boilerplate.

Select the model by commenting/uncommenting exactly one MODEL block below, and the
dataset (the prompt v3 sentences or their context-only copy) the same way. Only the
essential path/model arguments are passed, so everything else (batch size, device,
sequence length, ...) comes from the defaults set in compute_responses.py and
compute_expertise.py. Run from anywhere on the server:

    python scripts/run_expertise_pipeline.py

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

# Qwen3-1.7B-Base (pre-trained)
MODEL = "Qwen/Qwen3-1.7B-Base"
RUN   = "Qwen3_1.7Base_abstractiveness_Richie_HSJ_responses"

# GPT-2
# MODEL = "gpt2"
# RUN   = "GPT2_abstractiveness_Richie_HSJ_responses"

# --- Dataset: keep exactly one line uncommented. -----------------------------
# The context-only copy masks every concept word, built by
# `python -m core.data_preparation.build_context_only_dataset` from abstractiveness/scripts/.
DATASET, SUFFIX = "Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot_v3", "_v3"
# DATASET, SUFFIX = "Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot_v3_context_only", "_v3_context_only"

# Always a fresh folder: compute_responses skips concepts whose responses look complete.
DATA_PATH      = str(REPO_ROOT / "abstractiveness" / "assets" / DATASET)
RESPONSES_PATH = str(REPO_ROOT / "abstractiveness" / "responses" / (RUN + SUFFIX))
CONCEPTS_PATH  = str(pathlib.Path(DATA_PATH) / "concept_list.csv")

if __name__ == "__main__":
    print(f"Model {MODEL}\nData {DATA_PATH}\nResponses {RESPONSES_PATH}")
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
