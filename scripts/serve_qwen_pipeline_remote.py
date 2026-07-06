"""
Run the Qwen3-1.7B expertise pipeline (compute_responses → compute_expertise) directly
on a remote GPU server — no Modal, no cloud function boilerplate.

Run from anywhere on the server:

    python scripts/serve_qwen_pipeline_remote.py

Or with custom arguments:

    python scripts/serve_qwen_pipeline_remote.py \
        --model Qwen/Qwen3-1.7B \
        --device cuda \
        --batch-size 16

Prerequisites on the remote machine:
    pip install torch transformers pandas numpy matplotlib scipy scikit-learn tqdm
    export HF_TOKEN=<your huggingface token>  # only needed for gated models
"""
import argparse
import pathlib
import sys

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

# Make selfcond (repo root) and the compute_* scripts importable from anywhere.
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from compute_responses import run_response_computation
from compute_expertise import run_expertise_computation

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_DATA_PATH = str(REPO_ROOT / "assets" / "Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot")
DEFAULT_RESPONSES_PATH = str(REPO_ROOT / "responses" / "Qwen3_1.7B_abstractiveness_Richie_HSJ_responses")
DEFAULT_CONCEPTS = str(REPO_ROOT / "assets" / "Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot" / "concept_list.csv")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                        help="Path to the concept dataset (concept JSON files + concept_list.csv)")
    parser.add_argument("--responses-path", default=DEFAULT_RESPONSES_PATH,
                        help="Directory where intermediate model responses are cached")
    parser.add_argument("--concepts", default=DEFAULT_CONCEPTS,
                        help="Path to concept_list.csv, or comma-separated 'group/concept' pairs")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda, cpu, cuda:0, ...)")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size for compute_responses")
    parser.add_argument("--seq-len", type=int, default=1024, help="Max token sequence length")
    parser.add_argument("--num-per-concept", type=int, default=1000,
                        help="Max sentences per concept per label")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(">>> Step 1/2 — compute_responses")
    print(f"{'='*60}")
    run_response_computation(
        model_name_or_path=args.model,
        data_path=pathlib.Path(args.data_path),
        responses_path=pathlib.Path(args.responses_path),
        concepts=args.concepts,
        seq_len=args.seq_len,
        num_per_concept=args.num_per_concept,
        inf_batch_size=args.batch_size,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print(">>> Step 2/2 — compute_expertise")
    print(f"{'='*60}")
    run_expertise_computation(
        root_dir=pathlib.Path(args.responses_path),
        model_name=args.model,
        concepts=args.concepts,
    )

    print("\n--- PIPELINE COMPLETED ---")


if __name__ == "__main__":
    main()
