"""
Run the Qwen3-1.7B expertise pipeline (compute_responses → compute_expertise) directly
on a remote GPU server — no Modal, no cloud function boilerplate.

Run this over SSH (inside tmux/screen or as a systemd unit) from the scripts/ directory:

    python serve_qwen_pipeline_remote.py

Or with custom arguments:

    python serve_qwen_pipeline_remote.py \
        --model Qwen/Qwen3-1.7B \
        --responses-path ../responses/Qwen3_1.7B_abstractiveness_150_responses \
        --device cuda \
        --batch-size 16

Prerequisites on the remote machine:
    pip install torch transformers pandas numpy matplotlib scipy scikit-learn tqdm
    export HF_TOKEN=<your huggingface token>  # only needed for gated models
"""
import argparse
import pathlib
import subprocess
import sys

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

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
                        help="concept_list.csv or comma-separated 'group/concept' pairs; "
                             "leave blank to use concept_list.csv inside --data-path")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda, cpu, cuda:0, ...)")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size for compute_responses")
    parser.add_argument("--seq-len", type=int, default=1024, help="Max token sequence length")
    parser.add_argument("--num-per-concept", type=int, default=1000,
                        help="Max sentences per concept per label")
    return parser.parse_args()


def run(cmd: list, label: str) -> None:
    print(f"\n{'='*60}")
    print(f">>> {label}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    run([
        sys.executable, str(SCRIPTS_DIR / "compute_responses.py"),
        "--model-name-or-path", args.model,
        "--data-path", args.data_path,
        "--responses-path", args.responses_path,
        "--concepts", args.concepts,
        "--device", args.device,
        "--inf-batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--num-per-concept", str(args.num_per_concept),
    ], label="Step 1/2 — compute_responses.py")

    run([
        sys.executable, str(SCRIPTS_DIR / "compute_expertise.py"),
        "--root-dir", args.responses_path,
        "--model-name", args.model,
        "--concepts", args.concepts,
    ], label="Step 2/2 — compute_expertise.py")

    print("\n--- PIPELINE COMPLETED ---")


if __name__ == "__main__":
    main()
