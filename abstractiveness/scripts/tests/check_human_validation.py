"""Checks the human-similarity validation on one real scope. Exits non-zero on failure."""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modules.module_5_heatmaps import validate_against_human, profile_matrix_key
from utils.helpers import DEFAULT_PROFILE_METRIC
from utils.human_similarity import human_noise_ceiling

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MATRIX_DIR = (REPO_ROOT / "abstractiveness/results/research_plots_qwen_richie_hsj_sensefix"
              / "AP_0.5/5_heatmaps")

# The default metric's matrix, under whichever name the tree on disk uses. Trees written
# before the multi-metric extension call it layer_profile_matrix.csv, later ones name every
# matrix after its metric. The file is the same quantity either way, so accepting both lets
# this check run against an existing baseline tree without regenerating it.
PROFILE_MATRIX_NAMES = [f"{profile_matrix_key(DEFAULT_PROFILE_METRIC)}_matrix.csv",
                        "layer_profile_matrix.csv"]


def main() -> int:
    jaccard = pd.read_csv(MATRIX_DIR / "jaccard_matrix.csv", index_col=0)
    profile_path = next((MATRIX_DIR / name for name in PROFILE_MATRIX_NAMES
                         if (MATRIX_DIR / name).exists()), None)
    if profile_path is None:
        print(f"FAIL check_human_validation\n  no profile matrix in {MATRIX_DIR}, "
              f"tried {PROFILE_MATRIX_NAMES}")
        return 1
    profile = pd.read_csv(profile_path, index_col=0)
    concepts = list(jaccard.index)
    matrices = {"jaccard": jaccard.to_numpy(dtype=float),
                profile_matrix_key(DEFAULT_PROFILE_METRIC): profile.to_numpy(dtype=float)}

    out_dir = pathlib.Path("/tmp/check_human_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = validate_against_human(concepts, matrices, out_dir, make_plots=False)

    ceiling = human_noise_ceiling()
    per_category = table[~table.category.isin(["POOLED", "POOLED_MEAN"])]
    assert set(per_category.category.unique()) == set(ceiling), \
        "categories do not match the ceiling keys"
    assert len(table[table.metric == "jaccard"]) == len(ceiling) + 2, \
        "expected one row per category plus two pooled rows per metric"

    assert per_category.n_pairs.sum() == 2391 * len(matrices), "pair total is not 2391 per metric"
    assert per_category.rho.between(-1, 1).all(), "rho outside [-1, 1]"
    assert per_category.mantel_p.between(0, 1).all(), "mantel_p outside [0, 1]"
    assert (per_category.rho_over_ceiling.abs() >= per_category.rho.abs() - 1e-9).all(), \
        "dividing by a ceiling below 1 must not shrink the magnitude"
    assert (out_dir / "human_similarity_validation.csv").exists(), "CSV not written"

    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
