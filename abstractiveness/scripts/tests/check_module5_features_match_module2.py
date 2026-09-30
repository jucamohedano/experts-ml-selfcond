"""
Module 5's self-computed label-word features equal module 2's concept-to-label table on one run.

Both come from the same helpers over the same item list, so jaccard and the default profile
agreement and z must agree exactly, on the same concept set. Both CSVs must come from the same
results tree, since older trees hold profiles on the flat layer axis.

Checks:
  1. The concept sets of the two tables coincide.
  2. jaccard_pct, layer-profile similarity and z agree to 1e-9, NaN in the same rows.

Usage, from scripts/:
    python tests/check_module5_features_match_module2.py [results_root] [ap_folder]
Reads the concept-to-label table from 2_similarities/2.1_* or, in trees written before the
merge, from 3_*, and the module 5 table from 5_typicality_prediction or, in trees written before
the renumbering, from 9_*. The default root is the GPT-2 Richie-HSJ smoke tree.
"""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _report import run

DEFAULT_ROOT = "../results/research_plots_gpt2_richie_hsj_smoke_m9"
DEFAULT_AP = "AP_0.5"
MODULE5_TABLES = ["5_typicality_prediction/typicality_model_design.csv",
                  "9_*/typicality_model_design.csv"]
CONCEPT_LABEL_TABLES = ["2_similarities/2.1_*/category_concept_similarity_metrics.csv",
                        "3_*/category_concept_similarity_metrics.csv"]
COLUMN_PAIRS = [("jaccard_pct", "jaccard_pct"),
                ("layer_profile_similarity_pct", "profile_js_distance"),
                ("layer_profile_z", "profile_js_distance_z")]
TOLERANCE = 1e-9


def checks(failures: list) -> str:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT)
    tree = root / (sys.argv[2] if len(sys.argv) > 2 else DEFAULT_AP)
    assert tree.is_dir(), f"missing results root {tree}"
    module2_csv = next((p for pattern in CONCEPT_LABEL_TABLES for p in tree.glob(pattern)), None)
    module5_csv = next((p for pattern in MODULE5_TABLES for p in tree.glob(pattern)), None)
    assert module2_csv and module5_csv, f"tables not found: module 2 {module2_csv}, module 5 {module5_csv}"

    m2, m5 = pd.read_csv(module2_csv), pd.read_csv(module5_csv)
    only_2 = sorted(set(m2["concept"]) - set(m5["concept"]))
    only_5 = sorted(set(m5["concept"]) - set(m2["concept"]))
    if only_2 or only_5:
        failures.append(f"concept sets differ, module 2 only {only_2}, module 5 only {only_5}")

    merged = m2.merge(m5, on="concept", suffixes=("_2", "_5"))
    assert not merged.empty, "no concepts in common"
    for left, right in COLUMN_PAIRS:
        left_name = left if left in merged.columns else f"{left}_2"
        right_name = right if right in merged.columns else f"{right}_5"
        if left_name not in merged.columns or right_name not in merged.columns:
            failures.append(f"missing column {left_name} or {right_name}")
            continue
        n_missing = int((merged[left_name].isna() != merged[right_name].isna()).sum())
        if n_missing:
            failures.append(f"{left_name} and {right_name} disagree on {n_missing} NaN rows")
        worst = (merged[left_name] - merged[right_name]).abs().max()
        if not worst < TOLERANCE:
            failures.append(f"{left_name} != {right_name}, max abs diff {worst}")
    return f"{len(merged)} concepts, {len(COLUMN_PAIRS)} column pairs exact"


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
