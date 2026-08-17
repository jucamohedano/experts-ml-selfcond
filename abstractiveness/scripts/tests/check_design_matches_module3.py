"""Module 9's self-computed label-word columns equal module 3's CSV on the same run.

The restructure made module 9 self-sufficient, it derives jaccard_pct and the layer-profile
columns from the expert frame itself rather than reading module 3's table. The claim that
buys is that the two agree by construction, because both go through the same helpers over
the same item list. This script is the evidence for that claim on real data, joining the
two CSVs on concept and demanding exact agreement of the three shared columns.

Both CSVs MUST come from the SAME results root. Module 9's design frame lives on the BLOCK
layer axis, sublayers summed within each transformer block, and any results tree written
before the block-axis change holds module 3's profile columns on the FLAT layer axis. On
GPT-2 at AP 0.5 the two axes differ by about 23 points on the similarity scale, so reading
module 3 from an older tree would report a large difference that says nothing about this
module. The single path argument below makes that mistake hard to commit by accident.

Usage, from scripts/:
    python tests/check_design_matches_module3.py [results_root] [ap_folder]
The default points at the GPT-2 Richie-HSJ smoke tree. Passing the Qwen3 root runs the
same check on that architecture without editing this file.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

DEFAULT_ROOT = "../results/research_plots_gpt2_richie_hsj_smoke_m9"
DEFAULT_AP = "AP_0.5"

# Module 3 column against module 9 column. Module 9 names its profile columns after the
# registered metric, and js_distance is the default and only registered metric today, so
# these are the columns module 3 has always written under its own older names.
COLUMN_PAIRS = [("jaccard_pct", "jaccard_pct"),
                ("layer_profile_similarity_pct", "profile_js_distance"),
                ("layer_profile_z", "profile_js_distance_z")]

# The two frames come from the same helpers on the same inputs, so agreement is exact
# rather than approximate. The tolerance guards only against a formatting round trip.
TOLERANCE = 1e-9


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT)
    ap_folder = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_AP
    S = root / ap_folder
    if not S.is_dir():
        print(f"FAIL check_design_matches_module3\n  missing results root {S}")
        return 1

    module3_csv = next(S.glob("3_*/category_concept_similarity_metrics.csv"), None)
    module9_csv = next(S.glob("9_*/typicality_model_design.csv"), None)
    if module3_csv is None or module9_csv is None:
        print("FAIL check_design_matches_module3")
        print(f"  module 3 CSV {module3_csv}, module 9 CSV {module9_csv}")
        return 1

    m3 = pd.read_csv(module3_csv)
    m9 = pd.read_csv(module9_csv)
    failures = []

    # The row sets must coincide too, module 9 gates its rows on exactly module 3's
    # condition, so a concept in one and not the other would mean the gates have drifted.
    only_3 = sorted(set(m3["concept"]) - set(m9["concept"]))
    only_9 = sorted(set(m9["concept"]) - set(m3["concept"]))
    if only_3 or only_9:
        failures.append(f"concept sets differ, module 3 only {only_3}, module 9 only {only_9}")

    merged = m3.merge(m9, on="concept", suffixes=("_3", "_9"))
    if merged.empty:
        failures.append("no concepts in common")
    for left, right in COLUMN_PAIRS:
        left_name = left if left in merged.columns else f"{left}_3"
        right_name = right if right in merged.columns else f"{right}_9"
        if left_name not in merged.columns or right_name not in merged.columns:
            failures.append(f"missing column {left_name} or {right_name}")
            continue
        worst = (merged[left_name] - merged[right_name]).abs().max()
        n_missing = int((merged[left_name].isna() != merged[right_name].isna()).sum())
        if n_missing:
            failures.append(f"{left_name} and {right_name} disagree on {n_missing} NaN rows")
        if not worst < TOLERANCE:
            failures.append(f"{left_name} != {right_name}, max abs diff {worst}")

    if failures:
        print("FAIL check_design_matches_module3")
        for line in failures:
            print("  " + line)
        return 1
    print(f"OK check_design_matches_module3, {len(merged)} concepts, "
          f"{len(COLUMN_PAIRS)} column pairs exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
