"""The generated grid is the module's whole feature vocabulary.

Checks cell names, the metric column convention, nested-pair structure, and that
resolve_features maps every (cell, reference) to real design-frame column names
with no hand-written dictionary anywhere.
Run from scripts/: python tests/check_feature_grid.py.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.module_9_typicality_prediction import (
    build_feature_grid, nested_pairs, resolve_features, profile_columns, REFERENCES,
    study_a_columns, ACTIVE_METRICS, SUMMARY_METRIC, TARGET, _summary)

failures = []
grid = build_feature_grid(["js_distance"])

names = [(cell["model"], cell["metric"]) for cell in grid]
expected = [("jaccard", "none"), ("profile", "js_distance"), ("profile_z", "js_distance"),
            ("jaccard_profile", "js_distance"), ("jaccard_profile_z", "js_distance"),
            ("jaccard_profile_both", "js_distance")]
if names != expected:
    failures.append(f"grid cells {names} != {expected}")

# Two metrics: jaccard still appears exactly once, 5 profile cells per metric.
two = build_feature_grid(["js_distance", "hellinger"])
if sum(1 for cell in two if cell["model"] == "jaccard") != 1 or len(two) != 11:
    failures.append(f"two-metric grid has wrong shape, {len(two)} cells")

# nested_pairs guards against pairing one metric's cell with another's, which is
# trivially satisfied on a one-metric grid and so can only be tested here. A cross-metric
# pair would be a partial F test between two models that are not nested at all, which is
# not a well-defined comparison. The count is pinned too, so a later change cannot quietly
# drop or duplicate comparisons: 6 pairs per metric, jaccard being shared.
two_pairs = nested_pairs(two)
crossing = [(r["model"], r["metric"], f["model"], f["metric"])
            for r, f in two_pairs if r["metric"] not in ("none", f["metric"])]
if crossing:
    failures.append(f"nested pairs cross metrics: {crossing}")
if len(two_pairs) != 12:
    failures.append(f"two-metric grid yields {len(two_pairs)} nested pairs, expected 12")

pairs = nested_pairs(grid)
pair_names = {(r["model"], f["model"]) for r, f in pairs}
required = {("jaccard", "jaccard_profile"), ("jaccard", "jaccard_profile_z"),
            ("jaccard_profile", "jaccard_profile_both"),
            ("jaccard_profile_z", "jaccard_profile_both")}
if not required <= pair_names:
    failures.append(f"nested pairs missing {required - pair_names}")

cols = profile_columns("js_distance")
if cols != {"S": "profile_js_distance", "z": "profile_js_distance_z",
            "S_cen": "centroid_profile_js_distance",
            "z_cen": "centroid_profile_js_distance_z"}:
    failures.append(f"profile_columns wrong: {cols}")

# Exact column lists per (cell, reference), not their length. resolve_features is a
# comprehension over the cell's roles, so the count is right by construction and only the
# NAMES can catch a reference wired to the other reference's columns. That failure would
# fit the label-word feature under the centroid heading, or the reverse, and every number
# downstream would be attributed to the wrong reference while looking entirely healthy.
resolved_columns = {
    ("profile", "label_word"): [cols["S"]],
    ("profile", "member_centroid"): [cols["S_cen"]],
    ("profile_z", "label_word"): [cols["z"]],
    ("profile_z", "member_centroid"): [cols["z_cen"]],
    ("jaccard_profile", "label_word"): ["jaccard_pct", cols["S"]],
    ("jaccard_profile", "member_centroid"): ["mean_jaccard_to_members", cols["S_cen"]],
    ("jaccard_profile_z", "label_word"): ["jaccard_pct", cols["z"]],
    ("jaccard_profile_z", "member_centroid"): ["mean_jaccard_to_members", cols["z_cen"]],
    ("jaccard_profile_both", "label_word"): ["jaccard_pct", cols["S"], cols["z"]],
    ("jaccard_profile_both", "member_centroid"): ["mean_jaccard_to_members", cols["S_cen"],
                                                  cols["z_cen"]],
}
for (model, reference), wanted in resolved_columns.items():
    cell = next(c for c in grid if c["model"] == model)
    resolved = resolve_features(cell, reference, cols)
    if resolved != wanted:
        failures.append(f"{reference}: {model} resolved to {resolved} != {wanted}")
if resolve_features(grid[0], "label_word", cols) != ["jaccard_pct"]:
    failures.append("label_word jaccard cell must resolve to [jaccard_pct]")
if resolve_features(grid[0], "member_centroid", cols) != ["mean_jaccard_to_members"]:
    failures.append("member_centroid jaccard cell must resolve to [mean_jaccard_to_members]")
if set(REFERENCES) != {"label_word", "member_centroid"}:
    failures.append(f"REFERENCES keys {set(REFERENCES)}")


# --- Study A's identical-rows guarantee, the hand-written list against the generated grid --
# study_a_columns is written by hand while resolve_features is generated, and nothing else
# pins that the two agree. Study A drops its design frame over that list ONCE, so a
# disagreement is silent in both directions and asymmetric in consequence. An extra column
# OVER-GATES, dropping concept rows no cell ever needed and shrinking every accuracy in the
# study with no error raised anywhere, which is the defect that had to be corrected twice in
# Study B. A missing column UNDER-GATES, letting a cell meet NaN in a feature it does fit, so
# the cells stop being fitted on identical rows and stop being comparable. The two are
# reported separately because they are different faults.
def _grid_feature_columns(metrics: list) -> set:
    """Every design-frame column resolve_features can return over the grid and both references."""
    return {column
            for cell in build_feature_grid(metrics)
            for reference in REFERENCES
            for column in resolve_features(
                cell, reference,
                profile_columns(cell["metric"]) if cell["metric"] != "none" else {})}


# Checked on the live ACTIVE_METRICS, which is what the module actually runs, and on a
# six-metric list so registering the follow-up project's metrics cannot break the agreement
# without this check saying so.
for label, metric_list in (("ACTIVE_METRICS", list(ACTIVE_METRICS)),
                           ("six metrics", ["js_distance", "wasserstein", "cosine",
                                            "pearson", "spearman", "hellinger"])):
    declared = set(study_a_columns(metric_list)) - {TARGET}
    used = _grid_feature_columns(metric_list)
    over = sorted(declared - used)
    under = sorted(used - declared)
    if over:
        failures.append(f"study_a_columns OVER-GATES on {label}: {over} are dropped over but "
                        "no grid cell fits them, so Study A loses rows for nothing")
    if under:
        failures.append(f"study_a_columns UNDER-GATES on {label}: {under} are fitted by some "
                        "grid cell but not dropped over, so the cells see different rows")

# --- the summary row selects on (model, metric), not on position -------------------------
# _summary reports one number per column, so with several metrics active the model name alone
# names several rows. Selecting the first would make the reported metric whichever one the
# grid emitted first. The fixture puts a decoy metric AHEAD of SUMMARY_METRIC on purpose, so a
# positional selection returns the decoy's value and this check fails.
decoy = "decoy_metric_not_the_summary_one"
rank_fixture = pd.DataFrame([
    {"model": "jaccard", "metric": "none", "reference": "label_word",
     "accuracy": 0.10, "delta_accuracy_vs_jaccard": 0.0},
    {"model": "jaccard_profile_both", "metric": decoy, "reference": "label_word",
     "accuracy": 0.20, "delta_accuracy_vs_jaccard": 0.20},
    {"model": "jaccard_profile_both", "metric": SUMMARY_METRIC, "reference": "label_word",
     "accuracy": 0.30, "delta_accuracy_vs_jaccard": 0.30},
    {"model": "jaccard_profile_both", "metric": decoy, "reference": "member_centroid",
     "accuracy": 0.40, "delta_accuracy_vs_jaccard": 0.40},
    {"model": "jaccard_profile_both", "metric": SUMMARY_METRIC, "reference": "member_centroid",
     "accuracy": 0.50, "delta_accuracy_vs_jaccard": 0.50},
])
pair_fixture = pd.DataFrame([
    {"model": "jaccard", "metric": "none", "mean_category_spearman": 0.11},
    {"model": "jaccard_profile_both", "metric": decoy, "mean_category_spearman": 0.22},
    {"model": "jaccard_profile_both", "metric": SUMMARY_METRIC, "mean_category_spearman": 0.33},
])
summary = _summary(rank_fixture, pair_fixture)
expected_summary = {"rank_acc_jaccard": 0.10, "rank_acc_full_grid": 0.30,
                    "rank_acc_delta_profile": 0.30, "rank_acc_centroid_full_grid": 0.50,
                    "pair_sim_rho_jaccard": 0.11, "pair_sim_rho_full_grid": 0.33}
for key, wanted in expected_summary.items():
    got = summary.get(key)
    if got != wanted:
        failures.append(f"_summary[{key}] is {got}, expected {wanted}. The summary row must "
                        f"select on (model, metric) and report {SUMMARY_METRIC}, not the "
                        "first row carrying the model name")

if failures:
    print("FAIL check_feature_grid")
    for line in failures:
        print("  " + line)
    sys.exit(1)
print("OK check_feature_grid")
