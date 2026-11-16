"""
Module 5's generated feature grid is its whole feature vocabulary, wired to the right columns.

Checks:
  1. Cell names and order on one metric, the four metric-free cells (jaccard, jaccard_bands, text,
     jaccard_text) once and first, five cells per metric, 14 cells on two metrics, 12 without text.
  2. Nested pairs never cross metrics, 15 pairs on two metrics, the required nestings present.
  3. profile_columns naming, and resolve_features and resolve_pair_features returning the exact
     columns per (cell, reference), the band role giving its four columns, so neither reference can
     be wired to the other's columns.
  4. study_a_columns neither over-gates nor under-gates against the generated grid, on the active
     metrics and on six metrics, with and without text.
  5. The summary row selects on (model, metric), not on the first row carrying the model name, reports {J, S}
     with SUMMARY_METRIC and {J, J_1..J_4}, and labels every key.

Usage, from scripts/:
    python tests/check_module5_feature_grid.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.module_5_typicality_prediction import (
    build_feature_grid, nested_pairs, resolve_features, resolve_pair_features, profile_columns, REFERENCES,
    study_a_columns, ACTIVE_METRICS, SUMMARY_METRIC, SUMMARY_LABELS, TARGET, _summary, BAND_COLUMNS,
    MEMBER_BAND_COLUMNS)
from _report import run

SIX_METRICS = ["js_distance", "wasserstein", "cosine", "pearson", "spearman", "hellinger"]


def grid_feature_columns(metrics: list, with_text: bool = True) -> set:
    """Every design-frame column resolve_features can return over the grid and both references."""
    return {column
            for cell in build_feature_grid(metrics, with_text=with_text)
            for reference in REFERENCES
            for column in resolve_features(
                cell, reference, profile_columns(cell["metric"]) if cell["metric"] != "none" else {})}


def checks(failures: list) -> str:
    grid = build_feature_grid(["js_distance"])
    names = [(cell["model"], cell["metric"]) for cell in grid]
    expected = [("jaccard", "none"), ("jaccard_bands", "none"), ("text", "none"), ("jaccard_text", "none"),
                ("profile", "js_distance"), ("profile_z", "js_distance"),
                ("jaccard_profile", "js_distance"), ("jaccard_profile_z", "js_distance"),
                ("jaccard_profile_both", "js_distance")]
    if names != expected:
        failures.append(f"grid cells {names} != {expected}")
    no_text = build_feature_grid(["js_distance"], with_text=False)
    if len(no_text) != 7 or any("s" in cell["roles"] for cell in no_text):
        failures.append(f"grid without text has {len(no_text)} cells or a text role")

    two = build_feature_grid(["js_distance", "hellinger"])
    if sum(1 for cell in two if cell["model"] == "jaccard") != 1 or len(two) != 14:
        failures.append(f"two-metric grid has wrong shape, {len(two)} cells")
    if len(build_feature_grid(["js_distance", "hellinger"], with_text=False)) != 12:
        failures.append("two-metric grid without text must have 12 cells")
    two_pairs = nested_pairs(two)
    crossing = [(r["model"], r["metric"], f["model"], f["metric"])
                for r, f in two_pairs if r["metric"] not in ("none", f["metric"])]
    if crossing:
        failures.append(f"nested pairs cross metrics: {crossing}")
    if len(two_pairs) != 15:
        failures.append(f"two-metric grid yields {len(two_pairs)} nested pairs, expected 15")

    pair_names = {(r["model"], f["model"]) for r, f in nested_pairs(grid)}
    required = {("jaccard", "jaccard_profile"), ("jaccard", "jaccard_profile_z"),
                ("jaccard_profile", "jaccard_profile_both"), ("jaccard_profile_z", "jaccard_profile_both"),
                ("jaccard", "jaccard_bands"), ("jaccard", "jaccard_text"), ("text", "jaccard_text")}
    if not required <= pair_names:
        failures.append(f"nested pairs missing {required - pair_names}")

    cols = profile_columns("js_distance")
    if cols != {"S": "profile_js_distance", "z": "profile_js_distance_z",
                "S_cen": "centroid_profile_js_distance", "z_cen": "centroid_profile_js_distance_z"}:
        failures.append(f"profile_columns wrong: {cols}")
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
        ("jaccard_profile_both", "member_centroid"): ["mean_jaccard_to_members", cols["S_cen"], cols["z_cen"]],
    }
    for (model, reference), wanted in resolved_columns.items():
        cell = next(c for c in grid if c["model"] == model)
        resolved = resolve_features(cell, reference, cols)
        if resolved != wanted:
            failures.append(f"{reference}: {model} resolved to {resolved} != {wanted}")
    free = {c["model"]: c for c in grid if c["metric"] == "none"}
    free_columns = {
        ("jaccard_bands", "label_word"): ["jaccard_pct", *BAND_COLUMNS],
        ("jaccard_bands", "member_centroid"): ["mean_jaccard_to_members", *MEMBER_BAND_COLUMNS],
        ("text", "label_word"): ["text_to_label"],
        ("jaccard_text", "member_centroid"): ["mean_jaccard_to_members", "mean_text_to_members"],
    }
    for (model, reference), wanted in free_columns.items():
        resolved = resolve_features(free[model], reference, {})
        if resolved != wanted:
            failures.append(f"{reference}: {model} resolved to {resolved} != {wanted}")
    pair_columns = {"jaccard": ["jaccard_pct"], "jaccard_bands": ["jaccard_pct", *BAND_COLUMNS],
                    "jaccard_text": ["jaccard_pct", "text_similarity"]}
    for model, wanted in pair_columns.items():
        if resolve_pair_features(free[model], {}) != wanted:
            failures.append(f"pair cell {model} resolved to {resolve_pair_features(free[model], {})}")
    both = next(c for c in grid if c["model"] == "jaccard_profile_both")
    if resolve_pair_features(both, cols) != ["jaccard_pct", cols["S"], cols["z"]]:
        failures.append(f"pair cell jaccard_profile_both resolved to {resolve_pair_features(both, cols)}")
    if resolve_features(grid[0], "label_word", cols) != ["jaccard_pct"]:
        failures.append("label_word jaccard cell must resolve to [jaccard_pct]")
    if resolve_features(grid[0], "member_centroid", cols) != ["mean_jaccard_to_members"]:
        failures.append("member_centroid jaccard cell must resolve to [mean_jaccard_to_members]")
    if set(REFERENCES) != {"label_word", "member_centroid"}:
        failures.append(f"REFERENCES keys {set(REFERENCES)}")

    for label, metric_list, with_text in (("ACTIVE_METRICS", list(ACTIVE_METRICS), True),
                                          ("six metrics", SIX_METRICS, True), ("six metrics, no text", SIX_METRICS, False)):
        declared = set(study_a_columns(metric_list, with_text=with_text)) - {TARGET}
        used = grid_feature_columns(metric_list, with_text=with_text)
        if declared - used:
            failures.append(f"study_a_columns OVER-GATES on {label}: {sorted(declared - used)} "
                            "are dropped over but no grid cell fits them")
        if used - declared:
            failures.append(f"study_a_columns UNDER-GATES on {label}: {sorted(used - declared)} "
                            "are fitted but not dropped over, so cells see different rows")

    decoy = "decoy_metric_not_the_summary_one"
    rank_fixture = pd.DataFrame([
        {"model": "jaccard", "metric": "none", "reference": "label_word",
         "accuracy": 0.10, "delta_accuracy_vs_jaccard": np.nan},
        {"model": "jaccard_profile", "metric": decoy, "reference": "label_word",
         "accuracy": 0.20, "delta_accuracy_vs_jaccard": 0.20},
        {"model": "jaccard_profile", "metric": SUMMARY_METRIC, "reference": "label_word",
         "accuracy": 0.30, "delta_accuracy_vs_jaccard": 0.30},
        {"model": "jaccard_profile_both", "metric": SUMMARY_METRIC, "reference": "label_word",
         "accuracy": 0.35, "delta_accuracy_vs_jaccard": 0.35},
        {"model": "jaccard_bands", "metric": "none", "reference": "label_word",
         "accuracy": 0.12, "delta_accuracy_vs_jaccard": 0.02},
        {"model": "jaccard_profile", "metric": decoy, "reference": "member_centroid",
         "accuracy": 0.40, "delta_accuracy_vs_jaccard": 0.40},
        {"model": "jaccard_profile", "metric": SUMMARY_METRIC, "reference": "member_centroid",
         "accuracy": 0.50, "delta_accuracy_vs_jaccard": 0.50},
    ])
    pair_fixture = pd.DataFrame([
        {"model": "jaccard", "metric": "none", "mean_category_spearman": 0.11, "delta_vs_jaccard": np.nan},
        {"model": "jaccard_profile", "metric": decoy, "mean_category_spearman": 0.22, "delta_vs_jaccard": 0.11},
        {"model": "jaccard_profile", "metric": SUMMARY_METRIC, "mean_category_spearman": 0.33,
         "delta_vs_jaccard": 0.22},
        {"model": "jaccard_profile_both", "metric": SUMMARY_METRIC, "mean_category_spearman": 0.44,
         "delta_vs_jaccard": 0.33},
        {"model": "jaccard_bands", "metric": "none", "mean_category_spearman": 0.15, "delta_vs_jaccard": 0.04},
    ])
    summary = _summary(rank_fixture, pair_fixture)
    expected_summary = {"rank_acc_jaccard": 0.10, "rank_acc_jaccard_profile": 0.30,
                        "rank_acc_delta_profile": 0.30, "rank_acc_delta_bands": 0.02,
                        "rank_acc_centroid_jaccard_profile": 0.50,
                        "pair_sim_rho_jaccard": 0.11, "pair_sim_rho_jaccard_profile": 0.33,
                        "pair_sim_delta_profile": 0.22, "pair_sim_delta_bands": 0.04}
    if set(summary) != set(expected_summary) or not set(summary) <= set(SUMMARY_LABELS):
        failures.append(f"_summary keys {sorted(summary)}, expected {sorted(expected_summary)}, each with a label")
    for key, wanted in expected_summary.items():
        if summary.get(key) != wanted:
            failures.append(f"_summary[{key}] is {summary.get(key)}, expected {wanted}, "
                            f"it must select on (model, metric) and report {SUMMARY_METRIC}")
    return f"{len(grid)} cells per metric, {len(two_pairs)} nested pairs on two metrics"


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
