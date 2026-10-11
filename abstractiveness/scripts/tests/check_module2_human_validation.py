"""
Module 2's human validation uses every rated pair and reports coherent statistics.

Runs validate_against_human on synthetic symmetric matrices over every metadata concept, so each
rated pair has a finite model value and none may be lost.

Checks:
  1. Per metric and category, n_pairs equals the number of rated pairs.
  2. One row per category plus POOLED and POOLED_MEAN per metric, categories matching the ceilings.
  3. rho in [-1, 1], Mantel p in [0, 1], and dividing by a ceiling below 1 never shrinks |rho|.
  4. The validation CSV is written.
  5. The ranked mirror figure keeps every rated pair, and per category the least-squares slope
     between its standardised ranks equals the reported Spearman rho.

Usage, from scripts/:
    python tests/check_module2_human_validation.py
"""
import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_2_similarities import validate_against_human, profile_matrix_key, plot_human_vs_expert_ranked
from modules.shared_layer_profile_measures import DEFAULT_PROFILE_METRIC
from core.human_similarity_ratings import load_human_similarity, human_noise_ceiling
from _report import run

METADATA = pathlib.Path(__file__).resolve().parents[3] / "abstractiveness/assets/metadata_Richie_HSJ.json"


def checks(failures: list) -> str:
    concepts = [entry["concept"] for entry in json.load(METADATA.open())]
    rng = np.random.default_rng(0)
    matrices = {}
    for key in ("jaccard", profile_matrix_key(DEFAULT_PROFILE_METRIC)):
        values = rng.uniform(0, 100, size=(len(concepts), len(concepts)))
        matrices[key] = (values + values.T) / 2

    with tempfile.TemporaryDirectory() as out:
        table = validate_against_human(concepts, matrices, pathlib.Path(out), make_plots=False)
        if not (pathlib.Path(out) / "human_similarity_validation.csv").exists():
            failures.append("validation CSV not written")
        ranked_png = pathlib.Path(out) / "similarity_vs_jaccard_ranked.png"
        plot_human_vs_expert_ranked(concepts, matrices["jaccard"], "Jaccard", ranked_png)
        ranked = pd.read_csv(ranked_png.with_suffix(".csv"))

    rated = load_human_similarity().groupby("category").size()
    ceiling = human_noise_ceiling()
    per_category = table[~table.category.isin(["POOLED", "POOLED_MEAN"])]
    if set(per_category.category) != set(ceiling):
        failures.append("categories do not match the ceiling keys")
    for key in matrices:
        rows = table[table.metric == key]
        if len(rows) != len(ceiling) + 2:
            failures.append(f"{key}: expected one row per category plus two pooled rows, got {len(rows)}")
        used = rows[~rows.category.isin(["POOLED", "POOLED_MEAN"])].set_index("category").n_pairs
        lost = (rated - used.reindex(rated.index).fillna(0)).astype(int)
        if (lost != 0).any():
            failures.append(f"{key}: rated pairs not used, per category {lost[lost != 0].to_dict()}")

    if not per_category.rho.between(-1, 1).all():
        failures.append("rho outside [-1, 1]")
    if not per_category.mantel_p.between(0, 1).all():
        failures.append("mantel_p outside [0, 1]")
    if not (per_category.rho_over_ceiling.abs() >= per_category.rho.abs() - 1e-9).all():
        failures.append("dividing by a ceiling below 1 shrank a magnitude")

    reported = table[table.metric == "jaccard"].set_index("category").rho
    for category, group in ranked.groupby("category"):
        slope = np.polyfit(group.expert_rank_z, group.human_rank_z, 1)[0]
        if abs(slope - reported[category]) > 1e-3:
            failures.append(f"ranked mirror, {category}: slope {slope:.4f} against rho {reported[category]:.4f}")
    if len(ranked) != int(rated.sum()):
        failures.append(f"ranked mirror keeps {len(ranked)} of {int(rated.sum())} rated pairs")
    return f"{int(per_category.n_pairs.sum())} pair evaluations over {len(matrices)} matrices"


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
