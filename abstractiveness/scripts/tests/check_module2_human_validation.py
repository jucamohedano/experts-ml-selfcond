"""
Module 2's human validation reports, per matrix, each category's Spearman rho with its Mantel p and their pair-weighted
mean with a stratified Mantel p, on every rated pair.

Runs validate_against_human on synthetic symmetric matrices over every metadata concept, so each rated pair has a
finite model value and none may be lost, the Jaccard matrix tracking the ratings and the profile matrix unrelated.

Checks:
  1. Per metric, one row per category plus WEIGHTED_MEAN, the categories those of the ceilings, n_pairs the rated
     pairs.
  2. WEIGHTED_MEAN holds the pair-weighted means of the category rho and of the ceilings, with a p of its own.
  3. Every rho in [-1, 1] and p in [floor, 1], dividing by a ceiling below 1 never shrinks |rho|, a category's p equals
     the Mantel test of that category alone, and a matrix's rows do not change when another matrix is validated
     with it.
  4. The CSV and the per-category figure are written, the figure's CSV keeps every rated pair with its raw values,
     per category the least-squares slope between its rank z-scores equals the reported rho, and each panel draws
     the raw pairs with their least-squares line and no bootstrap band, since pairs share concepts.

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
from modules.module_2_similarities import WEIGHTED_MEAN, profile_matrix_key, validate_against_human
from modules.shared_layer_profile_measures import DEFAULT_PROFILE_METRIC
from core.human_similarity_ratings import load_human_similarity, human_noise_ceiling
from utils.statistics import PERMUTATIONS, spearman_test
from _figures import closed_figures, has_band, legend_texts, line_slope, points, same_points
from _report import run

METADATA = pathlib.Path(__file__).resolve().parents[3] / "abstractiveness/assets/metadata_Richie_HSJ.json"


def checks(failures: list) -> str:
    concepts = [entry["concept"] for entry in json.load(METADATA.open())]
    index = {concept: i for i, concept in enumerate(concepts)}
    pairs = load_human_similarity()
    rng = np.random.default_rng(0)
    jaccard = rng.uniform(0, 10, size=(len(concepts), len(concepts)))
    for a, b, rating in zip(pairs.concept_a, pairs.concept_b, pairs.mean_rating):
        jaccard[index[a], index[b]] = jaccard[index[b], index[a]] = rating + rng.normal(0, 1.5)
    profile = rng.uniform(0, 100, size=jaccard.shape)
    matrices = {"jaccard": jaccard, profile_matrix_key(DEFAULT_PROFILE_METRIC): (profile + profile.T) / 2}

    with tempfile.TemporaryDirectory() as out:
        table, figures = closed_figures(validate_against_human, concepts, matrices, pathlib.Path(out))
        assert (pathlib.Path(out) / "human_similarity_validation.csv").exists(), "validation CSV not written"
        assert (pathlib.Path(out) / "similarity_vs_jaccard.png").exists(), "per-category figure not written"
        ranked = pd.read_csv(pathlib.Path(out) / "similarity_vs_jaccard.csv")
    alone = validate_against_human(concepts, {"jaccard": jaccard}, None, make_plots=False)

    rated = pairs.groupby("category").size()
    ceiling = pd.Series(human_noise_ceiling())
    floor = 1 / (PERMUTATIONS + 1)
    for key in matrices:
        rows = table[table.metric == key].set_index("category")
        per_category = rows.drop(index=WEIGHTED_MEAN)
        assert set(per_category.index) == set(ceiling.index) and len(rows) == len(ceiling) + 1, \
            f"{key}: rows {list(rows.index)}"
        assert (per_category.n_pairs == rated.reindex(per_category.index)).all(), f"{key}: rated pairs lost"
        weights = per_category.n_pairs / per_category.n_pairs.sum()
        mean = rows.loc[WEIGHTED_MEAN]
        assert abs(mean.rho - (weights * per_category.rho).sum()) < 1e-12, f"{key}: weighted rho {mean.rho}"
        assert abs(mean.noise_ceiling - (weights * ceiling.reindex(per_category.index)).sum()) < 1e-12, \
            f"{key}: weighted ceiling {mean.noise_ceiling}"
        assert rows.rho.between(-1, 1).all() and rows.mantel_p.between(floor, 1).all(), f"{key}: rho or p out of range"
        assert (per_category.rho_over_ceiling.abs() >= per_category.rho.abs() - 1e-12).all(), \
            f"{key}: dividing by a ceiling below 1 shrank a magnitude"
    assert table[table.metric == "jaccard"].reset_index(drop=True).equals(alone), \
        "a matrix's rows must not depend on the matrices validated with it"

    birds = pairs[pairs.category == "birds"]
    first, second = birds.concept_a.map(index).to_numpy(), birds.concept_b.map(index).to_numpy()
    by_itself = spearman_test(jaccard[first, second], birds.mean_rating, groups=birds.category, pairs=(first, second))
    reported = table[(table.metric == "jaccard") & (table.category == "birds")].iloc[0]
    assert by_itself["p"] == reported.mantel_p and abs(by_itself["rho"] - reported.rho) < 1e-12, \
        f"birds: p {reported.mantel_p} against {by_itself['p']} tested alone"

    reported_rho = table[table.metric == "jaccard"].set_index("category").rho
    for category, group in ranked.groupby("category"):
        slope = np.polyfit(group.expert_rank_z, group.human_rank_z, 1)[0]
        assert abs(slope - reported_rho[category]) < 1e-9, \
            f"ranked mirror, {category}: slope {slope:.4f} against rho {reported_rho[category]:.4f}"
    assert len(ranked) == int(rated.sum()), f"the figure keeps {len(ranked)} of {int(rated.sum())} rated pairs"
    raw = jaccard[ranked.concept_a.map(index), ranked.concept_b.map(index)]
    assert np.allclose(ranked.expert_similarity, raw), "the figure's CSV must hold the raw model values"
    figure = next(f for f in figures if f._suptitle is not None and f._suptitle.get_text().startswith("Expert Jaccard"))
    visible = [axis for axis in figure.axes if axis.get_visible()]
    assert len(visible) == ranked.category.nunique(), f"{len(visible)} panels"
    for axis in visible:
        group = ranked[ranked.category == axis.get_title().lower()]
        assert same_points(points(axis), group.expert_similarity, group.human_rating), f"{axis.get_title()}: points"
        expected = np.polyfit(group.expert_similarity, group.human_rating, 1)[0]
        assert abs(line_slope(axis) - expected) < 1e-9 and not has_band(axis), f"{axis.get_title()}: line or band"
    assert legend_texts(visible[0]) == ["Least-squares line"] and not legend_texts(visible[1]), "legend once"
    assert visible[0].get_ylabel() == "Human similarity rating (1 to 7)", f"y label {visible[0].get_ylabel()!r}"
    weighted = table[table.category == WEIGHTED_MEAN].set_index("metric")
    return (f"{int(rated.sum())} rated pairs per matrix, weighted rho {weighted.rho['jaccard']:.3f} at p "
            f"{weighted.mantel_p['jaccard']} for the related matrix, p {weighted.mantel_p.iloc[1]:.3f} "
            f"for the unrelated")


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
