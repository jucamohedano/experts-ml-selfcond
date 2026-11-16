"""
Module 5 reports every gain over the Jaccard index with its t interval over categories, and every feature's partial
Spearman coefficient under each control set.

Runs Study B on a synthetic pair table whose rated pairs are complete inside each category, as the human ratings
are, so the Mantel draws keep every pair.

Checks:
  1. Study B: delta_vs_jaccard and its interval on the jaccard_profile row equal mean_difference_interval of the
     per-category rho differences from the jaccard cell.
  2. pair_partials of the profile given the Jaccard index equals spearman_test with Mantel draws over the word
     positions, the Jaccard index is never controlled for itself, and without text no text feature or control appears.
  3. typicality_partials reads each reference's own band columns.
  4. Study A: delta_ci_low and delta_ci_high equal mean_difference_interval of the per-category accuracy differences
     from the reference's own jaccard cell.

Usage, from scripts/:
    python tests/check_module5_inference.py
"""
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import modules.module_5_typicality_prediction as m5
from modules.module_5_typicality_prediction import (BAND_COLUMNS, MEMBER_BAND_COLUMNS, REFERENCES, TARGET,
                                                    pair_partials, run_pair_similarity, run_typicality_ranking,
                                                    typicality_partials)
from utils.statistics import mean_difference_interval, spearman_test
from _report import run


SIZES = {"birds": 8, "clothing": 10, "fruit": 12, "furniture": 14, "vehicles": 16}


def complete_pair_table(seed: int = 0) -> pd.DataFrame:
    """Every pair of members rated inside each category, the ratings tracking the Jaccard index and the text."""
    rng = np.random.default_rng(seed)
    pairs = [(category, f"{category}_{a}", f"{category}_{b}") for category, size in SIZES.items()
             for a, b in itertools.combinations(range(size), 2)]
    table = pd.DataFrame(pairs, columns=["category", "word_a", "word_b"])
    jaccard = rng.uniform(0, 100, len(table))
    text = rng.normal(size=len(table))
    table["human_similarity"] = 4 + (jaccard - 50) / 50 + text + rng.normal(0, 1, len(table))
    table["jaccard_pct"], table["text_similarity"] = jaccard, text
    table["profile_js_distance"] = jaccard + rng.normal(0, 40, len(table))
    table["profile_js_distance_z"] = rng.normal(0, 1, len(table))
    for column in BAND_COLUMNS:
        table[column] = jaccard + rng.normal(0, 30, len(table))
    return table


def study_a_design(seed: int = 3) -> pd.DataFrame:
    """Five categories of twelve concepts with every Study A column, the ratings loosely tracking the features."""
    rng = np.random.default_rng(seed)
    c = np.repeat(np.arange(5), 12)
    rating = rng.normal(size=len(c))
    noisy = lambda scale: rating + scale * rng.normal(size=len(c))
    columns = ["jaccard_pct", "mean_jaccard_to_members", *BAND_COLUMNS, *MEMBER_BAND_COLUMNS, "text_to_label",
               "mean_text_to_members", "profile_js_distance", "profile_js_distance_z",
               "centroid_profile_js_distance", "centroid_profile_js_distance_z"]
    return pd.DataFrame({"concept": [f"w{i}" for i in range(len(c))], "category": [f"c{i}" for i in c],
                         TARGET: rating + c, **{column: noisy(1.5) for column in columns}})


def checks(failures: list) -> str:
    table = complete_pair_table()
    real_build_design = m5.build_pair_similarity_design
    m5.build_pair_similarity_design = lambda *args, **kwargs: table.copy()
    try:
        results, per_category, _, fitted = run_pair_similarity(pd.DataFrame(), pd.DataFrame(), ["js_distance"],
                                                               text=pd.DataFrame())
    finally:
        m5.build_pair_similarity_design = real_build_design

    rho_k = per_category.pivot_table(index="category", columns=["model", "metric"], values="rho")
    delta = rho_k[("jaccard_profile", "js_distance")] - rho_k[("jaccard", "none")]
    row = results.set_index(["model", "metric"]).loc[("jaccard_profile", "js_distance")]
    assert np.allclose([row.delta_vs_jaccard, row.delta_ci_low_vs_jaccard, row.delta_ci_high_vs_jaccard],
                       mean_difference_interval(delta)), row

    partials = pair_partials(fitted, ["js_distance"], with_text=True, n_draws=99)
    assert set(partials.columns) == {"feature", "role", "metric", "controls", "rho", "p", "n"}, partials.columns
    got = partials.query("feature == 'profile_js_distance' and controls == 'jaccard'").iloc[0]
    codes = {c: i for i, c in enumerate(sorted(set(fitted.word_a) | set(fitted.word_b)))}
    positions = (fitted.word_a.map(codes).to_numpy(), fitted.word_b.map(codes).to_numpy())
    want = spearman_test(fitted.profile_js_distance, fitted.human_similarity, [fitted.jaccard_pct],
                         groups=fitted.category, pairs=positions, n_draws=99)
    assert got.n == len(fitted), f"Mantel completion dropped pairs: {got.n} of {len(fitted)}"
    assert abs(got.rho - want["rho"]) < 1e-12 and got.p == want["p"], (got, want["rho"], want["p"])
    assert partials.query("feature == 'jaccard_pct' and controls in ('jaccard', 'jaccard_text')").empty
    assert partials.query("feature == 'text_similarity' and controls in ('text', 'jaccard_text')").empty
    assert len(partials.query("feature == 'jaccard_band_2_pct'")) == 4, "a band takes all four control sets"
    no_text = pair_partials(fitted.drop(columns="text_similarity"), ["js_distance"], with_text=False, n_draws=0)
    assert not no_text.controls.str.contains("text").any() and "text_similarity" not in set(no_text.feature)

    design = study_a_design()
    typ = typicality_partials(design, ["js_distance"], with_text=True, n_draws=0)
    assert set(typ.query("reference == 'member_centroid'").feature) >= set(MEMBER_BAND_COLUMNS)
    assert not set(typ.query("reference == 'member_centroid'").feature) & set(BAND_COLUMNS)

    ranks, ranks_per_category, _, _ = run_typicality_ranking(design, ["js_distance"], with_text=True)
    for reference in REFERENCES:
        arm = ranks_per_category[ranks_per_category.reference == reference]
        accuracy = arm.pivot_table(index="category", columns=["model", "metric"], values="accuracy")
        difference = accuracy[("jaccard_bands", "none")] - accuracy[("jaccard", "none")]
        rank_row = ranks.set_index(["reference", "model", "metric"]).loc[(reference, "jaccard_bands", "none")]
        assert np.allclose([rank_row.delta_accuracy_vs_jaccard, rank_row.delta_ci_low, rank_row.delta_ci_high],
                           mean_difference_interval(difference)), rank_row
    return (f"Study B interval {row.delta_vs_jaccard:+.3f} [{row.delta_ci_low_vs_jaccard:+.3f}, "
            f"{row.delta_ci_high_vs_jaccard:+.3f}], {len(partials)} pair partials, {len(typ)} typicality partials, "
            f"Study A intervals on both references")


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
