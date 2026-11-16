"""
Module 5's added-value table compares every richer cell with the Jaccard cell of its own arm, category by category,
and draws the mean difference with its t interval.

Checks:
  1. added_value_summary's mean, interval and paired p equal mean_difference_interval and the one-sample t of the
     per-category differences, for every study, reference and cell.
  2. The other-members arm is compared with its own Jaccard cell, never with the category name's, and only the
     summary metric of a profile cell enters.
  3. plot_added_value draws three panels with one row per cell, four with the text cell and three without it, and
     names Study B's axis by the per-category rho_k it averages, not by the size-weighted rho_W.
  4. At print width the panels stack in one column, every panel names its cells, and each takes the x range given
     for its study, so two models can share their ranges.

Usage, from scripts/:
    python tests/check_module5_added_value.py
"""
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_5_typicality_prediction import (ADDED_VALUE_CELLS, added_value_deltas, added_value_summary,
                                                    plot_added_value)
from utils.statistics import mean_difference_interval
from _figures import closed_figures
from _report import run

CATEGORIES = ["birds", "clothing", "fruit", "furniture", "vehicles"]
CELLS = [("jaccard", "none"), ("jaccard_profile", "js_divergence"), ("jaccard_profile", "js_distance"),
         ("jaccard_profile_z", "js_divergence"), ("jaccard_bands", "none"), ("jaccard_text", "none")]


def fixtures(seed: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-category accuracies of both references and per-category rho of the pair study, every cell."""
    rng = np.random.default_rng(seed)
    rank = pd.DataFrame([{"category": category, "n_pairs": 50, "accuracy": rng.uniform(0.5, 0.8),
                          "regression_accuracy": 0.6, "model": model, "metric": metric, "reference": reference}
                         for reference in ("label_word", "member_centroid") for model, metric in CELLS
                         for category in CATEGORIES])
    rank.loc[rank.reference == "member_centroid", "accuracy"] += 0.2
    pair = pd.DataFrame([{"model": model, "metric": metric, "category": category, "n_pairs": 300,
                          "rho": rng.uniform(0.3, 0.6), "noise_ceiling": 0.9, "rho_over_ceiling": 0.5}
                         for model, metric in CELLS for category in CATEGORIES])
    return rank, pair


def checks(failures: list) -> str:
    rank_per_category, pair_per_category = fixtures()
    deltas = added_value_deltas(rank_per_category, pair_per_category, "js_divergence")
    assert set(deltas.columns) == {"study", "reference", "model", "metric", "category", "delta"}, deltas.columns
    assert set(deltas.model) == set(ADDED_VALUE_CELLS) and "js_distance" not in set(deltas.metric), deltas
    summary = added_value_summary(deltas).set_index(["study", "reference", "model"])
    for key, rows in deltas.groupby(["study", "reference", "model"]):
        assert np.allclose(summary.loc[key, ["delta_mean", "ci_low", "ci_high"]].astype(float),
                           mean_difference_interval(rows.delta)), key
        assert abs(summary.loc[key, "paired_t_p"] - stats.ttest_1samp(rows.delta, 0).pvalue) < 1e-12, key
        assert summary.loc[key, "categories_tested"] == len(rows), key
    member = deltas.query("reference == 'member_centroid' and model == 'jaccard_profile'").set_index("category").delta
    arm = rank_per_category.query("reference == 'member_centroid' and metric in ('js_divergence', 'none')")
    accuracy = arm.set_index(["model", "category"])["accuracy"]
    assert np.allclose(member, (accuracy["jaccard_profile"] - accuracy["jaccard"]).loc[member.index])
    pairs = deltas.query("study == 'B' and model == 'jaccard_bands'").set_index("category").delta
    rho = pair_per_category.set_index(["model", "category"])["rho"]
    assert np.allclose(pairs, (rho["jaccard_bands"] - rho["jaccard"]).loc[pairs.index])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        axes = closed_figures(plot_added_value, summary.reset_index(), deltas, tmp / "a.png")[1][0].axes
        assert len(axes) == 3 and all(len(ax.get_yticks()) == 4 for ax in axes), [ax.get_yticks() for ax in axes]
        assert "rho_k" in axes[2].get_xlabel() and "rho_W" not in axes[2].get_xlabel(), (
            f"Study B plots the mean of per-category rho_k differences: {axes[2].get_xlabel()}")
        no_text = deltas[deltas.model != "jaccard_text"]
        no_text_axes = closed_figures(plot_added_value, added_value_summary(no_text), no_text, tmp / "b.png")[1][0].axes
        assert all(len(ax.get_yticks()) == 3 for ax in no_text_axes), [ax.get_yticks() for ax in no_text_axes]
        assert (tmp / "a.png").exists()
        xlims = {"A": (-0.4, 0.5), "B": (-0.2, 0.6)}
        printed = closed_figures(plot_added_value, summary.reset_index(), deltas, tmp / "c.png", print_width=3.25,
                                 xlims=xlims)[1][0].axes
        assert [ax.get_xlim() for ax in printed] == [xlims["A"], xlims["A"], xlims["B"]], [ax.get_xlim() for ax in printed]
        assert len({round(ax.get_position().x0, 6) for ax in printed}) == 1, "print panels must share one column"
        tops = [ax.get_position().y0 for ax in printed]
        assert tops == sorted(tops, reverse=True), tops
        assert all(ax.yaxis.get_tick_params()["labelleft"] for ax in printed), "every print panel names its cells"
    return (f"{len(summary)} summary rows equal to their per-category differences, three panels of four and three "
            f"rows, print panels stacked on the given ranges")


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
