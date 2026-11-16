"""
The eight layer-profile metrics fall in three groups by what they compare, and module 5's compact figure keeps one
metric per group for every cell of Studies A and B.

Checks:
  1. Every registered metric sits in exactly one of the three groups, overlap of proportions, pattern across blocks
     and distance along the network, in that order, and each group's representative is its first metric:
     Jensen-Shannon divergence, cosine and Wasserstein, the summary metric among them.
  2. selected_metric_table keeps the representatives and the cells without a profile metric, with each value equal
     to the ranking accuracy (Study A, both references) or the rho_W (Study B) it was read from, and skips a
     representative the results lack.
  3. plot_selected_metric_comparison draws three panels with the nine cells in grid order and thesis notation, one
     marker per cell and metric, a legend naming the three metrics, and Study B's axis as rho_W.
  4. At full print width the figure is no wider than the page, the panels take the x range given for their study,
     only the first names the cells, and the legend sits under every panel.

Usage, from scripts/:
    python tests/check_module5_metric_groups.py
"""
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.shared_layer_profile_measures import (PROFILE_METRICS, PROFILE_METRIC_GROUPS,
                                                   REPRESENTATIVE_PROFILE_METRICS)
from modules.module_5_typicality_prediction import (GRID_CELLS, GRID_NOTATION, SUMMARY_METRIC, build_feature_grid,
                                                    plot_selected_metric_comparison, selected_metric_table)
from utils.plotting import print_style
from _figures import closed_figures, points
from _report import run

GROUPS = ["Overlap of proportions", "Pattern across blocks", "Distance along the network"]
ROWS_PER_ARM = 4 + 5 * 3


def fixtures(metrics=tuple(PROFILE_METRICS), seed: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ranking accuracies of both references and pair rho_W for every generated cell."""
    rng = np.random.default_rng(seed)
    grid = build_feature_grid(list(metrics))
    rank = pd.DataFrame([{"model": cell["model"], "metric": cell["metric"], "reference": reference,
                          "accuracy": rng.uniform(0.45, 0.8), "regression_accuracy": 0.6}
                         for reference in ("label_word", "member_centroid") for cell in grid])
    pair = pd.DataFrame([{"model": cell["model"], "metric": cell["metric"],
                          "mean_category_spearman": rng.uniform(0.1, 0.6), "noise_ceiling_mean": 0.88}
                         for cell in grid])
    return rank, pair


def checks(failures: list) -> str:
    members = [metric for group in PROFILE_METRIC_GROUPS.values() for metric in group]
    assert list(PROFILE_METRIC_GROUPS) == GROUPS, list(PROFILE_METRIC_GROUPS)
    assert sorted(members) == sorted(PROFILE_METRICS) and len(members) == len(set(members)), members
    assert REPRESENTATIVE_PROFILE_METRICS == ("js_divergence", "cosine", "wasserstein"), REPRESENTATIVE_PROFILE_METRICS
    assert all(group[0] == metric for group, metric in zip(PROFILE_METRIC_GROUPS.values(),
                                                           REPRESENTATIVE_PROFILE_METRICS))
    assert SUMMARY_METRIC in REPRESENTATIVE_PROFILE_METRICS, SUMMARY_METRIC

    rank, pair = fixtures()
    table = selected_metric_table(rank, pair)
    assert set(table.metric) == {"none", *REPRESENTATIVE_PROFILE_METRICS}, set(table.metric)
    sizes = table.groupby(["study", "reference"]).size()
    assert sizes.to_dict() == {("A", "label_word"): ROWS_PER_ARM, ("A", "member_centroid"): ROWS_PER_ARM,
                               ("B", "pairs"): ROWS_PER_ARM}, sizes
    source = pd.concat([rank.rename(columns={"accuracy": "value"}).assign(study="A"),
                        pair.rename(columns={"mean_category_spearman": "value"}).assign(study="B", reference="pairs")])
    merged = table.merge(source, on=["study", "reference", "model", "metric"], suffixes=("", "_source"))
    assert len(merged) == len(table) and np.allclose(merged.value, merged.value_source), "values must be the inputs'"
    partial = selected_metric_table(*fixtures(("js_divergence", "cosine", "pearson")))
    assert set(partial.metric) == {"none", "js_divergence", "cosine"}, set(partial.metric)

    notation = [GRID_NOTATION[model] for model in GRID_CELLS]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        fig = closed_figures(plot_selected_metric_comparison, table, tmp / "a.png")[1][0]
        axes = fig.axes
        assert len(axes) == 3, len(axes)
        assert [t.get_text() for t in axes[0].get_yticklabels()] == notation, axes[0].get_yticklabels()
        assert all(len(points(ax)) == ROWS_PER_ARM for ax in axes), [len(points(ax)) for ax in axes]
        entries = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
        for label in ("Jensen-Shannon divergence", "Cosine", "Wasserstein"):
            assert any(entry.startswith(label) for entry in entries), (label, entries)
        assert "rho_W" in axes[2].get_xlabel(), axes[2].get_xlabel()
        assert (tmp / "a.png").exists()

        xlims = {"A": (0.4, 0.85), "B": (0.0, 0.65)}
        printed = closed_figures(plot_selected_metric_comparison, table, tmp / "b.png", print_width=6.6,
                                 xlims=xlims)[1][0]
        axes = printed.axes
        assert [ax.get_xlim() for ax in axes] == [xlims["A"], xlims["A"], xlims["B"]], [ax.get_xlim() for ax in axes]
        assert [bool(ax.yaxis.get_tick_params()["labelleft"]) for ax in axes] == [True, False, False]
        with print_style():
            renderer = printed.canvas.get_renderer()
            width = printed.get_tightbbox(renderer).width
            legend_top = printed.legends[0].get_window_extent(renderer).y1
            lowest = min(ax.get_tightbbox(renderer).y0 for ax in axes)
        assert width <= 6.6 + 0.05, f"print figure {width:.2f} in wide"
        assert legend_top <= lowest + 1, f"legend top {legend_top:.0f} px above the panels' bottom {lowest:.0f} px"
    return (f"{len(members)} metrics in three groups, {len(table)} selected rows equal to their inputs, three panels "
            f"of nine cells, print figure {width:.2f} in wide")


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
