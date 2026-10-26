"""
Module 1's correlations follow the shared procedure: Spearman's rho with a two-sided permutation p, its own
generator per test, and no Pearson coefficient.

Checks:
  1. The per-category descriptor table carries no Pearson correlation between the label and member-average profiles.
  2. Sublayer informativeness reports the category alignment ROC-AUC with its Mantel p and no point-biserial r, the
     p only for the specific concepts, identical when recomputed after other random draws.
  3. Entropy against expert count reports a pooled Spearman rho with its permutation p, its CSV holds the rank
     scores whose least-squares slope is that rho, and its figure draws the raw values in bits and experts with the
     least-squares line, no band, and a legend naming the line beside the two abstraction levels.

Usage, from scripts/:
    python tests/check_module1_correlations.py
"""
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_1_expert_distribution import (compute_layer_descriptors, compute_sublayer_informativeness,
                                                   plot_entropy_vs_expert_count)
from utils.statistics import PERMUTATIONS
from _figures import closed_figures, has_band, legend_texts, line_slope, points, same_points
from _report import run

SUBLAYERS = ("mlp", "attn")
N_BLOCKS = 4


def synthetic(seed: int = 0) -> tuple:
    """Expert rows over two sublayers of four blocks for four categories of eight members and their labels."""
    rng = np.random.default_rng(seed)
    names = [f"{2 * block + s}.L.{block}.{sublayer}"
             for block in range(N_BLOCKS) for s, sublayer in enumerate(SUBLAYERS)]
    categories = {"fruit": 0, "tools": 1, "birds": 2, "cars": 3}
    rows, meta = [], []
    for category, base in categories.items():
        words = [category] + [f"{category}{i}" for i in range(8)]
        for word in words:
            meta.append({"concept": word, "category": None if word == category else category,
                         "abstraction_level": 1 if word == category else 2})
            units = np.concatenate([rng.choice(np.arange(base * 40, base * 40 + 40), 12, replace=False),
                                    rng.choice(np.arange(300, 500), 6 + rng.integers(0, 20), replace=False)])
            for unit in units:
                layer = int(rng.integers(0, len(names)))
                rows.append({"concept": word, "layer_name": names[layer], "layer_idx": layer, "unit": int(unit)})
    frame = pd.DataFrame(rows)
    frame["layer_name"] = pd.Categorical(frame["layer_name"], categories=names, ordered=True)
    return frame, pd.DataFrame(meta)


def check_descriptors() -> str:
    frame, meta = synthetic()
    _, category_df = compute_layer_descriptors(frame, meta)
    assert "pearson_correlation_distributions" not in category_df.columns, "the label-member Pearson r is dropped"
    return f"descriptors: {len(category_df)} categories without a Pearson column"


def check_informativeness() -> str:
    frame, meta = synthetic()
    with tempfile.TemporaryDirectory() as folder:
        first = compute_sublayer_informativeness(frame, meta, pathlib.Path(folder))
        np.random.default_rng(9).normal(size=100)
        second = compute_sublayer_informativeness(frame, meta, pathlib.Path(folder))
    assert "category_alignment_r" not in first.columns, "the point-biserial r is dropped"
    members = first[first["abstraction_level"] == 2]
    labels = first[first["abstraction_level"] == 1]
    assert members["mantel_p"].between(1 / (PERMUTATIONS + 1), 1).all(), f"p {members['mantel_p'].tolist()}"
    assert labels[["roc_auc", "mantel_p"]].isna().all().all(), "labels carry no alignment"
    assert first["mantel_p"].equals(second["mantel_p"]), "each sublayer's p must not depend on earlier draws"
    return f"informativeness: AUC {members['roc_auc'].round(3).tolist()} at p {members['mantel_p'].tolist()}"


def check_entropy_panel() -> str:
    frame, meta = synthetic()
    concept_df, category_df = compute_layer_descriptors(frame, meta)
    with tempfile.TemporaryDirectory() as folder:
        row, figures = closed_figures(plot_entropy_vs_expert_count, concept_df, category_df, pathlib.Path(folder),
                                      len(meta))
        table = pd.read_csv(pathlib.Path(folder) / "shannon_entropy_vs_expert_count.csv")
    axis = figures[0].axes[0]
    assert same_points(points(axis), table["shannon_entropy"], table["experts_count"]), "raw values must be drawn"
    expected = np.polyfit(table["shannon_entropy"], table["experts_count"], 1)[0]
    assert abs(line_slope(axis) - expected) < 1e-9 and not has_band(axis), "least-squares line of the raw values only"
    assert legend_texts(axis) == ["Specific Concepts", "Broad Categories", "Least-squares line"], \
        f"legend {legend_texts(axis)}"
    assert (axis.get_xlabel(), axis.get_ylabel()) == ("Shannon entropy (bits)", "Expert count"), "raw axis labels"
    assert "pearson_r" not in row and row["ranks"] == "pooled", f"row {row}"
    assert 1 / (PERMUTATIONS + 1) <= row["spearman_p"] <= 1, f"p {row['spearman_p']}"
    slope = np.polyfit(table["shannon_entropy_rank_z"], table["experts_count_rank_z"], 1)[0]
    assert abs(slope - row["spearman_rho"]) < 1e-9, f"slope {slope}, rho {row['spearman_rho']}"
    return f"entropy panel: rho {row['spearman_rho']:.3f} at p {row['spearman_p']}"


def checks(failures: list) -> str:
    return ", ".join([check_descriptors(), check_informativeness(), check_entropy_panel()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
