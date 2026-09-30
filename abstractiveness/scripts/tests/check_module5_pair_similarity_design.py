"""
Module 5's pair-similarity design is symmetric, joined to the right ratings, and shortcut-free.

Checks:
  1. On a four-fruit fixture, one row per rated pair, canonical order, target on the 1 to 7 scale,
     and every per-metric feature column present.
  2. Pairs of a word listed under two categories are joined on its per-sense concept key, so a
     rated squash__sports pair is found.
  3. Category identity alone cannot predict pair similarity under the grouped split, so the pair
     construction does not leak the category.

Usage, from scripts/:
    python tests/check_module5_pair_similarity_design.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_5_typicality_prediction import build_pair_similarity_design, ACTIVE_METRICS
from core.human_similarity_ratings import human_pair_lookup
from _report import run


def fixture(concepts: list, category: str) -> tuple:
    """Five expert rows per concept over three layers, and a one-category design frame."""
    rows = []
    for offset, concept in enumerate(concepts):
        for unit in range(5):
            layer = 1 + (unit + offset) % 3
            rows.append({"concept": concept, "layer_idx": layer, "unit": unit,
                         "layer_name": f"{layer}.L.{layer - 1}.mlp.c_fc"})
    experts = pd.DataFrame(rows)
    ordered = sorted(experts["layer_name"].unique(), key=lambda name: int(name.split(".")[0]))
    experts["layer_name"] = pd.Categorical(experts["layer_name"], categories=ordered, ordered=True)
    design = pd.DataFrame({"concept": concepts, "category": [category] * len(concepts),
                           "human_typicality": np.linspace(0.9, 0.1, len(concepts))})
    return design, experts


def checks(failures: list) -> str:
    concepts = ["apple", "banana", "pear", "olive"]
    rows = []
    for offset, concept in enumerate(concepts):
        for unit in range(5):
            layer = 1 + (unit + offset) % 3
            rows.append({"concept": concept, "layer_idx": layer, "unit": unit,
                         "layer_name": f"{layer}.L.{layer - 1}.mlp.c_fc"})
    experts = pd.DataFrame(rows)
    ordered = sorted(experts["layer_name"].unique(), key=lambda name: int(name.split(".")[0]))
    experts["layer_name"] = pd.Categorical(experts["layer_name"], categories=ordered, ordered=True)
    design = pd.DataFrame({"concept": concepts, "category": ["fruit"] * 4,
                           "human_typicality": [0.9, 0.8, 0.7, 0.1]})
    table = build_pair_similarity_design(design, experts, ACTIVE_METRICS)

    lookup = human_pair_lookup()
    expected = sum(1 for i in range(4) for j in range(i + 1, 4)
                   if (min(concepts[i], concepts[j]), max(concepts[i], concepts[j])) in lookup)
    assert len(table) == expected, f"expected {expected} rated pairs, got {len(table)}"
    assert (table.word_a < table.word_b).all(), "pairs not canonical"
    assert table.human_similarity.between(1, 7).all(), "target off the 1 to 7 scale"
    for column in ["jaccard_pct"] + [c for m in ACTIVE_METRICS for c in (f"profile_{m}", f"profile_{m}_z")]:
        if column not in table.columns:
            failures.append(f"{column} missing")
    assert table.category.nunique() == 1, "toy fixture should be single-category"

    sports = ["squash__sports", "tennis", "golf", "boxing"]
    sense_table = build_pair_similarity_design(*fixture(sports, "sports"), ACTIVE_METRICS)
    rated_with_squash = sum(1 for other in sports[1:] if tuple(sorted(("squash__sports", other))) in lookup)
    joined_with_squash = int(((sense_table.word_a == "squash__sports") | (sense_table.word_b == "squash__sports")).sum())
    if rated_with_squash == 0:
        failures.append("the squash fixture has no rated pair, pick other sports")
    if joined_with_squash != rated_with_squash:
        failures.append(f"squash__sports joined to {joined_with_squash} rated pairs, expected {rated_with_squash}")

    rng = np.random.default_rng(0)
    big = pd.DataFrame({"category": ["a"] * 50 + ["b"] * 50,
                        "human_similarity": np.r_[rng.normal(4, 1, 50), rng.normal(4, 1, 50)]})
    codes = pd.factorize(big.category)[0].reshape(-1, 1).astype(float)
    predicted = cross_val_predict(LinearRegression(), codes, big.human_similarity,
                                  cv=GroupKFold(n_splits=2), groups=big.category)
    leak = abs(spearmanr(predicted, big.human_similarity).statistic)
    if not leak < 0.2:
        failures.append(f"category identity predicted pair similarity (rho={leak:.3f}), the split leaks")
    return f"{len(table)} fixture pairs, {joined_with_squash} squash__sports pairs joined, category-shortcut rho {leak:.3f}"


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
