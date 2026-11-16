"""
Module 5 fits its models on within-category rank scores, so category means cannot steer a weight.

Checks:
  1. within_category_predictions on five categories of 40 pairs, y = x + noise inside each, whose
     category means of x rise while those of y fall: the weight of x is positive in all five
     folds and the out-of-fold predictions reach a within-category Spearman above 0.8, while a
     pooled least-squares fit on the raw values gets the sign wrong.
  2. A constant added to x per category leaves the predictions unchanged.
  3. x constant inside one category still gives finite predictions.
  4. evaluate_ranker's ridge baseline is invariant to one constant per category added to the
     feature, and the ranker's fold weights come back as held_out_category, feature, weight.

Usage, from scripts/:
    python tests/check_module5_within_category_fit.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_5_typicality_prediction import (TARGET, evaluate_ranker, within_category_pairs,
                                                    within_category_predictions)
from utils.statistics import spearman_test
from _report import run


def checks(failures: list) -> str:
    rng = np.random.default_rng(5)
    k = np.repeat(np.arange(5), 40)
    groups = np.array([f"c{i}" for i in k])
    u = rng.normal(size=len(k))
    x = u + 10 * k
    y = u + 0.3 * rng.normal(size=len(k)) - 3 * k

    predicted, weights = within_category_predictions(x[:, None], y, groups, ["x"])
    assert len(weights) == 5 and (weights.weight > 0).all(), weights
    assert set(weights.columns) == {"held_out_category", "feature", "weight"}, weights.columns
    rho = spearman_test(predicted, y, groups=groups, n_draws=0)["rho"]
    assert rho > 0.8, rho
    assert LinearRegression().fit(x[:, None], y).coef_[0] < 0, "the fixture must defeat a pooled fit"

    offset_by_category = rng.normal(scale=50, size=5)[k]
    shifted, _ = within_category_predictions((x + offset_by_category)[:, None], y, groups, ["x"])
    assert np.allclose(predicted, shifted), "a per-category constant changed the predictions"

    x_flat = x.copy()
    x_flat[groups == "c0"] = 1.0
    assert np.isfinite(within_category_predictions(x_flat[:, None], y, groups, ["x"])[0]).all()

    c = np.repeat(np.arange(5), 12)
    spread = rng.normal(size=len(c))
    design = pd.DataFrame({"concept": [f"w{i}" for i in range(len(c))],
                           "category": [f"c{i}" for i in c],
                           TARGET: spread + c})
    pairs, gap = within_category_pairs(design)
    per_concept_features = (spread + 0.5 * rng.normal(size=len(c)))[:, None]
    category_offsets = (-20.0 * c)[:, None]
    scores, _, _, ranker_weights = evaluate_ranker(per_concept_features, design, pairs, gap, ["f"])
    shifted_scores = evaluate_ranker(per_concept_features + category_offsets, design, pairs, gap, ["f"])[0]
    assert scores["regression_accuracy"] == shifted_scores["regression_accuracy"], (
        scores["regression_accuracy"], shifted_scores["regression_accuracy"])
    assert set(ranker_weights.columns) == {"held_out_category", "feature", "weight"}, ranker_weights.columns
    assert len(ranker_weights) == 5, ranker_weights
    return (f"x weight positive in 5 of 5 folds, rho_W {rho:.3f}, shift invariant, constant category "
            f"finite, ridge accuracy {scores['regression_accuracy']:.3f} under category offsets")


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
