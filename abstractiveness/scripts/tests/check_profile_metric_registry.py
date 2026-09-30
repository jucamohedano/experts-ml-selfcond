"""
Every registered layer-profile metric honours the PROFILE_METRICS contract.

Checks:
  1. Square form on one input, cross form on two, and f(P) equals f(P, P).
  2. Orientation: identical profiles score higher than disjoint ones.
  3. The default is js_distance and reproduces 100 * (1 - sqrt(JSD)) from a hand-rolled JSD.

Usage, from scripts/:
    python tests/check_profile_metric_registry.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.shared_layer_profile_measures import (PROFILE_METRICS, DEFAULT_PROFILE_METRIC,
                                                   _profile_jsd_matrix, _jsd_to_similarity)
from _report import run


def checks(failures: list) -> str:
    identical = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    disjoint = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(0)
    random_p = rng.dirichlet(np.ones(6), size=5)
    random_q = rng.dirichlet(np.ones(6), size=3)

    for name, func in PROFILE_METRICS.items():
        square = func(random_p)
        if square.shape != (5, 5):
            failures.append(f"{name}: square form returned {square.shape}, expected (5, 5)")
        cross = func(random_p, random_q)
        if cross.shape != (5, 3):
            failures.append(f"{name}: cross form returned {cross.shape}, expected (5, 3)")
        if not np.allclose(square, func(random_p, random_p)):
            failures.append(f"{name}: f(P) != f(P, P)")
        same, far = func(identical)[0, 1], func(disjoint)[0, 1]
        if not same > far:
            failures.append(f"{name}: orientation violated, identical={same:.4f} <= disjoint={far:.4f}")

    if DEFAULT_PROFILE_METRIC != "js_distance":
        failures.append(f"default metric is {DEFAULT_PROFILE_METRIC}, expected js_distance")
    if not np.allclose(PROFILE_METRICS["js_distance"](random_p),
                       _jsd_to_similarity(_profile_jsd_matrix(random_p))):
        failures.append("js_distance does not reproduce 100*(1-sqrt(JSD)) on the fixture")
    return f"contract holds for {sorted(PROFILE_METRICS)}"


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
