"""Contract checks for the PROFILE_METRICS registry.

Every registered metric must return a square matrix on one input, a cross matrix on
two, and must be oriented so identical profiles score at least as high as disjoint
ones. The default metric must reproduce the historical 100*(1-sqrt(JSD)) values.
Run from scripts/: python tests/check_profile_metrics.py. Exits non-zero on failure.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.helpers import (PROFILE_METRICS, DEFAULT_PROFILE_METRIC,
                           _profile_jsd_matrix, _jsd_to_similarity)

failures = []

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
        failures.append(f"{name}: f(P) != f(P, P), square form must equal explicit cross")
    same = func(identical)[0, 1]
    far = func(disjoint)[0, 1]
    if not same > far:
        failures.append(f"{name}: orientation violated, identical={same:.4f} <= disjoint={far:.4f}")

if DEFAULT_PROFILE_METRIC != "js_distance":
    failures.append(f"default metric is {DEFAULT_PROFILE_METRIC}, expected js_distance")
expected = _jsd_to_similarity(_profile_jsd_matrix(random_p))
actual = PROFILE_METRICS["js_distance"](random_p)
if not np.allclose(actual, expected):
    failures.append("js_distance does not reproduce 100*(1-sqrt(JSD)) on the fixture")

if failures:
    print("FAIL check_profile_metrics")
    for line in failures:
        print("  " + line)
    sys.exit(1)
print("OK check_profile_metrics: registry contract holds for "
      f"{sorted(PROFILE_METRICS)} ")
