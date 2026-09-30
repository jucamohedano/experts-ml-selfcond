"""
The count-matched null standardizes against comparable real pairs and stays pinned to known values.

Checks:
  1. _empirical_pair_null reproduces the checksum recorded before count_matched_z was factored out.
  2. count_matched_z excludes each entry from its own reference set, so an outlier keeps a large z.
  3. Too few entries return all NaN instead of a z from a handful of references.
  4. The masking branch, words below MIN_PROFILE_EXPERTS, reproduces its pinned checksum and
     leaves every unusable pair NaN.

Usage, from scripts/:
    python tests/check_count_matched_z.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.shared_layer_profile_measures import (count_matched_z, _empirical_pair_null,
                                                   NULL_NEIGHBORS_MIN, MIN_PROFILE_EXPERTS)
from _report import run

EXPECTED = 16002.256342290246
EXPECTED_MASKED = 65628.51072386315


def weighted_checksum(z: np.ndarray) -> float:
    """Position-weighted sum of a z matrix, sensitive to any value or placement change."""
    return float(np.nansum(z * np.arange(z.size).reshape(z.shape)))


def checks(failures: list) -> str:
    rng = np.random.default_rng(7)
    n = 60
    sim = rng.uniform(0, 100, size=(n, n))
    sim = (sim + sim.T) / 2
    counts = rng.integers(2, 500, size=n).astype(float)
    checksum = weighted_checksum(_empirical_pair_null(sim, counts, neighbors=45))
    if not np.isclose(checksum, EXPECTED, rtol=1e-10):
        failures.append(f"pair null changed: checksum {checksum!r} != {EXPECTED!r}")

    values = np.concatenate([rng.normal(0.0, 1.0, 2 * NULL_NEIGHBORS_MIN), [100.0]])
    coords = np.stack([np.linspace(0, 1, len(values)), np.zeros(len(values))], axis=1)
    flat = count_matched_z(values, coords, neighbors=NULL_NEIGHBORS_MIN)
    if not (np.isfinite(flat[-1]) and flat[-1] > 5):
        failures.append(f"outlier z should be large positive, got {flat[-1]}")

    if not np.all(np.isnan(count_matched_z(np.array([1.0, 2.0, 3.0]), np.zeros((3, 2))))):
        failures.append("tiny input should return all-NaN")

    rng_masked = np.random.default_rng(11)
    n_masked = 60
    sim_masked = rng_masked.uniform(0, 100, size=(n_masked, n_masked))
    sim_masked = (sim_masked + sim_masked.T) / 2
    counts_masked = np.concatenate([rng_masked.integers(2, 500, size=40),
                                    rng_masked.integers(0, MIN_PROFILE_EXPERTS, size=20)]).astype(float)
    iu = np.triu_indices(n_masked, k=1)
    above_a = counts_masked[iu[0]] >= MIN_PROFILE_EXPERTS
    above_b = counts_masked[iu[1]] >= MIN_PROFILE_EXPERTS
    if not ((above_a & above_b).any() and (~above_a & ~above_b).any() and (above_a != above_b).any()):
        failures.append("masking fixture must cover both-above, both-below and one-below pairs")
    z_masked = _empirical_pair_null(sim_masked, counts_masked, neighbors=45)
    if not np.all(np.isnan(z_masked[iu][~(above_a & above_b)])):
        failures.append("masking branch: entries outside usable must be NaN")
    masked_checksum = weighted_checksum(z_masked)
    if not np.isclose(masked_checksum, EXPECTED_MASKED, rtol=1e-10):
        failures.append(f"masking branch changed: checksum {masked_checksum!r} != {EXPECTED_MASKED!r}")
    return f"checksums {checksum:.6f} and {masked_checksum:.6f}"


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
