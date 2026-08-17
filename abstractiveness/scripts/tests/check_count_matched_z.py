"""count_matched_z is the coordinate-agnostic core of the pair null.

Checks: (1) _empirical_pair_null still produces identical z after the refactor,
compared against values recorded from the pre-refactor implementation on a fixed
fixture, (2) count_matched_z standardizes against neighbors excluding self,
(3) too few entries yields all-NaN rather than a z from a handful of references,
(4) a fixture with some words below MIN_PROFILE_EXPERTS exercises the masking branch
that check (1) never reaches, since check (1) always has every pair usable, pinned
against a checksum captured from the current, already refactored implementation.
Run from scripts/: python tests/check_count_matched_z.py.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.helpers import (count_matched_z, _empirical_pair_null, NULL_NEIGHBORS_MIN,
                           MIN_PROFILE_EXPERTS)

failures = []
rng = np.random.default_rng(7)

# (1) Regression: the pair null on a random similarity matrix must be reproduced by
# the delegating implementation, against the checksum recorded from the pre-refactor
# implementation on this exact fixture (rng seed 7, n=60, neighbors=45).
n = 60
sim = rng.uniform(0, 100, size=(n, n))
sim = (sim + sim.T) / 2
counts = rng.integers(2, 500, size=n).astype(float)
z = _empirical_pair_null(sim, counts, neighbors=45)
checksum = float(np.nansum(z * np.arange(z.size).reshape(z.shape)))
EXPECTED = 16002.256342290246
if not np.isclose(checksum, EXPECTED, rtol=1e-10):
    failures.append(f"pair null changed: checksum {checksum!r} != {EXPECTED!r}")
print(f"pair-null checksum: {checksum!r}")

# (2) Self-exclusion: an extreme outlier appended to noisy baseline values must get a
# large positive z. If the outlier leaked into its own reference set, the mean would
# absorb it and the z would shrink far below this bound. The baseline is noise rather
# than exact zeros so the reference sd is nonzero.
values = np.concatenate([rng.normal(0.0, 1.0, 2 * NULL_NEIGHBORS_MIN), [100.0]])
coords = np.stack([np.linspace(0, 1, len(values)), np.zeros(len(values))], axis=1)
flat = count_matched_z(values, coords, neighbors=NULL_NEIGHBORS_MIN)
if not (np.isfinite(flat[-1]) and flat[-1] > 5):
    failures.append(f"outlier z should be large positive, got {flat[-1]}")

# (3) Too few entries -> all NaN.
tiny = count_matched_z(np.array([1.0, 2.0, 3.0]), np.zeros((3, 2)))
if not np.all(np.isnan(tiny)):
    failures.append("tiny input should return all-NaN")

# (4) Masking branch: a fixture where some words fall below MIN_PROFILE_EXPERTS, so
# usable is a proper subset of all pairs, covering pairs where one word is below the
# gate, both are below, and both are above. Check (1) above never reaches this branch,
# its counts are always drawn from [2, 500) so usable is all-True there, and this is
# exactly the branch where finite inside count_matched_z must reconstruct usable from
# the NaN holes _empirical_pair_null punches into pair_values and coords.
#
# EXPECTED_MASKED below was captured by running the CURRENT, already refactored
# _empirical_pair_null on this fixture, the pre-refactor implementation no longer
# exists in the repository to compare against directly. This pins the masking branch
# against FUTURE drift, it is not evidence the original refactor was correct, check (1)
# and its checksum already establish that for the all-usable path.
rng_masked = np.random.default_rng(11)
n_masked = 60
sim_masked = rng_masked.uniform(0, 100, size=(n_masked, n_masked))
sim_masked = (sim_masked + sim_masked.T) / 2
counts_masked = np.concatenate([
    rng_masked.integers(2, 500, size=40),
    rng_masked.integers(0, MIN_PROFILE_EXPERTS, size=20),
]).astype(float)

iu_masked = np.triu_indices(n_masked, k=1)
above_a = counts_masked[iu_masked[0]] >= MIN_PROFILE_EXPERTS
above_b = counts_masked[iu_masked[1]] >= MIN_PROFILE_EXPERTS
both_above = int((above_a & above_b).sum())
both_below = int((~above_a & ~above_b).sum())
one_below = int((above_a != above_b).sum())
if not (both_above > 0 and both_below > 0 and one_below > 0):
    failures.append("masking fixture must cover both-above, both-below and one-below "
                     f"pairs, got both_above={both_above}, both_below={both_below}, "
                     f"one_below={one_below}")

z_masked = _empirical_pair_null(sim_masked, counts_masked, neighbors=45)
usable_masked = above_a & above_b
flat_masked = z_masked[iu_masked]
if not np.all(np.isnan(flat_masked[~usable_masked])):
    failures.append("masking branch: entries outside usable must be NaN")

masked_checksum = float(np.nansum(z_masked * np.arange(z_masked.size).reshape(z_masked.shape)))
EXPECTED_MASKED = 65628.51072386315
if not np.isclose(masked_checksum, EXPECTED_MASKED, rtol=1e-10):
    failures.append(f"masking branch changed: checksum {masked_checksum!r} != {EXPECTED_MASKED!r}")
print(f"masked-branch checksum: {masked_checksum!r}")

if failures:
    print("FAIL check_count_matched_z")
    for line in failures:
        print("  " + line)
    sys.exit(1)
print("OK check_count_matched_z")
