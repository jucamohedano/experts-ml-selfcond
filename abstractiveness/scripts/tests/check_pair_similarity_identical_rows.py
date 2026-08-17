"""Proves run_pair_similarity's identical-rows guarantee is genuinely enforced.

Why the obvious check is not enough. The results table's own n_pairs column is read once
from the single post-dropna table and copied onto every row, so set(results["n_pairs"]) ==
{k} would hold even if a per-cell dropna silently gave each cell a different row count,
because n_pairs never reflects what any individual cell was actually fitted on. That is a
tautology, not evidence.

What this check does instead. It wraps cross_val_predict itself, inside the module's own
namespace, to record the number of rows passed to X on every call, one call per grid cell.
That is the real row count each cell was fitted on, not a self-reported field. The fixture
is a synthetic pair table across four real categories (their names must exist in
human_noise_ceiling's registry) with a KNOWN, non-zero number of rows, N_NAN, carrying NaN
in profile_js_distance_z while jaccard_pct, profile_js_distance and human_similarity stay
valid everywhere. A per-cell dropna would give the cells that read profile_js_distance_z
(profile_z, jaccard_profile_z, jaccard_profile_both) TOTAL_ROWS - N_NAN rows and the other
three cells TOTAL_ROWS rows, a real and detectable divergence. run_pair_similarity's own
build_pair_similarity_design is monkeypatched out, since this check is about the row-identity
machinery inside run_pair_similarity, not about pair construction, which
check_pair_similarity_target.py already covers.

Three assertions, in increasing strength. One call per grid cell. Every call sees the SAME
row count. That count equals TOTAL_ROWS - N_NAN exactly, the value computed independently
from the fixture, not merely agreement among the six calls, which is what made the earlier
n_pairs-based evidence tautological. And that count is strictly less than TOTAL_ROWS, so the
shared dropna is proven to have actually removed the NaN rows rather than silently doing
nothing.

Run from scripts/: python tests/check_pair_similarity_identical_rows.py.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import modules.module_9_typicality_prediction as m9

# Real category names, human_noise_ceiling only knows entries of CATEGORY_FILES.
CATEGORIES = ["birds", "clothing", "fruit", "furniture"]
PER_CATEGORY = 80
N_NAN = 50
TOTAL_ROWS = len(CATEGORIES) * PER_CATEGORY
EXPECTED_POST_DROPNA = TOTAL_ROWS - N_NAN


def make_synthetic_table(seed: int = 0) -> pd.DataFrame:
    """A pair table shaped like build_pair_similarity_design's output, with a known, fixed
    number of rows carrying NaN in profile_js_distance_z only."""
    rng = np.random.default_rng(seed)
    rows = []
    for category in CATEGORIES:
        for i in range(PER_CATEGORY):
            rows.append({
                "category": category, "word_a": f"{category}_a{i}", "word_b": f"{category}_b{i}",
                "human_similarity": float(rng.uniform(1, 7)),
                "jaccard_pct": float(rng.uniform(0, 100)),
                "profile_js_distance": float(rng.uniform(0, 100)),
                "profile_js_distance_z": float(rng.normal()),
            })
    table = pd.DataFrame(rows)
    nan_rows = rng.choice(len(table), size=N_NAN, replace=False)
    table.loc[nan_rows, "profile_js_distance_z"] = np.nan
    return table


def run_against_real_module() -> tuple[list, list]:
    """Runs the REAL run_pair_similarity on the synthetic fixture, recording the row count
    fed to every cross_val_predict call. Returns (failures, recorded_counts)."""
    failures = []
    fixture = make_synthetic_table()

    real_cross_val_predict = m9.cross_val_predict
    real_build_design = m9.build_pair_similarity_design
    recorded = []

    def recording_cross_val_predict(estimator, X, y, **kwargs):
        recorded.append(len(X))
        return real_cross_val_predict(estimator, X, y, **kwargs)

    m9.build_pair_similarity_design = lambda *args, **kwargs: fixture.copy()
    m9.cross_val_predict = recording_cross_val_predict
    results = pd.DataFrame()
    try:
        results, _ = m9.run_pair_similarity(pd.DataFrame(), pd.DataFrame(), ["js_distance"])
    except Exception as exc:  # noqa: BLE001, a broken implementation may not even reach a
        # DataFrame, and a crash is itself evidence the identical-rows guarantee failed, not
        # something this check should let escape as an uncaught traceback.
        failures.append(f"run_pair_similarity raised {type(exc).__name__}: {exc}")
    finally:
        m9.build_pair_similarity_design = real_build_design
        m9.cross_val_predict = real_cross_val_predict

    if failures:
        # A crash does not excuse the row-count evidence, if the per-cell calls that ran
        # before the crash already diverged, say so explicitly rather than only reporting
        # the exception.
        if len(set(recorded)) > 1:
            failures.append(f"the per-cell fit calls made before the crash had already "
                            f"diverged: {recorded}")
        return failures, recorded

    if results.empty:
        failures.append("run_pair_similarity returned empty results on the synthetic fixture, "
                        "cannot verify")
        return failures, recorded

    if len(recorded) != len(results):
        failures.append(f"expected one cross_val_predict call per grid cell ({len(results)} "
                        f"cells), got {len(recorded)} calls")

    distinct = set(recorded)
    if len(distinct) != 1:
        failures.append(f"cells were fitted on DIFFERENT row counts, the identical-rows "
                        f"guarantee is broken: {recorded}")
    else:
        fitted_rows = distinct.pop()
        if fitted_rows != EXPECTED_POST_DROPNA:
            failures.append(f"every cell agreed on {fitted_rows} rows, but the fixture's "
                            f"independently computed post-dropna count is "
                            f"{EXPECTED_POST_DROPNA}, agreement alone is not enough evidence")
        if not (fitted_rows < TOTAL_ROWS):
            failures.append(f"fitted row count {fitted_rows} is not strictly less than the "
                            f"pre-dropna fixture size {TOTAL_ROWS}, the dropna may not have run "
                            f"at all")

    return failures, recorded


def main() -> int:
    failures, recorded = run_against_real_module()
    if failures:
        print("FAIL check_pair_similarity_identical_rows")
        for line in failures:
            print("  " + line)
        print(f"  recorded per-cell fitted row counts (one per grid cell, in cell order): "
             f"{recorded}")
        return 1
    print(f"OK check_pair_similarity_identical_rows, all {len(recorded)} grid cells fitted on "
         f"exactly {recorded[0]} rows ({TOTAL_ROWS} in the fixture, {N_NAN} dropped for NaN "
         f"profile_js_distance_z)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
