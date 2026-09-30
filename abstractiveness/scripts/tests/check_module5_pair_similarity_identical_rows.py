"""
Every grid cell of module 5's pair-similarity study is fitted on exactly the same rows.

The results table's n_pairs is copied onto every row, so it cannot show a per-cell dropna. This
check instead wraps cross_val_predict inside the module to record the rows each cell is really
fitted on, on a synthetic pair table with N_NAN rows missing only profile_js_distance_z.

Checks:
  1. One fit per grid cell, all on the same row count.
  2. That count equals TOTAL_ROWS - N_NAN, computed independently from the fixture, and is
     strictly below TOTAL_ROWS, so the shared dropna really ran.

Usage, from scripts/:
    python tests/check_module5_pair_similarity_identical_rows.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import modules.module_5_typicality_prediction as m5
from _report import run

CATEGORIES = ["birds", "clothing", "fruit", "furniture"]
PER_CATEGORY = 80
N_NAN = 50
TOTAL_ROWS = len(CATEGORIES) * PER_CATEGORY
EXPECTED_POST_DROPNA = TOTAL_ROWS - N_NAN


def make_synthetic_table(seed: int = 0) -> pd.DataFrame:
    """A pair table shaped like build_pair_similarity_design's, N_NAN rows missing only the z."""
    rng = np.random.default_rng(seed)
    rows = [{"category": category, "word_a": f"{category}_a{i}", "word_b": f"{category}_b{i}",
             "human_similarity": float(rng.uniform(1, 7)), "jaccard_pct": float(rng.uniform(0, 100)),
             "profile_js_distance": float(rng.uniform(0, 100)), "profile_js_distance_z": float(rng.normal())}
            for category in CATEGORIES for i in range(PER_CATEGORY)]
    table = pd.DataFrame(rows)
    table.loc[rng.choice(len(table), size=N_NAN, replace=False), "profile_js_distance_z"] = np.nan
    return table


def checks(failures: list) -> str:
    fixture = make_synthetic_table()
    real_cross_val_predict, real_build_design = m5.cross_val_predict, m5.build_pair_similarity_design
    recorded = []

    def recording_cross_val_predict(estimator, X, y, **kwargs):
        recorded.append(len(X))
        return real_cross_val_predict(estimator, X, y, **kwargs)

    m5.build_pair_similarity_design = lambda *args, **kwargs: fixture.copy()
    m5.cross_val_predict = recording_cross_val_predict
    try:
        results, _ = m5.run_pair_similarity(pd.DataFrame(), pd.DataFrame(), ["js_distance"])
    finally:
        m5.build_pair_similarity_design = real_build_design
        m5.cross_val_predict = real_cross_val_predict

    assert not results.empty, "run_pair_similarity returned empty results on the fixture"
    if len(recorded) != len(results):
        failures.append(f"expected one fit per grid cell ({len(results)}), got {len(recorded)}")
    if len(set(recorded)) != 1:
        failures.append(f"cells were fitted on different row counts: {recorded}")
    else:
        fitted_rows = recorded[0]
        if fitted_rows != EXPECTED_POST_DROPNA:
            failures.append(f"cells agree on {fitted_rows} rows, the fixture's post-dropna count is "
                            f"{EXPECTED_POST_DROPNA}")
        if not fitted_rows < TOTAL_ROWS:
            failures.append(f"fitted {fitted_rows} rows, not below the fixture's {TOTAL_ROWS}, "
                            "the dropna may not have run")
    return f"{len(recorded)} cells fitted on {recorded[0] if recorded else 0} rows each"


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
