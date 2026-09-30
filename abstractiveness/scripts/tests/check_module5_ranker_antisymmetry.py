"""
Module 5's pairwise typicality ranker is antisymmetric, f(b, a) = 1 - f(a, b), for every grid cell.

Checks, per reference and grid cell:
  1. Mirrored arm: the (a, b) plus (b, a) training set evaluate_ranker builds.
  2. Direct arm: an unmirrored, offset, class-imbalanced fit. The mirrored set alone would hide
     an intercept or a centring scaler, both of which break antisymmetry here.

Usage, from scripts/:
    python tests/check_module5_ranker_antisymmetry.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_5_typicality_prediction import (_ranker, build_feature_grid, resolve_features,
                                                    profile_columns, REFERENCES, ACTIVE_METRICS)
from _report import run

TOLERANCE = 1e-9
ROWS = 80


def checks(failures: list) -> str:
    rng = np.random.default_rng(1)
    grid = build_feature_grid(ACTIVE_METRICS)
    for reference in REFERENCES:
        for cell in grid:
            cols = profile_columns(cell["metric"]) if cell["metric"] != "none" else {}
            width = len(resolve_features(cell, reference, cols))
            label = f"{reference}/{cell['model']}/{cell['metric']}"

            difference = rng.normal(size=(ROWS, width))
            labels = (rng.random(ROWS) > 0.5).astype(int)
            model = _ranker().fit(np.vstack([difference, -difference]),
                                  np.concatenate([labels, 1 - labels]))
            total = model.predict_proba(difference)[:, 1] + model.predict_proba(-difference)[:, 1]
            if not np.allclose(total, 1.0, atol=TOLERANCE):
                failures.append(f"{label}: mirrored fit, max deviation {np.max(np.abs(total - 1.0)):.3e}")

            direct = rng.normal(size=(ROWS, width)) + 2.0
            direct_labels = (rng.random(ROWS) > 0.25).astype(int)
            if len(np.unique(direct_labels)) < 2:
                failures.append(f"{label}: the direct fixture is single-class, fix the seed")
                continue
            model = _ranker().fit(direct, direct_labels)
            total = model.predict_proba(direct)[:, 1] + model.predict_proba(-direct)[:, 1]
            if not np.allclose(total, 1.0, atol=TOLERANCE):
                failures.append(f"{label}: direct fit, max deviation {np.max(np.abs(total - 1.0)):.3e}")
    return f"{2 * len(REFERENCES) * len(grid)} fits"


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
