"""f(b, a) = 1 - f(a, b) must hold for every grid cell under every reference.

The ranker sees differences phi(a) - phi(b), so the question "which of these two is more
typical" only has a meaning if presenting the pair the other way round flips the answer
exactly. That requires the intercept-free logistic model and the uncentred scaler together,
which is what _ranker() is built for and what this check pins.

Two arms per cell, and the second one is the reason this file is longer than it looks.

The MIRRORED arm reproduces what evaluate_ranker actually trains on, each pair entered twice
as (a, b) labelled 1 and (b, a) labelled 0. That construction is what the module relies on,
so it is worth pinning, but on its own it cannot catch a broken _ranker(). The mirrored
training set is invariant under (x, y) -> (-x, 1 - y), so the fitted loss satisfies
L(w, b) = L(w, -b) and the optimum is forced to b = 0 whatever fit_intercept says, while the
mirrored columns have exact mean 0 so a centring scaler subtracts nothing. Both defects
would pass here, measured, at 1e-16.

The DIRECT arm therefore fits the same model on an unmirrored, class-imbalanced, offset
training set, where nothing outside _ranker() suppresses an intercept or a shift. It asks the
property the docstring of _ranker() actually claims, that the FITTED MODEL is an antisymmetric
function of its input. An intercept breaks it (measured at 5e-2) and a centring scaler breaks
it (measured at 3e-1).

Run from scripts/: python tests/check_ranker_antisymmetry.py.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modules.module_9_typicality_prediction import (
    _ranker, build_feature_grid, resolve_features, profile_columns, REFERENCES,
    ACTIVE_METRICS)

TOLERANCE = 1e-9
ROWS = 80

rng = np.random.default_rng(1)
failures = []
grid = build_feature_grid(ACTIVE_METRICS)

for reference in REFERENCES:
    for cell in grid:
        cols = profile_columns(cell["metric"]) if cell["metric"] != "none" else {}
        width = len(resolve_features(cell, reference, cols))
        label = f"{reference}/{cell['model']}/{cell['metric']}"

        # The mirrored construction of evaluate_ranker.
        difference = rng.normal(size=(ROWS, width))
        labels = (rng.random(ROWS) > 0.5).astype(int)
        model = _ranker().fit(np.vstack([difference, -difference]),
                              np.concatenate([labels, 1 - labels]))
        total = model.predict_proba(difference)[:, 1] + model.predict_proba(-difference)[:, 1]
        if not np.allclose(total, 1.0, atol=TOLERANCE):
            failures.append(f"{label}: mirrored fit, max deviation "
                            f"{np.max(np.abs(total - 1.0)):.3e}")

        # The direct arm. Offset features and a 3 to 1 class split, so an intercept has
        # something to fit and a centring scaler has something to subtract.
        direct = rng.normal(size=(ROWS, width)) + 2.0
        direct_labels = (rng.random(ROWS) > 0.25).astype(int)
        if len(np.unique(direct_labels)) < 2:
            failures.append(f"{label}: the direct fixture is single-class, fix the seed")
            continue
        model = _ranker().fit(direct, direct_labels)
        total = model.predict_proba(direct)[:, 1] + model.predict_proba(-direct)[:, 1]
        if not np.allclose(total, 1.0, atol=TOLERANCE):
            failures.append(f"{label}: direct fit, max deviation "
                            f"{np.max(np.abs(total - 1.0)):.3e}")

if failures:
    print("FAIL check_ranker_antisymmetry")
    for line in failures:
        print("  " + line)
    sys.exit(1)
print(f"OK check_ranker_antisymmetry ({2 * len(REFERENCES) * len(grid)} fits)")
