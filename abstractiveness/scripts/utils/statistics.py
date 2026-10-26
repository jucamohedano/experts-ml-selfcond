"""
Textbook statistics shared by several files: Spearman's rho over all items or within groups, its partial form, the
two-sided permutation p of both, and the Spearman-Brown correction.

Used by every correlation and permutation test of modules 1 to 5 and by the human-rating noise ceiling.

Explanations: documentation/README.md, section "Correlations and permutation tests".
"""

import functools
import itertools
import math
import zlib

import numpy as np
from scipy import stats

# Random reassignments behind every permutation p, replaced by every distinct one when there are at most this many.
PERMUTATIONS = 9999
MAX_EXACT_REASSIGNMENTS = math.factorial(9)

# Smallest group, beyond one item per control, that enters a within-group coefficient.
MIN_GROUP_SIZE = 3

_CHUNK_ELEMENTS = 4_000_000
_TOLERANCE = 1e-12


def spearman_brown(reliability: float) -> float:
    """Spearman-Brown correction for a split-half reliability, 2r / (1 + r)."""
    if not np.isfinite(reliability) or reliability <= -1.0:
        return np.nan
    return float(2.0 * reliability / (1.0 + reliability))


def _group_index(groups, n: int) -> tuple[list, list]:
    """Sorted group names and the positions of each, one unnamed group without groups."""
    if groups is None:
        return [None], [np.arange(n)]
    groups = np.asarray(groups)
    names = sorted(set(groups.tolist()))
    return names, [np.flatnonzero(groups == name) for name in names]


def rank_scores(values, groups=None) -> np.ndarray:
    """Average ranks standardised to mean 0 and population sd 1, over all items or within each group, 0 where constant."""
    values = np.asarray(values, dtype=float)
    scores = np.zeros(len(values))
    for rows in _group_index(groups, len(values))[1]:
        ranks = stats.rankdata(values[rows])
        if ranks.std() > 0:
            scores[rows] = (ranks - ranks.mean()) / ranks.std()
    return scores


def residuals(values, controls=(), groups=None) -> np.ndarray:
    """Values minus their least-squares fit on the intercept and the controls, fitted within each group."""
    values = np.asarray(values, dtype=float)
    controls = [np.asarray(control, dtype=float) for control in controls]
    out = np.empty(len(values))
    for rows in _group_index(groups, len(values))[1]:
        design = np.column_stack([np.ones(len(rows)), *(control[rows] for control in controls)])
        out[rows] = values[rows] - design @ np.linalg.lstsq(design, values[rows], rcond=None)[0]
    return out


def permutation_p(observed, null):
    """Two-sided permutation p per column of null, the observed value counted among the reassignments, NaN where the
    observed value is undefined."""
    observed, null = np.asarray(observed, dtype=float), np.asarray(null, dtype=float)
    hits = (np.abs(null) >= np.abs(observed) - _TOLERANCE).sum(axis=0)
    p = np.where(np.isfinite(observed), (1 + hits) / (1 + len(null)), np.nan)
    return float(p) if p.ndim == 0 else p


def _arrangement_count(codes: np.ndarray) -> int:
    """Number of distinct arrangements of a sequence of codes."""
    counts = np.unique(codes, return_counts=True)[1]
    return math.factorial(len(codes)) // math.prod(math.factorial(int(count)) for count in counts)


@functools.lru_cache(maxsize=8)
def _arrangements(codes: tuple) -> np.ndarray:
    """Positions rearranging the codes into each of their distinct arrangements, one row per arrangement."""
    codes = np.asarray(codes)
    if len(np.unique(codes)) == len(codes):
        return np.array(list(itertools.permutations(range(len(codes)))), dtype=np.intp).reshape(-1, len(codes))
    values, counts = np.unique(codes, return_counts=True)
    sources = [np.flatnonzero(codes == value) for value in values]
    rows, assignment = [], np.empty(len(codes), dtype=np.intp)

    def place(level: int, free: list) -> None:
        if level == len(values):
            rows.append(assignment.copy())
            return
        for chosen in itertools.combinations(free, counts[level]):
            assignment[list(chosen)] = sources[level]
            place(level + 1, [position for position in free if position not in chosen])

    place(0, list(range(len(codes))))
    return np.array(rows, dtype=np.intp)


def reassignments(codes, groups=None, n_draws: int = PERMUTATIONS, seed: int = 0) -> np.ndarray:
    """Index arrays reassigning the items within each group, every distinct arrangement but the observed one when
    few enough, else n_draws random ones."""
    codes = np.unique(np.asarray(codes), return_inverse=True)[1].ravel()
    names, index = _group_index(groups, len(codes))
    total = math.prod(_arrangement_count(codes[rows]) for rows in index)
    if total <= MAX_EXACT_REASSIGNMENTS:
        per_group = [rows[_arrangements(tuple(codes[rows].tolist()))] for rows in index]
        draws = np.empty((total, len(codes)), dtype=np.intp)
        if len(index) == 1:
            draws[:, index[0]] = per_group[0]
        for row, combination in enumerate(itertools.product(*per_group) if len(index) > 1 else ()):
            for rows, chosen in zip(index, combination):
                draws[row, rows] = chosen
        return draws[~(codes[draws] == codes).all(axis=1)]
    draws = np.empty((n_draws, len(codes)), dtype=np.intp)
    for name, rows in zip(names, index):
        rng = np.random.default_rng(seed if name is None else [seed, zlib.crc32(str(name).encode())])
        draws[:, rows] = rows[rng.permuted(np.tile(np.arange(len(rows)), (n_draws, 1)), axis=1)]
    return draws


def _complete_pairs(rows: np.ndarray, cols: np.ndarray, groups, usable: np.ndarray) -> np.ndarray:
    """The usable pairs left once, within each group, the item missing the most pairs is dropped until every two
    remaining items form a usable pair, ties going to the first item."""
    keep = np.zeros(len(rows), dtype=bool)
    for positions in _group_index(groups, len(rows))[1]:
        items, slots = np.unique(np.concatenate([rows[positions], cols[positions]]), return_inverse=True)
        first, second = slots[:len(positions)], slots[len(positions):]
        good = usable[positions]
        linked = np.eye(len(items), dtype=bool)
        linked[first[good], second[good]] = linked[second[good], first[good]] = True
        alive = np.ones(len(items), dtype=bool)
        while alive.any():
            missing = (~linked[np.ix_(alive, alive)]).sum(axis=1)
            if missing.max() == 0:
                break
            alive[np.flatnonzero(alive)[missing.argmax()]] = False
        keep[positions] = good & alive[first] & alive[second]
    return keep


def _usable(x, y, controls, groups, pairs) -> np.ndarray:
    """Values with x, y and every control finite, completed into whole pair sets, in groups large enough."""
    used = np.isfinite(x) & np.isfinite(y)
    for control in controls:
        used &= np.isfinite(control)
    if pairs is not None:
        used = _complete_pairs(*pairs, groups, used)
    if groups is not None:
        for name in set(groups[used].tolist()):
            if (used & (groups == name)).sum() < MIN_GROUP_SIZE + len(controls):
                used &= groups != name
    return used


def _prepare(x: np.ndarray, y: np.ndarray, controls: list, index: list) -> list:
    """Per group: positions, orthonormal basis of the intercept and controls, residual of y, its norm, and the
    reassignment-invariant sums of squares of x."""
    prepared = []
    for rows in index:
        design = np.column_stack([np.ones(len(rows)), *(control[rows] for control in controls)])
        u, s, _ = np.linalg.svd(design, full_matrices=False)
        basis = u[:, s > 1e-10 * s[0]]
        residual = y[rows] - basis @ (basis.T @ y[rows])
        norm = np.sqrt(residual @ residual)
        spread = ((y[rows] - y[rows].mean()) ** 2).sum()
        prepared.append((rows, basis, residual, norm if norm ** 2 > 1e-12 * spread else np.nan,
                         x[rows] @ x[rows], ((x[rows] - x[rows].mean()) ** 2).sum()))
    return prepared


def _group_correlations(X: np.ndarray, prepared: list) -> np.ndarray:
    """Per row of X and per group, the correlation of the residuals of that row and of y."""
    out = np.empty((X.shape[0], len(prepared)))
    for k, (rows, basis, residual, norm, squares, spread) in enumerate(prepared):
        block = X[:, rows]
        remaining = squares - ((block @ basis) ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, k] = np.where(remaining > 1e-12 * spread,
                                 (block @ residual) / (np.sqrt(np.maximum(remaining, 0.0)) * norm), np.nan)
    return out


def _standardised_residuals(values: np.ndarray, prepared: list) -> np.ndarray:
    """Residuals of values on each group's basis, standardised within the group, NaN where nothing is left."""
    out = np.full(len(values), np.nan)
    for rows, basis, *_ in prepared:
        residual = values[rows] - basis @ (basis.T @ values[rows])
        squares = residual @ residual
        if squares > 1e-12 * ((values[rows] - values[rows].mean()) ** 2).sum():
            out[rows] = residual * np.sqrt(len(rows) / squares)
    return out


def _draw_chunks(x: np.ndarray, groups, pairs, n_draws: int, seed: int):
    """Positions of the values under each reassignment, in chunks: the values of x reassigned among the items, or
    with pairs the labels of the items (Mantel)."""
    chunk = max(1, _CHUNK_ELEMENTS // max(1, len(x)))
    if pairs is None:
        draws = reassignments(x, groups, n_draws, seed)
        yield from (draws[start:start + chunk] for start in range(0, len(draws), chunk))
        return
    items, slots = np.unique(np.concatenate(pairs), return_inverse=True)
    first, second = slots[:len(x)], slots[len(x):]
    item_groups = None
    if groups is not None:
        item_groups = np.empty(len(items), dtype=groups.dtype)
        item_groups[first] = groups
        item_groups[second] = groups
        if (item_groups[first] != groups).any():
            raise ValueError("every pair must join two items of the same group")
    lookup = np.full((len(items), len(items)), -1, dtype=np.intp)
    lookup[first, second] = lookup[second, first] = np.arange(len(x))
    labels = reassignments(np.arange(len(items)), item_groups, n_draws, seed)
    for start in range(0, len(labels), chunk):
        block = labels[start:start + chunk]
        yield lookup[block[:, first], block[:, second]]


def spearman_test(x, y, controls=(), groups=None, pairs=None, ranks: bool = True,
                  n_draws: int = PERMUTATIONS, seed: int = 0) -> dict:
    """Spearman's rho of x and y, partial on the controls, pooled, size-weighted within groups or over item pairs,
    with its two-sided permutation p, the per-group results and the rank residuals whose slope is rho."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    controls = [np.asarray(control, dtype=float) for control in controls]
    groups = None if groups is None else np.asarray(groups)
    pairs = None if pairs is None else tuple(np.asarray(side) for side in pairs)
    used = _usable(x, y, controls, groups, pairs)

    kept_groups = None if groups is None else groups[used]
    names, index = _group_index(kept_groups, int(used.sum()))
    score = (lambda values: rank_scores(values[used], kept_groups)) if ranks else (lambda values: values[used])
    zx, zy = score(x), score(y)
    prepared = _prepare(zx, zy, [score(control) for control in controls], index) if used.any() else []
    sizes = np.array([len(rows) for rows in index]) if used.any() else np.zeros(0, dtype=int)
    observed = _group_correlations(zx[None, :], prepared)[0] if used.any() else np.zeros(0)
    valid = np.isfinite(observed)
    weights = np.where(valid, sizes, 0) / max(1, sizes[valid].sum()) if len(sizes) else sizes
    total = float(observed[valid] @ weights[valid]) if valid.any() else np.nan

    p, p_k = np.nan, np.full(len(observed), np.nan)
    if n_draws and valid.any():
        kept_pairs = None if pairs is None else tuple(side[used] for side in pairs)
        null_k = np.vstack([_group_correlations(zx[chunk], prepared)
                            for chunk in _draw_chunks(x[used], kept_groups, kept_pairs, n_draws, seed)])
        p = permutation_p(total, np.nan_to_num(null_k[:, valid]) @ weights[valid])
        p_k = permutation_p(observed, np.nan_to_num(null_k, nan=0.0))

    x_scores, y_scores = np.full(len(x), np.nan), np.full(len(x), np.nan)
    if used.any():
        x_scores[used], y_scores[used] = _standardised_residuals(zx, prepared), _standardised_residuals(zy, prepared)
    n = int(sizes[valid].sum()) if valid.any() else int(used.sum())
    return {"rho": total, "p": p, "n": n, "groups": names if used.any() else [],
            "n_k": sizes, "rho_k": observed, "p_k": p_k, "used": used, "x_scores": x_scores, "y_scores": y_scores}


def correlation(x, y, controls=(), groups=None, ranks: bool = True) -> float:
    """Spearman's rho of spearman_test without the test, or with ranks=False Pearson's r, partial on the controls."""
    return spearman_test(x, y, controls, groups, ranks=ranks, n_draws=0)["rho"]
