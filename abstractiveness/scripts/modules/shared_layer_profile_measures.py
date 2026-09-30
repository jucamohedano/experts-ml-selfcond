"""
Layer-profile agreement: the registry of eight profile metrics and the count-matched null.

Shared by modules 2 and 5. Every metric returns an agreement matrix oriented so higher means
more similar, and z standardizes it against real pairs of comparable expert counts.

Explanations: documentation/module_2_similarities.md.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import xlogy
from modules.shared_layer_distribution_measures import build_layer_probability_matrix


# Below this many experts a word has no usable layer profile.
MIN_PROFILE_EXPERTS = 2

_JSD_BLOCK_ROWS = 32

_SQRT_LN2 = np.sqrt(np.log(2.0))

# Count-matched null: neighbours as a fraction of available pairs, clipped.
NULL_NEIGHBOR_FRACTION = 0.01

NULL_NEIGHBORS_MIN = 40

NULL_NEIGHBORS_MAX = 300


def _entropy_bits(prob: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits along the last axis."""
    return -xlogy(prob, prob).sum(axis=-1) / np.log(2.0)


def _profile_jsd_matrix(prob: np.ndarray) -> np.ndarray:
    """Pairwise Jensen-Shannon divergence in bits between the rows of ``prob``, as a square (n, n) array."""
    n = len(prob)
    row_entropy = _entropy_bits(prob)
    mixture_entropy = np.empty((n, n), dtype=np.float64)

    for start in range(0, n, _JSD_BLOCK_ROWS):
        stop = min(start + _JSD_BLOCK_ROWS, n)
        mixture = 0.5 * (prob[start:stop, None, :] + prob[None, :, :])
        mixture_entropy[start:stop] = _entropy_bits(mixture)

    jsd = mixture_entropy - 0.5 * (row_entropy[:, None] + row_entropy[None, :])
    return np.clip(jsd, 0.0, 1.0)


def _jsd_to_similarity(jsd: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits to a 0-100 similarity, 100 * (1 - sqrt(JSD))."""
    return 100.0 * (1.0 - np.sqrt(jsd))


def _zero_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairs where either profile carries no mass at all, as an (n, m) boolean mask."""
    return (a.sum(axis=1)[:, None] <= 0) | (b.sum(axis=1)[None, :] <= 0)


def _js_distance_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Jensen-Shannon distance in bits, sqrt(JSD), from scipy's kernel."""
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = cdist(a, b, metric="jensenshannon") / _SQRT_LN2
    return np.where(_zero_mass(a, b), np.nan, distance)


def _js_distance_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """The historical profile agreement, 100 * (1 - sqrt(JSD bits))."""
    return 100.0 * (1.0 - _js_distance_bits(*_pair_inputs(p, q)))


def _js_divergence_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """Jensen-Shannon DIVERGENCE as an agreement, 100 * (1 - JSD bits)."""
    return 100.0 * (1.0 - _js_distance_bits(*_pair_inputs(p, q)) ** 2)


def _pair_inputs(p: np.ndarray, q: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """The (P, Q) a cross-form kernel operates on."""
    return (p, p) if q is None else (p, q)


def _correlation_agreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """100 * Pearson r between every row of a and every row of b, via cdist's `correlation` kernel, which returns 1 - r."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (1.0 - cdist(a, b, metric="correlation"))


def _cosine_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """Cosine agreement between layer profiles, 100 * cos(p, q), via cdist's `cosine` kernel, which returns 1 - cos."""
    a, b = _pair_inputs(p, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        agreement = 100.0 * (1.0 - cdist(a, b, metric="cosine"))
    return np.where(_zero_mass(a, b), np.nan, agreement)


def _pearson_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """Pearson correlation between layer profiles across bins, on a 100 * r scale."""
    return _correlation_agreement(*_pair_inputs(p, q))


def _spearman_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """Spearman rank correlation between layer profiles across bins, on a 100 * rho scale."""
    a, b = _pair_inputs(p, q)
    return _correlation_agreement(stats.rankdata(a, axis=1), stats.rankdata(b, axis=1))


def _hellinger_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """Hellinger agreement, 100 * (1 - H), with H = sqrt(1 - BC) and BC the Bhattacharyya coefficient sum_k sqrt(p_k q_k)."""
    a, b = _pair_inputs(p, q)
    distance = cdist(np.sqrt(a), np.sqrt(b), metric="euclidean") / np.sqrt(2.0)
    return np.where(_zero_mass(a, b), np.nan, 100.0 * (1.0 - distance))


def _profile_jaccard_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """Weighted Jaccard (Ruzicka) agreement between layer profiles, on a 0 to 100 scale."""
    a, b = _pair_inputs(p, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        total_variation = cdist(a, b, metric="braycurtis")
        jaccard = (1.0 - total_variation) / (1.0 + total_variation)
    return np.where(_zero_mass(a, b), np.nan, 100.0 * jaccard)


def _wasserstein_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """First Wasserstein (earth mover's) agreement over the bin axis, on a 0 to 100 scale."""
    a, b = _pair_inputs(p, q)
    n_bins = a.shape[1]
    if n_bins < 2:
        return np.full((len(a), len(b)), 100.0)

    bins = np.arange(n_bins, dtype=np.float64)
    empty = _zero_mass(a, b)
    distance = np.full((len(a), len(b)), np.nan)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            if not empty[i, j]:
                distance[i, j] = stats.wasserstein_distance(bins, bins, u, v)
    return 100.0 * (1.0 - np.clip(distance / (n_bins - 1), 0.0, 1.0))


# Registry contract: f(P, Q=None) -> agreement matrix, higher is more similar.
PROFILE_METRICS = {
    "js_distance": _js_distance_similarity,
    "wasserstein": _wasserstein_similarity,
    "cosine": _cosine_similarity,
    "pearson": _pearson_similarity,
    "spearman": _spearman_similarity,
    "js_divergence": _js_divergence_similarity,
    "hellinger": _hellinger_similarity,
    "profile_jaccard": _profile_jaccard_similarity,
}

DEFAULT_PROFILE_METRIC = "js_distance"

# Metrics every consumer computes, the default first.
ACTIVE_PROFILE_METRICS = list(PROFILE_METRICS)

PROFILE_METRIC_LABELS = {
    "js_distance": "Jensen-Shannon distance",
    "wasserstein": "Wasserstein",
    "cosine": "Cosine",
    "pearson": "Pearson",
    "spearman": "Spearman",
    "js_divergence": "Jensen-Shannon divergence",
    "hellinger": "Hellinger",
    "profile_jaccard": "Weighted Jaccard",
}

# Metrics whose agreement can be negative.
SIGNED_PROFILE_METRICS = {"pearson", "spearman"}


def profile_metric_label(metric: str) -> str:
    """Display name for a registered metric, falling back to the raw key."""
    return PROFILE_METRIC_LABELS.get(metric, metric)


def profile_metric_value_label(metric: str) -> str:
    """Axis label for a metric's raw agreement value."""
    if metric in SIGNED_PROFILE_METRICS:
        return f"{profile_metric_label(metric)} correlation x 100"
    return "Agreement % (100 = identical profiles)"


def count_matched_z(values: np.ndarray, coords: np.ndarray,
                    neighbors: int = None) -> np.ndarray:
    """Standardize each entry of ``values`` against its nearest entries in ``coords`` space, self excluded."""
    z = np.full(len(values), np.nan)
    finite = np.isfinite(values) & np.isfinite(coords).all(axis=1)
    if finite.sum() < 2 * NULL_NEIGHBORS_MIN:
        return z

    usable_values = values[finite]
    usable_coords = coords[finite]
    if neighbors is None:
        neighbors = int(np.clip(round(NULL_NEIGHBOR_FRACTION * len(usable_values)),
                                NULL_NEIGHBORS_MIN, NULL_NEIGHBORS_MAX))
    k = min(neighbors, len(usable_values) - 1)
    _, idx = cKDTree(usable_coords).query(usable_coords, k=k + 1)

    self_match = idx == np.arange(len(usable_values))[:, None]
    drop = self_match & (self_match.cumsum(axis=1) == 1)
    drop[~self_match.any(axis=1), -1] = True
    keep = idx[~drop].reshape(len(usable_values), k)

    reference = usable_values[keep]
    mu = reference.mean(axis=1)
    sigma = reference.std(axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        z[finite] = np.where(sigma > 0, (usable_values - mu) / sigma, np.nan)
    return z


def _empirical_pair_null(similarity: np.ndarray, counts: np.ndarray,
                         neighbors: int = None) -> np.ndarray:
    """Standardize each pair's similarity against OTHER REAL PAIRS of comparable expert counts, returning the square (n, n) array of z-scores."""
    n = len(counts)
    z = np.full((n, n), np.nan)
    iu = np.triu_indices(n, k=1)
    pair_values = similarity[iu]
    n_a, n_b = counts[iu[0]], counts[iu[1]]

    usable = np.isfinite(pair_values) & (n_a >= MIN_PROFILE_EXPERTS) & (n_b >= MIN_PROFILE_EXPERTS)
    pair_values_masked = np.where(usable, pair_values, np.nan)
    coords = np.full((len(pair_values), 2), np.nan)
    coords[usable] = np.stack([np.log(np.minimum(n_a, n_b)[usable]),
                               np.log(np.maximum(n_a, n_b)[usable])], axis=1)

    flat = count_matched_z(pair_values_masked, coords, neighbors)
    z[iu] = flat
    z.T[iu] = flat
    return z


def _layer_profiles(expert_allocation_df: pd.DataFrame, items: list) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalized layer profiles and expert counts for ``items``, in the given order."""
    count_matrix, prob_matrix = build_layer_probability_matrix(expert_allocation_df)
    prob = prob_matrix.reindex(items).to_numpy(dtype=np.float64)
    counts = count_matrix.reindex(items).sum(axis=1).to_numpy(dtype=np.float64)
    return prob, counts


def layer_profile_matrices(expert_allocation_df: pd.DataFrame, items: list, *,
                           null_neighbors: int = None,
                           metric: str = DEFAULT_PROFILE_METRIC
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise layer-profile agreement for every pair of ``items``, as three square (n, n) arrays: (jsd_bits, similarity_pct, z)."""
    return layer_profile_metric_matrices(expert_allocation_df, items, [metric],
                                         null_neighbors=null_neighbors)[metric]


def layer_profile_metric_matrices(expert_allocation_df: pd.DataFrame, items: list,
                                  metrics: list = None, *,
                                  null_neighbors: int = None) -> dict:
    """layer_profile_matrices for several metrics at once, as {metric: (jsd_bits, similarity, z)}."""
    metrics = list(ACTIVE_PROFILE_METRICS if metrics is None else metrics)
    prob, counts = _layer_profiles(expert_allocation_df, items)
    n = len(items)
    usable = np.isfinite(counts) & (counts >= MIN_PROFILE_EXPERTS) & np.isfinite(prob).all(axis=1)

    if usable.sum() < 2:
        return {metric: (np.full((n, n), np.nan), np.full((n, n), np.nan),
                         np.full((n, n), np.nan)) for metric in metrics}

    idx = np.flatnonzero(usable)
    block = np.ix_(idx, idx)
    jsd = np.full((n, n), np.nan)
    jsd[block] = _profile_jsd_matrix(prob[idx])

    out = {}
    for metric in metrics:
        similarity = np.full((n, n), np.nan)
        similarity[block] = PROFILE_METRICS[metric](prob[idx])
        z = _empirical_pair_null(similarity, counts, null_neighbors)
        np.fill_diagonal(z, np.nan)
        out[metric] = (jsd.copy(), similarity, z)
    return out


def pair_layer_profile_vectors(expert_allocation_df: pd.DataFrame, items: list,
                               **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Layer-profile similarity and its null z-score for every unordered pair of ``items``, as two flat vectors aligned with np.triu_indices(len(items), k=1), mirroring pair_similarity_vector."""
    _, similarity, z = layer_profile_matrices(expert_allocation_df, items, **kwargs)
    iu = np.triu_indices(len(items), k=1)
    return similarity[iu], z[iu]
