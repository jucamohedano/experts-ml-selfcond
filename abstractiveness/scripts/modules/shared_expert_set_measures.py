"""
Expert-set measures: which neurons two words share (Jaccard, overlap) and category alignment.

Shared by modules 1, 2, 4 and 5. Expert sets are held sparsely, never as a dense pivot.

Explanations: documentation/module_2_similarities.md and module_1_expert_distribution.md.
"""

import numpy as np
import pandas as pd
from scipy import sparse, stats

from utils.statistics import PERMUTATIONS, spearman_test


def expert_presence_matrix(expert_sets_df: pd.DataFrame, concepts: list) -> sparse.csr_matrix:
    """Binary concept-by-expert presence matrix, one row per entry of ``concepts`` in the given order and one column per distinct (layer_idx, unit) pair present in expert_sets_df."""
    row_of = {concept: i for i, concept in enumerate(concepts)}
    rows = expert_sets_df["concept"].map(row_of)
    keep = rows.notna().to_numpy()

    layer_idx = expert_sets_df.loc[keep, "layer_idx"].to_numpy(dtype=np.int64)
    unit = expert_sets_df.loc[keep, "unit"].to_numpy(dtype=np.int64)
    if len(layer_idx) == 0:
        return sparse.csr_matrix((len(concepts), 0), dtype=np.int32)

    pair_codes = pd.factorize(layer_idx * (unit.max() + 1) + unit)[0]

    presence = sparse.csr_matrix(
        (np.ones(len(pair_codes), dtype=np.int32),
         (rows.to_numpy()[keep].astype(np.int32), pair_codes)),
        shape=(len(concepts), pair_codes.max() + 1), dtype=np.int32)
    presence.data[:] = 1
    return presence


def expert_set_overlap_matrices(expert_sets_df: pd.DataFrame, concepts: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise expert-set overlap for every concept pair, as three square (n, n) arrays aligned with ``concepts``: (intersection, jaccard, overlap)."""
    presence = expert_presence_matrix(expert_sets_df, concepts)
    intersection = np.asarray((presence @ presence.T).todense(), dtype=np.float64)

    sizes = np.asarray(presence.sum(axis=1)).ravel()
    union = sizes[:, None] + sizes[None, :] - intersection
    min_size = np.minimum(sizes[:, None], sizes[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)
        overlap = np.where(min_size > 0, intersection / min_size, 0.0)
    return intersection, jaccard, overlap


def category_alignment_metrics(pair_similarity: np.ndarray, concept_categories: np.ndarray,
                               n_draws: int = PERMUTATIONS) -> dict:
    """ROC-AUC of pair similarity for same-category pairs, with the two-sided Mantel p of the equivalent Spearman rho
    against the same-category indicator, the mean similarity within and across categories and the number of pairs,
    over the pairs left complete once words with missing pairs are dropped. pair_similarity follows
    np.triu_indices(n, k=1)."""
    rows, cols = np.triu_indices(len(concept_categories), k=1)
    same = concept_categories[rows] == concept_categories[cols]
    test = spearman_test(same.astype(float), pair_similarity, pairs=(rows, cols), n_draws=n_draws)
    similarity, same = pair_similarity[test["used"]], same[test["used"]]
    n1, n0 = int(same.sum()), int((~same).sum())
    if not n1 or not n0:
        return {"roc_auc": np.nan, "mantel_p": np.nan, "within_mean": np.nan, "across_mean": np.nan, "n_pairs": n1 + n0}
    ranks = stats.rankdata(similarity)
    return {"roc_auc": (ranks[same].sum() - n1 * (n1 + 1) / 2) / (n1 * n0),
            "mantel_p": test["p"],
            "within_mean": similarity[same].mean(), "across_mean": similarity[~same].mean(), "n_pairs": n1 + n0}


def pair_similarity_vector(expert_sets_df: pd.DataFrame, concepts: list) -> np.ndarray:
    """Jaccard similarity of raw expert sets for every unordered pair of ``concepts``, as a flat vector aligned with np.triu_indices(len(concepts), k=1)."""
    _, jaccard, _ = expert_set_overlap_matrices(expert_sets_df, concepts)
    return jaccard[np.triu_indices(len(concepts), k=1)]


def pair_shared_count_vector(expert_sets_df: pd.DataFrame, concepts: list) -> np.ndarray:
    """Raw count of shared experts for every unordered concept pair, aligned with np.triu_indices(len(concepts), k=1)."""
    intersection, _, _ = expert_set_overlap_matrices(expert_sets_df, concepts)
    return intersection[np.triu_indices(len(concepts), k=1)]
