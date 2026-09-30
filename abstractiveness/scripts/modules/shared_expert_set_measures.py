"""
Expert-set measures: which neurons two words share (Jaccard, overlap) and category alignment.

Shared by modules 1, 2, 4 and 5. Expert sets are held sparsely, never as a dense pivot.

Explanations: documentation/module_2_similarities.md and module_1_expert_distribution.md.
"""

import numpy as np
import pandas as pd
from scipy import sparse, stats


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
                               n_permutations: int = 0, rng: np.random.Generator = None) -> dict:
    """How well expert-set similarity tracks category membership, over all concept pairs."""
    n = len(concept_categories)
    iu = np.triu_indices(n, k=1)
    same = (concept_categories[:, None] == concept_categories[None, :])[iu]

    if pair_similarity.min() == pair_similarity.max():
        return {"roc_auc": 0.5, "category_alignment_r": np.nan, "mantel_p": 1.0}

    ranks = stats.rankdata(pair_similarity)
    n1 = int(same.sum())
    n0 = len(same) - n1
    rank_offset = n1 * (n1 + 1) / 2

    def rank_auc(same_mask):
        return (ranks[same_mask].sum() - rank_offset) / (n1 * n0)

    auc = rank_auc(same)
    r, _ = stats.pearsonr(same.astype(float), pair_similarity)

    if not n_permutations:
        return {"roc_auc": auc, "category_alignment_r": r, "mantel_p": np.nan}

    n_at_least = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(concept_categories)
        same_perm = (permuted[:, None] == permuted[None, :])[iu]
        if rank_auc(same_perm) >= auc:
            n_at_least += 1
    mantel_p = (1 + n_at_least) / (1 + n_permutations)

    return {"roc_auc": auc, "category_alignment_r": r, "mantel_p": mantel_p}


def pair_similarity_vector(expert_sets_df: pd.DataFrame, concepts: list) -> np.ndarray:
    """Jaccard similarity of raw expert sets for every unordered pair of ``concepts``, as a flat vector aligned with np.triu_indices(len(concepts), k=1)."""
    _, jaccard, _ = expert_set_overlap_matrices(expert_sets_df, concepts)
    return jaccard[np.triu_indices(len(concepts), k=1)]


def pair_shared_count_vector(expert_sets_df: pd.DataFrame, concepts: list) -> np.ndarray:
    """Raw count of shared experts for every unordered concept pair, aligned with np.triu_indices(len(concepts), k=1)."""
    intersection, _, _ = expert_set_overlap_matrices(expert_sets_df, concepts)
    return intersection[np.triu_indices(len(concepts), k=1)]
