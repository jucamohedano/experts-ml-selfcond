import logging
import pandas as pd
import numpy as np
from utils.helpers import (save_dataframe, expert_set_overlap_matrices, category_alignment_metrics,
                           scope_out_dir, scope_summary_row, layer_profile_matrices)
from utils.plot_helpers import _plot_heatmap_with_leaders, build_category_color_map

log = logging.getLogger(__name__)

# Headline metrics for the cross-scope sublayer_comparison table. The two layer-profile
# entries are ROC-AUCs rather than contrasts because ROC-AUC is rank-based, hence immune
# to the same/different pair imbalance and directly comparable to the Jaccard one above.
# Together they answer whether the layer-profile metric carries categorical signal that
# Jaccard does not.
SUMMARY_LABELS = {
    "category_contrast_pct": "Within minus across category Jaccard %",
    "category_roc_auc": "Category alignment ROC-AUC",
    "within_category_jaccard_pct": "Within-category Jaccard %",
    "across_category_jaccard_pct": "Across-category Jaccard %",
    "layer_profile_roc_auc": "Category alignment ROC-AUC, layer profile",
    "layer_profile_z_roc_auc": "Category alignment ROC-AUC, layer-profile z",
}

def plot_all_heatmaps(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, heat_dir) -> tuple:
    """
    Calculate pairwise Jaccard, Overlap and layer-profile similarity matrices for all
    concepts. Generates corresponding heatmap visualizations and saves matrix CSVs.

    The first two ask which neurons a pair shares, the layer-profile pair asks whether
    they spread their experts over the layers alike, which is invisible to a set metric.
    Returns (concepts, jaccard_matrix, layer_profile_matrix, layer_profile_z_matrix) so
    the caller can summarize without recomputing.
    """
    concepts = concept_metadata['concept'].unique()

    # Color each concept's label by its category (shared map -> same colors as module 3).
    # A root category node (e.g. the "furniture" concept, whose own category is null) is
    # colored as its own category, so each block and its parent label share one color.
    color_map = build_category_color_map(concept_metadata.dropna(subset=["category"])["category"])
    concept_to_cat = concept_metadata.set_index("concept")["category"].to_dict()

    def _concept_color(concept):
        cat = concept_to_cat.get(concept)
        if (cat is None or pd.isna(cat)) and concept in color_map:
            cat = concept
        return color_map.get(cat, "#000000")

    concept_colors = [_concept_color(c) for c in concepts]
    present = set(concept_metadata.dropna(subset=["category"])["category"])
    color_legend = {cat: col for cat, col in color_map.items() if cat in present}

    # Category-block boundaries (concepts are ordered by category, each block led by its
    # root label, which counts as part of its own category).
    def _effective_category(concept):
        cat = concept_to_cat.get(concept)
        if (cat is None or pd.isna(cat)) and concept in color_map:
            cat = concept
        return cat

    effective_categories = [_effective_category(c) for c in concepts]
    category_boundaries = [i for i in range(1, len(concepts))
                           if effective_categories[i] != effective_categories[i - 1]]


    # Experts are keyed on (layer_idx, unit) pairs and held sparsely; see
    # expert_set_overlap_matrices for why the dense concept-by-expert form is avoided.
    intersection, jaccard, overlap = expert_set_overlap_matrices(expert_allocation_df, concepts)
    jaccard_matrix = jaccard * 100
    overlap_matrix = overlap * 100

    # Raw shared-expert counts behind both percentage matrices, for scale. The diagonal
    # holds each concept's own expert-set size.
    counts_df = pd.DataFrame(intersection.astype(int), index=concepts, columns=concepts)
    save_dataframe(counts_df, heat_dir / "shared_expert_counts_matrix.csv", index=True)

    # Layer-profile agreement over the same concept list module 3 and module 4 pass, which
    # the null grid requires (see layer_profile_matrices).
    _, profile_matrix, profile_z_matrix = layer_profile_matrices(expert_allocation_df, list(concepts))

    # Dictionary of metrics to streamline saving and plotting
    matrices = {
        "jaccard": (jaccard_matrix, "Pairwise Jaccard Similarity Index %", "magma"),
        "overlap": (overlap_matrix, "Pairwise Overlap Coefficient %", "magma"),
        "layer_profile": (profile_matrix, "Pairwise Layer-Profile Similarity %", "magma"),
        "layer_profile_z": (profile_z_matrix,
                            "Pairwise Layer-Profile Agreement vs Count-Matched Null (z)", "magma"),
    }

    for name, (mtx, title, cmap) in matrices.items():
        matrix_df = pd.DataFrame(mtx, index=concepts, columns=concepts)
        save_dataframe(matrix_df, heat_dir / f"{name}_matrix.csv", index=True)
        _plot_heatmap_with_leaders(
            mtx, concepts, title, heat_dir / f"{name}_heatmap.png", cmap,
            concept_colors=concept_colors, color_legend=color_legend,
            category_boundaries=category_boundaries,
        )

    return concepts, jaccard_matrix, profile_matrix, profile_z_matrix


def summarize_category_contrast(concepts, similarity_matrix: np.ndarray, concept_metadata: pd.DataFrame) -> dict:
    """
    Reduce a pairwise similarity matrix to the question the heatmaps exist to answer:
    are within-category concept pairs more similar than across-category pairs?

    Restricted to level-2 concepts carrying a category, so the eight category-label words
    (whose own category is null, and which have no same-category peers) never enter the
    pair pool. Reports both group means and their difference, plus the rank-based
    roc_auc from category_alignment_metrics, which is immune to the heavy same/different
    pair imbalance and to Jaccard's skew. Fewer than two categorized concepts leaves
    nothing to contrast, so the metrics come back NaN.

    Returns GENERIC keys (within_pct, across_pct, contrast_pct, roc_auc), which the caller
    renames per matrix.

    Concepts with no usable value are dropped whole, rather than dropping individual
    non-finite PAIRS, because category_alignment_metrics derives its same-category mask
    from the concept list and its permutation test shuffles labels across concepts, both
    of which need the pair pool to stay a complete triangle over one consistent concept
    set. This is a no-op for Jaccard, which is never NaN, and matters for the
    layer-profile matrices, where a word below MIN_PROFILE_EXPERTS is NaN throughout.
    """
    concept_to_cat = concept_metadata.set_index("concept")["category"].to_dict()
    categorized = [i for i, c in enumerate(concepts) if pd.notna(concept_to_cat.get(c))]
    empty = {"within_pct": np.nan, "across_pct": np.nan,
             "contrast_pct": np.nan, "roc_auc": np.nan}
    if len(categorized) < 2:
        return empty

    # Drop concepts carrying non-finite similarities, most-affected first, until the
    # remaining submatrix is a complete finite triangle. A word with no usable profile has
    # an entirely NaN row, so this normally converges in one step.
    keep = list(categorized)
    while len(keep) >= 2:
        block = similarity_matrix[np.ix_(keep, keep)].copy()
        np.fill_diagonal(block, 0.0)  # the z diagonal is NaN by design, never a data loss
        bad = (~np.isfinite(block)).sum(axis=1)
        if bad.max() == 0:
            break
        keep.pop(int(bad.argmax()))
    if len(keep) < 2:
        return empty

    categories = np.array([concept_to_cat[concepts[i]] for i in keep])
    sub_matrix = similarity_matrix[np.ix_(keep, keep)]
    iu = np.triu_indices(len(keep), k=1)
    pair_similarity = sub_matrix[iu]
    same = (categories[:, None] == categories[None, :])[iu]
    if not same.any() or same.all():
        return empty

    alignment = category_alignment_metrics(pair_similarity, categories)
    within, across = pair_similarity[same].mean(), pair_similarity[~same].mean()
    return {"within_pct": within, "across_pct": across,
            "contrast_pct": within - across, "roc_auc": alignment["roc_auc"]}


def execute_module_5_heatmaps(scope, concept_metadata: pd.DataFrame, heat_dir) -> dict:
    """Execute Module 5: Heatmaps.
    Generate all pairwise similarity heatmaps and CSV matrices for concepts.

    Purely set-based, so the whole-model scope is the unqualified pairwise geometry and
    each sublayer scope shows whether that same block structure survives in a single
    projection type. Returns the summary row for this module's sublayer_comparison table.
    """
    out_dir = scope_out_dir(heat_dir, scope)
    log.info(f"  [{scope.label}] Generating all pairwise heatmaps and CSV matrices "
             f"(Jaccard, Overlap & layer profile)...")
    concepts, jaccard_matrix, profile_matrix, profile_z_matrix = plot_all_heatmaps(
        scope.expert_df, concept_metadata, out_dir)

    jaccard = summarize_category_contrast(concepts, jaccard_matrix, concept_metadata)
    profile = summarize_category_contrast(concepts, profile_matrix, concept_metadata)
    profile_z = summarize_category_contrast(concepts, profile_z_matrix, concept_metadata)
    return scope_summary_row(
        scope,
        within_category_jaccard_pct=jaccard["within_pct"],
        across_category_jaccard_pct=jaccard["across_pct"],
        category_contrast_pct=jaccard["contrast_pct"],
        category_roc_auc=jaccard["roc_auc"],
        layer_profile_roc_auc=profile["roc_auc"],
        layer_profile_z_roc_auc=profile_z["roc_auc"],
    )