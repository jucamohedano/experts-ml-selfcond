import logging
import pandas as pd
import numpy as np
from utils.helpers import (save_dataframe, expert_set_overlap_matrices, category_alignment_metrics,
                           scope_out_dir, scope_summary_row)
from utils.plot_helpers import _plot_heatmap_with_leaders, build_category_color_map

log = logging.getLogger(__name__)

# Headline metrics for the cross-scope sublayer_comparison table.
SUMMARY_LABELS = {
    "category_contrast_pct": "Within minus across category Jaccard %",
    "category_roc_auc": "Category alignment ROC-AUC",
    "within_category_jaccard_pct": "Within-category Jaccard %",
    "across_category_jaccard_pct": "Across-category Jaccard %",
}

def plot_all_heatmaps(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, heat_dir) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate pairwise Jaccard and Overlap similarity matrices for all concepts.
    Generates corresponding heatmap visualizations and saves matrix CSVs.
    Returns (concepts, jaccard_matrix) so the caller can summarize without recomputing.
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

    # Dictionary of metrics to streamline saving and plotting
    matrices = {
        "jaccard": (jaccard_matrix, "Pairwise Jaccard Similarity Index %", "magma"),
        "overlap": (overlap_matrix, "Pairwise Overlap Coefficient %", "magma")
    }

    for name, (mtx, title, cmap) in matrices.items():
        matrix_df = pd.DataFrame(mtx, index=concepts, columns=concepts)
        save_dataframe(matrix_df, heat_dir / f"{name}_matrix.csv", index=True)
        _plot_heatmap_with_leaders(
            mtx, concepts, title, heat_dir / f"{name}_heatmap.png", cmap,
            concept_colors=concept_colors, color_legend=color_legend,
            category_boundaries=category_boundaries,
        )

    return concepts, jaccard_matrix


def summarize_category_contrast(concepts, jaccard_matrix: np.ndarray, concept_metadata: pd.DataFrame) -> dict:
    """
    Reduce the pairwise Jaccard matrix to the question the heatmaps exist to answer:
    are within-category concept pairs more similar than across-category pairs?

    Restricted to level-2 concepts carrying a category, so the eight category-label words
    (whose own category is null, and which have no same-category peers) never enter the
    pair pool. Reports both group means and their difference, plus the rank-based
    roc_auc from category_alignment_metrics, which is immune to the heavy same/different
    pair imbalance and to Jaccard's skew. Fewer than two categorized concepts leaves
    nothing to contrast, so the metrics come back NaN.
    """
    concept_to_cat = concept_metadata.set_index("concept")["category"].to_dict()
    keep = [i for i, c in enumerate(concepts) if pd.notna(concept_to_cat.get(c))]
    empty = {"within_category_jaccard_pct": np.nan, "across_category_jaccard_pct": np.nan,
             "category_contrast_pct": np.nan, "category_roc_auc": np.nan}
    if len(keep) < 2:
        return empty

    categories = np.array([concept_to_cat[concepts[i]] for i in keep])
    sub_matrix = jaccard_matrix[np.ix_(keep, keep)]
    iu = np.triu_indices(len(keep), k=1)
    pair_similarity = sub_matrix[iu]
    same = (categories[:, None] == categories[None, :])[iu]
    if not same.any() or same.all():
        return empty

    alignment = category_alignment_metrics(pair_similarity, categories)
    within, across = pair_similarity[same].mean(), pair_similarity[~same].mean()
    return {"within_category_jaccard_pct": within, "across_category_jaccard_pct": across,
            "category_contrast_pct": within - across, "category_roc_auc": alignment["roc_auc"]}


def execute_module_5_heatmaps(scope, concept_metadata: pd.DataFrame, heat_dir) -> dict:
    """Execute Module 5: Heatmaps.
    Generate all pairwise similarity heatmaps and CSV matrices for concepts.

    Purely set-based, so the whole-model scope is the unqualified pairwise geometry and
    each sublayer scope shows whether that same block structure survives in a single
    projection type. Returns the summary row for this module's sublayer_comparison table.
    """
    out_dir = scope_out_dir(heat_dir, scope)
    log.info(f"  [{scope.label}] Generating all pairwise heatmaps and CSV matrices (Jaccard & Overlap)...")
    concepts, jaccard_matrix = plot_all_heatmaps(scope.expert_df, concept_metadata, out_dir)
    return scope_summary_row(scope, **summarize_category_contrast(concepts, jaccard_matrix, concept_metadata))