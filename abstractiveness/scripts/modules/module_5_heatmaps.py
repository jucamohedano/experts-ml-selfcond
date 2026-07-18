import logging
import pandas as pd
import numpy as np
from utils.helpers import save_dataframe
from utils.plot_helpers import _plot_heatmap_with_leaders, build_category_color_map

log = logging.getLogger(__name__)

def plot_all_heatmaps(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, heat_dir) -> None:
    """
    Calculate pairwise Jaccard and Overlap similarity matrices for all concepts.
    Generates corresponding heatmap visualizations and saves matrix CSVs.
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


    # Columns are (layer_idx, unit) pairs: the raw `unit` column is only the neuron index
    # *within* a layer, so pivoting on it alone would merge identical indices from different
    # layers into one column, inflating pairwise intersections between unrelated experts.
    presence_matrix = expert_allocation_df.assign(present=1).pivot_table(index='concept', columns=['layer_idx', 'unit'], values='present', fill_value=0)
    presence_matrix = presence_matrix.reindex(concepts, fill_value=0)
    A = presence_matrix.values  
    
    # Set sizes and operations
    intersection = np.dot(A, A.T) 
    sizes = A.sum(axis=1)         
    union = sizes[:, None] + sizes[None, :] - intersection
    min_size = np.minimum(sizes[:, None], sizes[None, :])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_matrix = np.where(union > 0, (intersection / union) * 100, 0.0)    
        overlap_matrix = np.where(min_size > 0, (intersection / min_size) * 100, 0.0)

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

def execute_module_5_heatmaps(formatted_expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, heat_dir) -> None:
    """Execute Module 5: Heatmaps.
    Generate all pairwise similarity heatmaps and CSV matrices for concepts.
    """
    log.info("  Generating all pairwise heatmaps and CSV matrices (Jaccard & Overlap)...")
    plot_all_heatmaps(formatted_expert_allocation_df, concept_metadata, heat_dir)