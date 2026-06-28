import logging
import pandas as pd
import numpy as np
from utils.helpers import save_dataframe
from utils.plot_helpers import _plot_heatmap_with_leaders

log = logging.getLogger(__name__)

def plot_all_heatmaps(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, heat_dir) -> None:
    """
    Calculate pairwise Jaccard and Overlap similarity matrices for all concepts.
    Generates corresponding heatmap visualizations and saves matrix CSVs.
    """
    concepts = concept_metadata['concept'].unique()
    
    presence_matrix = expert_allocation_df.assign(present=1).pivot_table(index='concept', columns='unit', values='present', fill_value=0)
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

    # Dictionary of metrics to streamline saving and plotting
    matrices = {
        "jaccard": (jaccard_matrix, "Pairwise Jaccard Similarity Index %", "magma"),
        "overlap": (overlap_matrix, "Pairwise Overlap Coefficient %", "magma")
    }

    for name, (mtx, title, cmap) in matrices.items():
        matrix_df = pd.DataFrame(mtx, index=concepts, columns=concepts)
        save_dataframe(matrix_df, heat_dir / f"{name}_matrix.csv", index=True)
        _plot_heatmap_with_leaders(mtx, concepts, title, heat_dir / f"{name}_heatmap.png", cmap)

def execute_module_5_heatmaps(formatted_expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, heat_dir) -> None:
    """Execute Module 5: Heatmaps.
    Generate all pairwise similarity heatmaps and CSV matrices for concepts.
    """
    log.info("  Generating all pairwise heatmaps and CSV matrices (Jaccard & Overlap)...")
    plot_all_heatmaps(formatted_expert_allocation_df, concept_metadata, heat_dir)