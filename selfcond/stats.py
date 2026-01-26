"""
Statistical testing utilities for RSA and other analyses.

This module provides functions for:
- Multiple comparison correction (FDR)
- Pair-wise permutation testing (GPU-accelerated)
- Manhattan plot visualization
- Ranked table output
"""

import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

log = logging.getLogger(__name__)


# =============================================================================
# Core Utility Functions (kept from original)
# =============================================================================

def apply_fdr_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction using statsmodels.
    
    Args:
        p_values: Array of raw p-values
        
    Returns:
        Array of FDR-corrected p-values
    """
    from statsmodels.stats.multitest import multipletests
    _, p_fdr, _, _ = multipletests(p_values, method="fdr_bh")
    return p_fdr


def rankdata_2d(arr: np.ndarray) -> np.ndarray:
    """
    Rank data along axis 1 (each row independently). Vectorized.
    
    Args:
        arr: (n_rows, n_cols) array
        
    Returns:
        (n_rows, n_cols) array of ranks (1-indexed)
    """
    n_rows, n_cols = arr.shape
    order = np.argsort(arr, axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(n_rows)[:, np.newaxis]
    ranks[rows, order] = np.arange(1, n_cols + 1)
    return ranks


def permute_rdm(rdm: np.ndarray, perm_idx: np.ndarray) -> np.ndarray:
    """
    Permute rows and columns of an RDM according to perm_idx.
    
    Args:
        rdm: 60x60 RDM matrix
        perm_idx: Permutation indices
        
    Returns:
        Permuted RDM matrix
    """
    return rdm[perm_idx][:, perm_idx]


# =============================================================================
# GPU-Accelerated Pair-wise Permutation Testing
# =============================================================================

def _build_upper_triangle_permutation_indices(n_words: int = 60) -> np.ndarray:
    """
    Build a lookup table for permuting upper triangle indices.
    
    Given a permutation of word indices (0..59), we need to know how to
    reorder the upper triangle elements (1770 values).
    
    Returns:
        (1770,) array where output[i] = original index in upper triangle
        that maps to position i after permutation
    """
    # Original upper triangle indices
    triu_i, triu_j = np.triu_indices(n_words, k=1)
    n_elements = len(triu_i)
    
    # Create mapping from (i, j) to flat index
    pair_to_idx = {}
    for idx, (i, j) in enumerate(zip(triu_i, triu_j)):
        pair_to_idx[(i, j)] = idx
    
    return triu_i, triu_j, pair_to_idx


def _apply_permutation_to_upper_triangle(
    brain_vecs: np.ndarray,
    perm_idx: np.ndarray,
    triu_i: np.ndarray,
    triu_j: np.ndarray,
    n_words: int = 60,
) -> np.ndarray:
    """
    Apply a word permutation to all brain upper triangle vectors.
    
    Instead of reconstructing 60x60 RDMs, we directly reorder the upper triangle
    elements based on the permutation.
    
    Args:
        brain_vecs: (N_brain, 1770) array of upper triangle values
        perm_idx: (60,) permutation of word indices
        triu_i, triu_j: Upper triangle index arrays
        n_words: Number of words (60)
        
    Returns:
        (N_brain, 1770) permuted brain vectors
    """
    # Map original indices to permuted indices
    # If perm_idx = [3, 1, 0, 2, ...], then word 0 becomes word 3, etc.
    inv_perm = np.argsort(perm_idx)  # Inverse permutation
    
    # New upper triangle indices after permutation
    new_i = inv_perm[triu_i]
    new_j = inv_perm[triu_j]
    
    # Ensure i < j (upper triangle convention)
    swap_mask = new_i > new_j
    new_i_sorted = np.where(swap_mask, new_j, new_i)
    new_j_sorted = np.where(swap_mask, new_i, new_j)
    
    # Convert to flat indices
    # Formula: flat_idx = i * n_words - i * (i + 1) // 2 + (j - i - 1)
    # But easier: just use the triu_indices mapping
    new_flat_idx = new_i_sorted * n_words - new_i_sorted * (new_i_sorted + 1) // 2 + (new_j_sorted - new_i_sorted - 1)
    
    # Reorder brain vectors
    return brain_vecs[:, new_flat_idx]


def run_pairwise_permutation_test(
    brain_rdms: Dict[str, Dict[Tuple[int, int, int], np.ndarray]],
    model_rdms: Dict[str, np.ndarray],
    correlation_method: str = "spearman",
    n_permutations: int = 1000,
    use_gpu: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run pair-wise permutation test for all (brain region, model layer) pairs.
    
    Uses the Permutation-Loop-First algorithm for efficiency:
    1. Pre-compute ALL observed correlations in one batch operation
    2. Loop over permutations (not pairs)
    3. For each permutation, compute ALL null correlations in one batch
    4. Count how many nulls exceed observed for each pair
    
    Args:
        brain_rdms: {subject_id: {region_id: 60x60 RDM}}
        model_rdms: {layer_name: 60x60 RDM}
        correlation_method: "spearman" or "pearson"
        n_permutations: Number of permutations (default 1000)
        use_gpu: Whether to use GPU acceleration (requires PyTorch)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: subject, region, layer, rsa_correlation, 
                               p_value, p_value_fdr, significant
    """
    np.random.seed(seed)
    
    n_words = 60
    triu_idx = np.triu_indices(n_words, k=1)
    n_elements = len(triu_idx[0])  # 1770
    
    # Prepare upper triangle indices for permutation
    triu_i, triu_j, _ = _build_upper_triangle_permutation_indices(n_words)
    
    # Flatten brain RDMs and extract upper triangles
    brain_metadata = []  # List of (subject_id, region_id)
    brain_vecs_list = []
    
    for subject_id, subject_regions in brain_rdms.items():
        for region_id, brain_rdm in subject_regions.items():
            brain_metadata.append((subject_id, region_id))
            brain_vecs_list.append(brain_rdm[triu_idx])
    
    brain_vecs = np.array(brain_vecs_list, dtype=np.float64)  # (N_brain, 1770)
    n_brain = len(brain_vecs)
    
    # Extract model upper triangles
    layer_names = list(model_rdms.keys())
    model_vecs = np.array([model_rdms[name][triu_idx] for name in layer_names], dtype=np.float64)  # (N_layers, 1770)
    n_layers = len(layer_names)
    
    log.info(f"Running pair-wise permutation test:")
    log.info(f"  - {n_brain} brain regions × {n_layers} layers = {n_brain * n_layers} pairs")
    log.info(f"  - {n_permutations} permutations")
    log.info(f"  - GPU: {use_gpu}")
    
    # Check if GPU is available and requested
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                log.info(f"  - Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                log.warning("GPU requested but CUDA not available. Falling back to CPU.")
                use_gpu = False
                device = None
        except ImportError:
            log.warning("PyTorch not installed. Falling back to CPU (NumPy).")
            use_gpu = False
            device = None
    
    if use_gpu:
        # GPU implementation using PyTorch
        import torch
        
        brain_vecs_t = torch.tensor(brain_vecs, dtype=torch.float32, device=device)
        model_vecs_t = torch.tensor(model_vecs, dtype=torch.float32, device=device)
        
        # Pre-compute ranks for Spearman
        if correlation_method == "spearman":
            # Rank along axis 1
            brain_ranks_t = brain_vecs_t.argsort(dim=1).argsort(dim=1).float() + 1
            model_ranks_t = model_vecs_t.argsort(dim=1).argsort(dim=1).float() + 1
            
            # Normalize for correlation
            brain_centered = brain_ranks_t - brain_ranks_t.mean(dim=1, keepdim=True)
            model_centered = model_ranks_t - model_ranks_t.mean(dim=1, keepdim=True)
        else:
            brain_centered = brain_vecs_t - brain_vecs_t.mean(dim=1, keepdim=True)
            model_centered = model_vecs_t - model_vecs_t.mean(dim=1, keepdim=True)
        
        brain_norms = torch.sqrt((brain_centered ** 2).sum(dim=1, keepdim=True))
        model_norms = torch.sqrt((model_centered ** 2).sum(dim=1, keepdim=True))
        
        brain_normed = brain_centered / (brain_norms + 1e-10)
        model_normed = model_centered / (model_norms + 1e-10)
        
        # Step 1: Compute observed correlations
        observed_matrix = brain_normed @ model_normed.T  # (N_brain, N_layers)
        
        # Step 2: Initialize count matrix
        count_exceeds = torch.zeros_like(observed_matrix)
        
        # Step 3: Loop over permutations
        for perm_i in tqdm(range(n_permutations), desc="Permutation test (GPU)"):
            # Shuffle word indices
            perm_idx = np.random.permutation(n_words)
            
            # Apply permutation to brain vectors
            permuted_brain_vecs = _apply_permutation_to_upper_triangle(
                brain_vecs, perm_idx, triu_i, triu_j, n_words
            )
            permuted_brain_t = torch.tensor(permuted_brain_vecs, dtype=torch.float32, device=device)
            
            # Compute ranks/normalization for permuted data
            if correlation_method == "spearman":
                perm_ranks = permuted_brain_t.argsort(dim=1).argsort(dim=1).float() + 1
                perm_centered = perm_ranks - perm_ranks.mean(dim=1, keepdim=True)
            else:
                perm_centered = permuted_brain_t - permuted_brain_t.mean(dim=1, keepdim=True)
            
            perm_norms = torch.sqrt((perm_centered ** 2).sum(dim=1, keepdim=True))
            perm_normed = perm_centered / (perm_norms + 1e-10)
            
            # Compute null correlations
            null_matrix = perm_normed @ model_normed.T
            
            # Count where null >= observed
            count_exceeds += (null_matrix >= observed_matrix).float()
        
        # Step 4: Compute p-values
        p_values = (count_exceeds + 1) / (n_permutations + 1)
        
        # Move back to CPU for DataFrame construction
        observed_matrix = observed_matrix.cpu().numpy()
        p_values = p_values.cpu().numpy()
        
    else:
        # CPU implementation using NumPy
        
        # Pre-compute ranks for Spearman
        if correlation_method == "spearman":
            brain_ranks = rankdata_2d(brain_vecs)
            model_ranks = rankdata_2d(model_vecs)
            
            brain_centered = brain_ranks - brain_ranks.mean(axis=1, keepdims=True)
            model_centered = model_ranks - model_ranks.mean(axis=1, keepdims=True)
        else:
            brain_centered = brain_vecs - brain_vecs.mean(axis=1, keepdims=True)
            model_centered = model_vecs - model_vecs.mean(axis=1, keepdims=True)
        
        brain_norms = np.sqrt((brain_centered ** 2).sum(axis=1, keepdims=True))
        model_norms = np.sqrt((model_centered ** 2).sum(axis=1, keepdims=True))
        
        brain_normed = brain_centered / (brain_norms + 1e-10)
        model_normed = model_centered / (model_norms + 1e-10)
        
        # Step 1: Compute observed correlations
        observed_matrix = brain_normed @ model_normed.T  # (N_brain, N_layers)
        
        # Step 2: Initialize count matrix
        count_exceeds = np.zeros_like(observed_matrix)
        
        # Step 3: Loop over permutations
        for perm_i in tqdm(range(n_permutations), desc="Permutation test (CPU)"):
            # Shuffle word indices
            perm_idx = np.random.permutation(n_words)
            
            # Apply permutation to brain vectors
            permuted_brain_vecs = _apply_permutation_to_upper_triangle(
                brain_vecs, perm_idx, triu_i, triu_j, n_words
            )
            
            # Compute ranks/normalization for permuted data
            if correlation_method == "spearman":
                perm_ranks = rankdata_2d(permuted_brain_vecs)
                perm_centered = perm_ranks - perm_ranks.mean(axis=1, keepdims=True)
            else:
                perm_centered = permuted_brain_vecs - permuted_brain_vecs.mean(axis=1, keepdims=True)
            
            perm_norms = np.sqrt((perm_centered ** 2).sum(axis=1, keepdims=True))
            perm_normed = perm_centered / (perm_norms + 1e-10)
            
            # Compute null correlations
            null_matrix = perm_normed @ model_normed.T
            
            # Count where null >= observed
            count_exceeds += (null_matrix >= observed_matrix).astype(np.float64)
        
        # Step 4: Compute p-values
        p_values = (count_exceeds + 1) / (n_permutations + 1)
    
    # Step 5: Apply FDR correction
    p_values_flat = p_values.flatten()
    p_values_fdr = apply_fdr_correction(p_values_flat)
    p_values_fdr_matrix = p_values_fdr.reshape(n_brain, n_layers)
    
    # Build results DataFrame
    results = []
    for brain_idx, (subject_id, region_id) in enumerate(brain_metadata):
        for layer_idx, layer_name in enumerate(layer_names):
            results.append({
                "subject": subject_id,
                "region_x": region_id[0],
                "region_y": region_id[1],
                "region_z": region_id[2],
                "region": f"{region_id[0]}_{region_id[1]}_{region_id[2]}",
                "layer": layer_name,
                "rsa_correlation": observed_matrix[brain_idx, layer_idx],
                "p_value": p_values[brain_idx, layer_idx],
                "p_value_fdr": p_values_fdr_matrix[brain_idx, layer_idx],
                "significant": p_values_fdr_matrix[brain_idx, layer_idx] < 0.05,
            })
    
    df = pd.DataFrame(results)
    
    # Log summary
    n_significant = df["significant"].sum()
    log.info(f"Permutation test complete:")
    log.info(f"  - {n_significant}/{len(df)} pairs significant (FDR < 0.05)")
    
    return df


# =============================================================================
# Visualization: Manhattan Plot
# =============================================================================

def plot_manhattan_rsa(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    p_value_threshold: float = 0.01,
    ap_threshold: Optional[float] = None,
) -> None:
    """
    Create a Manhattan-style scatter plot of significant RSA pairs.
    
    Args:
        df: DataFrame from run_pairwise_permutation_test
        output_dir: Directory to save plot
        p_value_threshold: Only plot pairs with p_value_fdr < threshold
        ap_threshold: AP threshold for title (optional)
    """
    # Filter to significant pairs
    df_sig = df[df["p_value_fdr"] < p_value_threshold].copy()
    
    if len(df_sig) == 0:
        log.warning(f"No significant pairs found at p < {p_value_threshold}. Skipping Manhattan plot.")
        return
    
    log.info(f"Plotting {len(df_sig)} significant pairs (p_fdr < {p_value_threshold})")
    
    # Extract layer number for x-axis
    def get_layer_num(layer_name):
        parts = layer_name.split("_")
        for i, p in enumerate(parts):
            if p == "h" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return 0
    
    df_sig["layer_num"] = df_sig["layer"].apply(get_layer_num)
    
    # Create region index for y-axis (sorted by peak layer)
    region_peak_layer = df_sig.groupby("region")["layer_num"].apply(
        lambda x: df_sig.loc[x.idxmax(), "layer_num"] if len(x) > 0 else 0
    )
    region_order = region_peak_layer.sort_values().index.tolist()
    region_to_idx = {r: i for i, r in enumerate(region_order)}
    df_sig["region_idx"] = df_sig["region"].map(region_to_idx)
    
    # Compute -log10(p_value_fdr) for size encoding
    df_sig["neg_log_p"] = -np.log10(df_sig["p_value_fdr"] + 1e-10)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    scatter = ax.scatter(
        df_sig["layer_num"],
        df_sig["region_idx"],
        c=df_sig["rsa_correlation"],
        s=df_sig["neg_log_p"] * 20,  # Scale size
        cmap="RdBu_r",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("RSA Correlation", fontsize=12)
    
    # Labels and title
    ax.set_xlabel("LLM Layer Number", fontsize=12)
    ax.set_ylabel("Brain Region (sorted by peak layer)", fontsize=12)
    
    ap_str = f" (AP={ap_threshold})" if ap_threshold is not None else ""
    ax.set_title(
        f"Significant Brain-Model RSA Pairs{ap_str}\n"
        f"(p_fdr < {p_value_threshold}, n={len(df_sig)})\n"
        f"Size = -log10(p_fdr)",
        fontsize=14
    )
    
    # Set x-axis ticks
    unique_layers = sorted(df_sig["layer_num"].unique())
    ax.set_xticks(unique_layers)
    
    # Set y-axis ticks (show subset if too many)
    n_regions = len(region_order)
    if n_regions > 30:
        # Show every Nth region
        step = n_regions // 20
        tick_positions = list(range(0, n_regions, step))
        tick_labels = [region_order[i] for i in tick_positions]
    else:
        tick_positions = list(range(n_regions))
        tick_labels = region_order
    
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "rsa_manhattan_plot.png", dpi=150)
    plt.close()
    
    log.info(f"Saved Manhattan plot to {output_dir / 'rsa_manhattan_plot.png'}")


# =============================================================================
# Ranked Table Output
# =============================================================================

def generate_ranked_table(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    top_n: int = 100,
    p_value_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Generate a ranked table of significant layer-region pairs.
    
    Args:
        df: DataFrame from run_pairwise_permutation_test
        output_dir: Directory to save table
        top_n: Number of top pairs to include
        p_value_threshold: Only include pairs with p_value_fdr < threshold
        
    Returns:
        Ranked DataFrame
    """
    # Filter to significant pairs
    df_sig = df[df["p_value_fdr"] < p_value_threshold].copy()
    
    if len(df_sig) == 0:
        log.warning(f"No significant pairs found at p < {p_value_threshold}. Creating empty table.")
        ranked_df = pd.DataFrame(columns=[
            "rank", "subject", "region", "layer", "rsa_correlation", "p_value", "p_value_fdr"
        ])
        ranked_df.to_csv(output_dir / "rsa_ranked_pairs.csv", index=False)
        return ranked_df
    
    # Sort by RSA correlation (descending)
    df_sig = df_sig.sort_values("rsa_correlation", ascending=False)
    
    # Take top N
    df_top = df_sig.head(top_n).copy()
    
    # Add rank column
    df_top.insert(0, "rank", range(1, len(df_top) + 1))
    
    # Select columns for output
    output_cols = [
        "rank", "subject", "region", "layer", 
        "rsa_correlation", "p_value", "p_value_fdr"
    ]
    ranked_df = df_top[output_cols]
    
    # Save to CSV
    ranked_df.to_csv(output_dir / "rsa_ranked_pairs.csv", index=False)
    log.info(f"Saved ranked table ({len(ranked_df)} pairs) to {output_dir / 'rsa_ranked_pairs.csv'}")
    
    # Also save summary by best layer per region
    best_per_region = df_sig.loc[df_sig.groupby("region")["rsa_correlation"].idxmax()]
    best_per_region = best_per_region.sort_values("rsa_correlation", ascending=False)
    best_per_region.insert(0, "rank", range(1, len(best_per_region) + 1))
    best_per_region = best_per_region[output_cols]
    best_per_region.to_csv(output_dir / "rsa_best_layer_per_region.csv", index=False)
    log.info(f"Saved best layer per region ({len(best_per_region)} regions) to {output_dir / 'rsa_best_layer_per_region.csv'}")
    
    return ranked_df
