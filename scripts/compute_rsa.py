#!/usr/bin/env python3
"""
Simple RSA analysis: Paired t-tests per ROI.

- For each ROI: 9 subjects × 2 conditions paired t-test
- Optional FDR correction across ROIs
- Support for specific layer selection

Usage:
    python scripts/compute_rsa_simple.py \
        --brain-rdms path/to/brain_rdms.pkl \
        --expert-rdms path/to/expert/rdms \
        --full-rdms path/to/full/rdms \
        --layers attn.c_proj \
        --output results/rsa_simple/...
"""

import json
import logging
import pathlib
import pickle
import sys
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, os.getcwd())

log = logging.getLogger(__name__)

# Precompute upper triangle indices
TRIU_IDX = np.triu_indices(60, k=1)


def fisher_z(r: np.ndarray) -> np.ndarray:
    """Fisher z-transformation for correlation values.
    
    Transforms Spearman's rho to z-values which are approximately
    normally distributed, satisfying t-test assumptions.
    """
    r = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r)


def cohens_d_paired(diff: np.ndarray) -> float:
    """Cohen's d for paired samples.
    
    d = mean(diff) / std(diff)
    """
    return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

def spearman_upper_tri(rdm1: np.ndarray, rdm2: np.ndarray) -> float:
    """Spearman correlation between upper triangles."""
    vec1 = rdm1[TRIU_IDX]
    vec2 = rdm2[TRIU_IDX]
    
    if np.isnan(vec1).any() or np.isnan(vec2).any():
        return 0.0
    
    rho, _ = stats.spearmanr(vec1, vec2)
    return rho if not np.isnan(rho) else 0.0


def load_brain_rdms(path: pathlib.Path) -> Dict[str, Dict[Tuple, np.ndarray]]:
    """Load brain RDMs from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model_rdms(path: pathlib.Path) -> Dict[str, np.ndarray]:
    """Load model RDMs from directory."""
    rdms = {}
    for f in sorted(path.glob("*_rdm.npy")):
        layer = f.stem.replace("_rdm", "")
        rdms[layer] = np.load(f)
    return rdms


def filter_layers(
    layers: List[str],
    layer_filter: Optional[List[str]] = None,
) -> List[str]:
    """
    Filter layers based on pattern matching.
    
    Args:
        layers: All available layer names
        layer_filter: List of patterns to match (e.g., ["attn.c_proj", "mlp.c_fc"])
                     If None, returns all layers
    """
    if layer_filter is None:
        return layers
    
    filtered = []
    for layer in layers:
        for pattern in layer_filter:
            # Handle underscore vs dot notation
            pattern_variants = [pattern, pattern.replace(".", "_")]
            if any(p in layer for p in pattern_variants):
                filtered.append(layer)
                break
    
    return filtered


def run_rsa(
    brain_rdms_path: pathlib.Path,
    condition_rdms: Dict[str, pathlib.Path],
    output_dir: pathlib.Path,
    layer_filter: Optional[List[str]] = None,
    apply_fdr: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    RSA analysis with paired t-tests per ROI.
    
    Args:
        brain_rdms_path: Path to brain_rdms.pkl
        condition_rdms: Dict of {condition_name: rdms_path}
        output_dir: Output directory
        layer_filter: List of layer patterns to include (e.g., ["attn.c_proj"])
        apply_fdr: Whether to apply FDR correction (default: True)
        alpha: Significance threshold
        
    Returns:
        DataFrame with results per ROI
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    log.info("Loading brain RDMs...")
    brain_rdms = load_brain_rdms(brain_rdms_path)
    subjects = sorted(brain_rdms.keys())
    n_subjects = len(subjects)
    log.info(f"Loaded {n_subjects} subjects")
    
    log.info("Loading model RDMs...")
    condition_model_rdms = {}
    for cond, path in condition_rdms.items():
        condition_model_rdms[cond] = load_model_rdms(path)
    
    conditions = sorted(condition_rdms.keys())
    
    # Get common regions
    common_regions = set(brain_rdms[subjects[0]].keys())
    for subj in subjects[1:]:
        common_regions &= set(brain_rdms[subj].keys())
    common_regions = sorted(common_regions)
    log.info(f"Found {len(common_regions)} common regions")
    
    # Get and filter layers
    all_layers = sorted(condition_model_rdms[conditions[0]].keys())
    selected_layers = filter_layers(all_layers, layer_filter)
    log.info(f"Selected {len(selected_layers)}/{len(all_layers)} layers: {selected_layers}")
    
    if len(selected_layers) == 0:
        log.error("No layers matched the filter!")
        return pd.DataFrame()
    
    # Compute RSA values: [n_subjects, n_regions, n_layers, n_conditions]
    n_regions = len(common_regions)
    n_layers = len(selected_layers)
    n_conditions = len(conditions)
    
    rho_values = np.zeros((n_subjects, n_regions, n_layers, n_conditions))
    
    log.info("Computing RSA...")
    for si, subj in enumerate(subjects):
        for ri, region in enumerate(common_regions):
            brain_rdm = brain_rdms[subj][region]
            for li, layer in enumerate(selected_layers):
                for ci, cond in enumerate(conditions):
                    model_rdm = condition_model_rdms[cond][layer]
                    rho = spearman_upper_tri(brain_rdm, model_rdm)
                    rho_values[si, ri, li, ci] = rho
    
    # Average across selected layers -> [n_subjects, n_regions, n_conditions]
    rho_avg = rho_values.mean(axis=2)
    log.info(f"Averaged across {n_layers} layers")
    
    # Fisher-Z transform for t-test (correlations aren't normally distributed)
    z_avg = fisher_z(rho_avg)
    log.info("Applied Fisher-Z transformation")
    
    # Paired t-test per ROI
    log.info("Running paired t-tests...")
    
    results = []
    
    # For each condition pair
    for i, cond_a in enumerate(conditions):
        for cond_b in conditions[i+1:]:
            ci_a = conditions.index(cond_a)
            ci_b = conditions.index(cond_b)
            
            p_values = []
            
            for ri, region in enumerate(common_regions):
                # Use z-transformed values for t-test (normality assumption)
                z_a = z_avg[:, ri, ci_a]  # [9]
                z_b = z_avg[:, ri, ci_b]  # [9]
                z_diff = z_a - z_b
                
                # Also keep raw rho for reporting
                rho_a = rho_avg[:, ri, ci_a]
                rho_b = rho_avg[:, ri, ci_b]
                
                # Paired t-test on Fisher-Z values
                t_stat, p_value = stats.ttest_rel(z_a, z_b)
                
                # Effect size (Cohen's d)
                d = cohens_d_paired(z_diff)
                
                results.append({
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "region": f"{region[0]}_{region[1]}_{region[2]}",
                    "region_x": region[0],
                    "region_y": region[1],
                    "region_z": region[2],
                    # Descriptive stats (raw rho)
                    "mean_rho_a": rho_a.mean(),
                    "std_rho_a": rho_a.std(ddof=1),
                    "mean_rho_b": rho_b.mean(),
                    "std_rho_b": rho_b.std(ddof=1),
                    "mean_diff": (rho_a - rho_b).mean(),
                    "std_diff": (rho_a - rho_b).std(ddof=1),
                    # T-test results (on Fisher-Z values)
                    "t_stat": t_stat,
                    "df": n_subjects - 1,
                    "p_value": p_value,
                    # Effect size
                    "cohens_d": d,
                    # Metadata
                    "n_subjects": n_subjects,
                    "layers_used": ",".join(selected_layers),
                    "n_layers": n_layers,
                })
                p_values.append(p_value)
            
            # FDR correction
            if apply_fdr:
                from scipy.stats import false_discovery_control
                p_fdr = false_discovery_control(p_values)
                
                for j, res in enumerate(results[-n_regions:]):
                    res["p_fdr"] = p_fdr[j]
                    res["sig_raw"] = res["p_value"] < alpha
                    res["sig_fdr"] = p_fdr[j] < alpha
            else:
                for res in results[-n_regions:]:
                    res["sig_raw"] = res["p_value"] < alpha
    
    df = pd.DataFrame(results)
    
    # Save
    output_file = output_dir / "rsa_results.csv"
    df.to_csv(output_file, index=False)
    log.info(f"Saved to {output_file}")
    
    # Summary
    for cond_a, cond_b in [(conditions[0], conditions[1])]:
        mask = (df["condition_a"] == cond_a) & (df["condition_b"] == cond_b)
        n_sig_raw = df.loc[mask, "sig_raw"].sum()
        mean_diff = df.loc[mask, "mean_diff"].mean()
        mean_d = df.loc[mask, "cohens_d"].mean()
        direction = (df.loc[mask, "mean_diff"] > 0).sum()
        
        log.info(f"\n{cond_a} vs {cond_b}:")
        log.info(f"  Mean difference (rho): {mean_diff:.4f}")
        log.info(f"  Mean Cohen's d: {mean_d:.3f}")
        log.info(f"  Direction: {direction}/{n_regions} regions favor {cond_a}")
        log.info(f"  Significant (raw p<{alpha}): {n_sig_raw}/{n_regions}")
        
        if apply_fdr:
            n_sig_fdr = df.loc[mask, "sig_fdr"].sum()
            log.info(f"  Significant (FDR<{alpha}): {n_sig_fdr}/{n_regions}")
    
    # Save metadata
    metadata = {
        "brain_rdms_path": str(brain_rdms_path),
        "conditions": conditions,
        "n_subjects": n_subjects,
        "n_regions": n_regions,
        "layers_filter": layer_filter,
        "layers_used": selected_layers,
        "n_layers_averaged": n_layers,
        "apply_fdr": apply_fdr,
        "alpha": alpha,
    }
    with open(output_dir / "rsa_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Generate plots
    log.info("Generating plots...")
    _plot_paired_comparison(df, rho_avg, conditions, common_regions, output_dir, alpha, apply_fdr)
    _plot_top_effect_sizes(df, conditions, output_dir)
    _plot_diagonal_scatter(df, conditions, output_dir, apply_fdr)
    
    return df


def _plot_paired_comparison(
    df: pd.DataFrame,
    rho_avg: np.ndarray,  # [n_subjects, n_regions, n_conditions]
    conditions: List[str],
    regions: List[Tuple],
    output_dir: pathlib.Path,
    alpha: float = 0.05,
    apply_fdr: bool = True,
):
    """
    Paired comparison plot: lines connect conditions for each ROI.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.warning("matplotlib not available, skipping plots")
        return
    
    if len(conditions) != 2:
        log.warning("Paired plot requires exactly 2 conditions")
        return
    
    cond_a, cond_b = conditions[0], conditions[1]
    
    # Get mean RSA per ROI for each condition
    mean_rho_a = rho_avg[:, :, 0].mean(axis=0)  # [n_regions]
    mean_rho_b = rho_avg[:, :, 1].mean(axis=0)  # [n_regions]
    
    # Get significance from df
    mask = (df["condition_a"] == cond_a) & (df["condition_b"] == cond_b)
    sig_raw = df.loc[mask, "sig_raw"].values
    sig_fdr = df.loc[mask, "sig_fdr"].values if apply_fdr else np.zeros_like(sig_raw, dtype=bool)
    
    n_regions = len(mean_rho_a)
    
    # Sort by difference for better visualization
    diff = mean_rho_a - mean_rho_b
    sort_idx = np.argsort(diff)[::-1]  # Largest diff first
    
    fig, ax = plt.subplots(figsize=(10, max(6, n_regions * 0.12)))
    
    y_positions = np.arange(n_regions)
    
    for i, idx in enumerate(sort_idx):
        # Determine color based on significance
        if sig_fdr[idx]:
            color = "#2ecc71"  # Green for FDR significant
            lw = 2.0
        elif sig_raw[idx]:
            color = "#f39c12"  # Orange for raw significant
            lw = 1.5
        else:
            color = "#bdc3c7"  # Gray for ns
            lw = 1.0
        
        # Draw line connecting conditions
        ax.plot([mean_rho_a[idx], mean_rho_b[idx]], [i, i], 
                color=color, linewidth=lw, alpha=0.8)
        
        # Draw points
        ax.scatter([mean_rho_a[idx]], [i], color="#3498db", s=40, zorder=5)
        ax.scatter([mean_rho_b[idx]], [i], color="#e74c3c", s=40, zorder=5)
    
    ax.set_xlabel("Mean RSA (Spearman rho)", fontsize=12)
    ax.set_ylabel("Brain Region (sorted by difference)", fontsize=12)
    ax.set_title(f"Paired RSA: {cond_a} vs {cond_b}", fontsize=14)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color="#3498db", label=cond_a),
        mpatches.Patch(color="#e74c3c", label=cond_b),
        mpatches.Patch(color="#2ecc71", label="FDR significant"),
        mpatches.Patch(color="#f39c12", label="Raw significant"),
        mpatches.Patch(color="#bdc3c7", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    
    ax.set_yticks([])
    ax.axvline(0, color="black", linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "paired_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    log.info(f"Saved paired comparison plot to {output_dir / 'paired_comparison.png'}")


def _plot_top_effect_sizes(
    df: pd.DataFrame,
    conditions: List[str],
    output_dir: pathlib.Path,
    top_n: Optional[int] = None,
):
    """
    Horizontal bar plot of all regions sorted by Cohen's d effect size.
    Color indicates FDR significance (blue=significant, gray=not significant).
    
    Args:
        top_n: Number of top regions to show. If None, shows all regions.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return
    
    if len(conditions) != 2:
        return
    
    cond_a, cond_b = conditions[0], conditions[1]
    mask = (df["condition_a"] == cond_a) & (df["condition_b"] == cond_b)
    
    # Sort by Cohen's d (largest positive to largest negative)
    subset = df.loc[mask].copy()
    subset = subset.sort_values("cohens_d", ascending=False)
    
    # Limit to top_n if specified
    if top_n is not None:
        subset = subset.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(subset) * 0.18)))
    
    y_positions = np.arange(len(subset))
    
    # Color by FDR significance (consistent colors across all plots)
    has_fdr = "sig_fdr" in subset.columns
    colors = []
    for _, row in subset.iterrows():
        if has_fdr and row.get("sig_fdr", False):
            colors.append("#2ecc71")  # Green for FDR significant
        elif row.get("sig_raw", False):
            colors.append("#f39c12")  # Orange for raw significant only
        else:
            colors.append("#bdc3c7")  # Gray for not significant
    
    bars = ax.barh(y_positions, subset["cohens_d"].values, color=colors, 
                   edgecolor="black", linewidth=0.3, alpha=0.9, height=0.8)
    
    # Add region labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(subset["region"].values, fontsize=7)
    
    ax.set_xlabel("Cohen's d (effect size)", fontsize=12)
    ax.set_ylabel("Brain Region (sorted by effect size)", fontsize=12)
    ax.set_title(f"Effect Sizes: {cond_a} vs {cond_b}\n(positive = {cond_a} > {cond_b})", fontsize=14)
    
    # Add reference lines for effect size benchmarks
    ax.axvline(0, color="black", linestyle="-", linewidth=1.5)
    ax.axvline(0.2, color="#27ae60", linestyle="--", alpha=0.7, linewidth=1)
    ax.axvline(-0.2, color="#27ae60", linestyle="--", alpha=0.7, linewidth=1)
    ax.axvline(0.5, color="#f39c12", linestyle="--", alpha=0.7, linewidth=1)
    ax.axvline(-0.5, color="#f39c12", linestyle="--", alpha=0.7, linewidth=1)
    ax.axvline(0.8, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1)
    ax.axvline(-0.8, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color="#2ecc71", label="FDR significant"),
        mpatches.Patch(color="#f39c12", label="Raw significant"),
        mpatches.Patch(color="#bdc3c7", label="Not significant"),
        plt.Line2D([0], [0], color="#27ae60", linestyle="--", label="d=±0.2 (small)"),
        plt.Line2D([0], [0], color="#f39c12", linestyle="--", label="d=±0.5 (medium)"),
        plt.Line2D([0], [0], color="#e74c3c", linestyle="--", label="d=±0.8 (large)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "effect_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    log.info(f"Saved effect sizes plot to {output_dir / 'effect_sizes.png'}")


def _plot_diagonal_scatter(
    df: pd.DataFrame,
    conditions: List[str],
    output_dir: pathlib.Path,
    apply_fdr: bool = True,
):
    """
    Diagonal scatter plot comparing Expert vs Full RSA correlations.
    
    X-axis: Full model RSA
    Y-axis: Expert model RSA
    Diagonal: y=x line (above = Expert wins, below = Full wins)
    
    Creates two versions:
    - scatter_fdr.png: Shows FDR + raw significance
    - scatter_raw.png: Shows raw significance only
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return
    
    if len(conditions) != 2:
        return
    
    cond_a, cond_b = conditions[0], conditions[1]  # expert, full (alphabetically sorted)
    mask = (df["condition_a"] == cond_a) & (df["condition_b"] == cond_b)
    subset = df.loc[mask].copy()
    
    # Get RSA values for each condition
    x_vals = subset["mean_rho_b"].values  # Full (condition_b)
    y_vals = subset["mean_rho_a"].values  # Expert (condition_a)
    
    # --- Plot 1: With FDR correction ---
    if apply_fdr:
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Determine colors based on significance
        colors = []
        markers = []
        for _, row in subset.iterrows():
            if row.get("sig_fdr", False):
                colors.append("#2ecc71")  # Green for FDR significant
                markers.append("o")
            elif row.get("sig_raw", False):
                colors.append("#f39c12")  # Orange for raw significant
                markers.append("o")
            else:
                colors.append("#bdc3c7")  # Gray for not significant
                markers.append("o")
        
        # Plot points
        for i, (x, y, c) in enumerate(zip(x_vals, y_vals, colors)):
            ax.scatter(x, y, c=c, s=60, alpha=0.8, edgecolor="black", linewidth=0.5, zorder=3)
        
        # Add diagonal line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, zorder=1, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_xlabel(f"RSA: Full Model (ρ)", fontsize=12)
        ax.set_ylabel(f"RSA: Expert Model (ρ)", fontsize=12)
        ax.set_title("Expert vs Full RSA by Brain Region\n(above diagonal = Expert wins)", fontsize=13)
        
        # Count regions above/below diagonal
        n_expert_wins = (y_vals > x_vals).sum()
        n_full_wins = (y_vals < x_vals).sum()
        n_fdr_sig = subset["sig_fdr"].sum() if "sig_fdr" in subset.columns else 0
        n_raw_sig = subset["sig_raw"].sum() if "sig_raw" in subset.columns else 0
        
        # Add statistics text
        stats_text = f"Expert > Full: {n_expert_wins}/{len(subset)}\nFDR sig: {n_fdr_sig}, Raw sig: {n_raw_sig}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Legend
        legend_elements = [
            mpatches.Patch(color="#2ecc71", label="FDR significant"), # Green for FDR significant
            mpatches.Patch(color="#f39c12", label="Raw significant"), # Orange for raw significant
            mpatches.Patch(color="#bdc3c7", label="Not significant"), # Gray for not significant
            plt.Line2D([0], [0], color="black", linestyle="--", label="y = x"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
        
        ax.set_aspect('equal', adjustable='box')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "scatter_fdr.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        log.info(f"Saved FDR scatter plot to {output_dir / 'scatter_fdr.png'}")
    
    # --- Plot 2: Raw significance only ---
    fig, ax = plt.subplots(figsize=(7, 7))
    
    colors = []
    for _, row in subset.iterrows():
        if row.get("sig_raw", False):
            colors.append("#f39c12")  # Orange for raw significant
        else:
            colors.append("#bdc3c7")  # Gray for not significant
    
    for i, (x, y, c) in enumerate(zip(x_vals, y_vals, colors)):
        ax.scatter(x, y, c=c, s=60, alpha=0.8, edgecolor="black", linewidth=0.5, zorder=3)
    
    # Add diagonal line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, zorder=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.set_xlabel(f"RSA: Full Model (ρ)", fontsize=12)
    ax.set_ylabel(f"RSA: Expert Model (ρ)", fontsize=12)
    ax.set_title("Expert vs Full RSA by Brain Region\n(above diagonal = Expert wins)", fontsize=13)
    
    # Count regions
    n_expert_wins = (y_vals > x_vals).sum()
    n_raw_sig = subset["sig_raw"].sum() if "sig_raw" in subset.columns else 0
    
    stats_text = f"Expert > Full: {n_expert_wins}/{len(subset)}\nRaw sig: {n_raw_sig}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    legend_elements = [
        mpatches.Patch(color="#f39c12", label="Significant (p<0.05)"), # Orange for raw significant
        mpatches.Patch(color="#bdc3c7", label="Not significant"), # Gray for not significant
        plt.Line2D([0], [0], color="black", linestyle="--", label="y = x"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    
    ax.set_aspect('equal', adjustable='box')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_raw.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    log.info(f"Saved raw scatter plot to {output_dir / 'scatter_raw.png'}")


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Simple RSA analysis")
    parser.add_argument("--brain-rdms", type=pathlib.Path, required=True)
    parser.add_argument("--expert-rdms", type=pathlib.Path)
    parser.add_argument("--full-rdms", type=pathlib.Path)
    parser.add_argument("--layers", type=str, nargs="+",
                        help="Layer patterns to include (e.g., attn.c_proj mlp.c_fc)")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--no-fdr", action="store_true", help="Skip FDR correction")
    parser.add_argument("--alpha", type=float, default=0.05)
    
    args = parser.parse_args()
    
    condition_rdms = {}
    if args.expert_rdms:
        condition_rdms["expert"] = args.expert_rdms
    if args.full_rdms:
        condition_rdms["full"] = args.full_rdms
    
    if len(condition_rdms) < 2:
        parser.error("Need at least 2 conditions")
    
    run_rsa(
        brain_rdms_path=args.brain_rdms,
        condition_rdms=condition_rdms,
        output_dir=args.output,
        layer_filter=args.layers,
        apply_fdr=not args.no_fdr,
        alpha=args.alpha,
    )
