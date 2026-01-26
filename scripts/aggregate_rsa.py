#!/usr/bin/env python3
"""
Aggregate RSA results across multiple AP threshold runs.

Creates summary tables and comparison plots to identify the best-performing
expert neuron configuration.

Usage:
    python scripts/aggregate_rsa.py \
        --results-dir results/rsa/.../Qwen3-30B-A3B-Instruct-2507_gpt2 \
        --output results/rsa_summary
"""

import json
import logging
import pathlib
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def parse_condition_name(run_tag: str) -> Dict:
    """
    Parse condition info from run_tag.
    
    Examples:
        "cot-ap-0.5_vs_full" -> {"condition": "cot-ap-0.5", "ap": 0.5, "type": "standard"}
        "cot-ap-0.5-unique-corr_0.8_vs_full" -> {"condition": "...", "ap": 0.5, "corr": 0.8, "type": "unique"}
    """
    info = {"run_tag": run_tag, "condition": run_tag.replace("_vs_full", "")}
    
    # Extract AP threshold
    ap_match = re.search(r"ap-?([\d.]+)", run_tag)
    if ap_match:
        info["ap"] = float(ap_match.group(1))
    else:
        info["ap"] = None
    
    # Check for unique correlation threshold
    corr_match = re.search(r"unique-corr_?([\d.]+)", run_tag)
    if corr_match:
        info["corr_threshold"] = float(corr_match.group(1))
        info["type"] = "unique"
    else:
        info["corr_threshold"] = None
        info["type"] = "standard"
    
    return info


def load_run_results(run_dir: pathlib.Path) -> Optional[Dict]:
    """Load results from a single run directory."""
    
    # Find the latest timestamp subdirectory
    subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    
    latest = sorted(subdirs)[-1]
    
    results_file = latest / "rsa_results.csv"
    metadata_file = latest / "rsa_metadata.json"
    
    if not results_file.exists():
        log.warning(f"No results in {latest}")
        return None
    
    df = pd.read_csv(results_file)
    
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
    
    return {
        "df": df,
        "metadata": metadata,
        "path": latest,
    }


def aggregate_results(
    results_dir: pathlib.Path,
    output_dir: pathlib.Path,
):
    """
    Aggregate results from multiple RSA runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all run directories
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    log.info(f"Found {len(run_dirs)} run directories in {results_dir}")
    
    summary_rows = []
    
    for run_dir in sorted(run_dirs):
        run_tag = run_dir.name
        
        result = load_run_results(run_dir)
        if result is None:
            continue
        
        df = result["df"]
        
        # Parse condition info
        info = parse_condition_name(run_tag)
        
        # Compute summary stats
        n_regions = len(df)
        
        # Effect size
        mean_d = df["cohens_d"].mean()
        std_d = df["cohens_d"].std()
        
        # Direction: how many regions favor Expert
        n_expert_wins = (df["mean_diff"] > 0).sum()
        pct_expert_wins = n_expert_wins / n_regions * 100
        
        # Significance counts
        n_sig_raw = df["sig_raw"].sum() if "sig_raw" in df.columns else 0
        n_sig_fdr = df["sig_fdr"].sum() if "sig_fdr" in df.columns else 0
        
        # Mean RSA values
        mean_rsa_expert = df["mean_rho_a"].mean()
        mean_rsa_full = df["mean_rho_b"].mean()
        
        summary_rows.append({
            "condition": info["condition"],
            "ap_threshold": info.get("ap"),
            "corr_threshold": info.get("corr_threshold"),
            "type": info["type"],
            "n_regions": n_regions,
            "mean_cohens_d": mean_d,
            "std_cohens_d": std_d,
            "n_expert_wins": n_expert_wins,
            "pct_expert_wins": pct_expert_wins,
            "n_sig_raw": n_sig_raw,
            "n_sig_fdr": n_sig_fdr,
            "mean_rsa_expert": mean_rsa_expert,
            "mean_rsa_full": mean_rsa_full,
            "mean_rsa_diff": mean_rsa_expert - mean_rsa_full,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Sort by effect size (descending)
    summary_df = summary_df.sort_values("mean_cohens_d", ascending=False)
    
    # Save summary
    summary_file = output_dir / "rsa_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    log.info(f"Saved summary to {summary_file}")
    
    # Print summary table
    log.info("\n" + "="*80)
    log.info("RSA SUMMARY BY AP THRESHOLD")
    log.info("="*80)
    log.info(f"{'Condition':<30} {'Mean d':>8} {'Expert%':>8} {'FDR sig':>8} {'Raw sig':>8}")
    log.info("-"*80)
    for _, row in summary_df.iterrows():
        log.info(f"{row['condition']:<30} {row['mean_cohens_d']:>+8.3f} {row['pct_expert_wins']:>7.1f}% {row['n_sig_fdr']:>8} {row['n_sig_raw']:>8}")
    
    # Identify best
    best = summary_df.iloc[0]
    log.info("-"*80)
    log.info(f"BEST: {best['condition']} (d = {best['mean_cohens_d']:+.3f})")
    
    # Generate plots
    _plot_ap_comparison(summary_df, output_dir)
    
    return summary_df


def _plot_ap_comparison(df: pd.DataFrame, output_dir: pathlib.Path):
    """Bar chart comparing effect sizes across all AP thresholds."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return
    
    # Sort by effect size (descending) for display
    plot_df = df.sort_values("mean_cohens_d", ascending=False).copy()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(plot_df))
    
    # Color by sign of effect
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in plot_df["mean_cohens_d"]]
    
    bars = ax.bar(x, plot_df["mean_cohens_d"], color=colors, edgecolor="black", alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["condition"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Cohen's d (effect size)", fontsize=12)
    ax.set_xlabel("Expert Condition", fontsize=12)
    ax.set_title("RSA Effect Size by Expert Threshold\n(Expert vs Full Comparison)", fontsize=14)
    
    # Reference lines
    ax.axhline(0, color="black", linestyle="-", linewidth=1.5)
    ax.axhline(0.2, color="gray", linestyle="--", alpha=0.5, label="Small (d=0.2)")
    ax.axhline(-0.2, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Medium (d=0.5)")
    ax.axhline(-0.5, color="gray", linestyle=":", alpha=0.5)
    
    # Add value labels
    for i, (_, row) in enumerate(plot_df.iterrows()):
        offset = 0.03 if row["mean_cohens_d"] >= 0 else -0.08
        ax.text(i, row["mean_cohens_d"] + offset, f"{row['mean_cohens_d']:+.2f}", 
                ha="center", va="bottom" if row["mean_cohens_d"] >= 0 else "top", fontsize=9)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color="#2ecc71", label="Expert > Full"),
        mpatches.Patch(color="#e74c3c", label="Full > Expert"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="d = ±0.2"),
        plt.Line2D([0], [0], color="gray", linestyle=":", label="d = ±0.5"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    log.info(f"Saved AP comparison plot to {output_dir / 'ap_comparison.png'}")


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Aggregate RSA results")
    parser.add_argument("--results-dir", type=pathlib.Path, required=True,
                        help="Directory containing run subdirectories")
    parser.add_argument("--output", type=pathlib.Path, required=True,
                        help="Output directory for summary")
    
    args = parser.parse_args()
    
    aggregate_results(
        results_dir=args.results_dir,
        output_dir=args.output,
    )
