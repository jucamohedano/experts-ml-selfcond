#!/usr/bin/env python3
"""
Visualize global redundancy summary across all concepts.

This script:
1. Scans a directory for all `*_filtering_summary.json` files.
2. Aggregates per-concept statistics (unique vs dropped experts).
3. Generates plots for redundancy rates and expert counts per concept.
"""

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_summaries(root_dir: pathlib.Path, pattern: str = "**/*_filtering_summary.json"):
    """
    Find and load all filtering summary JSONs.
    
    Returns:
        Tuple of (per_concept_df, correlation_threshold or None)
    """
    summary_files = list(root_dir.glob(pattern))
    
    if not summary_files:
        raise ValueError(f"No summary files found in {root_dir} matching {pattern}")
    
    print(f"Found {len(summary_files)} summary files.")
    
    data = []
    correlation_threshold = None
    
    for f in summary_files:
        try:
            with open(f, 'r') as file:
                content = json.load(file)
                
            concept = content.get("concept", f.stem.replace("_filtering_summary", ""))
            totals = content.get("total_summary", {})
            
            # Extract correlation threshold from metadata (should be same for all)
            metadata = content.get("metadata", {})
            if correlation_threshold is None and "correlation_threshold" in metadata:
                correlation_threshold = metadata["correlation_threshold"]
            
            data.append({
                "Concept": concept,
                "Original Total": totals.get("total_original", 0),
                "Unique Experts": totals.get("total_unique", 0),
                "Dropped (Redundant)": totals.get("total_dropped", 0),
                "Redundancy Rate (%)": totals.get("overall_redundancy_rate", 0.0),
            })
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
    
    df = pd.DataFrame(data)
    
    return df, correlation_threshold


def plot_redundancy_rates(df: pd.DataFrame, output_dir: pathlib.Path, correlation_threshold: float = None):
    """Plot sorted redundancy rates per concept with percentage labels."""
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.25 + 2))
    
    # Sort by redundancy rate
    df_sorted = df.sort_values("Redundancy Rate (%)", ascending=True).reset_index(drop=True)
    
    concepts = df_sorted["Concept"]
    rates = df_sorted["Redundancy Rate (%)"].values
    
    # Create horizontal bars
    bars = ax.barh(concepts, rates, color="#4c72b0")
    
    # Add percentage labels at the end of each bar
    for i, rate in enumerate(rates):
        ax.text(
            rate + 0.01,  # Slight offset from bar end
            i,
            f"{rate:.1f}%",
            va='center',
            ha='left',
            fontsize=8,
            color='#4c72b0',
            fontweight='bold'
        )
    
    # Build title with correlation threshold if available
    title = "Redundancy Rate per Concept\n(% of each concept's experts dropped due to high correlation)"
    if correlation_threshold is not None:
        title += f"\nPearson correlation threshold: {correlation_threshold}"
    
    ax.set_title(title)
    ax.set_xlabel("Redundancy Rate (%)")
    ax.set_ylabel("Concept")
    ax.grid(axis='x', alpha=0.3)
    
    # Extend x-axis to fit labels
    x_max = rates.max() if len(rates) > 0 else 100
    ax.set_xlim(0, min(x_max * 1.15, 105))
    
    plt.tight_layout()
    
    save_path = output_dir / "global_redundancy_rates.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved redundancy rate plot: {save_path}")
    plt.close()


def plot_expert_breakdown(df: pd.DataFrame, output_dir: pathlib.Path, correlation_threshold: float = None):
    """Plot stacked bar chart of Unique vs Dropped experts with labels."""
    # Prepare data for stacked plot
    df_sorted = df.sort_values("Original Total", ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.25 + 2))
    
    concepts = df_sorted["Concept"]
    unique = df_sorted["Unique Experts"].values
    dropped = df_sorted["Dropped (Redundant)"].values
    original = df_sorted["Original Total"].values
    
    # Create horizontal bars
    p1 = ax.barh(concepts, unique, label='Unique Experts', color='#55a868')
    p2 = ax.barh(concepts, dropped, left=unique, label='Redundant (Dropped)', color='#c44e52')
    
    # Add dropped count labels at the end of each bar
    for i, (orig, drop) in enumerate(zip(original, dropped)):
        if drop > 0:
            # Position label just after the end of the full bar
            ax.text(
                orig + orig * 0.01,  # Slight offset from bar end
                i,
                f"-{int(drop)}",
                va='center',
                ha='left',
                fontsize=8,
                color='#c44e52',
                fontweight='bold'
            )
    
    # Build title with correlation threshold if available
    title = "Expert Breakdown per Concept: Unique vs. Redundant"
    if correlation_threshold is not None:
        title += f"\nPearson correlation threshold: {correlation_threshold}"
    
    ax.set_title(title)
    ax.set_xlabel("Number of Expert Neurons")
    ax.set_ylabel("Concept")
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    # Extend x-axis slightly to fit labels
    x_max = original.max() if len(original) > 0 else 1
    ax.set_xlim(0, x_max * 1.12)
    
    plt.tight_layout()
    
    save_path = output_dir / "global_expert_breakdown.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved breakdown plot: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize global redundancy summary.")
    parser.add_argument("--input-dir", type=pathlib.Path, required=True, help="Root directory containing concept subfolders")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Where to save the global plots")
    
    args = parser.parse_args()
    
    # Load Data
    try:
        df, correlation_threshold = load_summaries(args.input_dir)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    if df.empty:
        print("No data extracted.")
        sys.exit(1)
    
    if correlation_threshold is not None:
        print(f"Correlation threshold used: {correlation_threshold}")
        
    # Create Output Dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV summary
    csv_path = args.output_dir / "global_redundancy_summary.csv"
    df.sort_values("Redundancy Rate (%)", ascending=False).to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")
    
    # Generate Plots
    plot_redundancy_rates(df, args.output_dir, correlation_threshold)
    plot_expert_breakdown(df, args.output_dir, correlation_threshold)
    
    # Compute redundancy rate statistics
    avg_redundancy = df["Redundancy Rate (%)"].mean()
    min_redundancy = df["Redundancy Rate (%)"].min()
    max_redundancy = df["Redundancy Rate (%)"].max()
    
    # Print summary
    print(f"\n{'='*60}")
    print("REDUNDANCY SUMMARY")
    print(f"{'='*60}")
    print(f"Number of concepts: {len(df)}")
    print(f"Average redundancy rate: {avg_redundancy:.2f}%")
    print(f"Min redundancy rate: {min_redundancy:.2f}%")
    print(f"Max redundancy rate: {max_redundancy:.2f}%")
    
    # Build summary JSON
    summary_data = {
        "num_concepts": len(df),
        "correlation_threshold": correlation_threshold,
        "redundancy_rates": {
            "average": round(avg_redundancy, 2),
            "min": round(min_redundancy, 2),
            "max": round(max_redundancy, 2),
        },
    }
    
    # Save statistics
    summary_file = args.output_dir / "global_redundancy_summary.json"
    with summary_file.open('w') as f:
        json.dump(summary_data, f, indent=2) 
    print(f"\nSaved global redundancy summary: {summary_file}")


if __name__ == "__main__":
    main()
