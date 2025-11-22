#!/usr/bin/env python3
"""
Filter unique experts by removing redundant neurons.

This script loads correlation matrices computed by compute_neuron_correlations.py,
identifies redundant pairs (correlation > threshold), and selects unique representatives
based on highest AP score.
"""

import argparse
import json
import pathlib
import sys

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_correlation_data(correlations_dir: pathlib.Path, concept: str, expertise_csv: pathlib.Path = None):
    """
    Load correlation matrices and expert info for all layers.
    
    Args:
        correlations_dir: Directory containing correlation files
        concept: Concept name
        expertise_csv: Optional path to original expertise CSV to extract layer names
        
    Returns:
        Dict of {layer_name: (corr_matrix, expert_info)}
    """
    correlation_files = list(correlations_dir.glob(f"{concept}_*_correlation.npy"))
    
    if not correlation_files:
        raise ValueError(f"No correlation files found for concept '{concept}' in {correlations_dir}")
    
    # If expertise_csv provided, load it to get the mapping of uuid -> layer
    uuid_to_layer = {}
    if expertise_csv and expertise_csv.exists():
        df_expertise = pd.read_csv(expertise_csv)
        uuid_to_layer = dict(zip(df_expertise['uuid'], df_expertise['layer']))
    
    layer_data = {}
    
    for corr_file in correlation_files:
        # Extract layer name from filename
        # Format: {concept}_{layer_name}_correlation.npy
        filename = corr_file.stem
        sanitized_layer_name = filename.replace(f"{concept}_", "").replace("_correlation", "")
        
        # Load correlation matrix
        corr_matrix = np.load(corr_file)
        
        # Load corresponding expert info
        expert_file = corr_file.parent / f"{concept}_{sanitized_layer_name}_experts.csv"
        if not expert_file.exists():
            print(f"Warning: Expert info not found for {sanitized_layer_name}, skipping")
            continue
        
        expert_info = pd.read_csv(expert_file)
        
        # Add layer name to expert_info using uuid mapping
        if uuid_to_layer:
            expert_info['layer'] = expert_info['uuid'].map(uuid_to_layer)
            # Get the actual layer name from the first entry
            if len(expert_info) > 0 and 'layer' in expert_info.columns:
                original_layer_name = expert_info['layer'].iloc[0]
            else:
                print(f"Warning: Could not determine layer name for {sanitized_layer_name}")
                continue
        else:
            # Fallback: try to restore original layer name (not reliable)
            original_layer_name = sanitized_layer_name.replace('_', '.')
            expert_info['layer'] = original_layer_name
        
        layer_data[original_layer_name] = (corr_matrix, expert_info)
    
    print(f"Loaded correlation data for {len(layer_data)} layers")
    return layer_data


def identify_redundant_groups(corr_matrix: np.ndarray, threshold: float = 0.9):
    """
    Identify groups of redundant neurons using graph-based clustering.
    
    Args:
        corr_matrix: Correlation matrix (NxN)
        threshold: Correlation threshold for redundancy
        
    Returns:
        List of sets, each containing indices of redundant neurons
    """
    n = corr_matrix.shape[0]
    
    if n == 1:
        return [{0}]
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Add edges for high correlations
    corr_no_diag = corr_matrix.copy()
    np.fill_diagonal(corr_no_diag, 0)
    
    for i in range(n):
        for j in range(i + 1, n):
            if corr_no_diag[i, j] > threshold:
                G.add_edge(i, j, weight=corr_no_diag[i, j])
    
    # Find connected components (redundancy groups)
    redundancy_groups = list(nx.connected_components(G))
    
    return redundancy_groups


def select_unique_representatives(
    redundancy_groups: list,
    expert_info: pd.DataFrame,
    strategy: str = "highest_ap"
):
    """
    Select one representative from each redundancy group.
    
    Args:
        redundancy_groups: List of sets with redundant neuron indices
        expert_info: DataFrame with expert information (must have 'ap' column)
        strategy: Selection strategy ('highest_ap', 'highest_variance', etc.)
        
    Returns:
        Tuple of (unique_indices, dropped_indices, groups_info)
    """
    unique_indices = []
    dropped_indices = []
    groups_info = []
    
    ap_scores = expert_info['ap'].values
    
    for group in redundancy_groups:
        group_list = list(group)
        
        if len(group) == 1:
            # No redundancy, keep it
            unique_indices.append(group_list[0])
        else:
            # Select representative based on strategy
            if strategy == "highest_ap":
                # Keep neuron with highest AP score
                group_aps = [ap_scores[idx] for idx in group_list]
                best_idx = group_list[np.argmax(group_aps)]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            unique_indices.append(best_idx)
            dropped_from_group = [idx for idx in group_list if idx != best_idx]
            dropped_indices.extend(dropped_from_group)
            
            # Store group info for reporting
            groups_info.append({
                "representative_idx": int(best_idx),
                "representative_ap": float(ap_scores[best_idx]),
                "dropped_indices": [int(x) for x in dropped_from_group],
                "dropped_aps": [float(ap_scores[x]) for x in dropped_from_group],
                "group_size": len(group),
            })
    
    return unique_indices, dropped_indices, groups_info


def filter_experts_per_layer(
    layer_data: dict,
    correlation_threshold: float,
    selection_strategy: str = "highest_ap"
):
    """
    Filter experts for all layers.
    
    Args:
        layer_data: Dict of {layer_name: (corr_matrix, expert_info)}
        correlation_threshold: Threshold for redundancy
        selection_strategy: Strategy for selecting representatives
        
    Returns:
        Dict with filtering results per layer
    """
    results = {}
    
    for layer_name, (corr_matrix, expert_info) in tqdm(layer_data.items(), desc="Filtering layers"):
        # Identify redundancy groups
        redundancy_groups = identify_redundant_groups(corr_matrix, correlation_threshold)
        
        # Select unique representatives
        unique_indices, dropped_indices, groups_info = select_unique_representatives(
            redundancy_groups, expert_info, selection_strategy
        )
        
        # Get unique experts
        unique_experts = expert_info.iloc[unique_indices].copy()
        unique_experts['is_unique'] = True
        
        # Mark dropped experts
        if dropped_indices:
            dropped_experts = expert_info.iloc[dropped_indices].copy()
            dropped_experts['is_unique'] = False
        else:
            dropped_experts = pd.DataFrame()
        
        results[layer_name] = {
            "unique_experts": unique_experts,
            "dropped_experts": dropped_experts,
            "redundancy_groups": groups_info,
            "num_original": len(expert_info),
            "num_unique": len(unique_experts),
            "num_dropped": len(dropped_indices),
        }
    
    return results


def create_filtered_expertise_csv(
    original_expertise_csv: pathlib.Path,
    filtering_results: dict,
    output_path: pathlib.Path
):
    """
    Create a new expertise CSV marking redundant experts as non-experts.
    
    Keeps the FULL neuron space but sets AP=0 for dropped (redundant) neurons.
    This ensures compatibility with ExpertSet class which requires same neuron space.
    
    Args:
        original_expertise_csv: Path to original expertise.csv
        filtering_results: Results from filter_experts_per_layer
        output_path: Where to save filtered CSV
    """
    # Load original expertise (full neuron space)
    df_filtered = pd.read_csv(original_expertise_csv).copy()
    
    # Create set of unique (layer, unit) pairs
    unique_neurons = set()
    for layer_name, results in filtering_results.items():
        for _, row in results['unique_experts'].iterrows():
            unique_neurons.add((layer_name, row['unit']))
    
    # Mark redundant neurons by setting their AP to 0
    def is_unique(row):
        return (row['layer'], row['unit']) in unique_neurons
    
    df_filtered['is_unique_expert'] = df_filtered.apply(is_unique, axis=1)
    
    # Set AP=0 for dropped (redundant) neurons to mark them as non-experts
    original_experts = (df_filtered['ap'] > 0.5).sum()
    df_filtered.loc[~df_filtered['is_unique_expert'], 'ap'] = 0.0
    unique_experts = (df_filtered['ap'] > 0.5).sum()
    
    # Save full neuron space with redundant neurons marked
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    
    print(f"Filtered expertise saved: {output_path}")
    print(f"  Total neurons in space: {len(df_filtered)}")
    print(f"  Original experts (AP > 0.5): {original_experts}")
    print(f"  Unique experts (AP > 0.5): {unique_experts}")
    print(f"  Dropped (redundant): {original_experts - unique_experts}")


def save_filtering_results(
    output_dir: pathlib.Path,
    concept: str,
    filtering_results: dict,
    metadata: dict
):
    """
    Save filtering results and statistics.
    
    Args:
        output_dir: Output directory
        concept: Concept name
        filtering_results: Results from filtering
        metadata: Metadata dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results per layer
    results_dir = output_dir / "filtering_details"
    results_dir.mkdir(exist_ok=True)
    
    for layer_name, results in filtering_results.items():
        safe_layer_name = layer_name.replace(':', '_').replace('.', '_')
        
        # Save unique experts
        if len(results['unique_experts']) > 0:
            results['unique_experts'].to_csv(
                results_dir / f"{concept}_{safe_layer_name}_unique.csv",
                index=False
            )
        
        # Save dropped experts
        if len(results['dropped_experts']) > 0:
            results['dropped_experts'].to_csv(
                results_dir / f"{concept}_{safe_layer_name}_dropped.csv",
                index=False
            )
    
    # Save summary statistics
    summary_file = output_dir / f"{concept}_filtering_summary.json"
    
    summary = {
        "concept": concept,
        "metadata": metadata,
        "per_layer_summary": {
            layer_name: {
                "num_original": results["num_original"],
                "num_unique": results["num_unique"],
                "num_dropped": results["num_dropped"],
                "redundancy_rate": results["num_dropped"] / results["num_original"] * 100 if results["num_original"] > 0 else 0,
                "num_redundancy_groups": len(results["redundancy_groups"]),
            }
            for layer_name, results in filtering_results.items()
        },
        "total_summary": {
            "total_original": sum(r["num_original"] for r in filtering_results.values()),
            "total_unique": sum(r["num_unique"] for r in filtering_results.values()),
            "total_dropped": sum(r["num_dropped"] for r in filtering_results.values()),
        }
    }
    
    summary["total_summary"]["overall_redundancy_rate"] = (
        summary["total_summary"]["total_dropped"] / summary["total_summary"]["total_original"] * 100
        if summary["total_summary"]["total_original"] > 0 else 0
    )
    
    with summary_file.open('w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved: {summary_file}")
    
    return summary


def run_filtering(
    correlation_dir: pathlib.Path,
    expertise_csv: pathlib.Path,
    output_dir: pathlib.Path,
    correlation_threshold: float,
    selection_strategy: str,
    concept: str = None,
    save_filtered_csv: bool = False,
):
    """
    Run the expert filtering pipeline.
    
    Args:
        correlation_dir: Directory containing correlation results
        expertise_csv: Path to original expertise.csv file
        output_dir: Directory where to save filtering results
        correlation_threshold: Correlation threshold for defining redundancy
        selection_strategy: Strategy for selecting representatives
        concept: Concept name (optional, inferred if None)
        save_filtered_csv: Whether to save filtered expertise CSV
    """
    # Validate inputs
    correlations_dir_path = correlation_dir / "correlations"
    if not correlations_dir_path.exists():
        print(f"Error: Correlations directory not found: {correlations_dir_path}")
        sys.exit(1)
    
    if not expertise_csv.exists():
        print(f"Error: Expertise CSV not found: {expertise_csv}")
        sys.exit(1)
    
    # Get concept name
    if concept:
        concept_name = concept
    else:
        # Try to infer from expertise CSV
        df_temp = pd.read_csv(expertise_csv, nrows=1)
        concept_name = df_temp['concept'].iloc[0]
    
    print(f"Processing concept: {concept_name}")
    print(f"Correlation threshold: {correlation_threshold}")
    print(f"Selection strategy: {selection_strategy}")
    
    # Load correlation data
    print(f"\nLoading correlation data from {correlations_dir_path}")
    layer_data = load_correlation_data(correlations_dir_path, concept_name, expertise_csv)
    
    # Filter experts
    print(f"\nIdentifying redundant neurons and selecting unique experts...")
    filtering_results = filter_experts_per_layer(
        layer_data,
        correlation_threshold,
        selection_strategy
    )
    
    # Compute statistics
    total_original = sum(r["num_original"] for r in filtering_results.values())
    total_unique = sum(r["num_unique"] for r in filtering_results.values())
    total_dropped = sum(r["num_dropped"] for r in filtering_results.values())
    redundancy_rate = total_dropped / total_original * 100 if total_original > 0 else 0
    
    # Print results
    print(f"\n{'='*60}")
    print("FILTERING RESULTS")
    print(f"{'='*60}")
    print(f"Total original experts: {total_original}")
    print(f"Unique experts: {total_unique}")
    print(f"Dropped (redundant): {total_dropped}")
    print(f"Redundancy rate: {redundancy_rate:.1f}%")
    
    print(f"\nPer-layer breakdown:")
    for layer_name, results in filtering_results.items():
        if results["num_dropped"] > 0:
            layer_redundancy = results["num_dropped"] / results["num_original"] * 100
            print(f"  {layer_name}:")
            print(f"    {results['num_original']} → {results['num_unique']} "
                  f"({results['num_dropped']} dropped, {layer_redundancy:.1f}% redundancy)")
    
    # Save results
    metadata = {
        "correlation_threshold": correlation_threshold,
        "selection_strategy": selection_strategy,
    }
    
    summary = save_filtering_results(
        output_dir,
        concept_name,
        filtering_results,
        metadata
    )
    
    # Optionally create filtered CSV
    if save_filtered_csv:
        output_csv = output_dir / f"{concept_name}_expertise_unique.csv"
        create_filtered_expertise_csv(
            expertise_csv,
            filtering_results,
            output_csv
        )
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        prog="filter_unique_experts.py",
        description=(
            "Filter unique experts by identifying and removing redundant neurons. "
            "Redundant neurons are identified based on correlation matrices, and "
            "representatives are selected using the specified strategy (e.g., highest AP)."
        ),
    )
    
    parser.add_argument(
        "--correlation-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing correlation results from compute_neuron_correlations.py",
    )
    parser.add_argument(
        "--expertise-csv",
        type=pathlib.Path,
        required=True,
        help="Path to original expertise.csv file",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory where to save filtering results",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for defining redundancy (default: 0.9)",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        default="highest_ap",
        choices=["highest_ap"],
        help="Strategy for selecting representatives (default: highest_ap)",
    )
    parser.add_argument(
        "--concept",
        type=str,
        help="Concept name (if not provided, will be inferred)",
    )
    parser.add_argument(
        "--save-filtered-csv",
        action="store_true",
        help="Save filtered expertise.csv with only unique experts",
    )
    
    args = parser.parse_args()
    
    run_filtering(
        correlation_dir=args.correlation_dir,
        expertise_csv=args.expertise_csv,
        output_dir=args.output_dir,
        correlation_threshold=args.correlation_threshold,
        selection_strategy=args.selection_strategy,
        concept=args.concept,
        save_filtered_csv=args.save_filtered_csv,
    )


if __name__ == "__main__":
    main()

