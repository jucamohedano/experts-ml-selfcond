#!/usr/bin/env python3
"""
Compute neuron-to-neuron correlation matrices for expert neurons.

This script loads activation responses and expertise results for a concept,
identifies expert neurons (AP > threshold), and computes Pearson correlation
between their activation patterns.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from selfcond.responses import read_responses_from_cached


def load_expert_neurons(expertise_csv: pathlib.Path, ap_threshold: float = 0.5):
    """
    Load expert neurons from expertise CSV file.
    
    Args:
        expertise_csv: Path to expertise.csv file
        ap_threshold: AP threshold for defining experts
        
    Returns:
        DataFrame with expert neurons only
    """
    print(f"Loading expertise from {expertise_csv}")
    df = pd.read_csv(expertise_csv)
    
    # Filter to experts only
    experts_df = df[df['ap'] > ap_threshold].copy()
    
    print(f"Total neurons: {len(df)}")
    print(f"Expert neurons (AP > {ap_threshold}): {len(experts_df)} ({100*len(experts_df)/len(df):.1f}%)")
    
    return experts_df


def compute_correlation_matrix_per_layer(
    responses: dict,
    experts_df: pd.DataFrame,
    layer_name: str
) -> tuple:
    """
    Compute correlation matrix for expert neurons in a specific layer.
    
    Args:
        responses: Dict of {layer_name: activations array}
        experts_df: DataFrame with expert neurons
        layer_name: Name of the layer to process
        
    Returns:
        Tuple of (correlation_matrix, expert_indices, expert_info)
    """
    # Get experts for this layer
    layer_experts = experts_df[experts_df['layer'] == layer_name].copy()
    
    if len(layer_experts) == 0:
        return None, None, None
    
    # Get activation patterns for these experts
    layer_activations = responses[layer_name]  # Shape: [all_units, num_sentences]
    expert_units = layer_experts['unit'].values
    expert_activations = layer_activations[expert_units, :]  # Shape: [num_experts, num_sentences]
    
    # Compute Pearson correlation matrix
    if len(expert_units) > 1:
        correlation_matrix = np.corrcoef(expert_activations)
    else:
        correlation_matrix = np.array([[1.0]])
    
    # Create expert info for reference
    expert_info = layer_experts[['unit', 'ap', 'uuid']].reset_index(drop=True)
    
    return correlation_matrix, expert_units, expert_info


def save_correlation_results(
    output_dir: pathlib.Path,
    concept: str,
    layer_correlations: dict,
    metadata: dict
):
    """
    Save correlation matrices and metadata.
    
    Args:
        output_dir: Output directory
        concept: Concept name
        layer_correlations: Dict of {layer_name: (corr_matrix, units, info)}
        metadata: Metadata dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each layer's correlation matrix
    correlations_dir = output_dir / "correlations"
    correlations_dir.mkdir(exist_ok=True)
    
    for layer_name, (corr_matrix, units, expert_info) in layer_correlations.items():
        if corr_matrix is None:
            continue
            
        # Sanitize layer name for filename
        safe_layer_name = layer_name.replace(':', '_').replace('.', '_')
        
        # Save correlation matrix as numpy array
        np.save(
            correlations_dir / f"{concept}_{safe_layer_name}_correlation.npy",
            corr_matrix
        )
        
        # Save expert info as CSV
        expert_info.to_csv(
            correlations_dir / f"{concept}_{safe_layer_name}_experts.csv",
            index=False
        )
    
    # Save metadata
    metadata_file = output_dir / f"{concept}_correlation_metadata.json"
    with metadata_file.open('w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - Correlation matrices: {correlations_dir}")
    print(f"  - Metadata: {metadata_file}")


def compute_correlation_statistics(layer_correlations: dict, correlation_threshold: float = 0.9):
    """
    Compute statistics about correlations.
    
    Args:
        layer_correlations: Dict of layer correlation data
        correlation_threshold: Threshold for identifying high correlation
        
    Returns:
        Statistics dict
    """
    stats = {
        "total_experts": 0,
        "experts_per_layer": {},
        "high_correlation_pairs_per_layer": {},
        "total_high_correlation_pairs": 0,
    }
    
    for layer_name, (corr_matrix, units, expert_info) in layer_correlations.items():
        if corr_matrix is None:
            continue
            
        num_experts = len(units)
        stats["total_experts"] += num_experts
        stats["experts_per_layer"][layer_name] = num_experts
        
        # Count high correlation pairs (upper triangle only, exclude diagonal)
        if num_experts > 1:
            # Set diagonal to 0
            corr_matrix_no_diag = corr_matrix.copy()
            np.fill_diagonal(corr_matrix_no_diag, 0)
            
            # Count pairs above threshold
            high_corr_pairs = np.sum(np.triu(corr_matrix_no_diag > correlation_threshold, k=1))
            stats["high_correlation_pairs_per_layer"][layer_name] = int(high_corr_pairs)
            stats["total_high_correlation_pairs"] += int(high_corr_pairs)
    
    return stats


def run_correlation_analysis(
    responses_dir: pathlib.Path,
    expertise_csv: pathlib.Path,
    output_dir: pathlib.Path,
    ap_threshold: float,
    correlation_threshold: float,
    concept: str = None,
):
    """
    Run the correlation analysis pipeline.
    
    Args:
        responses_dir: Directory containing response files (*.pkl)
        expertise_csv: Path to expertise.csv file
        output_dir: Directory where to save correlation results
        ap_threshold: AP threshold for defining expert neurons
        correlation_threshold: Correlation threshold for reporting statistics
        concept: Concept name (optional, read from expertise CSV if None)
    """
    # Validate inputs
    if not responses_dir.exists():
        print(f"Error: Responses directory not found: {responses_dir}")
        sys.exit(1)
    
    if not expertise_csv.exists():
        print(f"Error: Expertise CSV not found: {expertise_csv}")
        sys.exit(1)
    
    # Load expert neurons
    experts_df = load_expert_neurons(expertise_csv, ap_threshold)
    
    if len(experts_df) == 0:
        print(f"Error: No expert neurons found with AP > {ap_threshold}")
        sys.exit(1)
    
    # Get concept name
    if concept is None:
        concept = experts_df['concept'].iloc[0]
    concept_group = experts_df['group'].iloc[0]
    
    print(f"\nProcessing concept: {concept_group}/{concept}")
    
    # Load responses
    print(f"\nLoading responses from {responses_dir}")
    responses, labels, response_names = read_responses_from_cached(
        responses_dir,
        concept=concept,
        verbose=False
    )
    
    # Get unique layers that have experts
    expert_layers = experts_df['layer'].unique()
    print(f"\nComputing correlations for {len(expert_layers)} layers with experts")
    
    # Compute correlation matrices per layer
    layer_correlations = {}
    for layer_name in tqdm(expert_layers, desc="Computing correlations"):
        corr_matrix, units, expert_info = compute_correlation_matrix_per_layer(
            responses, experts_df, layer_name
        )
        if corr_matrix is not None:
            layer_correlations[layer_name] = (corr_matrix, units, expert_info)
    
    # Compute statistics
    stats = compute_correlation_statistics(layer_correlations, correlation_threshold)
    
    # Prepare metadata
    metadata = {
        "concept": concept,
        "concept_group": concept_group,
        "ap_threshold": ap_threshold,
        "correlation_threshold": correlation_threshold,
        "num_sentences": responses[list(response_names)[0]].shape[1] if response_names else 0,
        "statistics": stats,
    }
    
    # Print statistics
    print(f"\n{'='*60}")
    print("CORRELATION STATISTICS")
    print(f"{'='*60}")
    print(f"Total expert neurons: {stats['total_experts']}")
    print(f"Layers with experts: {len(expert_layers)}")
    print(f"High correlation pairs (>{correlation_threshold}): {stats['total_high_correlation_pairs']}")
    
    if stats['total_high_correlation_pairs'] > 0:
        print(f"\nLayers with redundant neurons:")
        for layer_name, count in stats['high_correlation_pairs_per_layer'].items():
            if count > 0:
                print(f"  {layer_name}: {count} pairs")
    
    # Save results
    save_correlation_results(
        output_dir,
        concept,
        layer_correlations,
        metadata
    )
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        prog="compute_neuron_correlations.py",
        description=(
            "Compute Pearson correlation between expert neurons based on their "
            "activation patterns. This identifies redundant neurons that respond "
            "similarly across sentences."
        ),
    )
    
    parser.add_argument(
        "--responses-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing response files (*.pkl)",
    )
    parser.add_argument(
        "--expertise-csv",
        type=pathlib.Path,
        required=True,
        help="Path to expertise.csv file",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory where to save correlation results",
    )
    parser.add_argument(
        "--ap-threshold",
        type=float,
        default=0.5,
        help="AP threshold for defining expert neurons (default: 0.5)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for reporting statistics (default: 0.9)",
    )
    parser.add_argument(
        "--concept",
        type=str,
        help="Concept name (if not provided, will be read from expertise CSV)",
    )
    
    args = parser.parse_args()
    
    run_correlation_analysis(
        responses_dir=args.responses_dir,
        expertise_csv=args.expertise_csv,
        output_dir=args.output_dir,
        ap_threshold=args.ap_threshold,
        correlation_threshold=args.correlation_threshold,
        concept=args.concept,
    )


if __name__ == "__main__":
    main()

