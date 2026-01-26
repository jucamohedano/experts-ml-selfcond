#!/usr/bin/env python3
"""
Visualize neuron correlation matrices and redundancy patterns.

This script creates visualizations including:
- Correlation heatmaps per layer
- Distribution of correlation values
- Redundancy network graphs
"""

import argparse
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle


def load_correlation_data(correlations_dir: pathlib.Path, concept: str):
    """Load correlation matrices for all layers."""
    correlation_files = list(correlations_dir.glob(f"{concept}_*_correlation.npy"))
    
    if not correlation_files:
        raise ValueError(f"No correlation files found for concept '{concept}' in {correlations_dir}")
    
    layer_data = {}
    
    for corr_file in correlation_files:
        filename = corr_file.stem
        layer_name = filename.replace(f"{concept}_", "").replace("_correlation", "")
        
        corr_matrix = np.load(corr_file)
        
        expert_file = corr_file.parent / f"{concept}_{layer_name}_experts.csv"
        if expert_file.exists():
            expert_info = pd.read_csv(expert_file)
        else:
            expert_info = None
        
        layer_data[layer_name] = (corr_matrix, expert_info)
    
    return layer_data


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    layer_name: str,
    output_path: pathlib.Path,
    threshold: float = 0.9,
    max_neurons: int = 50
):
    """
    Plot correlation heatmap for a layer.
    
    Args:
        corr_matrix: Correlation matrix
        layer_name: Layer name for title
        output_path: Where to save the plot
        threshold: Threshold line to draw
        max_neurons: Maximum neurons to display (for readability)
    """
    n = corr_matrix.shape[0]
    
    if n > max_neurons:
        print(f"  Note: Showing only first {max_neurons} neurons (out of {n})")
        corr_matrix = corr_matrix[:max_neurons, :max_neurons]
        n = max_neurons
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation', rotation=270, labelpad=20)
    
    # Highlight high correlations
    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i, j] > threshold:
                rect = Rectangle((j-0.5, i-0.5), 1, 1, linewidth=1, 
                               edgecolor='green', facecolor='none')
                ax.add_patch(rect)
    
    # Labels
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Neuron Index')
    ax.set_title(f'Neuron Correlation Matrix\n{layer_name}', pad=20)
    
    # Add text annotation
    high_corr_count = np.sum(np.triu(corr_matrix > threshold, k=1))
    ax.text(0.02, 0.98, f'High correlations (>{threshold}): {high_corr_count}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_distribution(
    layer_data: dict,
    output_path: pathlib.Path,
    threshold: float = 0.9
):
    """
    Plot distribution of correlation values across all layers.
    
    Args:
        layer_data: Dict of {layer_name: (corr_matrix, expert_info)}
        output_path: Where to save the plot
        threshold: Threshold to mark on plot
    """
    # Collect all upper-triangle correlations (excluding diagonal)
    all_correlations = []
    
    for layer_name, (corr_matrix, _) in layer_data.items():
        if corr_matrix.shape[0] > 1:
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            all_correlations.extend(upper_triangle)
    
    if not all_correlations:
        print("No correlations to plot")
        return
    
    all_correlations = np.array(all_correlations)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(all_correlations, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold})')
    ax1.set_xlabel('Pearson Correlation')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Neuron-to-Neuron Correlations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Statistics text
    stats_text = f"""
    Total pairs: {len(all_correlations):,}
    Mean: {np.mean(all_correlations):.3f}
    Median: {np.median(all_correlations):.3f}
    Std: {np.std(all_correlations):.3f}
    
    High correlation (>{threshold}):
    Count: {np.sum(all_correlations > threshold):,}
    Percentage: {100 * np.sum(all_correlations > threshold) / len(all_correlations):.1f}%
    """
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Cumulative distribution
    sorted_corr = np.sort(all_correlations)
    cumulative = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
    ax2.plot(sorted_corr, cumulative, linewidth=2)
    ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold})')
    ax2.set_xlabel('Pearson Correlation')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def parse_layer_name(layer_name: str) -> dict:
    """
    Parse a layer name to extract meaningful information.
    
    Handles both original format (transformer.h.0.attn.c_attn:0) 
    and sanitized format (transformer.h.0.attn.c.attn.0).
    
    Args:
        layer_name: Layer name string
        
    Returns:
        Dict with 'layer_num', 'layer_type', 'component', 'full_display'
    """
    parts = layer_name.split('.')
    
    # Extract layer number (after 'h.')
    layer_num = None
    match = re.search(r'h\.(\d+)', layer_name)
    if match:
        layer_num = int(match.group(1))
    
    # Determine layer type (ATTN or MLP)
    layer_type = None
    if '.attn.' in layer_name or 'attn' in parts:
        layer_type = 'ATTN'
    elif '.mlp.' in layer_name or 'mlp' in parts:
        layer_type = 'MLP'
    
    # Determine component
    component = None
    if layer_type == 'ATTN':
        # For attention: c_attn or c_proj
        # Pattern: ...attn.c.attn... or ...attn.c.proj...
        if 'c.attn' in layer_name:
            component = 'c_attn'
        elif 'c.proj' in layer_name:
            component = 'c_proj'
        elif 'c_attn' in layer_name:
            component = 'c_attn'
        elif 'c_proj' in layer_name:
            component = 'c_proj'
    elif layer_type == 'MLP':
        # For MLP: c_fc or c_proj
        # Pattern: ...mlp.c.fc... or ...mlp.c.proj...
        if 'c.fc' in layer_name:
            component = 'c_fc'
        elif 'c.proj' in layer_name:
            component = 'c_proj'
        elif 'c_fc' in layer_name:
            component = 'c_fc'
        elif 'c_proj' in layer_name:
            component = 'c_proj'
    
    # Create display name
    if layer_num is not None and layer_type is not None and component is not None:
        display = f"h{layer_num} {layer_type} {component}"
    else:
        # Fallback: use meaningful parts
        if len(parts) >= 4:
            # Try to extract last 4 parts: e.g., "h.3.mlp.c.fc" -> "h3 MLP c_fc"
            relevant_parts = parts[-4:]
            if 'h' in relevant_parts:
                h_idx = relevant_parts.index('h')
                if h_idx + 1 < len(relevant_parts):
                    try:
                        num = int(relevant_parts[h_idx + 1])
                        display = f"h{num} {' '.join(relevant_parts[h_idx+2:])}"
                    except ValueError:
                        display = '.'.join(relevant_parts)
                else:
                    display = '.'.join(relevant_parts)
            else:
                display = '.'.join(relevant_parts)
        else:
            display = layer_name
    
    return {
        'layer_num': layer_num,
        'layer_type': layer_type,
        'component': component,
        'full_display': display
    }


def plot_filtering_summary(
    summary_file: pathlib.Path,
    output_path: pathlib.Path,
    correlation_dir: pathlib.Path = None,
    concept: str = None,
    correlation_threshold: float = 0.9,
    max_heatmaps: int = 5
):
    """
    Plot summary of filtering results.
    
    Creates a single figure with expert neuron distribution and heatmaps for
    layers with highest correlations.
    
    Args:
        summary_file: Path to filtering summary JSON
        output_path: Where to save the plot
        correlation_dir: Directory containing correlation results (for heatmaps)
        concept: Concept name (for loading correlation data)
        correlation_threshold: Threshold for correlations (default: 0.9)
        max_heatmaps: Maximum number of heatmaps to generate (default: 5)
    """
    with summary_file.open('r') as f:
        summary = json.load(f)
    
    # Extract and parse all layer data
    layer_names = list(summary['per_layer_summary'].keys())
    parsed_layers = []
    for name in layer_names:
        parsed = parse_layer_name(name)
        parsed_layers.append({
            'name': name,
            'layer_num': parsed['layer_num'] if parsed['layer_num'] is not None else 999,
            'layer_type': parsed['layer_type'] or 'UNKNOWN',
            'component': parsed['component'] or 'unknown',
            'display': parsed['full_display'],
            'original': summary['per_layer_summary'][name]['num_original'],
            'unique': summary['per_layer_summary'][name]['num_unique'],
            'dropped': summary['per_layer_summary'][name]['num_dropped'],
        })
    
    # Sort by layer number (primary) and component (secondary)
    parsed_layers.sort(key=lambda x: (x['layer_num'], x['component']))
    
    # Extract sorted data
    display_names = [p['display'] for p in parsed_layers]
    layer_types_list = [p['layer_type'] for p in parsed_layers]
    original_counts = [p['original'] for p in parsed_layers]
    unique_counts = [p['unique'] for p in parsed_layers]
    dropped_counts = [p['dropped'] for p in parsed_layers]
    
    # Calculate redundancy rates
    redundancy_rates = [
        d / o * 100 if o > 0 else 0 
        for d, o in zip(dropped_counts, original_counts)
    ]
    
    # Get overall statistics
    total_original = summary['total_summary']['total_original']
    total_unique = summary['total_summary']['total_unique']
    total_dropped = summary['total_summary']['total_dropped']
    overall_rate = summary['total_summary']['overall_redundancy_rate']
    
    # Calculate original redundancy rate (before filtering)
    original_redundancy_rate = (total_dropped / total_original * 100) if total_original > 0 else 0
    
    x = np.arange(len(display_names))
    width = 0.6
    
    # Color coding by layer type for unique neurons
    type_colors = {
        'ATTN': '#2ECC71',  # Green for attention
        'MLP': '#3498DB',   # Blue for MLP
        'UNKNOWN': '#95A5A6'  # Gray for unknown
    }
    dropped_color = '#E74C3C'  # Bright red for dropped
    
    # Create bar colors based on layer type
    unique_colors = [type_colors.get(lt, '#95A5A6') for lt in layer_types_list]
    
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Expert Neuron Distribution
    ax.bar(x, unique_counts, width, label='Unique', color=unique_colors, alpha=0.8)
    ax.bar(x, dropped_counts, width, bottom=unique_counts, label='Dropped (Redundant)', 
            color=dropped_color, alpha=0.8)
    ax.set_xlabel('Layer (Ordered)', fontsize=11)
    ax.set_ylabel('Number of Expert Neurons', fontsize=11)
    ax.set_title('Expert Neuron Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
    
    # Create legend with layer types and dropped
    legend_elements = [
        Patch(facecolor='#2ECC71', alpha=0.8, label='ATTN'),
        Patch(facecolor='#3498DB', alpha=0.8, label='MLP'),
        Patch(facecolor='#E74C3C', alpha=0.8, label='Dropped (Redundant)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Overall title with redundancy mapping
    fig.suptitle(
        f"Expert Filtering Summary: {total_original} → {total_unique} experts "
        f"({total_dropped} dropped) | "
        f"Redundancy: {original_redundancy_rate:.1f}% → 0% "
        f"({overall_rate:.1f}% removed)",
        fontsize=13, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate heatmaps for layers with highest correlations
    if correlation_dir and concept:
        print(f"\nGenerating heatmaps for layers with highest correlations...")
        
        # Load correlation metadata to find layers with most correlations
        correlation_metadata_file = correlation_dir / f"{concept}_correlation_metadata.json"
        if correlation_metadata_file.exists():
            with correlation_metadata_file.open('r') as f:
                corr_metadata = json.load(f)
            
            # Get high correlation pairs per layer
            high_corr_pairs = corr_metadata.get('statistics', {}).get('high_correlation_pairs_per_layer', {})
            
            # Sort layers by number of high correlation pairs (descending)
            sorted_layers = sorted(high_corr_pairs.items(), key=lambda x: x[1], reverse=True)
            
            # Take top layers with correlations
            top_layers = [(name, count) for name, count in sorted_layers if count > 0][:max_heatmaps]
            
            if top_layers:
                # Load correlation data
                correlations_dir = correlation_dir / "correlations"
                layer_data = load_correlation_data(correlations_dir, concept)
                
                # Create heatmaps directory
                heatmaps_dir = output_path.parent / "heatmaps"
                heatmaps_dir.mkdir(exist_ok=True)
                
                for layer_name, count in top_layers:
                    # Find matching layer in layer_data (handle sanitized names)
                    matching_layer = None
                    
                    # Try exact match first
                    if layer_name in layer_data:
                        matching_layer = layer_name
                    else:
                        # Try to match by sanitized name
                        sanitized_target = layer_name.replace(':', '_').replace('.', '_')
                        for stored_layer_name in layer_data.keys():
                            sanitized_stored = stored_layer_name.replace(':', '_').replace('.', '_')
                            # Check if they match (either direction)
                            if sanitized_target == sanitized_stored:
                                matching_layer = stored_layer_name
                                break
                            # Also try partial match (in case of slight differences)
                            elif sanitized_target in sanitized_stored or sanitized_stored in sanitized_target:
                                matching_layer = stored_layer_name
                                break
                    
                    if matching_layer and matching_layer in layer_data:
                        corr_matrix, _ = layer_data[matching_layer]
                        parsed = parse_layer_name(matching_layer)
                        display_name = parsed['full_display']
                        
                        safe_layer_name = matching_layer.replace(':', '_').replace('.', '_')
                        heatmap_path = heatmaps_dir / f"{concept}_{safe_layer_name}_heatmap.png"
                        
                        print(f"  Creating heatmap for {display_name} ({count} high correlation pairs)")
                        plot_correlation_heatmap(
                            corr_matrix,
                            display_name,
                            heatmap_path,
                            correlation_threshold
                        )
                    else:
                        print(f"  Warning: Could not find correlation data for {layer_name}")
            else:
                print("  No layers with high correlations found")
        else:
            print(f"  Warning: Correlation metadata not found at {correlation_metadata_file}")



def run_visualization(
    correlation_dir: pathlib.Path,
    output_dir: pathlib.Path,
    concept: str,
    correlation_threshold: float = 0.9,
    max_layers: int = 5,
    filtering_summary: pathlib.Path = None,
):
    """
    Run the visualization pipeline.
    
    Args:
        correlation_dir: Directory containing correlation results
        output_dir: Directory where to save visualizations
        concept: Concept name
        correlation_threshold: Correlation threshold to highlight
        max_layers: Maximum number of layer heatmaps to generate
        filtering_summary: Path to filtering summary JSON (optional)
    """
    # Validate inputs
    correlations_dir_path = correlation_dir / "correlations"
    if not correlations_dir_path.exists():
        print(f"Error: Correlations directory not found: {correlations_dir_path}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visualizations for concept: {concept}")
    
    # Load correlation data
    print(f"Loading correlation data from {correlations_dir_path}")
    layer_data = load_correlation_data(correlations_dir_path, concept)
    
    print(f"Found correlation data for {len(layer_data)} layers")
    
    # Plot correlation distribution across all layers
    print("Creating correlation distribution plot...")
    plot_correlation_distribution(
        layer_data,
        output_dir / f"{concept}_correlation_distribution.png",
        correlation_threshold
    )
    
    # Plot heatmaps for top layers (by number of experts)
    print(f"Creating correlation heatmaps (max {max_layers} layers)...")
    
    # Sort layers by number of experts
    layer_sizes = {name: matrix.shape[0] for name, (matrix, _) in layer_data.items()}
    sorted_layers = sorted(layer_sizes.items(), key=lambda x: x[1], reverse=True)
    
    heatmaps_dir = output_dir / "heatmaps"
    heatmaps_dir.mkdir(exist_ok=True)
    
    for i, (layer_name, size) in enumerate(sorted_layers[:max_layers]):
        print(f"  [{i+1}/{min(max_layers, len(sorted_layers))}] {layer_name} ({size} experts)")
        corr_matrix, _ = layer_data[layer_name]
        
        safe_layer_name = layer_name.replace(':', '_').replace('.', '_')
        output_path = heatmaps_dir / f"{concept}_{safe_layer_name}_heatmap.png"
        
        plot_correlation_heatmap(
            corr_matrix,
            layer_name,
            output_path,
            correlation_threshold
        )
    
    # Plot filtering summary if provided
    if filtering_summary and filtering_summary.exists():
        print("Creating filtering summary plot...")
        plot_filtering_summary(
            filtering_summary,
            output_dir / f"{concept}_filtering_summary.png",
            correlation_dir=correlation_dir,
            concept=concept,
            correlation_threshold=correlation_threshold,
            max_heatmaps=max_layers
        )
    
    print(f"\n{'='*60}")
    print("VISUALIZATIONS COMPLETE!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"  - Correlation distribution: {concept}_correlation_distribution.png")
    print(f"  - Layer heatmaps: heatmaps/")
    if filtering_summary:
        print(f"  - Filtering summary: {concept}_filtering_summary.png")


def main():
    parser = argparse.ArgumentParser(
        prog="visualize_neuron_correlations.py",
        description="Visualize neuron correlation matrices and redundancy patterns.",
    )
    
    parser.add_argument(
        "--correlation-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing correlation results",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory where to save visualizations",
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        help="Concept name",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold to highlight (default: 0.9)",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=5,
        help="Maximum number of layer heatmaps to generate (default: 5)",
    )
    parser.add_argument(
        "--filtering-summary",
        type=pathlib.Path,
        help="Path to filtering summary JSON (for summary plots)",
    )
    
    args = parser.parse_args()
    
    run_visualization(
        correlation_dir=args.correlation_dir,
        output_dir=args.output_dir,
        concept=args.concept,
        correlation_threshold=args.correlation_threshold,
        max_layers=args.max_layers,
        filtering_summary=args.filtering_summary,
    )


if __name__ == "__main__":
    main()

