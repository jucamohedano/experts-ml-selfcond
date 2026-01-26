#!/usr/bin/env python3
"""
Patch script to fix killed vector pickle files by copying missing concepts from global.

This fixes the bug where concepts with no expert neurons were excluded from killed vectors,
when they should equal the global vectors (nothing to kill = same as original).

Usage:
    python scripts/patch_killed_vectors.py /path/to/subspace_gaze/results/directory
    python scripts/patch_killed_vectors.py /path/to/results --regenerate-plots
"""

import argparse
import pickle
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def regenerate_plots(vectors_dict, output_dir, layer_name, suffix):
    """Regenerate heatmap and UMAP plots for a vector dictionary."""
    if not vectors_dict:
        return
    
    concepts = sorted(list(vectors_dict.keys()))
    if len(concepts) < 2:
        return
    
    # Stack vectors into matrix
    vectors_matrix = np.array([vectors_dict[c]['vector'] for c in concepts])
    
    # Compute cosine similarity
    sim_matrix = cosine_similarity(vectors_matrix)
    
    # Log stats
    mean_sim = (sim_matrix.sum() - len(concepts)) / (len(concepts) * (len(concepts) - 1))
    max_sim = sim_matrix[~np.eye(len(concepts), dtype=bool)].max() if len(concepts) > 1 else 1.0
    min_sim = sim_matrix[~np.eye(len(concepts), dtype=bool)].min() if len(concepts) > 1 else 1.0
    print(f"    Stats: Mean sim={mean_sim:.3f}, Max={max_sim:.3f}, Min={min_sim:.3f}")
    
    # Save similarity matrix CSV
    sim_df = pd.DataFrame(sim_matrix, index=concepts, columns=concepts)
    sim_df.to_csv(output_dir / f"similarity_matrix_{layer_name}{suffix}.csv")
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(sim_df, cmap='viridis', center=0, vmin=-1, vmax=1,
                xticklabels=True, yticklabels=True)
    plt.title(f"Cosine Similarity Matrix - {layer_name}{suffix}")
    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_{layer_name}{suffix}.png", dpi=150)
    plt.close()
    
    # UMAP plot
    try:
        from umap import UMAP
        if len(concepts) >= 5:
            n_neighbors = min(15, len(concepts) - 1)
            umap = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
            embeddings = umap.fit_transform(vectors_matrix)
            
            # Import concept groups from subspace_gaze
            from scripts.subspace_gaze import CONCEPT_GROUPS, GROUP_COLORS, get_concept_group
            
            df = pd.DataFrame({
                'x': embeddings[:, 0],
                'y': embeddings[:, 1],
                'concept': concepts,
                'group': [get_concept_group(c) for c in concepts],
            })
            
            df.to_csv(output_dir / f"umap_embeddings_{layer_name}{suffix}.csv", index=False)
            
            fig, ax = plt.subplots(figsize=(14, 10))
            for group in sorted(df['group'].unique()):
                mask = df['group'] == group
                color = GROUP_COLORS.get(group, '#cccccc')
                ax.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'], 
                          c=color, label=group, s=100, alpha=0.7, edgecolors='white', linewidth=0.5)
            
            for _, row in df.iterrows():
                ax.annotate(row['concept'], (row['x'], row['y']), 
                           fontsize=8, alpha=0.85, ha='center', va='bottom',
                           xytext=(0, 5), textcoords='offset points')
            
            ax.set_title(f"UMAP of Concept Vectors by Semantic Group\n{layer_name} {suffix}", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("UMAP 1", fontsize=11)
            ax.set_ylabel("UMAP 2", fontsize=11)
            ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
            ax.set_facecolor('#f5f5f5')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"umap_{layer_name}{suffix}.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    UMAP plot saved")
    except ImportError:
        print("    UMAP not available - skipping UMAP plot")
    except Exception as e:
        print(f"    UMAP failed: {e}")


def patch_killed_vectors(results_dir: str, dry_run: bool = False, regenerate: bool = False):
    """
    For each layer, find concepts in global that are missing from killed,
    and copy them over.
    """
    results_path = pathlib.Path(results_dir)
    
    # Find all global vector files
    global_files = list(results_path.glob("vectors_*_global.pkl"))
    
    if not global_files:
        print(f"No global vector files found in {results_dir}")
        return
    
    print(f"Found {len(global_files)} global vector files")
    patched_layers = []
    
    for global_file in sorted(global_files):
        # Derive killed filename
        layer_name = global_file.stem.replace("vectors_", "").replace("_global", "")
        killed_file = results_path / f"vectors_{layer_name}_killed.pkl"
        
        if not killed_file.exists():
            print(f"  [SKIP] No killed file for {layer_name}")
            continue
        
        # Load both files
        with open(global_file, "rb") as f:
            global_vectors = pickle.load(f)
        
        with open(killed_file, "rb") as f:
            killed_vectors = pickle.load(f)
        
        # Find missing concepts
        global_concepts = set(global_vectors.keys())
        killed_concepts = set(killed_vectors.keys())
        missing_concepts = global_concepts - killed_concepts
        
        if not missing_concepts:
            print(f"  [OK] {layer_name}: All {len(killed_concepts)} concepts present")
            continue
        
        print(f"  [PATCH] {layer_name}: Adding {len(missing_concepts)} missing concepts (was {len(killed_concepts)}, now {len(global_concepts)})")
        
        if dry_run:
            continue
        
        # Copy missing concepts from global to killed
        for concept in missing_concepts:
            killed_vectors[concept] = global_vectors[concept]
        
        # Save patched killed file
        with open(killed_file, "wb") as f:
            pickle.dump(killed_vectors, f)
        
        patched_layers.append((layer_name, killed_vectors))
        print(f"          Saved patched file: {killed_file.name}")
    
    # Regenerate plots if requested
    if regenerate and not dry_run and patched_layers:
        print(f"\nRegenerating plots for {len(patched_layers)} patched layers...")
        for layer_name, killed_vectors in patched_layers:
            print(f"  Regenerating {layer_name}_killed plots...")
            regenerate_plots(killed_vectors, results_path, layer_name, "_killed")
    
    print("\nDone!")
    if dry_run:
        print("(Dry run - no files were modified)")
    elif regenerate and patched_layers:
        print(f"Patched {len(patched_layers)} layers and regenerated their plots.")


def main():
    parser = argparse.ArgumentParser(description="Patch killed vector files")
    parser.add_argument("results_dir", help="Path to subspace_gaze results directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be patched without modifying files")
    parser.add_argument("--regenerate-plots", action="store_true", help="Regenerate heatmap and UMAP plots for patched layers")
    args = parser.parse_args()
    
    patch_killed_vectors(args.results_dir, args.dry_run, args.regenerate_plots)


if __name__ == "__main__":
    main()

