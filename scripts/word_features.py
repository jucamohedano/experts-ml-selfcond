#! /usr/bin/env python3
"""
Extract word-level features from expert neuron activations.

This script implements Steps 1-2 from the Expert-Based RSA analysis:
- Step 1: Build unified expert pool E_l = union of experts across all words
- Step 2: For each word, extract feature vector f_l(w_i) = activations from E_l

Output: Feature matrices per layer, shape [num_words, |E_l|]
"""

import pathlib
import sys
import os
import logging
import pickle
import json
import hydra
from omegaconf import DictConfig

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add project root to path to import selfcond modules
sys.path.append(os.getcwd())
from selfcond.responses import read_responses_from_cached

log = logging.getLogger(__name__)


def build_unified_expert_pool(expertise_path: pathlib.Path, 
                               concepts: list, 
                               ap_threshold: float,
                               unique_experts_path: pathlib.Path = None) -> dict:
    """
    Build the unified expert pool E_l for each layer.
    
    Args:
        expertise_path: Path to standard expertise CSVs
        concepts: List of concept names
        ap_threshold: AP threshold for expert filtering
        unique_experts_path: Optional path to unique expert CSVs (filters by is_unique_expert=True)
    
    Returns:
        {layer_name: np.array of unique expert unit indices}
    """
    all_dfs = []
    
    for concept in concepts:
        if unique_experts_path is not None:
            # Load unique expert CSVs
            csv_path = unique_experts_path / concept / "unique_experts" / f"{concept}_expertise_unique.csv"
        else:
            # Load standard expertise CSVs
            csv_path = expertise_path / concept / "expertise" / "expertise.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
    
    # Concatenate into single DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Apply filters
    if unique_experts_path is not None:
        # Filter by is_unique_expert column AND AP threshold
        experts_df = combined_df[
            (combined_df['is_unique_expert'] == True) & 
            (combined_df['ap'] > ap_threshold)
        ]
    else:
        # Standard AP threshold filter only
        experts_df = combined_df[combined_df['ap'] > ap_threshold]
    
    # Group by layer, get unique units (union across all concepts)
    unified_pool = {}
    for layer_name, group in experts_df.groupby('layer'):
        # Clean layer name (remove :0 suffix if present)
        clean_layer = layer_name.split(':')[0]
        # Get unique unit indices (the union)
        unique_units = group['unit'].unique()
        unified_pool[clean_layer] = np.sort(unique_units)
    
    return unified_pool


def compute_rdm(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the RDM (Representational Dissimilarity Matrix) from a feature matrix.
    
    RDM(i,j) = 1 - Pearson correlation between word i and word j
    
    Args:
        feature_matrix: Shape [num_words, num_features]
        
    Returns:
        RDM matrix of shape [num_words, num_words]
    """
    return 1 - np.corrcoef(feature_matrix)


def plot_rdm(rdm_matrix: np.ndarray, 
             concepts: list, 
             output_path: pathlib.Path, 
             layer_name: str) -> None:
    """
    Plot and save the RDM as a heatmap with concept labels.
    
    Args:
        rdm_matrix: RDM of shape [num_words, num_words]
        concepts: List of concept names (sorted alphabetically)
        output_path: Path to save the plot
        layer_name: Layer name for title
    """
    # Create DataFrame for better labeling
    df_rdm = pd.DataFrame(rdm_matrix, index=concepts, columns=concepts)
    
    # Log statistics
    off_diag = rdm_matrix[~np.eye(rdm_matrix.shape[0], dtype=bool)]
    log.info(f"Layer {layer_name}: Mean dissim={off_diag.mean():.3f}, "
             f"Max dissim={off_diag.max():.3f}, Min dissim={off_diag.min():.3f}")
    
    # Plot heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(df_rdm, cmap="viridis", vmin=0, vmax=2,  # RDM range: 0 (identical) to 2 (anti-correlated)
                square=True, linewidths=0.0, linecolor='white',
                cbar_kws={'label': 'Dissimilarity (1 - Pearson r)'})
    plt.title(f"Representational Dissimilarity Matrix (RDM)\n{layer_name}", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Concepts", fontsize=11)
    plt.ylabel("Concepts", fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save the RDM as CSV for easier inspection
    csv_path = output_path.with_suffix('.csv')
    df_rdm.to_csv(csv_path)
    log.info(f"Saved RDM heatmap: {output_path}")
    log.info(f"Saved RDM CSV: {csv_path}")

def extract_word_features(activations_path: pathlib.Path, 
                          expertise_path: pathlib.Path, 
                          ap_threshold: float, 
                          layer: str,
                          use_all_neurons: bool = False,
                          unique_experts_path: pathlib.Path = None) -> None:
    """
    Extract word-level features from the expertise data.
    
    Args:
        activations_path: Path to concept responses
        expertise_path: Path to expertise CSVs
        ap_threshold: AP threshold for expert filtering
        layer: Layer filter ("all" or specific layer name)
        use_all_neurons: If True, use ALL neurons (no expert filtering) for Full RDM
        unique_experts_path: Path to unique expert CSVs (for unique RDM computation)
    """

    log.info(f"Starting Word Features Extraction")
    log.info(f"Output directory: {os.getcwd()}")
    
    if not activations_path.is_absolute():
        activations_path = pathlib.Path(hydra.utils.get_original_cwd()) / activations_path

    if expertise_path is not None and not expertise_path.is_absolute():
        expertise_path = pathlib.Path(hydra.utils.get_original_cwd()) / expertise_path
        
    log.info(f"AP Threshold: {ap_threshold}")
    log.info(f"Layer: {layer}")
    log.info(f"Use all neurons: {use_all_neurons}")

    log.info(f"Reading activations from: {activations_path}")
    
    if not activations_path.exists():
        log.error(f"Activations path does not exist: {activations_path}")
        return

    concepts = [d.name for d in activations_path.iterdir() if d.is_dir()]
    concepts = sorted(concepts)
    log.info(f"Found {len(concepts)} concepts")

    # Build unified expert pool (or None if using all neurons)
    if use_all_neurons:
        log.info("Using ALL neurons (no expert filtering) for Full RDM computation")
        unified_expert_pool = None
    else:
        if unique_experts_path:
            log.info(f"Using UNIQUE experts from: {unique_experts_path}")
        unified_expert_pool = build_unified_expert_pool(
            expertise_path, concepts, ap_threshold, unique_experts_path=unique_experts_path
        )
        log.info(f"Built unified expert pool for {len(unified_expert_pool)} layers")
        for layer_name, units in unified_expert_pool.items():
            log.info(f"  {layer_name}: {len(units)} experts")

    # Pass 2: Extract word features for each concept
    # Data structure: {layer: {concept: feature_vector}}
    word_features = defaultdict(dict)
    
    for concept in tqdm(concepts, desc="Extracting word features"):
        concept_responses_path = activations_path / concept / "responses"
        if not concept_responses_path.exists():
            log.warning(f"Responses not found for {concept}, skipping")
            continue
        
        try:
            # Load activations: {layer_name: array[units, sentences]}
            data, labels, _ = read_responses_from_cached(concept_responses_path, concept)
            labels = np.array(labels)
            
            # Get positive sample indices (sentences about this word)
            pos_indices = np.where(labels == 1)[0]
            if len(pos_indices) == 0:
                log.warning(f"No positive samples for {concept}, skipping")
                continue
            
            for layer_name, activations in data.items():
                # Skip metadata keys
                if layer_name in ["labels", "input_ids", "attention_mask"]:
                    continue
                
                # Clean layer name (remove :0 suffix if present)
                clean_layer_name = layer_name.split(":")[0]
                
                # Filter by user-requested layer if specified
                if layer != "all" and clean_layer_name != layer:
                    continue
                
                # activations shape: [units, sentences]
                if use_all_neurons or unified_expert_pool is None:
                    # Use ALL neurons (for Full RDM)
                    selected_activations = activations
                else:
                    # Use only expert neurons
                    if clean_layer_name not in unified_expert_pool:
                        log.debug(f"No experts for {clean_layer_name}, skipping")
                        continue
                    expert_indices = unified_expert_pool[clean_layer_name]
                    selected_activations = activations[expert_indices, :]
                
                # Select only positive samples (columns)
                activations_pos = selected_activations[:, pos_indices]
                
                # Aggregate - mean across positive samples
                # Result: feature vector of shape [num_neurons] or [|E_l|]
                feature_vector = np.mean(activations_pos, axis=1)
                
                # Store the feature vector
                word_features[clean_layer_name][concept] = feature_vector
                
        except Exception as e:
            log.error(f"Error processing {concept}: {e}")
            continue

    # Save results
    output_dir = pathlib.Path(os.getcwd())
    
    # Create rdms directory
    rdms_dir = output_dir / "rdms"
    rdms_dir.mkdir(exist_ok=True)

    # Compute and save RDMs for each layer
    log.info(f"Computing and saving RDMs for each layer")
    rdm_matrices = {}
    for layer_name, concept_vectors in tqdm(word_features.items(), desc="Computing RDMs"):
        sorted_concepts = sorted(concept_vectors.keys())
        feature_matrix = np.stack([concept_vectors[c] for c in sorted_concepts])
        rdm = compute_rdm(feature_matrix)
        rdm_matrices[layer_name] = (rdm, sorted_concepts)
        
        # Save RDM as numpy array
        rdm_file = rdms_dir / f"{layer_name.replace('.', '_')}_rdm.npy"
        np.save(rdm_file, rdm)
        
        # Plot and save RDM heatmap
        plot_path = rdms_dir / f"{layer_name.replace('.', '_')}_rdm.png"
        plot_rdm(rdm, sorted_concepts, plot_path, layer_name)
    
    # Log summary
    log.info(f"\nExtraction complete:")
    for layer_name in sorted(word_features.keys()):
        num_words = len(word_features[layer_name])
        if num_words > 0:
            vec_dim = len(next(iter(word_features[layer_name].values())))
            log.info(f"  {layer_name}: {num_words} words, {vec_dim}-dim features")
    
    
    # Save as pickle: {layer: {concept: vector}}
    output_file = output_dir / "word_features.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(dict(word_features), f)
    log.info(f"Saved word features to {output_file}")
    
    # Also save as feature matrices per layer (for easier RDM computation)
    # Shape: [num_words, |E_l|]
    matrices_dir = output_dir / "feature_matrices"
    matrices_dir.mkdir(exist_ok=True)
    
    for layer_name, concept_vectors in word_features.items():
        # Sort concepts alphabetically for consistent ordering
        sorted_concepts = sorted(concept_vectors.keys())
        
        # Stack into matrix [num_words, |E_l|]
        feature_matrix = np.stack([concept_vectors[c] for c in sorted_concepts])
        
        # Save matrix
        matrix_file = matrices_dir / f"{layer_name.replace('.', '_')}_features.npy"
        np.save(matrix_file, feature_matrix)
        
        # Save concept order
        concepts_file = matrices_dir / f"{layer_name.replace('.', '_')}_concepts.txt"
        with open(concepts_file, 'w') as f:
            f.write('\n'.join(sorted_concepts))
        
        log.info(f"  Saved {layer_name}: matrix shape {feature_matrix.shape}")
    
    # Save metadata
    metadata = {
        'ap_threshold': ap_threshold,
        'use_all_neurons': use_all_neurons,
        'layer_filter': layer,
        'num_concepts': len(concepts),
        'concepts': concepts,
        'unified_expert_pool': {k: len(v) for k, v in unified_expert_pool.items()} if unified_expert_pool else "all_neurons",
    }
    metadata_file = output_dir / "word_features_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata to {metadata_file}")
    
    return word_features


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    activations_path = pathlib.Path(cfg.task.activations_path)
    if not activations_path.is_absolute():
        activations_path = pathlib.Path(hydra.utils.get_original_cwd()) / activations_path

    expertise_path = pathlib.Path(cfg.task.expertise_path)
    if not expertise_path.is_absolute():
        expertise_path = pathlib.Path(hydra.utils.get_original_cwd()) / expertise_path
    
    use_all_neurons = cfg.task.get("use_all_neurons", False)
    
    # Optional unique experts path
    unique_experts_path = cfg.task.get("unique_experts_path", None)
    if unique_experts_path:
        unique_experts_path = pathlib.Path(unique_experts_path)
        if not unique_experts_path.is_absolute():
            unique_experts_path = pathlib.Path(hydra.utils.get_original_cwd()) / unique_experts_path
        
    extract_word_features(
        activations_path=activations_path,
        expertise_path=expertise_path,
        ap_threshold=cfg.task.ap_threshold,
        layer=cfg.task.layer,
        use_all_neurons=use_all_neurons,
        unique_experts_path=unique_experts_path,
    )

if __name__ == "__main__":
    main()