import hydra
from omegaconf import DictConfig
import pathlib
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import sys
import os
from collections import defaultdict

# Add project root to path to import selfcond modules
sys.path.append(os.getcwd())
from selfcond.responses import read_responses_from_cached

log = logging.getLogger(__name__)

# =============================================================================
# Concept Groupings based on WordNet Hypernyms
# Categories: artifact.n.01, living_thing.n.01, body_part.n.01
# =============================================================================

CONCEPT_GROUPS = {
    # Artifacts - Tools
    'chisel': 'tool', 'hammer': 'tool', 'key': 'tool', 'knife': 'tool',
    'pliers': 'tool', 'saw': 'tool', 'screwdriver': 'tool', 'spoon': 'tool',
    
    # Artifacts - Vehicles
    'airplane': 'vehicle', 'bicycle': 'vehicle', 'car': 'vehicle',
    'train': 'vehicle', 'truck': 'vehicle',
    
    # Artifacts - Furniture
    'bed': 'furniture', 'chair': 'furniture', 'closet': 'furniture',
    'desk': 'furniture', 'dresser': 'furniture', 'refrigerator': 'furniture',
    'table': 'furniture',
    
    # Artifacts - Buildings/Structures
    'apartment': 'building', 'arch': 'building', 'barn': 'building',
    'chimney': 'building', 'church': 'building', 'door': 'building',
    'house': 'building', 'igloo': 'building', 'window': 'building',
    
    # Artifacts - Clothing
    'coat': 'clothing', 'dress': 'clothing', 'pants': 'clothing',
    'shirt': 'clothing', 'skirt': 'clothing',
    
    # Artifacts - Containers/Objects
    'bell': 'object', 'bottle': 'object', 'cup': 'object', 'glass': 'object',
    'telephone': 'object', 'watch': 'object',
    
    # Living Things - Animals (Mammals)
    'bear': 'animal', 'cat': 'animal', 'cow': 'animal',
    'dog': 'animal', 'horse': 'animal',
    
    # Living Things - Insects
    'ant': 'insect', 'bee': 'insect', 'beetle': 'insect',
    'butterfly': 'insect', 'fly': 'insect',
    
    # Living Things - Plants/Vegetables
    'carrot': 'plant', 'celery': 'plant', 'corn': 'plant',
    'lettuce': 'plant', 'tomato': 'plant',
    
    # Body Parts
    'arm': 'body_part', 'eye': 'body_part', 'foot': 'body_part',
    'hand': 'body_part', 'leg': 'body_part',
}

# Color palette for visualization
GROUP_COLORS = {
    'tool': '#e41a1c',       # Red
    'vehicle': '#377eb8',    # Blue
    'furniture': '#4daf4a',  # Green
    'building': '#984ea3',   # Purple
    'clothing': '#ff7f00',   # Orange
    'object': '#ffff33',     # Yellow
    'animal': '#a65628',     # Brown
    'insect': '#f781bf',     # Pink
    'plant': '#999999',      # Gray
    'body_part': '#66c2a5',  # Teal
    'other': '#cccccc',      # Light gray for unknown
}

def get_concept_group(concept: str) -> str:
    """Returns the semantic group for a concept."""
    return CONCEPT_GROUPS.get(concept.lower(), 'other')

def get_diffmean_vector(activations, labels):
    """
    Computes the concept vector using Difference-in-Means.
    v = normalize(mu_pos - mu_neg)
    """
    # Transpose to [samples, units]
    X = activations.T
    
    # Separate positive and negative samples
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return None
        
    mu_pos = np.mean(X[pos_indices], axis=0)
    mu_neg = np.mean(X[neg_indices], axis=0)
    diff_vector = mu_pos - mu_neg
    
    # Normalize
    norm = np.linalg.norm(diff_vector)
    if norm > 0:
        vector = diff_vector / norm
        # Compute max activation on positive samples
        # X[pos_indices] shape: [num_pos_samples, units]
        # vector shape: [units]
        projections = X[pos_indices] @ vector
        max_act = np.max(projections)
        return vector, max_act
    else:
        return np.zeros_like(diff_vector), 0.0

def get_pca_vector(activations, labels):
    """
    Computes the concept vector using PCA on positive samples only.
    1. Select positive samples H+.
    2. Center them: h - mu_pos.
    3. Compute first principal component.
    """
    # Transpose to [samples, units]
    X = activations.T
    
    # Select positive samples
    pos_indices = np.where(labels == 1)[0]
    if len(pos_indices) < 2: # Need at least 2 samples for PCA
        return None
        
    X_pos = X[pos_indices]
    
    # Center the data
    mu_pos = np.mean(X_pos, axis=0)
    X_pos_centered = X_pos - mu_pos
    
    # PCA
    try:
        pca = PCA(n_components=1)
        pca.fit(X_pos_centered)
        vector = pca.components_[0] # First principal component
        
        # Compute max activation on positive samples
        projections = X_pos @ vector
        max_act = np.max(projections)
        
        return vector, max_act
    except Exception as e:
        log.error(f"PCA failed: {e}")
        return None, 0.0


def get_lat_vector(activations, labels, n_pairs=None, seed=42):
    """
    Computes the concept vector using LAT (Linear Artificial Tomography).
    Based on Zou et al., 2023: https://arxiv.org/abs/2310.01405
    
    LAT isolates concept directions by analyzing difference vectors between
    positive and negative samples. This cancels out shared features (syntax,
    common words) and isolates the concept-specific direction.
    
    Steps:
    1. Create pairs of (positive, negative) samples
    2. Compute difference vectors: delta = h_pos - h_neg
    3. Normalize each difference vector
    4. Apply PCA to normalized differences
    5. Return first principal component
    
    Args:
        activations: Array of shape [units, samples]
        labels: Binary labels (1 for positive, 0 for negative)
        n_pairs: Number of random pairs to generate (default: min(n_pos * n_neg, 10000))
        seed: Random seed for reproducibility
    
    Returns:
        (vector, max_act) tuple or (None, 0.0) on failure
    """
    np.random.seed(seed)
    
    # Transpose to [samples, units]
    X = activations.T
    
    # Get positive and negative indices
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    if len(pos_indices) < 2 or len(neg_indices) < 2:
        log.warning("LAT requires at least 2 positive and 2 negative samples")
        return None, 0.0
    
    # Determine number of pairs
    n_pos, n_neg = len(pos_indices), len(neg_indices)
    max_pairs = n_pos * n_neg
    n_pairs = n_pairs or min(max_pairs, 10000)  # Cap for memory efficiency
    n_pairs = min(n_pairs, max_pairs)
    
    # Sample random pairs (with replacement if needed)
    pos_sample = np.random.choice(pos_indices, size=n_pairs, replace=(n_pairs > n_pos))
    neg_sample = np.random.choice(neg_indices, size=n_pairs, replace=(n_pairs > n_neg))
    
    # Compute difference vectors: h_pos - h_neg
    differences = X[pos_sample] - X[neg_sample]  # [n_pairs, units]
    
    # Normalize each difference vector (avoiding division by zero)
    norms = np.linalg.norm(differences, axis=1, keepdims=True)
    # Replace zero norms with 1 to avoid division by zero, result will be zero vector anyway
    norms = np.where(norms == 0, 1, norms)
    differences = differences / norms
    
    # Filter out any all-zero difference vectors
    valid_mask = np.any(differences != 0, axis=1)
    differences = differences[valid_mask]
    
    if len(differences) < 2:
        log.warning("LAT: Not enough valid difference vectors after normalization")
        return None, 0.0
    
    # Apply PCA to normalized differences
    try:
        pca = PCA(n_components=1)
        pca.fit(differences)
        vector = pca.components_[0]  # First principal component
        
        # Log explained variance for debugging
        variance = pca.explained_variance_ratio_[0]
        log.debug(f"LAT explains {variance:.2%} of the variance in differences")
        
        # Compute max activation on positive samples
        projections = X[pos_indices] @ vector
        max_act = np.max(projections)
        
        return vector, max_act
    except Exception as e:
        log.error(f"LAT PCA failed: {e}")
        return None, 0.0


def get_concept_vector(activations, labels, method="diff_mean"):
    """
    Dispatcher for concept vector computation methods.
    
    Args:
        activations: Array of shape [units, samples]
        labels: Binary labels (1 for positive, 0 for negative)
        method: One of "diff_mean", "pca", or "lat"
    
    Returns:
        (vector, max_act) tuple
    """
    if method == "diff_mean":
        return get_diffmean_vector(activations, labels)
    elif method == "pca":
        return get_pca_vector(activations, labels)
    elif method == "lat":
        return get_lat_vector(activations, labels)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: diff_mean, pca, lat")

def load_expertise(expertise_path, concept, ap_threshold):
    """
    Loads expertise for a concept and returns a dict of {layer: expert_units_indices}.
    """
    exp_file = expertise_path / concept / "expertise" / "expertise.csv"
    if not exp_file.exists():
        log.warning(f"Expertise file not found: {exp_file}")
        return {}
        
    try:
        df = pd.read_csv(exp_file)
        # Filter by AP threshold
        experts = df[df["ap"] > ap_threshold]
        
        # Group by layer
        layer_experts = defaultdict(list)
        for _, row in experts.iterrows():
            # Strip suffix from layer name if present
            layer_name = row["layer"].split(":")[0]
            layer_experts[layer_name].append(row["unit"])
            
        return {k: np.array(v) for k, v in layer_experts.items()}
    except Exception as e:
        log.error(f"Error loading expertise for {concept}: {e}")
        return {}

def plot_umap_concepts(concept_vectors, output_dir, layer_name, suffix=""):
    """
    Creates a UMAP visualization of concept vectors, colored by semantic group.
    
    UMAP (Uniform Manifold Approximation and Projection) reduces high-dimensional
    concept vectors to 2D for visualization while preserving local structure.
    This reveals clusters of semantically related concepts.
    
    Args:
        concept_vectors: Dict of {concept: {'vector': ndarray, 'max_act': float}}
        output_dir: Path to save the plot
        layer_name: Layer name for title
        suffix: Suffix for filename (e.g., "_global", "_masked")
    """
    try:
        from umap import UMAP
    except ImportError:
        log.warning("UMAP not installed. Skipping UMAP visualization. Install with: pip install umap-learn")
        return
    
    concepts_list = sorted(list(concept_vectors.keys()))
    if len(concepts_list) < 5:
        log.warning(f"Not enough concepts for UMAP ({len(concepts_list)} < 5)")
        return
    
    # Stack vectors into matrix
    vectors_matrix = np.array([concept_vectors[c]['vector'] for c in concepts_list])
    
    # Handle non-finite values
    if not np.all(np.isfinite(vectors_matrix)):
        log.warning(f"Non-finite values in vectors for UMAP - {layer_name}")
        return
    
    # Apply UMAP
    try:
        n_neighbors = min(15, len(concepts_list) - 1)
        umap = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
        embeddings = umap.fit_transform(vectors_matrix)
    except Exception as e:
        log.error(f"UMAP failed: {e}")
        return
    
    # Create DataFrame with groupings
    df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'concept': concepts_list,
        'group': [get_concept_group(c) for c in concepts_list],
    })
    
    # Save embeddings
    df.to_csv(output_dir / f"umap_embeddings_{layer_name}{suffix}.csv", index=False)
    
    # Create single figure colored by semantic group
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot points colored by semantic group
    for group in sorted(df['group'].unique()):
        mask = df['group'] == group
        color = GROUP_COLORS.get(group, '#cccccc')
        ax.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'], 
                  c=color, label=group, s=100, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Add labels for each point
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
    
    log.info(f"UMAP plot saved: umap_{layer_name}{suffix}.png")


def save_and_plot_layer(layer_name, concept_vectors, output_dir, suffix=""):
    """
    Saves concept vectors and creates visualizations (heatmap + UMAP).
    
    Args:
        layer_name: Name of the layer
        concept_vectors: Dict of {concept: {'vector': ndarray, 'max_act': float}}
        output_dir: Path to save outputs
        suffix: Suffix for filenames (e.g., "_global", "_masked", "_killed")
    """
    if not concept_vectors:
        log.warning(f"No vectors extracted for layer {layer_name} {suffix}")
        return

    # Save vectors
    save_path = output_dir / f"vectors_{layer_name}{suffix}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(concept_vectors, f)
        
    # Compute Similarity Matrix
    concepts_list = sorted(list(concept_vectors.keys()))
    if len(concepts_list) < 2:
        return

    # Extract vectors for similarity computation
    # concept_vectors[c] is now {'vector': ..., 'max_act': ...}
    vectors_matrix = np.array([concept_vectors[c]['vector'] for c in concepts_list])
    
    # Handle NaN/Inf in vectors if any (shouldn't happen with normalization but safe check)
    if not np.all(np.isfinite(vectors_matrix)):
        log.warning(f"Non-finite values in vectors for {layer_name}")
        return

    sim_matrix = cosine_similarity(vectors_matrix)
    
    # Log statistics
    off_diag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
    log.info(f"Layer {layer_name}{suffix}: Mean sim={off_diag.mean():.3f}, "
             f"Max sim={off_diag.max():.3f}, Min sim={off_diag.min():.3f}")
    
    # Save Matrix
    df_sim = pd.DataFrame(sim_matrix, index=concepts_list, columns=concepts_list)
    df_sim.to_csv(output_dir / f"similarity_matrix_{layer_name}{suffix}.csv")
    
    # Plot Heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(df_sim, cmap="viridis", vmin=-1, vmax=1)
    plt.title(f"Cosine Similarity Matrix - {layer_name} {suffix}")
    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_{layer_name}{suffix}.png")
    plt.close()
    
    # Plot UMAP visualization
    plot_umap_concepts(concept_vectors, output_dir, layer_name, suffix)

def run_subspace_gaze(activations_path: pathlib.Path,
                      expertise_path: pathlib.Path,
                      ap_threshold: float,
                      layer: str = "all",
                      method: str = "diff_mean"):
    log.info(f"Starting Subspace Gaze Analysis")
    log.info(f"Output directory: {os.getcwd()}")
    
    if not activations_path.is_absolute():
        activations_path = pathlib.Path(hydra.utils.get_original_cwd()) / activations_path

    if not expertise_path.is_absolute():
        expertise_path = pathlib.Path(hydra.utils.get_original_cwd()) / expertise_path
        
    log.info(f"AP Threshold: {ap_threshold}")
    log.info(f"Method: {method}")

    log.info(f"Reading activations from: {activations_path}")
    
    if not activations_path.exists():
        log.error(f"Activations path does not exist: {activations_path}")
        return

    concepts = [d.name for d in activations_path.iterdir() if d.is_dir()]
    concepts = sorted(concepts)
    log.info(f"Found {len(concepts)} concepts")
    
    # Data structure to hold vectors: layer -> {concept: vector}
    global_vectors = defaultdict(dict)
    masked_vectors = defaultdict(dict)
    killed_vectors = defaultdict(dict)
    
    for concept in tqdm(concepts, desc="Processing concepts"):
        concept_path = activations_path / concept / "responses"
        if not concept_path.exists():
            log.warning(f"Responses not found for {concept}")
            continue
            
        # Load expertise for this concept
        expert_units_map = load_expertise(expertise_path, concept, ap_threshold)
        
        try:
            data, labels, _ = read_responses_from_cached(concept_path, concept)
            
            for layer_name, activations in data.items():
                if layer_name in ["labels", "input_ids", "attention_mask"]:
                    continue
                    
                # Handle potential suffixes in layer names (e.g. :0)
                clean_layer_name = layer_name.split(":")[0]
                
                # If user requested specific layer, skip others
                if layer != "all" and clean_layer_name != layer:
                    continue
                
                # Use clean layer name for storage
                storage_layer_name = clean_layer_name
                
                # 1. Global Vector (Unmasked)
                vec_global, max_act_global = get_concept_vector(activations, labels, method=method)
                if vec_global is not None:
                    global_vectors[storage_layer_name][concept] = {
                        'vector': vec_global,
                        'max_act': max_act_global
                    }
                
                # 2. Masked & Killed Vectors
                # Get experts for this concept in this layer (if any)
                experts = expert_units_map.get(storage_layer_name, np.array([]))
                
                if len(experts) > 0:
                    # Create mask
                    mask = np.zeros(activations.shape[0])
                    mask[experts] = 1
                    
                    # Apply mask (element-wise multiplication along units dimension)
                    # activations shape: [units, samples]
                    
                    # Masked (Keep Experts)
                    masked_activations = activations * mask[:, np.newaxis]
                    vec_masked, max_act_masked = get_concept_vector(masked_activations, labels, method=method)
                    if vec_masked is not None:
                        masked_vectors[storage_layer_name][concept] = {
                            'vector': vec_masked,
                            'max_act': max_act_masked
                        }
                        
                    # Killed (Remove Experts)
                    inverse_mask = 1 - mask
                    killed_activations = activations * inverse_mask[:, np.newaxis]
                    vec_killed, max_act_killed = get_concept_vector(killed_activations, labels, method=method)
                    if vec_killed is not None:
                        killed_vectors[storage_layer_name][concept] = {
                            'vector': vec_killed,
                            'max_act': max_act_killed
                        }
                else:
                    # No experts in this layer for this concept
                    # Masked: undefined (all zeros) - skip
                    # Killed: equals Global (nothing to kill)
                    if vec_global is not None:
                        killed_vectors[storage_layer_name][concept] = {
                            'vector': vec_global,
                            'max_act': max_act_global
                        }
                
        except Exception as e:
            log.error(f"Error processing {concept}: {e}")
            continue
            
    # Save and plot results
    output_dir = pathlib.Path(os.getcwd())
    
    log.info(f"Saving Global Results for {len(global_vectors)} layers")
    for layer_name, vectors in tqdm(global_vectors.items(), desc="Saving Global"):
        save_and_plot_layer(layer_name, vectors, output_dir, suffix="_global")
        
    log.info(f"Saving Masked Results for {len(masked_vectors)} layers")
    for layer_name, vectors in tqdm(masked_vectors.items(), desc="Saving Masked"):
        save_and_plot_layer(layer_name, vectors, output_dir, suffix="_masked")
        
    log.info(f"Saving Killed Results for {len(killed_vectors)} layers")
    for layer_name, vectors in tqdm(killed_vectors.items(), desc="Saving Killed"):
        save_and_plot_layer(layer_name, vectors, output_dir, suffix="_killed")

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    activations_path = pathlib.Path(cfg.task.activations_path)
    if not activations_path.is_absolute():
        activations_path = pathlib.Path(hydra.utils.get_original_cwd()) / activations_path

    expertise_path = pathlib.Path(cfg.task.expertise_path)
    if not expertise_path.is_absolute():
        expertise_path = pathlib.Path(hydra.utils.get_original_cwd()) / expertise_path
        
    run_subspace_gaze(
        activations_path=activations_path,
        expertise_path=expertise_path,
        ap_threshold=cfg.task.ap_threshold,
        layer=cfg.task.layer,
        method=cfg.task.method,
    )

if __name__ == "__main__":
    main()
