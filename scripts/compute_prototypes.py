import argparse
import logging
import pathlib
from typing import Dict, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from sklearn.preprocessing import StandardScaler
from pdc_dp_means import DPMeans
from sklearn.metrics.pairwise import cosine_similarity

from selfcond.brain_data import load_brain_data

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Step 1-2. Identify expert voxels and return the index of the expert voxels, their winning word and activation profile
def identify_expert_voxels(
    word_activations: np.ndarray,
    word_labels: List[str],
    x_threshold_parameter: float = 1.0
) -> List[Tuple[int, str, float, np.ndarray]]:
    """
    Identify expert voxels that are most responsive to each noun.

    Args:
        word_activations: Array of shape [60, V] with averaged responses per word.
        word_labels: List of 60 word strings in alphabetical order.
        x_threshold_parameter: Parameter for setting the threshold for identifying expert voxels based on mean activity.

    Returns:
        expert_voxels: List of tuples (voxel_index, winning_word, gap, activation_profile) for voxels that are identified as experts.
    """
    # Transpose to get shape (V, 60) for easier and clearer processing
    voxel_activations = word_activations.T  # (V, 60)
    # Mean activity per voxel across all words
    voxel_mean_activity = np.mean(voxel_activations, axis=1)
    # In case activations are centered (negative values), use absolute mean so x acts as a positive scaling factor
    voxel_mean_activity_abs = np.abs(voxel_mean_activity)
    # Sort the activations for each voxel to find the top 2 words
    sorted_indices = np.argsort(voxel_activations, axis=1)[:, ::-1]  # (V, 60)
    
    expert_voxels = []
    
    V = voxel_activations.shape[0] # Number of voxels
    for v in range(V):
        # Index of the top 2 words for this voxel
        top1_idx = sorted_indices[v, 0]
        top2_idx = sorted_indices[v, 1]
        # Activation values for the top 2 words for this voxel
        a1 = voxel_activations[v, top1_idx]
        a2 = voxel_activations[v, top2_idx]
        gap = a1 - a2
        threshold = x_threshold_parameter * voxel_mean_activity_abs[v]
        # Save the voxel if the gap is above the threshold times the mean activity of this voxel
        if gap > threshold:
            winning_word = word_labels[top1_idx]
            activation_profile = voxel_activations[v, :]
            expert_voxels.append((v, winning_word, gap, activation_profile))
    return expert_voxels

# Step 3.1 Group expert voxels by their winning word
def group_expert_voxels_by_word(expert_voxels: List[Tuple[int, str, float, np.ndarray]]) -> Dict[str, List[Tuple[int, np.ndarray]]]:
    """
    Group expert voxels by their winning word.

    Args:
        expert_voxels: List of tuples (voxel_index, winning_word, gap, activation_profile) for expert voxels.

    Returns:
        grouped_experts: Dictionary mapping each winning word to a list of tuples (voxel_index, activation_profile) for the expert voxels that are most responsive to that word.
    """
    grouped_experts = {}
    for voxel_index, winning_word, _, activation_profile in expert_voxels:
        if winning_word not in grouped_experts:
            grouped_experts[winning_word] = []
        grouped_experts[winning_word].append((voxel_index, activation_profile))
    return grouped_experts

def tune_delta(profiles, delta_range=[5, 10, 15, 20, 30, 40, 50, 75, 100], refine=True, singleton_penalty=1) -> float:
    """
    Find a good delta for DPMeans using Calinski-Harabasz index over a fixed set of candidate deltas.
    Adds explicit singleton penalty for more robust non-singleton clustering.

    Args:
        profiles: ndarray of shape [N, features] with activation profiles.
        delta_range: List of fixed candidate delta values.
        refine: Whether to perform a fine-grained local search around best delta.
        singleton_penalty: Multiplier for singleton ratio penalty (0 disables penalty).

    Returns:
        best_delta: Delta that maximizes adjusted CH score.
    """
    best_delta = delta_range[0]
    best_score = -np.inf

    def adjusted_score(profiles, labels, penalty):
        ch = calinski_harabasz_score(profiles, labels)
        num_singletons = sum(1 for label in np.unique(labels) if np.sum(labels == label) == 1)
        singleton_frac = num_singletons / len(labels)
        return ch * (1.0 - penalty * singleton_frac)

    # --- Coarse search ---
    for d in delta_range:
        model = DPMeans(n_clusters=1, delta=d, n_init=10)
        labels = model.fit_predict(profiles)

        unique_labels = np.unique(labels)
        if 1 < len(unique_labels) < len(profiles):
            score = adjusted_score(profiles, labels, singleton_penalty)
            if score > best_score:
                best_score = score
                best_delta = d

    # Fine-grained refinement around best_delta (smaller step for precision)
    if refine:
        step = 2.0
        local_range = np.arange(max(best_delta - step * 2, 1), best_delta + step * 2 + 1, step)
        for d in local_range:
            model = DPMeans(n_clusters=1, delta=d, n_init=10)
            labels = model.fit_predict(profiles)
            unique_labels = np.unique(labels)
            if 1 < len(unique_labels) < len(profiles):
                score = adjusted_score(profiles, labels, singleton_penalty)
                if score > best_score:
                    best_score = score
                    best_delta = d

    return best_delta

# Step 3.2 Cluster expert voxels within each winning word group
# We use DP-means, which is a non-parametric clustering algorithm that does not require specifying the number of clusters in advance.
# This implementation is based on the pdc_dp_means library, which provides an efficient implementation of the DP-means algorithm.
# https://github.com/BGU-CS-VIL/pdc-dp-means
def cluster_grouped_expert_voxels(
    grouped_experts: Dict[str, List[Tuple[int, np.ndarray]]],
    delta: Union[bool, float] = 10.0,
    n_init: int = 10
) -> Tuple[Union[float, Dict[str, float]], Dict[str, List[np.ndarray]]]:
    """
    Groups expert voxels for each winning word using optimized DP-means.

    Args:
        grouped_experts: Dictionary mapping each word to a list of tuples (voxel_index, activation_profile).
        delta: Penalty parameter for creating a new cluster (equivalent to lambda).
            - If True: tune delta separately for each word.
            - If float: use this shared delta for all words.
        n_init: Number of initializations for the DPMeans algorithm.

    Returns:
        delta_tuned: If per-word tuning, a dict {word: delta}. Otherwise a float.
        clustered_experts: Dictionary mapping each word to a list of clusters (2D arrays).
    """
    clustered_experts = {}
    per_word_delta = {}

    shared_delta = None
    if not isinstance(delta, bool):
        shared_delta = float(delta)

    for word, voxel_data in grouped_experts.items():
        if not voxel_data:
            clustered_experts[word] = []
            if shared_delta is None:
                per_word_delta[word] = np.nan
            continue

        profiles = np.array([profile for _, profile in voxel_data])
        scaler = StandardScaler()
        profiles_normalized = scaler.fit_transform(profiles)

        # Edge case handling: if there is only one voxel, it trivially forms a single cluster
        if len(profiles_normalized) == 1:
            clustered_experts[word] = [profiles]
            per_word_delta[word] = shared_delta if shared_delta is not None else delta
            continue

        # Determine delta for this word
        if shared_delta is None:
            delta_tuned = tune_delta(profiles_normalized)
            per_word_delta[word] = delta_tuned
        else:
            delta_tuned = shared_delta
            per_word_delta[word] = shared_delta

        # Initialization and training of the DPMeans model
        dpmeans = DPMeans(n_clusters=1, n_init=n_init, delta=delta_tuned)
        dpmeans.fit(profiles_normalized)

        # Predict labels (which cluster each profile belongs to)
        labels = dpmeans.predict(profiles_normalized)

        # Split profiles into their respective clusters based on labels
        clusters = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_profiles = profiles[labels == label]
            clusters.append(cluster_profiles)

        clustered_experts[word] = clusters

    return (shared_delta if shared_delta is not None else per_word_delta), clustered_experts

# Step 4. Average withing each cluster to obtain prototypes for each word.
def compute_prototypes_from_clusters(clustered_experts: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    """
    Compute prototype activation profiles for each word by averaging the activation profiles of the voxels in each cluster.

    Args:
        clustered_experts: Dictionary mapping each word to a list of clusters (2D arrays) of activation profiles.

    Returns:
        prototypes: Dictionary mapping each word to a list of prototype activation profiles (1D arrays), one for each cluster.
    """
    prototypes = {}
    
    for word, clusters in clustered_experts.items():
        word_prototypes = []
        for cluster in clusters:
            if cluster.size == 0:
                continue  # Skip empty clusters
            prototype = np.mean(cluster, axis=0)  # Average across voxels in the cluster
            word_prototypes.append(prototype)
        prototypes[word] = word_prototypes
    
    return prototypes

# Step 5. Collect all prototypes vectors across all words and clusters to form the final prototype set.
def collect_prototype_set(prototypes: Dict[str, List[np.ndarray]]) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """
    Flatten all prototype vectors into a single matrix and keep track of labels.

    Args:
        prototypes: Dictionary mapping each word to a list of prototype vectors.

    Returns:
        prototype_set: np.ndarray of shape (num_prototypes, word_activations)
        labels: list of tuples (word, cluster_id)
    """
    prototype_set = []
    labels = []
    
    for word, prototypes_list in prototypes.items():
        for cluster_id, proto in enumerate(prototypes_list):
            prototype_set.append(proto)
            labels.append((word, cluster_id))
    prototype_set = np.vstack(prototype_set) if len(prototype_set) > 0 else np.array([])
    
    return prototype_set, labels


# Functions for computing and plotting similarity matrix of prototypes
# The similarity is computed as the average of the max cosine similarity between prototypes of word1 and prototypes of word2, 
# and vice versa, to ensure symmetry and avoid bias from differing numbers of prototypes.
def word_similarity(prototypes, word1, word2) -> float:
    """
    Compute a symmetric similarity between two words with potentially different numbers of prototypes.
    Uses max-over-prototypes for each direction to avoid bias from differing counts.
    
    Args:
        prototypes: Dict mapping words to lists of prototype vectors.
        word1: First word.
        word2: Second word.
    Returns:
        similarity: A single scalar representing the similarity between the two words.
    """
    P1 = np.vstack(prototypes[word1])  # Shape (n1, D)
    P2 = np.vstack(prototypes[word2])  # Shape (n2, D)

    sim_matrix = cosine_similarity(P1, P2)  # Shape (n1, n2)
    # Max over P2 for each P1, then mean
    sim1 = sim_matrix.max(axis=1).mean()
    # Max over P1 for each P2, then mean
    sim2 = sim_matrix.max(axis=0).mean()
    # Symmetric similarity
    return 0.5 * (sim1 + sim2)

def compute_similarity_matrix(prototypes) -> Tuple[List[str], np.ndarray]:
    """
    Compute a square similarity matrix between all words in prototypes.

    Args:
        prototypes: Dict mapping words to lists of prototype vectors.
    
    Returns:
        words: List of words corresponding to the rows/columns of the similarity matrix.
        sim_matrix: 2D numpy array containing similarity values.
    """
    words = list(prototypes.keys())
    n = len(words)

    sim_matrix = np.zeros((n, n))
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            sim_matrix[i, j] = word_similarity(prototypes, w1, w2)

    return words, sim_matrix

def plot_similarity_matrix(words, sim_matrix) -> plt.Figure:
    """
    Plot a similarity matrix with proper word labels.
    
    Args:
        words: List of word labels corresponding to the rows/columns of the similarity matrix.
        sim_matrix: 2D numpy array containing similarity values.
        
    Returns:
        fig: The matplotlib figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(sim_matrix, cmap='viridis')
    fig.colorbar(im, ax=ax, label="Cosine similarity")

    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=90)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_title("Word Similarity (cosine)")
    fig.tight_layout()
    
    return fig
    
    
# Functions for identifying the biggest x that keeps all 60 words present in the prototype set.
def get_word_counts_for_x(word_activations, word_labels, x) -> int:
    """ 
    Helper function to get the count of words that have at least one expert voxel for a given x. 
    
    Args:
        word_activations: Array of shape [60, V] with averaged responses per word.
        word_labels: List of 60 word strings in alphabetical order.
        x: Threshold parameter to identify expert voxels.
        
    Returns:
        present_words: Number of words that have at least one expert voxel.
    """
    experts = identify_expert_voxels(word_activations, word_labels, x_threshold_parameter=x)
    grouped = group_expert_voxels_by_word(experts)
    # Count words with at least one expert voxel
    present_words = sum(1 for voxels in grouped.values() if len(voxels) > 0)
    return present_words

def get_max_x_for_all_words(word_activations, word_labels, x_min=0.0, x_max=10000.0, tol=1e-3) -> float:
    """
    Use binary search to find the maximum x that keeps all 60 words present in the prototype
    set.
    
    Args:
        word_activations: Array of shape [60, V] with averaged responses per word.
        word_labels: List of 60 word strings in alphabetical order.
        x_min: Minimum x to consider.
        x_max: Maximum x to consider.
        tol: Tolerance for convergence of the binary search.
    
    Returns:
        best_x: The maximum x that keeps all 60 words present in the prototype set.
    """
    # First check if all words have experts at x=0
    count_at_zero = get_word_counts_for_x(word_activations, word_labels, 0.0)
    if count_at_zero < 60:
        log.warning(f"Only {count_at_zero}/60 words have expert voxels at x=0. Identifying missing words...")
        experts = identify_expert_voxels(word_activations, word_labels, x_threshold_parameter=0.0)
        grouped = group_expert_voxels_by_word(experts)
        present_words = set(grouped.keys())
        missing_words = set(word_labels) - present_words
        log.warning(f"Missing words at x=0: {sorted(missing_words)}")
    
    lo, hi = x_min, x_max
    best = x_min
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        count = get_word_counts_for_x(word_activations, word_labels, mid)
        if count == 60:
            best = mid
            lo = mid
        else:
            hi = mid
    return best
    
    

def run_prototype_computation(
    brain_data_path: Union[pathlib.Path, str],
    x_threshold_parameter: Union[bool, float] = 10.0,
    delta: Union[bool, float] = 10.0,
    n_init: int = 10
) -> None:
    """
    Run the full prototype computation pipeline and save the results in the 'prototypes' directory.

    Args:
        mat_path: Path to the .mat file containing brain data.
    """
    
    brain_data_path = pathlib.Path(brain_data_path) if isinstance(brain_data_path, str) else brain_data_path

    if not brain_data_path.exists():
        raise FileNotFoundError(f"Brain data path not found: {brain_data_path}")

    # Collect .mat files: single file or directory
    if brain_data_path.is_file():
        mat_files = [brain_data_path]
    elif brain_data_path.is_dir():
        mat_files = sorted(brain_data_path.glob("*.mat"))
        if not mat_files:
            raise ValueError(f"No .mat files found in directory: {brain_data_path}")
    else:
        raise ValueError(f"Invalid brain_data_path: {brain_data_path} (must be a file or directory)")
    
    log.info(f"Found {len(mat_files)} .mat file(s)")
    
    # Create a timestamp for saving results
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    for mat_idx, mat_file in enumerate(mat_files):
        log.info(f"Processing file: {mat_file}")
    
        # Get the directory where the current script is located
        script_dir = pathlib.Path(__file__).parent
        # Build paths relative to the script
        target_path = script_dir / "../" / "prototypes" / date_str / f"P{mat_idx+1}"
        # Create folders if missing
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Define paths for saving logs, prototypes and similarity matrix
        log_file_path = target_path / "pipeline.log"
        proto_saving_path = target_path / "prototypes.npz"
        sim_matrix_saving_path = target_path / "similarity_matrix.png"
        
        # Add file handler to capture logs for this participant
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    
        # Load brain data
        log.info(f"Loading brain data from {mat_file}...")
        word_activations, _, word_labels = load_brain_data(mat_file)
        log.info(f"Loaded brain data with shape: {word_activations.shape}")
        
        # If x_threshold_parameter is True, we will compute the best x that keeps all 60 words. Otherwise, we will use the provided float value.
        if isinstance(x_threshold_parameter, bool) and x_threshold_parameter is True:
            actual_x = get_max_x_for_all_words(word_activations, word_labels)
            log.info(f"Auto-selected x threshold keeping all 60 words: {actual_x:.3f}")
        else:
            actual_x = float(x_threshold_parameter)
            log.info(f"Using manual x threshold: {actual_x:.3f}")

        # Identify expert voxels
        log.info(f"Identifying expert voxels with x = {actual_x}...")
        expert_voxels = identify_expert_voxels(word_activations, word_labels, x_threshold_parameter=actual_x)
        log.info(f"Identified {len(expert_voxels)} expert voxels.")
        
        # Group expert voxels by their winning word
        log.info("Grouping expert voxels by their winning word...")
        grouped_experts = group_expert_voxels_by_word(expert_voxels)
        # Sort by word alphabetically for consistent ordering throughout the pipeline
        grouped_experts = {word: grouped_experts[word] for word in sorted(grouped_experts.keys())}
        for idx, (word, voxels) in enumerate(grouped_experts.items(), start=1):
            log.info(f"{idx}. Word '{word}' has {len(voxels)} expert voxels.")
        
        # Cluster expert voxels within each winning word group
        log.info("Clustering expert voxels within each winning word group using DP-means...")
        delta_tuned, clustered_experts = cluster_grouped_expert_voxels(grouped_experts, delta, n_init=n_init)

        # log per-word delta for debugging
        for idx, (word, clusters) in enumerate(clustered_experts.items(), start=1):
            if isinstance(delta_tuned, dict):
                word_delta = delta_tuned.get(word, float('nan'))
                log.info(f"{idx}. Word '{word}' has delta = {word_delta:.3f} and {len(clusters)} clusters of expert voxels, with cluster sizes: {[cluster.shape[0] for cluster in clusters]}")
            else:
                log.info(f"{idx}. Word '{word}' has delta = {delta_tuned:.3f} and {len(clusters)} clusters of expert voxels, with cluster sizes: {[cluster.shape[0] for cluster in clusters]}")

        # Compute prototypes from clusters
        log.info("Computing prototypes from clusters...")
        prototypes = compute_prototypes_from_clusters(clustered_experts)
        log.info(f"Prototypes computed.")
        
        # Collect all prototypes into a single set
        log.info("Collecting all prototypes into a single set...")
        prototype_set, labels = collect_prototype_set(prototypes)
        log.info(f"Collected {prototype_set.shape[0]} prototypes across all words and clusters.")
        
        # Compute similarity matrix and plot
        log.info("Computing similarity matrix...")
        words, sim_matrix = compute_similarity_matrix(prototypes)
        log.info("Plotting similarity matrix...")
        fig = plot_similarity_matrix(words, sim_matrix)
        
        # Save prototypes and similarity matrix
        np.savez(proto_saving_path, prototypes=prototype_set, labels=np.array(labels, dtype=object))
        fig.savefig(sim_matrix_saving_path, dpi=300)
        log.info(f"Saved prototypes to {proto_saving_path} and similarity matrix plot to {sim_matrix_saving_path}")
        log.info(f"Saved pipeline log to {log_file_path}")
        
        # Clean up to free memory
        del word_activations, word_labels, expert_voxels, grouped_experts, clustered_experts, prototypes, prototype_set, labels, words, sim_matrix
        plt.close(fig)
        
        # Remove file handler to avoid duplicate logs in next iteration
        log.removeHandler(file_handler)
        file_handler.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compute_prototypes.py",
        description=(
            "Compute prototype activation profiles from brain data using expert voxel selection "
            "and DP-means clustering. The pipeline identifies expert voxels per word, clusters them, "
            "derives prototype representations, and evaluates semantic consistency via cosine similarity "
            "between words."
        )
    )

    parser.add_argument(
        "--brain_data_path",
        type=str,
        default="../brain_data/",
        help="Path to the input brain data file (.mat format)."
    )

    """ parser.add_argument(
        "--x_threshold",
        type=float,
        default=10.0,
        help=(
            "Threshold parameter for expert voxel selection. Higher values make the selection stricter "
            "(fewer expert voxels retained). Default: 10.0"
        )
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=10.0,
        help=(
            "DP-means clustering penalty parameter controlling the creation of new clusters. "
            "Higher values result in fewer clusters. Default: 10.0"
        )
    ) """

    parser.add_argument(
        "--n_init",
        type=int,
        default=10,
        help="Number of initializations for DP-means clustering. Default: 10"
    )

    parser.add_argument(
        "--no_auto_x",
        action="store_false",
        dest="auto_x",
        help="Disable automatic search of x threshold"
    )
    
    parser.add_argument(
        "--no_auto_delta",
        action="store_false",
        dest="auto_delta",
        help="Disable automatic tuning of DP-means delta parameter"
    )
    
    args = parser.parse_args()
    
    x_threshold_parameter = True if args.auto_x else args.x_threshold
    delta = True if args.auto_delta else args.delta

    run_prototype_computation(
        brain_data_path=pathlib.Path(args.brain_data_path),
        x_threshold_parameter=x_threshold_parameter,
        delta=delta,
        n_init=args.n_init
    )