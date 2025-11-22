#!/usr/bin/env python3
"""
Simplified script:
- Loads a Jaccard similarity matrix from CSV
- Plots a heatmap of similarities
- Builds a threshold-based graph (edges if J > threshold)
- Plots the graph with labeled nodes at MDS coordinates
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
# from sklearn.manifold import MDS
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------- CONFIG (defaults) ----------------
# Edge width mapping constants
DEFAULT_MIN_EDGE_WIDTH = 0.2
DEFAULT_MAX_EDGE_WIDTH = 15.0
DEFAULT_FIG_DPI = 200
DEFAULT_BASE_FONTSIZE = 9
DEFAULT_SPRING_K = 0.5
DEFAULT_HYBRID_ITERATIONS = 100
DEFAULT_SEED = 0
# ----------------------------------------

def load_similarity_matrix(csv_path: str):
    print(f"Loading matrix from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Matrix file not found: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    concepts = list(df.index.astype(str))
    J = df.values
    N = len(concepts)
    print(f"Loaded {J.shape[0]}×{J.shape[1]} Jaccard matrix with {len(concepts)} concepts.")
    # Validate matrix shape
    if J.shape[0] != J.shape[1]:
        raise ValueError(f"Matrix is not square: {J.shape}")
    if J.shape[0] != len(concepts):
        raise ValueError(f"Matrix size {J.shape[0]} doesn't match concepts count {len(concepts)}")
    # Ensure diagonal = 1
    np.fill_diagonal(J, 1.0)
    return J, concepts

def compute_top_k_pairs(J: np.ndarray, labels: list[str], k: int = 10):
    N = len(labels)
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            sim = float(J[i, j])
            pairs.append((labels[i], labels[j], sim))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]

# -------- Heatmap --------
def plot_heatmap(J, labels, outpath, dpi=DEFAULT_FIG_DPI):
    n = len(labels)
    # Scale figure size with number of labels (capped for practicality)
    side = min(28, max(8, 0.05 * n + 8))
    fig, ax = plt.subplots(figsize=(side, side))

    im = ax.imshow(J, interpolation='nearest', aspect='equal', cmap='viridis')
    ax.set_title("Jaccard Similarity Heatmap")

    # Build a separate axis for the colorbar so the heatmap axis keeps its width
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.4)
    cb = fig.colorbar(im, cax=cax, label="Jaccard similarity")

    # Tick labeling strategy
    if n <= 60:
        fontsize = 8 if n > 40 else 9
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=fontsize)
    else:
        # For large N, sample ticks every k to keep labels readable
        step = max(1, n // 50)  # show ~50 labels
        idx = list(range(0, n, step))
        ax.set_xticks(idx)
        ax.set_xticklabels([labels[i] for i in idx], rotation=90, fontsize=6)
        ax.set_yticks(idx)
        ax.set_yticklabels([labels[i] for i in idx], fontsize=6)
        # Light grid to help navigation
        ax.grid(True, which='major', color='w', alpha=0.15, linestyle='--')

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# -------- Build graph (threshold) --------
def first_build_graph_threshold(J, concepts, threshold=0.1):
    """Undirected weighted graph: edges if J > threshold."""
    G = nx.Graph()
    for i, c in enumerate(concepts):
        G.add_node(i, label=c)
    N = J.shape[0]
    print(f"Building graph from {N}×{J.shape[1]} matrix with threshold {threshold}")
    
    edges_added = 0
    for i in range(N):
        for j in range(i+1, N):
            if i >= J.shape[0] or j >= J.shape[1]:
                print(f"WARNING: Index out of bounds: i={i}, j={j}, matrix shape={J.shape}")
                continue
            w = float(J[i,j])
            if w > threshold:
                G.add_edge(i, j, weight=w)
                edges_added += 1
    
    print(f"Added {edges_added} edges with weight > {threshold}")
    return G

def build_graph_threshold_linear(J, labels,
                                 low_threshold=0.1,
                                 min_edge_width=DEFAULT_MIN_EDGE_WIDTH,
                                 max_edge_width=DEFAULT_MAX_EDGE_WIDTH,
                                 symmetrize=True):
    """
    Build an undirected weighted graph from similarity matrix J using a low threshold for edge inclusion.

    - Keep edges only if sim >= low_threshold.
    - Edge width is linear: width = min_edge_width + (sim - low_threshold) / (1.0 - low_threshold) * (max_edge_width - min_edge_width)
      for sim >= low_threshold. This scales width proportionally to sim above the threshold.
    - symmetrize: uses max(J, J.T) before thresholding

    Returns: networkx.Graph with 'weight' attribute on edges and a dict `edge_widths` mapping (i,j)->width.
    """
    if symmetrize:
        mat = np.maximum(J, J.T).copy()
    else:
        mat = J.copy()

    n = mat.shape[0]
    G = nx.Graph()
    for i, lab in enumerate(labels):
        G.add_node(i, label=lab)

    edge_widths = {}
    for i in range(n):
        for j in range(i+1, n):
            sim = float(mat[i, j])
            if sim >= low_threshold:
                # Linear width mapping based on sim value (no high threshold)
                if low_threshold >= 1.0:
                    w = float(max_edge_width)  # edge case: all sims >=1, use max
                else:
                    frac = (sim - low_threshold) / (1.0 - low_threshold)
                    w = float(min_edge_width + frac * (max_edge_width - min_edge_width))
                G.add_edge(i, j, weight=sim)
                edge_widths[(i, j)] = w

    return G, edge_widths

# print(f"Threshold graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (J>{THRESHOLD}).")

# -------- Plot graph with MDS layout --------
def plot_graph(
    G, 
    J, 
    concepts, 
    outpath, 
    edge_widths, 
    layout_mode, 
    low_threshold: float,
    min_edge_width=DEFAULT_MIN_EDGE_WIDTH,
    dpi=DEFAULT_FIG_DPI,
    fontsize=DEFAULT_BASE_FONTSIZE,
    spring_k=DEFAULT_SPRING_K,
    hybrid_iterations=DEFAULT_HYBRID_ITERATIONS,
    seed=DEFAULT_SEED,
):
    # Remove isolated nodes (no edges kept by thresholds)
    isolated = [n for n, d in G.degree() if d == 0]
    if isolated:
        G = G.copy()
        G.remove_nodes_from(isolated)

    # If all nodes removed, skip
    if G.number_of_nodes() == 0:
        print("No nodes meet threshold criteria; skipping graph plot.")
        return

    # Build index mapping for subgraph to extract submatrix for MDS/hybrid
    kept_nodes = sorted(G.nodes())
    idx_map = {node: i for i, node in enumerate(kept_nodes)}
    labels_sub = [concepts[i] for i in kept_nodes]

    # Submatrix of J for kept nodes
    J_sub = J[np.ix_(kept_nodes, kept_nodes)]
    dist = 1.0 - J_sub

    # Compute positions based on layout mode
    if layout_mode == 'mds':
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed)
        coords = mds.fit_transform(dist)
        pos = {node: (coords[idx_map[node], 0], coords[idx_map[node], 1]) for node in kept_nodes}
        layout_title = 'MDS'
    elif layout_mode == 'hybrid':
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed)
        coords = mds.fit_transform(dist)
        init_pos = {node: (coords[idx_map[node], 0], coords[idx_map[node], 1]) for node in kept_nodes}
        pos = nx.spring_layout(G, weight='weight', seed=seed, k=spring_k, pos=init_pos, iterations=hybrid_iterations)
        layout_title = f'Hybrid (MDS init + spring {hybrid_iterations} it)'
    else:  # 'spring'
        pos = nx.spring_layout(G, weight='weight', seed=seed, k=spring_k)
        layout_title = 'Spring'
    
    plt.figure(figsize=(10,10))
    # edges: use precomputed widths mapped to current edge order
    widths = []
    ordered_edges = list(G.edges())
    for (u, v) in ordered_edges:
        key = (u, v) if u < v else (v, u)
        w = edge_widths.get(key, min_edge_width)
        widths.append(w)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#444", alpha=0.8)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='red', edgecolors='k')
    # labels
    for node, (x,y) in pos.items():
        plt.text(x, y, concepts[node], fontsize=fontsize,
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
    plt.title(f"Graph of Concepts [{layout_title}] (edges where J ≥ {low_threshold}; linear width by sim)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()



def run_overlap_visualization(
    csv_path: str,
    output_dir: str,
    low_threshold: float = 0.2,
    layout_mode: str = "mds",
    topk: int = 10,
    min_edge_width: float = DEFAULT_MIN_EDGE_WIDTH,
    max_edge_width: float = DEFAULT_MAX_EDGE_WIDTH,
    dpi: int = DEFAULT_FIG_DPI,
    fontsize: int = DEFAULT_BASE_FONTSIZE,
    spring_k: float = DEFAULT_SPRING_K,
    hybrid_iterations: int = DEFAULT_HYBRID_ITERATIONS,
    seed: int = DEFAULT_SEED,
):
    """
    Run the expert overlap visualization pipeline.
    
    Args:
        csv_path: Path to the CSV file containing the similarity matrix
        output_dir: Path to the output directory
        low_threshold: Low threshold for edge inclusion
        layout_mode: Layout mode ('spring', 'mds', 'hybrid')
        topk: Number of top similar pairs to save
        min_edge_width: Minimum edge width
        max_edge_width: Maximum edge width
        dpi: Figure DPI
        fontsize: Base font size
        spring_k: Spring layout constant
        hybrid_iterations: Number of iterations for hybrid layout
        seed: Random seed
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load matrix
    J, concepts = load_similarity_matrix(csv_path)

    # Heatmap
    heatmap_path = os.path.join(output_dir, "jaccard_heatmap.png")
    plot_heatmap(J, concepts, heatmap_path, dpi=dpi)
    print("Saved heatmap →", heatmap_path)

    # Build graph and edge widths from similarity matrix
    G, edge_widths = build_graph_threshold_linear(
        J, concepts,
        low_threshold=low_threshold,
        min_edge_width=min_edge_width,
        max_edge_width=max_edge_width,
        symmetrize=True
    )

    # Plot graph
    graph_path = os.path.join(output_dir, f"graph_threshold_{low_threshold}_{layout_mode}.png")
    plot_graph(
        G, J, concepts, graph_path, edge_widths, layout_mode, low_threshold,
        min_edge_width=min_edge_width, dpi=dpi, fontsize=fontsize,
        spring_k=spring_k, hybrid_iterations=hybrid_iterations, seed=seed
    )
    print("Saved threshold graph →", graph_path)

    # Save top 10 similarities to JSON
    top_ks = compute_top_k_pairs(J, concepts, k=topk)
    top_k_json = {
        "top_k": topk,
        "pairs": [
            {"concept1": c1, "concept2": c2, "similarity": sim}
            for c1, c2, sim in top_ks
        ]
    }
    json_path = os.path.join(output_dir, "top_similarities.json")
    with open(json_path, 'w') as f:
        json.dump(top_k_json, f, indent=2)
    print("Saved top similarities →", json_path)

    print("\nDone! Outputs are in:", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Visualize expert overlap")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file containing the similarity matrix")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--low-threshold", type=float, default=0.2, help="Low threshold for edge inclusion")
    parser.add_argument("--layout-mode", type=str, default="mds", choices=["spring","mds","hybrid"], help="Layout mode")
    parser.add_argument("--topk", type=int, default=10, help="Save JSON with top-K most similar pairs")
    
    args = parser.parse_args()
    
    run_overlap_visualization(
        csv_path=args.csv,
        output_dir=args.output_dir,
        low_threshold=args.low_threshold,
        layout_mode=args.layout_mode,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
