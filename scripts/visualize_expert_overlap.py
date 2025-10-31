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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
# from sklearn.manifold import MDS
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------- CONFIG ----------------
CSV_PATH = "/home/juancm/trento/abns/project/juans_fork/ml-selfcond/pythia_openai_custom_60_expert_sim/similarity_matrix_tau_0.5.csv"   # path to your matrix CSV
OUT_DIR = "/home/juancm/trento/abns/project/juans_fork/ml-selfcond/pythia_openai_custom_60_expert_sim/graph_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Thresholds (tune for your matrix)
LOW_THRESHOLD = 0.3    # edges with sim >= LOW_THRESHOLD are kept

# Edge width mapping
MIN_EDGE_WIDTH = 0.2
MAX_EDGE_WIDTH = 15.0
FIG_DPI = 200
BASE_FONTSIZE = 9
# Layout options: 'spring' | 'mds' | 'hybrid'
LAYOUT_MODE = 'mds'
SPRING_K = 0.4
HYBRID_ITERATIONS = 100
SEED = 0
# ----------------------------------------

# -------- Load matrix --------
print(f"Loading matrix from: {CSV_PATH}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Matrix file not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, index_col=0)
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

# Extract top 10 highest similarities for JSON output
top_similarities = []
for i in range(N):
    for j in range(i+1, N):
        sim = J[i, j]
        top_similarities.append((concepts[i], concepts[j], float(sim)))

# Sort by similarity descending and take top 10
top_similarities.sort(key=lambda x: x[2], reverse=True)
top_10 = top_similarities[:10]

# -------- Heatmap --------
def plot_heatmap(J, labels, outpath):
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
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)

heatmap_path = os.path.join(OUT_DIR, "jaccard_heatmap.png")
plot_heatmap(J, concepts, heatmap_path)
print("Saved heatmap →", heatmap_path)

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
                                 min_edge_width=0.2,
                                 max_edge_width=15.0,
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

G, edge_widths = build_graph_threshold_linear(J, concepts,
                                             low_threshold=LOW_THRESHOLD,
                                             min_edge_width=MIN_EDGE_WIDTH,
                                             max_edge_width=MAX_EDGE_WIDTH,
                                             symmetrize=True)

# print(f"Threshold graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (J>{THRESHOLD}).")

# -------- Plot graph with MDS layout --------
def plot_graph(G, J, concepts, outpath, edge_widths):
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
    if LAYOUT_MODE == 'mds':
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=SEED)
        coords = mds.fit_transform(dist)
        pos = {node: (coords[idx_map[node], 0], coords[idx_map[node], 1]) for node in kept_nodes}
        layout_title = 'MDS'
    elif LAYOUT_MODE == 'hybrid':
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=SEED)
        coords = mds.fit_transform(dist)
        init_pos = {node: (coords[idx_map[node], 0], coords[idx_map[node], 1]) for node in kept_nodes}
        pos = nx.spring_layout(G, weight='weight', seed=SEED, k=SPRING_K, pos=init_pos, iterations=HYBRID_ITERATIONS)
        layout_title = f'Hybrid (MDS init + spring {HYBRID_ITERATIONS} it)'
    else:  # 'spring'
        pos = nx.spring_layout(G, weight='weight', seed=SEED, k=SPRING_K)
        layout_title = 'Spring'
    
    plt.figure(figsize=(10,10))
    # edges: use precomputed widths mapped to current edge order
    widths = []
    ordered_edges = list(G.edges())
    for (u, v) in ordered_edges:
        key = (u, v) if u < v else (v, u)
        w = edge_widths.get(key, MIN_EDGE_WIDTH)
        widths.append(w)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#444", alpha=0.8)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='red', edgecolors='k')
    # labels
    for node, (x,y) in pos.items():
        plt.text(x, y, concepts[node], fontsize=BASE_FONTSIZE,
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
    plt.title(f"Graph of Concepts [{layout_title}] (edges where J ≥ {LOW_THRESHOLD}; linear width by sim)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()

graph_path = os.path.join(OUT_DIR, f"graph_threshold_{LOW_THRESHOLD}_{LAYOUT_MODE}.png")
plot_graph(G, J, concepts, graph_path, edge_widths)
print("Saved threshold graph →", graph_path)

# Save top 10 similarities to JSON
top_10_json = {
    "top_10_similarities": [
        {"concept1": c1, "concept2": c2, "similarity": sim}
        for c1, c2, sim in top_10
    ]
}
json_path = os.path.join(OUT_DIR, "top_10_similarities.json")
with open(json_path, 'w') as f:
    json.dump(top_10_json, f, indent=2)
print("Saved top 10 similarities →", json_path)

print("\nDone! Outputs are in:", OUT_DIR)