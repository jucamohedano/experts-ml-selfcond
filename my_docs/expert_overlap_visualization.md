# Expert Overlap Visualization

This document describes the visualization methods used to display expert neuron overlap between concepts.

## Overview

After computing Jaccard similarity between expert neuron sets (via `analyze_expert_overlap.py`), we visualize the results using:

1. **Heatmap**: Direct visualization of the similarity matrix
2. **Graph**: Network visualization where edges connect similar concepts

The graph visualization supports multiple **layout algorithms** to position the concept nodes.

---

## Layout Algorithms

### 1. MDS (Multidimensional Scaling)

**Purpose**: Position nodes so that distances between them reflect actual dissimilarities.

**Algorithm**:
1. Compute distance matrix: `D = 1 - J` (where J is Jaccard similarity)
2. Use MDS to find 2D coordinates that minimize stress: `Σ(d_ij - D_ij)²`
3. Concepts with high similarity → close together in the plot

**Pros**:
- Preserves global structure
- Deterministic (given same seed)
- Good for seeing overall concept relationships

**Cons**:
- Ignores edge weights/thresholds
- May produce overlapping labels for dense clusters

**Code** (from `visualize_expert_overlap.py`):
```python
from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed)
coords = mds.fit_transform(1.0 - J)  # J is Jaccard similarity matrix
```

---

### 2. Spring (Force-Directed) Layout

**Purpose**: Use physics simulation where edges act as springs.

**Algorithm**:
1. Initialize nodes at random positions
2. Iterate physics simulation:
   - **Attraction**: Connected nodes (edges with J > threshold) pull together proportional to edge weight
   - **Repulsion**: All nodes repel each other (prevents overlap)
3. Converge to equilibrium

**Pros**:
- Good local clustering
- Naturally separates disconnected components
- Weighted edges → stronger connections appear closer

**Cons**:
- Random initialization → different results each run
- May lose global distance structure
- Can get stuck in local minima

**Code**:
```python
pos = nx.spring_layout(G, weight='weight', seed=seed, k=spring_k)
```

**Parameters**:
- `k`: Optimal distance between nodes (higher = more spread out)
- `weight`: Edge attribute to use for attraction strength

---

### 3. Hybrid (MDS + Spring) Layout ⭐ Recommended

**Purpose**: Combine MDS's global structure preservation with Spring's local refinement.

**Algorithm**:

```
┌────────────────────────────────────────────────────────────────────┐
│                     HYBRID LAYOUT ALGORITHM                        │
└────────────────────────────────────────────────────────────────────┘

   STAGE 1: MDS (Global Positioning)
   ─────────────────────────────────
   
   Input: Jaccard similarity matrix J
          ┌───────────────────────┐
          │ 1.0  0.8  0.3  0.1    │
          │ 0.8  1.0  0.2  0.4    │
          │ 0.3  0.2  1.0  0.7    │
          │ 0.1  0.4  0.7  1.0    │
          └───────────────────────┘

   Step 1: Convert to distance matrix
           D = 1 - J
          ┌───────────────────────┐
          │ 0.0  0.2  0.7  0.9    │
          │ 0.2  0.0  0.8  0.6    │
          │ 0.7  0.8  0.0  0.3    │
          │ 0.9  0.6  0.3  0.0    │
          └───────────────────────┘

   Step 2: MDS optimization
           Find 2D coordinates (x,y) that minimize:
           Stress = Σ_ij (distance(node_i, node_j) - D_ij)²

   Output: Initial positions
                    ┌──────────────┐
           node_0 → │ (0.8, 0.2)   │
           node_1 → │ (0.6, 0.3)   │  ← Similar concepts
           node_2 → │ (-0.5, -0.4) │    start close
           node_3 → │ (-0.7, -0.2) │  ← together
                    └──────────────┘

   STAGE 2: Spring Refinement (Local Optimization)
   ────────────────────────────────────────────────

   Input: Graph G with weighted edges (only edges where J ≥ threshold)
          Initial positions from MDS

   Physics Simulation:
   ┌─────────────────────────────────────────────────────────────────┐
   │  For each iteration (default: 100):                            │
   │                                                                 │
   │    For each pair of nodes (i, j):                              │
   │      ┌─────────────────────────────────────────────────────┐   │
   │      │ If edge exists (J_ij ≥ threshold):                  │   │
   │      │   ATTRACT:  Move i ← j proportional to J_ij         │   │
   │      │             (spring force)                          │   │
   │      │                                                     │   │
   │      │ Always:                                             │   │
   │      │   REPEL:    Push i away from j                      │   │
   │      │             (electrostatic force)                   │   │
   │      └─────────────────────────────────────────────────────┘   │
   │                                                                 │
   │    Update positions, reduce step size                          │
   └─────────────────────────────────────────────────────────────────┘

   Output: Final positions (refined)
```

**Code** (from `visualize_expert_overlap.py`, lines 205-211):
```python
# Stage 1: MDS for initial positions
from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed)
dist = 1.0 - J_sub  # Convert similarity to distance
coords = mds.fit_transform(dist)
init_pos = {node: (coords[i, 0], coords[i, 1]) for i, node in enumerate(kept_nodes)}

# Stage 2: Spring refinement starting from MDS positions
pos = nx.spring_layout(
    G, 
    weight='weight',      # Use Jaccard similarity as edge weight
    seed=seed,            # Reproducibility
    k=spring_k,           # Node spacing parameter (default: 0.5)
    pos=init_pos,         # Start from MDS positions (not random!)
    iterations=hybrid_iterations  # Number of spring iterations (default: 100)
)
```

**Why Hybrid is Recommended**:

| Aspect | MDS Only | Spring Only | Hybrid |
|--------|----------|-------------|--------|
| Global structure | ✅ Preserved | ❌ Random | ✅ Preserved |
| Local clustering | ⚠️ Moderate | ✅ Good | ✅ Good |
| Stability | ✅ Deterministic | ❌ Random | ✅ Deterministic |
| Edge weights | ❌ Ignored | ✅ Used | ✅ Used |
| Label overlap | ⚠️ Possible | ⚠️ Possible | ✅ Minimized |

**Parameters**:
- `spring_k` (default: 0.5): Ideal distance between nodes
  - Higher = more spread out
  - Lower = tighter clusters
- `hybrid_iterations` (default: 100): Number of spring refinement steps
  - More iterations = more refined, but slower
  - 50-200 is typically sufficient

---

### 4. UMAP Layout

**Purpose**: Non-linear dimensionality reduction that preserves local neighborhood structure.

**Algorithm**:
1. Build a fuzzy topological representation of the data
2. Optimize a low-dimensional representation to match

**Pros**:
- Excellent at preserving local clusters
- Handles non-linear relationships

**Cons**:
- May distort global distances
- Requires additional dependency (`umap-learn`)

---

## Edge Thresholding and Width

### Edge Inclusion

Only edges where `J ≥ low_threshold` are drawn. This reduces visual clutter.

**Default**: `low_threshold = 0.2` (20% Jaccard similarity)

### Edge Width Mapping

Edge width is **linearly scaled** based on similarity:

```python
# From build_graph_threshold_linear() in visualize_expert_overlap.py
frac = (sim - low_threshold) / (1.0 - low_threshold)
width = min_edge_width + frac * (max_edge_width - min_edge_width)
```

| Similarity | Fraction | Width (if min=0.2, max=15.0) |
|------------|----------|------------------------------|
| 0.2 (threshold) | 0.0 | 0.2 (thinnest) |
| 0.5 | 0.375 | 5.75 |
| 0.8 | 0.75 | 11.3 |
| 1.0 | 1.0 | 15.0 (thickest) |

---

## Usage

### Command Line

```bash
# MDS layout (default)
python scripts/visualize_expert_overlap.py \
    --csv results/.../jaccard_similarity.csv \
    --output-dir results/.../visualizations \
    --layout-mode mds

# Hybrid layout (recommended)
python scripts/visualize_expert_overlap.py \
    --csv results/.../jaccard_similarity.csv \
    --output-dir results/.../visualizations \
    --layout-mode hybrid

# Spring layout
python scripts/visualize_expert_overlap.py \
    --csv results/.../jaccard_similarity.csv \
    --output-dir results/.../visualizations \
    --layout-mode spring \
    --low-threshold 0.3  # Higher threshold for cleaner graph
```

### Via Hydra (run_pipeline.py)

```bash
# Hybrid layout with custom parameters
python run_pipeline.py task=visualize_overlap \
    task.csv_path=/path/to/similarity_matrix.csv \
    task.visualization.layout_mode=hybrid \
    task.visualization.low_threshold=0.2 \
    task.visualization.hybrid_iterations=150
```

---

## Output Files

| File | Description |
|------|-------------|
| `jaccard_heatmap.png` | N×N heatmap of similarity matrix |
| `graph_threshold_{threshold}_{layout}.png` | Network graph with specified layout |
| `top_similarities.json` | Top-K most similar concept pairs |

---

## Interpretation Guide

### Heatmap

- **Bright cells**: High Jaccard similarity (concepts share many expert neurons)
- **Dark cells**: Low similarity (concepts have distinct expert sets)
- **Diagonal**: Always 1.0 (self-similarity)
- **Clusters**: Look for bright rectangular blocks indicating concept groups

### Graph

- **Node proximity**: Concepts close together share expert neurons
- **Edge thickness**: Thicker = higher Jaccard similarity
- **Isolated nodes**: Concepts with no similarity above threshold
- **Clusters**: Groups of interconnected nodes = semantically related concepts

### Expected Patterns

- **vehicle** concepts (airplane, car, train) should cluster together
- **animal** concepts should form a separate cluster
- **cross-category edges** (e.g., vehicle–animal) should be thin or absent

---

## Mathematical Background

### Jaccard Similarity

For two expert sets A and B:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

- **J = 1.0**: Identical expert sets
- **J = 0.0**: No shared experts
- **J = 0.5**: Half of the neurons are shared

### MDS Stress Function

MDS minimizes:

$$
\text{Stress} = \sqrt{\frac{\sum_{i<j}(d_{ij} - \delta_{ij})^2}{\sum_{i<j}\delta_{ij}^2}}
$$

Where:
- $d_{ij}$ = Euclidean distance between nodes i and j in 2D
- $\delta_{ij}$ = Original dissimilarity (1 - Jaccard)

### Spring Layout Energy

The spring layout minimizes:

$$
E = \sum_{(i,j) \in E} w_{ij} \cdot ||p_i - p_j||^2 - \sum_{i \neq j} \frac{1}{||p_i - p_j||}
$$

Where:
- First term: Attraction (weighted by similarity)
- Second term: Repulsion (prevents overlap)
