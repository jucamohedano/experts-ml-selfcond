# Expert Overlap Analysis & Visualization

This documentation covers the methods for analyzing and visualizing the **overlap** (similarity) between expert neuron sets of different concepts.

## 📌 Motivation

If "expert neurons" truly capture semantic meaning, we expect concepts with similar meanings (e.g., *cat* and *dog*) to share more expert neurons than unrelated concepts (e.g., *cat* and *hammer*). By measuring and visualizing this overlap, we can reconstruct the model's internal semantic structure.

---

## 🛠️ Analysis Scripts

### 1. Jaccard Similarity (`analyze_expert_overlap.py`)
This script computes the pairwise similarity between the expert sets of all concepts.

- **Input**: A collection of `expertise.csv` files (one per concept).
- **Metric**: Jaccard Similarity
  $$
  J(A, B) = \frac{|A \cap B|}{|A \cup B|}
  $$
  where $A$ and $B$ are the sets of expert neuron IDs for two concepts.
- **Output**: A symmetric similarity matrix (CSV) where cell $(i, j)$ is the Jaccard similarity between concept $i$ and concept $j$.

### 2. Shared Experts Statistics (`compute_shared_experts_stats.py`)
Calculates high-level statistics about neuron sharing across detailed groups.

- **Intersection**: Neurons that are experts for *every* concept in the group (e.g., "core animal experts").
- **Union**: All neurons that are expert for *at least one* concept.
- **Output**: JSON summary of unique vs. shared counts.

---

## 🎨 Visualization Methods

The `visualize_expert_overlap.py` script turns the similarity matrix into interpretable plots.

### 1. Heatmap
A simple $N \times N$ grid visualizing the similarity matrix directly.
- **Bright cells**: High overlap (related concepts).
- **Dark cells**: Low overlap (unrelated concepts).
- **Clusters**: Block-diagonal structures indicate semantic categories.

### 2. Network Graph Layouts
We model concepts as **nodes** and their similarity as **weighted edges**. The challenge is positioning these nodes in 2D space to reflect their high-dimensional similarity. We support three algorithms:

#### A. MDS (Multidimensional Scaling)
- **Logic**: Positions nodes such that their 2D Euclidean distance approximates their dissimilarity ($1 - J$).
- **Pros**: Preserves **global structure** (distances are meaningful).
- **Cons**: Can result in cluttered local clusters.

#### B. Spring (Force-Directed)
- **Logic**: Physics simulation. Edges act as springs pulling similar nodes together; all nodes repel each other (like magnets) to prevent overlap.
- **Pros**: Excellent **local segregation** of clusters.
- **Cons**: Random initialization; may distort global distances.

#### C. Hybrid Layout (⭐ Recommended)
combines the best of both worlds.
1.  **Stage 1 (MDS)**: Compute initial positions using MDS to establish the correct global topology.
2.  **Stage 2 (Spring)**: Run a short physics simulation starting from the MDS positions to refine local clusters and reduce overlap.

---

## 🚀 Usage

### Running via Pipeline
The easiest way is to use `run_pipeline.py` with the `expert_overlap` task.

```bash
# Standard analysis with Hybrid layout
python run_pipeline.py task=expert_overlap \
    task.ap_threshold=0.5 \
    task.visualization.layout_mode=hybrid
```

### Visualizing Existing Data
If you already have a similarity matrix and just want to tweak the plot:

```bash
python run_pipeline.py task=visualize_overlap \
    task.csv_path=results/.../jaccard_similarity.csv \
    task.visualization.layout_mode=spring \
    task.visualization.low_threshold=0.3
```

## 📊 Output Files

| File | Description |
|------|-------------|
| `jaccard_similarity.csv` | Raw $N \times N$ similarity matrix. |
| `jaccard_heatmap.png` | Visual heatmap of the matrix. |
| `graph_threshold_0.2_hybrid.png` | Network graph (e.g., using Hybrid layout). |
| `analysis_metadata.json` | Run configuration and top-10 similarity pairs. |