# Subspace Gaze & Steering Validation

This document details the methodology and usage for Subspace Gaze, Masked Subspace Gaze, and Causal Steering Validation implemented in this project.

## 1. Subspace Gaze

**Goal**: To study the geometry of the model's latent space by extracting "concept vectors" and analyzing their relationships.

**Methodology**:
1.  **Data Collection**: We use cached activations (responses) from the model for a set of concepts (e.g., 60 concepts).
2.  **Concept Extraction**: For each concept and each layer, we extract a concept vector $\mathbf{v}_{concept}$ and its maximum activation using one of three methods.

### Extraction Methods

#### DiffMean Method (`method=diff_mean`)
The simplest approach - computes the direction that separates positive and negative sample means.

-   Compute mean activations: $\mu_{pos}$ and $\mu_{neg}$
-   Concept vector: $\mathbf{v} = \frac{\mu_{pos} - \mu_{neg}}{||\mu_{pos} - \mu_{neg}||}$
-   **Max Activation**: $\text{max\_act} = \max(\mathbf{X}_{pos} \cdot \mathbf{v})$

**Pros**: Fast, deterministic, works well when classes are well-separated.
**Cons**: Only captures the direction between class centers, ignores within-class variance.

#### PCA Method (`method=pca`)
Finds the direction of maximum variance within positive samples only.

-   Center positive samples: $\mathbf{X}_{centered} = \mathbf{X}_{pos} - \mu_{pos}$
-   Concept vector: First principal component of $\mathbf{X}_{centered}$
-   **Max Activation**: $\text{max\_act} = \max(\mathbf{X}_{pos} \cdot \mathbf{v})$

**Pros**: Captures the dominant direction of variation in positive samples.
**Cons**: May capture irrelevant variance (e.g., sentence length) instead of concept-specific direction.

#### LAT Method (`method=lat`)
**Linear Artificial Tomography** (Zou et al., 2023) - isolates concept directions by analyzing normalized difference vectors between positive and negative samples.

**Algorithm**:
1.  **Create Pairs**: Sample random pairs $(x^+, x^-)$ where $x^+ \in N_c^+$ and $x^- \in N_c^-$
2.  **Compute Differences**: $\delta_i = \mathbf{h}(x_i^+) - \mathbf{h}(x_i^-)$
3.  **Normalize**: $\hat{\delta}_i = \frac{\delta_i}{||\delta_i||}$
4.  **PCA on Differences**: Fit PCA on $\{\hat{\delta}_1, \hat{\delta}_2, ...\}$
5.  **Concept Vector**: First principal component (PC1)
6.  **Max Activation**: $\text{max\_act} = \max(\mathbf{X}_{pos} \cdot \mathbf{v})$

**Why LAT Works**:
-   By taking differences between positive and negative samples, shared features (syntax, common words, sentence structure) are **cancelled out**.
-   Normalizing each difference ensures that all pairs contribute equally, regardless of their magnitude.
-   PCA on normalized differences finds the direction that best explains the **concept-specific** variation.

**Pros**: Isolates concept-specific direction, cancels out confounds, often better for steering.
**Cons**: Slower (requires sampling pairs), stochastic (depends on random pairs).

### Method Comparison

| Method | Input | What PC1 Captures | Best For |
|--------|-------|-------------------|----------|
| DiffMean | $\mu_{pos} - \mu_{neg}$ | Direction between class centers | Well-separated concepts |
| PCA | Positive samples only | Max variance in positive class | Homogeneous positive sets |
| LAT | Normalized $(pos - neg)$ pairs | Concept-specific direction | Steering, noisy data |

3.  **Storage Format**: Vectors are stored as dictionaries: `{'vector': ndarray, 'max_act': float}`.
4.  **Geometric Analysis**: We compute the **Cosine Similarity Matrix** between all pairs of concept vectors in a given layer.
    -   **Heatmaps**: We visualize this matrix to see clusters of related concepts (e.g., animals, furniture).

### UMAP Visualization

In addition to heatmaps, we generate **UMAP plots** to visualize the concept vectors in 2D space.

**What is UMAP?**
UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that projects high-dimensional vectors (768D for GPT-2) to 2D while preserving local neighborhood structure. Similar concepts should cluster together.

**Semantic Groups**:
The 60 standard concepts are organized into 10 semantic groups:

| Group | Concepts |
|-------|----------|
| **tool** | chisel, hammer, key, knife, pliers, saw, screwdriver, spoon |
| **vehicle** | airplane, bicycle, car, train, truck |
| **furniture** | bed, chair, closet, desk, dresser, refrigerator, table |
| **building** | apartment, arch, barn, chimney, church, door, house, igloo, window |
| **clothing** | coat, dress, pants, shirt, skirt |
| **object** | bell, bottle, cup, glass, telephone, watch |
| **animal** | bear, cat, cow, dog, horse |
| **insect** | ant, bee, beetle, butterfly, fly |
| **plant** | carrot, celery, corn, lettuce, tomato |
| **body_part** | arm, eye, foot, hand, leg |

**Interpretation**:
- **Clusters** indicate semantically related concepts share similar representations
- **Separation** between groups suggests the model distinguishes between categories
- **LAT vs DiffMean**: LAT should show cleaner separation (fewer spurious similarities)

## 2. Masked Subspace Gaze (Expert Subspace)

**Goal**: To refine the analysis by focusing only on the "expert neurons" that are relevant to each concept.

**Methodology**:
1.  **Expert Identification**: We use the **Average Precision (AP)** scores from the `expertise` task.
2.  **Selection**: Neurons with $AP > \text{threshold}$ (default 0.5) are selected as experts for a given concept.
3.  **Masking**:
    -   We create a binary mask $\mathbf{m}$ where $1$ indicates an expert neuron.
    -   We apply this mask to the activations: $\mathbf{x}_{masked} = \mathbf{x} \odot \mathbf{m}$.
4.  **Analysis**: We perform the standard Subspace Gaze extraction and geometric analysis on these *masked* activations.

**Comparison**:
-   **Global Subspace Gaze**: Captures the global direction in the entire activation space.
-   **Masked Subspace Gaze**: Captures the direction within the specific subspace of neurons that "care" about the concept. This often results in sharper, more distinct clusters in the heatmaps.

## 3. Kill Experts Subspace Gaze (Inverse Masking)

**Goal**: To perform an ablation study by analyzing the latent geometry when expert neurons are suppressed.

**Methodology**:
1.  **Expert Identification**: Same as Masked LAT (using AP scores).
2.  **Inverse Masking**:
    -   We create an inverse binary mask: $\mathbf{m}_{kill} = 1 - \mathbf{m}_{expert}$.
    -   This mask has $0$ for expert neurons and $1$ for non-experts.
3.  **Projection**: We apply this mask to suppress experts: $\mathbf{x}_{killed} = \mathbf{x} \odot \mathbf{m}_{kill}$.
4.  **Analysis**: We perform standard Subspace Gaze extraction on these *killed* activations.

**Purpose**:
-   **Ablation Study**: By removing experts, we can see if the concept vector disappears or changes dramatically.
-   **Steering Validation**: When steering with a "killed" vector, the concept should be weakened or absent in generation, confirming that the experts are indeed responsible for the concept.

## 4. Steering Validation

**Goal**: To causally validate that the extracted vectors actually represent the target concept in the model.

**Methodology**:
1.  **Vector Loading**: Load the concept vector and its maximum activation from the pickle file.
2.  **Layer Validation**: Automatically validate that the steering layer matches the layer from which the vector was extracted.
    -   A warning is issued if there's a mismatch (e.g., steering on `transformer.h.6` with a vector from `transformer.h.6.attn.c_proj`).
3.  **Injection**: We use a PyTorch forward hook to intervene during the model's forward pass.
4.  **Steering with Max Activation Scaling**: We add the concept vector to the hidden states of a target layer, scaled by both a coefficient $\alpha$ and the maximum activation:
    $$ \mathbf{h}' = \mathbf{h} + \alpha \cdot \text{max\_act} \cdot \mathbf{v}_{concept} $$
    -   This scaling matches the implementation in `axbench` and ensures the steering magnitude is calibrated to the concept's natural activation strength.
5.  **Generation**: We generate text with this intervention across a range of coefficients.
    -   Temperature is set to 0.9 for diverse sampling.
6.  **Scoring**: We evaluate how well the generated text relates to the concept using **Sentence Embedding Similarity**.
    
    **Sentence Embedding Scorer**:
    -   Uses `sentence-transformers/all-MiniLM-L6-v2` (lightweight, 80MB model).
    -   Embeds both the generated text and the concept word.
    -   Computes **cosine similarity** between embeddings.
    -   Returns a score in range [-1, 1], where higher = more semantically related.
    -   **Advantage**: Captures semantic meaning (e.g., "train" is similar to "railway", "locomotive", "tracks") without requiring exact word matches or LLM API calls.
    
7.  **Analysis**: Plot steering factor vs. concept score to visualize the relationship.
    -   **Killed Vector Test**: Steering with a "killed" vector should weaken or remove the concept from generation.

## 5. Usage

### Running Subspace Gaze Analysis (Global, Masked & Killed)

This script computes concept vectors and generates heatmaps for Global, Masked, and Killed Subspace Gaze.

```bash
# Using DiffMean (default, fast)
python run_pipeline.py task=subspace_gaze \
    task.activations_path=results/compute_responses/.../responses-1024 \
    task.expertise_path=results/compute_responses/.../responses-1024 \
    task.ap_threshold=0.8 \
    task.layer=transformer.h.6 \
    task.method=diff_mean

# Using LAT (recommended for steering)
python run_pipeline.py task=subspace_gaze \
    task.activations_path=results/compute_responses/.../responses-1024 \
    task.expertise_path=results/compute_responses/.../responses-1024 \
    task.ap_threshold=0.8 \
    task.layer=transformer.h.6 \
    task.method=lat
```

**Configuration (`conf/task/subspace_gaze.yaml`)**:
-   `activations_path`: Path to cached activations (concept subfolders with `responses/`).
-   `expertise_path`: Path to the expertise outputs (concept subfolders with `expertise/`).
-   `ap_threshold`: Threshold for expert selection (default: 0.5).
-   `layer`: Specific layer to analyze (or `"all"` for every layer).
-   `method`: Extraction method - one of:
    -   `diff_mean` - Difference of means (fast, deterministic)
    -   `pca` - PCA on positive samples only
    -   `lat` - Linear Artificial Tomography (recommended for steering)

**Output**:
-   `vectors_{layer}_{suffix}.pkl`: Pickle files containing `{concept: {'vector': ndarray, 'max_act': float}}`.
-   `similarity_matrix_{layer}_{suffix}.csv`: Cosine similarity matrices.
-   `heatmap_{layer}_{suffix}.png`: Cosine similarity heatmaps.
-   `umap_{layer}_{suffix}.png`: UMAP visualization colored by semantic group.
-   `umap_embeddings_{layer}_{suffix}.csv`: UMAP 2D coordinates with concept groupings.

**Note**: UMAP requires the `umap-learn` package. Install with: `pip install umap-learn`

### Running Steering Validation

This script loads a specific concept vector and steers the model generation across multiple coefficients.

```bash
python run_pipeline.py task=steering_validation \
    task.vector_path=results/subspace_gaze/.../vectors_transformer.h.6_global.pkl \
    task.concept=train \
    task.layer=transformer.h.6 \
    task.coeff_start=-5.0 \
    task.coeff_end=60.0 \
    task.coeff_steps=20 \
    task.prompt="I work on" \
    task.max_new_tokens=50 \
    task.num_samples=10
```

**Configuration (`conf/task/steering_validation.yaml`)**:
-   `vector_path`: Path to the `.pkl` file containing the vector (output from Subspace Gaze analysis).
-   `concept`: Name of the concept to steer (must match key in pkl file).
-   `layer`: Target layer name (must match model module). **Important**: Should match the layer from which the vector was extracted.
-   `coeff_start`, `coeff_end`, `coeff_steps`: Range and number of steering coefficients to test.
-   `prompt`: Initial text prompt for generation.
-   `max_new_tokens`: Maximum tokens to generate per sample.
-   `num_samples`: Number of samples to generate per coefficient for scoring.
-   `seed`: Random seed for reproducibility.

**Output**:
-   `steering_results_{concept}_{layer}.csv`: CSV with columns: `factor`, `score`, `texts`.
-   `steering_plot_{concept}_{layer}.png`: Plot of steering factor vs. concept score (embedding similarity).

**Important Notes**:
1.  **Layer Matching**: For best results, the `task.layer` should match the layer from which the vector was extracted. The script will warn you if there's a mismatch.
2.  **Residual Stream vs. Submodules**: Steering the residual stream (e.g., `transformer.h.6`) is generally more effective than steering submodules (e.g., `transformer.h.6.attn.c_proj`).
3.  **Max Activation Scaling**: The effective steering magnitude is `coeff × max_act × vector`. If `max_act` is large (e.g., 50), you may need smaller coefficients than expected.
4.  **Scoring Method**: The sentence embedding scorer is semantic-based. If you prefer exact word matching, you can modify the `SentenceEmbeddingScorer` class in `scripts/steering_validation.py`.

