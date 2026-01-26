# Brain Data Processing, RDM Computation, and RSA Analysis

This document describes the implementation of Steps 4 and 5 of the JuanProject analysis plan:
- **Step 4**: Brain RDMs per region
- **Step 5**: Representational Similarity Analysis (RSA)
- **Step 6**: Permutation testing and multiple comparison correction

---

## 1. Brain Data Source: Mitchell 2008 Dataset

### 1.1 Dataset Overview

The brain data comes from the Mitchell et al. (2008) fMRI study. Files are stored in `brain_data/` as MATLAB `.mat` files:

| File | Subject |
|------|---------|
| `data-science-P1.mat` | Subject 1 |
| `data-science-P2.mat` | Subject 2 |
| ... | ... |
| `data-science-P9.mat` | Subject 9 |

### 1.2 File Structure

Each `.mat` file contains three main components:

```
mat_data['data']   # (360, 1) - Trial responses
mat_data['meta']   # Voxel metadata
mat_data['info']   # (1, 360) - Trial metadata
```

**Trial Data (`data`)**:
- Shape: `(360, 1)` where each cell contains a `(1, V)` voxel response vector
- 360 trials = 60 words x 6 presentations each
- V = number of voxels (varies by subject: ~19,750 to ~21,764)

**Voxel Coordinates (`meta`)**:
- `meta['colToCoord']`: Shape `(V, 3)` with x, y, z coordinates for each voxel
- Coordinates are in subject-specific brain space (not normalized across subjects)

**Trial Metadata (`info`)**:
- `info[0, i]['word']`: Word label for trial i (e.g., "bear", "hammer")
- `info[0, i]['cond']`: Semantic category (e.g., "animal", "tool")
- `info[0, i]['word_number']`: Word ID (1-60)
- `info[0, i]['cond_number']`: Category ID (1-12)

### 1.3 The 60 Words

The same 60 concrete nouns are used across all subjects and match the LM concepts:

```
airplane, ant, apartment, arch, arm, barn, bear, bed, bee, beetle,
bell, bicycle, bottle, butterfly, car, carrot, cat, celery, chair,
chimney, chisel, church, closet, coat, corn, cow, cup, desk, dog,
door, dress, dresser, eye, fly, foot, glass, hammer, hand, horse,
house, igloo, key, knife, leg, lettuce, pants, pliers, refrigerator,
saw, screwdriver, shirt, skirt, spoon, table, telephone, tomato,
train, truck, watch, window
```

---

## 2. Brain Data Loading

### 2.1 Implementation: `selfcond/brain_data.py`

```python
def load_brain_data(mat_paths: Path | List[Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load Mitchell 2008 brain data and average trials per word.
    
    Returns:
        word_activations: [60, V] averaged responses per word
        voxel_coords: [V, 3] x, y, z coordinates
        word_labels: List of 60 words (alphabetical)
    """
```

### 2.2 Trial Averaging

Raw data contains 360 individual trials. We average the 6 presentations of each word to obtain a stable response estimate.

**Formal definition**:

Let r^(w,t) be the voxel response vector for word w on trial t (where t in {1, ..., 6}).

The averaged word activation is computed as:

r_bar^(w) = (1/6) * sum_{t=1}^{6} r^(w,t)

**Output**: `word_activations` of shape `(60, V)` - one averaged response vector per word.

### 2.3 Subject Differences

Each subject has different voxel counts due to individual brain anatomy:

| Subject | Voxels |
|---------|--------|
| P1 | 21,764 |
| P2 | 21,253 |
| P3 | 20,651 |
| P4 | 20,395 |
| P5 | 20,601 |
| P6 | 19,919 |
| P7 | 19,750 |
| P8 | 20,082 |
| P9 | 21,344 |

**Important**: Because voxel counts and coordinates differ across subjects, we process each subject independently and **do not average brain data across subjects**.

---

## 3. Voxel Division: 3D Grid Tessellation

### 3.1 Approach

Rather than using a voxel-centered searchlight (computationally expensive), we tessellate the brain volume into a regular 3D grid.

### 3.2 Implementation: `create_grid_regions()`

```python
def create_grid_regions(coords: np.ndarray, 
                        grid_shape: Tuple[int, int, int] = (11, 11, 9)
                       ) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Tessellate brain into 3D grid regions.
    
    Args:
        coords: [V, 3] voxel coordinates
        grid_shape: Number of bins along (x, y, z) axes
        
    Returns:
        regions: Dict mapping (xi, yi, zi) -> list of voxel indices
    """
```

### 3.3 Algorithm

1. **Compute bin edges** for each axis using `np.linspace`:
   ```
   x_bins = linspace(x_min, x_max, nx + 1)  # 12 edges for 11 bins
   y_bins = linspace(y_min, y_max, ny + 1)
   z_bins = linspace(z_min, z_max, nz + 1)
   ```

2. **Assign each voxel** to a grid cell using `np.digitize`:
   ```
   x_idx[i] = bin index for coords[i, 0]
   y_idx[i] = bin index for coords[i, 1]
   z_idx[i] = bin index for coords[i, 2]
   ```

3. **Group voxels** by their (xi, yi, zi) cell:
   ```
   regions[(xi, yi, zi)] = [list of voxel indices in this cell]
   ```

### 3.4 Grid Parameters

- **Grid shape**: Configurable via `grid_shape` parameter (e.g., `(2, 10, 5)` or `(11, 11, 9)`)
- **Example**: `(2, 10, 5)` = 100 possible cells, `(11, 11, 9)` = 1,089 possible cells
- **Actual regions**: Non-empty cells vary by subject (brain doesn't fill entire bounding box)
- **Regions kept**: Cells with >= `min_voxels` (default: 5)

### 3.5 Example Region Counts

With grid `(2, 10, 5)`:

| Subject | Total Regions | Kept (>=5 voxels) |
|---------|---------------|-------------------|
| P1 | ~95 | ~90 |
| P2 | ~98 | ~92 |
| ... | ... | ... |

With grid `(11, 11, 9)`:

| Subject | Total Regions | Kept (>=5 voxels) |
|---------|---------------|-------------------|
| P1 | 781 | 694 |
| P2 | 863 | 775 |
| P3 | 768 | 681 |
| ... | ... | ... |

---

## 4. Brain RDM Computation

### 4.1 Implementation: `scripts/compute_brain_rdm.py`

```python
def compute_brain_rdms(word_activations: np.ndarray,
                       regions: Dict[Tuple[int,int,int], List[int]],
                       min_voxels: int = 5
                      ) -> Dict[Tuple[int,int,int], np.ndarray]:
    """
    Compute a 60x60 RDM for each brain region.
    """
```

### 4.2 RDM Formula

For each region r with voxel indices `[v1, v2, ..., vk]`:

1. **Extract region activations**:
   ```
   region_acts = word_activations[:, [v1, v2, ..., vk]]  # Shape: (60, k)
   ```

2. **Compute Pearson correlation** between all word pairs:
   ```
   corr_matrix = np.corrcoef(region_acts)  # Shape: (60, 60)
   ```

3. **Convert to dissimilarity**:
   ```
   RDM[i, j] = 1 - corr_matrix[i, j]
   ```

**Properties**:
- RDM diagonal = 0 (word is identical to itself)
- RDM range: [0, 2] where 0 = identical, 2 = perfectly anti-correlated
- RDM is symmetric

### 4.3 Output Structure

```python
brain_rdms = {
    "data-science-P1": {
        (1, 6, 4): np.array([60, 60]),  # Region at grid cell (1,6,4)
        (2, 6, 4): np.array([60, 60]),
        ...
    },
    "data-science-P2": {...},
    ...
}
```

### 4.4 Running the Task

```bash
python run_pipeline.py task=brain_rdm
```

**Outputs** (in `results/brain_rdm/.../`):
- `brain_rdms.pkl`: All RDMs `{subject_id: {region_id: RDM}}`
- `word_labels.txt`: 60 words in order
- `subject_info.json`: Per-subject metadata
- `brain_rdm_metadata.json`: Run metadata
- `sample_rdms/`: Visualization of top 10 regions

---

## 5. RSA Analysis (Step 5)

### 5.1 Goal

Compare the representational geometry of:
- **Model RDMs**: From LM expert-based word representations (Step 3)
- **Brain RDMs**: From fMRI regional activations (Step 4)

### 5.2 Inputs

**Model RDMs** (from `word_features` task):
- Location: `results/word_features/.../rdms/`
- Files: `{layer_name}_rdm.npy` (e.g., `transformer_h_0_mlp_c_fc_rdm.npy`)
- Structure: One 60x60 RDM per layer (48 layers for GPT-2)
- Layer names use underscore format: `transformer_h_0_mlp_c_fc`, `transformer_h_0_mlp_c_proj`, `transformer_h_0_attn_c_attn`, `transformer_h_0_attn_c_proj`

**Brain RDMs** (from `brain_rdm` task):
- Location: `results/brain_rdm/.../brain_rdms.pkl`
- Structure: `{subject_id: {region_id: 60x60 RDM}}`
- ~700 regions per subject, 9 subjects

### 5.3 RSA Procedure

For each **subject** s, **brain region** r, and **model layer** l:

1. **Extract upper triangle** of both RDMs (excluding diagonal):
   ```python
   triu_idx = np.triu_indices(60, k=1)
   brain_vec = brain_rdm[triu_idx]   # 1770 values
   model_vec = model_rdm[triu_idx]   # 1770 values
   ```

2. **Compute Spearman correlation** (rank-based, standard for RSA):
   ```python
   rsa_correlation = spearmanr(brain_vec, model_vec).correlation
   ```

3. **Result**: One correlation value per (subject, region, layer) triple

### 5.4 Statistical Aggregation

Results are aggregated in three ways:

**By layer** (mean across subjects and regions):
```python
by_layer = df.groupby("layer")["rsa_correlation"].agg(["mean", "std", "count"])
```

**By region** (mean across subjects and layers):
```python
by_region = df.groupby("region")["rsa_correlation"].agg(["mean", "std", "count"])
```

**By subject** (mean across regions and layers):
```python
by_subject = df.groupby("subject")["rsa_correlation"].agg(["mean", "std", "count"])
```

### 5.5 Outputs

**Data files** (in `results/rsa/.../`):
- `rsa_results.csv`: Raw results (subject, region, layer, correlation)
- `rsa_by_layer.csv`: Aggregated by layer (mean, std, count)
- `rsa_by_region.csv`: Aggregated by region
- `rsa_by_subject.csv`: Aggregated by subject
- `rsa_metadata.json`: Run metadata and top results

**Visualizations**:
- `rsa_by_layer.png`: Bar plot of RSA correlation per layer (mean +/- 95% CI)
- `rsa_layer_type_comparison.png`: Line plot comparing attention vs MLP layers
- `rsa_heatmap_top_regions.png`: Heatmap of top 20 regions x all layers

### 5.6 Running the Task

```bash
python run_pipeline.py task=rsa
```

Or with custom paths:
```bash
python run_pipeline.py task=rsa \
  task.brain_rdms_path=results/brain_rdm/.../brain_rdms.pkl \
  task.model_rdms_path=results/word_features/.../
```

### 5.7 Configuration

```yaml
# conf/task/rsa.yaml
name: rsa
run_tag: default

# Input paths
brain_rdms_path: results/brain_rdm/.../brain_rdms.pkl
model_rdms_path: results/word_features/.../

# RSA parameters
correlation_method: spearman  # spearman (rank-based) or pearson
```

### 5.8 Expected Insights

1. **Layer progression**: Do deeper LM layers correlate better with brain representations?
2. **Attention vs MLP**: Which component type better matches brain activity?
3. **Regional specificity**: Which brain regions show highest LM-brain alignment?
4. **Expert selection effect**: Does the AP threshold affect RSA results?

---

## 6. Permutation Testing for Statistical Significance

### 6.1 Motivation

RSA correlations can be spuriously high due to autocorrelation in both brain and model RDMs. To assess statistical significance, we use **pair-wise permutation testing** that respects the structure of the data.

### 6.2 Implementation: `selfcond/stats.py`

```python
def run_pairwise_permutation_test(
    brain_rdms: Dict[str, Dict[Tuple[int, int, int], np.ndarray]],
    model_rdms: Dict[str, np.ndarray],
    correlation_method: str = "spearman",
    n_permutations: int = 1000,
    use_gpu: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run pair-wise permutation test for all (brain region, model layer) pairs.
    """
```

### 6.3 Permutation Procedure

For each permutation iteration:

1. **Generate random word permutation**: Shuffle the 60 word indices
   ```python
   perm_idx = np.random.permutation(60)  # e.g., [34, 12, 0, 58, ...]
   ```

2. **Apply permutation to brain RDM upper triangles**:
   - The 60x60 brain RDM is represented as a 1770-element vector (upper triangle)
   - Permuting words means reordering both rows and columns of the RDM
   - We directly reorder the upper triangle elements based on the permutation

3. **Compute null correlations**:
   ```python
   null_corr = spearman(permuted_brain_vec, model_vec)
   ```

4. **Count exceedances**:
   ```python
   count_exceeds += (null_corr >= observed_corr)
   ```

5. **Compute p-value** (after all permutations):
   ```python
   p_value = (count_exceeds + 1) / (n_permutations + 1)
   ```

### 6.4 Efficient Upper Triangle Permutation

Rather than reconstructing 60x60 matrices for each permutation, we directly permute the 1770-element upper triangle vectors:

```python
def _apply_permutation_to_upper_triangle(brain_vecs, perm_idx, triu_i, triu_j, n_words=60):
    """
    Apply word permutation to upper triangle vectors without matrix reconstruction.
    
    Given permutation [3, 1, 0, 2, ...]:
    - Word 0 -> position 3
    - Word 1 -> position 1
    - etc.
    
    The RDM element (i, j) becomes element (perm[i], perm[j]).
    """
```

### 6.5 GPU Acceleration

When `use_gpu=True` and PyTorch/CUDA is available:

- Brain and model vectors are moved to GPU as tensors
- All correlations computed via matrix multiplication
- Permutation loop runs on GPU
- ~10-50x speedup over CPU for large experiments

```python
# GPU computation
observed_matrix = brain_normed @ model_normed.T  # (N_brain, N_layers)
for perm_i in range(n_permutations):
    null_matrix = permuted_brain_normed @ model_normed.T
    count_exceeds += (null_matrix >= observed_matrix).float()
```

### 6.6 Running Permutation Tests

```bash
python run_pipeline.py task=rsa \
    task.run_permutation_test=True \
    task.n_permutations=10000 \
    task.use_gpu=True
```

### 6.7 Outputs

Additional outputs when permutation testing is enabled:

- `rsa_permutation_results.csv`: Full results with p-values
- `rsa_manhattan_plot.png`: Visualization of significant pairs
- `rsa_ranked_pairs.csv`: Top pairs ranked by RSA correlation
- `rsa_best_layer_per_region.csv`: Best-matching layer for each region

---

## 7. Multiple Comparison Correction (FDR)

### 7.1 The Problem

With n_regions brain regions x n_layers model layers tests per subject, we face multiple comparison issues. 

**Examples**:
- Grid `(2, 10, 5)`: ~100 regions x 48 layers = ~4,800 tests
- Grid `(11, 11, 9)`: ~700 regions x 48 layers = ~33,600 tests

At alpha=0.05, we'd expect 5% false positives by chance.

### 7.2 FDR Correction Method

We apply **Benjamini-Hochberg FDR correction** using statsmodels:

```python
from statsmodels.stats.multitest import multipletests

def apply_fdr_correction(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    _, p_fdr, _, _ = multipletests(p_values, method="fdr_bh")
    return p_fdr
```

### 7.3 Current Status: FDR Not Yielding Significant Results

**Observation**: After FDR correction, very few or no pairs remain significant (p_fdr < 0.05).

**Possible reasons**:

1. **True effect size is small**: Brain-model RSA correlations are typically r ~ 0.05-0.15, which may require more statistical power to detect reliably.

2. **High number of comparisons**: With thousands of tests (e.g., 4,800 or 33,600), even moderate raw p-values become non-significant after correction.

3. **Permutation test may be too conservative**: The row/column permutation approach preserves certain RDM properties that may make the null distribution too similar to the observed distribution.

4. **Subject variability**: Individual differences in brain anatomy and function add noise.

### 7.4 Alternative Approaches (Not Yet Implemented)

1. **Reduce number of comparisons**:
   - Test only a priori hypothesized regions (e.g., language areas)
   - Aggregate layers into groups (early, middle, late)
   - Use cluster-based correction instead of voxel-wise

2. **Increase statistical power**:
   - More permutations (10,000+)
   - Average across subjects before testing
   - Use parametric tests with higher power

3. **Different null model**:
   - Phase randomization of RDMs
   - Shuffle only within semantic categories

### 7.5 Interpreting Results Without FDR

For exploratory analysis, we report:
- Raw p-values from permutation test
- Effect sizes (RSA correlations)
- Visualizations showing patterns across layers and regions

**Caution**: Results without FDR correction should be interpreted as exploratory, not confirmatory.

---

## 8. Visualization Outputs

### 8.1 Manhattan Plot

When permutation testing is enabled, a Manhattan-style plot shows significant pairs:

- **X-axis**: LLM layer number (0-47)
- **Y-axis**: Brain regions (sorted by peak layer)
- **Color**: RSA correlation strength
- **Size**: -log10(p_fdr) - larger dots = more significant

### 8.2 Layer Type Comparison

`rsa_layer_type_comparison.png` shows:
- Mean RSA correlation by layer number
- Separate lines for attention vs MLP components
- Helps identify whether attention or MLP better matches brain representations

### 8.3 Heatmaps

Two heatmaps are generated:
- `rsa_heatmap_top_regions.png`: Top 20 regions by mean RSA
- `rsa_heatmap_all_regions.png`: All regions (can be large)

---

## 9. Key Implementation Details

### 9.1 Per-Subject Processing

**Critical design decision**: Brain RDMs are computed and stored per-subject, NOT averaged across subjects.

```python
all_brain_rdms = {
    "data-science-P1": {region_id: RDM, ...},
    "data-science-P2": {region_id: RDM, ...},
    ...
}
```

**Rationale**:
- Different subjects have different voxel counts and coordinates
- Averaging RDMs across subjects would require spatial normalization
- RSA correlations should be computed per-subject, then aggregated

### 9.2 Subject Selection in RSA

The `subject_selection` parameter controls which subjects are analyzed:

```yaml
task.subject_selection: "0"    # Only first subject (data-science-P1)
task.subject_selection: "all"  # All 9 subjects
task.subject_selection: "3"    # Only subject at index 3 (P4)
```

### 9.3 Vectorized Correlation Computation

For efficiency, all RSA correlations are computed in a single matrix multiplication:

```python
# Pre-compute normalized vectors
brain_normed = (brain_ranks - mean) / norm  # (N_brain, 1770)
model_normed = (model_ranks - mean) / norm  # (N_layers, 1770)

# All correlations at once
corr_matrix = brain_normed @ model_normed.T  # (N_brain, N_layers)
```

This avoids Python loops over thousands of pairs.

---

## 10. Summary of Pipeline Steps

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Expert Units | `compute_expertise.py` | Activations | `expertise.csv` per concept |
| 2. Word Features | `word_features.py` | Expertise + Activations | Feature matrices |
| 3. Model RDMs | `word_features.py` | Feature matrices | 60x60 RDMs per layer |
| 4. Brain RDMs | `compute_brain_rdm.py` | `.mat` files | 60x60 RDMs per subject/region |
| 5. RSA | `compute_rsa.py` | Model + Brain RDMs | Correlation maps |
| 6. Permutation Test | `compute_rsa.py` | RSA correlations | p-values, rankings |
| 7. FDR Correction | `selfcond/stats.py` | Raw p-values | Corrected p-values |

---

## References

- Mitchell, T. M., et al. (2008). Predicting human brain activity associated with the meanings of nouns. *Science*, 320(5880), 1191-1195.
- Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis. *Frontiers in Systems Neuroscience*, 2, 4.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
