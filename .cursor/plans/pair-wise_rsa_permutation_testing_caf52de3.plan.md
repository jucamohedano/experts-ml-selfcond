---
name: Pair-wise RSA Permutation Testing
overview: Replace current layer-wise significance testing with pair-wise (layer-region) permutation testing, add GPU acceleration, subject selection, Manhattan plot visualization, and ranked table output.
todos: []
---

# Pair-wise RSA Permutation Testing Implementation

## Overview

Replace the current statistical significance testing (analytical, t-test, layer-wise permutation) with a new pair-wise permutation testing approach that:

1. Tests each (layer, region) pair individually
2. Uses GPU acceleration for computational efficiency
3. Supports per-subject testing with configurable subject selection
4. Provides Manhattan-style scatter plot visualization
5. Outputs ranked table of significant pairs

## Architecture Changes

### Data Flow

```
Brain RDMs (per subject) → Filter by subject selection → 
Pair-wise permutation test (GPU-accelerated) → 
P-value calculation → FDR correction → 
Top-K extraction → Visualization + Table output
```

## Implementation Steps

### 1. Update Hydra Configuration (`conf/task/rsa.yaml`)

- Remove `significance_method`, `n_permutations`, `permutation_n_samples`
- Add new parameters:
  ```yaml
  # Pair-wise permutation testing
  run_permutation_test: true
  n_permutations: 1000
  p_value_threshold: 0.01  # For top-K extraction
  use_gpu: true
  subject_selection: "0"  # "0" for first subject, "all" for all, or specific index
  
  # Visualization
  manhattan_plot: true
  ranked_table_top_n: 100  # Number of top pairs to show in table
  ```


### 2. Remove Old Significance Testing Functions (`selfcond/stats.py`)

- Remove `run_significance_analytical()`
- Remove `run_significance_ttest()`
- Remove `run_significance_permutation()` (old layer-wise version)
- Remove `plot_significance_results()` (old version)
- Keep: `apply_fdr_correction()`, `rankdata_2d()`, `fast_spearman_matrix()`, `fast_pearson_matrix()`, `permute_rdm()`

### 3. Implement GPU-Accelerated Pair-wise Permutation Test (`selfcond/stats.py`)

**New function: `run_pairwise_permutation_test()`**

- **Inputs:**
                                - `brain_rdms`: Dict[subject_id, Dict[region_id, RDM]] (filtered by subject_selection)
                                - `model_rdms`: Dict[layer_name, RDM]
                                - `correlation_method`: "spearman" or "pearson"
                                - `n_permutations`: int (default 1000)
                                - `use_gpu`: bool
                                - `seed`: int

- **Process (Permutation-Loop-First Algorithm):**

**CRITICAL:** Loop over permutations (1000), NOT over pairs (~288K). This reduces Python overhead from 288M iterations to just 1000.

  ```
  Step 1: Pre-compute ALL observed correlations (one-time batch operation)
  ========================================================================
  - Extract upper triangles: brain_vecs (N_brain, 1770), model_vecs (N_layers, 1770)
  - Compute ranks (for Spearman): brain_ranks, model_ranks
  - Batch correlation via matrix multiplication:
      observed_matrix = brain_normed @ model_normed.T  # Shape: (N_brain, N_layers)
  - Store observed_matrix on GPU
  
  Step 2: Initialize count matrix
  ================================
  - count_exceeds = zeros(N_brain, N_layers)  # Tracks how many nulls >= observed
  
  Step 3: Loop over PERMUTATIONS (1000 iterations, not 288K)
  ===========================================================
  for perm_idx in range(n_permutations):
      # Shuffle word indices ONCE for this permutation
      shuffle_idx = random_permutation(60)
      
      # Permute ALL brain RDMs at once (shuffle rows/columns of the 60x60 RDMs)
      # This is equivalent to re-extracting upper triangles after permutation
      permuted_brain_vecs = apply_permutation_to_upper_triangles(brain_vecs, shuffle_idx)
      
      # Compute ALL correlations in one GPU batch operation
      null_matrix = permuted_brain_normed @ model_normed.T  # Shape: (N_brain, N_layers)
      
      # Count where null >= observed (element-wise comparison, all 288K pairs at once)
      count_exceeds += (null_matrix >= observed_matrix).float()
  
  Step 4: Compute p-values
  ========================
  p_values = (count_exceeds + 1) / (n_permutations + 1)  # Shape: (N_brain, N_layers)
  
  Step 5: Apply FDR correction
  =============================
  p_values_flat = p_values.flatten()
  p_values_fdr = apply_fdr_correction(p_values_flat)
  ```

- **Return:** DataFrame with columns: `subject`, `region`, `layer`, `rsa_correlation`, `p_value`, `p_value_fdr`, `significant`

- **GPU Memory Estimation (RTX 3060 = 6GB):**
        - observed_matrix: 700 regions × 48 layers × 4 bytes = ~135 KB per subject
        - brain_vecs: 700 × 1770 × 4 bytes = ~5 MB per subject
        - brain_ranks (Spearman): 700 × 1770 × 4 bytes = ~5 MB per subject
        - model_vecs: 48 × 1770 × 4 bytes = ~340 KB
        - model_ranks: 48 × 1770 × 4 bytes = ~340 KB
        - count_exceeds: same as observed_matrix = ~135 KB
        - null_matrix (per iteration): 700 × 48 × 4 bytes = ~135 KB
        - **Total per subject: ~15-20 MB → Well within 6GB capacity**
        - Note: Processing 9 subjects simultaneously would use ~180 MB, still safe

- **Helper function: `apply_permutation_to_upper_triangles()`**
                                - Given a permutation index, efficiently reorder the upper triangle elements
                                - This avoids reconstructing 60x60 RDMs; instead, precompute index mapping

### 4. Implement Manhattan Plot (`selfcond/stats.py`)

**New function: `plot_manhattan_rsa()`**

- **Input:** DataFrame from permutation test (filtered by p-value threshold)
- **Visualization:**
                                - X-axis: Layer number (0 to N)
                                - Y-axis: Brain region index (sorted by peak layer or region ID)
                                - Scatter plot: Only plot points where `p_value_fdr < threshold`
                                - Color: RSA correlation value (colormap: RdBu_r)
                                - Size: `-log10(p_value_fdr)` (larger = more significant)
- **Output:** Save to `rsa_manhattan_plot.png`

### 5. Implement Ranked Table Output (`selfcond/stats.py`)

**New function: `generate_ranked_table()`**

- **Input:** DataFrame from permutation test
- **Process:**
                                - Sort by RSA correlation (descending) or p-value (ascending)
                                - Filter by significance threshold
                                - Extract top N pairs
                                - Format as table with columns: Rank, Region ID, Best Layer, RSA Score, p-value, p-value (FDR)
- **Output:** Save to `rsa_ranked_pairs.csv`

### 6. Update RSA Main Script (`scripts/compute_rsa.py`)

**Changes to `run_rsa()`:**

- Remove old significance testing block (lines 292-347)
- Add subject filtering logic:
  ```python
  if subject_selection == "all":
      subjects_to_test = list(brain_rdms.keys())
  elif subject_selection.isdigit():
      subject_idx = int(subject_selection)
      subjects_to_test = [list(brain_rdms.keys())[subject_idx]]
  else:
      raise ValueError(f"Invalid subject_selection: {subject_selection}")
  
  filtered_brain_rdms = {sid: brain_rdms[sid] for sid in subjects_to_test}
  ```

- Replace significance testing with:
  ```python
  if run_permutation_test:
      pairwise_results = run_pairwise_permutation_test(
          brain_rdms=filtered_brain_rdms,
          model_rdms=model_rdms,
          correlation_method=correlation_method,
          n_permutations=n_permutations,
          use_gpu=use_gpu,
      )
      # Generate visualizations
      plot_manhattan_rsa(pairwise_results, output_dir, p_value_threshold, ap_threshold)
      generate_ranked_table(pairwise_results, output_dir, ranked_table_top_n)
  ```


**Update `main()`:**

- Read new config parameters: `run_permutation_test`, `n_permutations`, `p_value_threshold`, `use_gpu`, `subject_selection`, `ranked_table_top_n`

### 7. Dependencies

- Add `torch` (PyTorch) for GPU operations if not already present
- Consider `cupy` as alternative (optional, for pure NumPy-like GPU operations)

## Key Design Decisions

1. **Permutation-Loop-First (CRITICAL):** Loop over permutations (1000), NOT over pairs (~288K). This reduces Python overhead from 288 million iterations to just 1000, making the computation tractable.
2. **Permutation Strategy:** Permute brain RDM (shuffle word labels) while keeping model RDM fixed, as suggested in the methodology
3. **GPU Acceleration:** Use PyTorch tensors for batch matrix multiplication; compute all pairs in one GPU operation per permutation
4. **Subject Handling:** Test each subject separately, but allow filtering to single subject for initial testing
5. **FDR Correction:** Apply across all pairs (all subjects × all regions × all layers)
6. **Default Behavior:** Process first subject only (`subject_selection: "0"`) to save computation time

## Files to Modify

1. `conf/task/rsa.yaml` - Update configuration
2. `selfcond/stats.py` - Remove old functions, add new pair-wise testing and visualization
3. `scripts/compute_rsa.py` - Update main RSA function to use new testing approach

## Testing Considerations

- Start with single subject (`subject_selection: "0"`) to validate implementation
- Verify GPU memory usage doesn't exceed RTX 3060 capacity (6GB) - estimated ~20MB per subject, safe
- Test with small number of permutations (e.g., 100) first for debugging
- Ensure reproducibility with random seed