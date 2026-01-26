# RSA Experiment: Expert Neurons vs Full Embeddings

## Background: What are "Expert Neurons"?

Large language models (LLMs) contain millions of neurons. For any given concept (e.g., "airplane"), only a **small subset** of neurons reliably activate when that concept appears in text. We call these **"expert neurons"** for that concept.

### How we find them:
- Show the model many sentences containing "airplane" (positive) and sentences without it (negative)
- For each neuron, compute **Average Precision (AP)** — how well that neuron's activation distinguishes positive from negative
- Neurons with AP ≥ 0.5 (better than chance) are "experts" for that concept

### The intuition:
- **Experts** = concept-selective neurons (signal)
- **Non-experts** = neurons responding to syntax, other concepts, or noise

---

## The Experiment

### Central Question

> When representing 60 concrete concepts, do expert neurons encode **more brain-like** structure than using all neurons?

### Approach

1. **Build Representational Dissimilarity Matrices (RDMs):**
   - For LLM: How different are the 60 concepts based on expert neurons vs all neurons?
   - For brain: How different are the 60 concepts in human fMRI patterns?

2. **Compare via Representational Similarity Analysis (RSA):**
   - Spearman correlation between LLM RDM and Brain RDM
   - **Fisher-Z transformation** applied before t-test (correlations aren't normally distributed)
   - Higher correlation = more brain-aligned representation

3. **Statistical Test:**
   - **Within-subject design**: Same 9 subjects, 2 conditions (Expert RDM, Full RDM)
   - **Per brain region**: 87 regions tested
   - **Paired t-test**: Is mean(RSA_expert - RSA_full) ≠ 0 across subjects?
   - **FDR correction**: Benjamini-Hochberg applied across 87 tests

---

## What We're Testing

| Condition | Neurons Used | Hypothesis |
|-----------|--------------|------------|
| **Expert** | Only neurons selective for each concept | Cleaner signal → better brain match |
| **Full** | All ~3000 neurons per layer | Includes irrelevant neurons → noisier |

### Expected outcome if experts matter:
- RSA_expert > RSA_full consistently across subjects
- Significant paired t-tests in brain regions involved in semantic processing

---

## Statistical Details

### Per-Region Paired T-Test

For each of the 87 brain regions:

| Subject | RSA (Expert) | RSA (Full) | Difference |
|---------|--------------|------------|------------|
| P1 | 0.082 | 0.075 | +0.007 |
| P2 | 0.091 | 0.088 | +0.003 |
| ... | ... | ... | ... |
| P9 | 0.078 | 0.081 | -0.003 |

**Test statistic (on Fisher-Z transformed values):**
```
z = arctanh(rho)  # Fisher-Z transform
t = mean(z_diff) / (std(z_diff) / √n)
df = n - 1 = 8
```

### Effect Size

**Cohen's d** (paired samples):
```
d = mean(z_diff) / std(z_diff)
```
| d | Interpretation |
|---|----------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### Multiple Comparison Correction

With 87 tests at α = 0.05:
- **Expected false positives**: 0.05 × 87 ≈ 4.35
- **FDR correction**: `scipy.stats.false_discovery_control(p_values)`
- Reports both `sig_raw` (p < 0.05) and `sig_fdr` (FDR-corrected)

---

## Output Statistics

| Column | Description |
|--------|-------------|
| `mean_rho_a` | Mean RSA for Expert condition (raw Spearman ρ) |
| `std_rho_a` | Std dev for Expert condition |
| `mean_rho_b` | Mean RSA for Full condition |
| `std_rho_b` | Std dev for Full condition |
| `mean_diff` | Mean of paired differences |
| `std_diff` | Std dev of paired differences |
| `t_stat` | T-statistic (computed on Fisher-Z values) |
| `df` | Degrees of freedom (8) |
| `p_value` | Two-tailed p-value |
| `p_fdr` | FDR-corrected p-value |
| `cohens_d` | Effect size |
| `sig_raw` | Significant at raw p < 0.05 |
| `sig_fdr` | Significant after FDR correction |

---

## Visualizations

### 1. Paired Comparison Plot (`paired_comparison.png`)
Horizontal lines connecting Expert vs Full RSA for each brain region. Color indicates significance.

### 2. Effect Size Bar Plot (`effect_sizes.png`)
All 87 regions sorted by Cohen's d. Includes reference lines at d = ±0.2, ±0.5, ±0.8.

### 3. Diagonal Scatter Plot (`scatter_fdr.png`, `scatter_raw.png`)
- X-axis: Full model RSA
- Y-axis: Expert model RSA
- Diagonal line at y = x
- Points above diagonal = Expert wins
- Color indicates significance

### Color Scheme (IEEE-friendly)
| Significance | Color | Hex |
|--------------|-------|-----|
| FDR significant | Green | `#2ecc71` |
| Raw significant | Orange | `#f39c12` |
| Not significant | Gray | `#bdc3c7` |

---

## Data Sources

- **Brain data**: Mitchell et al. (2008) fMRI dataset
  - 9 subjects
  - 60 concrete nouns
  - 87 brain regions (tessellated voxel groups)
  
- **Model data**: GPT-2 layer activations
  - Expert neurons identified via Average Precision
  - RDMs computed from all 4 sub-layers (`attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`)

---

## Configuration

### Layer Filter Options

```yaml
# conf/task/rsa.yaml
task.layers: null                    # All 48 layers (averaged)
task.layers: [attn_c_proj]           # 12 attention output projections
task.layers: [mlp_c_fc]              # 12 MLP first layers
task.layers: [attn_c_proj, mlp_c_fc] # 24 layers combined
```

### FDR Control

```yaml
task.apply_fdr: true   # Apply FDR correction (default)
task.apply_fdr: false  # Raw p-values only
```

---

## Code Reference

- **Script**: `scripts/compute_rsa.py`
- **Config**: `conf/task/rsa.yaml`
- **Batch run**: `run_rsa_batch.sh`

### Run Example
```bash
python run_pipeline.py \
  task=rsa \
  model=Qwen3-30B-A3B-Instruct-2507_gpt2 \
  task.brain_rdms_path=results/brain_rdm/.../brain_rdms.pkl \
  task.expert_rdms_path=results/word_features/.../cot-ap-0.5/.../rdms \
  task.full_rdms_path=results/word_features/.../full/.../rdms \
  'task.layers=[attn_c_proj]' \
  task.run_tag=expert_vs_full
```


## Aggregated Results (Generated)

### Table 1: RSA Statistics - Attention Layers (`attn.c_proj`)

| Condition | Sig (Raw) | Sig (FDR) | Mean d | Max d | Mean Expert (SD) | Mean Full (SD) |
|---|---|---|---|---|---|---|
| cot-ap-0.7 | 22 | 0 | +0.536 | +1.356 | 0.010 (0.013) | -0.001 (0.017) |
| cot-ap-0.6-unique-corr_0.8 | 11 | 2 | +0.326 | +2.993 | 0.004 (0.015) | -0.001 (0.018) |
| cot-ap-0.6-unique-corr_0.9 | 11 | 2 | +0.326 | +2.993 | 0.004 (0.015) | -0.001 (0.018) |
| cot-ap-0.6 | 11 | 2 | +0.326 | +2.993 | 0.004 (0.015) | -0.001 (0.018) |
| cot-ap-0.5-unique-corr_0.8 | 7 | 0 | +0.027 | +1.754 | -0.001 (0.017) | -0.001 (0.018) |
| cot-ap-0.5-unique-corr_0.9 | 7 | 0 | +0.027 | +1.754 | -0.001 (0.017) | -0.001 (0.018) |
| cot-ap-0.5 | 7 | 0 | +0.027 | +1.754 | -0.001 (0.017) | -0.001 (0.018) |
| cot-ap-0.8 | 5 | 0 | -0.069 | +1.322 | -0.002 (0.016) | -0.000 (0.015) |
| cot-ap-0.9 | 58 | 43 | -0.929 | +0.230 | -0.028 (0.017) | 0.031 (0.024) |


### Table 2: RSA Statistics - MLP Layers (`mlp.c_fc`)

| Condition | Sig (Raw) | Sig (FDR) | Mean d | Max d | Mean Expert (SD) | Mean Full (SD) |
|---|---|---|---|---|---|---|
| cot-ap-0.5-unique-corr_0.8 | 19 | 4 | +0.443 | +2.106 | 0.017 (0.020) | 0.013 (0.021) |
| cot-ap-0.5-unique-corr_0.9 | 20 | 2 | +0.439 | +1.938 | 0.017 (0.020) | 0.013 (0.021) |
| cot-ap-0.5 | 20 | 2 | +0.437 | +1.918 | 0.017 (0.020) | 0.013 (0.021) |
| cot-ap-0.6 | 20 | 4 | +0.397 | +2.012 | 0.018 (0.019) | 0.013 (0.021) |
| cot-ap-0.6-unique-corr_0.9 | 20 | 4 | +0.397 | +2.006 | 0.018 (0.019) | 0.013 (0.021) |
| cot-ap-0.6-unique-corr_0.8 | 21 | 5 | +0.388 | +2.161 | 0.018 (0.019) | 0.013 (0.021) |
| cot-ap-0.7 | 13 | 2 | +0.157 | +2.124 | 0.015 (0.017) | 0.013 (0.021) |
| cot-ap-0.9 | 10 | 0 | -0.183 | +0.707 | 0.005 (0.014) | 0.013 (0.021) |
| cot-ap-0.8 | 12 | 4 | -0.207 | +0.976 | 0.007 (0.015) | 0.013 (0.021) |


### Table 3: RSA Statistics - Attention Layers (`attn.c_attn`)

| Condition | Sig (Raw) | Sig (FDR) | Mean d | Max d | Mean Expert (SD) | Mean Full (SD) |
|---|---|---|---|---|---|---|
| cot-ap-0.5-unique-corr_0.8 | 4 | 0 | +0.026 | +1.019 | 0.018 (0.021) | 0.018 (0.022) |
| cot-ap-0.5 | 5 | 0 | +0.020 | +1.012 | 0.018 (0.021) | 0.018 (0.022) |
| cot-ap-0.5-unique-corr_0.9 | 5 | 0 | +0.018 | +1.011 | 0.018 (0.021) | 0.018 (0.022) |
| cot-ap-0.6-unique-corr_0.8 | 2 | 0 | -0.068 | +1.000 | 0.017 (0.021) | 0.018 (0.022) |
| cot-ap-0.6-unique-corr_0.9 | 2 | 0 | -0.079 | +0.955 | 0.017 (0.021) | 0.018 (0.022) |
| cot-ap-0.6 | 2 | 0 | -0.085 | +0.959 | 0.017 (0.021) | 0.018 (0.022) |
| cot-ap-0.9 | 9 | 0 | -0.230 | +0.464 | 0.005 (0.008) | 0.018 (0.022) |
| cot-ap-0.8 | 12 | 0 | -0.307 | +1.017 | 0.010 (0.016) | 0.018 (0.022) |
| cot-ap-0.7 | 15 | 0 | -0.353 | +0.740 | 0.011 (0.018) | 0.018 (0.022) |


### Table 4: RSA Statistics - MLP Layers (`mlp.c_proj`)

| Condition | Sig (Raw) | Sig (FDR) | Mean d | Max d | Mean Expert (SD) | Mean Full (SD) |
|---|---|---|---|---|---|---|
| cot-ap-0.6-unique-corr_0.9 | 30 | 3 | +0.627 | +1.755 | 0.011 (0.016) | -0.005 (0.020) |
| cot-ap-0.6 | 30 | 3 | +0.627 | +1.755 | 0.011 (0.016) | -0.005 (0.020) |
| cot-ap-0.6-unique-corr_0.8 | 30 | 3 | +0.625 | +1.761 | 0.010 (0.016) | -0.005 (0.020) |
| cot-ap-0.5 | 17 | 0 | +0.550 | +1.714 | 0.005 (0.019) | -0.005 (0.020) |
| cot-ap-0.5-unique-corr_0.9 | 17 | 0 | +0.548 | +1.676 | 0.005 (0.019) | -0.005 (0.020) |
| cot-ap-0.5-unique-corr_0.8 | 17 | 0 | +0.541 | +1.652 | 0.005 (0.019) | -0.005 (0.020) |
| cot-ap-0.7 | 11 | 0 | +0.403 | +1.381 | 0.007 (0.013) | -0.006 (0.019) |
| cot-ap-0.8 | 9 | 0 | +0.321 | +1.030 | 0.006 (0.013) | -0.004 (0.018) |
| cot-ap-0.9 | 4 | 0 | -0.164 | +1.278 | 0.000 (0.017) | 0.012 (0.023) |
