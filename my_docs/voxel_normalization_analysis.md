# Voxel Normalization Robustness Analysis

## Executive Summary

This document presents a comprehensive comparison between the original RSA results (non-voxel-normalized brain RDMs) and the new results obtained with **per-voxel normalized** brain RDMs. The per-voxel normalization was applied as suggested in the original paper's limitations section to verify the robustness of the expert neuron findings.

**Key Finding:** Per-voxel normalization **fundamentally reverses** the main results across all layer components. The expert neuron advantage observed in the original analysis disappears or becomes negative when voxels are normalized to contribute equally.

---

## Methodological Difference

### Original Approach (Non-Normalized)
- Brain RDMs computed using Pearson correlation on raw voxel activations
- `np.corrcoef(region_acts)` where `region_acts` has shape [n_words, n_voxels]
- Correlation computed across words (rows), normalizing each word's activation vector across voxels
- **Issue:** Voxels with larger response variance contribute disproportionately to dissimilarities

### Voxel-Normalized Approach
- Per-voxel z-scoring applied before computing correlations
- Each voxel's activation vector is normalized across words (column-wise normalization)
- All voxels contribute equally to the representational geometry
- **Formula:**
  ```python
  voxel_mean = region_acts.mean(axis=0, keepdims=True)
  voxel_std = region_acts.std(axis=0, keepdims=True)
  region_acts_norm = (region_acts - voxel_mean) / voxel_std
  rdm = 1.0 - np.corrcoef(region_acts_norm)
  ```

---

## Comparative Results

### Table 1: MLP Projection Layer (`mlp.c_proj`) - Main Finding

| AP Threshold | Original Mean d | Voxel-Normalized Mean d | Change | Interpretation |
|--------------|-----------------|-------------------------|--------|----------------|
| 0.5 | **+0.550** | **-0.419** | **Reversed** | Expert advantage → Dense advantage |
| 0.6 | **+0.627** | **-0.523** | **Reversed** | Expert advantage → Dense advantage |
| 0.7 | **+0.403** | **-0.653** | **Reversed** | Expert advantage → Dense advantage |
| 0.8 | **+0.321** | **-0.644** | **Reversed** | Expert advantage → Dense advantage |
| 0.9 | -0.164 | -0.186 | Similar | Both show dense advantage |

**Original Paper Claim (Section 3.3.1):**
> "MLP layers showed the strongest and most consistent expert neuron benefits... At its optimal setting (AP ≥ 0.6), expert neurons achieved a substantial effect size relative to the dense baseline (mean d = 0.627), with 30 regions showing uncorrected significance and 3 surviving FDR correction."

**Voxel-Normalized Reality:**
> At AP ≥ 0.6, the effect **reverses** (mean d = -0.523), with dense embeddings outperforming expert neurons in 78/87 regions (90%).

---

### Table 2: Attention Projection Layer (`attn.c_proj`)

| AP Threshold | Original Mean d | Voxel-Normalized Mean d | Change | Interpretation |
|--------------|-----------------|-------------------------|--------|----------------|
| 0.5 | +0.027 | **+0.290** | **More positive** | Unexpected increase |
| 0.6 | **+0.326** | **-0.163** | **Reversed** | Expert advantage → Dense advantage |
| 0.7 | **+0.536** | **-0.551** | **Reversed** | Expert advantage → Dense advantage |
| 0.8 | -0.069 | -0.545 | More negative | Stronger dense advantage |
| 0.9 | -0.929 | -0.786 | Similar | Both show strong dense advantage |

**Original Paper Claim (Section 3.3.2):**
> "Attention layers were highly sensitive to the AP threshold... Performance increased with stricter thresholds, peaking at AP ≥ 0.7 (mean d = 0.536, 22 significant regions)."

**Voxel-Normalized Reality:**
> The peak effect at AP ≥ 0.7 **reverses completely** (mean d = -0.551). Interestingly, AP = 0.5 shows an unexpected increase (+0.290 vs +0.027), suggesting a more complex interaction.

---

### Table 3: MLP Expansion Layer (`mlp.c_fc`)

| AP Threshold | Original Mean d | Voxel-Normalized Mean d | Change | Interpretation |
|--------------|-----------------|-------------------------|--------|----------------|
| 0.5 | **+0.443** | **+0.088** | **Reduced to near-zero** | Expert advantage eliminated |
| 0.6 | **+0.443** | **-0.010** | **Reversed** | Expert advantage → Dense advantage |
| 0.7 | **+0.312** | **-0.214** | **Reversed** | Expert advantage → Dense advantage |
| 0.8 | **+0.198** | **-0.461** | **Reversed** | Expert advantage → Dense advantage |
| 0.9 | -0.089 | -0.642 | More negative | Stronger dense advantage |

**Original Paper Claim (Section 3.3.1):**
> "The expansion layer (mlp.c_fc) also benefitted from expert filtering, but the effect was strongest at the projection bottleneck."

**Voxel-Normalized Reality:**
> The expert advantage in `mlp.c_fc` is **eliminated** at AP = 0.5 (d = +0.088, near zero) and **reverses** at higher thresholds.

---

### Table 4: Attention Input Projection (`attn.c_attn`)

| AP Threshold | Original Mean d | Voxel-Normalized Mean d | Change | Interpretation |
|--------------|-----------------|-------------------------|--------|----------------|
| 0.5 | +0.156 | **-0.148** | **Reversed** | Expert advantage → Dense advantage |
| 0.6 | +0.289 | **-0.365** | **Reversed** | Expert advantage → Dense advantage |
| 0.7 | +0.412 | **-0.522** | **Reversed** | Expert advantage → Dense advantage |
| 0.8 | +0.234 | **-0.577** | **Reversed** | Expert advantage → Dense advantage |
| 0.9 | -0.123 | -0.690 | More negative | Stronger dense advantage |

**Original Paper Claim (Appendix):**
> Attention input projections showed moderate expert benefits across thresholds.

**Voxel-Normalized Reality:**
> **Consistent reversal** across all AP thresholds from 0.5 to 0.8.

---

## Summary: Complete Layer Comparison

### Mean Cohen's d Across All Conditions

| AP Threshold | mlp.c_proj | attn.c_proj | mlp.c_fc | attn.c_attn |
|--------------|------------|-------------|----------|-------------|
| | **Original** → **Voxel-Norm** | **Original** → **Voxel-Norm** | **Original** → **Voxel-Norm** | **Original** → **Voxel-Norm** |
| 0.5 | +0.550 → **-0.419** | +0.027 → **+0.290** | +0.443 → **+0.088** | +0.156 → **-0.148** |
| 0.6 | +0.627 → **-0.523** | +0.326 → **-0.163** | +0.443 → **-0.010** | +0.289 → **-0.365** |
| 0.7 | +0.403 → **-0.653** | +0.536 → **-0.551** | +0.312 → **-0.214** | +0.412 → **-0.522** |
| 0.8 | +0.321 → **-0.644** | -0.069 → **-0.545** | +0.198 → **-0.461** | +0.234 → **-0.577** |
| 0.9 | -0.164 → -0.186 | -0.929 → -0.786 | -0.089 → -0.642 | -0.123 → -0.690 |

**Legend:** Bold indicates reversal of effect direction

---

## Interpretation

### What This Means

1. **The expert neuron advantage is NOT robust to per-voxel normalization.**
   - In the original analysis, expert neurons outperformed dense embeddings across most layer types and AP thresholds
   - After per-voxel normalization, dense embeddings consistently outperform expert neurons

2. **High-variance voxels were driving the original signal.**
   - Voxels with larger response variance contributed disproportionately to the original brain RDMs
   - When these voxels are normalized to equal variance, the expert neuron advantage disappears

3. **The finding suggests a methodological confound.**
   - The original brain alignment signal may have been driven by voxel-specific variance patterns rather than genuine representational similarity
   - Expert neurons may align with high-variance voxels, but not with the underlying semantic structure

4. **AP = 0.9 shows consistent negative effects in both conditions.**
   - This suggests that at very high thresholds, expert sets become too sparse to capture semantic content
   - The consistency here validates the experimental setup

### Implications for the Paper

The original discussion section states:

> "Our brain RDMs rely on Pearson correlation, which normalizes activation patterns per word but not per voxel. Consequently, voxels with larger response variance may contribute disproportionately to the computed dissimilarities. Future work could apply per-voxel normalization to ensure all voxels contribute equally to the representational geometry, verifying the robustness of these findings."

**This robustness check reveals that the findings are NOT stable.** The expert neuron hypothesis, as tested through RSA with brain RDMs, does not hold when per-voxel normalization is applied.

### Possible Explanations

1. **High-variance voxels track expert neuron activity:**
   - Voxels with large response variance may happen to align with the activation patterns of expert neurons
   - This could be coincidental or reflect that both capture high-amplitude signal components

2. **Expert neurons capture amplitude, not geometry:**
   - Expert neurons may encode concepts through activation amplitude rather than representational geometry
   - Per-voxel normalization removes amplitude information, leaving only geometry

3. **Dense embeddings better capture semantic structure:**
   - When all voxels contribute equally, the distributed dense embeddings may better capture the brain's semantic organization
   - Expert neurons may be too sparse or specialized to match the brain's integrative representation

---

## Recommendations for Paper Revision

### 1. Add a New Results Section

Include the voxel-normalization analysis as a robustness check, presenting the complete tables and the reversal of effects.

### 2. Revise the Discussion

Acknowledge that:
- The expert neuron advantage is sensitive to voxel normalization
- High-variance voxels may have driven the original findings
- The robustness check raises questions about the interpretation of brain alignment

### 3. Consider Alternative Interpretations

- Expert neurons may encode concepts through mechanisms other than representational geometry
- The amplitude of expert neuron activity may be more important than their relative activation patterns
- Dense embeddings may better capture the brain's integrative semantic representation

### 4. Future Work Directions

- Investigate whether expert neurons predict absolute activation levels in specific voxels
- Test whether combining expert neurons with amplitude information improves prediction
- Explore whether different normalization strategies reveal different aspects of brain-LLM alignment

---

## Conclusion

The per-voxel normalization robustness check fundamentally challenges the main finding of the paper. While expert neurons showed significant brain alignment in the original analysis, this effect disappears or reverses when voxels are normalized to contribute equally.

This suggests that the original findings may have been confounded by voxel-specific variance patterns, and that the expert neuron hypothesis requires re-evaluation under more rigorous methodological controls.

**The expert neuron advantage is not robust to per-voxel normalization.**
