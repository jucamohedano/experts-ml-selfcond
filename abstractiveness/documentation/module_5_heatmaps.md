# Module 5: Pairwise Expert-Overlap Heatmaps

## Research question

Does the expert structure form recognizable clusters, with concepts that belong to the same category sharing more expert units with each other than with concepts from other categories?

## Analysis

Notation. Let $\mathcal{C}$ be the full concept list from the metadata (module 3 compared each concept only to its own category, whereas here every concept is compared to every other concept). As in module 3, each concept's expert set $E_c$ contains **(layer, unit) pairs**, because the raw `unit` column holds only the neuron index within a layer. The feature axis below is therefore indexed by pairs $(\ell, u)$ rather than by bare unit indices, so that identically indexed neurons in different layers remain distinct.

### 5.1 All-pairs expert-set similarity via binary matrix products

**Mathematical formulation.** Instead of looping over pairs, the module computes all pairwise set operations at once through linear algebra on a binary presence matrix. Define

$$A \in \{0,1\}^{|\mathcal{C}| \times |\mathcal{F}|}, \qquad \mathcal{F} = \{(\ell, u)\ \text{pairs observed in the data}\}, \qquad A_{c,(\ell,u)} = \begin{cases} 1 & (\ell, u) \in E_c \\ 0 & \text{otherwise,} \end{cases}$$

built with `pivot_table(index='concept', columns=['layer_idx', 'unit'], values='present', fill_value=0)` and reindexed over every concept in the metadata, so concepts with no retained experts appear as all-zero rows. Three derived matrices follow. The Gram matrix of $A$ counts shared (layer, unit) experts, because the inner product of two binary rows counts positions where both are 1:

$$I = A A^{\top}, \qquad I_{cd} = \sum_{(\ell,u)} A_{c,(\ell,u)} A_{d,(\ell,u)} = |E_c \cap E_d| .$$

The row sums give set sizes, $s_c = \sum_{(\ell,u)} A_{c,(\ell,u)} = |E_c| = n_c$, from which union and minimum-size matrices are assembled by broadcasting:

$$U_{cd} = s_c + s_d - I_{cd} = |E_c \cup E_d|, \qquad m_{cd} = \min(s_c, s_d).$$

The two similarity matrices are then the element-wise ratios, as percentages (with cells where the denominator is 0 set to 0):

$$J_{cd} = 100 \cdot \frac{I_{cd}}{U_{cd}} \quad \text{(Jaccard index)}, \qquad O_{cd} = 100 \cdot \frac{I_{cd}}{m_{cd}} \quad \text{(overlap coefficient)}.$$

Both are symmetric with diagonal 100 (each set is identical to itself), and $J_{cd} \le O_{cd}$ everywhere. As in module 3, $J$ demands near-identity of the two sets while $O$ rewards containment of the smaller set in the larger. Computed here for *all* $\binom{|\mathcal{C}|}{2}$ pairs, they answer the clustering question: if categories organize the expert space, same-category pairs $(c, d)$ should show visibly higher $J_{cd}$ or $O_{cd}$ than different-category pairs, appearing as bright blocks along the diagonal when concepts are ordered by category.

**Generated data structures.** Three square CSV matrices and two heatmap images.

#### 1. jaccard_matrix.csv

The matrix $J$: rows and columns are concepts, entries are Jaccard similarity percentages $J_{cd}$. The unnamed first column holds the row concept label, and the diagonal is always 100.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| (index) | string | $c$ | Concept label for the row. |
| `<concept>` (one column per concept) | float | $J_{cd}$ | Jaccard similarity percentage between the row concept and this column concept. |

Example (head, first 6 columns, of `AP_0.6/5_heatmaps/jaccard_matrix.csv` in `research_plots_150_revised_executor_again`):

| (index) | animal | alligator | frog | goldfish | iguana |
|---|---|---|---|---|---|
| animal | 100.0 | 4.02 | 9.78 | 4.77 | 4.21 |
| alligator | 4.02 | 100.0 | 5.75 | 4.12 | 6.38 |
| frog | 9.78 | 5.75 | 100.0 | 7.53 | 4.78 |
| goldfish | 4.77 | 4.12 | 7.53 | 100.0 | 3.96 |
| iguana | 4.21 | 6.38 | 4.78 | 3.96 | 100.0 |

#### 2. overlap_matrix.csv

The matrix $O$: same shape and indexing as `jaccard_matrix.csv`, entries are overlap-coefficient percentages $O_{cd}$.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| (index) | string | $c$ | Concept label for the row. |
| `<concept>` (one column per concept) | float | $O_{cd}$ | Overlap coefficient percentage between the row concept and this column concept. |

Example (head, first 6 columns, of `AP_0.6/5_heatmaps/overlap_matrix.csv` in `research_plots_150_revised_executor_again`):

| (index) | animal | alligator | frog | goldfish | iguana |
|---|---|---|---|---|---|
| animal | 100.0 | 19.91 | 21.60 | 9.52 | 20.78 |
| alligator | 19.91 | 100.0 | 37.65 | 19.05 | 12.02 |
| frog | 21.60 | 37.65 | 100.0 | 17.90 | 31.48 |
| goldfish | 9.52 | 19.05 | 17.90 | 100.0 | 18.25 |
| iguana | 20.78 | 12.02 | 31.48 | 18.25 | 100.0 |

#### 3. shared_expert_counts_matrix.csv

The intersection (Gram) matrix $I$ itself, saved as integers: the raw shared-expert counts behind both percentage matrices, with the diagonal holding each concept's own expert-set size $I_{cc} = n_c$. It is the scale reference for the percentages (the same caveat as module 1, subchapter 1.3, and module 3): a given Jaccard percentage backed by hundreds of shared experts is a far more stable measurement than the same percentage backed by a handful, so any cell of $J$ or $O$ can be traced back to the integers that produced it.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| (index) | string | $c$ | Concept label for the row. |
| `<concept>` (one column per concept) | int | $I_{cd} = \|E_c \cap E_d\|$ | Raw number of shared (layer, unit) experts, with the diagonal holding the own set size $n_c$. |

Example (head, first 5 columns, of `AP_0.6/5_heatmaps/shared_expert_counts_matrix.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`):

| (index) | furniture | bed | bench | bookcase | cabinet |
|---|---|---|---|---|---|
| furniture | 259 | 75 | 22 | 135 | 127 |
| bed | 75 | 264 | 20 | 64 | 58 |
| bench | 22 | 20 | 181 | 19 | 12 |
| bookcase | 135 | 64 | 19 | 1055 | 193 |
| cabinet | 127 | 58 | 12 | 193 | 419 |

**Plots.** Two **heatmap** images (`seaborn`-style matrix heatmap rendered via the `_plot_heatmap_with_leaders` helper, `magma` colormap), with concept labels on both axes (colored by category) and similarity encoded by color. Concepts are ordered by category, each block led by its root label (which counts as part of its own category), and two layers of white separator lines are drawn on both axes so every cell can be traced back to its row and column concept. A faint grid line (0.3 px, 25% opacity) is drawn at every single concept boundary, and a bold, fully opaque 1 px line is drawn on top of it at every category-block boundary, so the fine per-concept grid and the coarser category structure, along with the expected bright same-category blocks along the diagonal, can both be read directly off the matrix:

- `jaccard_heatmap.png`, encodes the values $J_{cd}$ of `jaccard_matrix.csv`.
- `overlap_heatmap.png`, encodes the values $O_{cd}$ of `overlap_matrix.csv`.

## Results

At AP=0.6, the same-category effect is strong in relative terms. Splitting all 13,366 concept pairs into same category (619 pairs) and different category (12,747 pairs), within-category pairs average **3.46%** Jaccard against **0.52%** across categories, a factor of **6.6**, and **10.29%** overlap against **1.93%**, a factor of **5.3**. Same-category concepts share disproportionately many specific (layer, unit) experts, whereas a random different-category pair shares almost none. In absolute terms, even within-category similarity is small, with a mean of 3.5% Jaccard, the largest off-diagonal pair at 33%, and the 99th percentile at just 6.8%.

This absolute smallness is the reason the effect is clearer in the tabulated ratio than in the rendered heatmaps. On a 0 to 100 color scale the diagonal, which is 100 by construction, occupies the top of the color range, and nearly every off-diagonal cell, including the elevated same-category ones, falls in the bottom few percent of the scale and appears near-black. The relative signal is therefore evident in the numbers but faint in the image.

### Across AP thresholds

| AP | Jaccard within / across | Jaccard ratio | Overlap within / across | Overlap ratio |
|---|---|---|---|---|
| 0.5 | 5.53% / 1.16% | 4.76x | 14.96% / 3.73% | 4.01x |
| 0.6 | 3.46% / 0.52% | 6.61x | 10.29% / 1.93% | 5.32x |
| 0.7 | 2.00% / 0.23% | 8.80x | 6.65% / 1.00% | 6.68x |
| 0.8 | 1.12% / 0.07% | 16.64x | 4.15% / 0.37% | 11.08x |
| 0.9 | 0.18% / 0.01% | 18.29x | 0.65% / 0.05% | 13.75x |

The pattern is consistent and monotone. The *relative* within-versus-across-category effect strengthens markedly as AP tightens, from 4.8x to 18.3x for Jaccard and from 4.0x to 13.8x for overlap, while the *absolute* similarities shrink toward zero, from 5.53% to 0.18% within-category Jaccard. In other words, the experts that survive strict AP filtering are increasingly *category-specific*, so that a pair of same-category concepts at AP=0.9 shares 18x more of its expert sets than a cross-category pair does. This is the strongest evidence in the whole analysis suite that the expert space is organized along category lines. The effect lives in the ratio rather than in the raw magnitudes, and it is more evident in the tabulated within-versus-across ratio than in the rendered heatmaps, where the values compress toward the bottom of the color scale.
