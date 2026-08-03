# Module 5: Pairwise Expert-Overlap Heatmaps

## Research question

Does the expert structure form recognizable clusters, with concepts that belong to the same category sharing more expert units with each other than with concepts from other categories?

## Analysis

Notation. Let $\mathcal{C}$ be the full concept list from the metadata (module 3 compared each concept only to its own category, whereas here every concept is compared to every other concept). As in module 3, each concept's expert set $E_c$ contains **(layer, unit) pairs**, because the raw `unit` column holds only the neuron index within a layer. The feature axis below is therefore indexed by pairs $(\ell, u)$ rather than by bare unit indices, so that identically indexed neurons in different layers remain distinct.

**Analysis scopes.** Subchapter 5.1 is *set-based*, meaning it reads expert rows as an unordered collection of (layer, unit) pairs, so the layer axis never enters. Subchapter 5.2 reads the layer axis only as a set of bins to normalize over, so it is permutation-invariant too. Each scope therefore runs the module exactly once. It runs on the whole model first, writing into the module folder itself, then once per sublayer type into `sublayers/<rank>_<sublayer>/`. Module 1, subchapter 1.7 defines the scopes and the rank prefix. A cross-scope summary lands in `sublayer_comparison.csv` and `sublayer_comparison.png` at the module's top level. Its columns are the mean within-category and across-category Jaccard, their difference, the rank-based category-alignment ROC-AUC, and the same ROC-AUC computed on each of the two layer-profile matrices of subchapter 5.2. The ROC-AUC is the sharper form of this module's question and is computed over level-2 concepts only, since category-label words have no same-category peers.

The layer-profile matrices are given a ROC-AUC rather than a within-minus-across contrast because ROC-AUC is rank-based, hence immune to the heavy same/different pair imbalance and expressed on a scale directly comparable to the Jaccard ROC-AUC beside it. Read together, those three numbers answer whether the layer-profile metric carries categorical signal that the Jaccard index does not.

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

The category color system behind the tick labels and the boundary lines is shared with module 3's bar charts, so a category reads as the same color in both modules. The palette alternates cool and warm hues across adjacent categories in the sorted order, rather than assigning colors by simple index, so that neighboring category blocks in the heatmap stay visually distinct even when the category order places similar categories next to each other.

### 5.2 All-pairs agreement in layer distribution

**Mathematical formulation.** Module 3, subchapter 3.2 defines the layer profile $p_c$, the Jensen-Shannon similarity $S(c,d) = 100(1 - \sqrt{\mathrm{JSD}(p_c, p_d)})$, and its count-matched null z-score $z(c,d)$, along with the full rationale for choosing Jensen-Shannon and for controlling on expert-set size. Here the same two quantities are computed for every pair in $\mathcal{C}$ rather than only for concept-to-parent pairs, giving two further square matrices alongside $J$, $O$ and $I$.

The distinction being drawn is the one subchapter 5.1 cannot see. $J_{cd}$ and $O_{cd}$ ask *which neurons* two concepts share, so two concepts allocating the same fraction of their experts to the same depths, without sharing a single neuron, sit at $J_{cd} = O_{cd} = 0$. $S(c,d)$ scores that case high. Whether the categorical block structure visible in $J$ survives in $S$, and in $z$ after size is conditioned out, is what the three ROC-AUC columns of `sublayer_comparison.csv` report.

**Generated data structures.** Two further square CSV matrices and their heatmaps, sharing the layout, colormap, category-colored labels and white boundary lines described above.

#### 4. layer_profile_matrix.csv

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| (index) | string | $c$ | Concept label for the row. |
| `<concept>` (one column per concept) | float | $S(c,d)$ | Layer-profile similarity as a percentage, diagonal 100, empty for words below 2 experts. |

#### 5. layer_profile_z_matrix.csv

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| (index) | string | $c$ | Concept label for the row. |
| `<concept>` (one column per concept) | float | $z(c,d)$ | Standard deviations by which $S(c,d)$ exceeds the count-matched null, diagonal empty. |

- `layer_profile_heatmap.png`, encodes the values $S(c,d)$ of `layer_profile_matrix.csv`.
- `layer_profile_z_heatmap.png`, encodes the values $z(c,d)$ of `layer_profile_z_matrix.csv`.

These two matrices are the intended source for any downstream model consuming pairwise expert features, since together with `jaccard_matrix.csv` they cover both readings of pairwise agreement, shared identity and shared depth allocation, over the same concept ordering.

A note on coverage. Words holding fewer than 2 experts have no usable layer profile and are empty throughout both matrices. When the cross-scope summary reduces a matrix to its category-alignment ROC-AUC, such words are dropped as whole concepts rather than as scattered pairs, because the alignment statistic derives its same-category mask from the concept list and permutes labels across concepts, both of which require the pair pool to remain a complete triangle over one consistent concept set.

## Results

*Scope note.* All values are **whole-model scope** from the corrected `_sensefix` runs, 197 concepts with a defined category. Per-sublayer replications sit in `sublayers/<rank>_<sublayer>/`, and the cross-scope contrast is summarised in `sublayer_comparison.csv`.

At AP=0.6, the same-category effect is strong in relative terms. Within-category pairs average **5.04%** Jaccard against **0.44%** across categories in Qwen3, a factor of **11.4**, and **3.42%** against **0.37%** in GPT-2, a factor of **9.3**. Same-category concepts share disproportionately many specific (layer, unit) experts, whereas a random different-category pair shares almost none. In absolute terms even within-category similarity is small, which is why the effect is far clearer in the tabulated ratio than in the rendered heatmap.

This absolute smallness is the reason the effect is clearer in the tabulated ratio than in the rendered heatmaps. On a 0 to 100 color scale the diagonal, which is 100 by construction, occupies the top of the color range, and nearly every off-diagonal cell, including the elevated same-category ones, falls in the bottom few percent of the scale and appears near-black. The relative signal is therefore evident in the numbers but faint in the image.

### Across AP thresholds

| Model | AP | experts | Jaccard within / across | Jaccard ratio |
|---|---|---|---|---|
| GPT-2 | 0.5 | 216,555 | 5.89% / 0.93% | 6.3x |
| GPT-2 | 0.6 | 76,416 | 3.42% / 0.37% | 9.3x |
| GPT-2 | 0.7 | 28,983 | 1.84% / 0.14% | 13.1x |
| GPT-2 | 0.8 | 9,222 | 0.85% / 0.04% | 19.8x |
| GPT-2 | 0.9 | 1,417 | 0.15% / 0.001% | 141.6x |
| Qwen3 | 0.5 | 1,188,772 | 7.00% / 1.02% | 6.8x |
| Qwen3 | 0.6 | 418,529 | 5.04% / 0.44% | 11.4x |
| Qwen3 | 0.7 | 165,497 | 3.27% / 0.19% | 17.5x |
| Qwen3 | 0.8 | 59,797 | 1.68% / 0.08% | 20.3x |
| Qwen3 | 0.9 | 12,212 | 0.43% / 0.02% | 19.6x |

The pattern is consistent and monotone in both architectures. The *relative* within-versus-across-category effect strengthens markedly as AP tightens, from about 6x to 20x, while the *absolute* similarities shrink toward zero. The experts that survive strict AP filtering are increasingly *category-specific*. The GPT-2 AP=0.9 figure of 141x should not be read as a stronger effect than Qwen3's 19.6x, since its denominator is an across-category mean of 0.001% computed over only 1,417 surviving experts, so the ratio is dominated by how close to zero the denominator has fallen rather than by any gain within categories.

This is the strongest evidence in the analysis suite that the expert space is organized along category lines. The effect lives in the ratio rather than in the raw magnitudes, and it is more evident in the tabulated ratio than in the rendered heatmaps, where the values compress toward the bottom of the color scale.

### Layer-profile agreement (subchapter 5.2)

The ratio table above is expressed in absolute percentages, which the layer-profile metric cannot be compared against directly, so all three matrices are reduced here to the rank-based category alignment ROC-AUC, which is on one scale and where 0.5 is chance:

| Model | AP | Jaccard AUC | layer-profile AUC | layer-profile $z$ AUC |
|---|---|---|---|---|
| GPT-2 | 0.5 | 0.934 | 0.618 | 0.610 |
| GPT-2 | 0.6 | 0.890 | 0.610 | 0.620 |
| GPT-2 | 0.7 | 0.742 | 0.596 | 0.629 |
| GPT-2 | 0.8 | 0.571 | 0.594 | 0.618 |
| GPT-2 | 0.9 | 0.506 | 0.566 | 0.553 |
| Qwen3 | 0.5 | 0.958 | 0.716 | 0.710 |
| Qwen3 | 0.6 | 0.952 | 0.707 | 0.707 |
| Qwen3 | 0.7 | 0.924 | 0.709 | 0.713 |
| Qwen3 | 0.8 | 0.768 | 0.709 | 0.724 |
| Qwen3 | 0.9 | 0.546 | 0.685 | 0.706 |

Two readings, and the second is the important one.

At lenient thresholds the Jaccard index is the far better category detector, 0.958 against 0.716 for Qwen3 at AP=0.5. Which specific neurons two concepts share is simply more diagnostic of category membership than how they distribute those neurons over depth, and this is the expected result.

**The ordering reverses at strict thresholds, in both architectures.** Jaccard's AUC collapses to 0.506 (GPT-2) and 0.546 (Qwen3) by AP=0.9, both essentially chance: the expert sets are so thinned that shared-neuron identity carries almost no category information, which is the mirror image of the ratio table above, where the surviving relative effect rests on a handful of pairs sharing anything at all. Layer-profile agreement is nearly flat over the same range, 0.618 to 0.566 in GPT-2 and 0.716 to 0.685 in Qwen3, and its $z$ form is flatter still. The crossover happens earlier in GPT-2, at AP 0.8 (0.594 profile against 0.571 Jaccard), than in Qwen3, at AP 0.9 (0.685 against 0.546), consistent with GPT-2's expert sets thinning faster at every threshold.

The interpretation is that depth allocation is the more robust carrier of category structure. Two same-category concepts continue to place their experts at similar depths even once the AP filter has stripped away nearly every shared neuron, so the categorical organization of the expert space survives in the layer profile after it has effectively vanished from set identity.

That the raw and $z$ columns track each other so closely, never differing by more than 0.03, is itself a useful check. It says the categorical signal in the layer profile is not an artifact of expert-set size, since conditioning on size leaves it intact. At AP=0.8 and 0.9 the $z$ form is in fact slightly the stronger of the two, which is consistent with size noise diluting the raw reading precisely where expert sets are smallest.

**Per sublayer at AP=0.6**, the two metrics agree on the ranking but differ in spread:

| Scope | experts | Jaccard AUC | layer-profile AUC |
|---|---|---|---|
| whole model | 414,089 | 0.931 | 0.692 |
| mlp.gate_proj | 191,555 | 0.927 | 0.683 |
| mlp.up_proj | 102,650 | 0.923 | 0.657 |
| self_attn.o_proj | 42,504 | 0.879 | 0.675 |
| mlp.down_proj | 34,184 | 0.626 | 0.581 |
| self_attn.q_proj | 17,185 | 0.698 | 0.604 |
| self_attn.v_proj | 15,376 | 0.779 | 0.625 |
| self_attn.k_proj | 10,635 | 0.714 | 0.603 |

Both put the FFN expansion projections at the top and `mlp.down_proj` at the bottom, consistent with the sublayer informativeness ranking of module 1. The layer-profile AUC has a much narrower range, 0.581 to 0.692 against 0.626 to 0.931, which is expected: within a single sublayer scope the layer profile has only 28 bins to work with, one per block, so it has far less resolution than the whole-model scope's 196.
