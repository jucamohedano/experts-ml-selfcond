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

**The layer axis.** Both matrices are built on the BLOCK-AGGREGATED axis, sublayers summed within each transformer block, giving 28 bins on Qwen3 and 12 on GPT-2 rather than the flat 196 and 48. A bin is then a transformer block and nothing else, so $S(c,d)$ reads as agreement in depth allocation, whereas on the flat axis a bin is a block crossed with a sublayer and the quantity confounded allocating deep with allocating to a particular projection type. The block axis also carries roughly one seventh of the flat axis's plug-in entropy bias, which scales as $1/n$ and therefore contaminates small expert sets most, which is the confound $z(c,d)$ exists to remove. This is not a claim about ordering. The Jensen-Shannon divergence sums over bins independently and never touches bin adjacency, so it is permutation invariant and neither axis measures depth in an ordered sense, and what changed is what a bin MEANS together with how noisily it is estimated. Every `sublayers/` scope is numerically unchanged, because aggregating a single-sublayer frame is an identity relabel. Subchapter 5.1's set matrices are untouched, since they read neuron identity rather than a layer distribution, and both matrix files keep their existing names.

**One pair of matrices per registered metric.** $S(c,d)$ is not one quantity but a family, defined by whichever entry of the profile metric registry is asked for. Module 3, subchapter 3.3 lists the eight registered measures, gives their formulas, and explains the four families they fall into and why their raw scales are deliberately not comparable. This module computes every active one over the same profiles, built once, so each gets its own agreement matrix, its own count-matched $z$ matrix, and a heatmap for each.

Note that `wasserstein` is the only registered measure that reads bin ORDER, which is exactly why the block axis above is what makes it meaningful. On the flat layer axis the transport cost it charges would be mostly sublayer alternation rather than depth.

**Generated data structures.** Two further square CSV matrices per registered metric and their heatmaps, sharing the layout, colormap, category-colored labels and white boundary lines described above. `<metric>` below is a registry key such as `js_distance` or `wasserstein`.

#### 4. layer_profile_&lt;metric&gt;_matrix.csv

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| (index) | string | $c$ | Concept label for the row. |
| `<concept>` (one column per concept) | float | $S(c,d)$ | Layer-profile agreement under that metric, diagonal 100, empty for words below 2 experts. |

#### 5. layer_profile_&lt;metric&gt;_z_matrix.csv

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| (index) | string | $c$ | Concept label for the row. |
| `<concept>` (one column per concept) | float | $z(c,d)$ | Standard deviations by which $S(c,d)$ exceeds the count-matched null, diagonal empty. |

- `layer_profile_<metric>_heatmap.png`, encodes the values $S(c,d)$ of the matching matrix.
- `layer_profile_<metric>_z_heatmap.png`, encodes the values $z(c,d)$ of the matching matrix.

Every layer-profile artifact carries its metric in the name, the default included, so the folder reads as one family rather than as a privileged file plus six additions. Results trees written before the multi-metric extension name the default metric's pair `layer_profile_matrix.csv` and `layer_profile_z_matrix.csv`, and those files are the same quantity as today's `layer_profile_js_distance_matrix.csv` and `layer_profile_js_distance_z_matrix.csv`, verified equal at a maximum absolute difference of exactly 0.0 on all four GPT-2 sublayer scopes at AP 0.5.

#### 6. profile_metric_comparison.csv

One row per registered metric, reducing its two matrices to the two questions this module can answer about them, so the metrics can be compared inside a scope the way `sublayer_comparison.csv` compares scopes inside a module.

| Column | Type | Description |
|--------|------|-------------|
| `metric`, `metric_label` | string | Registry key and its display name. |
| `within_pct`, `across_pct`, `contrast_pct` | float | Mean agreement over within-category pairs, over across-category pairs, and their difference. Comparable across scopes for one metric, NOT across metrics, since the registry fixes orientation and not scale. |
| `category_roc_auc` | float | Category alignment ROC-AUC of the agreement matrix. Rank based, so it IS comparable across metrics. |
| `category_roc_auc_z` | float | The same statistic on the $z$ matrix. |
| `human_rho` | float | Mean over categories of the subchapter 5.3 correlation against the human ratings. |
| `human_rho_z` | float | The same, on the $z$ matrix. |

The two readings sit side by side on purpose, and module 8 already showed they can disagree, since the sublayer best at recovering the category partition was not the one best at reproducing human similarity. `category_roc_auc` asks whether the metric separates within-category pairs from across-category ones, a partition question with a large contrast behind it. `human_rho` asks whether it orders *within-category* pairs the way people do, which is the harder and more externally valid test. The accompanying `profile_metric_comparison.png` draws the same four columns, with the default metric in the figure's reference color.

These matrices are the intended source for any downstream model consuming pairwise expert features, since together with `jaccard_matrix.csv` they cover both readings of pairwise agreement, shared identity and shared depth allocation, over the same concept ordering.

A note on coverage. Words holding fewer than 2 experts have no usable layer profile and are empty throughout both matrices. When the cross-scope summary reduces a matrix to its category-alignment ROC-AUC, such words are dropped as whole concepts rather than as scattered pairs, because the alignment statistic derives its same-category mask from the concept list and permutes labels across concepts, both of which require the pair pool to remain a complete triangle over one consistent concept set.

### 5.3 Validation against human similarity judgments

**Research question.** Every result above compares the model against itself or against the category labels of the stimulus design. Subchapter 5.3 asks a different and harder question: does the similarity of two words' expert sets predict how similar *people* judge those two words to be?

**The human data.** Richie and Bhatia's Study 1 collected pairwise similarity ratings on a 1 to 7 scale, higher meaning more similar, for every within-category pair of this exact word list. The files are per category under `assets/Richie_and_Bhatia-HSJ/study1_pairwise_data/data_individual_level/`, one row per subject. After excluding the `squash` pairs in sports, whose vegetables sense is the one this dataset admits, **2,391 pairs** remain across the 8 categories, with 19 to 39 raters each. `utils/human_similarity.py` loads them and is shared with module 9.

**Why this is the harder test.** Humans rated within-category pairs only, so this comparison cannot use the large across-category contrast that gives the category alignment ROC-AUC its size. Telling a `robin` from a `truck` is not on the table. The question is whether, among birds alone, the expert sets know that a `chicken` is more like a `rooster` than a `crow` is like a `penguin`.

**Mathematical formulation.** For a category $k$ with rated pairs $P_k$, let $m_{cd}$ be the model similarity of pair $(c,d)$ taken from one of the matrices above, and $h_{cd}$ the subject-averaged human rating. The agreement is the Spearman rank correlation

$$\rho_k = \operatorname{corr_{Spearman}}\big(\{m_{cd}\}_{(c,d) \in P_k},\ \{h_{cd}\}_{(c,d) \in P_k}\big).$$

Ranks rather than raw values for three reasons. The layer-profile metric is compressed into a narrow high band by construction (subchapter 5.2) so only its ordering is meaningful, the Jaccard index is bounded below at zero and heavily right-skewed, and the ratings are bounded averages of ordinal judgments. A Pearson coefficient on those scales would report how *linear* the relationship is mixed in with how strong it is, and the relationship is visibly not linear, rising steeply near zero and then saturating.

**Which coefficient and which test are independent choices.** They are easy to conflate, so both are named on every figure and in the table. Spearman against Pearson is a choice of *coefficient*, monotone association against linear association. Parametric against permutation is a choice of *null distribution*, and it applies to either coefficient. The parametric p attached to a Spearman or a Pearson coefficient alike assumes the observations are independent, which is false here, so the significance comes from the Mantel permutation instead.

**Significance.** A category's $N_k = \binom{m_k}{2}$ rated pairs are built from only $m_k$ concepts, each appearing in $m_k - 1$ of them, so the pairs are not independent and the effective information is closer to $m_k$ than to $N_k$. For birds that is 30 concepts behind 435 pairs, an overstatement of the degrees of freedom by roughly $(m_k - 1)/2 \approx 14.5$. The permutation test shuffles **concept labels within the category** and rebuilds the model vector from the same matrix, which moves every pair containing a given concept together, exactly as the dependence in the observed data does. Writing $\mathfrak{S}_{m_k}$ for the permutations of the category's concept labels and $B = 999$ draws,

$$\hat p_k = \frac{1 + \#\{b : |\rho_k^{(\pi_b)}| \ge |\rho_k^{\mathrm{obs}}|\}}{B + 1},$$

with the $+1$ in both places making the test exact rather than anticonservative and flooring the attainable value at $1/(B+1) = 0.001$, which is why every supported row reads exactly `0.001`. Measured on Qwen3 at AP 0.6 for the layer profile, the parametric Pearson p calls 4 of the 8 categories significant and this test calls 1, with professions at parametric $p = 0.0008$ against Mantel $p = 0.169$. The parametric column is therefore not reported anywhere.

**Which pairs enter.** A rated pair is dropped for three reasons, all of which appear only at strict thresholds. Either concept can be missing from the concept list, the matrix cell can be NaN, which happens for the layer-profile matrices when a word holds fewer than 2 experts and so has no usable profile, or a word can hold **no experts at all**. The third case needs explicit handling because it does not produce a NaN: `expert_set_overlap_matrices` defines the Jaccard index as 0 when the union is empty, deliberately, so that an empty-set concept does not fill the heatmap with NaNs. Read as data, though, that 0 asserts that two expert sets share nothing when one of them does not exist, and feeding it to the correlation scores a missing measurement as maximal dissimilarity. Those pairs are therefore excluded by name.

Consequently $n$ is metric-dependent at strict thresholds. On Qwen3 at AP 0.9, where 10 of 205 concepts retain no experts and 4 more hold exactly one, the Jaccard vectors keep 2,277 of 2,391 rated pairs while the layer-profile vectors keep 2,195. Excluding the empty-set pairs moves the per-category $\rho$ by at most 0.026 and leaves every threshold below 0.9 untouched, so it is a correctness fix rather than a result change.

The genuine zeros are kept, since two words that both hold experts and share none is a real measurement. They do dominate at AP 0.9, where 2,030 of the rated pairs are tied at exactly 0, which is the honest explanation for the Jaccard correlation falling there: Spearman has almost no ordering left to read. That is a reason to treat AP 0.9 as past the useful range for this test rather than a reason to prune further.

**Noise ceiling.** Subjects are split into halves, each half averaged per pair, the two pair vectors correlated, and the result Spearman-Brown corrected to full-sample reliability. This is the largest correlation any model could achieve against ratings this noisy, and every $\rho_k$ is also reported divided by it. The measured ceilings run from 0.836 (birds) to 0.935 (vehicles). This is also what makes the uneven rater counts harmless: unequal $n$ attenuates a correlation rather than inflating it, so it biases toward missing an effect, and whatever attenuation remains is absorbed into the ceiling.

**Two pooled figures, both reported.** `POOLED` is the rank correlation over all 2,391 pairs at once, which lets between-category differences in mean similarity contribute. `POOLED_MEAN` is the unweighted mean of the eight per-category values, which does not. The second is the conservative reading and is the one carried into `sublayer_comparison.csv`.

**Which matrices are validated.** The Jaccard index, plus the agreement and the $z$ matrix of every registered profile metric, so the list grows with the registry rather than being written out. At the eight metrics active today that is 17 series against the 3 of the single-metric era. The cost is linear in the list, one Mantel permutation sweep per series per category, which is roughly a second per series per scope on the Richie-HSJ item set. Narrowing `ACTIVE_PROFILE_METRICS` in `utils/helpers.py` narrows this along with everything else.

**Generated data structures.** `human_similarity_validation.csv`, one row per (metric, category) plus the two pooled rows per metric, with columns `metric`, `metric_label`, `category`, `coefficient`, `test`, `n_pairs`, `rho`, `mantel_p`, `noise_ceiling`, `rho_over_ceiling`. The `metric` key of a layer-profile row is the matrix name, such as `layer_profile_wasserstein` or `layer_profile_wasserstein_z`. The `coefficient` and `test` columns state `spearman` and `mantel` explicitly rather than leaving `rho` to be guessed at. The column keeps the bare name `rho` because `scripts/tests/check_human_validation.py` reads it.

Two figures.

`human_vs_expert_similarity_<metric>.png`, one scatter panel per category. Each panel title carries the coefficient, its value, the Mantel p and $n$, in the form `birds  Spearman rho=0.40  Mantel p<0.001  n=435`, with values at the permutation floor printed as `p<0.001` rather than a figure 999 draws cannot justify. The p comes from the same table the figure accompanies, so the two can never disagree.

The trend drawn on each panel is a **LOWESS smooth** (`frac = 0.4`, matching `plot_helpers.plot_hexbin_with_trends` so the two scatter families in the pipeline smooth at the same scale), not a least-squares line. A straight fit is a Pearson-shaped object whose slope tracks the linear association, so placing one beside a Spearman coefficient invites reading the line as the illustration of the number when the two can disagree, professions at AP 0.6 being Pearson 0.171 against Spearman 0.105. The smooth shows the monotone shape the coefficient actually measures, and it exposes structure a line hides, notably furniture saturating above a Jaccard of about 10 and vegetables staying flat before rising only at its top end.

`human_similarity_by_category.png`, grouped bars, one x group per category and one bar per metric. The height is $\rho_k$ divided by that category's noise ceiling, not raw $\rho_k$, so a category whose raters disagreed with each other is not charged for the model's inability to predict their noise. A bar reaching 1.0 would mean the metric agrees with the raters as well as the raters agree with each other. A `*` marks each bar whose Mantel p clears 0.05. The stars are uncorrected across the tests a scope runs, 8 categories by one series per validated matrix, which is the Jaccard index plus an agreement and a $z$ matrix for each registered metric, so 136 tests at the eight metrics active today against 24 when only the default was registered. A single starred category is therefore weaker evidence than the mark suggests, and more so now than before. The two pooled rows are excluded from this figure.

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

### Agreement with human similarity (subchapter 5.3)

Whole-model scope, both models, over the 2,391 rated within-category pairs. `mean` is the unweighted average of the eight per-category Spearman correlations, which is the conservative reading, and `pooled` is the correlation over all pairs at once. The `significant` column counts categories whose within-category Mantel test clears p < 0.05.

> **Stale, pending refresh, `layer profile, mean` column only.** That column comes from the PRE-RESTRUCTURE FLAT-AXIS run and the gated full sweep has not run. On the one scope re-run so far, GPT-2 at AP 0.5, it moves from 0.123 to 0.0708, a 43 percent relative drop, so the paragraph below headed "Layer-profile agreement does not transfer to human judgments" holds in direction and understates the gap in magnitude. The Jaccard columns, `pooled`, `mean` and `mean / ceiling`, are axis-free and unchanged, verified at 0.3608 in both trees, as are the noise ceilings and the significant counts. The `sublayers/` replications are unaffected, verified at a maximum difference of exactly 0.0 on the per-category human correlations of all four GPT-2 sublayer scopes.

| Model | AP | pooled | mean | mean / ceiling | layer profile, mean | significant |
|---|---|---|---|---|---|---|
| GPT-2 | 0.5 | 0.345 | 0.361 | 0.405 | 0.123 | 5/8 |
| GPT-2 | 0.6 | 0.261 | 0.341 | 0.382 | 0.102 | 6/8 |
| GPT-2 | 0.7 | 0.193 | 0.312 | 0.350 | 0.073 | 6/8 |
| GPT-2 | 0.8 | 0.236 | 0.279 | 0.313 | 0.144 | 6/8 |
| GPT-2 | 0.9 | 0.109 | 0.126 | 0.141 | 0.199 | 5/8 |
| Qwen3 | 0.5 | 0.613 | 0.508 | 0.570 | 0.066 | 8/8 |
| Qwen3 | 0.6 | 0.538 | 0.478 | 0.536 | 0.070 | 7/8 |
| Qwen3 | 0.7 | 0.503 | 0.516 | 0.579 | 0.104 | 8/8 |
| Qwen3 | 0.8 | 0.480 | 0.508 | 0.571 | 0.121 | 8/8 |
| Qwen3 | 0.9 | 0.302 | 0.312 | 0.350 | 0.100 | 7/8 |

(The GPT-2 AP 0.9 mean is over the 6 categories that retain enough finite pairs to yield a correlation, `fruit` and `vegetables` having fallen below the floor.)

**Expert-set overlap does predict human similarity judgments.** On Qwen3 the agreement reaches 0.508 at AP 0.5, which is 57 percent of what the raters themselves achieve, and every one of the eight categories is individually significant. Per category at AP 0.5 it runs from 0.26 for `vegetables` to 0.73 for `vehicles`. This is a within-category result, so none of it rests on the easy across-category contrast, and it is the most direct external validation of the expert-set methodology anywhere in this suite.

**The agreement is far more robust to thresholding than category alignment is.** The category alignment ROC-AUC of subchapter 5.1 collapses to near chance by AP 0.9, 0.546 on Qwen3 and 0.506 on GPT-2, while human agreement over the same expert sets is still 0.312 and 0.126. Whatever survives strict filtering continues to carry graded similarity information after it has stopped carrying the category partition.

**Layer-profile agreement does not transfer to human judgments.** It sits between 0.066 and 0.199 everywhere, an order below the Jaccard column, and is the weaker reading at every threshold in both models. This is consistent with module 9's finding that the layer profile carries no ordering information on its own. The quoted range is the flat-axis one and is pending refresh with the rest of that column. The one block-axis figure available, GPT-2 at AP 0.5, is 0.0708 against the 0.123 tabulated, so moving to the block axis widens the gap against Jaccard rather than closing it and the finding is unchanged in direction.

**The two architectures differ by more than a constant.** Qwen3 roughly doubles GPT-2 at every threshold. Since both were run on identical sentences and identical human ratings, the gap is a property of the models, and it goes the same way as the expert counts: the larger model has more units clearing the AP bar, so its expert sets are better estimated.

**A caution for reading the sublayer table.** The sublayer with the best category alignment is not the one that best matches human judgment. At AP 0.5 on Qwen3, `mlp.gate_proj` leads on category alignment (AUC 0.956) but reaches only 0.468 against the human ratings, while `self_attn.o_proj` scores 0.949 and 0.580 respectively. Recovering a partition and reproducing graded similarity are different tasks and the same sublayer need not win both.

### Layer-profile agreement (subchapter 5.2)

The ratio table above is expressed in absolute percentages, which the layer-profile metric cannot be compared against directly, so all three matrices are reduced here to the rank-based category alignment ROC-AUC, which is on one scale and where 0.5 is chance:

> **Stale, pending refresh, the two layer-profile columns only.** Both come from the PRE-RESTRUCTURE FLAT-AXIS run and the gated full sweep has not run. On the one scope re-run so far, GPT-2 at AP 0.5, the layer-profile AUC moves from 0.618 to 0.6055 and its $z$ form from 0.610 to 0.5909. The Jaccard AUC column is axis-free and unchanged, verified at 0.9343 in both trees, so the crossover reading below rests on one moving column and one fixed one and should be re-checked against the sweep. The `sublayers/` replications are unaffected, verified at a maximum difference of exactly 0.0 on both columns for all four GPT-2 sublayer scopes.

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

### Metric comparison (subchapter 5.2)

> **Provisional.** One scope of one architecture, GPT-2 whole model at AP 0.5. The full sweep on both architectures is gated and has not run, so read the ordering rather than the levels.

| Metric | category AUC | category AUC, $z$ | human $\rho$ | human $\rho$, $z$ |
|---|---|---|---|---|
| `js_distance` | 0.6055 | 0.5909 | 0.0708 | 0.0660 |
| `wasserstein` | 0.5768 | 0.5680 | 0.0322 | 0.0331 |
| `cosine` | 0.6010 | 0.5840 | 0.0758 | 0.0711 |
| `pearson` | 0.5715 | 0.5687 | 0.0605 | 0.0656 |
| `spearman` | 0.5415 | 0.5466 | 0.0221 | 0.0418 |
| `js_divergence` | 0.6055 | 0.5890 | 0.0708 | 0.0675 |
| `hellinger` | 0.6055 | 0.5909 | 0.0706 | 0.0663 |

Four readings.

First, and as a correctness check rather than a finding, `js_distance` and `js_divergence` agree to six decimal places on `category_roc_auc` and exactly on `human_rho`. They must, since one is a monotone transform of the other and both statistics are rank based, so this is the arithmetic confirming itself. Their $z$ columns differ slightly, 0.5909 against 0.5890, which is also correct: the count-matched null is computed on the raw scale, so a monotone transform changes the local standard deviation it divides by.

Second, `hellinger` joins them at 0.6055, matching `js_distance` to four decimal places on both AUC columns despite being a different function. The divergence family is close to interchangeable on this target, so registering all three buys resolution only for the linear models of module 9.

Third, the spread across the whole registry is narrow, 0.5415 to 0.6055 on category AUC, and every metric sits far below the Jaccard index's 0.9343 on the same scope. The conclusion of subchapter 5.2 is a property of the layer profile rather than of any particular way of comparing profiles.

Fourth, the two orderings do not agree, which is the reason both columns are reported. `cosine` is second on category AUC at 0.6010 but first on human agreement at 0.0758, ahead of `js_distance`. `wasserstein` is second worst on category AUC and worst on human agreement, so the one measure that reads depth ORDER gains nothing here, and the natural reading is that what distinguishes two concepts' profiles is which blocks they occupy rather than how far apart those blocks are. `pearson` and `spearman` are weakest on the partition question, 0.5715 and 0.5415, which matches module 3's finding that removing the global density baseline removes most of what related words share.

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

Both put the FFN expansion projections at the top and `mlp.down_proj` at the bottom, consistent with the sublayer informativeness ranking of module 1. The layer-profile AUC has a much narrower range, 0.581 to 0.692 against 0.626 to 0.931, and the reason is sample size rather than resolution. Both scopes now bin on the same 28 blocks, since the whole-model scope moved to the block axis and a sublayer scope was always one layer per block, so a sublayer profile has exactly the resolution the whole-model profile has. What separates them is that a sublayer profile is estimated from that sublayer's expert rows alone, which is a fraction of the word's experts, so it is the noisier estimate of the same quantity and its category alignment compresses toward chance accordingly.
