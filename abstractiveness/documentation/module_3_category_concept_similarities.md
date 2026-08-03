# Module 3: Concept-to-Category Expert Similarity

## Research question

Do the expert units associated with a concept overlap strongly with the expert units associated with the category that contains it, and would that support a hierarchical organization where category labels capture a meaningful parent structure?

## Analysis

Notation. For each item $c$ (concept or category label), let $E_c$ denote its set of expert units after AP filtering, where each element is a **(layer, unit) pair**:

$$E_c = \{(\ell, u) : \text{unit } u \text{ in layer } \ell \text{ is an expert for } c\} \subseteq \{1,\dots,L\} \times \mathcal{U}.$$

A neuron is identified by its (layer, unit) pair because the raw `unit` column holds only the neuron index within a layer, and these indices repeat across layers. The sets are therefore built by zipping `layer_idx` with `unit` (`set(zip(layer_idx, unit))` per concept), which gives $|E_c| = n_c$, the concept's expert count. The module iterates over every `(concept, category)` row of the metadata with a non-null category, that is, every hierarchical pair $(c, k)$ where $k$ is the parent category of concept $c$. Pairs where either $E_c$ or $E_k$ is empty are skipped.

**Analysis scopes.** Subchapter 3.1 is *set-based*, meaning it reads expert rows as an unordered collection of (layer, unit) pairs, so the layer axis never enters. Subchapter 3.2 reads the layer axis, but only as a set of bins to normalize over, so it is permutation-invariant too: unlike Geary's C or peak layer, nothing in it treats adjacent layer indices as adjacent depths. Each scope therefore runs the module exactly once, and the whole-model scope needs no `_by_block` variant. It runs on the whole model first, writing into the module folder itself, then once per sublayer type into `sublayers/<rank>_<sublayer>/`. Module 1, subchapter 1.7 defines the scopes and the rank prefix. A cross-scope summary lands in `sublayer_comparison.csv` and `sublayer_comparison.png` at the module's top level. Its columns are the pair count, the mean and median of both set-based percentages, and the mean of both layer-profile readings.

The whole-model scope uses the flat layer axis at full resolution, which is both the axis the Jaccard index already lives in and the one that retains sublayer identity, and sublayer identity is where the categorical signal is concentrated.

### 3.1 Expert-set similarity between a concept and its parent category

**Mathematical formulation.** Two set-similarity metrics are computed for each pair $(c, k)$, both expressed as percentages. The Jaccard index measures *global equivalence* of the two sets, the intersection over the union:

$$J(c, k) = 100 \cdot \frac{|E_c \cap E_k|}{|E_c \cup E_k|} \in [0, 100].$$

$J$ is symmetric and penalizes any mismatch in either direction: it can only be high when the two sets are close to identical, so a small concept set inside a much larger category set scores low even if the concept's experts are all shared. The overlap coefficient instead measures *subset containment*, the intersection over the smaller set:

$$O(c, k) = 100 \cdot \frac{|E_c \cap E_k|}{\min(|E_c|,\, |E_k|)} \in [0, 100],$$

which reaches 100 whenever one set is entirely contained in the other, regardless of the size difference. Read together, the two metrics separate two hypotheses. A high $J$ would mean concept and category recruit *the same* experts, whereas a high $O$ with a low $J$ would mean the smaller set (typically the concept's or the category's, whichever is smaller) is largely *inside* the other, a signature compatible with hierarchical containment rather than identity. Note that $J(c,k) \le O(c,k)$ always, since $|E_c \cup E_k| \ge \min(|E_c|, |E_k|)$.

**Generated data structures.** One CSV, `category_concept_similarity_metrics.csv`, with one row per concept-category pair. Alongside the two percentages, the raw set sizes behind them are stored, because a percentage alone hides its scale: an overlap of 20% resting on a 15-expert concept set is far less stable than the same 20% resting on a 1,500-expert set (one shared expert more or less moves the former by nearly 7 points), which is the same small-denominator caveat module 1 (subchapter 1.3) raises for layer percentages.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | The concept being compared. |
| category | string | $k$ | The parent category label. |
| hierarchy | string | (none) | String describing the concept-category relationship ("$k$ -> $c$"). |
| jaccard_pct | float | $J(c,k)$ | Percentage similarity computed with the Jaccard index. |
| overlap_pct | float | $O(c,k)$ | Percentage similarity computed with the overlap coefficient. |
| shared_expert_units | int | $\|E_c \cap E_k\|$ | Raw number of (layer, unit) experts shared by concept and category. |
| concept_expert_units | int | $\|E_c\|$ | Size of the concept's expert set. |
| category_expert_units | int | $\|E_k\|$ | Size of the category's expert set. |
| layer_profile_similarity_pct | float | $S(c,k)$ | Layer-profile similarity, see subchapter 3.2. |
| layer_profile_jsd_bits | float | $\mathrm{JSD}(p_c,p_k)$ | Raw Jensen-Shannon divergence between the two layer profiles, in bits. |
| layer_profile_z | float | $z(c,k)$ | Standardized excess of $S$ over the count-matched null, see subchapter 3.2. |

Example (head of `AP_0.6/3_category_concept_similarities/category_concept_similarity_metrics.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`):

| concept | category | hierarchy | jaccard_pct | overlap_pct | shared_expert_units | concept_expert_units | category_expert_units |
|---|---|---|---|---|---|---|---|
| bed | furniture | furniture -> bed | 16.7411 | 28.9575 | 75 | 264 | 259 |
| bench | furniture | furniture -> bench | 5.2632 | 12.1547 | 22 | 181 | 259 |

Two **bar charts** plot the same table, one per metric, with each pair's `hierarchy` tick label colored in its category's color (matching the palette used by module 5's heatmaps), so the category blocks are readable directly from the axis:

- `jaccard_hierarchy.png`, `jaccard_pct` ($J(c,k)$, x-axis) plotted against `hierarchy` (y-axis, one bar per concept-category pair), showing Jaccard similarity as a percentage for each concept-category hierarchy.
- `overlap_hierarchy.png`, `overlap_pct` ($O(c,k)$, x-axis) plotted against `hierarchy` (y-axis), showing the overlap coefficient as a percentage for each concept-category hierarchy.

### 3.2 Agreement in the layer distribution of the two expert sets

**Motivation.** Both metrics of subchapter 3.1 ask *which neurons* the two words share, so a concept and its category that allocate the same proportion of their experts to the same depths, without ever recruiting the same neuron, score zero on both. That is a real form of representational agreement, and it is invisible to any set metric. Subchapter 3.2 measures it directly.

**Mathematical formulation.** Let $n_{c,\ell}$ be the number of experts word $c$ holds in layer $\ell$, and $n_c = \sum_\ell n_{c,\ell}$ its total expert count. The **layer profile** of $c$ is the normalized vector

$$p_c[\ell] = \frac{n_{c,\ell}}{n_c}, \qquad \sum_{\ell=1}^{L} p_c[\ell] = 1,$$

with $L$ the number of layers in the current scope. For a pair $(c, k)$, writing $m = \tfrac{1}{2}(p_c + p_k)$ for the mixture of the two profiles and $H(p) = -\sum_\ell p[\ell]\log_2 p[\ell]$ for the Shannon entropy in bits, with the convention $0\log_2 0 = 0$, the Jensen-Shannon divergence is

$$\mathrm{JSD}(p_c, p_k) = H(m) - \tfrac{1}{2}\big(H(p_c) + H(p_k)\big) \in [0, 1].$$

It is reported as a similarity on the same scale and direction as $J$ and $O$,

$$S(c, k) = 100 \cdot \Big(1 - \sqrt{\mathrm{JSD}(p_c, p_k)}\Big) \in [0, 100].$$

The square root serves two purposes. $\sqrt{\mathrm{JSD}}$ is the Jensen-Shannon *distance*, a true metric obeying the triangle inequality, and the observed divergences bunch near zero, a region the square root expands, so the reported similarity keeps usable variance instead of saturating.

**Why Jensen-Shannon.** It is symmetric, bounded, and finite even when the two profiles have disjoint support. That last property is what rules out the alternatives: a concept holding experts in layers where its category label holds none is common, and grows more common as the AP threshold tightens, so cross-entropy and Kullback-Leibler divergence, which are infinite there and additionally asymmetric with no principled direction between a concept and a category, cannot be used. A Pearson correlation across layer bins would be dominated by the model's global density profile, which every word inherits, leaving it pinned at a high baseline with compressed range. A Spearman correlation discards magnitude and collapses into ties at strict thresholds, where most bins are empty. The histogram intersection $\sum_\ell \min(p_c[\ell], p_k[\ell])$, which is the natural distributional analogue of the overlap coefficient, carries a finite-sample bias of order $\sqrt{K/n}$ against Jensen-Shannon's $K/n$, with $K$ the number of occupied bins, because writing $\min(x,y) = \tfrac{1}{2}(x + y - |x - y|)$ shows the absolute value accumulating sampling noise rather than cancelling it. It is therefore more sensitive to expert-set size, which is the opposite of what is wanted.

**The size control.** $\mathrm{JSD}$ on normalized profiles is scale-invariant algebraically, since multiplying every count of a word by a constant leaves $p_c$ unchanged. What survives is estimation bias: $p_c$ is a multinomial estimate from $n_c$ draws, and plug-in entropy is biased low by approximately $(K-1)/(2 n \ln 2)$ bits, so a word with few experts yields a spuriously spiky profile and an inflated divergence. This cannot be normalized away, because it is a property of the sample rather than of the quantity estimated.

The control is a count-matched null built from **other real pairs**. Each pair is placed at the coordinate $(\log \min(n_c, n_k),\ \log \max(n_c, n_k))$, which is symmetric in the pair by construction, and compared against the $k$ nearest other pairs in that space, itself excluded. Writing $\mathcal{N}(c,k)$ for that reference set, $\mu_{\mathcal{N}}$ and $\sigma_{\mathcal{N}}$ for the mean and standard deviation of $S$ over it, the reported control is the standardized excess

$$z(c, k) = \frac{S(c,k) - \mu_{\mathcal{N}(c,k)}}{\sigma_{\mathcal{N}(c,k)}}.$$

A positive $z$ means the two words agree in layer allocation more than two arbitrary words of those expert counts do, and a negative $z$ means less.

The choice of reference set matters more than the arithmetic. A null that instead draws both profiles from the pooled global layer profile, at the pair's own expert counts, was tried first and rejected: it models words with no word-specific layer structure whatsoever, and since real words plainly do have idiosyncratic profiles, every real pair scored far below it, with a mean $z$ around $-12$ on Qwen3 at AP 0.6 and no pair reaching the baseline. That is a true statement about the model, layer allocation is strongly word-specific, but it is a global one, and it left the sign of $z$ carrying no information about the individual pair. The question the metric is asked is whether *these two words* agree more than *two arbitrary words* of these sizes, so the comparison set has to be arbitrary words.

This is why $z$, not $S$, is the reading to trust. On synthetic data where every word is drawn from one shared global profile, so that no pair has any true agreement, raw $S$ still correlates with the pair's smaller expert count at Spearman 0.965, that is, it is almost purely a size readout, while $z$ cuts that dependence to 0.04 at the full 204-word population.

The reference set is drawn by $k$ nearest neighbours rather than from a fixed grid over the count space, because the count distribution is skewed enough that a grid would leave its sparse corners with too few pairs to estimate $\sigma_{\mathcal{N}}$ from. Its size scales with the pool, $k = \operatorname{clip}(0.01\,m,\ 40,\ 300)$ for $m$ available pairs, so the neighbourhood stays local rather than absorbing the count gradient as the pool shrinks.

Because $z$ is defined relative to the pairs actually present, it is a **within-run ranking**. Its mean over all pairs is near zero by construction, so it answers which pairs agree more than comparable pairs, not whether words agree on depth in absolute terms, and comparisons of $z$ across AP thresholds or across scopes are comparisons of relative structure rather than of level.

Words holding fewer than 2 experts have no usable profile, so all three quantities are left empty for them, rather than set to zero as the Jaccard index is for an empty set. Zero is a true statement for a set metric, the word shares no experts, but it would be a false one here, asserting a maximally different layer distribution about a word that has no layer distribution at all. When fewer than 80 usable pairs survive in total, $z$ is left empty throughout, since a count-matched reference set cannot be formed from so few.

**Generated data structures.** Three columns added to `category_concept_similarity_metrics.csv` (see the table above) and two bar charts sharing the layout and category colors of the pair in subchapter 3.1:

- `layer_profile_similarity_hierarchy.png`, `layer_profile_similarity_pct` ($S(c,k)$, x-axis) against `hierarchy` (y-axis), the direct counterpart of `jaccard_hierarchy.png`.
- `layer_profile_z_hierarchy.png`, `layer_profile_z` ($z(c,k)$, x-axis) against `hierarchy` (y-axis), with a dashed reference line at $z = 0$ marking the count-matched null.

The same quantities are computed for every pair of words, not only concept-to-parent pairs, in module 5's `layer_profile_matrix.csv` and `layer_profile_z_matrix.csv`, and module 4 relates them to the Jaccard index over both populations. Note that the $z$ values in this module's table are drawn from the all-pairs reference set, so a concept-to-parent pair is standardized against arbitrary word pairs of comparable size rather than against other concept-to-parent pairs. A mean $z$ above zero across this table is therefore itself a finding: it would say that concept-to-parent pairs agree on depth more than arbitrary word pairs of the same sizes do.

## Results

*Scope note.* The 150-concept figures in this subsection describe the **analysis sublayer** only. The Richie-HSJ tables below are whole-model scope from the corrected `_sensefix` runs.

At AP=0.6, across the 147 concept-category pairs of the 150-concept run, Jaccard similarity is consistently low: mean 4.0%, median 2.5%, and 90% of pairs fall below 10%. By the more forgiving overlap-coefficient measure (which ignores how much larger the category's expert set is), the picture improves but is still modest: mean 15.2%, median 10.0%.

So a concept's expert set is, on average, far from identical to its category's expert set (low Jaccard), and even by the containment reading only a limited minority of the smaller set's experts are shared with the other (overlap near 15% on average). The bar charts make the pair-to-pair variation visible, but on their own they do not establish whether this is more than chance. Module 5's pairwise heatmaps test whether within-category pairs are systematically higher than across-category pairs, which is the sharper version of this question and the place where the categorical signal is clearest.

### Across AP thresholds

Both metrics decline steadily as the AP threshold tightens:

| Model | AP | n pairs | Jaccard mean / median | Overlap mean / median | % pairs with jaccard < 10% |
|---|---|---|---|---|---|
| GPT-2 | 0.5 | 197 | 8.0% / 7.3% | 23.4% / 23.4% | 68% |
| GPT-2 | 0.6 | 197 | 5.2% / 3.8% | 20.6% / 17.1% | 83% |
| GPT-2 | 0.7 | 197 | 3.1% / 0.3% | 15.6% / 5.0% | 90% |
| GPT-2 | 0.8 | 111 | 1.7% / 0.0% | 7.7% / 0.0% | 92% |
| GPT-2 | 0.9 | n/a | no output, too few surviving pairs | | |
| Qwen3 | 0.5 | 197 | 9.0% / 8.2% | 28.3% / 28.5% | 64% |
| Qwen3 | 0.6 | 197 | 7.9% / 6.1% | 28.9% / 28.2% | 70% |
| Qwen3 | 0.7 | 197 | 5.6% / 2.6% | 23.6% / 18.1% | 82% |
| Qwen3 | 0.8 | 197 | 1.9% / 0.2% | 13.9% / 3.6% | 95% |
| Qwen3 | 0.9 | 71 | 0.2% / 0.0% | 2.5% / 0.0% | 100% |

(The 150-concept sublayer-only figures this table previously carried are superseded. Whole-model scope roughly doubles every similarity, because a concept and its label can now share experts in any projection type rather than only in the FFN expansion.)

Concept-to-own-category identity overlap is low at *every* threshold: even at the most lenient AP=0.5, the median pair shares only 7.3% (GPT-2) or 8.2% (Qwen3) Jaccard, and roughly two thirds of pairs sit below 10%. By AP=0.8 the median concept-category pair shares essentially nothing (median 0.0% GPT-2, 0.2% Qwen3), and at AP=0.9 GPT-2 has too few surviving pairs to produce the table at all. Part of the decline is mechanical, since stricter AP keeps fewer experts per concept overall, shrinking every set and therefore every intersection, and n also shrinks at AP=0.8/0.9 because some concepts have too few retained experts to compute a set-based similarity at all. The overall conclusion is that a concept and its category-label word recruit largely *different* specific neurons at every threshold tested. The hierarchical relationship, to the extent modules 4 and 5 detect one, is carried by a modest shared minority of experts rather than by set identity.

### Layer-profile agreement (subchapter 3.2)

Whole-model scope, both models on Richie-HSJ, 197 concept-to-parent pairs:

| Model | AP | n | mean Jaccard | mean $S$ | median $S$ | mean $z$ | % of pairs with $z > 0$ |
|---|---|---|---|---|---|---|---|
| GPT-2 | 0.5 | 197 | 8.0% | 71.1% | 72.7% | +0.30 | 63% |
| GPT-2 | 0.6 | 197 | 5.2% | 53.8% | 54.2% | +0.13 | 55% |
| GPT-2 | 0.7 | 197 | 3.1% | 32.8% | 33.0% | +0.44 | 66% |
| GPT-2 | 0.8 | 111 | 1.7% | 29.2% | 26.0% | +1.09 | 86% |
| Qwen3 | 0.5 | 197 | 9.0% | 64.3% | 64.1% | +0.13 | 52% |
| Qwen3 | 0.6 | 197 | 7.9% | 54.7% | 55.2% | +0.03 | 51% |
| Qwen3 | 0.7 | 197 | 5.6% | 44.5% | 43.8% | +0.16 | 55% |
| Qwen3 | 0.8 | 197 | 1.9% | 29.3% | 31.5% | +0.14 | 56% |
| Qwen3 | 0.9 | 71 | 0.2% | 16.3% | 16.6% | +0.12 | 62% |

Raw $S$ looks far more encouraging than the Jaccard index, 54.7% against 7.9% for Qwen3 at AP=0.6, but that comparison is exactly the one the subchapter warns against, since $S$ is measured against a ceiling every pair approaches for reasons that have nothing to do with the pair. The interpretable column is $z$.

On Qwen3 it is flat at zero, mean between $+0.03$ and $+0.16$ with the share of pairs above zero within a few points of half, so **a concept and its own category label agree on layer allocation no more than two arbitrary words of the same expert counts do**. That is a genuine negative result and it strengthens subchapter 3.1's conclusion rather than softening it: the natural objection to a low Jaccard, that a concept and its category might occupy the same depths through different neurons, is testable and does not hold.

On GPT-2 the picture is weakly positive and grows with the threshold ($z$ = +0.30, +0.13, +0.44, +1.09 across AP 0.5 to 0.8, with 86% of pairs above zero at AP 0.8). The AP 0.8 figure should not be read as a strengthening effect: only 111 of 197 pairs survive there, and the survivors are the words with the most experts, whose profiles are the best estimated. Treat the GPT-2 column as a mild positive at lenient thresholds and as selection at strict ones. The two architectures do not agree here, so the safe statement is the Qwen3 one, that this pairing carries no reliable depth agreement.

This does not mean the metric is uninformative, only that the concept-to-label pairing is the wrong place to look for the effect. Module 5 computes the same quantity over all word pairs and finds that same-category concept *pairs* do agree on depth well above chance, with a category alignment ROC-AUC near 0.69 at every threshold. The two results together say the category-label word behaves unlike its own members, which is the same dissociation module 6 reports between the category prototype and its exemplar average.
