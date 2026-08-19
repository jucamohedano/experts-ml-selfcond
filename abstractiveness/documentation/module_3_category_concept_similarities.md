# Module 3: Concept-to-Category Expert Similarity

## Research question

Do the expert units associated with a concept overlap strongly with the expert units associated with the category that contains it, and would that support a hierarchical organization where category labels capture a meaningful parent structure?

## Analysis

Notation. For each item $c$ (concept or category label), let $E_c$ denote its set of expert units after AP filtering, where each element is a **(layer, unit) pair**:

$$E_c = \{(\ell, u) : \text{unit } u \text{ in layer } \ell \text{ is an expert for } c\} \subseteq \{1,\dots,L\} \times \mathcal{U}.$$

A neuron is identified by its (layer, unit) pair because the raw `unit` column holds only the neuron index within a layer, and these indices repeat across layers. The sets are therefore built by zipping `layer_idx` with `unit` (`set(zip(layer_idx, unit))` per concept), which gives $|E_c| = n_c$, the concept's expert count. The module iterates over every `(concept, category)` row of the metadata with a non-null category, that is, every hierarchical pair $(c, k)$ where $k$ is the parent category of concept $c$. Pairs where either $E_c$ or $E_k$ is empty are skipped.

**Analysis scopes.** Subchapter 3.1 is *set-based*, meaning it reads expert rows as an unordered collection of (layer, unit) pairs, so the layer axis never enters. Subchapter 3.2 reads the layer axis, but only as a set of bins to normalize over, so it is permutation-invariant too: unlike Geary's C or peak layer, nothing in it treats adjacent layer indices as adjacent depths. Each scope therefore runs the module exactly once, and the whole-model scope needs no `_by_block` variant. It runs on the whole model first, writing into the module folder itself, then once per sublayer type into `sublayers/<rank>_<sublayer>/`. Module 1, subchapter 1.7 defines the scopes and the rank prefix. A cross-scope summary lands in `sublayer_comparison.csv` and `sublayer_comparison.png` at the module's top level. Its columns are the pair count, the mean and median of both set-based percentages, and the mean of both layer-profile readings.

**The layer axis of subchapter 3.2.** Layer-profile agreement is measured on the BLOCK-AGGREGATED axis, sublayers summed within each transformer block, giving 28 bins on Qwen3 and 12 on GPT-2 in place of the flat 196 and 48. A bin is then a transformer block and nothing else, so $S$ and $z$ read as agreement in depth allocation, whereas on the flat axis a bin is a block crossed with a sublayer, since `build_layer_mapping_from_layers` numbers layers as `block * n_sublayers + sublayer_position + 1` and indices 1 to 7 are all block 0 on Qwen3, so the same quantity confounded allocating deep with allocating to a particular projection type. The block axis is also the less noisy estimator, since the plug-in entropy bias of about $(K-1)/(2 n \ln 2)$ bits falls sevenfold from $K = 196$ to $K = 28$ and, scaling as $1/n$, falls hardest on the small expert sets the $z$ column exists to protect against. Two things this change does NOT do. The Jensen-Shannon divergence sums over bins independently and never touches bin adjacency, so it is permutation invariant and neither axis measures depth in an ordered sense. What changes is what a bin MEANS and how noisily it is estimated, nothing about ordering. And every `sublayers/` scope is numerically unchanged, because aggregating a single-sublayer frame is an identity relabel, such a frame already holding exactly one layer per block. Set-based subchapter 3.1 is untouched, since the Jaccard index reads neuron identity and never sees the layer axis as anything but part of a unit's key. The columns of `category_concept_similarity_metrics.csv` keep their existing names, `layer_profile_similarity_pct`, `layer_profile_jsd_bits` and `layer_profile_z`, since renaming this module's reporting surface was out of scope for the change.

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

These are the reasons Jensen-Shannon is the **default**, and since the multi-metric extension of subchapter 3.3 they are no longer only arguments. The alternatives named here are now computed alongside it, so the claims about Pearson and Spearman can be read off `profile_metric_comparison.csv` rather than taken on trust, and the Results below report that both do behave as predicted.

**The size control.** $\mathrm{JSD}$ on normalized profiles is scale-invariant algebraically, since multiplying every count of a word by a constant leaves $p_c$ unchanged. What survives is estimation bias: $p_c$ is a multinomial estimate from $n_c$ draws, and plug-in entropy is biased low by approximately $(K-1)/(2 n \ln 2)$ bits, so a word with few experts yields a spuriously spiky profile and an inflated divergence. This cannot be normalized away, because it is a property of the sample rather than of the quantity estimated.

The control is a count-matched null built from **other real pairs**. Each pair is placed at the coordinate $(\log \min(n_c, n_k),\ \log \max(n_c, n_k))$, which is symmetric in the pair by construction, and compared against the $k$ nearest other pairs in that space, itself excluded. Writing $\mathcal{N}(c,k)$ for that reference set, $\mu_{\mathcal{N}}$ and $\sigma_{\mathcal{N}}$ for the mean and standard deviation of $S$ over it, the reported control is the standardized excess

$$z(c, k) = \frac{S(c,k) - \mu_{\mathcal{N}(c,k)}}{\sigma_{\mathcal{N}(c,k)}}.$$

A positive $z$ means the two words agree in layer allocation more than two arbitrary words of those expert counts do, and a negative $z$ means less.

The choice of reference set matters more than the arithmetic. A null that instead draws both profiles from the pooled global layer profile, at the pair's own expert counts, was tried first and rejected: it models words with no word-specific layer structure whatsoever, and since real words plainly do have idiosyncratic profiles, every real pair scored far below it, with a mean $z$ around $-12$ on Qwen3 at AP 0.6 and no pair reaching the baseline. That is a true statement about the model, layer allocation is strongly word-specific, but it is a global one, and it left the sign of $z$ carrying no information about the individual pair. The question the metric is asked is whether *these two words* agree more than *two arbitrary words* of these sizes, so the comparison set has to be arbitrary words.

This is why $z$, not $S$, is the reading to trust. On synthetic data where every word is drawn from one shared global profile, so that no pair has any true agreement, raw $S$ still correlates with the pair's smaller expert count at Spearman 0.965, that is, it is almost purely a size readout, while $z$ cuts that dependence to 0.04 at the full 205-word population.

One caveat belongs with that 0.04, and it is established in module 9 and recorded in full in that module's Limitations section. The residual is measured over the WHOLE pair population, which is the population the null is defined on, and it does not carry over to the concept-to-parent slice this module reports. On that slice the residual against the concept's own expert count is about $+0.21$ in Spearman rank correlation on GPT-2 at AP 0.5, on the flat axis and the block axis alike, $+0.2100$ over 1,576 pairs flat against $+0.2357$ block. The cause is structural rather than a defect of the axis or of the arithmetic. A concept-to-label pair is systematically asymmetric, since a category label word holds one of the largest expert sets in the population, and the pair's count neighbourhood is drawn from that same asymmetric family, so the null has little left to subtract. The consequence for this module is narrow but real, the $z$ column of `category_concept_similarity_metrics.csv` should not be described as size-free on the concept-to-parent population, and the near-zero $z$ reported in the Results below is a statement about depth agreement measured with a column that retains some size information.


The reference set is drawn by $k$ nearest neighbours rather than from a fixed grid over the count space, because the count distribution is skewed enough that a grid would leave its sparse corners with too few pairs to estimate $\sigma_{\mathcal{N}}$ from. Its size scales with the pool, $k = \operatorname{clip}(0.01\,m,\ 40,\ 300)$ for $m$ available pairs, so the neighbourhood stays local rather than absorbing the count gradient as the pool shrinks.

Because $z$ is defined relative to the pairs actually present, it is a **within-run ranking**. Its mean over all pairs is near zero by construction, so it answers which pairs agree more than comparable pairs, not whether words agree on depth in absolute terms, and comparisons of $z$ across AP thresholds or across scopes are comparisons of relative structure rather than of level.

Words holding fewer than 2 experts have no usable profile, so all three quantities are left empty for them, rather than set to zero as the Jaccard index is for an empty set. Zero is a true statement for a set metric, the word shares no experts, but it would be a false one here, asserting a maximally different layer distribution about a word that has no layer distribution at all. When fewer than 80 usable pairs survive in total, $z$ is left empty throughout, since a count-matched reference set cannot be formed from so few.

**Generated data structures.** Two columns per registered metric added to `category_concept_similarity_metrics.csv` (see subchapter 3.3 for the list), plus the metric-independent `layer_profile_jsd_bits`, and two bar charts per metric sharing the layout and category colors of the pair in subchapter 3.1:

- `layer_profile_<metric>_hierarchy.png`, `layer_profile_<metric>` ($S(c,k)$, y-axis) against `hierarchy` (x-axis), the direct counterpart of `jaccard_hierarchy.png`.
- `layer_profile_<metric>_z_hierarchy.png`, `layer_profile_<metric>_z` ($z(c,k)$, y-axis) against `hierarchy` (x-axis), with a dashed reference line at $z = 0$ marking the count-matched null.

The default metric additionally keeps its original column names, `layer_profile_similarity_pct` and `layer_profile_z`, holding the same values as `layer_profile_js_distance` and `layer_profile_js_distance_z`. Module 4 merges on the original names, and every results tree written before the multi-metric extension carries them, so the aliases are what keep those comparisons possible.

The same quantities are computed for every pair of words, not only concept-to-parent pairs, in module 5's `layer_profile_<metric>_matrix.csv` and `layer_profile_<metric>_z_matrix.csv`, and module 4 relates them to the Jaccard index over both populations. Note that the $z$ values in this module's table are drawn from the all-pairs reference set, so a concept-to-parent pair is standardized against arbitrary word pairs of comparable size rather than against other concept-to-parent pairs. A mean $z$ above zero across this table is therefore itself a finding: it would say that concept-to-parent pairs agree on depth more than arbitrary word pairs of the same sizes do.

### 3.3 The profile metric registry

**Motivation.** Subchapter 3.2 argues for Jensen-Shannon on grounds that rule out several alternatives, and those arguments are predictions rather than measurements. The registry turns them into measurements. "Do these two words allocate their experts to the same depths" has no single correct formalization, the candidate measures disagree by construction, and running all of them costs one extra agreement matrix each over profiles that are built once, so the choice is now reported rather than assumed.

**The seven measures.** All take the row-normalized profiles $p_c, p_k$ of subchapter 3.2 and are oriented so that a higher value means more similar, with 100 for identical profiles. Writing $P_c[\ell] = \sum_{j \le \ell} p_c[j]$ for the cumulative profile, $\langle\cdot,\cdot\rangle$ for the inner product over bins, $\bar{p} = 1/L$ for the mean of any normalized profile, and $r(p_c)$ for the vector of average ranks of $p_c$'s own bins:

| Metric | Definition | Range |
|---|---|---|
| `js_distance` | $100\,(1 - \sqrt{\mathrm{JSD}})$ | 0 to 100 |
| `js_divergence` | $100\,(1 - \mathrm{JSD})$ | 0 to 100 |
| `hellinger` | $100\,(1 - \sqrt{1 - \sum_\ell \sqrt{p_c[\ell]\,p_k[\ell]}}\,)$ | 0 to 100 |
| `cosine` | $100\,\langle p_c, p_k\rangle / (\lVert p_c\rVert\,\lVert p_k\rVert)$ | 0 to 100 |
| `pearson` | $100\,\langle p_c - \bar{p}, p_k - \bar{p}\rangle / (\lVert p_c - \bar{p}\rVert\,\lVert p_k - \bar{p}\rVert)$ | $-100$ to 100 |
| `spearman` | the `pearson` formula applied to $r(p_c)$ and $r(p_k)$ | $-100$ to 100 |
| `wasserstein` | $100\,\big(1 - \tfrac{1}{L-1}\sum_{\ell=1}^{L-1} \lvert P_c[\ell] - P_k[\ell]\rvert\big)$ | 0 to 100 |
| `profile_jaccard` | $100\,\sum_\ell \min(p_c[\ell], p_k[\ell]) \big/ \sum_\ell \max(p_c[\ell], p_k[\ell])$ | 0 to 100 |

**Implementation.** Every kernel is a thin wrapper over a scipy primitive rather than a hand-rolled formula, so the definition a reader has to trust is the library's documented one and the pipeline cannot drift from it. `js_distance` and `js_divergence` come from `cdist`'s `jensenshannon`, divided by $\sqrt{\ln 2}$ because `cdist` does not forward a `base` argument and therefore returns nats. `cosine` and `pearson` come from `cdist`'s `cosine` and `correlation`, `spearman` from `scipy.stats.rankdata` followed by the same `correlation` kernel, `hellinger` from `cdist`'s `euclidean` over square-rooted profiles divided by $\sqrt{2}$, `profile_jaccard` from `cdist`'s `braycurtis`, and `wasserstein` from `scipy.stats.wasserstein_distance`.

Two of those substitutions are worth stating because they are not merely cosmetic. Hellinger was previously computed as $\sqrt{1 - \mathrm{BC}}$, which cancels catastrophically when two profiles are similar and $\mathrm{BC}$ approaches 1: on a pair whose true distance is $2.16 \times 10^{-9}$ it returned essentially zero, a 100 percent relative error, where the Euclidean form errs by $7 \times 10^{-18}$. Similar pairs are exactly the regime a clustering reads, so this was a real defect rather than a rounding preference. Wasserstein, by contrast, was verified correct: the library agrees with the previous cumulative-difference implementation to $1.8 \times 10^{-14}$, at a cost of about 0.6 seconds per $205 \times 205$ matrix against 2 milliseconds, which is affordable at one matrix per scope.

Every kernel marks a pair NaN when either profile carries no mass at all, since an all-zero row is the absence of a profile rather than a disagreement with one. That restates at the kernel what `MIN_PROFILE_EXPERTS` enforces upstream, and it matters because the library functions disagree about the degenerate case: `jensenshannon` divides by the row sum and returns infinity, `braycurtis` returns NaN, and `wasserstein_distance` raises.

They fall into four families. The **divergence family**, `js_distance`, `js_divergence` and `hellinger`, compares the profiles as distributions and is bounded, symmetric and finite on disjoint support. The first two are monotone transforms of one another, so they rank every pair identically and differ only in spacing, which matters solely to the linear models of module 9, where a linear function of $\sqrt{\mathrm{JSD}}$ is not a linear function of $\mathrm{JSD}$. Hellinger differs genuinely, since the square root inside its sum lifts the small bins and makes it more sensitive to agreement in a profile's thin tail.

The **correlation family**, `cosine`, `pearson` and `spearman`, compares the profiles as vectors over bins. Centring is what separates the last two from the first: it removes the shared baseline every word inherits, the model's global expert density over depth, so those two read the deviation from that baseline and can go negative, meaning one word is heavy where the other is light. No member of the divergence family can express that.

`profile_jaccard` is the **L1 family**, and it is in the registry for a specific comparison rather than for variety. It is the weighted (Ruzicka) Jaccard, and on set indicator vectors it reduces exactly to the ordinary $|A \cap B| / |A \cup B|$ that subchapter 3.1 computes on expert sets, verified to machine precision. Registering it lets the pipeline's two geometries be contrasted under one functional form: "which neurons does a word recruit" and "where along depth does it put them" can then be compared without the statistic itself changing between them, which is the only way the difference between them can be attributed to the representation. Module 9's Study C is where that comparison is drawn.

Its relation to the rest is exact rather than approximate. Profiles are normalized, so $\sum_\ell \max = 2 - \sum_\ell \min$ and $\sum_\ell \min(p_c, p_k) = 1 - \mathrm{TV}$ with $\mathrm{TV}$ the total variation distance, giving

$$ J = \frac{1 - \mathrm{TV}}{1 + \mathrm{TV}}, $$

a strictly decreasing function of total variation, confirmed numerically at Spearman exactly $-1$. It therefore ranks every pair as $\mathrm{TV}$ does, so any rank-based statistic reads the two identically, while average linkage and the linear consumers of module 9 do not, since a mean of transformed distances is not the transform of a mean. One minus this agreement is a true metric on the simplex, which is what makes it safe to cluster on.

`wasserstein` stands alone as the only measure in the registry that reads bin **order**. Every other one is permutation invariant, so a word peaking at block 3 and a word peaking at block 4 are exactly as different to them as a word peaking at block 27, which is the wrong reading of a depth axis. Wasserstein charges the distance the mass has to travel, so near misses in depth score as near misses. The consequence is that it is meaningful only on an axis whose adjacency is real, that is on the block axis this module uses. On the flat layer axis adjacent bins are different projection types of the same block, so what it would measure there is largely sublayer alternation rather than depth.

**Scale is deliberately not part of the contract.** The registry fixes orientation and nothing else, because every consumer standardizes or ranks the feature, and forcing a shared scale would manufacture a false comparability between measures that are not on one scale in the first place. The practical consequence is that raw agreement columns must not be compared across metrics. The $z$ columns can be, since each is a standard score against the same count-matched null over the same pairs, and so can any rank-based statistic computed from them.

**Generated data structures.** `profile_metric_comparison.csv` and `profile_metric_comparison.png`, one row per metric, carrying `mean_agreement` and `median_agreement` for scale, `mean_z`, and `share_z_positive`, the fraction of concepts agreeing with their category label on depth more than two arbitrary words of those expert counts do. That last column is the blunt reading, and 0.5 is where a metric is finding nothing, since $z$ is centred on the null by construction. The default metric is drawn in the figure's reference color, and the title says so.

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

> **Stale, pending refresh.** Every $S$ and $z$ figure in this table comes from the PRE-RESTRUCTURE FLAT-AXIS run and is not what the current code produces, since layer-profile agreement is now measured on the block axis, see the layer-axis note in the Analysis section above. The full sweep that will refresh them is gated and has not run. On the one scope re-run so far, GPT-2 at AP 0.5, mean $S$ moves from 71.1 to 82.4 percent, median $S$ from 72.7 to 84.1, mean $z$ from +0.30 to +0.32 and the share of pairs above zero from 63 percent (124 of 197) to 67.5 percent (133 of 197), with a per-pair maximum difference of 23.19 points on $S$ and 2.23 on $z$. The mean Jaccard column is unaffected, since a set metric never reads the layer axis. Read the $S$ and $z$ columns as indicative of the previous axis only until the sweep lands. The per-sublayer replications in `sublayers/` are NOT affected and need no refresh, because aggregating a single-sublayer frame is an identity relabel, verified on all four GPT-2 sublayer scopes at a maximum difference of exactly 0.0 on both columns.


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

### Metric comparison (subchapter 3.3)

> **Provisional.** One scope of one architecture, GPT-2 whole model at AP 0.5, 197 concept-to-parent pairs. The full sweep on both architectures is gated and has not run, so read the ordering rather than the levels.

| Metric | mean $S$ | median $S$ | mean $z$ | % of pairs with $z > 0$ |
|---|---|---|---|---|
| `js_distance` | 82.4 | 84.1 | +0.319 | 67.5% |
| `wasserstein` | 92.4 | 93.5 | +0.268 | 64.0% |
| `cosine` | 92.0 | 94.1 | +0.248 | 70.1% |
| `pearson` | 70.1 | 78.2 | +0.106 | 66.0% |
| `spearman` | 70.9 | 82.1 | +0.116 | 69.5% |
| `js_divergence` | 96.4 | 97.5 | +0.280 | 70.6% |
| `hellinger` | 85.2 | 86.8 | +0.319 | 68.0% |

Three readings, and the mean $S$ column supports none of them, which is the point of reporting scale separately from orientation. `js_divergence` leads it at 96.4 purely because dropping the square root compresses everything toward the ceiling, and that is a property of the transform rather than of the data.

First, the qualitative conclusion of subchapter 3.2 does not depend on the metric. Every one of the seven gives a small positive mean $z$, between $+0.11$ and $+0.32$, with the share of pairs above zero between 64 and 71 percent. No measure turns the weak GPT-2 positive into a strong effect and none reverses it, so the choice of formalization is not what is holding the result down.

Second, the centred measures are the weakest on this pairing, $+0.106$ for `pearson` and $+0.116$ for `spearman` against $+0.319$ for `js_distance`, which is what subchapter 3.2 predicted for them and the first direct evidence for it. Removing the global density baseline removes most of what a concept and its category label share.

Third, `js_distance` and `hellinger` agree to three decimal places on mean $z$, $+0.3194$ against $+0.3193$, despite being different functions. They order the pairs almost identically, so registering both buys very little on this target, and the same near-duplication shows up in module 5.
