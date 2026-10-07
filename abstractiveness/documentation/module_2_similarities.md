# Module 2: Similarities

Module 2 measures how alike two words' expert sets and layer profiles are, and relates those similarities to human judgments of similarity, to typicality and to word frequency. It gathers what used to be four modules: the concept-to-category similarities (former module 3), the pairwise heatmaps and their validation against human ratings (former module 5), cosine typicality (former module 7), and every correlation panel that involves a similarity, a typicality or a frequency (former module 4). The one correlation about the layer distribution itself, entropy against expert count, is section 1.2 of module 1.

Outputs are written to `AP_<t>/2_similarities/`, one folder per section plus one folder grouping the per-metric layer-profile outputs, so the eight registered profile metrics do not flood the section folders:

| Section | Folder | Contents |
|---|---|---|
| 2.1 Category concept similarities | `2.1_category_concept_similarities/` | each concept against its category label: Jaccard, overlap, layer-profile agreement, and Jaccard against layer profile on these pairs |
| 2.2 Pairwise similarities | `2.2_pairwise_similarities/` | every word pair: Jaccard, overlap and shared-count matrices, heatmaps, category contrast, and Jaccard against layer profile over all pairs |
| 2.3 Human similarity validation | `2.3_human_similarity_validation/` | every pairwise matrix against Richie and Bhatia's human pair ratings |
| 2.4 Typicality | `2.4_typicality/` | cosine typicality, and every correlation with human typicality |
| 2.5 Frequency correlations | `2.5_frequency_correlations/` | the corpus frequency matched to the model, OpenWebText for GPT-2 and FineWeb for Qwen3, against the expert measures and human typicality, and Jaccard against that frequency controlling for human typicality |
| per-metric layer profiles | `layer_profile_measures/` | `hierarchy/` (section 2.1), `heatmaps/` (sections 2.2 and 2.3) and `similarity_correlation/` (section 2.3) |

Every section runs once per analysis scope. The whole model writes to the folders above, and a sublayer scope computes only its `sublayer_comparison` rows unless the executor's `WRITE_SUBLAYER_OUTPUTS` is switched on, in which case it writes the same folders under `sublayers/<rank>_<sublayer>/`. Module 1 defines the scopes, the rank prefix and the two layer axes in *Analysis scopes and the two layer axes*. Example tables below cite the run that produced them, in that run's folder layout.

## Research question

Do the expert units associated with a concept overlap strongly with those of the category that contains it, which would support a hierarchical organization where category labels capture a meaningful parent structure? Beyond that one pairing, does the expert structure form recognizable clusters, with concepts of the same category sharing more experts with each other than with concepts of other categories, and does the similarity of two words' expert sets predict how similar people judge them to be?

Two further questions concern single words. Does the expert structure of a category recover human judgments of typicality, with concepts humans rate as more typical also closer to the category's expert-allocation prototype? And are concepts that are more typical or more frequent also associated with more experts or stronger category alignment, and does the typicality-alignment relationship survive controlling for frequency?

## Analysis

Notation. For each item $c$ (concept or category label), let $E_c$ denote its set of expert units after AP filtering, where each element is a **(layer, unit) pair**:

$$E_c = \{(\ell, u) : \text{unit } u \text{ in layer } \ell \text{ is an expert for } c\} \subseteq \{1,\dots,L\} \times \mathcal{U}.$$

A neuron is identified by its (layer, unit) pair because the raw `unit` column holds only the neuron index within a layer, and these indices repeat across layers. The sets are therefore built by zipping `layer_idx` with `unit` (`set(zip(layer_idx, unit))` per concept), which gives $|E_c| = n_c$, the concept's expert count. Section 2.1 iterates over every `(concept, category)` row of the metadata with a non-null category, that is, every hierarchical pair $(c, k)$ where $k$ is the parent category of concept $c$. Pairs where either $E_c$ or $E_k$ is empty are skipped.

### Foundations, shared by the five sections

#### Analysis scopes and the layer axis

Sections 2.1 to 2.3 are *set-based* or read the layer axis only as a set of bins to normalize over, so they are permutation-invariant and each scope runs them once, with no axis variants. Section 2.4 is mixed. Its global prototype is set-based and is fed the scope's own flat expert rows, since on a block-aggregated frame units sharing an index across different projections of one block would collapse into a single feature and inflate every prototype. Its per-layer prototypes are order-based, so in the whole-model scope they are produced twice, suffixed `_by_block` and `_by_layer`, and once unsuffixed in a sublayer scope.

Each section writes one `sublayer_comparison.csv` and `.png` in its folder, one row per scope. Section 2.1 reports the pair count, the mean and median Jaccard and overlap, the mean layer-profile agreement and its $z$, and the $r$ of Jaccard against layer profile. Section 2.2 reports the mean within-category and across-category Jaccard, their difference, the category-alignment ROC-AUC of the Jaccard matrix and of the default metric's agreement and $z$ matrices, and the $r$ of both all-pairs panels. Section 2.3 reports the mean over categories of the Spearman agreement with the human ratings, for the Jaccard index and the default layer profile. Section 2.4 reports the pooled Pearson $r$ of cosine against human typicality with its concept count, and the $r$ of human typicality against Jaccard and of Jaccard against cosine typicality. Section 2.5 reports the $r$ of US frequency against expert count. The ROC-AUC columns are the density-robust ones, since they contrast pairs within a scope, and they are computed over level-2 concepts only, since category-label words have no same-category peers. With `WRITE_SUBLAYER_OUTPUTS` off, a sublayer scope computes only the default profile metric and skips the Mantel permutations of section 2.3, since the comparison rows report nothing else.

**The layer axis of the layer-profile measures.** Layer-profile agreement is measured on the BLOCK-AGGREGATED axis, sublayers summed within each transformer block, giving 28 bins on Qwen3 and 12 on GPT-2 in place of the flat 196 and 48. A bin is then a transformer block and nothing else, so $S$ and $z$ read as agreement in depth allocation, whereas on the flat axis a bin is a block crossed with a sublayer, since `build_layer_mapping_from_layers` numbers layers as `block * n_sublayers + sublayer_position + 1` and indices 1 to 7 are all block 0 on Qwen3, so the same quantity confounded allocating deep with allocating to a particular projection type. The block axis is also the less noisy estimator, since the plug-in entropy bias of about $(K-1)/(2 n \ln 2)$ bits falls sevenfold from $K = 196$ to $K = 28$ and, scaling as $1/n$, falls hardest on the small expert sets the $z$ column exists to protect against. Two things this change does NOT do. The Jensen-Shannon divergence sums over bins independently and never touches bin adjacency, so it is permutation invariant and neither axis measures depth in an ordered sense. What changes is what a bin MEANS and how noisily it is estimated, nothing about ordering. And every `sublayers/` scope is numerically unchanged, because aggregating a single-sublayer frame is an identity relabel, such a frame already holding exactly one layer per block. The set-based measures are untouched, since the Jaccard index reads neuron identity and never sees the layer axis as anything but part of a unit's key. The columns of `category_concept_similarity_metrics.csv` keep their existing names, `layer_profile_similarity_pct`, `layer_profile_jsd_bits` and `layer_profile_z`, since renaming this module's reporting surface was out of scope for the change.

#### Correlation panels and the coverage rule

Sections 2.1, 2.2, 2.4 and 2.5 carry regression panels, and module 1's section 1.2 carries one more, all drawn through `core/correlation_reporting.py`. `regression_row` applies the coverage rule below and computes the Pearson statistic, `run_regression_panel` draws a single panel with its CSV, and `run_regression_grid` draws one row of panels sharing an x variable, each panel dropping only its own missing rows. Every panel is a scatter in the shared correlation colours with its least-squares line and 95% band, titled with what it plots and with the coefficient, $p$ and $n$ on a grey line under the title.

**Mathematical formulation.** Every panel tests one variable pair $(x, y)$ for a linear association. For a pair with $n$ complete observations $(x_i, y_i)$ (rows with a missing value in either variable are dropped per panel), the Pearson correlation coefficient is

$$r_{xy} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2}\;\sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}} \in [-1, 1],$$

with the two-sided p-value obtained from the exact null distribution used by `scipy.stats.pearsonr` (equivalently, from the statistic $t = r\sqrt{(n-2)/(1-r^2)}$ under a $t_{n-2}$ reference). $r_{xy}$ measures only the *linear* component of the relationship, while $r_{xy}^2$ is the fraction of variance in $y$ that a linear function of $x$ would explain, which is the effect-size reading used in the Results. Each panel also fits and draws the ordinary-least-squares line $\hat{y} = \beta x + \alpha$ with $\beta = \operatorname{Cov}(x,y)/\operatorname{Var}(x)$, $\alpha = \bar{y} - \beta\bar{x}$ (rendered by `seaborn.regplot` with its confidence band).

**Where the missing values come from.** The panels never generate missingness themselves, they only inherit gaps that were created upstream, and a concept can drop out of a panel for two structurally different reasons.

The first is a value that is genuinely null in the source metadata, independent of the AP threshold. `human_typicality` is null for every category-label word (`furniture`, `clothing`, and so on), since those are root nodes with no HSJ pairwise rating of their own, and it can also be null for the small number of concepts the HSJ rating procedure did not cover. `frequency` (and therefore `log_frequency`, computed only when `frequency > 0` in `expert_counts_with_metadata`) is null or non-positive for the handful of concepts absent from the Wikipedia frequency source. These gaps show up as `NaN` inside an otherwise present row, so `dropna` removes only that row from only that panel.

The second, and the dominant one as AP tightens, is a concept disappearing entirely from an upstream table because it failed a structural condition, not because a cell is `NaN`. Three such conditions feed the panels:

- `expert_count` and everything derived from it: `expert_counts_with_metadata` builds the base table with `expert_allocation_df.groupby("concept").size()` then an inner merge onto the metadata. A concept with zero expert units surviving the AP filter has no rows to group, so it is absent from the grouped table, not present with a zero. The inner merge then drops it from `merged_metadata_df` entirely, and every panel built on that table (the expert-count panels of sections 2.4 and 2.5, and module 1's `shannon_entropy_vs_expert_count`) loses it along with it.
- `jaccard_pct` and `overlap_pct`: section 2.1 only emits a row for a concept when *both* the concept's own expert set and its parent category's expert set are non-empty (`if u_concept and u_category` in `plot_hierarchy_similarities`). Root category-label words are excluded even earlier, by `concept_metadata.dropna(subset=["category"])`, since a label is never its own child. As a consequence, if the *category label itself* loses all its experts at a strict threshold, every member concept of that category loses its Jaccard and overlap value too, not just the label.
- `global_cosine_typicality`: section 2.4 skips any category with fewer than two member concepts that still have expert data (`if len(valid_members) < 2: continue`), so every concept in an under-populated category is absent from the Jaccard against cosine typicality panel.

In short, `NaN` in a retained row means a rating was never collected, while a concept's total absence from a panel means it (or its category) ran out of surviving expert units at the current AP threshold. The two are handled the same way by `dropna`, but only the second one is threshold-dependent, and it drives the coverage collapse at strict thresholds.

**When a correlation is reported.** Every panel is gated on two conditions together, both defined as module constants (`MIN_ABSOLUTE_N = 30`, `MIN_COVERAGE_FRACTION = 0.75`):

$$n \ge \texttt{MIN\_ABSOLUTE\_N} \quad \text{and} \quad \frac{n}{n_{\text{total}}} \ge \texttt{MIN\_COVERAGE\_FRACTION},$$

where $n$ is the number of complete pairs after `dropna` and $n_{\text{total}}$ is a fixed count of concepts that could in principle have contributed to that panel, read once from `concept_metadata` and independent of the AP threshold: the full metadata row count for panels that only need metadata and expert counts, or the count of concepts with a defined category for the Jaccard/overlap/Cosine-Typicality panels (root labels never appear there by construction and are excluded from the denominator). Because $n_{\text{total}}$ does not move with the AP threshold, `coverage_pct` in the summary reports how much of the concept set a given panel rests on at any threshold. The gate suppresses reporting on small or non-representative subsets, since survivors at a strict AP are systematically the higher-frequency or better-populated concepts rather than a random sample.

When a panel clears both bars, the Pearson statistic and the fitted regression line are computed and drawn. When it does not, the scatter of the remaining points is drawn unfitted (with the coverage shortfall on the grey line under the title) and the CSV holds the cleaned data, but `pearson_r`/`pearson_p` are left empty in `correlation_summary.csv`. Every panel gets a summary row either way, with `n_points`, `n_total_relevant`, and `coverage_pct` always populated. A skipped panel is not a fault, so the log carries one line per scope counting the skipped panels with their range of $n$ and coverage, while the per-panel detail is written at DEBUG level only.

**Generated data structures.** Each section with panels writes its own `correlation_summary.csv`, one row per panel of that section, recording the Pearson statistic computed on the corresponding $(x, y)$ columns.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| plot_name | string | (none) | Identifier of the corresponding regression/plot (matches the `.png` file stem). |
| x_variable | string | $x$ | Name of the column used on the x-axis. |
| y_variable | string | $y$ | Name of the column used on the y-axis. |
| pearson_r | float or empty | $r_{xy}$ | Pearson correlation coefficient, left empty when the panel did not clear the coverage bar (see above). |
| pearson_p | float or empty | $p$ | Two-sided p-value, left empty under the same condition as pearson_r. |
| n_points | int | $n$ | Number of $(x, y)$ pairs used after dropping missing values. |
| n_total_relevant | int | $n_{\text{total}}$ | Fixed count of concepts that could in principle contribute to this panel (read from `concept_metadata`, independent of AP), the coverage denominator. |
| coverage_pct | float | $100 \, n / n_{\text{total}}$ | Percentage of the relevant concepts actually retained. A correlation is only computed when `n_points >= 30` and `coverage_pct >= 75`. |
| controlling_for | string | $z$ | Control variable, only populated for the partial-correlation row of section 2.4, and empty otherwise. |

#### The profile metric registry

**Motivation.** Section 2.1 argues for Jensen-Shannon on grounds that rule out several alternatives, and those arguments are predictions rather than measurements. The registry turns them into measurements. "Do these two words allocate their experts to the same depths" has no single correct formalization, the candidate measures disagree by construction, and running all of them costs one extra agreement matrix each over profiles that are built once, so the choice is now reported rather than assumed.

**The eight measures.** All take the row-normalized profiles $p_c, p_k$ of section 2.1 and are oriented so that a higher value means more similar, with 100 for identical profiles. Writing $P_c[\ell] = \sum_{j \le \ell} p_c[j]$ for the cumulative profile, $\langle\cdot,\cdot\rangle$ for the inner product over bins, $\bar{p} = 1/L$ for the mean of any normalized profile, and $r(p_c)$ for the vector of average ranks of $p_c$'s own bins:

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

They fall into four families. The **divergence family**, `js_distance`, `js_divergence` and `hellinger`, compares the profiles as distributions and is bounded, symmetric and finite on disjoint support. The first two are monotone transforms of one another, so they rank every pair identically and differ only in spacing, which matters solely to the linear models of module 5, where a linear function of $\sqrt{\mathrm{JSD}}$ is not a linear function of $\mathrm{JSD}$. Hellinger differs genuinely, since the square root inside its sum lifts the small bins and makes it more sensitive to agreement in a profile's thin tail.

The **correlation family**, `cosine`, `pearson` and `spearman`, compares the profiles as vectors over bins. Centring is what separates the last two from the first: it removes the shared baseline every word inherits, the model's global expert density over depth, so those two read the deviation from that baseline and can go negative, meaning one word is heavy where the other is light. No member of the divergence family can express that.

`profile_jaccard` is the **L1 family**, and it is in the registry for a specific comparison rather than for variety. It is the weighted (Ruzicka) Jaccard, and on set indicator vectors it reduces exactly to the ordinary $|A \cap B| / |A \cup B|$ that section 2.1 computes on expert sets, verified to machine precision. Registering it lets the pipeline's two geometries be contrasted under one functional form: "which neurons does a word recruit" and "where along depth does it put them" can then be compared without the statistic itself changing between them, which is the only way the difference between them can be attributed to the representation. Module 5's Study C is where that comparison is drawn.

Its relation to the rest is exact rather than approximate. Profiles are normalized, so $\sum_\ell \max = 2 - \sum_\ell \min$ and $\sum_\ell \min(p_c, p_k) = 1 - \mathrm{TV}$ with $\mathrm{TV}$ the total variation distance, giving

$$ J = \frac{1 - \mathrm{TV}}{1 + \mathrm{TV}}, $$

a strictly decreasing function of total variation, confirmed numerically at Spearman exactly $-1$. It therefore ranks every pair as $\mathrm{TV}$ does, so any rank-based statistic reads the two identically, while average linkage and the linear consumers of module 5 do not, since a mean of transformed distances is not the transform of a mean. One minus this agreement is a true metric on the simplex, which is what makes it safe to cluster on.

`wasserstein` stands alone as the only measure in the registry that reads bin **order**. Every other one is permutation invariant, so a word peaking at block 3 and a word peaking at block 4 are exactly as different to them as a word peaking at block 27, which is the wrong reading of a depth axis. Wasserstein charges the distance the mass has to travel, so near misses in depth score as near misses. The consequence is that it is meaningful only on an axis whose adjacency is real, that is on the block axis this module uses. On the flat layer axis adjacent bins are different projection types of the same block, so what it would measure there is largely sublayer alternation rather than depth.

**Scale is deliberately not part of the contract.** The registry fixes orientation and nothing else, because every consumer standardizes or ranks the feature, and forcing a shared scale would manufacture a false comparability between measures that are not on one scale in the first place. The practical consequence is that raw agreement columns must not be compared across metrics. The $z$ columns can be, since each is a standard score against the same count-matched null over the same pairs, and so can any rank-based statistic computed from them.

**Generated data structures.** `profile_metric_comparison.csv` and `profile_metric_comparison.png`, one row per metric, carrying `mean_agreement` and `median_agreement` for scale, `mean_z`, and `share_z_positive`, the fraction of concepts agreeing with their category label on depth more than two arbitrary words of those expert counts do. That last column is the blunt reading, and 0.5 is where a metric is finding nothing, since $z$ is centred on the null by construction. The default metric is drawn in the figure's reference color, and the title says so.

### 2.1 Category concept similarities

Each concept against its own category label. Written to `2.1_category_concept_similarities/`, with the per-metric layer-profile files in `layer_profile_measures/hierarchy/`.

#### Expert-set similarity between a concept and its parent category

**Mathematical formulation.** Two set-similarity metrics are computed for each pair $(c, k)$, both expressed as percentages. The Jaccard index measures *global equivalence* of the two sets, the intersection over the union:

$$J(c, k) = 100 \cdot \frac{|E_c \cap E_k|}{|E_c \cup E_k|} \in [0, 100].$$

$J$ is symmetric and penalizes any mismatch in either direction: it can only be high when the two sets are close to identical, so a small concept set inside a much larger category set scores low even if the concept's experts are all shared. The overlap coefficient instead measures *subset containment*, the intersection over the smaller set:

$$O(c, k) = 100 \cdot \frac{|E_c \cap E_k|}{\min(|E_c|,\, |E_k|)} \in [0, 100],$$

which reaches 100 whenever one set is entirely contained in the other, regardless of the size difference. Read together, the two metrics separate two hypotheses. A high $J$ would mean concept and category recruit *the same* experts, whereas a high $O$ with a low $J$ would mean the smaller set (typically the concept's or the category's, whichever is smaller) is largely *inside* the other, a signature compatible with hierarchical containment rather than identity. Note that $J(c,k) \le O(c,k)$ always, since $|E_c \cup E_k| \ge \min(|E_c|, |E_k|)$.

**Generated data structures.** One CSV, `category_concept_similarity_metrics.csv`, with one row per concept-category pair. Alongside the two percentages, the raw set sizes behind them are stored, because a percentage alone hides its scale: an overlap of 20% resting on a 15-expert concept set is far less stable than the same 20% resting on a 1,500-expert set (one shared expert more or less moves the former by nearly 7 points), which is the same small-denominator caveat module 1 raises for its layer percentages.

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
| layer_profile_similarity_pct | float | $S(c,k)$ | Layer-profile similarity, see below. |
| layer_profile_jsd_bits | float | $\mathrm{JSD}(p_c,p_k)$ | Raw Jensen-Shannon divergence between the two layer profiles, in bits. |
| layer_profile_z | float | $z(c,k)$ | Standardized excess of $S$ over the count-matched null, see below. |

Example (head of `AP_0.6/3_category_concept_similarities/category_concept_similarity_metrics.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`):

| concept | category | hierarchy | jaccard_pct | overlap_pct | shared_expert_units | concept_expert_units | category_expert_units |
|---|---|---|---|---|---|---|---|
| bed | furniture | furniture -> bed | 16.7411 | 28.9575 | 75 | 264 | 259 |
| bench | furniture | furniture -> bench | 5.2632 | 12.1547 | 22 | 181 | 259 |

Two **bar charts** plot the same table, one per metric, with each pair's `hierarchy` tick label colored in its category's color (matching the palette used by section 2.2's heatmaps), so the category blocks are readable directly from the axis:

- `jaccard_hierarchy.png`, `jaccard_pct` ($J(c,k)$, x-axis) plotted against `hierarchy` (y-axis, one bar per concept-category pair), showing Jaccard similarity as a percentage for each concept-category hierarchy.
- `overlap_hierarchy.png`, `overlap_pct` ($O(c,k)$, x-axis) plotted against `hierarchy` (y-axis), showing the overlap coefficient as a percentage for each concept-category hierarchy.

#### Agreement in the layer distribution of the two expert sets

**Motivation.** Both set metrics above ask *which neurons* the two words share, so a concept and its category that allocate the same proportion of their experts to the same depths, without ever recruiting the same neuron, score zero on both. That is a real form of representational agreement, and it is invisible to any set metric. The layer profile measures it directly.

**Mathematical formulation.** Let $n_{c,\ell}$ be the number of experts word $c$ holds in layer $\ell$, and $n_c = \sum_\ell n_{c,\ell}$ its total expert count. The **layer profile** of $c$ is the normalized vector

$$p_c[\ell] = \frac{n_{c,\ell}}{n_c}, \qquad \sum_{\ell=1}^{L} p_c[\ell] = 1,$$

with $L$ the number of layers in the current scope. For a pair $(c, k)$, writing $m = \tfrac{1}{2}(p_c + p_k)$ for the mixture of the two profiles and $H(p) = -\sum_\ell p[\ell]\log_2 p[\ell]$ for the Shannon entropy in bits, with the convention $0\log_2 0 = 0$, the Jensen-Shannon divergence is

$$\mathrm{JSD}(p_c, p_k) = H(m) - \tfrac{1}{2}\big(H(p_c) + H(p_k)\big) \in [0, 1].$$

It is reported as a similarity on the same scale and direction as $J$ and $O$,

$$S(c, k) = 100 \cdot \Big(1 - \sqrt{\mathrm{JSD}(p_c, p_k)}\Big) \in [0, 100].$$

The square root serves two purposes. $\sqrt{\mathrm{JSD}}$ is the Jensen-Shannon *distance*, a true metric obeying the triangle inequality, and the observed divergences bunch near zero, a region the square root expands, so the reported similarity keeps usable variance instead of saturating.

**Why Jensen-Shannon.** It is symmetric, bounded, and finite even when the two profiles have disjoint support. That last property is what rules out the alternatives: a concept holding experts in layers where its category label holds none is common, and grows more common as the AP threshold tightens, so cross-entropy and Kullback-Leibler divergence, which are infinite there and additionally asymmetric with no principled direction between a concept and a category, cannot be used. A Pearson correlation across layer bins would be dominated by the model's global density profile, which every word inherits, leaving it pinned at a high baseline with compressed range. A Spearman correlation discards magnitude and collapses into ties at strict thresholds, where most bins are empty. The histogram intersection $\sum_\ell \min(p_c[\ell], p_k[\ell])$, which is the natural distributional analogue of the overlap coefficient, carries a finite-sample bias of order $\sqrt{K/n}$ against Jensen-Shannon's $K/n$, with $K$ the number of occupied bins, because writing $\min(x,y) = \tfrac{1}{2}(x + y - |x - y|)$ shows the absolute value accumulating sampling noise rather than cancelling it. It is therefore more sensitive to expert-set size, which is the opposite of what is wanted.

These are the reasons Jensen-Shannon is the **default**, and since the multi-metric extension of the profile metric registry they are no longer only arguments. The alternatives named here are now computed alongside it, so the claims about Pearson and Spearman can be read off `profile_metric_comparison.csv` rather than taken on trust, and the Results below report that both do behave as predicted.

**The size control.** $\mathrm{JSD}$ on normalized profiles is scale-invariant algebraically, since multiplying every count of a word by a constant leaves $p_c$ unchanged. What survives is estimation bias: $p_c$ is a multinomial estimate from $n_c$ draws, and plug-in entropy is biased low by approximately $(K-1)/(2 n \ln 2)$ bits, so a word with few experts yields a spuriously spiky profile and an inflated divergence. This cannot be normalized away, because it is a property of the sample rather than of the quantity estimated.

The control is a count-matched null built from **other real pairs**. Each pair is placed at the coordinate $(\log \min(n_c, n_k),\ \log \max(n_c, n_k))$, which is symmetric in the pair by construction, and compared against the $k$ nearest other pairs in that space, itself excluded. Writing $\mathcal{N}(c,k)$ for that reference set, $\mu_{\mathcal{N}}$ and $\sigma_{\mathcal{N}}$ for the mean and standard deviation of $S$ over it, the reported control is the standardized excess

$$z(c, k) = \frac{S(c,k) - \mu_{\mathcal{N}(c,k)}}{\sigma_{\mathcal{N}(c,k)}}.$$

A positive $z$ means the two words agree in layer allocation more than two arbitrary words of those expert counts do, and a negative $z$ means less.

The choice of reference set matters more than the arithmetic. A null that instead draws both profiles from the pooled global layer profile, at the pair's own expert counts, was tried first and rejected: it models words with no word-specific layer structure whatsoever, and since real words plainly do have idiosyncratic profiles, every real pair scored far below it, with a mean $z$ around $-12$ on Qwen3 at AP 0.6 and no pair reaching the baseline. That is a true statement about the model, layer allocation is strongly word-specific, but it is a global one, and it left the sign of $z$ carrying no information about the individual pair. The question the metric is asked is whether *these two words* agree more than *two arbitrary words* of these sizes, so the comparison set has to be arbitrary words.

This is why $z$, not $S$, is the reading to trust. On synthetic data where every word is drawn from one shared global profile, so that no pair has any true agreement, raw $S$ still correlates with the pair's smaller expert count at Spearman 0.965, that is, it is almost purely a size readout, while $z$ cuts that dependence to 0.04 at the full 205-word population.

One caveat belongs with that 0.04, and it is established in module 5 and recorded in full in that module's Limitations section. The residual is measured over the WHOLE pair population, which is the population the null is defined on, and it does not carry over to the concept-to-parent slice this module reports. On that slice the residual against the concept's own expert count is about $+0.21$ in Spearman rank correlation on GPT-2 at AP 0.5, on the flat axis and the block axis alike, $+0.2100$ over 1,576 pairs flat against $+0.2357$ block. The cause is structural rather than a defect of the axis or of the arithmetic. A concept-to-label pair is systematically asymmetric, since a category label word holds one of the largest expert sets in the population, and the pair's count neighbourhood is drawn from that same asymmetric family, so the null has little left to subtract. The consequence for this module is narrow but real, the $z$ column of `category_concept_similarity_metrics.csv` should not be described as size-free on the concept-to-parent population, and the near-zero $z$ reported in the Results below is a statement about depth agreement measured with a column that retains some size information.

The reference set is drawn by $k$ nearest neighbours rather than from a fixed grid over the count space, because the count distribution is skewed enough that a grid would leave its sparse corners with too few pairs to estimate $\sigma_{\mathcal{N}}$ from. Its size scales with the pool, $k = \operatorname{clip}(0.01\,m,\ 40,\ 300)$ for $m$ available pairs, so the neighbourhood stays local rather than absorbing the count gradient as the pool shrinks.

Because $z$ is defined relative to the pairs actually present, it is a **within-run ranking**. Its mean over all pairs is near zero by construction, so it answers which pairs agree more than comparable pairs, not whether words agree on depth in absolute terms, and comparisons of $z$ across AP thresholds or across scopes are comparisons of relative structure rather than of level.

Words holding fewer than 2 experts have no usable profile, so all three quantities are left empty for them, rather than set to zero as the Jaccard index is for an empty set. Zero is a true statement for a set metric, the word shares no experts, but it would be a false one here, asserting a maximally different layer distribution about a word that has no layer distribution at all. When fewer than 80 usable pairs survive in total, $z$ is left empty throughout, since a count-matched reference set cannot be formed from so few.

**Generated data structures.** Two columns per registered metric added to `category_concept_similarity_metrics.csv` (see the profile metric registry for the list), plus the metric-independent `layer_profile_jsd_bits`, and two bar charts per metric sharing the layout and category colors of the set-metric bar charts above:

- `layer_profile_<metric>_hierarchy.png`, `layer_profile_<metric>` ($S(c,k)$, y-axis) against `hierarchy` (x-axis), the direct counterpart of `jaccard_hierarchy.png`.
- `layer_profile_<metric>_z_hierarchy.png`, `layer_profile_<metric>_z` ($z(c,k)$, y-axis) against `hierarchy` (x-axis), with a dashed reference line at $z = 0$ marking the count-matched null.

The default metric additionally keeps its original column names, `layer_profile_similarity_pct` and `layer_profile_z`, holding the same values as `layer_profile_js_distance` and `layer_profile_js_distance_z`. The panels of sections 2.1 and 2.4 read the original names, and every results tree written before the multi-metric extension carries them, so the aliases are what keep those comparisons possible.

The same quantities are computed for every pair of words, not only concept-to-parent pairs, in section 2.2's `layer_profile_<metric>_matrix.csv` and `layer_profile_<metric>_z_matrix.csv`, and sections 2.1 and 2.2 relate them to the Jaccard index over both populations. Note that the $z$ values in this module's table are drawn from the all-pairs reference set, so a concept-to-parent pair is standardized against arbitrary word pairs of comparable size rather than against other concept-to-parent pairs. A mean $z$ above zero across this table is therefore itself a finding: it would say that concept-to-parent pairs agree on depth more than arbitrary word pairs of the same sizes do.

#### Jaccard against layer-profile agreement, concept to label

Sharing neurons with the category label and allocating experts to the same depths as the label may be the same thing or two different things. The panel `jaccard_vs_layer_profile` answers this on the concept-to-label pairs, plotting $J(c,k)$ against the default metric's $S(c,k)$ with the Pearson statistic, over the categorized concepts as the coverage denominator. Section 2.2 repeats the comparison over every word pair, which is the population a downstream model consuming both quantities would see.

**Generated data structures.** `jaccard_vs_layer_profile.csv` with the two columns plotted, `jaccard_vs_layer_profile.png`, and one row in this section's `correlation_summary.csv`.

### 2.2 Pairwise similarities

Every pair of words. Written to `2.2_pairwise_similarities/`, with the per-metric matrices and heatmaps in `layer_profile_measures/heatmaps/`.

Notation. Let $\mathcal{C}$ be the full concept list from the metadata (section 2.1 compared each concept only to its own category, whereas here every concept is compared to every other concept). As in section 2.1, each concept's expert set $E_c$ contains **(layer, unit) pairs**, because the raw `unit` column holds only the neuron index within a layer. The feature axis below is therefore indexed by pairs $(\ell, u)$ rather than by bare unit indices, so that identically indexed neurons in different layers remain distinct.

#### All-pairs expert-set similarity via binary matrix products

**Mathematical formulation.** Instead of looping over pairs, the section computes all pairwise set operations at once through linear algebra on a binary presence matrix. Define

$$A \in \{0,1\}^{|\mathcal{C}| \times |\mathcal{F}|}, \qquad \mathcal{F} = \{(\ell, u)\ \text{pairs observed in the data}\}, \qquad A_{c,(\ell,u)} = \begin{cases} 1 & (\ell, u) \in E_c \\ 0 & \text{otherwise,} \end{cases}$$

built sparsely by `expert_presence_matrix` in `modules/shared_expert_set_measures.py`, which folds each (layer, unit) pair into one integer key and stores the matrix as a `scipy.sparse.csr_matrix`, reindexed over every concept in the metadata, so concepts with no retained experts appear as all-zero rows. Three derived matrices follow. The Gram matrix of $A$ counts shared (layer, unit) experts, because the inner product of two binary rows counts positions where both are 1:

$$I = A A^{\top}, \qquad I_{cd} = \sum_{(\ell,u)} A_{c,(\ell,u)} A_{d,(\ell,u)} = |E_c \cap E_d| .$$

The row sums give set sizes, $s_c = \sum_{(\ell,u)} A_{c,(\ell,u)} = |E_c| = n_c$, from which union and minimum-size matrices are assembled by broadcasting:

$$U_{cd} = s_c + s_d - I_{cd} = |E_c \cup E_d|, \qquad m_{cd} = \min(s_c, s_d).$$

The two similarity matrices are then the element-wise ratios, as percentages (with cells where the denominator is 0 set to 0):

$$J_{cd} = 100 \cdot \frac{I_{cd}}{U_{cd}} \quad \text{(Jaccard index)}, \qquad O_{cd} = 100 \cdot \frac{I_{cd}}{m_{cd}} \quad \text{(overlap coefficient)}.$$

Both are symmetric with diagonal 100 (each set is identical to itself), and $J_{cd} \le O_{cd}$ everywhere. As in section 2.1, $J$ demands near-identity of the two sets while $O$ rewards containment of the smaller set in the larger. Computed here for *all* $\binom{|\mathcal{C}|}{2}$ pairs, they answer the clustering question: if categories organize the expert space, same-category pairs $(c, d)$ should show visibly higher $J_{cd}$ or $O_{cd}$ than different-category pairs, appearing as bright blocks along the diagonal when concepts are ordered by category.

**Generated data structures.** Three square CSV matrices and two heatmap images.

##### jaccard_matrix.csv

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

##### overlap_matrix.csv

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

##### shared_expert_counts_matrix.csv

The intersection (Gram) matrix $I$ itself, saved as integers: the raw shared-expert counts behind both percentage matrices, with the diagonal holding each concept's own expert-set size $I_{cc} = n_c$. It is the scale reference for the percentages (the same caveat as module 1's allocation profiles and section 2.1): a given Jaccard percentage backed by hundreds of shared experts is a far more stable measurement than the same percentage backed by a handful, so any cell of $J$ or $O$ can be traced back to the integers that produced it.

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

**Plots.** Two **heatmap** images (`seaborn`-style matrix heatmap rendered via the `_plot_heatmap_with_leaders` helper, `magma` colormap for similarities), with concept labels on both axes (colored by category) and similarity encoded by color. Concepts are ordered by category, each block led by its root label (which counts as part of its own category), and two layers of white separator lines are drawn on both axes so every cell can be traced back to its row and column concept. A faint grid line (0.3 px, 25% opacity) is drawn at every single concept boundary, and a bold, fully opaque 1 px line is drawn on top of it at every category-block boundary, so the fine per-concept grid and the coarser category structure, along with the expected bright same-category blocks along the diagonal, can both be read directly off the matrix:

- `jaccard_heatmap.png`, encodes the values $J_{cd}$ of `jaccard_matrix.csv`.
- `overlap_heatmap.png`, encodes the values $O_{cd}$ of `overlap_matrix.csv`.

The category color system behind the tick labels and the boundary lines is shared with section 2.1's bar charts, so a category reads as the same color in both modules. The palette alternates cool and warm hues across adjacent categories in the sorted order, rather than assigning colors by simple index, so that neighboring category blocks in the heatmap stay visually distinct even when the category order places similar categories next to each other.

#### All-pairs agreement in layer distribution

**Mathematical formulation.** Section 2.1 defines the layer profile $p_c$, the Jensen-Shannon similarity $S(c,d) = 100(1 - \sqrt{\mathrm{JSD}(p_c, p_d)})$, and its count-matched null z-score $z(c,d)$, along with the full rationale for choosing Jensen-Shannon and for controlling on expert-set size. Here the same two quantities are computed for every pair in $\mathcal{C}$ rather than only for concept-to-parent pairs, giving two further square matrices alongside $J$, $O$ and $I$.

The distinction being drawn is the one the set matrices cannot see. $J_{cd}$ and $O_{cd}$ ask *which neurons* two concepts share, so two concepts allocating the same fraction of their experts to the same depths, without sharing a single neuron, sit at $J_{cd} = O_{cd} = 0$. $S(c,d)$ scores that case high. Whether the categorical block structure visible in $J$ survives in $S$, and in $z$ after size is conditioned out, is what the three ROC-AUC columns of `sublayer_comparison.csv` report.

**The layer axis.** Both matrices are built on the BLOCK-AGGREGATED axis, sublayers summed within each transformer block, giving 28 bins on Qwen3 and 12 on GPT-2 rather than the flat 196 and 48. A bin is then a transformer block and nothing else, so $S(c,d)$ reads as agreement in depth allocation, whereas on the flat axis a bin is a block crossed with a sublayer and the quantity confounded allocating deep with allocating to a particular projection type. The block axis also carries roughly one seventh of the flat axis's plug-in entropy bias, which scales as $1/n$ and therefore contaminates small expert sets most, which is the confound $z(c,d)$ exists to remove. This is not a claim about ordering. The Jensen-Shannon divergence sums over bins independently and never touches bin adjacency, so it is permutation invariant and neither axis measures depth in an ordered sense, and what changed is what a bin MEANS together with how noisily it is estimated. Every `sublayers/` scope is numerically unchanged, because aggregating a single-sublayer frame is an identity relabel. The set matrices above are untouched, since they read neuron identity rather than a layer distribution, and both matrix files keep their existing names.

**One pair of matrices per registered metric.** $S(c,d)$ is not one quantity but a family, defined by whichever entry of the profile metric registry is asked for. The profile metric registry lists the eight registered measures, gives their formulas, and explains the four families they fall into and why their raw scales are deliberately not comparable. This module computes every active one over the same profiles, built once, so each gets its own agreement matrix, its own count-matched $z$ matrix, and a heatmap for each.

Note that `wasserstein` is the only registered measure that reads bin ORDER, which is exactly why the block axis above is what makes it meaningful. On the flat layer axis the transport cost it charges would be mostly sublayer alternation rather than depth.

**Generated data structures.** Two further square CSV matrices per registered metric and their heatmaps, sharing the layout and category-colored labels described above. The agreement heatmaps keep the sequential `magma` colormap, while the $z$ heatmaps use the diverging `RdBu_r` colormap centred at zero, so a positive and a negative standard score read as opposite colours, with dark rather than white category boundary lines, which a near-white centre would hide. `<metric>` below is a registry key such as `js_distance` or `wasserstein`.

##### layer_profile_&lt;metric&gt;_matrix.csv

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| (index) | string | $c$ | Concept label for the row. |
| `<concept>` (one column per concept) | float | $S(c,d)$ | Layer-profile agreement under that metric, diagonal 100, empty for words below 2 experts. |

##### layer_profile_&lt;metric&gt;_z_matrix.csv

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| (index) | string | $c$ | Concept label for the row. |
| `<concept>` (one column per concept) | float | $z(c,d)$ | Standard deviations by which $S(c,d)$ exceeds the count-matched null, diagonal empty. |

- `layer_profile_<metric>_heatmap.png`, encodes the values $S(c,d)$ of the matching matrix.
- `layer_profile_<metric>_z_heatmap.png`, encodes the values $z(c,d)$ of the matching matrix.

Every layer-profile artifact carries its metric in the name, the default included, so the folder reads as one family rather than as a privileged file plus six additions. Results trees written before the multi-metric extension name the default metric's pair `layer_profile_matrix.csv` and `layer_profile_z_matrix.csv`, and those files are the same quantity as today's `layer_profile_js_distance_matrix.csv` and `layer_profile_js_distance_z_matrix.csv`, verified equal at a maximum absolute difference of exactly 0.0 on all four GPT-2 sublayer scopes at AP 0.5.

These matrices are the intended source for any downstream model consuming pairwise expert features, since together with `jaccard_matrix.csv` they cover both readings of pairwise agreement, shared identity and shared depth allocation, over the same concept ordering.

A note on coverage. Words holding fewer than 2 experts have no usable layer profile and are empty throughout both matrices. When the cross-scope summary reduces a matrix to its category-alignment ROC-AUC, such words are dropped as whole concepts rather than as scattered pairs, because the alignment statistic derives its same-category mask from the concept list and permutes labels across concepts, both of which require the pair pool to remain a complete triangle over one consistent concept set.

#### Jaccard against layer-profile agreement over all word pairs

**Research question.** Sections 2.1 and 2.2 describe a word pair in two ways: the Jaccard index, which asks which neurons the two words share, and the layer-profile similarity of section 2.1, which asks whether they spread their experts over the layers in the same proportions. This subchapter asks how much those two readings actually differ. If they are near-equivalent, the second carries little beyond the first. If they diverge, then two words can share neurons without agreeing on depth, or agree on depth while sharing no neuron, and the two are complementary descriptions.

Section 2.1's `jaccard_vs_layer_profile` panel answers this on the 197 concept-to-parent pairs. This panel answers it on **every unordered pair of the $|\mathcal{C}|$ words**, $\binom{205}{2} = 20{,}910$ pairs on the Richie-HSJ set, which is the population any downstream model consuming both quantities as pairwise features would see.

**Mathematical formulation.** Both quantities are read off the flat pair vectors in $\texttt{triu}(|\mathcal{C}|, k=1)$ order, the Jaccard index from `pair_similarity_vector` and the two layer-profile readings from `pair_layer_profile_vectors`. Two panels are drawn, one against the raw similarity $S$ and one against its null z-score $z$, because $S$ is largely a readout of expert-set size while $z$ is what survives conditioning on it.

Spearman's $\rho$ is the headline statistic here rather than Pearson's $r$, and both are reported. The relation is expected to be monotone but not linear: the Jaccard index is strongly zero-inflated, with most pairs sharing no expert at all, while the layer-profile similarity saturates toward its ceiling, so a linear coefficient understates an association that the ranks capture cleanly. Writing $d_i$ for the difference in ranks of pair $i$ under the two measures, over $n$ pairs with no ties,

$$\rho = 1 - \frac{6\sum_{i=1}^{n} d_i^2}{n(n^2 - 1)} \in [-1, 1].$$

**The layer axis.** Both layer-profile readings, here and in section 2.1's `jaccard_vs_layer_profile` panel, are computed on the BLOCK-AGGREGATED axis, sublayers summed within each transformer block, so 28 bins on Qwen3 and 12 on GPT-2 rather than the flat 196 and 48. A bin is then a transformer block and nothing else, so the agreement reads as depth allocation, whereas on the flat axis a bin is a block crossed with a sublayer and the quantity confounded allocating deep with allocating to a particular projection type. The block axis also carries roughly one seventh of the flat axis's plug-in entropy bias, which falls hardest on small expert sets and is exactly the contamination the $z$ column exists to remove. Neither axis measures depth in an ordered sense, since the Jensen-Shannon divergence sums over bins independently and is permutation invariant, so what changed is what a bin MEANS and how noisily it is estimated. Every `sublayers/` scope is numerically unchanged, because aggregating a single-sublayer frame is an identity relabel. The Jaccard axis of both panels is unaffected, since a set metric reads neuron identity rather than a layer distribution, and the column names of these CSVs are unchanged.

**Coverage.** The concept-level coverage rule above does not apply, because the population is pairs rather than concepts. Instead `n_total_relevant` holds the total number of unordered pairs and `n_points` the number finite on both axes, so `coverage_pct` keeps its meaning: the share of pairs that could in principle have contributed and did. Pairs are lost only when a word holds fewer than 2 experts, which leaves it without a usable layer profile.

**Generated data structures.** Two CSVs and two PNGs. Each CSV holds the per-pair values actually plotted, with a `same_category` flag, so the pair-level data behind the correlation is inspectable and reusable:

- `jaccard_vs_layer_profile_allpairs.csv`, columns `jaccard_pct`, `layer_profile_similarity_pct`, `same_category`.
- `jaccard_vs_layer_profile_z_allpairs.csv`, columns `jaccard_pct`, `layer_profile_z`, `same_category`.

At twenty thousand points a scatter plot is a solid block of ink, so both PNGs use the density treatment module 4 established for the same problem: a log-scaled **hexbin** split into same-category and different-category panels, each carrying a binned median, a straight OLS fit, and a LOWESS smooth. Showing the OLS line against the LOWESS smooth is the point, since it puts the true, often saturating, trend next to the straight line a naive linear correlation would draw.

- `jaccard_vs_layer_profile_allpairs.png`, x: `jaccard_pct` ($J_{cd}$), y: `layer_profile_similarity_pct` ($S(c,d)$).
- `jaccard_vs_layer_profile_z_allpairs.png`, x: `jaccard_pct` ($J_{cd}$), y: `layer_profile_z` ($z(c,d)$).

Both add a row to this section's `correlation_summary.csv`, which for these rows also carries the `spearman_rho` and `spearman_p` columns (left empty on every concept-level panel).

### 2.3 Human similarity validation

Every pairwise matrix of section 2.2 against the human pair ratings. Written to `2.3_human_similarity_validation/`, with the per-metric scatter panels in `layer_profile_measures/similarity_correlation/` and the metric comparison in `layer_profile_measures/heatmaps/`.

**Research question.** Every result above compares the model against itself or against the category labels of the stimulus design. Section 2.3 asks a different and harder question: does the similarity of two words' expert sets predict how similar *people* judge those two words to be?

**The human data.** Richie and Bhatia's Study 1 collected pairwise similarity ratings on a 1 to 7 scale, higher meaning more similar, for every within-category pair of this exact word list. The files are per category under `assets/Richie_and_Bhatia-HSJ/study1_pairwise_data/data_individual_level/`, one row per subject. Both senses of `squash` are admitted, as `squash__sports` and `squash__vegetables`, and rated pairs are joined to the matrices on concept keys rather than on words, so **2,418 pairs** remain across the 8 categories, with 19 to 39 raters each. `core/human_similarity_ratings.py` loads them and is shared with module 5.

**Why this is the harder test.** Humans rated within-category pairs only, so this comparison cannot use the large across-category contrast that gives the category alignment ROC-AUC its size. Telling a `robin` from a `truck` is not on the table. The question is whether, among birds alone, the expert sets know that a `chicken` is more like a `rooster` than a `crow` is like a `penguin`.

**Mathematical formulation.** For a category $k$ with rated pairs $P_k$, let $m_{cd}$ be the model similarity of pair $(c,d)$ taken from one of the matrices above, and $h_{cd}$ the subject-averaged human rating. The agreement is the Spearman rank correlation

$$\rho_k = \operatorname{corr_{Spearman}}\big(\{m_{cd}\}_{(c,d) \in P_k},\ \{h_{cd}\}_{(c,d) \in P_k}\big).$$

Ranks rather than raw values for three reasons. The layer-profile metric is compressed into a narrow high band by construction (section 2.2) so only its ordering is meaningful, the Jaccard index is bounded below at zero and heavily right-skewed, and the ratings are bounded averages of ordinal judgments. A Pearson coefficient on those scales would report how *linear* the relationship is mixed in with how strong it is, and the relationship is visibly not linear, rising steeply near zero and then saturating.

**Which coefficient and which test are independent choices.** They are easy to conflate, so both are named on every figure and in the table. Spearman against Pearson is a choice of *coefficient*, monotone association against linear association. Parametric against permutation is a choice of *null distribution*, and it applies to either coefficient. The parametric p attached to a Spearman or a Pearson coefficient alike assumes the observations are independent, which is false here, so the significance comes from the Mantel permutation instead.

**Significance.** A category's $N_k = \binom{m_k}{2}$ rated pairs are built from only $m_k$ concepts, each appearing in $m_k - 1$ of them, so the pairs are not independent and the effective information is closer to $m_k$ than to $N_k$. For birds that is 30 concepts behind 435 pairs, an overstatement of the degrees of freedom by roughly $(m_k - 1)/2 \approx 14.5$. The permutation test shuffles **concept labels within the category** and rebuilds the model vector from the same matrix, which moves every pair containing a given concept together, exactly as the dependence in the observed data does. Writing $\mathfrak{S}_{m_k}$ for the permutations of the category's concept labels and $B = 999$ draws,

$$\hat p_k = \frac{1 + \#\{b : |\rho_k^{(\pi_b)}| \ge |\rho_k^{\mathrm{obs}}|\}}{B + 1},$$

with the $+1$ in both places making the test exact rather than anticonservative and flooring the attainable value at $1/(B+1) = 0.001$, which is why every supported row reads exactly `0.001`. When either vector is constant, as at strict thresholds on thin sublayers where every rated pair of a category has Jaccard 0, $\rho_k$ is undefined and the row carries NaN for both $\rho_k$ and $\hat p_k$ (see `fixes.md`). Measured on Qwen3 at AP 0.6 for the layer profile, the parametric Pearson p calls 4 of the 8 categories significant and this test calls 1, with professions at parametric $p = 0.0008$ against Mantel $p = 0.169$. The parametric column is therefore not reported anywhere.

**Which pairs enter.** A rated pair is dropped for three reasons, all of which appear only at strict thresholds. Either concept can be missing from the concept list, the matrix cell can be NaN, which happens for the layer-profile matrices when a word holds fewer than 2 experts and so has no usable profile, or a word can hold **no experts at all**. The third case needs explicit handling because it does not produce a NaN: `expert_set_overlap_matrices` defines the Jaccard index as 0 when the union is empty, deliberately, so that an empty-set concept does not fill the heatmap with NaNs. Read as data, though, that 0 asserts that two expert sets share nothing when one of them does not exist, and feeding it to the correlation scores a missing measurement as maximal dissimilarity. Those pairs are therefore excluded by name.

Consequently $n$ is metric-dependent at strict thresholds. On Qwen3 at AP 0.9, where 10 of 205 concepts retain no experts and 4 more hold exactly one, the Jaccard vectors kept 2,277 of the 2,391 rated pairs of the runs before the squash fix while the layer-profile vectors keep 2,195. Excluding the empty-set pairs moves the per-category $\rho$ by at most 0.026 and leaves every threshold below 0.9 untouched, so it is a correctness fix rather than a result change.

The genuine zeros are kept, since two words that both hold experts and share none is a real measurement. They do dominate at AP 0.9, where 2,030 of the rated pairs are tied at exactly 0, which is the honest explanation for the Jaccard correlation falling there: Spearman has almost no ordering left to read. That is a reason to treat AP 0.9 as past the useful range for this test rather than a reason to prune further.

**Noise ceiling.** Subjects are split into halves, each half averaged per pair, the two pair vectors correlated, and the result Spearman-Brown corrected to full-sample reliability. This is the largest correlation any model could achieve against ratings this noisy, and every $\rho_k$ is also reported divided by it. The measured ceilings run from 0.836 (birds) to 0.935 (vehicles). This is also what makes the uneven rater counts harmless: unequal $n$ attenuates a correlation rather than inflating it, so it biases toward missing an effect, and whatever attenuation remains is absorbed into the ceiling.

**Two pooled figures, both reported.** `POOLED` is the rank correlation over all rated pairs at once, which lets between-category differences in mean similarity contribute. `POOLED_MEAN` is the unweighted mean of the eight per-category values, which does not. The second is the conservative reading and is the one carried into `sublayer_comparison.csv`.

**Which matrices are validated.** The Jaccard index, plus the agreement and the $z$ matrix of every registered profile metric, so the list grows with the registry rather than being written out. At the eight metrics active today that is 17 series against the 3 of the single-metric era. The cost is linear in the list, one Mantel permutation sweep per series per category, which is roughly a second per series per scope on the Richie-HSJ item set. Narrowing `ACTIVE_PROFILE_METRICS` in `modules/shared_layer_profile_measures.py` narrows this along with everything else.

**Generated data structures.** `human_similarity_validation.csv`, one row per (metric, category) plus the two pooled rows per metric, with columns `metric`, `metric_label`, `category`, `coefficient`, `test`, `n_pairs`, `rho`, `mantel_p`, `noise_ceiling`, `rho_over_ceiling`. The `metric` key of a layer-profile row is the matrix name, such as `layer_profile_wasserstein` or `layer_profile_wasserstein_z`. The `coefficient` and `test` columns state `spearman` and `mantel` explicitly rather than leaving `rho` to be guessed at. The column keeps the bare name `rho` because `scripts/tests/check_module2_human_validation.py` reads it.

Two figures.

`similarity_vs_<metric>.png` (formerly `human_vs_expert_similarity_<metric>.png`), one scatter panel per category. Each panel is titled with its category and carries the coefficient, its value, the Mantel p and $n$ on the grey line under the title, in the form `Spearman rho = 0.40, Mantel p < 0.001, n = 435`, with values at the permutation floor printed as `p < 0.001` rather than a figure 999 draws cannot justify. The p comes from the same table the figure accompanies, so the two can never disagree.

The trend drawn on each panel is an **isotonic regression**, fitted by pool adjacent violators, not a least-squares line. A straight fit is a Pearson-shaped object whose slope tracks the linear association, so placing one beside a Spearman coefficient invites reading the line as the illustration of the number when the two can disagree, professions at AP 0.6 being Pearson 0.171 against Spearman 0.105. Isotonic regression returns the best monotone function of the data, which is precisely the shape a Spearman coefficient measures, and it still exposes structure a line hides, notably furniture saturating above a Jaccard of about 10 and vegetables staying flat before rising only at its top end.

The direction is taken from the panel's own $\rho$ rather than fitted independently, so the curve can never rise while the number beside it is negative. It is fitted on the raw values, which is equivalent to fitting the ranks, since the procedure reads only the order of $x$, while the output stays in rating units and overlays the scatter directly. A useful side effect is that the number of distinct levels in the curve reads as signal strength: on Qwen3 at AP 0.5 the Jaccard panels resolve into 11 to 26 levels while the layer-profile panels flatten to 7 to 10.

This replaced a LOWESS smooth, and the reason is worth recording because the figure was actively misleading. A local smoother carries no monotonicity constraint, so the curve doubled back on itself on most panels and invited the reading that the reported $\rho$ had turned negative over part of its range. It had not. A $\rho$ of 0.16 implies a Kendall $\tau$ near 0.10, so roughly 45 percent of the point pairs run the wrong way, which is ample local disorder to bend any neighbourhood fit without saying anything about the coefficient. Two properties of this data made it worse. The ratings are averages of ordinal judgments and stack into horizontal bands, and the tails of the expert-similarity axis hold few neighbours, so the least trustworthy parts of the old curve were its two ends, which is where a reader looks first. One tail artifact does survive the change in a milder form, since an isolated extreme point becomes its own block under pool adjacent violators and the curve then jumps to meet it, so the final segment of a panel can rest on very few pairs.

`human_similarity_by_category.png`, grouped bars, one x group per category and one bar per metric. The height is $\rho_k$ divided by that category's noise ceiling, not raw $\rho_k$, so a category whose raters disagreed with each other is not charged for the model's inability to predict their noise. A bar reaching 1.0 would mean the metric agrees with the raters as well as the raters agree with each other. A `*` marks each bar whose Mantel p clears 0.05. The stars are uncorrected across the tests a scope runs, 8 categories by one series per validated matrix, which is the Jaccard index plus an agreement and a $z$ matrix for each registered metric, so 136 tests at the eight metrics active today against 24 when only the default was registered. A single starred category is therefore weaker evidence than the mark suggests, and more so now than before. The two pooled rows are excluded from this figure.

#### Ranking the profile metrics, profile_metric_comparison.csv

One row per registered metric, reducing its two matrices to the two questions this module can answer about them, so the metrics can be compared inside a scope the way `sublayer_comparison.csv` compares scopes inside a module.

| Column | Type | Description |
|--------|------|-------------|
| `metric`, `metric_label` | string | Registry key and its display name. |
| `within_pct`, `across_pct`, `contrast_pct` | float | Mean agreement over within-category pairs, over across-category pairs, and their difference. Comparable across scopes for one metric, NOT across metrics, since the registry fixes orientation and not scale. |
| `category_roc_auc` | float | Category alignment ROC-AUC of the agreement matrix. Rank based, so it IS comparable across metrics. |
| `category_roc_auc_z` | float | The same statistic on the $z$ matrix. |
| `human_rho` | float | Mean over categories of the section 2.3 correlation against the human ratings. |
| `human_rho_z` | float | The same, on the $z$ matrix. |

The two readings sit side by side on purpose, and module 4 already showed they can disagree, since the sublayer best at recovering the category partition was not the one best at reproducing human similarity. `category_roc_auc` asks whether the metric separates within-category pairs from across-category ones, a partition question with a large contrast behind it. `human_rho` asks whether it orders *within-category* pairs the way people do, which is the harder and more externally valid test. The accompanying `profile_metric_comparison.png` draws the same four columns, with the default metric in the figure's reference color.

### 2.4 Typicality

Cosine typicality and every correlation with human typicality. Written to `2.4_typicality/`, one subfolder per category for the per-category reports.

Notation. For a category $k$, let $M_k$ be its member concepts that appear in the current scope's expert data. Only categories with $|M_k| \ge 2$ are processed (a centroid of one member is just that member). The model-derived score computed here is called **Cosine Typicality**, to distinguish it from the metadata-sourced **Human Typicality** $t_c$ it is compared against.

#### Global prototype cosine typicality

**Mathematical formulation.** Within category $k$, each member concept $c \in M_k$ is encoded as a binary allocation vector over the *(layer, unit)* feature space observed among the category's members. Let $\mathcal{F}_k = \{(\ell, u) : \text{some } m \in M_k \text{ has expert } u \text{ in layer } \ell\}$ with $d_k = |\mathcal{F}_k|$. Then

$$\mathbf{x}_c \in \{0,1\}^{d_k}, \qquad (\mathbf{x}_c)_{(\ell,u)} = \begin{cases} 1 & \text{concept } c \text{ has expert unit } u \text{ in layer } \ell \\ 0 & \text{otherwise.} \end{cases}$$

(Restricting the feature space to pairs observed within the category is harmless: coordinates that are 0 for every member would change no inner product or norm.) The category's *global prototype* is the centroid, the element-wise mean of the member vectors,

$$\boldsymbol{\mu}_k = \frac{1}{|M_k|} \sum_{m \in M_k} \mathbf{x}_m \in [0,1]^{d_k},$$

whose $(\ell, u)$ coordinate is the fraction of members that use that expert. The Cosine Typicality of member $c$ is its cosine similarity to the centroid (via `sklearn.metrics.pairwise.cosine_similarity`):

$$T^{\cos}_c = \frac{\mathbf{x}_c \cdot \boldsymbol{\mu}_k}{\lVert \mathbf{x}_c \rVert \, \lVert \boldsymbol{\mu}_k \rVert} \in [0, 1],$$

which is 1 when the concept's expert pattern points in exactly the same direction as the category average and 0 when the concept shares no experts with any other member (an all-zero $\mathbf{x}_c$ is assigned similarity 0 by convention). This is the geometric analogue of prototype theory: the centroid plays the role of the category prototype, and $T^{\cos}_c$ ranks members by how well they instantiate it. Note that $c$ itself contributes $1/|M_k|$ of the centroid, so scores are inflated for tiny categories.

**Generated data structures.** One CSV, `global_prototype_typicality.csv`, with one row per concept-category pair:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| category | string | $k$ | Category label. |
| concept | string | $c$ | Concept identifier. |
| human_typicality | float | $t_c$ | Human-judged typicality score for the concept, read from the metadata. |
| global_cosine_typicality | float | $T^{\cos}_c$ | Cosine Typicality: cosine similarity between the concept vector and the global prototype vector for the category. |

Example (head of `AP_0.6/7_typicality_analysis/global_prototype_typicality.csv` in `research_plots_150_revised_executor_again`):

| category | concept | human_typicality | global_cosine_typicality |
|---|---|---|---|
| animal | alligator | 0.459 | 0.5333 |
| animal | frog | 0.575 | 0.4424 |
| animal | goldfish | 0.506 | 0.3512 |
| animal | iguana | 0.377 | 0.5413 |
| animal | leech | 0.065 | 0.4161 |

#### Per-layer prototype cosine typicality and the most-typical evolution

**Mathematical formulation.** The global prototype averages over the whole network at once, so it cannot say *where* in the network a concept is or isn't prototypical. The per-layer version repeats the construction of 7.1 within each layer $\ell$: the feature space is the set of units the category's members use at that layer, each member gets a binary vector $\mathbf{x}^{(\ell)}_c$ over those units, the layer prototype is the centroid

$$\boldsymbol{\mu}^{(\ell)}_k = \frac{1}{|M_k|} \sum_{m \in M_k} \mathbf{x}^{(\ell)}_m,$$

and the layer-wise Cosine Typicality is

$$T^{\cos}_{c,\ell} = \frac{\mathbf{x}^{(\ell)}_c \cdot \boldsymbol{\mu}^{(\ell)}_k}{\lVert \mathbf{x}^{(\ell)}_c \rVert \, \lVert \boldsymbol{\mu}^{(\ell)}_k \rVert},$$

set to 0 for every member when the category has no expert units at all in layer $\ell$ (empty feature space), and 0 for any member with no experts there. From these scores, each layer's *most typical member* is extracted:

$$c^{\ast}_{k,\ell} = \arg\max_{c \in M_k} T^{\cos}_{c,\ell},$$

so tracking $c^{\ast}_{k,\ell}$ across $\ell = 1, \dots, L$ shows whether the same concept anchors the category at every depth or the "best exemplar" rotates layer by layer.

**Generated data structures.** One module-level CSV and, per category, one CSV and one plot:

- `layer_prototype_typicality.csv`, one row per concept-category-layer combination:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| category | string | $k$ | Category label. |
| concept | string | $c$ | Concept identifier. |
| layer_idx | int | $\ell$ | Numeric layer index. |
| layer_name | string | $\ell$ (label) | Formatted layer label. |
| human_typicality | float | $t_c$ | Human-judged typicality score for the concept, read from the metadata. |
| layer_cosine_typicality | float | $T^{\cos}_{c,\ell}$ | Cosine Typicality: cosine similarity between the concept vector and the prototype vector at that layer. |

Example (head of `AP_0.6/7_typicality_analysis/layer_prototype_typicality.csv` in `research_plots_150_revised_executor_again`):

| category | concept | layer_idx | layer_name | human_typicality | layer_cosine_typicality |
|---|---|---|---|---|---|
| animal | alligator | 1 | 1.L.0.attn.c_attn | 0.459 | 0.2577 |
| animal | frog | 1 | 1.L.0.attn.c_attn | 0.575 | 0.3266 |
| animal | goldfish | 1 | 1.L.0.attn.c_attn | 0.506 | 0.4092 |
| animal | iguana | 1 | 1.L.0.attn.c_attn | 0.377 | 0.5324 |
| animal | leech | 1 | 1.L.0.attn.c_attn | 0.065 | 0.3314 |

- `<category>_most_typical_per_layer.csv` (per-category folder, e.g. `2.4_typicality/animal/animal_most_typical_per_layer_by_block.csv`): `layer_prototype_typicality.csv` filtered to the category, with the row of maximum `layer_cosine_typicality` selected per `layer_idx` (`groupby('layer_idx')['layer_cosine_typicality'].idxmax()`, i.e. $c^{\ast}_{k,\ell}$), renamed and sorted by layer. One row per model layer:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| layer_idx | int | $\ell$ | Numeric layer index. |
| layer_name | string | $\ell$ (label) | Formatted layer label. |
| most_typical_model_concept | string | $c^{\ast}_{k,\ell}$ | Concept with the highest Cosine Typicality to the layer prototype. |
| cosine_similarity_score | float | $T^{\cos}_{c^{\ast},\ell}$ | Cosine Typicality of that concept to the layer prototype. |
| hardcoded_human_score | float | $t_{c^{\ast}}$ | Human Typicality score of that same concept. |

Example (head of `AP_0.6/7_typicality_analysis/animal/animal_most_typical_per_layer.csv` in `research_plots_150_revised_executor_again`):

| layer_idx | layer_name | most_typical_model_concept | cosine_similarity_score | hardcoded_human_score |
|---|---|---|---|---|
| 1 | 1.L.0.attn.c_attn | iguana | 0.5324 | 0.377 |
| 2 | 2.L.0.attn.c_proj | iguana | 0.5687 | 0.377 |
| 3 | 3.L.0.mlp.c_fc | iguana | 0.5706 | 0.377 |
| 4 | 4.L.0.mlp.c_proj | iguana | 0.6686 | 0.377 |
| 5 | 5.L.1.attn.c_attn | iguana | 0.5887 | 0.377 |

- `<category>_most_typical_evolution.png`, a **lollipop/stem plot**: x-axis is the model layer (`layer_name`, $\ell$), y-axis is `cosine_similarity_score` ($T^{\cos}_{c^{\ast},\ell}$, "Cosine Typicality (Layer Prototype)"), with each point labeled by `most_typical_model_concept` ($c^{\ast}_{k,\ell}$).

#### Comparison with human typicality

**Mathematical formulation.** The research question is answered by correlating the model-derived and human scores within each category. Over the members of category $k$ with both scores defined,

$$r_k = \operatorname{corr}\big(t_c,\; T^{\cos}_c\big)_{c \in M_k},$$

the Pearson correlation (computed only when both variables have nonzero variance within the category), with its two-sided p-value. A positive, significant $r_k$ means members humans call typical also sit close to the category centroid in expert space. The pooled correlation over all concept-category rows, ignoring category identity, is reported in the Results as well. Note that the per-category sample sizes are small ($n = |M_k|$, typically 3 to 12), so individual $r_k$ values carry wide confidence intervals.

**Generated data structures.** Two plots per category (no new CSV, both read `global_prototype_typicality.csv`):

- `<category>_typicality_comparison_bar.png`, a vertical grouped **bar chart** of `Score` (y-axis) versus `concept` ($c$, x-axis, ordered by descending $t_c$), with bars colored by `Metric`, either "Human Typicality" ($t_c$) or "Cosine Typicality" ($T^{\cos}_c$), and titled "<Category>, human typicality and cosine typicality". Visual agreement would show both bar heights declining together left to right.
- `<category>_typicality_correlation_scatter.png`, a **scatter plot with a linear regression line**, x: `human_typicality` ($t_c$), y: `global_cosine_typicality` ($T^{\cos}_c$), one point per concept (annotated with the concept name) and the Pearson $r_k$, $p$ and $n$ on the grey line under the title.

#### Human typicality against expert count, Jaccard, overlap and frequency

**Mathematical formulation.** One grouped figure of four regression panels sharing the x variable, human typicality $t_c$, against expert count $n_c$, Jaccard $J_c$ and overlap $O_c$ with the category label, and the training-exposure frequency $f_c$ of the model being run on the Zipf scale, `frequency_zipf_openwebtext_lemma` for GPT-2 and `frequency_zipf_fineweb_lemma` for Qwen3, chosen by the config's `frequency_corpus` field. Each panel is the Pearson statistic of the coverage rule above, so the question per panel is whether more typical concepts recruit more experts, share more experts with their category label under either set measure, or are more frequent words. The Jaccard and overlap panels use the categorized concepts as their coverage denominator, the other two the full metadata count.

**Generated data structures.** `human_typicality_correlations.png`, one row of four panels, and `human_typicality_correlations.csv`, one wide table with `concept`, `category`, `human_typicality` and the four y columns. Each panel adds a row named `human_typicality_vs_<y>` to this section's `correlation_summary.csv`, with `<y>` one of `expert_count`, `jaccard`, `overlap` and `frequency_<corpus>`, where `<corpus>` is `openwebtext` or `fineweb`.

#### Partial correlation: Jaccard against human typicality, controlling for frequency

**Mathematical formulation.** If frequency were correlated with both typicality and Jaccard similarity, their pairwise correlation could be partly a frequency artifact rather than a direct typicality-alignment association. To isolate the direct component, both variables are residualized against the control $z_c = f_c$, the Zipf frequency of the corpus matched to the model, OpenWebText for GPT-2 and FineWeb for Qwen3. Zipf is $\log_{10}$ of the count plus a constant, so this is a log-frequency control on that corpus. Until 1 October 2026 the control was $\log_{10}$ of the raw Wikipedia count of the listed surface form, the old `log_frequency` column. For a variable $v$ regressed on $z$ by simple OLS,

$$\beta_v = \frac{\operatorname{Cov}(z, v)}{\operatorname{Var}(z)}, \qquad \alpha_v = \bar{v} - \beta_v \bar{z}, \qquad \tilde{v}_c = v_c - (\beta_v z_c + \alpha_v),$$

the residual $\tilde{v}_c$ is the part of $v_c$ that a linear function of frequency cannot explain. Applying this to both variables gives the residual vectors $\tilde{J}_c$ (`jaccard_resid`) and $\tilde{t}_c$ (`human_typicality_resid`), and the partial correlation is the plain Pearson correlation of the residuals:

$$r_{Jt \cdot f} = \operatorname{corr}\!\left(\tilde{J}, \tilde{t}\right),$$

which is algebraically identical to the standard partial-correlation formula $r_{Jt\cdot f} = \dfrac{r_{Jt} - r_{Jf}\, r_{tf}}{\sqrt{(1 - r_{Jf}^2)(1 - r_{tf}^2)}}$. A non-zero $r_{Jt\cdot f}$ indicates a typicality-alignment association not attributable to frequency. Missingness here is the union of the sources listed under the coverage rule for `jaccard_pct` and `human_typicality`, plus the frequency column, and the same coverage rule applies, against the categorized-concepts denominator, since this panel needs a defined category throughout.

**Generated data structures.** One CSV, one PNG, and one row appended to `correlation_summary.csv`, whose `controlling_for` names the frequency column. Both are written through `_partial_correlation`, which the frequency-side partial of section 2.5 shares:

- `partial_correlation_jaccard_human_typicality.csv`, the full working table (only rows complete in all three variables):

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Concept identifier. |
| category | string | $k$ | Parent category. |
| frequency_zipf_<corpus>_lemma | float | $z_c = f_c$ | Control variable, the model's corpus frequency on the Zipf scale. |
| jaccard_pct | float | $J_c$ | Raw Jaccard similarity to the category (section 2.1). |
| human_typicality | float | $t_c$ | Raw Human Typicality score. |
| jaccard_resid | float | $\tilde{J}_c$ | Jaccard residual after removing the linear frequency component. |
| human_typicality_resid | float | $\tilde{t}_c$ | Typicality residual after removing the linear frequency component. |

Example (head of `AP_0.6/4_correlations/partial_correlation_jaccard_typicality.csv` in `research_plots_150_revised_executor_again`, written before the rename and before the model-matched control, so its last column is today's `human_typicality_resid` and its control is the old `log_frequency`):

| concept | category | log_frequency | jaccard_pct | human_typicality | jaccard_resid | typicality_resid |
|---|---|---|---|---|---|---|
| airplane | vehicle | 4.2468 | 8.3994 | 0.58 | 3.7112 | 0.0335 |
| alligator | animal | 3.7168 | 4.0175 | 0.459 | 0.2522 | -0.0336 |
| anklet | jewelry | 2.4362 | 1.0776 | 0.378 | -0.4574 | 0.0156 |
| anvil | tool | 3.4951 | 0.4454 | 0.347 | -2.9337 | -0.1231 |
| bag | container | 4.4427 | 2.2727 | 0.696 | -2.7566 | 0.1296 |

- `partial_correlation_jaccard_human_typicality.png`, a **scatter plot with a linear regression line**, x: `human_typicality_resid` ($\tilde{t}_c$, "Human typicality, residual after OpenWebText frequency" or FineWeb), y: `jaccard_resid` ($\tilde{J}_c$, "Jaccard with category label (%), residual after" the same), with the partial $r$, $p$ and $n$ on the grey line under the title.

#### Jaccard against cosine typicality

**Mathematical formulation.** This panel reuses the global cosine typicality $T^{\cos}_c$ defined above, the cosine similarity between concept $c$'s binary expert-allocation vector and its category's centroid vector (see *Global prototype cosine typicality*), and asks whether it tracks a concept's set-based category alignment the way human typicality does in the human typicality against Jaccard panel. It computes the same Pearson statistic on the pair

$$x = T^{\cos}_c, \qquad y = J_c,$$

over the inner join of section 2.1's similarity table with this section's `global_prototype_typicality.csv` on (`concept`, `category`). It runs after the cosine typicality within this section and its row joins the same `correlation_summary.csv` as the human typicality panels, so the two "typicality vs. Jaccard" rows (human-based and model-based) can be compared directly in one file. Beyond the Jaccard-side missingness already described under the coverage rule, this panel loses a concept whenever the cosine typicality drops its entire category for having fewer than two members with expert data, and it is judged against the same categorized-concepts denominator (the cosine typicality's extra per-category minimum is not itself subtracted from that denominator, so the reported coverage is, if anything, a slight underestimate of the true eligible set, never an overestimate).

**Generated data structures.** One CSV, one PNG, and one appended summary row:

- `jaccard_vs_cosine_typicality.csv`, columns `global_cosine_typicality` ($T^{\cos}_c$), `jaccard_pct` ($J_c$).

Example (head of `AP_0.6/4_correlations/jaccard_vs_cosine_typicality.csv` in `research_plots_150_revised_executor_again`):

| global_cosine_typicality | jaccard_pct |
|---|---|
| 0.5333 | 4.0175 |
| 0.4424 | 9.7765 |
| 0.3512 | 4.7722 |
| 0.5413 | 4.2105 |
| 0.4161 | 3.4924 |

- `jaccard_vs_cosine_typicality.png`, a **scatter plot with a linear regression line**, x: `global_cosine_typicality` ($T^{\cos}_c$, "Cosine Typicality"), y: `jaccard_pct` ($J_c$, "Jaccard Similarity Index %"), with the Pearson $r$, $p$ and $n$ on the grey line under the title.

### 2.5 Frequency correlations

Whether word frequency relates to the expert measures. Written to `2.5_frequency_correlations/`.

**Mathematical formulation.** One grouped figure of four regression panels, the training-exposure frequency $f_c$ of the model being run on x, `frequency_zipf_openwebtext_lemma` for GPT-2 and `frequency_zipf_fineweb_lemma` for Qwen3, against expert count, Jaccard and overlap with the category label, and human typicality. The corpus is the config's `frequency_corpus` field, a key of `FREQUENCY_CORPORA`, and the reasons for each corpus are in the dataset and metadata page, section 6.1. The frequency against human typicality panel reads no expert data, so it is identical in every scope, and it is drawn in every scope's figure so each figure reads as a complete row. Until 1 October 2026 this section drew two figures, SUBTLEX-US and Wikipedia, for every model.

**Generated data structures.** `frequency_<corpus>_correlations.png` with its wide table `frequency_<corpus>_correlations.csv`, and one `correlation_summary.csv` row per panel named `frequency_<corpus>_vs_<y>`.

#### Partial correlation: Jaccard against frequency, controlling for human typicality

**Mathematical formulation.** The mirror image of the section 2.4 partial. There the question is whether Jaccard tracks typicality beyond what frequency explains, here it is whether Jaccard tracks frequency beyond what typicality explains. Both Jaccard $J_c$ and frequency $f_c$ are residualized on human typicality $t_c$ by the same OLS step, and

$$r_{Jf \cdot t} = \operatorname{corr}\!\left(\tilde{J}, \tilde{f}\right) = \dfrac{r_{Jf} - r_{Jt}\, r_{ft}}{\sqrt{(1 - r_{Jt}^2)(1 - r_{ft}^2)}}.$$

The two partials share the three pairwise correlations but not their numerators, $r_{Jt} - r_{Jf} r_{tf}$ against $r_{Jf} - r_{Jt} r_{ft}$, so they coincide only when $r_{Jt} = r_{Jf}$. Read together they apportion the Jaccard signal. If both survive, typicality and frequency each carry their own share. If one vanishes once the other is held fixed, the other accounts for it. The coverage rule and the categorized-concepts denominator are those of the section 2.4 partial.

**Generated data structures.** `partial_correlation_jaccard_frequency.csv`, with `concept`, `category`, the control `human_typicality`, `frequency_zipf_<corpus>_lemma`, `jaccard_pct`, `frequency_resid` and `jaccard_resid`, `partial_correlation_jaccard_frequency.png`, a scatter of `jaccard_resid` against `frequency_resid` with its regression line and the partial $r$, $p$ and $n$ under the title, and a `correlation_summary.csv` row with `controlling_for = human_typicality`.

**Caveat shared by both partials.** Human typicality is min-max normalised within each category, so it orders the members of one category and is not comparable across categories, and both partials pool all categorized concepts into one regression. A category whose members all sit high on Jaccard or on frequency can therefore move either partial through between-category differences that typicality cannot express. Adding category as a second control, or computing each partial within category, would remove that component.

## Results

### Concept-to-label similarity (section 2.1)

*Scope note.* The 150-concept figures in this subsection describe the **analysis sublayer** only. The Richie-HSJ tables below are whole-model scope from the corrected `_sensefix` runs.

At AP=0.6, across the 147 concept-category pairs of the 150-concept run, Jaccard similarity is consistently low: mean 4.0%, median 2.5%, and 90% of pairs fall below 10%. By the more forgiving overlap-coefficient measure (which ignores how much larger the category's expert set is), the picture improves but is still modest: mean 15.2%, median 10.0%.

So a concept's expert set is, on average, far from identical to its category's expert set (low Jaccard), and even by the containment reading only a limited minority of the smaller set's experts are shared with the other (overlap near 15% on average). The bar charts make the pair-to-pair variation visible, but on their own they do not establish whether this is more than chance. Section 2.2's pairwise heatmaps test whether within-category pairs are systematically higher than across-category pairs, which is the sharper version of this question and the place where the categorical signal is clearest.

#### Across AP thresholds

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

Concept-to-own-category identity overlap is low at *every* threshold: even at the most lenient AP=0.5, the median pair shares only 7.3% (GPT-2) or 8.2% (Qwen3) Jaccard, and roughly two thirds of pairs sit below 10%. By AP=0.8 the median concept-category pair shares essentially nothing (median 0.0% GPT-2, 0.2% Qwen3), and at AP=0.9 GPT-2 has too few surviving pairs to produce the table at all. Part of the decline is mechanical, since stricter AP keeps fewer experts per concept overall, shrinking every set and therefore every intersection, and n also shrinks at AP=0.8/0.9 because some concepts have too few retained experts to compute a set-based similarity at all. The overall conclusion is that a concept and its category-label word recruit largely *different* specific neurons at every threshold tested. The hierarchical relationship, to the extent sections 2.2 to 2.4 detect one, is carried by a modest shared minority of experts rather than by set identity.

#### Layer-profile agreement

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

On Qwen3 it is flat at zero, mean between $+0.03$ and $+0.16$ with the share of pairs above zero within a few points of half, so **a concept and its own category label agree on layer allocation no more than two arbitrary words of the same expert counts do**. That is a genuine negative result and it strengthens the set-based conclusion above rather than softening it: the natural objection to a low Jaccard, that a concept and its category might occupy the same depths through different neurons, is testable and does not hold.

On GPT-2 the picture is weakly positive and grows with the threshold ($z$ = +0.30, +0.13, +0.44, +1.09 across AP 0.5 to 0.8, with 86% of pairs above zero at AP 0.8). The AP 0.8 figure should not be read as a strengthening effect: only 111 of 197 pairs survive there, and the survivors are the words with the most experts, whose profiles are the best estimated. Treat the GPT-2 column as a mild positive at lenient thresholds and as selection at strict ones. The two architectures do not agree here, so the safe statement is the Qwen3 one, that this pairing carries no reliable depth agreement.

This does not mean the metric is uninformative, only that the concept-to-label pairing is the wrong place to look for the effect. Section 2.2 computes the same quantity over all word pairs and finds that same-category concept *pairs* do agree on depth well above chance, with a category alignment ROC-AUC near 0.69 at every threshold. The two results together say the category-label word behaves unlike its own members, which is the same dissociation module 3 reports between the category prototype and its exemplar average.

#### Metric comparison

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

First, the qualitative conclusion on layer-profile agreement does not depend on the metric. Every one of the seven gives a small positive mean $z$, between $+0.11$ and $+0.32$, with the share of pairs above zero between 64 and 71 percent. No measure turns the weak GPT-2 positive into a strong effect and none reverses it, so the choice of formalization is not what is holding the result down.

Second, the centred measures are the weakest on this pairing, $+0.106$ for `pearson` and $+0.116$ for `spearman` against $+0.319$ for `js_distance`, which is what section 2.1's metric argument predicted for them and the first direct evidence for it. Removing the global density baseline removes most of what a concept and its category label share.

Third, `js_distance` and `hellinger` agree to three decimal places on mean $z$, $+0.3194$ against $+0.3193$, despite being different functions. They order the pairs almost identically, so registering both buys very little on this target, and the same near-duplication shows up in section 2.2.

#### Jaccard against layer profile, concept to label

The concept-to-parent panel of section 2.1 agrees in magnitude on its much smaller population. Qwen3 gives Pearson $r$ of 0.367, 0.361, 0.518 and 0.389 at AP 0.5 through 0.8 (197, 197, 197 and 168 pairs), and GPT-2 gives 0.575, 0.417 and 0.593 at AP 0.5 through 0.7 before its coverage falls to 44% at AP 0.8 and the correlation is withheld. At AP 0.9 both models fall far below the 75% floor, so the scatter is drawn unfitted, as designed.

### Pairwise similarity (section 2.2)

*Scope note.* All values are **whole-model scope** from the corrected `_sensefix` runs, 197 concepts with a defined category. Per-sublayer replications sit in `sublayers/<rank>_<sublayer>/`, and the cross-scope contrast is summarised in `sublayer_comparison.csv`.

At AP=0.6, the same-category effect is strong in relative terms. Within-category pairs average **5.04%** Jaccard against **0.44%** across categories in Qwen3, a factor of **11.4**, and **3.42%** against **0.37%** in GPT-2, a factor of **9.3**. Same-category concepts share disproportionately many specific (layer, unit) experts, whereas a random different-category pair shares almost none. In absolute terms even within-category similarity is small, which is why the effect is far clearer in the tabulated ratio than in the rendered heatmap.

This absolute smallness is the reason the effect is clearer in the tabulated ratio than in the rendered heatmaps. On a 0 to 100 color scale the diagonal, which is 100 by construction, occupies the top of the color range, and nearly every off-diagonal cell, including the elevated same-category ones, falls in the bottom few percent of the scale and appears near-black. The relative signal is therefore evident in the numbers but faint in the image.

#### Across AP thresholds

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

#### Layer-profile agreement

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

#### Metric comparison

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

Second, `hellinger` joins them at 0.6055, matching `js_distance` to four decimal places on both AUC columns despite being a different function. The divergence family is close to interchangeable on this target, so registering all three buys resolution only for the linear models of module 5.

Third, the spread across the whole registry is narrow, 0.5415 to 0.6055 on category AUC, and every metric sits far below the Jaccard index's 0.9343 on the same scope. The conclusion on layer-profile agreement is a property of the layer profile rather than of any particular way of comparing profiles.

Fourth, the two orderings do not agree, which is the reason both columns are reported. `cosine` is second on category AUC at 0.6010 but first on human agreement at 0.0758, ahead of `js_distance`. `wasserstein` is second worst on category AUC and worst on human agreement, so the one measure that reads depth ORDER gains nothing here, and the natural reading is that what distinguishes two concepts' profiles is which blocks they occupy rather than how far apart those blocks are. `pearson` and `spearman` are weakest on the partition question, 0.5715 and 0.5415, which matches section 2.1's finding that removing the global density baseline removes most of what related words share.

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

Both put the FFN expansion projections at the top and `mlp.down_proj` at the bottom, consistent with the sublayer informativeness ranking of module 1's section 1.3. The layer-profile AUC has a much narrower range, 0.581 to 0.692 against 0.626 to 0.931, and the reason is sample size rather than resolution. Both scopes now bin on the same 28 blocks, since the whole-model scope moved to the block axis and a sublayer scope was always one layer per block, so a sublayer profile has exactly the resolution the whole-model profile has. What separates them is that a sublayer profile is estimated from that sublayer's expert rows alone, which is a fraction of the word's experts, so it is the noisier estimate of the same quantity and its category alignment compresses toward chance accordingly.

#### Jaccard against layer-profile agreement over all word pairs

Whole-model scope, over all 20,910 word pairs:

> **Stale, pending refresh.** Every layer-profile column in this table comes from the PRE-RESTRUCTURE FLAT-AXIS run, since $S$ and $z$ are now computed on the block axis, see the layer-axis note in section 2.2 above. The gated full sweep has not run. On the one scope re-run so far, GPT-2 at AP 0.5, Spearman $\rho$ of $J$ against $S$ moves from 0.40 to 0.3026, Pearson $r$ from 0.337 to 0.2618 and Spearman $\rho$ of $J$ against $z$ from 0.29 to 0.2344, all still over the same 20,910 pairs. The direction of every reading below is preserved, the two metrics remaining positively but weakly related, but the magnitudes are not current. The concept-to-parent figure quoted at the end of this subsection moves the same way, GPT-2 at AP 0.5 going from Pearson $r$ 0.575 to 0.4673. Every panel that does not read a layer profile, including every typicality and frequency panel, is axis-free and unchanged, verified on typicality against Jaccard at $r = 0.3196$ in both trees. The `sublayers/` replications are NOT affected, because aggregating a single-sublayer frame is an identity relabel, verified at a maximum difference of exactly 0.0 on all four GPT-2 sublayer scopes.

| Model | AP | pairs used | Spearman $\rho$, $J$ vs $S$ | Pearson $r$, $J$ vs $S$ | Spearman $\rho$, $J$ vs $z$ |
|---|---|---|---|---|---|
| GPT-2 | 0.5 | 20,910 | 0.40 | 0.337 | 0.29 |
| GPT-2 | 0.6 | 20,910 | 0.42 | 0.276 | 0.28 |
| GPT-2 | 0.7 | 20,910 | 0.37 | 0.189 | 0.20 |
| GPT-2 | 0.8 | 16,836 | 0.25 | 0.155 | 0.13 |
| GPT-2 | 0.9 | 3,916 | 0.10 | 0.100 | 0.08 |
| Qwen3 | 0.5 | 20,910 | 0.45 | 0.369 | 0.40 |
| Qwen3 | 0.6 | 20,910 | 0.39 | 0.288 | 0.34 |
| Qwen3 | 0.7 | 20,910 | 0.34 | 0.230 | 0.28 |
| Qwen3 | 0.8 | 20,503 | 0.31 | 0.191 | 0.23 |
| Qwen3 | 0.9 | 18,145 | 0.16 | 0.104 | 0.12 |

The two metrics are **positively but weakly related, and they diverge further as the AP threshold tightens**, from $\rho \approx 0.4$ down to $\rho \approx 0.1$. Sharing neurons and allocating experts to the same depths are therefore largely different things, which is precisely what makes the layer-profile reading worth computing rather than a restatement of the Jaccard index. At the loosest threshold the shared rank variance is about 16 to 20 percent, at the strictest about 1 to 3 percent. The pair count itself is worth noting: GPT-2 drops from 20,910 usable pairs to 3,916 by AP 0.9 while Qwen3 still has 18,145, the same thinning documented in module 1.

Pearson sits consistently below Spearman, by 0.07 to 0.10, which is the expected signature of the non-linearity that motivated reporting the rank statistic as the headline. The hexbin panels show its source directly. The Jaccard axis is heavily zero-inflated, so a dense column of pairs sits at $J = 0$ spanning the entire range of layer-profile similarity, from roughly 30% to 80% at AP=0.6. Those are pairs that share no expert at all, which the Jaccard index cannot tell apart, and which the layer-profile metric separates across nearly its whole range. Above $J = 0$ the LOWESS smooth flattens well below the OLS line, so the linear fit substantially overstates the association at high Jaccard.

The same-category and different-category panels differ sharply. At AP=0.6 the within-panel OLS gives $r = 0.34$ while the across-panel gives $r = 0.18$, and the across-panel LOWESS is nearly flat above $J \approx 2\%$. For pairs drawn from different categories the two metrics are close to unrelated.

Substituting $z$ for $S$ lowers the correlation slightly at every threshold, by 0.03 to 0.07, so the small shared component between the Jaccard index and the layer profile is partly a common dependence on expert-set size, and removing that dependence makes the two readings more nearly independent still.

### Agreement with human similarity (section 2.3)

Whole-model scope, both models, over the 2,391 rated within-category pairs of the runs before the squash fix, which dropped the 46 pairs of the two squash senses (see `fixes.md`). `mean` is the unweighted average of the eight per-category Spearman correlations, which is the conservative reading, and `pooled` is the correlation over all pairs at once. The `significant` column counts categories whose within-category Mantel test clears p < 0.05.

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

**The agreement is far more robust to thresholding than category alignment is.** The category alignment ROC-AUC of section 2.2 collapses to near chance by AP 0.9, 0.546 on Qwen3 and 0.506 on GPT-2, while human agreement over the same expert sets is still 0.312 and 0.126. Whatever survives strict filtering continues to carry graded similarity information after it has stopped carrying the category partition.

**Layer-profile agreement does not transfer to human judgments.** It sits between 0.066 and 0.199 everywhere, an order below the Jaccard column, and is the weaker reading at every threshold in both models. This is consistent with module 5's finding that the layer profile carries no ordering information on its own. The quoted range is the flat-axis one and is pending refresh with the rest of that column. The one block-axis figure available, GPT-2 at AP 0.5, is 0.0708 against the 0.123 tabulated, so moving to the block axis widens the gap against Jaccard rather than closing it and the finding is unchanged in direction.

**The two architectures differ by more than a constant.** Qwen3 roughly doubles GPT-2 at every threshold. Since both were run on identical sentences and identical human ratings, the gap is a property of the models, and it goes the same way as the expert counts: the larger model has more units clearing the AP bar, so its expert sets are better estimated.

**A caution for reading the sublayer table.** The sublayer with the best category alignment is not the one that best matches human judgment. At AP 0.5 on Qwen3, `mlp.gate_proj` leads on category alignment (AUC 0.956) but reaches only 0.468 against the human ratings, while `self_attn.o_proj` scores 0.949 and 0.580 respectively. Recovering a partition and reproducing graded similarity are different tasks and the same sublayer need not win both.

### Cosine typicality (section 2.4)

*Scope note.* The 150-concept figures below use 17 categories and the analysis sublayer only. The Richie-HSJ tables are whole-model scope from the corrected `_sensefix` runs, 197 concepts across 8 categories, with `typicality_HSJ_pairwise` as the human measure.

**The two datasets give opposite answers, and this is the most consequential disagreement in the analysis suite.** Both are reported here rather than reconciling them, since the difference is real and its cause is not isolated to one factor.

#### The 150-concept run, 17 categories, analysis sublayer

At AP=0.6, pooled across all 147 concept-category rows, the correlation between Human Typicality and global Cosine Typicality is **r=-0.001, p=0.99, no relationship at all**, and the pooled value moves monotonically from weakly positive to significantly negative as AP tightens (0.091, -0.001, -0.101, -0.149, -0.242 across AP 0.5 to 0.9), the last being significant at p=0.011. The number of categories with any positive correlation shrinks steadily (10, 9, 7, 5, 4 of 17), and by AP=0.9 zero categories are significantly positive while 3 are significantly negative. `weapon` is positive and significant at every AP from 0.5 to 0.8 (r=0.87 to 0.79), while `container` is negative and significant at every AP (r=-0.68 to -0.96).

#### The Richie-HSJ runs, 8 categories, whole model

| Model | AP | n | Pooled r | p | Positive / 8 | Sig. positive | Sig. negative |
|---|---|---|---|---|---|---|---|
| GPT-2 | 0.5 | 197 | +0.248 | <0.001 | 7 | 3 | 0 |
| GPT-2 | 0.6 | 197 | +0.198 | 0.005 | 7 | 4 | 0 |
| GPT-2 | 0.7 | 197 | +0.145 | 0.043 | 7 | 2 | 0 |
| GPT-2 | 0.8 | 188 | +0.141 | 0.054 | 7 | 2 | 0 |
| GPT-2 | 0.9 | 114 | +0.055 | 0.56 | 4 | 0 | 1 |
| Qwen3 | 0.5 | 197 | +0.454 | <0.001 | 8 | 4 | 0 |
| Qwen3 | 0.6 | 197 | +0.393 | <0.001 | 7 | 6 | 0 |
| Qwen3 | 0.7 | 197 | +0.307 | <0.001 | 7 | 6 | 0 |
| Qwen3 | 0.8 | 197 | +0.211 | 0.003 | 6 | 4 | 0 |
| Qwen3 | 0.9 | 192 | +0.060 | 0.41 | 5 | 1 | 0 |

Here the pooled correlation is **positive and significant in both models across AP 0.5 to 0.8**, reaching r=0.454 for Qwen3 at AP 0.5, and there is not a single significantly negative category anywhere in the sweep except one GPT-2 cell at AP 0.9. The decay toward zero at AP 0.9 is the familiar thinning effect rather than a reversal.

Per-category correlations (AP 0.5 / 0.6 / 0.7 / 0.8, asterisk marks p<0.05):

| Category | GPT-2 | Qwen3 |
|---|---|---|
| professions | +0.68* +0.65* +0.61* +0.51* | +0.62* +0.62* +0.65* +0.65* |
| sports | +0.40* +0.46* +0.50* +0.60* | +0.34 +0.41* +0.49* +0.50* |
| vegetables | +0.12 +0.26 +0.43 +0.33 | +0.54* +0.51* +0.62* +0.47* |
| fruit | +0.31 +0.45* +0.35 +0.33 | +0.68* +0.70* +0.60* +0.41 |
| clothing | +0.50* +0.43* +0.25 +0.01 | +0.62* +0.62* +0.52* +0.18 |
| vehicles | +0.27 +0.15 +0.07 +0.09 | +0.38 +0.49* +0.55* +0.47* |
| birds | -0.07 +0.03 +0.01 +0.06 | +0.24 +0.30 +0.17 -0.00 |
| furniture | +0.02 -0.19 -0.30 -0.31 | +0.26 -0.02 -0.13 -0.19 |

`professions` and `sports` are positive in both architectures at every threshold, and `furniture` is the one category that trends negative in both, though never significantly.

#### Reading the disagreement

Three things differ between the two runs at once, so the reversal cannot be attributed to any single one: the stimulus set (150 concepts across 17 categories versus Richie-HSJ's 197 across 8), the analysis scope (single sublayer versus whole model), and the human measure (`typicality` versus `typicality_HSJ_pairwise`). The Richie-HSJ result additionally benefits from the word-sense correction documented in `fixes.md`, which raised category alignment across the board.

The defensible statement is therefore narrower than either run alone suggests: **on the Richie-HSJ norms, with whole-model expert sets, model-derived cosine typicality does recover human typicality judgments at a modest but reliable level in both architectures**, and the earlier negative result should be scoped to the 150-concept dataset and its single-sublayer view rather than treated as the general finding. A direct test of which factor drives the difference would require rerunning the 150-concept dataset at whole-model scope, which has not been done.

### Typicality and frequency correlations (sections 2.4 and 2.5)

> **Provenance of these figures.** Since 1 October 2026 every frequency panel and the section 2.4 partial read the model-matched corpus frequency, OpenWebText for GPT-2 and FineWeb for Qwen3, so the frequency numbers below are stale until the full sweep is refreshed. On GPT-2 at AP 0.5, whole model, the new code gives frequency against expert count r = -0.245, against Jaccard 0.262, the section 2.4 partial 0.324 (0.315 under the old Wikipedia control) and the new Jaccard against frequency partial 0.287, and on Qwen3 -0.384, 0.271, 0.455 and 0.301. The correlation numbers below come from the whole-model `_sensefix` runs, which drew one single panel per pair and used the log10 of the raw Wikipedia count as frequency. The grouped figures of sections 2.4 and 2.5 reproduce every human typicality panel exactly, since the typicality, expert count, Jaccard and overlap columns did not change, while the frequency panels now read the Zipf columns and will move slightly when the full sweep is refreshed.

*Scope note.* All values below are **whole-model scope** from the corrected `_sensefix` runs, 205 words = 197 concepts with a defined category (including `squash`) plus 8 category labels. Per-sublayer replications sit in each panel's `sublayers/<rank>_<sublayer>/` folder.

**Main correlations, AP=0.5.**

| Relationship | GPT-2 r, p | Qwen3 r, p |
|---|---|---|
| Frequency vs. Expert Count | -0.195, p=5e-3 | -0.267, p=1e-4 |
| Human Typicality vs. Expert Count | 0.021, p=0.77 (n.s.) | 0.052, p=0.47 (n.s.) |
| Frequency vs. Human Typicality | -0.026, p=0.72 (n.s.) | -0.026, p=0.72 (n.s.) |
| Human Typicality vs. Jaccard | 0.320, p=5e-6 | 0.452, p=3e-11 |
| Human Typicality vs. Overlap | 0.284, p=5e-5 | 0.433, p=2e-10 |
| Frequency vs. Jaccard | 0.194, p=6e-3 | 0.136, p=0.06 |

Human-Typicality-vs-Jaccard is positive and significant in both models (r = 0.32 to 0.45), indicating that more typical concepts share more experts with their category label, and it holds under the overlap coefficient too (0.28 and 0.43). Frequency-vs-expert-count is negative and modest (r = -0.20 to -0.27). Typicality-vs-expert-count and frequency-vs-typicality remain null.

**Frequency-vs-Jaccard is no longer null.** This panel was reported as null at every threshold in the previous, sublayer-restricted runs. On whole-model expert sets it is positive and significant in GPT-2 at AP 0.5 to 0.7 (r = 0.194, 0.236, 0.255) and in Qwen3 at AP 0.6 to 0.8 (r = 0.221, 0.290, 0.204), with the Qwen3 AP 0.5 row just short of the bar (r = 0.136, p = 0.06). The effect is small, about 4 to 8 percent of variance, and it is not independent of the typicality result, since frequency and typicality are themselves uncorrelated here (r = -0.026) while both relate positively to alignment. The honest reading is that more frequent words share somewhat more experts with their category label, an effect the single-sublayer view did not have the resolution to detect.

**Concept expert count vs category alignment.** Using the raw expert-set sizes stored by section 2.1 alongside each Jaccard value:

| Relationship (concept's own expert set size vs alignment) | GPT-2 AP0.5 | GPT-2 AP0.6 | Qwen3 AP0.5 | Qwen3 AP0.6 |
|---|---|---|---|---|
| concept expert count vs Jaccard | -0.293, p=2.9e-5 | -0.337, p=1.3e-6 | -0.372, p=7.7e-8 | -0.419, p=9.0e-10 |
| concept expert count vs Overlap coefficient | 0.169, p=0.02 | 0.089, p=0.21 (n.s.) | 0.085, p=0.24 (n.s.) | 0.029, p=0.69 (n.s.) |

Concept expert count and Jaccard are negatively correlated in both models (r = -0.29 to -0.42, p < 1e-4), so a larger expert set is associated with lower Jaccard.

This inverse relationship is a property of the Jaccard index rather than a semantic effect. Jaccard is $J = |A \cap B| / |A \cup B|$ with $A$ the concept's expert set and $B$ the category label's, and here the category label is the smaller and roughly fixed set (labels hold fewer experts than concepts). As the concept set $A$ grows, the union in the denominator grows with it while the intersection stays capped by the small label set $B$, so Jaccard falls by dilution. The overlap coefficient $|A \cap B| / \min(|A|, |B|)$, which normalizes the set-size disparity, shows no relationship with concept count (r = 0.0 to 0.15, non-significant), confirming the effect is arithmetic. The diagnostic figure `count_vs_jaccard_overlap.png` (produced separately, not part of the standard module output) shows the downward Jaccard trend in the left column flattening under the overlap coefficient in the right column, for both models.

**Typicality-alignment link and the size effect.** The Human-Typicality-vs-Jaccard result is independent of the Jaccard size effect. The partial correlation controlling for frequency is essentially unchanged (see the sweep table), and the relationship holds under the overlap coefficient (Qwen3 typicality-vs-overlap r=0.381 versus typicality-vs-Jaccard r=0.381 at AP=0.5). The typicality-alignment association is therefore size-independent.

#### Across AP thresholds

Frequency-vs-expert-count and the three null relationships, both models:

| AP | GPT-2 Freq-ExpCount | Qwen3 Freq-ExpCount | GPT-2 Freq-Jaccard | Qwen3 Freq-Jaccard |
|---|---|---|---|---|
| 0.5 | -0.195 (5e-3) | -0.267 (1e-4) | 0.194 (6e-3) | 0.136 (0.06, n.s.) |
| 0.6 | -0.247 (4e-4) | -0.303 (1e-5) | 0.236 (8e-4) | 0.221 (2e-3) |
| 0.7 | -0.286 (3e-5) | -0.321 (3e-6) | 0.255 (3e-4) | 0.290 (4e-5) |
| 0.8 | -0.277 (9e-5) | -0.300 (1e-5) | gated, 56% coverage | 0.204 (4e-3) |
| 0.9 | gated, 56% coverage | -0.160 (3e-2) | gated | gated, 36% coverage |

Frequency-vs-expert-count is negative in both models at every reported threshold, peaking near AP 0.7 at about -0.29 to -0.32 and weakening at AP 0.9 where only the highest-count words survive. Frequency-vs-Jaccard, discussed above, is now positive and significant across the mid range in both models. Frequency-vs-typicality is constant at -0.026 across all thresholds within each model, since neither variable depends on the expert data and the threshold only changes which concepts survive the join.

The gated cells are the coverage rule working as intended: GPT-2 retains only 111 of 197 categorized concepts at AP 0.8 (56%) and Qwen3 71 of 197 at AP 0.9 (36%), both below the 75% floor, so those correlations are withheld rather than computed on a survivor-biased subset.

The typicality-alignment panels, both models:

| AP | GPT-2 Typ-Jaccard | GPT-2 partial (ctrl freq) | GPT-2 Jaccard-CosineTyp | Qwen3 Typ-Jaccard | Qwen3 partial | Qwen3 Jaccard-CosineTyp |
|---|---|---|---|---|---|---|
| 0.5 | 0.320 (5e-6) | 0.331 (2e-6) | 0.647 (1e-24) | 0.452 (3e-11) | 0.460 (1e-11) | 0.639 (5e-24) |
| 0.6 | 0.291 (3e-5) | 0.306 (1e-5) | 0.543 (2e-16) | 0.395 (9e-9) | 0.411 (2e-9) | 0.686 (9e-29) |
| 0.7 | 0.209 (3e-3) | 0.223 (2e-3) | 0.450 (3e-11) | 0.328 (3e-6) | 0.350 (5e-7) | 0.689 (4e-29) |
| 0.8 | gated (56%) | gated (56%) | gated (56%) | 0.345 (7e-7) | 0.358 (2e-7) | 0.456 (2e-11) |
| 0.9 | n/a | n/a | n/a | gated (36%) | gated (36%) | gated (36%) |

Typicality-vs-Jaccard is significant across AP 0.5 to 0.8 in Qwen3 and AP 0.5 to 0.7 in GPT-2, the remaining cells being withheld by the coverage rule rather than non-significant. The partial correlation controlling for frequency slightly *exceeds* the raw value at every threshold in both models, so the modest positive frequency-alignment link now present is a mild suppressor here rather than the driver of the typicality effect. Qwen3 shows the stronger and more stable effect.

The Jaccard-vs-Cosine-Typicality panel is high in both models and holds up much better than previously reported: GPT-2 falls from 0.647 to 0.450 across AP 0.5 to 0.7, while Qwen3 stays near 0.64 to 0.69 through AP 0.7 before dropping to 0.456 at AP 0.8. Model-derived centroid typicality therefore tracks category alignment across the usable threshold range, not only at the most lenient cut.

**Overlap and entropy panels.** Human-Typicality-vs-Overlap tracks typicality-vs-Jaccard closely in both models (GPT-2 0.284 against 0.320 at AP 0.5, Qwen3 0.433 against 0.452), consistent with the typicality-alignment link being independent of Jaccard's size sensitivity.

## Conclusions

- Human Typicality is positively associated with category alignment in both models (typicality-vs-Jaccard r = 0.32 GPT-2 and 0.45 Qwen3 at AP=0.5), significant across the lenient-to-moderate thresholds, and this association is independent of both frequency (the partial correlation slightly exceeds the raw value) and Jaccard's set-size sensitivity (matched under the overlap coefficient).
- Frequency has a modest negative association with expert count (r = -0.20 to -0.32, strongest near AP 0.7) and, on whole-model expert sets, a modest *positive* association with category alignment (r = 0.14 to 0.29). The latter reverses the previous null and is the clearest change from the sublayer-restricted runs.
- Concept expert count is negatively associated with Jaccard alignment (r = -0.29 to -0.42), but this is an arithmetic property of the Jaccard index for a fixed smaller comparison set and vanishes under the overlap coefficient, so it does not reflect a size-dependent tendency in category alignment.
- The model-derived Cosine Typicality tracks Jaccard alignment across the usable threshold range in both models (GPT-2 0.65 to 0.45 over AP 0.5 to 0.7, Qwen3 near 0.64 to 0.69 through AP 0.7), rather than only at the most lenient cut.

- The Jaccard index and layer-profile agreement are only weakly related over all word pairs (Spearman about 0.4 at AP=0.5 falling to about 0.1 at AP=0.9 in both models), so which neurons two words share and how they distribute those neurons over depth are largely independent descriptions. The layer-profile reading is therefore a genuine addition to the pairwise feature set rather than a restatement of the Jaccard index, and the two diverge most exactly where section 2.2 shows the Jaccard index losing its category signal.
