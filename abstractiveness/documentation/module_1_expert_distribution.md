# Module 1: Expert Distribution

Module 1 describes where along depth each word's experts sit, how spread they are, and which sublayer types carry category information. It gathers what used to be three modules: the layer distribution (former module 1), the Shannon entropy and layer-distribution descriptors (former module 2), and the one correlation panel that concerns the distribution itself, entropy against expert count (former module 4). Everything that compares two words' expert sets or layer profiles lives in module 2.

Outputs are written to `AP_<t>/1_expert_distribution/`, one folder per section:

| Section | Folder | Contents |
|---|---|---|
| 1.1 Layer expert distribution | `1.1_layer_expert_distribution/` | expert counts, mean and cumulative layer distributions, per-concept plots, peak layer, average layer and peak dominance gap |
| 1.2 Distribution shape | `1.2_distribution_shape/` | Shannon entropy, Geary's C, the profile-shape map, entropy against expert count |
| 1.3 Sublayer informativeness | `1.3_sublayer_informativeness/` | sublayer types ranked by category alignment, whole model only |

Sections 1.1 and 1.2 run once per analysis scope, the whole model in the section folders and each sublayer type under `sublayers/<rank>_<sublayer>/<section>/`. The per-concept plots and the sublayer scopes are written only at the thresholds in the executor's `DETAILED_AP_THRESHOLDS`, currently 0.5, 0.6 and 0.7. Example tables below cite the run that produced them, in that run's folder layout.

## Research question

Are expert allocations concentrated in a subset of model layers rather than spread evenly, and does that concentration pattern differ between specific concepts (abstraction level 2) and broad categories (abstraction level 1)? And, since every layer of a transformer block plays a different functional role (attention vs. MLP projections), *which sublayer type* carries the most category-relevant expert structure, which is the analysis sublayer module 4 reports as its headline?

Beyond that, are specific concepts more sharply localized in a few layers than the broader semantic categories that contain them, i.e. is expert allocation more concentrated at the concept level than at the category level? And beyond concentration alone: do the two abstraction levels differ in *where* along the depth axis their experts sit, in how *smooth* their depth profiles are, and in how *unambiguous* their peak layers are?

Finally, since a word's entropy is bounded by its number of experts, is entropy independent of the expert count in the range analyzed, as the concentration comparisons require?

## Analysis

Notation. Let $\mathcal{C}$ be the set of analyzed items (concepts and category labels), partitioned by abstraction level into $\mathcal{C}_1$ (broad categories) and $\mathcal{C}_2$ (specific concepts). Let $L$ be the number of model layers, 48 for GPT-2 (12 blocks × 4 projections) and 196 for Qwen3-1.7B (28 blocks × 7 projections), where the layer label encodes its position as `{layer_idx}.L.{block}.{sublayer}`, e.g. `5.L.0.mlp.gate_proj`. $E_c$ is the set of expert units retained for item $c$ after AP filtering, $n_c = |E_c|$ its expert count, and $N_{c\ell}$ the number of expert rows of item $c$ in layer $\ell$. Every module, this one included, runs once per **analysis scope**: first the whole model, then one scope per sublayer type. *Analysis scopes and the two layer axes* below defines the scopes and the two layer axes they can be read on, and every other module doc refers back to it.

### Foundations, shared by the three sections

#### From responses to expert sets (data provenance)

**Procedure.** The expertise data this whole pipeline consumes is produced upstream by the self-conditioning expertise stage (`compute_responses.py` + `compute_expertise.py`). For each word $c$, a labeled sentence corpus $S_c = S_c^{\text{on}} \cup S_c^{\text{off}}$ is assembled: "on" sentences that use the word ($y_s = 1$) and "off" sentences that do not ($y_s = 0$). Every sentence is run through the model, and each unit $(\ell, u)$ records one scalar response per sentence, $z_{\ell u}(s)$ (its maximum activation over the sentence's tokens). The unit's *expertise* for the word is then the Average Precision obtained when its responses are used as a score for detecting the "on" label:

$$\mathrm{AP}_c(\ell, u) = \mathrm{AP}\big(\{(z_{\ell u}(s),\, y_s) : s \in S_c\}\big) \in [0, 1],$$

i.e. the area under the precision–recall curve of the ranking that the unit's activations induce over the sentences. $\mathrm{AP} = 1$ means the unit's activation perfectly separates on-sentences from off-sentences, while a value near the positive-class base rate means the unit carries no information about the word. Each word's results are written to `expertise/expertise.csv`, one row per unit, with columns including `ap`, the activation statistics `off_mean`, `on_p50`, `on_p90` (summaries over the off/on sentence sets), `layer`, `unit`, and `concept`. The analysis pipeline (`load_experts_data`) then applies the AP threshold $\tau$ to define the expert set

$$E_c = \{(\ell, u) : \mathrm{AP}_c(\ell, u) \ge \tau\},$$

and concatenates all words' surviving rows into the (concept, layer, unit) table every module works on. All results folders are parameterized by this $\tau$ (`AP_0.5` … `AP_0.9`).

**A neuron can be an expert for many words.** Because $\mathrm{AP}_c(\ell,u)$ is computed independently per word, nothing restricts a unit to a single word, so the expert sets $E_c$ *overlap* rather than partition the network. This is not a corner case but the norm. In the Qwen3 Richie-HSJ run at AP=0.6, the 414,089 expert rows collapse to 170,001 distinct $(\ell, u)$ neurons, and **54.9% of expert neurons are experts for two or more of the 205 words** (mean 2.44 words per expert neuron, maximum 33). Restricted to the `mlp.gate_proj` sublayer the sharing is even denser (63.7% serve ≥ 2 words, mean 3.10). Three consequences matter for reading everything downstream. First, $n_c$ counts word-neuron *associations*, not private neurons, so $\sum_c n_c$ greatly exceeds the number of distinct experts. Second, two words' layer profiles are not built from disjoint resources, so comparing them compares allocation *shapes*, not competing claims on separate neurons. Third, the overlap itself is signal, since sections 2.1 and 2.2 of module 2 measure exactly this cross-word sharing, and section 1.3 uses it to rank sublayers.

**A word can also end up with zero experts.** The definition of $E_c$ makes $E_c = \varnothing$ perfectly possible: if no unit reaches $\tau$ for word $c$, the word contributes no rows at all. Such words are *silently absent* from every downstream table (they do not appear as zero-count rows), which shrinks sample sizes at strict thresholds without any explicit marker in the CSVs. The Results section quantifies where this actually happens.

#### Analysis scopes and the two layer axes

This part defines the machinery every module shares. The other module docs refer back to it rather than restating it.

**Scopes.** Each module computes its analysis once on the entire model and once per sublayer type $s$, that is on the restricted expert sets $E_c^{(s)}$. The whole-model scope writes into the module's own folder, so the big picture is what a reader sees first, and each sublayer scope writes into `<module>/sublayers/<rank>_<sublayer>/`. The rank is the sublayer's position by expert count, computed once at the most lenient AP threshold of the sweep and cached in `results/<output_subdir>/sublayer_rank.csv`. Freezing it there rather than recomputing per threshold means a given prefix names the same sublayer in every `AP_*` folder, so paths stay comparable across the sweep, which matters because stricter thresholds thin the sublayers unevenly and would otherwise reshuffle the prefixes. A sublayer left with no experts at a strict threshold is skipped with a warning instead of producing an empty folder. Within a sublayer scope the retained `layer_idx` values are non-contiguous by design (5, 12, 19, and so on for `mlp.gate_proj`) and layers keep their absolute whole-model indices, so depth statements stay comparable across scopes.

**The two layer axes.** Analyses divide into two kinds, and only one of them is affected. *Set-based* analyses (sections 2.1 to 2.3, the global prototype of section 2.4, every Jaccard computation, and section 1.3 here) treat $E_c$ as an unordered set of $(\ell, u)$ pairs, so layer order never enters and the whole-model scope needs no special handling. *Order-based* analyses (the mean and cumulative distributions and the layer descriptors of sections 1.1 and 1.2, module 3, and the per-layer prototypes of section 2.4) read $\ell$ as depth, and there the whole-model layer index is not a depth coordinate. Layers are numbered

$$\ell = b \cdot S + s + 1,$$

with $b$ the block, $S$ the number of sublayers per block, and $s$ the sublayer's position inside the block, so on Qwen3 indices 1 to 7 all belong to block 0. Adjacent indices are therefore different projection types of the *same* block, not different depths. Geary's C, which differences adjacent indices, then measures the alternation between projection types rather than depth smoothness, and the peak layer $\arg\max_\ell \bar{P}_\ell$ returns whichever sublayer is densest for nearly every word. Measured on GPT-2 at AP 0.6, the flat whole-model axis gives a mean per-concept Geary's C of 1.036, which is exactly the value meaning *no spatial structure at all*, and 68% of concepts peak in an `mlp.c_fc` layer.

The whole-model scope therefore produces every order-based output twice:

- **`_by_block`** (canonical): the sublayers within each block are summed into one bin, $N_{cb} = \sum_{s} N_{c, bS+s+1}$, giving $B$ bins (12 for GPT-2, 28 for Qwen3) that carry the model's full expert mass on a genuine depth axis. The same GPT-2 run gives mean Geary's C 0.361 here. This is the variant downstream modules consume.
- **`_by_layer`**: the flat axis at full resolution ($L$ = 48 or 196 bins), kept because it is the only view that separates projection types, with the caveat above attached to any depth reading of it.

A sublayer scope holds exactly one layer per block, so block aggregation would be an identity relabel there and only one variant is written, with no suffix.

**Whole-model-only outputs.** Two things are not replicated per scope. Section 1.3's `sublayer_informativeness.csv` is a cross-sublayer table by construction, so it has one natural home in the whole-model folder. The per-concept files of section 1.1 are written for the whole model only, on both layer axes, since one set per concept per sublayer per threshold would be 206 × 7 × 5 × 4 figures on Qwen3, and even the whole-model set is the slowest step of module 1.

**Which scopes write what.** Modules 1 and 5 write every sublayer scope in full. Sublayer scopes of modules 2 and 3 compute only their `sublayer_comparison` rows unless the executor's `WRITE_SUBLAYER_OUTPUTS` is switched on, since their per-sublayer plots add hours and gigabytes to a sweep while the comparison rows carry the numbers. Module 4 runs its own sweep over the sublayers. The executor's `DETAILED_AP_THRESHOLDS` limits the costly detail to thresholds 0.5, 0.6 and 0.7: above them modules 1 and 5 run the whole model alone, module 4 its analysis sublayer alone, and module 1 skips the per-concept plots.

**Cross-scope comparison.** Each module writes `sublayer_comparison.csv` and `sublayer_comparison.png`, modules 1 and 2 one pair per section in the section folder, one row per scope (whole model first) with that module's headline metrics, `n_experts` on every row so each percentage or correlation is read against the mass it rests on. These are the readable surface of the sweep, and the per-scope folders are the evidence behind them. Section 1.1's row carries the mean expert count per word at each abstraction level and the mean concept peak gap, and section 1.2's the mean entropy and Geary's C at each level, the rank-based AUC of label entropy over concept entropy, and the Pearson $r$ of entropy against expert count. The AUC is the density-robust column, because it contrasts the two levels within a scope, whereas the raw entropy means fall in sparse sublayers simply because fewer experts spread over the same bins. A comparison table left with the whole-model row alone, as at thresholds outside `DETAILED_AP_THRESHOLDS`, is not written. Because the block axis puts one bin per block and a sublayer scope holds exactly one layer per block, all scopes in a comparison table share the same bin count, so bin-count-sensitive quantities such as Shannon entropy are on a common scale across rows.

**How to read these tables, and the one thing they do not control for.** Every scope-level metric here is confounded with expert density, because sublayers differ enormously in how many experts they hold (on GPT-2 at AP 0.6, `mlp.c_fc` has 40,564 expert rows against `mlp.c_proj`'s 5,271, a factor of 7.7). Measured on that run, the cross-sublayer rankings produced by section 1.3's `roc_auc`, section 2.2's category contrast, and module 4's raw $\rho$ are *all* Spearman +1.00 with each other **and with the raw expert count**, so nothing in those four rankings distinguishes "this projection type carries more category structure" from "this projection type simply holds more experts". Module 4 is the only place that controls for it, by recomputing $\rho$ on a count-matched top-$k$ expert set, and doing so **re-ranks the sublayers**: `attn.c_proj` goes from third on raw $\rho$ (0.313) to first on the count-matched value (0.275), ahead of `mlp.c_fc` (0.486 raw, 0.189 matched) and `attn.c_attn` (0.392 raw, 0.190 matched). Read the comparison tables of modules 1, 2, 3 and 5 as descriptions of what each scope's expert set actually looks like, which is what they are, and read `4_embedding_rsa/sublayer_comparison.csv` for the density-controlled ranking. Extending count-matching to the other modules is an open item.

The same confound explains the sparse-scope readings that would otherwise look like findings. `attn.c_proj` and `mlp.c_proj` carry roughly 31 and 26 experts per word, so their layer distributions are sparse, which mechanically depresses Shannon entropy (section 1.2 reports 1.94 and 1.33 bits for concepts, against about 3.1 in the dense scopes) and mechanically inflates Jensen-Shannon divergence (module 3 reports 0.36 and 0.33, against 0.09 in `mlp.c_fc`). Section 1.2's `shannon_entropy_vs_expert_count` panel measures this relationship directly, at $r \approx 0.6$ to $0.7$ in every scope, and the small-denominator caveat of section 1.1's allocation profiles is the same point at the level of a single layer share.

#### Layer descriptors

Notation. Let $L$ be the number of bins in the analyzed support, which depends on the scope and layer axis: the block-aggregated whole-model axis has one bin per transformer block (12 for GPT-2, 28 for Qwen3-1.7B), the flat whole-model axis has one per layer (48 and 196), and a sublayer scope has one per layer of that projection type (12 and 28). Sections 1.1 and 1.2 receive the expert table for the current scope, so in the sublayer-restricted runs the support is one projection type (e.g. the 28 `mlp.gate_proj` layers of Qwen3-1.7B, or the 12 `mlp.c_fc` layers of GPT-2), while layers keep their absolute whole-model indices $\ell$ (1…196 resp. 1…48), non-contiguous within the support. The module starts from the concept-by-layer count matrix $N$ (built by the shared `build_layer_probability_matrix` helper, also used by module 3), where $N_{c\ell}$ counts the expert rows of item $c$ in layer $\ell$, reindexed to the full support so layers with zero surviving experts still appear. Row-normalizing gives each item's layer probability distribution

$$p_{c\ell} = \frac{N_{c\ell}}{\sum_{\ell'} N_{c\ell'}}, \qquad p_c = (p_{c\ell})_{\ell \in \text{support}}, \quad \sum_{\ell} p_{c\ell} = 1 .$$

For a category $k$, $M_k$ denotes its member concepts (restricted to those present in the matrix).

Two structural facts about the underlying data qualify every comparison below (both are established quantitatively in *From responses to expert sets* above). First, **expert sets overlap across words**: a single neuron is typically an expert for several words (54.9% of expert neurons serve ≥ 2 of the 205 words in the Qwen3 run at AP=0.6, and 63.7% within `mlp.gate_proj`), so the distributions $p_c$ of different words are not built from disjoint neuron populations. Comparing two words' entropies therefore compares the *shapes* of their allocations over depth, not how they divide a fixed pool of neurons between them. Second, **expert counts differ systematically between the groups being compared**, addressed in the dedicated paragraph below.

**Mathematical formulation.** For each item $c$ (every word with at least one expert, including category labels, since they are words with expert sets too), six scalar descriptors of the distribution $p_c$ are computed by the shared `layer_distribution_descriptors` helper. The Shannon entropy (via `scipy.stats.entropy`, base 2, in bits) measures how spread out the allocation is:

$$H(c) = -\sum_{\ell} p_{c\ell}\,\log_2 p_{c\ell},$$

with the convention $0 \log_2 0 = 0$. Entropy is minimal ($H = 0$) when all experts sit in a single layer, and maximal ($H = \log_2 L$, e.g. $\log_2 28 \approx 4.807$ bits on the Qwen3 `mlp.gate_proj` support) when the allocation is uniform, so *lower entropy = more concentrated*. Entropy is invariant under any permutation of the layers, so it sees concentration but is blind to depth order. Its order-sensitive complement is **Geary's C**, the adjacent-layer spatial autocorrelation of the profile in depth order:

$$C(c) = \frac{\sum_{i} \left(p_{c,\ell_{i+1}} - p_{c,\ell_i}\right)^2}{2 \sum_{i} \left(p_{c,\ell_i} - \overline{p_c}\right)^2},$$

where $\ell_1 < \ell_2 < \dots$ enumerate the support in depth order. $C \approx 1$ means no depth structure (neighbors no more alike than random), $C < 1$ a smooth/clumped profile (neighboring layers alike, mass in contiguous bands), $C > 1$ a jagged, alternating profile. It is scale-invariant (counts and shares give the same value) and NaN for degenerate inputs (fewer than 3 layers or zero variance). Two words can have identical entropy but opposite C, one broad contiguous hump versus mass scattered in isolated spikes, which is exactly the distinction the shape map of section 1.2 displays.

The remaining four descriptors summarize location. The peak layer is the mode of the distribution,

$$\ell^{\ast}_c = \arg\max_{\ell} \; p_{c\ell},$$

and its reliability is quantified by the **peak dominance gap**, the percentage-point lead of the peak over the runner-up layer:

$$\gamma_c = 100\,\big(p_{(1)} - p_{(2)}\big),$$

with $p_{(1)} \ge p_{(2)}$ the two largest shares. When $\gamma_c$ is near zero the "peak layer" is effectively a tie, and its location is not a trustworthy statistic, so the peak-layer histogram and the peak-gap ECDF of section 1.1 use the threshold $\gamma < 1$ pp to mark such ambiguous peaks. The average layer is the distribution's center of mass, treating the absolute layer index as a numeric depth position:

$$\bar{\ell}_c = \sum_{\ell} \ell \cdot p_{c\ell},$$

and its trimmed variant restricts the average to the $\lceil L/4 \rceil$ most-loaded layers, renormalized:

$$\bar{\ell}^{\,25\%}_c = \frac{\sum_{\ell \in T_c} \ell \cdot p_{c\ell}}{\sum_{\ell \in T_c} p_{c\ell}}, \qquad T_c = \text{top } \lceil L/4 \rceil \text{ layers of } p_c .$$

The full average is dragged toward mid-network by the low-mass tail (a near-uniform tail pulls $\bar{\ell}_c$ toward $L/2$ regardless of where the bulk sits), while the trimmed average locates the bulk of the expertise itself. $\ell^{\ast}_c$ says where the single strongest concentration sits, $\bar{\ell}_c$ and $\bar{\ell}^{\,25\%}_c$ where the mass centers, and they can differ substantially for multi-modal or skewed profiles.

**Entropy is not independent of the expert count.** A word's entropy is bounded by its number of experts as well as by the layer count:

$$H(c) \;\le\; \log_2 \min(n_c,\, L),$$

since $n_c$ experts can occupy at most $n_c$ distinct layers. A word with many experts *can in principle spread quite evenly over the layers*, while a word with few experts *cannot produce an even distribution at all*: with $n_c < L$ a uniform allocation is impossible, and even for $n_c \gtrsim L$ the attainable shares are coarse multiples of $1/n_c$ whose sampling noise biases entropy downward. Group comparisons must therefore be read against the groups' counts. This matters concretely here because category labels systematically have *fewer* experts than concepts (Qwen3 at AP=0.6: label mean 331 experts vs. concept mean 964, and in the 150-concept run, e.g. `animal` 231 vs. member concepts often 500–1,000), so part of any "labels are more concentrated" gap could in principle be a count artifact rather than an organizational fact. Two checks bound this concern. At moderate thresholds the counts are large relative to $L$ (hundreds to thousands of experts over 28–48 layers), so the bound is slack, and section 1.2's entropy against expert count panel finds essentially no relationship at AP=0.6 (Pearson r = 0.02, p = 0.87, n = 205 in the Qwen3 run), so the entropy differences there are not driven by set size. At strict thresholds, however, counts collapse into the regime where the bound binds (the Results of section 1.1 report minimum counts of 2 at AP=0.7 and 1 at AP ≥ 0.8 in the 150-concept run, with most surviving words under 30 experts at AP ≥ 0.8), so the across-AP entropy collapse in the Results table below is partly mechanical, and level comparisons at AP ≥ 0.8 inherit the count confound.

**Generated data structures.** The descriptors are computed once per axis variant and written as two tables per level, split by what they describe. Section 1.1 writes the location descriptors, `layer_location_concepts.csv` with `peak_layer`, `peak_gap_pct`, `avg_layer`, `avg_layer_top25pct` and `experts_count`, and section 1.2 writes the shape descriptors, `shannon_entropy_concepts.csv` with `shannon_entropy`, `gearys_c` and `experts_count`. Both tables have one row per word, concepts and category labels alike, keyed by `concept`:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Word identifier (concepts *and* category labels). |
| shannon_entropy | float | $H(c)$ | Entropy of the word's layerwise expert distribution. |
| gearys_c | float | $C(c)$ | Adjacent-layer autocorrelation of the profile in depth order. |
| peak_layer | int | $\ell^{\ast}_c$ | Absolute index of the layer with the strongest concentration. |
| peak_gap_pct | float | $\gamma_c$ | Percentage-point lead of the peak layer over the runner-up. |
| avg_layer | float | $\bar{\ell}_c$ | Expected (absolute) layer value over the full distribution. |
| avg_layer_top25pct | float | $\bar{\ell}^{\,25\%}_c$ | Expected layer over the top 25% most-loaded layers, renormalized. |
| experts_count | int | $n_c$ | Number of expert rows used to compute the distribution. |

Example (head of the combined table the run wrote before the split, `AP_0.6/2_shannon_entropy_analysis/shannon_entropy_concepts.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`, where peak/average layers are absolute indices in 1–196, on the 28-layer `mlp.gate_proj` support):

| concept | shannon_entropy | gearys_c | peak_layer | peak_gap_pct | avg_layer | avg_layer_top25pct | experts_count |
|---|---|---|---|---|---|---|---|
| accountant | 4.4946 | 0.1866 | 166 | 0.3052 | 101.83 | 113.46 | 3277 |
| actor | 4.5028 | 0.5474 | 12 | 6.0443 | 93.02 | 75.98 | 1489 |
| airplane | 4.1070 | 0.2584 | 173 | 0.5579 | 119.88 | 152.29 | 717 |

#### Dual category definitions: label distribution against member average

**Mathematical formulation.** Each category $k$ is represented in two independent ways, and the same six descriptors defined above are computed for both:

- *Definition A (the label itself)*: the category label is a word with its own row in the matrix, so its distribution is simply $p_k$, giving $H(p_k)$, $C(p_k)$, $\ell^{\ast}_k$, $\gamma_k$, $\bar{\ell}_k$, $\bar{\ell}^{\,25\%}_k$.
- *Definition B (average of members)*: the members' distributions are averaged element-wise,

$$\bar{q}_{k\ell} = \frac{1}{|M_k|} \sum_{m \in M_k} p_{m\ell}, \qquad \bar{q}_k = (\bar{q}_{k\ell})_{\ell},$$

which is again a valid probability distribution (a uniform mixture of the members), giving the same six descriptors, stored with the `_average` suffix. Because entropy is concave, mixing distributions can only preserve or increase entropy relative to the average of the members' entropies, so averaging tends to *wash out* concentration unless all members peak in the same layers.

To quantify whether the two definitions at least *point at the same layers*, the Pearson correlation between the two support-length vectors is computed:

$$r_k = \operatorname{corr}(p_k, \bar{q}_k),$$

set to 0 when both vectors exist but either has zero variance, and NaN when one of the two representations is missing (e.g. a label with zero retained experts), while categories missing *both* representations are skipped entirely. High $r_k$ means the label's allocation shape mirrors the members' average shape, even if their entropies differ.

**Generated data structures.** The category tables, one row per category, and one plot. Every column is listed here, and the split between the two sections follows the table:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| category | string | $k$ | Category label. |
| shannon_entropy, gearys_c, peak_layer, peak_gap_pct, avg_layer, avg_layer_top25pct | float/int | $H(p_k)$, $C(p_k)$, $\ell^{\ast}_k$, $\gamma_k$, $\bar{\ell}_k$, $\bar{\ell}^{\,25\%}_k$ | The six descriptors of the category-label distribution (Definition A). |
| shannon_entropy_average, gearys_c_average, peak_layer_average, peak_gap_pct_average, avg_layer_average, avg_layer_average_top25pct | float/int | $H(\bar{q}_k)$, $C(\bar{q}_k)$, … | The same six descriptors of the average member distribution (Definition B). |
| pearson_correlation_distributions | float | $r_k$ | Correlation between the label distribution and the member-average distribution. |
| member_count | int | $|M_k|$ | Number of valid members contributing to the category summary. |

The category table is split the same way as the concept table. `layer_location_categories.csv` in section 1.1 holds the four location descriptors for both definitions plus `pearson_correlation_distributions` and `member_count`, and `shannon_entropy_categories.csv` in section 1.2 holds entropy and Geary's C for both definitions plus `member_count`.

Example (first rows of the combined table the run wrote before the split, `AP_0.6/2_shannon_entropy_analysis/shannon_entropy_categories.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`, location columns abbreviated):

| category | shannon_entropy | gearys_c | peak_layer | peak_gap_pct | shannon_entropy_average | gearys_c_average | peak_layer_average | pearson_correlation_distributions | member_count |
|---|---|---|---|---|---|---|---|---|---|
| furniture | 4.1739 | 0.3904 | 47 | 0.0 | 4.3876 | 0.2708 | 19 | 0.7158 | 20 |
| clothing | 4.0994 | 0.4497 | 47 | 6.1404 | 4.2417 | 0.2587 | 19 | 0.4351 | 29 |

- `category_shannon_entropies_bar.png` (section 1.2), a horizontal **bar chart** of `shannon_entropy` ($H(p_k)$, x-axis) per `category` ($k$, y-axis), sorted descending, ranking categories by how distributed (top) versus localized (bottom) their label's expert allocation is.

#### Level comparisons for the location and shape descriptors

**Mathematical formulation.** The concentration test of section 1.2 is complemented by the same non-parametric comparison, two-sided, applied to each of the other descriptors: peak dominance gap $\gamma$, full average layer $\bar{\ell}$, trimmed average layer $\bar{\ell}^{\,25\%}$, and Geary's C, between category labels and concepts. Each comparison reports the two-sided Mann–Whitney $U$, its p-value, and the effect size $\mathrm{AUC} = U / (n_1 n_2) = P(\text{label value} > \text{concept value})$.

**Generated data structures.** Logged to `main.log` only (four lines, one per descriptor), and the corresponding visual comparisons are the peak, average-layer, peak-gap and shape-map figures of sections 1.1 and 1.2.

### 1.1 Layer expert distribution

Where along depth each word's experts sit. Written to `1.1_layer_expert_distribution/`.

#### Expert counts merged with concept metadata

**Mathematical formulation.** The first analysis is a per-item aggregation of the filtered expert table. For each item $c \in \mathcal{C}$, the total expert count is

$$n_c = \sum_{\ell=1}^{L} N_{c\ell} = |E_c|,$$

i.e. the number of expert rows that survive the AP threshold for that item, counted within the current scope, so the whole-model scope reports the item's total across the network and each sublayer scope reports only that projection type's share of it. Each count is then joined with the item's metadata, and the raw Wikipedia frequency $f_c$ is mapped to a logarithmic scale,

$$\tilde{f}_c = \log_{10} f_c, \qquad f_c > 0,$$

with items at $f_c \le 0$ (missing frequency) dropped from the merged table. The log transform compresses the heavy right tail of word-frequency distributions so that downstream linear correlation operates on a roughly scale-invariant quantity rather than being dominated by a few extremely frequent words. The result is one row per item pairing the model-derived quantity $n_c$ with the human-sourced covariates $f_c$, $\tilde{f}_c$, and Human Typicality $t_c$. Words with $E_c = \varnothing$ (see *From responses to expert sets*) have no expert rows and therefore no row here either, so the table's row count *is* the count of words that retained at least one expert.

**Generated data structures.** One CSV, `expert_counts_with_metadata.csv`, built by grouping the expert rows with `groupby("concept").size()` and merging on `concept` with `concept_metadata` (the metadata table, with its typicality field renamed to `human_typicality` at load time to distinguish it from the model-computed Cosine Typicality of section 2.4). It also carries the metadata's Zipf frequency columns, `frequency_zipf_subtlex_us_lemma` and `frequency_zipf_wikipedia_lemma`, which the frequency panels of module 2 read. Rows are sorted by `(abstraction_level, concept)`, so all level-1 items form one block followed by the level-2 block. This table is not plotted here. It is the input for the correlation panels and the typicality analysis of module 2.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Concept identifier. |
| category | string | $k$ | Semantic category assigned to the concept. |
| abstraction_level | int | $a \in \{1,2\}$ | Abstraction level from the metadata. |
| frequency | float | $f_c$ | Raw frequency value from the metadata source. |
| log_frequency | float | $\tilde{f}_c$ | Base-10 logarithm of the frequency. |
| frequency_zipf_subtlex_us_lemma | float | | SUBTLEX-US lemma frequency on the Zipf scale. |
| frequency_zipf_wikipedia_lemma | float | | Wikipedia lemma frequency on the Zipf scale. |
| human_typicality | float | $t_c$ | Human-judged typicality score from the metadata. |
| expert_count | int | $n_c$ | Number of expert rows retained for the concept. |

Example (head of `AP_0.6/1_layer_expert_distribution/expert_counts_with_metadata.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`, where the level-1 block comes first because of the sort):

| concept | category | abstraction_level | frequency | log_frequency | human_typicality | expert_count |
|---|---|---|---|---|---|---|
| birds | | 1 | 132116 | 5.1210 | | 431 |
| clothing | | 1 | 69314 | 4.8408 | | 114 |
| fruit | | 1 | 86371 | 4.9364 | | 182 |
| furniture | | 1 | 44365 | 4.6470 | | 259 |
| professions | | 1 | 10176 | 4.0076 | | 612 |

#### Concept-by-layer allocation profiles (percentages and counts)

**Mathematical formulation.** The core object of the module is the percentage allocation matrix $P \in [0,100]^{|\mathcal{C}| \times L}$, built from the raw count crosstab $N$ over a `(abstraction_level, concept)` row MultiIndex:

$$P_{c\ell} = 100 \cdot \frac{N_{c\ell}}{\sum_{\ell'=1}^{L} N_{c\ell'}} = 100 \cdot \frac{N_{c\ell}}{n_c}, \qquad \sum_{\ell=1}^{L} P_{c\ell} = 100 .$$

Each row of $P$ is item $c$'s *layer profile*: the share of that item's experts sitting in each layer, expressed as a percentage. Because every row sums to 100 regardless of $n_c$, profiles of items with very different expert counts are directly comparable, since the normalization removes overall expert volume and keeps only *where* in the network the experts sit. A perfectly uniform profile would have $P_{c\ell} = 100/L$ in every layer, whereas concentration shows up as a few layers holding a much larger share. The count matrix $N$ is now kept alongside $P$ (rather than being discarded inside the crosstab), so every percentage can be traced back to the raw count that produced it.

**Percentages inherit the reliability of their denominator.** The normalization that makes profiles comparable also hides their very different statistical stability: a share computed from a large total is far more reliable than the same share computed from a small one. Moving a *single* expert between layers shifts $P_{c\ell}$ by $100/n_c$ percentage points, a negligible 0.05 pp for a concept with $n_c = 2{,}160$ experts but a massive 4.3 pp for one with $n_c = 23$ (both real AP=0.6 values in the GPT-2 Richie-HSJ run). Treating $N_{c\ell}$ as a binomial count, the standard error of an estimated share $\hat{p} = P_{c\ell}/100$ is

$$\mathrm{SE}(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n_c}},$$

so e.g. a 10% layer share is measured to about ±0.6 pp at $n_c = 2{,}160$ but only to about ±6 pp at $n_c = 23$, an uncertainty as large as the value itself. An apparent shift in a layer's percentage between two low-count words (or between AP thresholds, which change $n_c$ drastically) is therefore greatly *amplified* by the small denominator and can be pure sampling noise rather than a real reallocation. This is why the raw counts are now written next to every percentage (the `expert_count` column of the per-concept tables below, and `total_expert_count` of the mean profile): every reported share carries its scale, and shares backed by small counts should be read with proportionally wide error bars. The problem is most acute at strict AP thresholds, where many words drop to single-digit expert counts (see Results).

**Generated data structures.** For every concept, on the whole model only, a folder `per_concept/<concept>/` holding the same four cumulative plots as the section itself and the two tables behind them. They are written only at the thresholds in `DETAILED_AP_THRESHOLDS`.

- `<concept>_expert_layer_distribution_by_{block,layer}.csv`, the concept's row $P_{c\,\cdot}$ with the matching counts $N_{c\,\cdot}$ and the cumulative share, in depth order.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| abstraction_level | int | $a$ | The concept's abstraction level. |
| layer_name | string | $\ell$ (label) | Layer or block label, in model order. |
| expert_allocation_pct | float | $P_{c\ell}$ | Percentage of the concept's experts in this layer or block. |
| expert_count | int | $N_{c\ell}$ | Raw number of experts behind that percentage (the scale reference). |
| cumulative_pct | float | | Cumulative share up to this bin, in depth order. |
| experts_&lt;sublayer&gt; | int | | Block table only, one column per sublayer type with its count inside the block. |

- `<concept>_expert_layer_distribution_cumulative[_sorted]_by_{block,layer}.png`, the depth-ordered and Pareto-sorted cumulative plots described below, drawn for this one concept and titled with its name. On the block axis every bar is stacked by sublayer type, so the concept's full-mass profile and its projection-type composition read off one figure.

#### Mean allocation profile per abstraction level

**Mathematical formulation.** To compare broad categories against specific concepts as groups, the per-item profiles are averaged within each abstraction level $a \in \{1, 2\}$:

$$\bar{P}^{(a)}_{\ell} = \frac{1}{|\mathcal{C}_a|} \sum_{c \in \mathcal{C}_a} P_{c\ell}.$$

This is an unweighted mean of percentages: each item contributes equally to its group profile, regardless of its expert count $n_c$, so a concept with 100 experts and one with 1,000 shape $\bar{P}^{(a)}$ identically. Note that this equal weighting interacts with the reliability caveat of the allocation profiles, since low-count items inject their noisier profiles into the group mean with full weight, which is one more reason the raw counts are carried along. The two resulting vectors $\bar{P}^{(1)}, \bar{P}^{(2)} \in [0,100]^L$ each still sum to 100 across layers and represent the *typical* layer profile of a category label versus a specific concept. Two companion quantities are attached per (level, layer): the raw expert total

$$T^{(a)}_{\ell} = \sum_{c \in \mathcal{C}_a} N_{c\ell}$$

(the scale behind the mean percentage), and the within-level cumulative allocation in model-depth order,

$$C^{(a)}_{\ell} = \sum_{\ell' \le \ell} \bar{P}^{(a)}_{\ell'},$$

which rises from $\bar{P}^{(a)}_{1}$ to 100 across the depth axis and feeds the cumulative-mass plots below.

**Generated data structures.** One CSV, since the plain bar chart that used to accompany it was retired and the cumulative plots below now carry the same bars plus the cumulative curves:

- `mean_expert_layer_distribution{suffix}.csv`, in long format, sorted into two blocks by `abstraction_level` with layers in model order inside each block (so each block reads as a depth profile and its `cumulative_pct` ends at 100). `{suffix}` is `_by_block` or `_by_layer` in the whole-model scope and empty in a sublayer scope, per *Analysis scopes and the two layer axes*.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| abstraction_level | int | $a$ | Abstraction group being summarized. |
| layer_name | string | $\ell$ (label) | Layer label used for plotting. |
| mean_expert_allocation_pct | float | $\bar{P}^{(a)}_{\ell}$ | Mean percentage of experts assigned to that layer within the abstraction group. |
| total_expert_count | int | $T^{(a)}_{\ell}$ | Raw number of that level's experts in that layer (scale reference). |
| cumulative_pct | float | $C^{(a)}_{\ell}$ | Cumulative mean allocation up to this layer, in depth order, within the level. |

Example (head of `AP_0.6/1_layer_expert_distribution/mean_expert_layer_distribution_by_layer.csv`):

| abstraction_level | layer_name | mean_expert_allocation_pct | total_expert_count | cumulative_pct |
|---|---|---|---|---|
| 1 | 1.L.0.self_attn.q_proj | 0.6495 | 25 | 0.6495 |
| 1 | 2.L.0.self_attn.k_proj | 0.5478 | 22 | 1.1973 |
| 1 | 3.L.0.self_attn.v_proj | 0.0725 | 3 | 1.2698 |
| 1 | 4.L.0.self_attn.o_proj | 0.1004 | 4 | 1.3702 |
| 1 | 5.L.0.mlp.gate_proj | 3.4257 | 132 | 4.7959 |

#### Cumulative-mass plots (depth-ordered and Pareto-sorted)

**Mathematical formulation.** Two complementary visualizations answer "how concentrated is the allocation" directly from the quantities of the mean allocation profile. The *depth-ordered* view keeps the layers in model order and overlays the cumulative curve $C^{(a)}_{\ell}$ on the mean-allocation bars, so the curve rises steeply exactly where the distribution peaks and the depth at which the mass accumulates is visible: the marked milestones are the first layers at which the curve crosses 50% and 90%,

$$\ell^{(a)}_{50} = \min\{\ell : C^{(a)}_{\ell} \ge 50\}, \qquad \ell^{(a)}_{90} = \min\{\ell : C^{(a)}_{\ell} \ge 90\}.$$

The *Pareto-sorted* view instead orders the layers of each level by descending $\bar{P}^{(a)}_{\ell}$ and accumulates in that order, which answers "how many layers hold the bulk of the mass" independent of where they sit in the network: the milestones are the smallest $k$ such that the top-$k$ layers hold 50% (resp. 90%) of the total mass. The two views are deliberately different summaries, since a distribution can be very concentrated (small $k_{50}$) while reaching depth-ordered 50% late, if its dominant layers sit deep in the network.

**Generated data structures.** No new CSVs, since both plots are drawn from `mean_expert_layer_distribution*.csv`. Two plots per layer-axis variant, so two in a sublayer scope and four in the whole-model scope:

- `mean_expert_layer_distribution_cumulative{suffix}.png`: depth-ordered bars ($\bar{P}^{(a)}_{\ell}$, both abstraction levels side by side, left y-axis) with each level's cumulative curve $C^{(a)}_{\ell}$ on a secondary 0–100% axis. The 50% and 90% crossings are flagged with diamond markers, dotted projection lines onto both axes, and legend labels naming the crossing layers.
- `mean_expert_layer_distribution_cumulative_sorted{suffix}.png`: the Pareto view, one panel per abstraction level (the sort order differs between levels, so they cannot share an x-axis). Bars are the sorted $\bar{P}^{(a)}_{\ell}$, the line is the cumulative mass recomputed in sorted order, diamonds mark the top-$k_{50}$/top-$k_{90}$ milestones and the panel title states them (e.g. "top 8 layers hold 50% of the mass").

`{suffix}` is `_by_block` or `_by_layer` in the whole-model scope and empty in a sublayer scope. Within a sublayer scope each concept's percentages are renormalized to sum to 100 *within* that sublayer's layers, so the profiles describe where inside the projection type the experts sit, not how much of the model's total mass the type holds, which is `share_of_all_experts_pct` in section 1.3.

#### Peak and average layer across groups

**Mathematical formulation.** This analysis compares *where* in the network the three representations of the layer descriptors and the dual category definitions place their mass, using the per-item summary positions rather than the full distributions. Three groups $g$ are formed: Specific Concepts, Broad Categories (labels), and Broad Categories (Avg) (member averages). For the bars, each layer $\ell$ receives the percentage of group members peaking there:

$$\text{peak\%}_g(\ell) = 100 \cdot \frac{|\{c \in g : \ell^{\ast}_c = \ell\}|}{|g|},$$

which sums to 100 within each group and makes groups of very different sizes comparable. Each bar is additionally split by peak reliability: the solid segment counts members whose peak dominance gap is $\gamma_c \ge 1$ pp, and a hatched segment stacked on top counts members with an *ambiguous* peak ($\gamma_c < 1$ pp), so the histogram shows at a glance how much of the peak-location mass is actually trustworthy. For the curves, a Gaussian kernel density estimate is fitted per group to the average-layer positions (normalized within each group, `common_norm=False`), in two line styles: solid for the full-distribution average $\bar{\ell}_c$ and dashed for the trimmed average $\bar{\ell}^{\,25\%}_c$, whose comparison shows how strongly the low-mass tail pulls the centers toward mid-network. Because the sublayer-restricted support is non-contiguous in absolute layer indices, all positions are mapped to consecutive axis slots via the support's label order (fractional averages are placed between the retained layers by linear interpolation).

**Generated data structures.** One plot (no standalone CSV, since the inputs are columns of the two CSVs above):

- `peak_average_layers.png`, a combined **bar chart + KDE overlay** on twin y-axes: bars show $\text{peak\%}_g(\ell)$ (left y-axis, solid = dominant peak, hatched = ambiguous peak) against the model layer (x-axis), while the overlaid density curves show the KDE of the average layers (right y-axis, solid = full, dashed = trimmed). Both are colored by group.

#### Average-layer comparison across abstraction levels

**Mathematical formulation.** The direct test of whether one abstraction level's experts sit earlier or later in the network than the other's: the violin comparison of section 1.2's concentration test repeated on the depth descriptor. It uses the *trimmed* average $\bar{\ell}^{\,25\%}$ rather than the full average, because the full average is dragged toward mid-network by the low-mass tail (see *Layer descriptors*) and thereby compresses exactly the group differences this plot is meant to show. The full averages remain available in the CSVs and in the KDE curves of the peak and average layer plot.

**Generated data structures.** One plot:

- `category_concept_avg_layer_violin.png`, the same **violin + strip** composition as the entropy violins of section 1.2, comparing `avg_layer_top25pct` ($\bar{\ell}^{\,25\%}$, y-axis) between Specific Concepts and Broad Categories.

#### Peak dominance gap ECDF

**Mathematical formulation.** The peak-reliability question of the peak-layer histogram in full distributional form: for each abstraction level, the empirical cumulative distribution of the gap,

$$\mathrm{ECDF}_g(x) = \frac{|\{c \in g : \gamma_c < x\}|}{|g|},$$

read as "the fraction of the group whose peak layer leads the runner-up by less than $x$ percentage points". The steeper the curve near zero, the less meaningful the group's peak-layer statistics are.

**Generated data structures.** One plot:

- `peak_gap_ecdf.png`, one **ECDF curve** per level of `peak_gap_pct` ($\gamma$), with the ambiguity threshold ($\gamma = 1$ pp, matching the hatching of the peak-layer histogram) drawn as a dashed line and annotated with the share of each level below it.

### 1.2 Distribution shape

How spread each word's experts are over depth, and whether that spread is independent of the expert count. Written to `1.2_distribution_shape/`.

#### Concept-against-category concentration test (Mann-Whitney U with AUC effect size)

**Mathematical formulation.** The research question is a comparison of two samples of entropies: $\{H(c) : c \in \mathcal{C}_2\}$ (concepts) versus $\{H(p_k) : k\}$ (category labels). Because entropies are bounded and not normally distributed, a non-parametric one-sided Mann–Whitney U test is used with the alternative hypothesis "category entropies are stochastically *smaller* than concept entropies" (`alternative='less'`), i.e. categories are more concentrated. Writing $n_1, n_2$ for the two sample sizes and $R_1$ for the sum of ranks of the category sample in the pooled ranking,

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1 ,$$

and the p-value is the probability, under the null of identical distributions, of a $U$ at least as extreme in the "less" direction. Alongside the p-value, the test's own effect size is logged:

$$\mathrm{AUC} = 1 - \frac{U}{n_1 n_2} = P\big(H(\text{random category}) < H(\text{random concept})\big),$$

where 0.5 means no effect and 1.0 means every category label is more concentrated than every concept, which keeps "significant" and "large" visibly separate. A significant result means category labels' allocations are systematically more concentrated (lower $H$) than concepts', not just different in shape. Per the count-dependence paragraph of *Layer descriptors*, the comparison should be read jointly with the entropy against expert count panel below, since the two groups differ in typical expert count.

**Generated data structures.** The $U$ statistic, p-value, and AUC are logged to `main.log` (not saved as CSV). The comparison is visualized in:

- `category_concept_shannon_entropies.png`, a **violin plot** (drawn by the shared `plot_comparison_violin` helper: quartile lines with Q1/median/Q3 text labels, violins clipped to the observed data range, and a strip plot of the raw points overlaid so small groups stay honest) comparing `shannon_entropy` ($H$, y-axis) across the two groups Specific Concepts vs. Broad Categories (x-axis).

#### Profile-shape map: Geary's C against Shannon entropy

**Mathematical formulation.** Entropy and Geary's C answer orthogonal questions about a layer profile, namely *how concentrated* (order-blind) versus *how smooth in depth* (order-sensitive), so plotting every word at coordinates $(H(c), C(c))$ maps the space of profile shapes. The four corners of the map read: concentrated & smooth (low $H$, low $C$: one compact band of layers), concentrated & jagged (low $H$, high $C$: a few isolated spikes), spread & smooth (high $H$, low $C$: broad contiguous mass), spread & jagged (high $H$, high $C$: scattered mass with no depth locality). The dashed horizontal line at $C = 1$ marks "no depth structure", and dotted lines through the middle of each axis quarter the map.

**Generated data structures.** One plot (inputs are the descriptor tables):

- `gearys_entropy_scatter.png`, a **scatter plot**: every specific concept as a small dot (category-label words are excluded from the dots, since they appear in the concepts CSV too, and would otherwise be plotted twice), every category label as a larger named diamond, the 5 most extreme concepts on each end of *both* axes annotated by name, and quadrant guides and the $C=1$ reference line drawn as described, with corner glosses naming the four shape regimes.

#### Shannon entropy against expert count (both abstraction levels)

**Mathematical formulation.** This panel is the empirical check behind the count-dependence caveat of *Layer descriptors*: entropy is bounded by the expert count ($H(c) \le \log_2 \min(n_c, L)$), so if entropy and expert count were strongly correlated in the analyzed range, any entropy comparison between groups with different typical counts (category labels hold systematically fewer experts than concepts) would partly measure set size rather than layer organization. The test is the pooled Pearson correlation on the pair

$$x = H(c), \qquad y = n_c,$$

over *every* word in the concept descriptor table, which contains both abstraction levels, since category labels are words with expert distributions too. The two levels are separated only for display (words are classified by membership in the category list), while $r$ and $p$ are computed on the pooled sample. A near-zero $r$ licenses reading this section's entropy contrasts as organizational, while a strong positive $r$ would flag them as count artifacts. A word is missing here only when the descriptor computation dropped it for having zero experts across every layer, so this panel's coverage denominator is the full metadata count, the same one module 2's expert-count panels use.

**Generated data structures.** One CSV, one PNG, and one row in `correlation_summary.csv`:

- `shannon_entropy_vs_expert_count.csv`, the working table:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Word identifier. |
| group | string | | "Specific Concepts" or "Broad Categories" (display split). |
| shannon_entropy | float | $H(c)$ | Entropy of the word's layer distribution. |
| experts_count | int | $n_c$ | Number of expert rows behind the distribution. |

Example (head of `AP_0.6/4_correlations/shannon_entropy_vs_expert_count.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`):

| concept | group | shannon_entropy | experts_count |
|---|---|---|---|
| accountant | Specific Concepts | 4.4946 | 3277 |
| actor | Specific Concepts | 4.5028 | 1489 |
| airplane | Specific Concepts | 4.1070 | 717 |
| apple | Specific Concepts | 4.0195 | 160 |
| apricot | Specific Concepts | 3.9250 | 1975 |

- `shannon_entropy_vs_expert_count.png`, a **scatter plot with a linear regression line** over the pooled sample, x: `shannon_entropy` ($H(c)$), y: `experts_count` ($n_c$), with concepts drawn as small dots and category labels as larger diamonds, and the pooled Pearson $r$, $p$ and $n$ on the grey line under the title.

### 1.3 Sublayer informativeness

Which sublayer types carry the most category structure. Written once, for the whole model, to `1.3_sublayer_informativeness/`.

**Mathematical formulation.** Transformer layers come in a small number of *sublayer types* (projections), parsed from the layer label `{idx}.L.{block}.{sublayer}`, with 4 types for GPT-2 (`attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`) and 7 for Qwen3 (`self_attn.q/k/v/o_proj`, `mlp.gate/up/down_proj`). This analysis ranks the types by how much category structure their expert sets carry, which is the empirical justification for the analysis sublayer module 4 reports as its headline. For each sublayer $s$, restrict every item's expert set to that sublayer's layers, $E_c^{(s)}$, and compute for every unordered pair of level-2 concepts the Jaccard similarity of the restricted sets,

$$J^{(s)}_{cd} = \frac{|E^{(s)}_c \cap E^{(s)}_d|}{|E^{(s)}_c \cup E^{(s)}_d|},$$

together with the indicator of whether $c$ and $d$ belong to the same category. Three alignment statistics follow (level-2 concepts only, since category labels have no same-category peers, so level-1 rows carry NaN here):

- **`roc_auc` (primary)**: the probability that a randomly drawn same-category pair is more similar than a randomly drawn different-category pair. It is computed rank-based via the Mann–Whitney identity, $\mathrm{AUC} = \big(\sum_{\text{same}} R_i - \tfrac{n_1(n_1+1)}{2}\big) / (n_1 n_0)$ with $R_i$ the ranks of the pair similarities (average ranks for ties), which makes it immune to the heavy same/different pair imbalance and to the skewed shape of Jaccard values. 0.5 = no category signal, 1.0 = every within-category pair beats every across-category pair.
- **`category_alignment_r` (companion)**: the same same-category-vs-different-category comparison as `roc_auc`, but as a plain Pearson correlation instead of a rank statistic. Writing $\mathrm{same}_{cd} \in \{0,1\}$ for the indicator that $c$ and $d$ share a category,

$$r = \operatorname{corr}\big(\mathrm{same}_{cd},\, J^{(s)}_{cd}\big) \in [-1, 1],$$

taken over every concept pair. This is a point-biserial correlation, the name for a Pearson correlation with one binary input, so positive $r$ means same-category pairs run more similar on average, and $r^2$ is the share of the pair-similarity variance explained by category membership alone, the same variance-explained reading the correlation panels of modules 1 and 2 use. Because Pearson correlation assumes a roughly linear relationship, `category_alignment_r` can diverge from the rank-based `roc_auc` when the similarity distribution is skewed or dominated by a few extreme pairs, for example a handful of very high same-category similarities can inflate $r$ well beyond what the AUC's pairwise ranking shows, which is why the two are reported side by side rather than one replacing the other.
- **`mantel_p`**: a permutation p-value for the AUC. Pairs are not independent observations (each concept participates in $|\mathcal{C}_2|-1$ pairs), so the null distribution is built by shuffling category labels across *concepts* (never across pairs), rebuilding the indicator, and recomputing the AUC, 9,999 times. $p = (1 + \#\{\mathrm{AUC}_{\text{perm}} \ge \mathrm{AUC}\}) / (1 + 9{,}999)$, so the smallest reportable value is $10^{-4}$. Degenerate sublayers whose pair similarities are all identical short-circuit to AUC = 0.5, $r$ = NaN, $p$ = 1.

Each (level, sublayer) row also carries a *depth-smoothness* descriptor: the mean, over that level's words, of Geary's C of the word's expert counts across the sublayer's layers in depth order,

$$C(x) = \frac{\sum_{i} (x_{i+1} - x_i)^2}{2 \sum_i (x_i - \bar{x})^2},$$

the adjacent-neighbor spatial autocorrelation, where $C \approx 1$ means no depth structure, $C < 1$ means smooth or clumped profiles with neighboring layers alike, and $C > 1$ means jagged, alternating profiles. It is scale-invariant, so counts and shares give the same value, as used per word in section 1.2. Computing it within one sublayer type uses one layer per block, so it is free of the mechanical high/low alternation between neighboring projection types that whole-model depth order would introduce. Finally, each row records that level's expert volume in the sublayer, as a count and as a share of *all* experts (shares sum to 100 over the whole table).

**Generated data structures.** One CSV, `sublayer_informativeness.csv`, one row per (abstraction_level, sublayer), level-1 block first, sublayers ordered by the level-2 AUC ranking within each block:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| abstraction_level | int | $a$ | Level the row summarizes. |
| sublayer | string | $s$ | Sublayer (projection) type. |
| gearys_c | float | $\overline{C}$ | Mean per-word Geary's C across the sublayer's layers in depth order. |
| roc_auc | float | AUC | Category alignment of the sublayer's expert sets (level 2 only, NaN for level 1). |
| category_alignment_r | float | $r$ | Pearson (point-biserial) correlation between same-category membership $\mathrm{same}_{cd}$ and pair similarity $J^{(s)}_{cd}$ (level 2 only). |
| mantel_p | float | $p$ | Concept-label permutation p-value for the AUC (level 2 only). |
| share_of_all_experts_pct | float | | That level's experts in this sublayer as % of all experts. |
| n_experts | int | | Raw expert count behind the share. |
| n_words | int | | Words of that level with at least one expert in the sublayer. |
| mean_entropy_bits | float | $\overline{H}$ | Mean Shannon entropy of the words' profiles on the sublayer's own block axis. |
| reliable_peak_pct | float | | Share of words whose peak leads the runner-up by at least `PEAK_GAP_RELIABLE_PP` (1 pp). |
| mean_depth_top25_block | float | $\overline{\ell}^{\,25\%}$ | Mean top-25% average depth, in blocks. |
| mean_expert_count | float | $\bar{n}$ | Mean experts per word in the sublayer. |

Example (level-2 block of `AP_0.6/1_layer_expert_distribution/sublayer_informativeness.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`, values rounded):

| abstraction_level | sublayer | gearys_c | roc_auc | category_alignment_r | mantel_p | share_of_all_experts_pct | n_experts |
|---|---|---|---|---|---|---|---|
| 2 | mlp.gate_proj | 0.317 | 0.927 | 0.620 | 0.0001 | 45.62 | 188907 |
| 2 | mlp.up_proj | 0.241 | 0.923 | 0.562 | 0.0001 | 24.62 | 101964 |
| 2 | self_attn.o_proj | 0.955 | 0.879 | 0.549 | 0.0001 | 10.20 | 42256 |
| 2 | self_attn.v_proj | 0.492 | 0.779 | 0.417 | 0.0001 | 3.68 | 15220 |
| 2 | self_attn.k_proj | 0.510 | 0.714 | 0.362 | 0.0001 | 2.55 | 10565 |
| 2 | self_attn.q_proj | 0.452 | 0.698 | 0.304 | 0.0001 | 4.13 | 17122 |
| 2 | mlp.down_proj | 0.785 | 0.626 | 0.168 | 0.0001 | 8.24 | 34118 |

The descriptor columns are computed on the sublayer's own block axis, one bin per block, so they sit on a common scale across sublayers. `sublayer_informativeness.png` draws seven of the columns as small multiples, one panel per column, sublayers as rows in level-2 ROC-AUC order and one bar per abstraction level: category alignment ROC-AUC, mean Geary's C, mean entropy, reliable-peak share, mean top-25% depth, mean experts per word and share of all experts.

## Results

### Layer distribution (section 1.1)

At AP=0.6 in the GPT-2 150-concept reference run (`research_plots_150_revised_executor_again`, whole-model expert sets), layer allocation is genuinely concentrated, not uniform. With 48 layers, a uniform allocation would put 10.4% of experts in any 5 layers. Instead, the top 5 layers hold **44.0%** of all expert mass for broad categories (abstraction level 1) and **35.4%** for specific concepts (level 2). The single busiest layer for both levels is `3.L.0.mlp.c_fc` (an early MLP layer), at 13.5% for categories and 11.3% for concepts. This is visible directly in the mean-distribution bars of the cumulative plot: a handful of sharp early-to-mid-layer peaks stand out clearly against a low, noisy baseline across the rest of the network.

#### Across AP thresholds

The concentration effect is not a one-threshold artifact. It holds at every AP from 0.5 to 0.9, and it gets dramatically stronger as the threshold tightens:

| AP | Top-5-layer % (categories, level 1) | Top-5-layer % (concepts, level 2) | Layers with zero experts |
|---|---|---|---|
| 0.5 | 36.4% | 31.1% | 0 |
| 0.6 | 44.0% | 35.4% | 0 |
| 0.7 | 50.0% | 40.6% | 0 |
| 0.8 | 66.6% | 45.4% | 3 |
| 0.9 | 88.6% | 54.2% | 5 |

At AP=0.9, the top 5 layers hold **88.6%** of all category-level expert mass, so almost everything is concentrated in a handful of layers. The level-1 (category) vs. level-2 (concept) gap also widens at stricter thresholds (5.3 points at AP=0.5 vs. 34.4 points at AP=0.9), consistent with section 1.2's finding that categories concentrate faster than concepts as the AP bar rises.

A second, structural finding only visible by comparing thresholds: at AP=0.8 and AP=0.9, **3 and 5 of the 48 layers respectively have zero retained experts across every single concept** (i.e. no expert anywhere in the dataset survives that strict an AP filter in those layers).

Section 1.1 only reports descriptive percentages, it does not run a significance test on the level-1-vs-level-2 gap at any threshold. Section 1.2's Shannon entropy analysis is what tests concept-vs-category concentration statistically, while this section's contribution is establishing that *some* layers clearly dominate (more so as AP increases), which is the precondition for that later test to be meaningful.

#### Zero-expert words and low-count words across thresholds

Not only layers but *words* can end up empty (see *From responses to expert sets*). In the 150-concept reference run (the same run as the tables above, 164 metadata words = 147 concepts + 17 category labels):

| AP | Words with ≥1 expert (of 164) | Zero-expert words | Words with <30 experts | Smallest count |
|---|---|---|---|---|
| 0.5 | 164 | 0 | 0 | 162 |
| 0.6 | 164 | 0 | 1 | 25 (`clothing`) |
| 0.7 | 164 | 0 | 13 | 2 (`clothing`) |
| 0.8 | 161 | 3 | 94 | 1 |
| 0.9 | 119 | 45 | 101 | 1 |

At AP ≤ 0.7 there are **no zero-expert cases**: every word keeps at least one expert. The first casualties appear at AP=0.8 (`clothing`, `hopscotch`, `toy`, noting that `clothing` and `toy` are category *labels*, so a whole category loses its Definition-A representation), and at AP=0.9 the phenomenon is widespread: **45 of 164 words (27%) retain zero experts**, including the labels `animal` and `clothing`. This is exactly what silently shrinks the downstream sample sizes reported by the other modules (section 2.1's pair count falling 147 → 128 → 59 across AP 0.7–0.9, module 3's category count falling 17 → 15 → 9). The Richie-HSJ runs behave the same way, and the two architectures differ sharply in how fast they thin out. Of the 205 metadata words:

| AP | GPT-2 surviving | GPT-2 zero | GPT-2 <30 experts | Qwen3 surviving | Qwen3 zero | Qwen3 <30 experts |
|---|---|---|---|---|---|---|
| 0.5 | 205 | 0 | 0 | 205 | 0 | 0 |
| 0.6 | 205 | 0 | 1 | 205 | 0 | 0 |
| 0.7 | 205 | 0 | 51 | 205 | 0 | 0 |
| 0.8 | 193 | 12 | 122 | 205 | 0 | 11 |
| 0.9 | 114 | 91 | 96 | 195 | 10 | 102 |

GPT-2 loses its first words at AP=0.8 and by AP=0.9 has lost 91 of 205, while Qwen3 keeps every word through AP=0.8 and loses only 10 at AP=0.9. The larger model simply has more units available to clear a strict bar, so results at AP≥0.8 rest on a much thinner GPT-2 sample than the same threshold implies for Qwen3.

The middle column matters just as much as the zero column. Well before a word reaches zero, it passes through the low-count regime where the reliability caveat of the allocation profiles bites: at AP=0.8, 94 of the 161 surviving words have fewer than 30 experts, so a single expert is 3–100 percentage points of the layer profile, and per-word percentage profiles (and any shift in them between thresholds) are dominated by sampling noise. Group-level statements remain meaningful at strict thresholds because they pool words, but per-word layer percentages at AP ≥ 0.8 should not be over-interpreted.

The cumulative plots quantify concentration for the Qwen3 run at AP=0.6. On the flat 196-layer axis, the top 11 layers hold 50% of the level-1 mass (top 20 for level 2), and 48 and 70 layers respectively are needed for 90%. On the block axis, which is the one that reads as depth, 7 of 28 blocks hold 50% of the mass at both levels, and 17 and 16 blocks reach 90%. GPT-2 is more concentrated still: 4 of 12 blocks hold 50% at both levels, 9 and 10 blocks reach 90%, and on its flat 48-layer axis 7 and 9 layers hold 50%.

### Distribution shape (section 1.2)

*Scope note.* The 150-concept figures below come from the runs that predate the whole-model refactor, so they describe the **analysis sublayer** only. The Richie-HSJ sections are now reported from the corrected `_sensefix` runs at both scopes, and the scope turns out to matter more than any other choice in this module, so the two are kept separate and labelled throughout.

At AP=0.6 in the GPT-2 150-concept reference run, the hypothesis holds, but only modestly in magnitude. Mean entropy for the 164 concepts is **4.49 bits** (SD 0.35) versus **4.18 bits** (SD 0.33) for the 17 category labels, so categories are lower-entropy (more concentrated) than concepts, and a Mann-Whitney U test confirms this is unlikely to be chance (U=698.5, p=3.6×10⁻⁴). But the absolute gap is small relative to the scale: the maximum possible entropy over 48 layers is log₂(48) ≈ 5.58 bits, so both groups sit well below the ceiling and only about 6% apart from each other. `category_concept_shannon_entropies.png` shows this directly: the two violins clearly overlap, with the category distribution shifted down and narrower, not cleanly separated from the concept distribution.

A more striking, and easy to miss, pattern sits in the same CSV: averaging a category's *members* together (`shannon_entropy_average`, mean 4.79 bits) produces the **least** concentrated distribution of the three, even less concentrated than individual concepts, let alone the category label itself. In other words, the category-label word is the sharpest of the three representations, but smoothing across its members washes that sharpness out rather than preserving it. This is exactly the gap that module 3 quantifies directly with Jensen-Shannon divergence, and it complicates a simple "category labels are just an average of their members" story.

#### Across AP thresholds

| AP | Concept entropy (mean, SD) | Category-label entropy (mean, SD) | Avg-member entropy (mean) | Mann-Whitney p | Label-vs-avg-member r |
|---|---|---|---|---|---|
| 0.5 | 4.83 (0.21) | 4.66 (0.14) | 5.00 | 3.3e-04 | 0.83 |
| 0.6 | 4.49 (0.35) | 4.18 (0.33) | 4.79 | 3.6e-04 | 0.82 |
| 0.7 | 4.03 (0.63) | 3.32 (0.90) | 4.56 | 3.3e-04 | 0.73 |
| 0.8 | 3.19 (1.08) | 2.08 (1.31) | 4.22 | 6.8e-04 | 0.62 |
| 0.9 | 1.94 (1.41) | 1.05 (1.05) | 3.34 | 2.6e-02 | 0.59 |

Three findings are robust across all five thresholds, which is the strongest form of evidence this analysis produces anywhere:

1. **Categories are always significantly more concentrated than concepts** (Mann-Whitney p<0.05 at every threshold, p<0.001 at four of five).
2. **The category-label word is always the most concentrated of the three representations, and the member-average is always the least.** This ordering (label < concepts < avg-member, lower=more concentrated) holds at every single AP value, not just AP=0.6.
3. **Label-vs-member-average shape correlation weakens monotonically** as AP tightens (0.83, 0.82, 0.73, 0.62, and 0.59 across AP 0.5 to 0.9). At lenient thresholds the label and its members peak in very similar places, whereas at stricter thresholds they diverge more, consistent with there being fewer, noisier experts per concept to average over.

The *relative* size of the concept-vs-category gap also grows sharply with AP. At AP=0.5 concepts and categories differ by only 0.17 bits (4.83 against 4.66), and by AP=0.9 they differ by 0.89 bits on a much smaller overall scale (1.94 against 1.05), so categories collapse toward near-total concentration faster than concepts do as the threshold strips away marginal experts. Two mechanical contributions to this pattern should be kept in mind (see *Layer descriptors*): at strict thresholds many words' expert counts fall into the regime where $H \le \log_2 n_c$ binds, so entropies are pushed down by shrinking counts and not only by sharpening organization, and category labels reach that regime earlier because they hold fewer experts to begin with.

#### The Richie-HSJ runs, and why the analysis scope decides the answer

Reported from the corrected `_sensefix` runs (205 words = 197 concepts including `squash`, plus 8 category labels). The concentration hypothesis is answered differently depending on whether the expert set is read over the whole model or restricted to the analysis sublayer, and that difference is the main finding of this section. Entropy tests are one-sided (categories more concentrated), depth tests two-sided, computed over the 8 labels against the 197 concepts.

**Whole model.** The hypothesis holds, clearly, in both architectures:

| Model | AP | Entropy, label vs concept (p) | Avg layer, label vs concept (p) | Trimmed avg (p) |
|---|---|---|---|---|
| GPT-2 | 0.5 | 4.62 vs 4.78 (0.003) | 25.6 vs 23.6 (0.027) | 26.4 vs 23.5 (0.12) |
| GPT-2 | 0.6 | 4.06 vs 4.42 (0.001) | 25.9 vs 21.7 (0.008) | 26.1 vs 20.7 (0.019) |
| Qwen3 | 0.5 | 5.95 vs 6.20 (<0.001) | 105.6 vs 110.6 (0.12) | 107.8 vs 113.3 (0.13) |
| Qwen3 | 0.6 | 5.34 vs 5.84 (<0.001) | 108.2 vs 106.3 (0.80) | 108.7 vs 107.7 (0.94) |

**Analysis sublayer only** (`mlp.c_fc`, `mlp.gate_proj`). The same test on the same words returns nothing:

| Model | AP | Entropy, label vs concept (p) | Avg layer, label vs concept (p) |
|---|---|---|---|
| GPT-2 | 0.5 | 3.22 vs 3.27 (0.16) | 26.6 vs 24.0 (0.030) |
| GPT-2 | 0.6 | 2.98 vs 3.05 (0.08) | 26.9 vs 21.9 (0.021) |
| Qwen3 | 0.5 | 4.39 vs 4.31 (0.88) | 112.4 vs 106.7 (0.12) |
| Qwen3 | 0.6 | 4.19 vs 4.15 (0.60) | 115.0 vs 103.7 (0.013) |

Three readings follow.

First, **the concentration contrast is a whole-model property**. Restricted to one projection type the levels are indistinguishable (p 0.08 to 0.88), while pooled over all sublayers the labels are reliably sharper (p ≤ 0.003 everywhere). The mechanism is visible in section 1.1: a category label spreads its experts over fewer sublayer types than a concept does, so restricting to a single sublayer discards exactly the axis along which labels are concentrated. Any claim about level-1 versus level-2 concentration must therefore state its scope, and the earlier sublayer-only null should not be read as a null for the models.

Second, **depth separates the levels in GPT-2 but not in Qwen3**. GPT-2 labels sit consistently deeper than their members (p 0.008 to 0.030 whole model, 0.021 to 0.030 in `mlp.c_fc`), while on Qwen3 the whole-model comparison is flat (p 0.12 and 0.80) and only the sublayer-restricted AP 0.6 comparison separates (p 0.013). Depth is the more architecture-dependent of the two contrasts, and the reverse of what the earlier sublayer-only reading suggested.

Third, **the three-way ordering label < concepts < avg-member holds on Richie-HSJ too**, matching the 150-concept run: at AP 0.6 GPT-2 gives 4.06 < 4.42 < 4.74 and Qwen3 gives 5.34 < 5.84 < 6.15. Averaging a category's members produces the least concentrated of the three representations in every run and at every threshold, which is the gap module 3 measures directly. Label-versus-member shape correlation is 0.85 and 0.71 for GPT-2 at AP 0.5 and 0.6, and 0.84 and 0.77 for Qwen3, weakening as the threshold tightens exactly as in the 150-run.

**Peak reliability and smoothness.** Ambiguous peaks ($\gamma < 1$ pp) are common and far more so in the larger model: at AP 0.6, 23% of GPT-2 concepts and 50% of its labels are ambiguous, against 56% and 62% for Qwen3, so the hatched segments of the peak-layer histogram carry most of the mass in the Qwen3 figures and peak locations there should not be read as point estimates. On the block axis, mean Geary's C is 0.35 (concepts) and 0.43 (labels) for GPT-2 at AP 0.6, and 0.33 and 0.47 for Qwen3, all far below the $C=1$ no-structure line, so expert mass sits in contiguous depth bands in both models. Read on the flat layer axis instead, the same GPT-2 statistic is 1.02, the textbook value for no spatial structure at all, which is the artifact *Analysis scopes and the two layer axes* warns about and the reason the block axis is canonical for every depth-ordered descriptor here.

#### Entropy against expert count

Shannon-entropy-vs-expert-count behaves very differently in the two architectures, and this matters for how the entropy contrasts of section 1.2 are read. In Qwen3 it is near zero at lenient thresholds (r = 0.109 at AP 0.5, -0.012 at AP 0.6, 0.011 at AP 0.7) and only becomes substantial at strict ones (0.192 at AP 0.8, 0.490 at AP 0.9), exactly the $H \le \log_2 n_c$ ceiling effect the entropy against expert count panel predicts. In GPT-2 it is strong at *every* threshold (0.655, 0.647, 0.561, 0.545). GPT-2 words hold far fewer experts than Qwen3 words at the same AP, so GPT-2 sits in the count-limited regime from the start. The entropy contrasts are therefore count-independent for Qwen3 at AP 0.5 to 0.7, but for GPT-2 an entropy difference between two groups is partly a count difference at any threshold, and the level-1 versus level-2 entropy gap there should be read with the label-versus-concept count gap in view.

### Sublayer informativeness (section 1.3)

The sublayer ranking cleanly separates the projection types, in the same order for both architectures' MLP-vs-attention contrast. For Qwen3 (table in section 1.3): `mlp.gate_proj` and `mlp.up_proj` are nearly tied at the top (AUC 0.948 and 0.943) and jointly hold 70% of all experts, attention projections rank middle (o_proj 0.901 down to q_proj 0.710), and `mlp.down_proj` is by far the least category-aligned (AUC 0.628, r 0.173). Every sublayer's `mantel_p` sits at the $10^{-4}$ permutation floor, so *all* sublayers carry statistically real category signal, and the ranking is about how much, not whether. `mlp.gate_proj` also dominates the level-1 side, holding 2,849 of the 4,163 category-label experts (68%). For GPT-2 on the same dataset, the winner is `mlp.c_fc` (AUC 0.889, against 0.796 for `attn.c_attn`, 0.691 for `attn.c_proj`, 0.571 for `mlp.c_proj`). These rankings are what the configured `sublayer_filter` values (`mlp.gate_proj`, `mlp.c_fc`), the analysis sublayer of module 4, implement. The Geary's C column adds a shape observation: the category-aligned MLP input projections have smooth depth profiles (mean C ≈ 0.24–0.31), while `self_attn.o_proj` (C ≈ 0.96) is essentially depth-unstructured.

At the more lenient AP=0.5 the same ordering holds with higher values throughout (Qwen3: gate_proj 0.956, up_proj 0.954, o_proj 0.949, v_proj 0.906, q_proj 0.872, k_proj 0.872, down_proj 0.754, and for GPT-2: c_fc 0.933, attn.c_attn 0.906, attn.c_proj 0.901, mlp.c_proj 0.695), so the ranking is stable across thresholds rather than an artifact of one cut.
