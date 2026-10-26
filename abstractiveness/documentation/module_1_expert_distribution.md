# Module 1: Expert Distribution

Module 1 describes where along depth each word's experts sit, how spread they are, and which sublayer types carry category information. It gathers what used to be three modules: the layer distribution (former module 1), the Shannon entropy and layer-distribution descriptors (former module 2), and the one correlation panel that concerns the distribution itself, entropy against expert count (former module 4). Everything that compares two words' expert sets or layer profiles lives in module 2.

Outputs are written to `AP_<t>/1_expert_distribution/`, one folder per section:

| Section | Folder | Contents |
|---|---|---|
| 1.1 Layer expert distribution | `1.1_layer_expert_distribution/` | expert counts, mean and cumulative layer distributions with their per-level summary, per-concept plots, peak layer, average layer and peak dominance gap |
| 1.2 Distribution shape | `1.2_distribution_shape/` | Shannon entropy, normalised entropy, Geary's C, each category label against its own members, entropy against expert count |
| 1.3 Sublayer informativeness | `1.3_sublayer_informativeness/` | sublayer types ranked by category alignment, a compact level-2 summary and the ROC-AUC against expert share scatter, whole model only |

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

**A neuron can be an expert for many words.** Because $\mathrm{AP}_c(\ell,u)$ is computed independently per word, nothing restricts a unit to a single word, so the expert sets $E_c$ *overlap* rather than partition the network. This is not a corner case but the norm. In the Qwen3-1.7B-Base Richie-HSJ run at AP=0.6, the 569,820 expert rows collapse to 214,988 distinct $(\ell, u)$ neurons, and **58.4% of expert neurons are experts for two or more of the 206 words** (mean 2.65 words per expert neuron, maximum 34). Restricted to the `mlp.gate_proj` sublayer the sharing is even denser (65.9% serve ≥ 2 words, mean 3.31). Three consequences matter for reading everything downstream. First, $n_c$ counts word-neuron *associations*, not private neurons, so $\sum_c n_c$ greatly exceeds the number of distinct experts. Second, two words' layer profiles are not built from disjoint resources, so comparing them compares allocation *shapes*, not competing claims on separate neurons. Third, the overlap itself is signal, since sections 2.1 and 2.2 of module 2 measure exactly this cross-word sharing, and section 1.3 uses it to rank sublayers.

**A word can also end up with zero experts.** The definition of $E_c$ makes $E_c = \varnothing$ perfectly possible: if no unit reaches $\tau$ for word $c$, the word contributes no rows at all. Such words are *silently absent* from every downstream table (they do not appear as zero-count rows), which shrinks sample sizes at strict thresholds without any explicit marker in the CSVs. The Results section quantifies where this actually happens.

#### Analysis scopes and the two layer axes

This part defines the machinery every module shares. The other module docs refer back to it rather than restating it.

**Scopes.** Each module computes its analysis once on the entire model and once per sublayer type $s$, that is on the restricted expert sets $E_c^{(s)}$. The whole-model scope writes into the module's own folder, so the big picture is what a reader sees first, and each sublayer scope writes into `<module>/sublayers/<rank>_<sublayer>/`. The rank is the sublayer's position by expert count, computed once at the most lenient AP threshold of the sweep and cached in `results/<output_subdir>/sublayer_rank.csv`. Freezing it there rather than recomputing per threshold means a given prefix names the same sublayer in every `AP_*` folder, so paths stay comparable across the sweep, which matters because stricter thresholds thin the sublayers unevenly and would otherwise reshuffle the prefixes. A sublayer left with no experts at a strict threshold is skipped with a warning instead of producing an empty folder. Within a sublayer scope the retained `layer_idx` values are non-contiguous by design (5, 12, 19, and so on for `mlp.gate_proj`) and layers keep their absolute whole-model indices, so depth statements stay comparable across scopes.

**The two layer axes.** Analyses divide into two kinds, and only one of them is affected. *Set-based* analyses (sections 2.1 to 2.3, the global prototype of section 2.4, every Jaccard computation, and section 1.3 here) treat $E_c$ as an unordered set of $(\ell, u)$ pairs, so layer order never enters and the whole-model scope needs no special handling. *Order-based* analyses (the mean and cumulative distributions and the layer descriptors of sections 1.1 and 1.2, module 3, and the per-layer prototypes of section 2.4) read $\ell$ as depth, and there the whole-model layer index is not a depth coordinate. Layers are numbered

$$\ell = b \cdot S + s + 1,$$

with $b$ the block, $S$ the number of sublayers per block, and $s$ the sublayer's position inside the block, so on Qwen3 indices 1 to 7 all belong to block 0. Adjacent indices are therefore different projection types of the *same* block, not different depths. Geary's C, which differences adjacent indices, then measures the alternation between projection types rather than depth smoothness, and the peak layer $\arg\max_\ell \bar{P}_\ell$ returns whichever sublayer is densest for nearly every word. Measured on GPT-2 at AP 0.6 in `research_plots_gpt2_richie_hsj_restructured_final`, the flat whole-model axis gives a mean per-word Geary's C of 1.027, which is the value meaning *no spatial structure at all*, and 85% of the words peak in an `mlp.c_fc` layer.

The whole-model scope therefore produces every order-based output twice:

- **`_by_block`** (canonical): the sublayers within each block are summed into one bin, $N_{cb} = \sum_{s} N_{c, bS+s+1}$, giving $B$ bins (12 for GPT-2, 28 for Qwen3) that carry the model's full expert mass on a genuine depth axis. The same GPT-2 tree gives mean Geary's C 0.357 here. This is the variant downstream modules consume.
- **`_by_layer`**: the flat axis at full resolution ($L$ = 48 or 196 bins), kept because it is the only view that separates projection types, with the caveat above attached to any depth reading of it.

A sublayer scope holds exactly one layer per block, so block aggregation would be an identity relabel there and only one variant is written, with no suffix.

**Whole-model-only outputs.** Two things are not replicated per scope. Section 1.3's `sublayer_informativeness.csv` is a cross-sublayer table by construction, so it has one natural home in the whole-model folder. The per-concept files of section 1.1 are written for the whole model only, on both layer axes, since one set per concept per sublayer per threshold would be 206 × 7 × 5 × 4 figures on Qwen3, and even the whole-model set is the slowest step of module 1.

**Which scopes write what.** Modules 1 and 5 write every sublayer scope in full. Sublayer scopes of modules 2 and 3 compute only their `sublayer_comparison` rows unless the executor's `WRITE_SUBLAYER_OUTPUTS` is switched on, since their per-sublayer plots add hours and gigabytes to a sweep while the comparison rows carry the numbers. Module 4 runs its own sweep over the sublayers. The executor's `DETAILED_AP_THRESHOLDS` limits the costly detail to thresholds 0.5, 0.6 and 0.7: above them modules 1 and 5 run the whole model alone, module 4 its analysis sublayer alone, and module 1 skips the per-concept plots.

**Cross-scope comparison.** Each module writes `sublayer_comparison.csv` and `sublayer_comparison.png`, modules 1 and 2 one pair per section in the section folder, one row per scope (whole model first) with that module's headline metrics, `n_experts` on every row so each percentage or correlation is read against the mass it rests on. These are the readable surface of the sweep, and the per-scope folders are the evidence behind them. Section 1.1's row carries the mean expert count per word at each abstraction level and the mean concept peak gap, and section 1.2's the mean entropy and Geary's C at each level, the rank-based AUC of label entropy over concept entropy, and the Spearman $\rho$ of entropy against expert count. The AUC is the density-robust column, because it contrasts the two levels within a scope, whereas the raw entropy means fall in sparse sublayers simply because fewer experts spread over the same bins. A comparison table left with the whole-model row alone, as at thresholds outside `DETAILED_AP_THRESHOLDS`, is not written. Because the block axis puts one bin per block and a sublayer scope holds exactly one layer per block, all scopes in a comparison table share the same bin count, so bin-count-sensitive quantities such as Shannon entropy are on a common scale across rows.

**How to read these tables, and the one thing they do not control for.** Every scope-level metric here is confounded with expert density, because sublayers differ enormously in how many experts they hold (on GPT-2 at AP 0.6, `mlp.c_fc` has 40,564 expert rows against `mlp.c_proj`'s 5,271, a factor of 7.7). Measured on that run, the cross-sublayer rankings produced by section 1.3's `roc_auc`, section 2.2's category contrast, and module 4's raw $\rho$ are *all* Spearman +1.00 with each other **and with the raw expert count**, so nothing in those four rankings distinguishes "this projection type carries more category structure" from "this projection type simply holds more experts". Module 4 is the only place that controls for it, by recomputing $\rho$ on a count-matched top-$k$ expert set, and doing so **re-ranks the sublayers**: `attn.c_proj` goes from third on raw $\rho$ (0.313) to first on the count-matched value (0.275), ahead of `mlp.c_fc` (0.486 raw, 0.189 matched) and `attn.c_attn` (0.392 raw, 0.190 matched). Read the comparison tables of modules 1, 2, 3 and 5 as descriptions of what each scope's expert set actually looks like, which is what they are, and read `4_embedding_rsa/sublayer_comparison.csv` for the density-controlled ranking. Extending count-matching to the other modules is an open item.

The same confound explains the sparse-scope readings that would otherwise look like findings. `attn.c_proj` and `mlp.c_proj` carry roughly 31 and 26 experts per word, so their layer distributions are sparse, which mechanically depresses Shannon entropy (section 1.2 reports 1.94 and 1.33 bits for concepts, against about 3.1 in the dense scopes) and mechanically inflates Jensen-Shannon divergence (module 3 reports 0.36 and 0.33, against 0.09 in `mlp.c_fc`). Section 1.2's `shannon_entropy_vs_expert_count` panel measures this relationship directly, at $r \approx 0.6$ to $0.7$ in every scope, and the small-denominator caveat of section 1.1's allocation profiles is the same point at the level of a single layer share.

#### Layer descriptors

Notation. Let $L$ be the number of bins in the analyzed support, which depends on the scope and layer axis: the block-aggregated whole-model axis has one bin per transformer block (12 for GPT-2, 28 for Qwen3-1.7B), the flat whole-model axis has one per layer (48 and 196), and a sublayer scope has one per layer of that projection type (12 and 28). Sections 1.1 and 1.2 receive the expert table for the current scope, so in the sublayer-restricted runs the support is one projection type (e.g. the 28 `mlp.gate_proj` layers of Qwen3-1.7B, or the 12 `mlp.c_fc` layers of GPT-2), while layers keep their absolute whole-model indices $\ell$ (1…196 resp. 1…48), non-contiguous within the support. The module starts from the concept-by-layer count matrix $N$ (built by the shared `build_layer_probability_matrix` helper, also used by module 3), where $N_{c\ell}$ counts the expert rows of item $c$ in layer $\ell$, reindexed to the full support so layers with zero surviving experts still appear. Row-normalizing gives each item's layer probability distribution

$$p_{c\ell} = \frac{N_{c\ell}}{\sum_{\ell'} N_{c\ell'}}, \qquad p_c = (p_{c\ell})_{\ell \in \text{support}}, \quad \sum_{\ell} p_{c\ell} = 1 .$$

For a category $k$, $M_k$ denotes its member concepts (restricted to those present in the matrix).

Two structural facts about the underlying data qualify every comparison below (both are established quantitatively in *From responses to expert sets* above). First, **expert sets overlap across words**: a single neuron is typically an expert for several words (58.4% of expert neurons serve ≥ 2 of the 206 words in the Qwen3-1.7B-Base run at AP=0.6, and 65.9% within `mlp.gate_proj`), so the distributions $p_c$ of different words are not built from disjoint neuron populations. Comparing two words' entropies therefore compares the *shapes* of their allocations over depth, not how they divide a fixed pool of neurons between them. Second, **expert counts differ systematically between the groups being compared**, addressed in the dedicated paragraph below.

**Mathematical formulation.** For each item $c$ (every word with at least one expert, including category labels, since they are words with expert sets too), six scalar descriptors of the distribution $p_c$ are computed by the shared `layer_distribution_descriptors` helper. The Shannon entropy (via `scipy.stats.entropy`, base 2, in bits) measures how spread out the allocation is:

$$H(c) = -\sum_{\ell} p_{c\ell}\,\log_2 p_{c\ell},$$

with the convention $0 \log_2 0 = 0$. Entropy is minimal ($H = 0$) when all experts sit in a single layer, and maximal ($H = \log_2 L$, e.g. $\log_2 28 \approx 4.807$ bits on the Qwen3 `mlp.gate_proj` support) when the allocation is uniform, so *lower entropy = more concentrated*. Dividing by that maximum gives the normalised entropy $\tilde H(c) = H(c)/\log_2 L \in [0, 1]$, the form to use whenever two axes of different length are compared, for instance the 12 GPT-2 and the 28 Qwen3 blocks. Entropy is invariant under any permutation of the layers, so it sees concentration but is blind to depth order. Its order-sensitive complement is **Geary's C**, the adjacent-layer spatial autocorrelation of the profile in depth order:

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

since $n_c$ experts can occupy at most $n_c$ distinct layers. A word with many experts *can in principle spread quite evenly over the layers*, while a word with few experts *cannot produce an even distribution at all*: with $n_c < L$ a uniform allocation is impossible, and even for $n_c \gtrsim L$ the attainable shares are coarse multiples of $1/n_c$ whose sampling noise biases entropy downward. Group comparisons must therefore be read against the groups' counts. This matters concretely here because category labels systematically have *fewer* experts than concepts (Qwen3-1.7B-Base at AP=0.6, whole model: label mean 598 experts vs. concept mean 2,854, and in the legacy 150-concept run, e.g. `animal` 231 vs. member concepts often 500 to 1,000), so part of any "labels are more concentrated" gap could in principle be a count artifact rather than an organizational fact. Two checks bound this concern. At moderate thresholds the counts are large relative to $L$ (hundreds to thousands of experts over 28–48 layers), so the bound is slack, and section 1.2's entropy against expert count panel finds essentially no relationship at AP=0.6 (Pearson r = 0.02, p = 0.87, n = 205 in the Qwen3 run), so the entropy differences there are not driven by set size. At strict thresholds, however, counts collapse into the regime where the bound binds (the Results of section 1.1 report minimum counts of 2 at AP=0.7 and 1 at AP ≥ 0.8 in the legacy 150-concept run, with most surviving words under 30 experts at AP ≥ 0.8), so the across-AP entropy collapse in the Results table below is partly mechanical, and level comparisons at AP ≥ 0.8 inherit the count confound.

**Generated data structures.** The descriptors are computed once per axis variant and written as two tables per level, split by what they describe. Section 1.1 writes the location descriptors, `layer_location_concepts.csv` with `peak_layer`, `peak_gap_pct`, `avg_layer`, `avg_layer_top25pct` and `experts_count`, and section 1.2 writes the shape descriptors, `shannon_entropy_concepts.csv` with `shannon_entropy`, `shannon_entropy_norm`, `gearys_c` and `experts_count`. Both tables have one row per word, concepts and category labels alike, keyed by `concept`:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Word identifier (concepts *and* category labels). |
| shannon_entropy | float | $H(c)$ | Entropy of the word's layerwise expert distribution. |
| shannon_entropy_norm | float | $\tilde H(c)$ | Entropy divided by $\log_2 L$, in $[0, 1]$. |
| gearys_c | float | $C(c)$ | Adjacent-layer autocorrelation of the profile in depth order. |
| peak_layer | int | $\ell^{\ast}_c$ | Absolute index of the layer with the strongest concentration. |
| peak_gap_pct | float | $\gamma_c$ | Percentage-point lead of the peak layer over the runner-up. |
| avg_layer | float | $\bar{\ell}_c$ | Expected (absolute) layer value over the full distribution. |
| avg_layer_top25pct | float | $\bar{\ell}^{\,25\%}_c$ | Expected layer over the top 25% most-loaded layers, renormalized. |
| experts_count | int | $n_c$ | Number of expert rows used to compute the distribution. |

Examples (heads of `AP_0.6/1_expert_distribution/1.1_layer_expert_distribution/layer_location_concepts_by_block.csv` and `AP_0.6/1_expert_distribution/1.2_distribution_shape/shannon_entropy_concepts_by_block.csv` in `research_plots_qwen_base_richie_hsj_FINAL`, whole model on the block axis, so peak and average layers are bin indices from 1 to 28, one per block):

| concept | peak_layer | peak_gap_pct | avg_layer | avg_layer_top25pct | experts_count |
|---|---|---|---|---|---|
| accountant | 7 | 0.1612 | 12.70 | 6.47 | 8065 |
| actor | 2 | 4.7651 | 12.56 | 5.71 | 4470 |
| airplane | 7 | 2.0526 | 14.83 | 17.26 | 1900 |

| concept | shannon_entropy | shannon_entropy_norm | gearys_c | experts_count |
|---|---|---|---|---|
| accountant | 4.5131 | 0.9388 | 0.2040 | 8065 |
| actor | 4.5557 | 0.9476 | 0.3479 | 4470 |
| airplane | 4.2981 | 0.8941 | 0.4307 | 1900 |

#### Dual category definitions: label distribution against member average

**Mathematical formulation.** Each category $k$ is represented in two independent ways, and the same six descriptors defined above are computed for both:

- *Definition A (the label itself)*: the category label is a word with its own row in the matrix, so its distribution is simply $p_k$, giving $H(p_k)$, $C(p_k)$, $\ell^{\ast}_k$, $\gamma_k$, $\bar{\ell}_k$, $\bar{\ell}^{\,25\%}_k$.
- *Definition B (average of members)*: the members' distributions are averaged element-wise,

$$\bar{q}_{k\ell} = \frac{1}{|M_k|} \sum_{m \in M_k} p_{m\ell}, \qquad \bar{q}_k = (\bar{q}_{k\ell})_{\ell},$$

which is again a valid probability distribution (a uniform mixture of the members), giving the same six descriptors, stored with the `_average` suffix. Because entropy is concave, mixing distributions can only preserve or increase entropy relative to the average of the members' entropies, so averaging tends to *wash out* concentration unless all members peak in the same layers.

To quantify whether the two definitions at least *point at the same layers*, trees written before 19 October 2026 hold the Pearson correlation between the two support-length vectors,

$$r_k = \operatorname{corr}(p_k, \bar{q}_k),$$

set to 0 when both vectors exist but either has zero variance, and NaN when one of the two representations is missing (e.g. a label with zero retained experts). High $r_k$ means the label's allocation shape mirrors the members' average shape, even if their entropies differ. This column, `pearson_correlation_distributions`, is **not generated by the current code**, since no analysis read it and module 3 compares the same two distributions through their Jensen-Shannon divergence.

**Generated data structures.** The category tables, one row per category, and one plot. Every column is listed here, and the split between the two sections follows the table:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| category | string | $k$ | Category label. |
| shannon_entropy, gearys_c, peak_layer, peak_gap_pct, avg_layer, avg_layer_top25pct | float/int | $H(p_k)$, $C(p_k)$, $\ell^{\ast}_k$, $\gamma_k$, $\bar{\ell}_k$, $\bar{\ell}^{\,25\%}_k$ | The six descriptors of the category-label distribution (Definition A). |
| shannon_entropy_average, gearys_c_average, peak_layer_average, peak_gap_pct_average, avg_layer_average, avg_layer_average_top25pct | float/int | $H(\bar{q}_k)$, $C(\bar{q}_k)$, … | The same six descriptors of the average member distribution (Definition B). |
| pearson_correlation_distributions | float | $r_k$ | Correlation between the label distribution and the member-average distribution, **not generated by the current code**. |
| member_count | int | $\lvert M_k \rvert$ | Number of valid members contributing to the category summary. |

The category table is split the same way as the concept table. `layer_location_categories.csv` in section 1.1 holds the four location descriptors for both definitions plus `member_count`, and `pearson_correlation_distributions` in trees written before 19 October 2026, as in the second example below, and `shannon_entropy_categories.csv` in section 1.2 holds entropy, normalised entropy and Geary's C for both definitions plus `member_count`.

Examples (first rows of `AP_0.6/1_expert_distribution/1.2_distribution_shape/shannon_entropy_categories_by_block.csv` and `AP_0.6/1_expert_distribution/1.1_layer_expert_distribution/layer_location_categories_by_block.csv` in `research_plots_qwen_base_richie_hsj_FINAL`, block axis, layers as bin indices from 1 to 28):

| category | shannon_entropy | shannon_entropy_norm | gearys_c | shannon_entropy_average | shannon_entropy_norm_average | gearys_c_average | member_count |
|---|---|---|---|---|---|---|---|
| furniture | 4.1462 | 0.8625 | 0.5162 | 4.3710 | 0.9092 | 0.3751 | 20 |
| clothing | 3.9478 | 0.8212 | 0.6822 | 4.2860 | 0.8916 | 0.3317 | 29 |

| category | peak_layer | peak_gap_pct | avg_layer | avg_layer_top25pct | peak_layer_average | peak_gap_pct_average | avg_layer_average | avg_layer_average_top25pct | pearson_correlation_distributions | member_count |
|---|---|---|---|---|---|---|---|---|---|---|
| furniture | 7 | 2.0000 | 15.86 | 16.75 | 1 | 0.3693 | 13.95 | 14.39 | 0.8487 | 20 |
| clothing | 7 | 12.3529 | 15.31 | 16.89 | 1 | 0.5998 | 14.20 | 14.87 | 0.6082 | 29 |

- `category_shannon_entropies_bar{suffix}.png` (section 1.2, **not generated by the current code**, its generator line being commented out in favour of the per-category figure of section 1.2), a horizontal **bar chart** of `shannon_entropy` ($H(p_k)$, x-axis) per `category` ($k$, y-axis), sorted descending, ranking categories by how distributed (top) versus localized (bottom) their label's expert allocation is.

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

**Generated data structures.** One CSV, `expert_counts_with_metadata.csv`, built by grouping the expert rows with `groupby("concept").size()` and merging on `concept` with `concept_metadata` (the metadata table, with its typicality field renamed to `human_typicality` at load time to distinguish it from the model-computed Cosine Typicality of section 2.4). It also carries the metadata's four Zipf frequency columns, SUBTLEX-US, Wikipedia, OpenWebText and FineWeb, of which the frequency panels of module 2 read the one named by the config's `frequency_corpus`, OpenWebText for GPT-2 and FineWeb for Qwen3. Rows are sorted by `(abstraction_level, concept)`, so all level-1 items form one block followed by the level-2 block. This table is not plotted here. It is the input for the correlation panels and the typicality analysis of module 2.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Concept identifier. |
| category | string | $k$ | Semantic category assigned to the concept. |
| abstraction_level | int | $a \in \{1,2\}$ | Abstraction level from the metadata. |
| frequency | float | $f_c$ | Raw frequency value from the metadata source. |
| log_frequency | float | $\tilde{f}_c$ | Base-10 logarithm of the frequency. |
| frequency_zipf_subtlex_us_lemma | float | | SUBTLEX-US lemma frequency on the Zipf scale. |
| frequency_zipf_wikipedia_lemma | float | | Wikipedia lemma frequency on the Zipf scale. |
| frequency_zipf_openwebtext_lemma | float | | OpenWebText lemma frequency on the Zipf scale, read by module 2 for GPT-2. |
| frequency_zipf_fineweb_lemma | float | | FineWeb lemma frequency on the Zipf scale, read by module 2 for Qwen3. |
| human_typicality | float | $t_c$ | Human-judged typicality score from the metadata. |
| expert_count | int | $n_c$ | Number of expert rows retained for the concept. |

Example (head of `AP_0.6/1_expert_distribution/1.1_layer_expert_distribution/expert_counts_with_metadata.csv` in `research_plots_qwen_base_richie_hsj_FINAL`, where the level-1 block comes first because of the sort):

| concept | category | abstraction_level | frequency | log_frequency | frequency_zipf_subtlex_us_lemma | frequency_zipf_wikipedia_lemma | frequency_zipf_openwebtext_lemma | frequency_zipf_fineweb_lemma | human_typicality | expert_count |
|---|---|---|---|---|---|---|---|---|---|---|
| birds | | 1 | 132116 | 5.1210 | 4.8533 | 4.9814 | 4.7259 | 4.8426 | | 801 |
| clothing | | 1 | 69314 | 4.8408 | 4.0350 | 4.4473 | 4.3961 | 4.6136 | | 170 |
| fruit | | 1 | 86371 | 4.9364 | 4.3897 | 4.6830 | 4.5626 | 4.9036 | | 434 |
| furniture | | 1 | 44365 | 4.6470 | 4.1784 | 4.2535 | 4.0447 | 4.7484 | | 400 |
| professions | | 1 | 10176 | 4.0076 | 3.9736 | 4.3066 | 4.1692 | 4.3917 | | 1048 |

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

**Generated data structures.** One summary table and two plots per layer-axis variant, so three files in a sublayer scope and six in the whole-model scope:

- `mean_expert_layer_distribution_summary{suffix}.csv`, one row per abstraction level with the milestones of both views: `n_words`, `n_experts`, `n_bins` ($L$), `cumulative_50pct_position` and `cumulative_90pct_position` (the position of $\ell^{(a)}_{50}$ and $\ell^{(a)}_{90}$ along the axis counted from 0, which on the block axis is the block number) with their `_layer_name`, `top_bins_50pct` and `top_bins_90pct` ($k_{50}$ and $k_{90}$ of the Pareto view), and the share of the mean profile in each third of depth, `first_third_pct`, `middle_third_pct` and `last_third_pct`, each beside its expert count (`_experts`). The thirds cut the $L$ bins into three contiguous runs as equal as possible, longer runs first, so 4, 4 and 4 blocks for GPT-2 and 10, 9 and 9 for Qwen3. Writing the table also checks that each level's mean profile sums to 100% and logs a warning otherwise.

- `mean_expert_layer_distribution_cumulative{suffix}.png`: depth-ordered bars ($\bar{P}^{(a)}_{\ell}$, both abstraction levels side by side, left y-axis) with each level's cumulative curve $C^{(a)}_{\ell}$ on a secondary 0–100% axis. The 50% and 90% crossings are flagged with diamond markers, dotted projection lines onto both axes, and legend labels naming the crossing layers. On the block axis the ticks are block numbers counted from 0, as in the models' own module names (`h.0`, `layers.0`), and the legend names the crossing block, while the flat layer axis keeps the full layer names.
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

**Generated data structures.** The $U$ statistic, p-value, and AUC are logged to `main.log` (not saved as CSV). The comparison used to be drawn as the plot below, whose generator line is now commented out because pooling 198 members against 8 labels hides the categories, and the per-category figure further down replaces it:

- `category_concept_shannon_entropies{suffix}.png` (**not generated by the current code**, its generator line being commented out in favour of the per-category figure below), a **violin plot** (drawn by the shared `plot_comparison_violin` helper: quartile lines with Q1/median/Q3 text labels, violins clipped to the observed data range, and a strip plot of the raw points overlaid so small groups stay honest) comparing `shannon_entropy` ($H$, y-axis) across the two groups Specific Concepts vs. Broad Categories (x-axis).

#### Profile-shape map: Geary's C against Shannon entropy

No longer drawn. Its generator line is commented out, since the map does not show which members belong to which label, and the per-category figure below replaces it. The description stays for reference.

**Mathematical formulation.** Entropy and Geary's C answer orthogonal questions about a layer profile, namely *how concentrated* (order-blind) versus *how smooth in depth* (order-sensitive), so plotting every word at coordinates $(H(c), C(c))$ maps the space of profile shapes. The four corners of the map read: concentrated & smooth (low $H$, low $C$: one compact band of layers), concentrated & jagged (low $H$, high $C$: a few isolated spikes), spread & smooth (high $H$, low $C$: broad contiguous mass), spread & jagged (high $H$, high $C$: scattered mass with no depth locality). The dashed horizontal line at $C = 1$ marks "no depth structure", and dotted lines through the middle of each axis quarter the map.

**Generated data structures.** One plot (inputs are the descriptor tables):

- `gearys_entropy_scatter{suffix}.png` (**not generated by the current code**, commented out in favour of the same per-category figure), a **scatter plot**: every specific concept as a small dot (category-label words are excluded from the dots, since they appear in the concepts CSV too, and would otherwise be plotted twice), every category label as a larger named diamond, the 5 most extreme concepts on each end of *both* axes annotated by name, and quadrant guides and the $C=1$ reference line drawn as described, with corner glosses naming the four shape regimes.

#### Per-category spread and smoothness, members against their category label

**Mathematical formulation.** The violin and the shape map above pooled the 8 category labels against all members, which hid the categories and let one label be compared with another category's members. This figure compares each label $k$ with its own members $M_k$ only. Spread is the entropy normalised by the number of bins, $\tilde H(c) = H(c)/\log_2 L$, which lies in $[0, 1]$ and is comparable between the 12 GPT-2 and the 28 Qwen3 blocks, and smoothness is Geary's C as defined above. For either descriptor $x$, the label's percentile among its members is

$$\pi_k = \frac{1}{|M_k|} \sum_{c \in M_k} \Big( \mathbb{1}[x_c < x_k] + \tfrac{1}{2}\, \mathbb{1}[x_c = x_k] \Big),$$

0.5 when the label sits at its members' median. Labels hold fewer experts than their members, and a sparse profile is by construction both less spread and rougher, so the comparison is repeated at the label's own expert count $n_k$. Each member with at least $n_k$ experts is subsampled to exactly $n_k$ of them, without replacement and with its block shares preserved in expectation (200 multivariate hypergeometric draws, seed 0), its descriptor is averaged over the draws, and $\pi_k$ is recomputed against these count-matched values. Members with fewer experts than their label cannot be subsampled up and are left out of the matched comparison, and `n_members_matched` reports how many remain.

**Generated data structures.** Per axis variant, `category_entropy_gearys{_by_block,_by_layer}.png` and its table `category_entropy_gearys{_by_block,_by_layer}.csv`, one row per category with `n_members`, `label_expert_count`, `n_members_matched` and, for `entropy_norm` and `gearys_c`, the members' median and quartiles (`_members_median`, `_members_q1`, `_members_q3`), the label's value (`_label`), its percentile (`_label_percentile`), the members' median at the label's count (`_members_matched_median`) and the label's percentile against those (`_label_percentile_matched`). The figure has two panels sharing one row per category, the members as a box over their quartiles with their points, the label as a diamond, a grey tick at the count-matched member median, and the two percentiles printed above each row. `check_module1_category_shape` pins the percentile, the normalised entropy, the agreement with `gearys_c`, and the exclusion of members too small to match.

**What it shows**, at AP 0.6 on the block axis. In Qwen3 the label's percentiles barely move under count matching, except `professions` in smoothness (0.61 to 0.32), three of the eight labels sit above their members' median in spread (four once counts are matched), and six are rougher than their members' median (five once matched). In GPT-2 matching changes the picture: the labels hold only 25 to 189 experts, and their Geary's C percentiles fall once the members are cut to the same count, in professions from 0.61 to 0.04, vehicles from 0.59 to 0.23 and vegetables from 0.30 to 0.06, so GPT-2's label-member differences in smoothness are mostly an effect of the labels' small expert sets.

#### Shannon entropy against expert count (both abstraction levels)

**Mathematical formulation.** This panel is the empirical check behind the count-dependence caveat of *Layer descriptors*: entropy is bounded by the expert count ($H(c) \le \log_2 \min(n_c, L)$), so if entropy and expert count were strongly correlated in the analyzed range, any entropy comparison between groups with different typical counts (category labels hold systematically fewer experts than concepts) would partly measure set size rather than layer organization. The test is the pooled Spearman correlation, the pooled modality of the README's *Correlations and permutation tests*, on the pair

$$x = H(c), \qquad y = n_c,$$

over *every* word in the concept descriptor table, which contains both abstraction levels, since category labels are words with expert distributions too. The two levels are separated only for display (words are classified by membership in the category list), while $\rho$ and its two-sided permutation $p$ (9,999 reassignments) are computed on the pooled sample. A near-zero $\rho$ licenses reading this section's entropy contrasts as organizational, while a strong positive $\rho$ would flag them as count artifacts. Spearman replaced Pearson on 17 October 2026, since the expert count is strongly right-skewed (skewness about 2.2) and Pearson's $r$ read mostly the few largest sets, 0.30 against $\rho$ 0.40 on Qwen3 and 0.65 against 0.74 on GPT-2 at AP 0.6 on the nopad trees. Since 19 October 2026 Pearson's $r$ is no longer written, see module 2's correlation panels. A word is missing here only when the descriptor computation dropped it for having zero experts across every layer, so this panel's coverage denominator is the full metadata count, the same one module 2's expert-count panels use.

**Generated data structures.** One CSV, one PNG, and one row in `correlation_summary.csv`:

- `shannon_entropy_vs_expert_count.csv`, the working table:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Word identifier. |
| group | string | | "Specific Concepts" or "Broad Categories" (display split). |
| shannon_entropy | float | $H(c)$ | Entropy of the word's layer distribution. |
| experts_count | int | $n_c$ | Number of expert rows behind the distribution. |
| shannon_entropy_rank_z, experts_count_rank_z | float | | Rank scores of the two variables, average ranks z-scored over the words of the panel, whose least-squares slope is the reported $\rho$. |

Example (head of `AP_0.6/1_expert_distribution/1.2_distribution_shape/shannon_entropy_vs_expert_count.csv` in `research_plots_qwen_base_richie_hsj_FINAL`, block axis, written before the rank-score columns):

| concept | group | shannon_entropy | experts_count |
|---|---|---|---|
| accountant | Specific Concepts | 4.5131 | 8065 |
| actor | Specific Concepts | 4.5557 | 4470 |
| airplane | Specific Concepts | 4.2981 | 1900 |
| apple | Specific Concepts | 3.9472 | 564 |
| apricot | Specific Concepts | 4.0371 | 5592 |

- `shannon_entropy_vs_expert_count.png`, a scatter of the raw values over the pooled sample, x: `shannon_entropy` labelled "Shannon entropy (bits)", y: `experts_count` labelled "Expert count", with concepts drawn as small dots and category labels as larger diamonds, the least-squares line of the raw values, a legend naming the two levels and the line, and the pooled Spearman $\rho$, its permutation $p$ and $n$ on the grey line under the title. The line is a visual guide, as the README's *Correlations and permutation tests* explains, while $\rho$ is computed on the ranks. Trees written before 19 October 2026 hold the same figure with Pearson's r (the FINAL trees) or with Spearman's $\rho$ and the t-based p of `scipy.stats.spearmanr` (the nopad trees written before 20 October, kept as `research_plots_*_nopad_before_2026-10-20`) on the grey line, both with a 95% bootstrap band about the line.

### 1.3 Sublayer informativeness

Which sublayer types carry the most category structure. Written once, for the whole model, to `1.3_sublayer_informativeness/`.

**Mathematical formulation.** Transformer layers come in a small number of *sublayer types* (projections), parsed from the layer label `{idx}.L.{block}.{sublayer}`, with 4 types for GPT-2 (`attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`) and 7 for Qwen3 (`self_attn.q/k/v/o_proj`, `mlp.gate/up/down_proj`). This analysis ranks the types by how much category structure their expert sets carry, which is the empirical justification for the analysis sublayer module 4 reports as its headline. For each sublayer $s$, restrict every item's expert set to that sublayer's layers, $E_c^{(s)}$, and compute for every unordered pair of level-2 concepts the Jaccard similarity of the restricted sets,

$$J^{(s)}_{cd} = \frac{|E^{(s)}_c \cap E^{(s)}_d|}{|E^{(s)}_c \cup E^{(s)}_d|},$$

together with the indicator of whether $c$ and $d$ belong to the same category. Two alignment statistics follow (level-2 concepts only, since category labels have no same-category peers, so level-1 rows carry NaN here), both from `shared_expert_set_measures.category_alignment_metrics`, which section 2.2 of module 2 uses too:

- **`roc_auc` (primary)**: the probability that a randomly drawn same-category pair is more similar than a randomly drawn different-category pair. It is computed rank-based via the Mann–Whitney identity, $\mathrm{AUC} = \big(\sum_{\text{same}} R_i - \tfrac{n_1(n_1+1)}{2}\big) / (n_1 n_0)$ with $R_i$ the ranks of the pair similarities (average ranks for ties), which makes it immune to the heavy same/different pair imbalance and to the skewed shape of Jaccard values. 0.5 = no category signal, 1.0 = every within-category pair beats every across-category pair.
- **`mantel_p`**: the two-sided Mantel p of the AUC. Writing $\mathrm{same}_{cd} \in \{0,1\}$ for the indicator that $c$ and $d$ share a category, the AUC is an affine function of Spearman's $\rho\big(\mathrm{same}_{cd}, J^{(s)}_{cd}\big)$ over the same pairs, so testing $\rho$ tests the AUC. Pairs are not independent observations (each concept participates in $|\mathcal{C}_2|-1$ pairs), so the null distribution is built by reassigning category labels across *concepts* (never across pairs) and recomputing $\rho$, 9,999 times from a generator of its own per sublayer, the matrix modality of the README's *Correlations and permutation tests*, and $p = (1 + \#\{|\rho_b| \ge |\rho|\}) / (1 + 9{,}999)$, so the smallest reportable value is $10^{-4}$. Degenerate sublayers whose pair similarities are all identical give AUC = 0.5 and $p$ = 1.

Until 19 October 2026 the p counted only permutations with an AUC at least as high, one-sided, from one generator seeded at 42 and consumed in turn by every sublayer, and a companion column, `category_alignment_r`, held the point-biserial Pearson correlation of $\mathrm{same}_{cd}$ with $J^{(s)}_{cd}$. That column is **not generated by the current code**, since its Spearman counterpart is the AUC rescaled.

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
| mantel_p | float | $p$ | Two-sided concept-label permutation p-value for the AUC (level 2 only). |
| share_of_all_experts_pct | float | | That level's experts in this sublayer as % of all experts. |
| n_experts | int | | Raw expert count behind the share. |
| n_words | int | | Words of that level with at least one expert in the sublayer. |
| mean_entropy_bits | float | $\overline{H}$ | Mean Shannon entropy of the words' profiles on the sublayer's own block axis. |
| mean_entropy_norm | float | $\overline{\tilde H}$ | The same mean divided by $\log_2 L$. |
| reliable_peak_pct | float | | Share of words whose peak leads the runner-up by at least `PEAK_GAP_RELIABLE_PP` (1 pp). |
| mean_depth_top25_block | float | $\overline{\ell}^{\,25\%}$ | Mean top-25% average depth, in blocks. |
| mean_depth_norm | float | $\bar d$ | Mean full average depth rescaled to $[0, 1]$, $d_c = (\bar{\ell}_c - 1)/(L - 1)$ with blocks numbered 1 to $L$. |
| mean_expert_count | float | $\bar{n}$ | Mean experts per word in the sublayer. |

Example (level-2 block of `AP_0.6/1_expert_distribution/1.3_sublayer_informativeness/sublayer_informativeness.csv` in `research_plots_qwen_base_richie_hsj_FINAL`, first eight columns, values rounded, with the `category_alignment_r` column and the one-sided `mantel_p` of that date):

| abstraction_level | sublayer | gearys_c | roc_auc | category_alignment_r | mantel_p | share_of_all_experts_pct | n_experts |
|---|---|---|---|---|---|---|---|
| 2 | mlp.gate_proj | 0.327 | 0.951 | 0.647 | 0.0001 | 42.77 | 243707 |
| 2 | mlp.up_proj | 0.246 | 0.947 | 0.596 | 0.0001 | 26.92 | 153370 |
| 2 | self_attn.o_proj | 0.963 | 0.920 | 0.589 | 0.0001 | 10.94 | 62361 |
| 2 | self_attn.v_proj | 0.531 | 0.863 | 0.490 | 0.0001 | 3.82 | 21757 |
| 2 | self_attn.k_proj | 0.520 | 0.797 | 0.443 | 0.0001 | 2.69 | 15324 |
| 2 | self_attn.q_proj | 0.453 | 0.786 | 0.421 | 0.0001 | 4.26 | 24269 |
| 2 | mlp.down_proj | 0.777 | 0.691 | 0.244 | 0.0001 | 7.77 | 44247 |

The descriptor columns are computed on the sublayer's own block axis, one bin per block, so they sit on a common scale across sublayers. `sublayer_informativeness.png` draws seven of the columns as small multiples, one panel per column, sublayers as rows in level-2 ROC-AUC order and one bar per abstraction level: category alignment ROC-AUC, mean Geary's C, mean entropy, reliable-peak share, mean top-25% depth, mean experts per word and share of all experts.

`sublayer_informativeness_summary.csv` keeps only what a thesis table needs. One row per sublayer, in level-2 ROC-AUC order, gives its `group` (MLP or attention, read from the name), the share of the specific concepts' experts it holds (`concept_share_pct`, beside `concept_experts`) and the share of the category labels' experts (`label_share_pct`, beside `label_experts`), each share taken within its own level so that it sums to 100% over the sublayers, then the level-2 `roc_auc`, `mean_depth_norm` and `mean_entropy_norm`. Two grouped rows close the table, all MLP sublayers and all attention sublayers, with shares and counts summed. `sublayer_auc_vs_share.png` draws the sublayer rows as one scatter, ROC-AUC against the concepts' share, one point per sublayer coloured by projection type, with chance (0.5) dashed, a one-panel alternative to the strip above.

## Results

### Layer distribution (section 1.1)

**Legacy.** At AP=0.6 in the GPT-2 150-concept reference run (`research_plots_150_revised_executor_again`, whole-model expert sets), layer allocation is genuinely concentrated, not uniform. With 48 layers, a uniform allocation would put 10.4% of experts in any 5 layers. Instead, the top 5 layers hold **44.0%** of all expert mass for broad categories (abstraction level 1) and **35.4%** for specific concepts (level 2). The single busiest layer for both levels is `3.L.0.mlp.c_fc` (an early MLP layer), at 13.5% for categories and 11.3% for concepts. This is visible directly in the mean-distribution bars of the cumulative plot: a handful of sharp early-to-mid-layer peaks stand out clearly against a low, noisy baseline across the rest of the network.

#### Across AP thresholds (legacy, 150-concept run)

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

Not only layers but *words* can end up empty (see *From responses to expert sets*). **Legacy.** In the 150-concept reference run (the same run as the tables above, 164 metadata words, 147 concepts and 17 category labels):

| AP | Words with ≥1 expert (of 164) | Zero-expert words | Words with <30 experts | Smallest count |
|---|---|---|---|---|
| 0.5 | 164 | 0 | 0 | 162 |
| 0.6 | 164 | 0 | 1 | 25 (`clothing`) |
| 0.7 | 164 | 0 | 13 | 2 (`clothing`) |
| 0.8 | 161 | 3 | 94 | 1 |
| 0.9 | 119 | 45 | 101 | 1 |

At AP ≤ 0.7 there are **no zero-expert cases**: every word keeps at least one expert. The first casualties appear at AP=0.8 (`clothing`, `hopscotch`, `toy`, noting that `clothing` and `toy` are category *labels*, so a whole category loses its Definition-A representation), and at AP=0.9 the phenomenon is widespread: **45 of 164 words (27%) retain zero experts**, including the labels `animal` and `clothing`. This is exactly what silently shrinks the downstream sample sizes reported by the other modules (section 2.1's pair count falling 147 → 128 → 59 across AP 0.7–0.9, module 3's category count falling 17 → 15 → 9). The Richie-HSJ runs behave the same way, and the two architectures differ sharply in how fast they thin out. Of the 206 words of the final trees, GPT-2 in `research_plots_gpt2_richie_hsj_restructured_final` and Qwen3-1.7B-Base in `research_plots_qwen_base_richie_hsj_FINAL`, which every Qwen3 figure of these Results comes from:

| AP | GPT-2 surviving | GPT-2 zero | GPT-2 <30 experts | Qwen3 surviving | Qwen3 zero | Qwen3 <30 experts |
|---|---|---|---|---|---|---|
| 0.5 | 206 | 0 | 0 | 206 | 0 | 0 |
| 0.6 | 206 | 0 | 2 | 206 | 0 | 0 |
| 0.7 | 206 | 0 | 51 | 206 | 0 | 0 |
| 0.8 | 189 | 17 | 115 | 206 | 0 | 8 |
| 0.9 | 115 | 91 | 98 | 201 | 5 | 92 |

GPT-2 loses its first words at AP=0.8 and by AP=0.9 has lost 91 of 206, while Qwen3 keeps every word through AP=0.8 and loses only 5 at AP=0.9. The larger model simply has more units available to clear a strict bar, so results at AP≥0.8 rest on a much thinner GPT-2 sample than the same threshold implies for Qwen3.

The middle column matters just as much as the zero column. Well before a word reaches zero, it passes through the low-count regime where the reliability caveat of the allocation profiles bites: in the legacy 150-concept run at AP=0.8, 94 of the 161 surviving words have fewer than 30 experts, so a single expert is 3–100 percentage points of the layer profile, and per-word percentage profiles (and any shift in them between thresholds) are dominated by sampling noise. Group-level statements remain meaningful at strict thresholds because they pool words, but per-word layer percentages at AP ≥ 0.8 should not be over-interpreted.

The cumulative plots quantify concentration for Qwen3 at AP=0.6. On the flat 196-layer axis, the top 13 layers hold 50% of the level-1 mass (top 23 for level 2), and 53 and 78 layers respectively are needed for 90%. On the block axis, which is the one that reads as depth, 7 and 8 of 28 blocks hold 50% of the mass, and 18 and 17 blocks reach 90%. GPT-2 is more concentrated still: 4 of 12 blocks hold 50% at both levels, 9 and 10 blocks reach 90%, and on its flat 48-layer axis 7 and 9 layers hold 50%.

The per-level summary, `mean_expert_layer_distribution_summary_by_block.csv`, adds where along depth that mass accumulates (final trees). On Qwen3 the specific concepts' mean profile reaches 50% at block 20 at AP 0.5 and block 19 at AP 0.6, and 90% at block 26 at both, and puts 40 to 44% of its mass in the first third of depth (blocks 0 to 9), only 6 to 9% in the middle third (blocks 10 to 18) and 50 to 52% in the last (blocks 19 to 27), so the experts sit in an early and a late band with little between them. GPT-2 reaches 50% earlier, at block 5 at AP 0.5 and block 4 at AP 0.6, and 90% at block 11, with 43 to 49% in the first third (blocks 0 to 3), 15 to 18% in the middle and 35 to 39% in the last. In GPT-2 the category labels reach 50% two blocks later than their members at both thresholds, in Qwen3 two blocks earlier at AP 0.5 and at the same block at AP 0.6.

### Distribution shape (section 1.2)

*Scope note.* The 150-concept figures below are **legacy**: they come from the runs that predate the whole-model refactor, so they describe the **analysis sublayer** only. The Richie-HSJ sections are reported from the final trees at both scopes, Qwen3 being Qwen3-1.7B-Base, and the scope turns out to matter more than any other choice in this module, so the two are kept separate and labelled throughout.

**Legacy.** At AP=0.6 in the GPT-2 150-concept reference run, the hypothesis holds, but only modestly in magnitude. Mean entropy for the 164 concepts is **4.49 bits** (SD 0.35) versus **4.18 bits** (SD 0.33) for the 17 category labels, so categories are lower-entropy (more concentrated) than concepts, and a Mann-Whitney U test confirms this is unlikely to be chance (U=698.5, p=3.6×10⁻⁴). But the absolute gap is small relative to the scale: the maximum possible entropy over 48 layers is log₂(48) ≈ 5.58 bits, so both groups sit well below the ceiling and only about 6% apart from each other. `category_concept_shannon_entropies.png`, a figure not generated by the current code, showed this directly: the two violins clearly overlap, with the category distribution shifted down and narrower, not cleanly separated from the concept distribution.

A more striking, and easy to miss, pattern sits in the same CSV: averaging a category's *members* together (`shannon_entropy_average`, mean 4.79 bits) produces the **least** concentrated distribution of the three, even less concentrated than individual concepts, let alone the category label itself. In other words, the category-label word is the sharpest of the three representations, but smoothing across its members washes that sharpness out rather than preserving it. This is exactly the gap that module 3 quantifies directly with Jensen-Shannon divergence, and it complicates a simple "category labels are just an average of their members" story.

#### Across AP thresholds (legacy, 150-concept run)

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

Reported from the final trees (206 words, the 198 members with both `squash` homonyms plus the 8 category labels), the whole model read on the flat layer axis. The concentration hypothesis is answered differently depending on whether the expert set is read over the whole model or restricted to the analysis sublayer, and that difference is the main finding of this section. Entropy tests are one-sided (categories more concentrated), depth tests two-sided, computed over the 8 labels against the 198 members.

**Whole model, flat layer axis.** The hypothesis holds, clearly, in both architectures:

| Model | AP | Entropy, label vs concept (p) | Avg layer, label vs concept (p) | Trimmed avg (p) |
|---|---|---|---|---|
| GPT-2 | 0.5 | 4.62 vs 4.78 (0.003) | 25.6 vs 23.5 (0.023) | 26.2 vs 23.4 (0.13) |
| GPT-2 | 0.6 | 4.05 vs 4.41 (0.001) | 25.3 vs 21.5 (0.021) | 25.6 vs 20.5 (0.030) |
| Qwen3 | 0.5 | 6.07 vs 6.34 (<0.001) | 106.6 vs 109.5 (0.29) | 110.0 vs 112.9 (0.36) |
| Qwen3 | 0.6 | 5.48 vs 6.01 (<0.001) | 107.6 vs 104.2 (0.38) | 109.0 vs 106.0 (0.56) |

**Analysis sublayer only** (`mlp.c_fc`, `mlp.gate_proj`). The same test on the same words returns nothing:

| Model | AP | Entropy, label vs concept (p) | Avg layer, label vs concept (p) |
|---|---|---|---|
| GPT-2 | 0.5 | 3.23 vs 3.27 (0.17) | 26.6 vs 23.9 (0.032) |
| GPT-2 | 0.6 | 2.94 vs 3.04 (0.05) | 26.8 vs 21.8 (0.017) |
| Qwen3 | 0.5 | 4.42 vs 4.36 (0.72) | 113.8 vs 106.9 (0.039) |
| Qwen3 | 0.6 | 4.21 vs 4.22 (0.34) | 114.6 vs 102.0 (0.004) |

Three readings follow.

First, **the concentration contrast is a whole-model property**. Restricted to one projection type the levels are indistinguishable (p 0.05 to 0.72), while pooled over all sublayers the labels are reliably sharper (p ≤ 0.004 everywhere). The mechanism is visible in section 1.1: a category label spreads its experts over fewer sublayer types than a concept does, so restricting to a single sublayer discards exactly the axis along which labels are concentrated. The block axis, which sums the sublayer types within each block, discards that axis too, and there the labels are no sharper than their members either (p 0.25 at both thresholds in GPT-2, 0.85 and 0.49 in Qwen3). Any claim about level-1 versus level-2 concentration must therefore state its scope and its layer axis, and the earlier sublayer-only null should not be read as a null for the models.

Second, **depth separates the levels in GPT-2 but not in Qwen3**. GPT-2 labels sit consistently deeper than their members on the average layer (p 0.021 and 0.023 whole model, 0.017 and 0.032 in `mlp.c_fc`), while on Qwen3 the whole-model comparison is flat (p 0.29 and 0.38) and only the sublayer-restricted comparisons separate, with the labels deeper (p 0.039 and 0.004). Depth is the more architecture-dependent of the two contrasts, and the reverse of what the earlier sublayer-only reading suggested.

Third, **the three-way ordering label < concepts < avg-member holds on Richie-HSJ too**, matching the 150-concept run: at AP 0.6 GPT-2 gives 4.05 < 4.41 < 4.74 and Qwen3 gives 5.48 < 6.01 < 6.29. Averaging a category's members produces the least concentrated of the three representations in every run and at every threshold, which is the gap module 3 measures directly. Label-versus-member shape correlation (`pearson_correlation_distributions`, **not generated by the current code**) is 0.85 and 0.73 for GPT-2 at AP 0.5 and 0.6, and 0.85 and 0.77 for Qwen3, weakening as the threshold tightens exactly as in the 150-run.

**Peak reliability and smoothness.** Ambiguous peaks ($\gamma < 1$ pp) are common and far more so in the larger model: at AP 0.6, 27% of GPT-2 concepts and 50% of its labels are ambiguous, against 57% and 75% for Qwen3, so the hatched segments of the peak-layer histogram carry most of the mass in the Qwen3 figures and peak locations there should not be read as point estimates. On the block axis, mean Geary's C is 0.35 (concepts) and 0.41 (labels) for GPT-2 at AP 0.6, and 0.36 and 0.46 for Qwen3, all far below the $C=1$ no-structure line, so expert mass sits in contiguous depth bands in both models. Read on the flat layer axis instead, the same GPT-2 statistic is 1.02, the textbook value for no spatial structure at all, which is the artifact *Analysis scopes and the two layer axes* warns about and the reason the block axis is canonical for every depth-ordered descriptor here.

#### Entropy against expert count

Shannon-entropy-vs-expert-count behaves very differently in the two architectures, and this matters for how the entropy contrasts of section 1.2 are read. The values below are **Earlier definition**, Pearson's $r$, where the current Spearman $\rho$ with its permutation p reads 0.40 / 0.74 (Qwen3 / GPT-2) on the nopad trees at AP 0.6. In Qwen3 it is weak at lenient thresholds (r = 0.252 at AP 0.5, 0.160 at AP 0.6, 0.132 at AP 0.7) and rises at the strict ones (0.221 at AP 0.8, 0.467 at AP 0.9), the $H \le \log_2 n_c$ ceiling effect the entropy against expert count panel predicts. In GPT-2 it is strong at *every* threshold (0.655, 0.640, 0.569, 0.554). GPT-2 words hold far fewer experts than Qwen3 words at the same AP, so GPT-2 sits in the count-limited regime from the start. The entropy contrasts are therefore largely count-independent for Qwen3 at AP 0.5 to 0.7, but for GPT-2 an entropy difference between two groups is partly a count difference at any threshold, and the level-1 versus level-2 entropy gap there should be read with the label-versus-concept count gap in view.

### Sublayer informativeness (section 1.3)

The sublayer ranking cleanly separates the projection types, in the same order for both architectures' MLP-vs-attention contrast. For Qwen3 (table in section 1.3): `mlp.gate_proj` and `mlp.up_proj` are nearly tied at the top (AUC 0.951 and 0.947 at AP 0.6) and jointly hold 70% of all experts, attention projections rank middle (o_proj 0.920 down to q_proj 0.786), and `mlp.down_proj` is by far the least category-aligned (AUC 0.691, point-biserial r 0.244 in the `category_alignment_r` column, **not generated by the current code**). Every sublayer's `mantel_p` sits at the $10^{-4}$ permutation floor (**Earlier definition**, one-sided, while the current test is two-sided, and the whole-model $\rho$ behind the AUC, 0.520 / 0.443 on the nopad trees at AP 0.6, also has p 0.0001), so *all* sublayers carry statistically real category signal, and the ranking is about how much, not whether. `mlp.gate_proj` also dominates the level-1 side, holding 3,153 of the 4,785 category-label experts (66%). For GPT-2 on the same dataset, the winner is `mlp.c_fc` (AUC 0.879, against 0.796 for `attn.c_attn`, 0.691 for `attn.c_proj`, 0.571 for `mlp.c_proj`). These rankings are what the configured `sublayer_filter` values (`mlp.gate_proj`, `mlp.c_fc`), the analysis sublayer of module 4, implement. The Geary's C column adds a shape observation: the category-aligned MLP input projections have smooth depth profiles (mean C of 0.25 and 0.33), while `self_attn.o_proj` (C of 0.96) is essentially depth-unstructured.

At the more lenient AP=0.5 the same ordering holds with higher values throughout (Qwen3: gate_proj 0.958, up_proj 0.957, o_proj 0.949, v_proj 0.933, q_proj 0.915, k_proj 0.915, down_proj 0.808, and for GPT-2: c_fc 0.931, attn.c_attn 0.903, attn.c_proj 0.901, mlp.c_proj 0.688), so the ranking is stable across thresholds rather than an artifact of one cut.

`sublayer_informativeness_summary.csv` splits each level's experts over the sublayers, every share within its own level (final trees). On Qwen3 the MLP sublayers hold 76.2% of the specific concepts' experts at AP 0.5 and 78.1% at AP 0.6, and a larger 81.3% and 86.7% of the category labels' experts, `mlp.gate_proj` alone holding 43% of the concepts' experts and 66% of the labels' at AP 0.6. On GPT-2 the two levels split alike, 57.7% and 57.9% MLP at AP 0.5, 60.5% and 62.3% at AP 0.6. `sublayer_auc_vs_share.png` shows that in GPT-2 the share of experts and the category alignment fall together down the four sublayers, while in Qwen3 `mlp.down_proj` breaks that order, the least aligned (0.69) although it holds 8% of the concepts' experts, more than any query, key or value projection, and `self_attn.o_proj` stays close to the MLP input projections, aligned at 0.92 from 11%.
