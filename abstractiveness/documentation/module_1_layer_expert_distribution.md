# Module 1: Layer-wise Expert Distribution

## Research question

Are expert allocations concentrated in a subset of model layers rather than spread evenly, and does that concentration pattern differ between specific concepts (abstraction level 2) and broad categories (abstraction level 1)? And, since every layer of a transformer block plays a different functional role (attention vs. MLP projections), *which sublayer type* carries the most category-relevant expert structure, so that the downstream modules can focus on it?

## Analysis

Notation. Let $\mathcal{C}$ be the set of analyzed items (concepts and category labels), partitioned by abstraction level into $\mathcal{C}_1$ (broad categories) and $\mathcal{C}_2$ (specific concepts). Let $L$ be the number of model layers, 48 for GPT-2 (12 blocks × 4 projections) and 196 for Qwen3-1.7B (28 blocks × 7 projections), where the layer label encodes its position as `{layer_idx}.L.{block}.{sublayer}`, e.g. `5.L.0.mlp.gate_proj`. $E_c$ is the set of expert units retained for item $c$ after AP filtering, $n_c = |E_c|$ its expert count, and $N_{c\ell}$ the number of expert rows of item $c$ in layer $\ell$. Every module, this one included, runs once per **analysis scope**: first the whole model, then one scope per sublayer type. Subchapter 1.7 defines the scopes and the two layer axes they can be read on, and every other module doc refers back to it.

### 1.1 From responses to expert sets (data provenance)

**Procedure.** The expertise data this whole pipeline consumes is produced upstream by the self-conditioning expertise stage (`compute_responses.py` + `compute_expertise.py`). For each word $c$, a labeled sentence corpus $S_c = S_c^{\text{on}} \cup S_c^{\text{off}}$ is assembled: "on" sentences that use the word ($y_s = 1$) and "off" sentences that do not ($y_s = 0$). Every sentence is run through the model, and each unit $(\ell, u)$ records one scalar response per sentence, $z_{\ell u}(s)$ (its maximum activation over the sentence's tokens). The unit's *expertise* for the word is then the Average Precision obtained when its responses are used as a score for detecting the "on" label:

$$\mathrm{AP}_c(\ell, u) = \mathrm{AP}\big(\{(z_{\ell u}(s),\, y_s) : s \in S_c\}\big) \in [0, 1],$$

i.e. the area under the precision–recall curve of the ranking that the unit's activations induce over the sentences. $\mathrm{AP} = 1$ means the unit's activation perfectly separates on-sentences from off-sentences, while a value near the positive-class base rate means the unit carries no information about the word. Each word's results are written to `expertise/expertise.csv`, one row per unit, with columns including `ap`, the activation statistics `off_mean`, `on_p50`, `on_p90` (summaries over the off/on sentence sets), `layer`, `unit`, and `concept`. The analysis pipeline (`load_experts_data`) then applies the AP threshold $\tau$ to define the expert set

$$E_c = \{(\ell, u) : \mathrm{AP}_c(\ell, u) \ge \tau\},$$

and concatenates all words' surviving rows into the (concept, layer, unit) table every module works on. All results folders are parameterized by this $\tau$ (`AP_0.5` … `AP_0.9`).

**A neuron can be an expert for many words.** Because $\mathrm{AP}_c(\ell,u)$ is computed independently per word, nothing restricts a unit to a single word, so the expert sets $E_c$ *overlap* rather than partition the network. This is not a corner case but the norm. In the Qwen3 Richie-HSJ run at AP=0.6, the 414,089 expert rows collapse to 170,001 distinct $(\ell, u)$ neurons, and **54.9% of expert neurons are experts for two or more of the 204 words** (mean 2.44 words per expert neuron, maximum 33). Restricted to the `mlp.gate_proj` sublayer the sharing is even denser (63.7% serve ≥ 2 words, mean 3.10). Three consequences matter for reading everything downstream. First, $n_c$ counts word-neuron *associations*, not private neurons, so $\sum_c n_c$ greatly exceeds the number of distinct experts. Second, two words' layer profiles are not built from disjoint resources, so comparing them compares allocation *shapes*, not competing claims on separate neurons. Third, the overlap itself is signal, since modules 3 and 5 measure exactly this cross-word sharing, and subchapter 1.6 uses it to rank sublayers.

**A word can also end up with zero experts.** The definition of $E_c$ makes $E_c = \varnothing$ perfectly possible: if no unit reaches $\tau$ for word $c$, the word contributes no rows at all. Such words are *silently absent* from every downstream table (they do not appear as zero-count rows), which shrinks sample sizes at strict thresholds without any explicit marker in the CSVs. The Results section quantifies where this actually happens.

### 1.2 Expert counts merged with concept metadata

**Mathematical formulation.** The first analysis is a per-item aggregation of the filtered expert table. For each item $c \in \mathcal{C}$, the total expert count is

$$n_c = \sum_{\ell=1}^{L} N_{c\ell} = |E_c|,$$

i.e. the number of expert rows that survive the AP threshold for that item, counted within the current scope, so the whole-model scope reports the item's total across the network and each sublayer scope reports only that projection type's share of it. Each count is then joined with the item's metadata, and the raw Wikipedia frequency $f_c$ is mapped to a logarithmic scale,

$$\tilde{f}_c = \log_{10} f_c, \qquad f_c > 0,$$

with items at $f_c \le 0$ (missing frequency) dropped from the merged table. The log transform compresses the heavy right tail of word-frequency distributions so that downstream linear correlation (module 4) operates on a roughly scale-invariant quantity rather than being dominated by a few extremely frequent words. The result is one row per item pairing the model-derived quantity $n_c$ with the human-sourced covariates $f_c$, $\tilde{f}_c$, and Human Typicality $t_c$. Words with $E_c = \varnothing$ (subchapter 1.1) have no expert rows and therefore no row here either, so the table's row count *is* the count of words that retained at least one expert.

**Generated data structures.** One CSV, `expert_counts_with_metadata.csv`, built by grouping the expert rows with `groupby("concept").size()` and merging on `concept` with `concept_metadata` (the metadata table, with its typicality field renamed to `human_typicality` at load time to distinguish it from the model-computed Cosine Typicality of module 7). Rows are sorted by `(abstraction_level, concept)`, so all level-1 items form one block followed by the level-2 block. This table is not plotted here. It is the input for the correlation (module 4) and typicality (module 7) analyses.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Concept identifier. |
| category | string | $k$ | Semantic category assigned to the concept. |
| abstraction_level | int | $a \in \{1,2\}$ | Abstraction level from the metadata. |
| frequency | float | $f_c$ | Raw frequency value from the metadata source. |
| log_frequency | float | $\tilde{f}_c$ | Base-10 logarithm of the frequency. |
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

### 1.3 Concept-by-layer allocation profiles (percentages and counts)

**Mathematical formulation.** The core object of the module is the percentage allocation matrix $P \in [0,100]^{|\mathcal{C}| \times L}$, built from the raw count crosstab $N$ over a `(abstraction_level, concept)` row MultiIndex:

$$P_{c\ell} = 100 \cdot \frac{N_{c\ell}}{\sum_{\ell'=1}^{L} N_{c\ell'}} = 100 \cdot \frac{N_{c\ell}}{n_c}, \qquad \sum_{\ell=1}^{L} P_{c\ell} = 100 .$$

Each row of $P$ is item $c$'s *layer profile*: the share of that item's experts sitting in each layer, expressed as a percentage. Because every row sums to 100 regardless of $n_c$, profiles of items with very different expert counts are directly comparable, since the normalization removes overall expert volume and keeps only *where* in the network the experts sit. A perfectly uniform profile would have $P_{c\ell} = 100/L$ in every layer, whereas concentration shows up as a few layers holding a much larger share. The count matrix $N$ is now kept alongside $P$ (rather than being discarded inside the crosstab), so every percentage can be traced back to the raw count that produced it.

**Percentages inherit the reliability of their denominator.** The normalization that makes profiles comparable also hides their very different statistical stability: a share computed from a large total is far more reliable than the same share computed from a small one. Moving a *single* expert between layers shifts $P_{c\ell}$ by $100/n_c$ percentage points, a negligible 0.05 pp for a concept with $n_c = 2{,}160$ experts but a massive 4.3 pp for one with $n_c = 23$ (both real AP=0.6 values in the GPT-2 Richie-HSJ run). Treating $N_{c\ell}$ as a binomial count, the standard error of an estimated share $\hat{p} = P_{c\ell}/100$ is

$$\mathrm{SE}(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n_c}},$$

so e.g. a 10% layer share is measured to about ±0.6 pp at $n_c = 2{,}160$ but only to about ±6 pp at $n_c = 23$, an uncertainty as large as the value itself. An apparent shift in a layer's percentage between two low-count words (or between AP thresholds, which change $n_c$ drastically) is therefore greatly *amplified* by the small denominator and can be pure sampling noise rather than a real reallocation. This is why the raw counts are now written next to every percentage (the `expert_count` column below, and `total_expert_count` in subchapter 1.4): every reported share carries its scale, and shares backed by small counts should be read with proportionally wide error bars. The problem is most acute at strict AP thresholds, where many words drop to single-digit expert counts (see Results).

**Generated data structures.** One CSV and one plot per item, under `per_concept/<concept>/`. These are written for the analysis sublayer only, and only in the whole-model scope, for the volume reason given in subchapter 1.7:

- `<concept>_data.csv` is the single row $P_{c\,\cdot}$ of the matrix paired with the matching row $N_{c\,\cdot}$ of the count matrix.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| layer_name | string | $\ell$ (label) | Layer label, in model order. |
| expert_allocation_pct | float | $P_{c\ell}$ | Percentage of that concept's experts located in this layer. |
| expert_count | int | $N_{c\ell}$ | Raw number of experts behind that percentage (the scale reference). |

Example (head of `AP_0.6/1_layer_expert_distribution/per_concept/accountant/accountant_data.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`, where the analysis is restricted to the 28 `mlp.gate_proj` layers):

| layer_name | expert_allocation_pct | expert_count |
|---|---|---|
| 5.L.0.mlp.gate_proj | 6.0421 | 198 |
| 12.L.1.mlp.gate_proj | 4.3943 | 144 |
| 19.L.2.mlp.gate_proj | 2.7769 | 91 |
| 26.L.3.mlp.gate_proj | 2.4107 | 79 |
| 33.L.4.mlp.gate_proj | 4.6689 | 153 |

- `<concept>_distribution.png`, one vertical **bar chart** per concept plotting `layer_name` ($\ell$, x-axis) versus `expert_allocation_pct` ($P_{c\ell}$, y-axis), colored by the concept's abstraction level $a$, titled with the concept name, abstraction level, and total expert count $n_c$.

### 1.4 Mean allocation profile per abstraction level

**Mathematical formulation.** To compare broad categories against specific concepts as groups, the per-item profiles are averaged within each abstraction level $a \in \{1, 2\}$:

$$\bar{P}^{(a)}_{\ell} = \frac{1}{|\mathcal{C}_a|} \sum_{c \in \mathcal{C}_a} P_{c\ell}.$$

This is an unweighted mean of percentages: each item contributes equally to its group profile, regardless of its expert count $n_c$, so a concept with 100 experts and one with 1,000 shape $\bar{P}^{(a)}$ identically. Note that this equal weighting interacts with the reliability caveat of subchapter 1.3, since low-count items inject their noisier profiles into the group mean with full weight, which is one more reason the raw counts are carried along. The two resulting vectors $\bar{P}^{(1)}, \bar{P}^{(2)} \in [0,100]^L$ each still sum to 100 across layers and represent the *typical* layer profile of a category label versus a specific concept. Two companion quantities are attached per (level, layer): the raw expert total

$$T^{(a)}_{\ell} = \sum_{c \in \mathcal{C}_a} N_{c\ell}$$

(the scale behind the mean percentage), and the within-level cumulative allocation in model-depth order,

$$C^{(a)}_{\ell} = \sum_{\ell' \le \ell} \bar{P}^{(a)}_{\ell'},$$

which rises from $\bar{P}^{(a)}_{1}$ to 100 across the depth axis and feeds the cumulative-mass plots of subchapter 1.5.

**Generated data structures.** One CSV, since the plain bar chart that used to accompany it was retired and the cumulative plots of subchapter 1.5 now carry the same bars plus the cumulative curves:

- `mean_expert_layer_distribution{suffix}.csv`, in long format, sorted into two blocks by `abstraction_level` with layers in model order inside each block (so each block reads as a depth profile and its `cumulative_pct` ends at 100). `{suffix}` is `_by_block` or `_by_layer` in the whole-model scope and empty in a sublayer scope, per subchapter 1.7.

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

### 1.5 Cumulative-mass plots (depth-ordered and Pareto-sorted)

**Mathematical formulation.** Two complementary visualizations answer "how concentrated is the allocation" directly from the quantities of subchapter 1.4. The *depth-ordered* view keeps the layers in model order and overlays the cumulative curve $C^{(a)}_{\ell}$ on the mean-allocation bars, so the curve rises steeply exactly where the distribution peaks and the depth at which the mass accumulates is visible: the marked milestones are the first layers at which the curve crosses 50% and 90%,

$$\ell^{(a)}_{50} = \min\{\ell : C^{(a)}_{\ell} \ge 50\}, \qquad \ell^{(a)}_{90} = \min\{\ell : C^{(a)}_{\ell} \ge 90\}.$$

The *Pareto-sorted* view instead orders the layers of each level by descending $\bar{P}^{(a)}_{\ell}$ and accumulates in that order, which answers "how many layers hold the bulk of the mass" independent of where they sit in the network: the milestones are the smallest $k$ such that the top-$k$ layers hold 50% (resp. 90%) of the total mass. The two views are deliberately different summaries, since a distribution can be very concentrated (small $k_{50}$) while reaching depth-ordered 50% late, if its dominant layers sit deep in the network.

**Generated data structures.** No new CSVs, since both plots are drawn from `mean_expert_layer_distribution*.csv`. Two plots per layer-axis variant (subchapter 1.7), so two in a sublayer scope and four in the whole-model scope:

- `mean_expert_layer_distribution_cumulative{suffix}.png`: depth-ordered bars ($\bar{P}^{(a)}_{\ell}$, both abstraction levels side by side, left y-axis) with each level's cumulative curve $C^{(a)}_{\ell}$ on a secondary 0–100% axis. The 50% and 90% crossings are flagged with diamond markers, dotted projection lines onto both axes, and legend labels naming the crossing layers.
- `mean_expert_layer_distribution_cumulative_sorted{suffix}.png`: the Pareto view, one panel per abstraction level (the sort order differs between levels, so they cannot share an x-axis). Bars are the sorted $\bar{P}^{(a)}_{\ell}$, the line is the cumulative mass recomputed in sorted order, diamonds mark the top-$k_{50}$/top-$k_{90}$ milestones and the panel title states them (e.g. "top 8 layers hold 50% of the mass").

`{suffix}` is `_by_block` or `_by_layer` in the whole-model scope and empty in a sublayer scope. Within a sublayer scope each concept's percentages are renormalized to sum to 100 *within* that sublayer's layers, so the profiles describe where inside the projection type the experts sit, not how much of the model's total mass the type holds, which is `share_of_all_experts_pct` in subchapter 1.6.

### 1.6 Sublayer informativeness (category alignment and depth smoothness)

**Mathematical formulation.** Transformer layers come in a small number of *sublayer types* (projections), parsed from the layer label `{idx}.L.{block}.{sublayer}`, with 4 types for GPT-2 (`attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`) and 7 for Qwen3 (`self_attn.q/k/v/o_proj`, `mlp.gate/up/down_proj`). This analysis ranks the types by how much category structure their expert sets carry, which is the empirical justification for restricting the downstream modules to one sublayer. For each sublayer $s$, restrict every item's expert set to that sublayer's layers, $E_c^{(s)}$, and compute for every unordered pair of level-2 concepts the Jaccard similarity of the restricted sets,

$$J^{(s)}_{cd} = \frac{|E^{(s)}_c \cap E^{(s)}_d|}{|E^{(s)}_c \cup E^{(s)}_d|},$$

together with the indicator of whether $c$ and $d$ belong to the same category. Three alignment statistics follow (level-2 concepts only, since category labels have no same-category peers, so level-1 rows carry NaN here):

- **`roc_auc` (primary)**: the probability that a randomly drawn same-category pair is more similar than a randomly drawn different-category pair. It is computed rank-based via the Mann–Whitney identity, $\mathrm{AUC} = \big(\sum_{\text{same}} R_i - \tfrac{n_1(n_1+1)}{2}\big) / (n_1 n_0)$ with $R_i$ the ranks of the pair similarities (average ranks for ties), which makes it immune to the heavy same/different pair imbalance and to the skewed shape of Jaccard values. 0.5 = no category signal, 1.0 = every within-category pair beats every across-category pair.
- **`category_alignment_r` (companion)**: the same same-category-vs-different-category comparison as `roc_auc`, but as a plain Pearson correlation instead of a rank statistic. Writing $\mathrm{same}_{cd} \in \{0,1\}$ for the indicator that $c$ and $d$ share a category,

$$r = \operatorname{corr}\big(\mathrm{same}_{cd},\, J^{(s)}_{cd}\big) \in [-1, 1],$$

taken over every concept pair. This is a point-biserial correlation, the name for a Pearson correlation with one binary input, so positive $r$ means same-category pairs run more similar on average, and $r^2$ is the share of the pair-similarity variance explained by category membership alone, the same variance-explained reading module 4 uses for its regression panels. Because Pearson correlation assumes a roughly linear relationship, `category_alignment_r` can diverge from the rank-based `roc_auc` when the similarity distribution is skewed or dominated by a few extreme pairs, for example a handful of very high same-category similarities can inflate $r$ well beyond what the AUC's pairwise ranking shows, which is why the two are reported side by side rather than one replacing the other.
- **`mantel_p`**: a permutation p-value for the AUC. Pairs are not independent observations (each concept participates in $|\mathcal{C}_2|-1$ pairs), so the null distribution is built by shuffling category labels across *concepts* (never across pairs), rebuilding the indicator, and recomputing the AUC, 9,999 times. $p = (1 + \#\{\mathrm{AUC}_{\text{perm}} \ge \mathrm{AUC}\}) / (1 + 9{,}999)$, so the smallest reportable value is $10^{-4}$. Degenerate sublayers whose pair similarities are all identical short-circuit to AUC = 0.5, $r$ = NaN, $p$ = 1.

Each (level, sublayer) row also carries a *depth-smoothness* descriptor: the mean, over that level's words, of Geary's C of the word's expert counts across the sublayer's layers in depth order,

$$C(x) = \frac{\sum_{i} (x_{i+1} - x_i)^2}{2 \sum_i (x_i - \bar{x})^2},$$

the adjacent-neighbor spatial autocorrelation, where $C \approx 1$ means no depth structure, $C < 1$ means smooth or clumped profiles with neighboring layers alike, and $C > 1$ means jagged, alternating profiles. It is scale-invariant, so counts and shares give the same value, as used per word in module 2. Computing it within one sublayer type uses one layer per block, so it is free of the mechanical high/low alternation between neighboring projection types that whole-model depth order would introduce. Finally, each row records that level's expert volume in the sublayer, as a count and as a share of *all* experts (shares sum to 100 over the whole table).

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

### 1.7 Analysis scopes and the two layer axes

This subchapter defines the machinery every module shares. The other module docs refer back to it rather than restating it.

**Scopes.** Each module computes its analysis once on the entire model and once per sublayer type $s$, that is on the restricted expert sets $E_c^{(s)}$. The whole-model scope writes into the module's own folder, so the big picture is what a reader sees first, and each sublayer scope writes into `<module>/sublayers/<rank>_<sublayer>/`. The rank is the sublayer's position by expert count, computed once at the most lenient AP threshold of the sweep and cached in `results/<output_subdir>/sublayer_rank.csv`. Freezing it there rather than recomputing per threshold means a given prefix names the same sublayer in every `AP_*` folder, so paths stay comparable across the sweep, which matters because stricter thresholds thin the sublayers unevenly and would otherwise reshuffle the prefixes. A sublayer left with no experts at a strict threshold is skipped with a warning instead of producing an empty folder. Within a sublayer scope the retained `layer_idx` values are non-contiguous by design (5, 12, 19, and so on for `mlp.gate_proj`) and layers keep their absolute whole-model indices, so depth statements stay comparable across scopes.

**The two layer axes.** Analyses divide into two kinds, and only one of them is affected. *Set-based* analyses (module 3, module 5, module 7's global prototype, every Jaccard computation, and subchapter 1.6 here) treat $E_c$ as an unordered set of $(\ell, u)$ pairs, so layer order never enters and the whole-model scope needs no special handling. *Order-based* analyses (subchapters 1.4 and 1.5, module 2's descriptors, module 6, module 7's per-layer prototypes) read $\ell$ as depth, and there the whole-model layer index is not a depth coordinate. Layers are numbered

$$\ell = b \cdot S + s + 1,$$

with $b$ the block, $S$ the number of sublayers per block, and $s$ the sublayer's position inside the block, so on Qwen3 indices 1 to 7 all belong to block 0. Adjacent indices are therefore different projection types of the *same* block, not different depths. Geary's C, which differences adjacent indices, then measures the alternation between projection types rather than depth smoothness, and the peak layer $\arg\max_\ell \bar{P}_\ell$ returns whichever sublayer is densest for nearly every word. Measured on GPT-2 at AP 0.6, the flat whole-model axis gives a mean per-concept Geary's C of 1.036, which is exactly the value meaning *no spatial structure at all*, and 68% of concepts peak in an `mlp.c_fc` layer.

The whole-model scope therefore produces every order-based output twice:

- **`_by_block`** (canonical): the sublayers within each block are summed into one bin, $N_{cb} = \sum_{s} N_{c, bS+s+1}$, giving $B$ bins (12 for GPT-2, 28 for Qwen3) that carry the model's full expert mass on a genuine depth axis. The same GPT-2 run gives mean Geary's C 0.361 here. This is the variant downstream modules consume.
- **`_by_layer`**: the flat axis at full resolution ($L$ = 48 or 196 bins), kept because it is the only view that separates projection types, with the caveat above attached to any depth reading of it.

A sublayer scope holds exactly one layer per block, so block aggregation would be an identity relabel there and only one variant is written, with no suffix.

**Whole-model-only outputs.** Two things are not replicated per scope. `sublayer_informativeness.csv` (subchapter 1.6) is a cross-sublayer table by construction, so it has one natural home at the module's top level. The per-concept files of subchapter 1.3 stay restricted to the configured analysis sublayer, `mlp.c_fc` for GPT-2 and `mlp.gate_proj` for Qwen3, in both cases the winner of the level-2 AUC ranking for that architecture: one plot per concept per sublayer per threshold would be 204 × 7 × 5 figures on Qwen3, and this is already the slowest step in the pipeline.

**Cross-scope comparison.** Each module writes `sublayer_comparison.csv` and `sublayer_comparison.png` at its top level, one row per scope (whole model first) with that module's headline metrics, `n_experts` on every row so each percentage or correlation is read against the mass it rests on. These are the readable surface of the sweep, and the per-scope folders are the evidence behind them. Module 1's row carries the mean expert count per word at each abstraction level, alongside the deeper per-sublayer ranking in `sublayer_informativeness.csv`. Because the block axis puts one bin per block and a sublayer scope holds exactly one layer per block, all scopes in a comparison table share the same bin count, so bin-count-sensitive quantities such as Shannon entropy are on a common scale across rows.

**How to read these tables, and the one thing they do not control for.** Every scope-level metric here is confounded with expert density, because sublayers differ enormously in how many experts they hold (on GPT-2 at AP 0.6, `mlp.c_fc` has 40,564 expert rows against `mlp.c_proj`'s 5,271, a factor of 7.7). Measured on that run, the cross-sublayer rankings produced by module 1's `roc_auc`, module 5's category contrast, and module 8's raw $\rho$ are *all* Spearman +1.00 with each other **and with the raw expert count**, so nothing in those four rankings distinguishes "this projection type carries more category structure" from "this projection type simply holds more experts". Module 8 is the only place that controls for it, by recomputing $\rho$ on a count-matched top-$k$ expert set, and doing so **re-ranks the sublayers**: `attn.c_proj` goes from third on raw $\rho$ (0.313) to first on the count-matched value (0.275), ahead of `mlp.c_fc` (0.486 raw, 0.189 matched) and `attn.c_attn` (0.392 raw, 0.190 matched). Read the comparison tables of modules 1 to 7 as descriptions of what each scope's expert set actually looks like, which is what they are, and read `8_embedding_rsa/sublayer_comparison.csv` for the density-controlled ranking. Extending count-matching to the other modules is an open item.

The same confound explains the sparse-scope readings that would otherwise look like findings. `attn.c_proj` and `mlp.c_proj` carry roughly 31 and 26 experts per word, so their layer distributions are sparse, which mechanically depresses Shannon entropy (module 2 reports 1.94 and 1.33 bits for concepts, against about 3.1 in the dense scopes) and mechanically inflates Jensen-Shannon divergence (module 6 reports 0.36 and 0.33, against 0.09 in `mlp.c_fc`). Module 4's `shannon_entropy_vs_expert_count` panel measures this relationship directly, at $r \approx 0.6$ to $0.7$ in every scope, and subchapter 1.3's small-denominator caveat is the same point at the level of a single layer share.

## Results

At AP=0.6 in the GPT-2 150-concept reference run (`research_plots_150_revised_executor_again`, whole-model expert sets), layer allocation is genuinely concentrated, not uniform. With 48 layers, a uniform allocation would put 10.4% of experts in any 5 layers. Instead, the top 5 layers hold **44.0%** of all expert mass for broad categories (abstraction level 1) and **35.4%** for specific concepts (level 2). The single busiest layer for both levels is `3.L.0.mlp.c_fc` (an early MLP layer), at 13.5% for categories and 11.3% for concepts. This is visible directly in the mean-distribution bars of the cumulative plot: a handful of sharp early-to-mid-layer peaks stand out clearly against a low, noisy baseline across the rest of the network.

### Across AP thresholds

The concentration effect is not a one-threshold artifact. It holds at every AP from 0.5 to 0.9, and it gets dramatically stronger as the threshold tightens:

| AP | Top-5-layer % (categories, level 1) | Top-5-layer % (concepts, level 2) | Layers with zero experts |
|---|---|---|---|
| 0.5 | 36.4% | 31.1% | 0 |
| 0.6 | 44.0% | 35.4% | 0 |
| 0.7 | 50.0% | 40.6% | 0 |
| 0.8 | 66.6% | 45.4% | 3 |
| 0.9 | 88.6% | 54.2% | 5 |

At AP=0.9, the top 5 layers hold **88.6%** of all category-level expert mass, so almost everything is concentrated in a handful of layers. The level-1 (category) vs. level-2 (concept) gap also widens at stricter thresholds (5.3 points at AP=0.5 vs. 34.4 points at AP=0.9), consistent with module 2's later finding that categories concentrate faster than concepts as the AP bar rises.

A second, structural finding only visible by comparing thresholds: at AP=0.8 and AP=0.9, **3 and 5 of the 48 layers respectively have zero retained experts across every single concept** (i.e. no expert anywhere in the dataset survives that strict an AP filter in those layers).

This module only reports descriptive percentages, it does not run a significance test on the level-1-vs-level-2 gap at any threshold. Module 2's Shannon entropy analysis is what actually tests concept-vs-category concentration statistically, while this module's contribution is establishing that *some* layers clearly dominate (more so as AP increases), which is the precondition for that later test to be meaningful.

### Zero-expert words and low-count words across thresholds

Not only layers but *words* can end up empty (subchapter 1.1). In the 150-concept reference run (the same run as the tables above, 164 metadata words = 147 concepts + 17 category labels):

| AP | Words with ≥1 expert (of 164) | Zero-expert words | Words with <30 experts | Smallest count |
|---|---|---|---|---|
| 0.5 | 164 | 0 | 0 | 162 |
| 0.6 | 164 | 0 | 1 | 25 (`clothing`) |
| 0.7 | 164 | 0 | 13 | 2 (`clothing`) |
| 0.8 | 161 | 3 | 94 | 1 |
| 0.9 | 119 | 45 | 101 | 1 |

At AP ≤ 0.7 there are **no zero-expert cases**: every word keeps at least one expert. The first casualties appear at AP=0.8 (`clothing`, `hopscotch`, `toy`, noting that `clothing` and `toy` are category *labels*, so a whole category loses its Definition-A representation), and at AP=0.9 the phenomenon is widespread: **45 of 164 words (27%) retain zero experts**, including the labels `animal` and `clothing`. This is exactly what silently shrinks the downstream sample sizes reported by the other modules (module 3's pair count falling 147 → 128 → 59 across AP 0.7–0.9, module 6's category count falling 17 → 15 → 9). The Richie-HSJ GPT-2 run behaves the same way: 0 zero-expert words of 204 at AP 0.5–0.7, then 12 at AP=0.8 and 94 at AP=0.9.

The middle column matters just as much as the zero column. Well before a word reaches zero, it passes through the low-count regime where subchapter 1.3's reliability caveat bites: at AP=0.8, 94 of the 161 surviving words have fewer than 30 experts, so a single expert is 3–100 percentage points of the layer profile, and per-word percentage profiles (and any shift in them between thresholds) are dominated by sampling noise. Group-level statements remain meaningful at strict thresholds because they pool words, but per-word layer percentages at AP ≥ 0.8 should not be over-interpreted.

### New analyses at AP=0.6 (Richie-HSJ runs)

The sublayer ranking cleanly separates the projection types, in the same order for both architectures' MLP-vs-attention contrast. For Qwen3 (table in subchapter 1.6): `mlp.gate_proj` and `mlp.up_proj` are nearly tied at the top (AUC 0.927 and 0.923) and jointly hold 70% of all experts, attention projections rank middle (o_proj 0.879 down to q_proj 0.698), and `mlp.down_proj` is by far the least category-aligned (AUC 0.626, r 0.168). Every sublayer's `mantel_p` sits at the $10^{-4}$ permutation floor, so *all* sublayers carry statistically real category signal, and the ranking is about how much, not whether. `mlp.gate_proj` also dominates the level-1 side, holding 2,648 of the 3,937 category-label experts (67%). For GPT-2 on the same dataset, the winner is `mlp.c_fc` (AUC 0.872, against 0.786 for `attn.c_attn`, 0.684 for `attn.c_proj`, 0.570 for `mlp.c_proj`). These rankings are what the configured `sublayer_filter` values (`mlp.gate_proj`, `mlp.c_fc`) implement. The Geary's C column adds a shape observation: the category-aligned MLP input projections have smooth depth profiles (mean C ≈ 0.24–0.32), while `self_attn.o_proj` (C ≈ 0.95) is essentially depth-unstructured.

The cumulative plots quantify concentration for the Qwen3 run at AP=0.6: over the whole 196-layer model, the top 12 layers hold 50% of the level-1 mass (top 20 for level 2), yet the depth-ordered cumulative curves cross 50% only around layer 142–144, so the allocation is simultaneously very concentrated *and* skewed toward the deeper half of the network. Within the `mlp.gate_proj` focus sublayer, 7 of 28 layers hold 50% of the level-1 mass (8 of 28 for level 2), and both levels need 17 of 28 layers to reach 90%.
