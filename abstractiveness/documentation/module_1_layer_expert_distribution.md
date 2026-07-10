# Module 1: Layer-wise Expert Distribution

## Research question

Are expert allocations concentrated in a subset of model layers rather than spread evenly, and does that concentration pattern differ between specific concepts (abstraction level 2) and broad categories (abstraction level 1)?

## Analysis

Notation. Let $\mathcal{C}$ be the set of analyzed items (concepts and category labels), partitioned by abstraction level into $\mathcal{C}_1$ (broad categories) and $\mathcal{C}_2$ (specific concepts). Let $L = 48$ be the number of model layers, $E_c$ the set of expert units retained for item $c$ after AP filtering, $n_c = |E_c|$ its expert count, and $N_{c\ell}$ the number of expert rows of item $c$ in layer $\ell$.

### 1.1 Expert counts merged with concept metadata

**Mathematical formulation.** The first analysis is a per-item aggregation of the filtered expert table. For each item $c \in \mathcal{C}$, the total expert count is

$$n_c = \sum_{\ell=1}^{L} N_{c\ell} = |E_c|,$$

i.e. the number of expert rows that survive the AP threshold for that item. Each count is then joined with the item's metadata, and the raw Wikipedia frequency $f_c$ is mapped to a logarithmic scale,

$$\tilde{f}_c = \log_{10} f_c, \qquad f_c > 0,$$

with items at $f_c \le 0$ (missing frequency) dropped from the merged table. The log transform compresses the heavy right tail of word-frequency distributions so that downstream linear correlation (module 4) operates on a roughly scale-invariant quantity rather than being dominated by a few extremely frequent words. The result is one row per item pairing the model-derived quantity $n_c$ with the human-sourced covariates $f_c$, $\tilde{f}_c$, and Human Typicality $t_c$.

**Generated data structures.** One CSV, `expert_counts_with_metadata.csv`, built by grouping the expert rows with `groupby("concept").size()` and merging on `concept` with `concept_metadata` (the table loaded from [assets/metadata_150.json](../assets/metadata_150.json), with its `typicality` field renamed to `human_typicality` at load time to distinguish it from the model-computed Cosine Typicality of module 7). This table is not plotted here. It is the input for the correlation (module 4) and typicality (module 7) analyses.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Concept identifier. |
| category | string | $k$ | Semantic category assigned to the concept. |
| abstraction_level | int | $a \in \{1,2\}$ | Abstraction level from the metadata. |
| frequency | float | $f_c$ | Raw frequency value from the metadata source. |
| log_frequency | float | $\tilde{f}_c$ | Base-10 logarithm of the frequency. |
| human_typicality | float | $t_c$ | Human-judged typicality score from the metadata. |
| expert_count | int | $n_c$ | Number of expert rows retained for the concept. |

Example (head of `AP_0.6/1_layer_expert_distribution/expert_counts_with_metadata.csv` in `research_plots_150_revised_executor_again`):

| concept | category | abstraction_level | frequency | log_frequency | human_typicality | expert_count |
|---|---|---|---|---|---|---|
| airplane | vehicle | 2 | 17652 | 4.2468 | 0.58 | 523 |
| alligator | animal | 2 | 5210 | 3.7168 | 0.459 | 960 |
| animal | | 1 | 139522 | 5.1446 | | 231 |
| anklet | jewelry | 2 | 273 | 2.4362 | 0.378 | 397 |
| anvil | tool | 2 | 3127 | 3.4951 | 0.347 | 831 |

### 1.2 Concept-by-layer percentage allocation profiles

**Mathematical formulation.** The core object of the module is the percentage allocation matrix $P \in [0,100]^{|\mathcal{C}| \times L}$, built with `pd.crosstab(..., normalize='index') * 100` over a `(abstraction_level, concept)` row MultiIndex:

$$P_{c\ell} = 100 \cdot \frac{N_{c\ell}}{\sum_{\ell'=1}^{L} N_{c\ell'}} = 100 \cdot \frac{N_{c\ell}}{n_c}, \qquad \sum_{\ell=1}^{L} P_{c\ell} = 100 .$$

Each row of $P$ is item $c$'s *layer profile*: the share of that item's experts sitting in each layer, expressed as a percentage. Because every row sums to 100 regardless of $n_c$, profiles of items with very different expert counts are directly comparable, since the normalization removes overall expert volume and keeps only *where* in the network the experts sit. A perfectly uniform profile would have $P_{c\ell} = 100/L \approx 2.08\%$ in every layer, whereas concentration shows up as a few layers holding a much larger share.

**Generated data structures.** One CSV and one plot per item, under `per_concept/<concept>/`:

- `<concept>_data.csv` is the single row $P_{c\,\cdot}$ of the matrix, reset to a two-column table.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| layer_name | string | $\ell$ (label) | Layer label, in model order. |
| expert_allocation_pct | float | $P_{c\ell}$ | Percentage of that concept's experts located in this layer. |

Example (head of `AP_0.6/1_layer_expert_distribution/per_concept/airplane/airplane_data.csv` in `research_plots_150_revised_executor_again`):

| layer_name | expert_allocation_pct |
|---|---|
| 1.L.0.attn.c_attn | 5.3537 |
| 2.L.0.attn.c_proj | 4.0153 |
| 3.L.0.mlp.c_fc | 6.3098 |
| 4.L.0.mlp.c_proj | 2.2945 |
| 5.L.1.attn.c_attn | 5.3537 |

- `<concept>_distribution.png`, one vertical **bar chart** per concept plotting `layer_name` ($\ell$, x-axis) versus `expert_allocation_pct` ($P_{c\ell}$, y-axis), colored by the concept's abstraction level $a$, titled with the concept name, abstraction level, and total expert count $n_c$.

### 1.3 Mean allocation profile per abstraction level

**Mathematical formulation.** To compare broad categories against specific concepts as groups, the per-item profiles are averaged within each abstraction level $a \in \{1, 2\}$:

$$\bar{P}^{(a)}_{\ell} = \frac{1}{|\mathcal{C}_a|} \sum_{c \in \mathcal{C}_a} P_{c\ell}.$$

This is an unweighted mean of percentages: each item contributes equally to its group profile, regardless of its expert count $n_c$, so a concept with 100 experts and one with 1,000 shape $\bar{P}^{(a)}$ identically. The two resulting vectors $\bar{P}^{(1)}, \bar{P}^{(2)} \in [0,100]^L$ each still sum to 100 across layers and represent the *typical* layer profile of a category label versus a specific concept. Comparing them layer by layer answers the second half of the research question: whether concentration lives in different places, or at different strengths, for the two abstraction levels.

**Generated data structures.** One CSV and one plot:

- `mean_expert_layer_distribution.csv` is the pair of group profiles melted to long format.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| abstraction_level | int | $a$ | Abstraction group being summarized. |
| layer_name | string | $\ell$ (label) | Layer label used for plotting. |
| mean_expert_allocation_pct | float | $\bar{P}^{(a)}_{\ell}$ | Mean percentage of experts assigned to that layer within the abstraction group. |

Example (head of `AP_0.6/1_layer_expert_distribution/mean_expert_layer_distribution.csv` in `research_plots_150_revised_executor_again`):

| abstraction_level | layer_name | mean_expert_allocation_pct |
|---|---|---|
| 1 | 1.L.0.attn.c_attn | 7.8471 |
| 2 | 1.L.0.attn.c_attn | 6.6557 |
| 1 | 2.L.0.attn.c_proj | 1.5484 |
| 2 | 2.L.0.attn.c_proj | 1.6560 |
| 1 | 3.L.0.mlp.c_fc | 13.4527 |

- `mean_expert_layer_distribution.png`, a vertical **bar chart** of `layer_name` ($\ell$, x-axis) versus `mean_expert_allocation_pct` ($\bar{P}^{(a)}_{\ell}$, y-axis), with bars grouped/colored by `abstraction_level` ($a$, hue), titled "Mean Expert Distribution across model layers".

## Results

At AP=0.6, layer allocation is genuinely concentrated, not uniform. With 48 layers, a uniform allocation would put 10.4% of experts in any 5 layers. Instead, the top 5 layers hold **44.0%** of all expert mass for broad categories (abstraction level 1) and **35.4%** for specific concepts (level 2). The single busiest layer for both levels is `3.L.0.mlp.c_fc` (an early MLP layer), at 13.5% for categories and 11.3% for concepts. This is visible directly in `mean_expert_layer_distribution.png`: a handful of sharp early-to-mid-layer peaks stand out clearly against a low, noisy baseline across the rest of the network.

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
