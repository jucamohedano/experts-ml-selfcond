# Module 1: Layer-wise Expert Distribution

## Research question

Are expert allocations concentrated in a subset of model layers rather than spread evenly, and does that concentration pattern differ between specific concepts (abstraction level 2) and broad categories (abstraction level 1)?

## Method

This module analyzes how expert allocations are distributed across model layers. It loads expert rows from the response result folders, keeps only rows whose AP score meets the selected threshold, and merges them with concept metadata by concept ID. After standardizing the layer naming scheme so each row has a numeric layer index and a human-readable layer label, it builds a concept-by-layer percentage matrix with `pd.crosstab(..., normalize='index') * 100`, so each concept contributes a layer profile that sums to 100%. Those profiles are then averaged within abstraction level for the global summary, and saved individually per concept for the per-concept breakdown.

## Outputs

### CSV tables

#### 1. expert_counts_with_metadata.csv

**Input:** the raw expert-allocation rows (one row per expert unit retained for a concept after AP filtering), grouped by `concept` with `groupby("concept").size()`, then merged on `concept` with `concept_metadata` (the table loaded from [assets/metadata_150.json](../assets/metadata_150.json), with its `typicality` field renamed to `human_typicality` at load time to distinguish it from the model-computed cosine typicality used in module 7).

This table combines expert counts with concept metadata and is used later by the correlation and typicality analyses.

| Column | Type | Description |
|--------|------|-------------|
| concept | string | Concept identifier. |
| category | string | Semantic category assigned to the concept. |
| abstraction_level | int | Abstraction level from the metadata. |
| frequency | float | Raw frequency value from the metadata source. |
| log_frequency | float | Base-10 logarithm of the frequency. |
| human_typicality | float | Human-judged typicality score from the metadata. |
| expert_count | int | Number of expert rows retained for the concept. |

Example (head of `AP_0.6/1_layer_expert_distribution/expert_counts_with_metadata.csv` in `research_plots_150_revised_executor`):

| concept | category | abstraction_level | frequency | log_frequency | human_typicality | expert_count |
|---|---|---|---|---|---|---|
| airplane | vehicle | 2 | 17652 | 4.2468 | 0.58 | 523 |
| alligator | animal | 2 | 5210 | 3.7168 | 0.459 | 960 |
| animal | | 1 | 139522 | 5.1446 | | 231 |
| anklet | jewelry | 2 | 273 | 2.4362 | 0.378 | 397 |
| anvil | tool | 2 | 3127 | 3.4951 | 0.347 | 831 |

#### 2. mean_expert_layer_distribution.csv

**Input:** the concept-by-layer percentage matrix built with `pd.crosstab(index=[abstraction_level, concept], columns=layer_name, normalize='index') * 100`, averaged across concepts within each `abstraction_level` and melted back to long format.

This CSV is saved alongside the global plot and contains the averaged percentage allocation by abstraction level and layer.

| Column | Type | Description |
|--------|------|-------------|
| abstraction_level | int | Abstraction group being summarized. |
| layer_name | string | Layer label used for plotting. |
| mean_expert_allocation_pct | float | Mean percentage of experts assigned to that layer within the abstraction group. |

Example (head of `AP_0.6/1_layer_expert_distribution/mean_expert_layer_distribution.csv` in `research_plots_150_revised_executor`):

| abstraction_level | layer_name | mean_expert_allocation_pct |
|---|---|---|
| 1 | 1.L.0.attn.c_attn | 7.8471 |
| 2 | 1.L.0.attn.c_attn | 6.6557 |
| 1 | 2.L.0.attn.c_proj | 1.5484 |
| 2 | 2.L.0.attn.c_proj | 1.6560 |
| 1 | 3.L.0.mlp.c_fc | 13.4527 |

#### 3. Per-concept CSVs in the per_concept folder

**Input:** one row of the concept-by-layer percentage matrix described above, isolated per `(abstraction_level, concept)` pair and reset to a two-column table (`<concept>_data.csv`, e.g. `per_concept/airplane/airplane_data.csv`).

Each concept gets its own CSV with the layer-wise percentage profile and a corresponding plot image.

| Column | Type | Description |
|--------|------|-------------|
| layer_name | string | Layer label, in model order. |
| expert_allocation_pct | float | Percentage of that concept's experts located in this layer. |

Example (head of `AP_0.6/1_layer_expert_distribution/per_concept/airplane/airplane_data.csv` in `research_plots_150_revised_executor`):

| layer_name | expert_allocation_pct |
|---|---|
| 1.L.0.attn.c_attn | 5.3537 |
| 2.L.0.attn.c_proj | 4.0153 |
| 3.L.0.mlp.c_fc | 6.3098 |
| 4.L.0.mlp.c_proj | 2.2945 |
| 5.L.1.attn.c_attn | 5.3537 |

### Plots

- `mean_expert_layer_distribution.png`, a vertical **bar chart** of `layer_name` (x-axis) versus `mean_expert_allocation_pct` (y-axis), with bars grouped/colored by `abstraction_level` (hue), titled "Mean Expert Distribution across model layers".
- `<concept>_distribution.png`, one vertical **bar chart** per concept of `layer_name` (x-axis) versus `expert_allocation_pct` (y-axis), colored by the concept's `abstraction_level`, titled with the concept name, abstraction level, and total expert count.

## Results

At AP=0.6, layer allocation is genuinely concentrated, not uniform. With 48 layers, a uniform allocation would put 10.4% of experts in any 5 layers; instead, the top 5 layers hold **44.0%** of all expert mass for broad categories (abstraction level 1) and **35.4%** for specific concepts (level 2). The single busiest layer for both levels is `3.L.0.mlp.c_fc` (an early MLP layer), at 13.5% for categories and 11.3% for concepts. This is visible directly in `mean_expert_layer_distribution.png`: a handful of sharp early-to-mid-layer peaks stand out clearly against a low, noisy baseline across the rest of the network.

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

This module only reports descriptive percentages, it does not run a significance test on the level-1-vs-level-2 gap at any threshold. Module 2's Shannon entropy analysis is what actually tests concept-vs-category concentration statistically; this module's contribution is establishing that *some* layers clearly dominate (more so as AP increases), which is the precondition for that later test to be meaningful.
