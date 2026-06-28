# Module 7: Cosine Typicality vs. Human Typicality

## Research question

Does the expert structure of a category recover human judgments of typicality, with concepts that humans rate as more typical of a category also more similar (by cosine similarity) to that category's expert-allocation prototype?

## Method

The module builds prototype vectors from the expert allocations and measures how similar each concept is to the category-level prototype using cosine similarity. This model-derived score is called **Cosine Typicality**, to distinguish it from the metadata-sourced **Human Typicality** score it is compared against. Two levels of prototype are used: a global prototype averaged across all layers, and a per-layer prototype for each depth. For each concept, the concept's allocation vector is compared to the relevant prototype vector with `sklearn.metrics.pairwise.cosine_similarity`.

## Outputs

### CSV tables

#### global_prototype_typicality.csv

**Input:** for each category with at least 2 valid members, a binary `concept x (layer_name_unit)` pivot table across all layers; the category's prototype is the column-wise mean of that pivot (the centroid), and each member concept is compared to the centroid with `sklearn.metrics.pairwise.cosine_similarity`.

Contains one row per concept-category pair.

| Column | Type | Description |
|--------|------|-------------|
| category | string | Category label. |
| concept | string | Concept identifier. |
| human_typicality | float | Human-judged typicality score for the concept, read from the metadata. |
| global_cosine_typicality | float | Cosine Typicality: cosine similarity between the concept vector and the global prototype vector for the category. |

Example (head of `AP_0.6/7_typicality_analysis/global_prototype_typicality.csv` in `research_plots_150_revised_executor`):

| category | concept | human_typicality | global_cosine_typicality |
|---|---|---|---|
| animal | alligator | 0.459 | 0.5333 |
| animal | frog | 0.575 | 0.4424 |
| animal | goldfish | 0.506 | 0.3512 |
| animal | iguana | 0.377 | 0.5413 |
| animal | leech | 0.065 | 0.4161 |

#### layer_prototype_typicality.csv

**Input:** the same per-category member set as above, but with one binary `concept x unit` pivot table per `layer_idx`; the per-layer prototype is the column-wise mean within that layer, and each member concept is compared to it with cosine similarity.

Contains one row per concept-category-layer combination.

| Column | Type | Description |
|--------|------|-------------|
| category | string | Category label. |
| concept | string | Concept identifier. |
| layer_idx | int | Numeric layer index. |
| layer_name | string | Formatted layer label. |
| human_typicality | float | Human-judged typicality score for the concept, read from the metadata. |
| layer_cosine_typicality | float | Cosine Typicality: cosine similarity between the concept vector and the prototype vector at that layer. |

Example (head of `AP_0.6/7_typicality_analysis/layer_prototype_typicality.csv` in `research_plots_150_revised_executor`):

| category | concept | layer_idx | layer_name | human_typicality | layer_cosine_typicality |
|---|---|---|---|---|---|
| animal | alligator | 1 | 1.L.0.attn.c_attn | 0.459 | 0.2577 |
| animal | frog | 1 | 1.L.0.attn.c_attn | 0.575 | 0.3266 |
| animal | goldfish | 1 | 1.L.0.attn.c_attn | 0.506 | 0.4092 |
| animal | iguana | 1 | 1.L.0.attn.c_attn | 0.377 | 0.5324 |
| animal | leech | 1 | 1.L.0.attn.c_attn | 0.065 | 0.3314 |

#### `<category>_most_typical_per_layer.csv` (per-category folder, e.g. `7_typicality_analysis/animal/animal_most_typical_per_layer.csv`)

**Input:** `layer_prototype_typicality.csv` filtered to the category, with the row of maximum `layer_cosine_typicality` selected for each `layer_idx` (`groupby('layer_idx')['layer_cosine_typicality'].idxmax()`), then renamed and sorted by layer.

Contains one row per model layer, identifying which member concept is most "typical" (by Cosine Typicality) of the category prototype at that depth.

| Column | Type | Description |
|--------|------|-------------|
| layer_idx | int | Numeric layer index. |
| layer_name | string | Formatted layer label. |
| most_typical_model_concept | string | Concept with the highest Cosine Typicality to the layer prototype. |
| cosine_similarity_score | float | Cosine Typicality of that concept to the layer prototype. |
| hardcoded_human_score | float | Human Typicality score of that same concept. |

Example (head of `AP_0.6/7_typicality_analysis/animal/animal_most_typical_per_layer.csv` in `research_plots_150_revised_executor`):

| layer_idx | layer_name | most_typical_model_concept | cosine_similarity_score | hardcoded_human_score |
|---|---|---|---|---|
| 1 | 1.L.0.attn.c_attn | iguana | 0.5324 | 0.377 |
| 2 | 2.L.0.attn.c_proj | iguana | 0.5687 | 0.377 |
| 3 | 3.L.0.mlp.c_fc | iguana | 0.5706 | 0.377 |
| 4 | 4.L.0.mlp.c_proj | iguana | 0.6686 | 0.377 |
| 5 | 5.L.1.attn.c_attn | iguana | 0.5887 | 0.377 |

### Plots

- `<category>_typicality_comparison_bar.png`, a vertical **bar chart** (grouped/hue) of `Score` (y-axis) versus `concept` (x-axis, ordered by descending human typicality), with bars colored by `Metric` ("Human Typicality" vs. "Cosine Typicality"), titled "'{category}' Human Typicality vs. Cosine Typicality".
- `<category>_typicality_correlation_scatter.png`, a **scatter plot with a linear regression line**, x: `human_typicality` ("Human Typicality"), y: `global_cosine_typicality` ("Cosine Typicality (Global Prototype)"), one point per concept (annotated with the concept name) and the Pearson $r$/$p$ in the title.
- `<category>_most_typical_evolution.png`, a **lollipop/stem plot**: x-axis is the model layer (`layer_name`), y-axis is `cosine_similarity_score` ("Cosine Typicality (Layer Prototype)"), with each point labeled by `most_typical_model_concept`.

## Results

At AP=0.6, pooled across all 147 concept-category rows, ignoring which category each concept belongs to, the correlation between Human Typicality and global Cosine Typicality is **r=-0.001, p=0.99, no relationship at all**. Per-category correlations show real heterogeneity: **weapon** (r=0.88, p<0.001, n=8) and **vehicle** (r=0.74, p=0.006, n=12) are strong and positive; **jewelry** (r=-0.98), **container** (r=-0.74, p=0.01), and **drink** (r=-0.72, p=0.03) are significantly *negative*. Only 9 of 17 categories have a positive r at all, and only 2 of 17 are both positive and significant.

### Across AP thresholds

| AP | Pooled r | Pooled p | n | Positive / 17 cat. | Sig. positive | Sig. negative |
|---|---|---|---|---|---|---|
| 0.5 | 0.091 | 0.27 (n.s.) | 147 | 10 | 2 | 2 |
| 0.6 | -0.001 | 0.99 (n.s.) | 147 | 9 | 2 | 3 |
| 0.7 | -0.101 | 0.22 (n.s.) | 147 | 7 | 1 | 3 |
| 0.8 | -0.149 | 0.073 (n.s.) | 146 | 5 | 1 | 2 |
| 0.9 | -0.242 | **0.011 (sig.)** | 110 | 4 | 0 | 3 |

This sweep reveals a clear and somewhat damning trend that the single-threshold view hides: the pooled correlation doesn't just hover near zero. It **moves monotonically from weakly positive to significantly negative** as AP tightens (0.091 → -0.001 → -0.101 → -0.149 → -0.242). At the strictest threshold, the pooled relationship is the *opposite* of the hypothesis, and is itself statistically significant. The number of categories with any positive correlation also shrinks steadily (10 → 9 → 7 → 5 → 4 out of 17), and by AP=0.9 **zero** categories show a significant positive relationship while 3 show significant negative ones.

Two categories are worth calling out specifically because they're consistent across every threshold tested, which is rare in this analysis:

- **`weapon` is positive and significant at every single AP from 0.5 to 0.8** (r=0.87, 0.88, 0.85, 0.79), the most reliable piece of evidence anywhere in this module that the hypothesis can hold. It's notably absent from the AP=0.9 "significant categories" list only because the per-category n shrinks too far to reach significance at that threshold, not because the relationship reverses.
- **`container` is negative and significant at every single AP from 0.5 to 0.9** (r=-0.68, -0.74, -0.81, -0.84, -0.96), an equally reliable finding in the *wrong* direction. `drink` follows a similar pattern from AP=0.6 onward (r=-0.72 to -0.79).

So the honest summary, now strengthened by the threshold sweep rather than weakened by it: this module does not show that the model's expert-allocation geometry recovers human typicality judgments in general, and the strictest, most selective AP threshold makes that conclusion worse, not better. A small number of categories (weapon, and to a lesser extent vehicle) show a real, repeatable positive relationship; a small number of others (container, drink) show an equally real, repeatable relationship in the opposite direction. Any claim about "the expert representation captures psychologically plausible structure" needs to be scoped to specific categories, not stated as a general result, and should note that the relationship gets worse, not better, as the expert-selection criterion is tightened.
