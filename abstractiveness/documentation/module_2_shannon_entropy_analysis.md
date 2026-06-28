# Module 2: Shannon Entropy and Peak/Average Layer Allocation

## Research question

Are specific concepts more sharply localized in a few layers than the broader semantic categories that contain them, i.e. is expert allocation more concentrated at the concept level than at the category level?

## Method

This module measures how concentrated expert allocations are for individual concepts and for broader category labels. It groups expert allocations by concept and layer index to form a count matrix (via the shared `build_layer_probability_matrix` helper, also used by module 6), normalizes each row to a probability distribution, and computes Shannon entropy with `scipy.stats.entropy` (base 2, in bits) for each concept and category definition. For each category it computes entropy under two separate definitions, the category-label distribution itself and the average distribution of its member concepts, and reports the Pearson correlation between those two distributions. It also records the peak layer and the mean (expected) layer position for each distribution, and runs a Mann-Whitney U test comparing concept-level entropy against category-level entropy.

## Outputs

### CSV tables

#### 1. shannon_entropy_concepts.csv

**Input:** the concept-by-layer probability matrix, passed through `scipy.stats.entropy`.

Contains one row per concept.

| Column | Type | Description |
|--------|------|-------------|
| concept | string | Concept identifier. |
| shannon_entropy | float | Entropy of the concept’s layerwise expert distribution. |
| peak_layer | int | Layer where the concept has its strongest expert concentration. |
| avg_layer | float | Expected layer value for the concept distribution. |
| experts_count | int | Number of expert rows used to compute the distribution. |

Example (head of `AP_0.6/2_shannon_entropy_analysis/shannon_entropy_concepts.csv` in `research_plots_150_revised_executor`):

| concept | shannon_entropy | peak_layer | avg_layer | experts_count |
|---|---|---|---|---|
| airplane | 4.5767 | 47 | 25.8547 | 523 |
| alligator | 5.0540 | 15 | 26.3125 | 960 |
| animal | 4.5295 | 3 | 20.7489 | 231 |
| anklet | 4.5219 | 3 | 17.9093 | 397 |
| anvil | 4.9849 | 19 | 24.5018 | 831 |

#### 2. shannon_entropy_categories.csv

**Input:** the same per-layer probability distributions as above, looked up for the category-label row itself (Definition A) and averaged across the category's member concepts (Definition B), plus a Pearson correlation between the two distributions.

Contains one row per category.

| Column | Type | Description |
|--------|------|-------------|
| category | string | Category label. |
| shannon_entropy | float | Entropy of the category-label distribution. |
| peak_layer | int | Layer of the category-label peak. |
| avg_layer | float | Expected layer of the category-label distribution. |
| shannon_entropy_average | float | Entropy of the average member distribution. |
| peak_layer_average | int | Peak layer of the average member distribution. |
| avg_layer_average | float | Expected layer of the average member distribution. |
| pearson_correlation_distributions | float | Correlation between the label distribution and the member-average distribution. |
| member_count | int | Number of valid members contributing to the category summary. |

Example (head of `AP_0.6/2_shannon_entropy_analysis/shannon_entropy_categories.csv` in `research_plots_150_revised_executor`):

| category | shannon_entropy | peak_layer | avg_layer | shannon_entropy_average | peak_layer_average | avg_layer_average | pearson_correlation_distributions | member_count |
|---|---|---|---|---|---|---|---|---|
| animal | 4.5295 | 3 | 20.7489 | 4.8733 | 3 | 22.5487 | 0.8996 | 10 |
| clothing | 3.7835 | 38 | 27.6000 | 4.8622 | 3 | 23.0316 | 0.7791 | 10 |
| container | 4.0633 | 3 | 15.1250 | 4.6756 | 3 | 18.5659 | 0.9154 | 11 |
| drink | 4.5192 | 1 | 21.4049 | 4.7349 | 3 | 23.3514 | 0.7940 | 9 |
| fastener | 4.8612 | 15 | 16.9444 | 4.5435 | 3 | 18.0433 | 0.5227 | 5 |

### Plots

1. `peak_average_layers.png`, a combined **bar chart + KDE (kernel density) overlay** on twin y-axes: bars show `% of group peaking in this layer` against the model layer (derived from `peak_layer`), while the overlaid density curves show the distribution of `avg_layer` (expected layer position); both are colored/grouped by `group_type` (Specific Concepts, Broad Categories, Broad Categories Avg).
2. `category_concept_shannon_entropies.png`, a **violin plot** (with quartile lines) overlaid with a **strip plot** of individual points, comparing `shannon_entropy` (y-axis) across the two groups `group_type` = concepts vs. categories (x-axis).
3. `category_shannon_entropies_bar.png`, a horizontal **bar chart** of `shannon_entropy` (x-axis) per `category` (y-axis), sorted descending, ranking categories by entropy.

## Results

At AP=0.6, the hypothesis holds, but only modestly in magnitude. Mean entropy for the 164 concepts is **4.49 bits** (SD 0.35) versus **4.18 bits** (SD 0.33) for the 17 category labels, so categories are lower-entropy (more concentrated) than concepts, and a Mann-Whitney U test confirms this is unlikely to be chance (U=698.5, p=3.6×10⁻⁴). But the absolute gap is small relative to the scale: the maximum possible entropy over 48 layers is log₂(48) ≈ 5.58 bits, so both groups sit well below the ceiling and only about 6% apart from each other. `category_concept_shannon_entropies.png` shows this directly: the two violins clearly overlap, with the category distribution shifted down and narrower, not cleanly separated from the concept distribution.

A more striking, and easy to miss, pattern sits in the same CSV: averaging a category's *members* together (`shannon_entropy_average`, mean 4.79 bits) produces the **least** concentrated distribution of the three, even less concentrated than individual concepts, let alone the category label itself. In other words, the category-label word is the sharpest of the three representations, but smoothing across its members washes that sharpness out rather than preserving it. This is exactly the gap that module 6 quantifies directly with Jensen-Shannon divergence, and it complicates a simple "category labels are just an average of their members" story.

### Across AP thresholds

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
3. **Label-vs-member-average shape correlation weakens monotonically** as AP tightens (0.83 → 0.82 → 0.73 → 0.62 → 0.59). At lenient thresholds the label and its members peak in very similar places; at stricter thresholds they diverge more, consistent with there being fewer, noisier experts per concept to average over.

The *relative* size of the concept-vs-category gap also grows sharply with AP: at AP=0.5 concepts and categories differ by only 0.17 bits (4.83 vs 4.66); by AP=0.9 they differ by 0.89 bits on a much smaller overall scale (1.94 vs 1.05), so categories collapse toward near-total concentration faster than concepts do as the threshold strips away marginal experts.
