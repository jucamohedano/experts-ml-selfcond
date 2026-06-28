# Module 6: Category Prototype vs. Exemplar Divergence (JSD)

## Research question

Are categories better described as a single prototype distribution (the category-label word's own layer profile) or as the average of their member concepts' layer profiles, and does that choice matter more for some categories than others?

## Method

This module compares two ways of representing a category: the category label itself as a prototype distribution (P) and the average of the member concepts as an exemplar distribution (Q), both built from the same concept-by-layer probability matrix used in module 2 (via the shared `build_layer_probability_matrix` helper). It uses Jensen-Shannon divergence (JSD), computed via `scipy.spatial.distance.jensenshannon`, to quantify how different P and Q are, storing the result as a squared JSD value. It also measures each category's internal diversity, the average pairwise JSD among its own member concepts, to test whether categories with more internally dispersed members also diverge more from their own label.

## Outputs

### CSV table

#### dual_category_jsd.csv

**Input:** for each category, the label distribution `P` and the member-average distribution `Q`, compared with `scipy.spatial.distance.jensenshannon(P, Q, base=2)` squared; `avg_member_diversity_jsd` is the mean of all pairwise squared JS divergences among the category's member concepts (`scipy.spatial.distance.pdist`). The raw `P_dist`/`Q_dist` arrays are kept in memory for the micro plot but dropped before saving the CSV.

Contains one row per category, sorted ascending by `jensen_shannon_divergence`.

| Column | Type | Description |
|--------|------|-------------|
| category | string | Category label. |
| member_count | int | Number of member concepts used in the average distribution. |
| avg_human_typicality | float | Mean human typicality of the category members. |
| shannon_entropy_label | float | Entropy of the category-label distribution. |
| jensen_shannon_divergence | float | Divergence between the label distribution and the member-average distribution. |
| avg_member_diversity_jsd | float | Mean pairwise divergence among the category's own member concepts (internal diversity). |

Example (head of `AP_0.6/6_dual_category_jsd/dual_category_jsd.csv` in `research_plots_150_revised_executor`):

| category | member_count | avg_human_typicality | shannon_entropy_label | jensen_shannon_divergence | avg_member_diversity_jsd |
|---|---|---|---|---|---|
| animal | 10 | 0.3596 | 4.5295 | 0.0642 | 0.1055 |
| drink | 9 | 0.5021 | 4.5192 | 0.0646 | 0.1283 |
| headwear | 3 | 0.3997 | 4.6060 | 0.0650 | 0.1627 |
| hardware | 9 | 0.4602 | 4.2198 | 0.0684 | 0.1709 |
| container | 11 | 0.5622 | 4.0633 | 0.0801 | 0.1585 |

### Plots

1. `scatter_jsd_vs_member_diversity.png`, a **scatter plot with a linear regression line**, x: `jensen_shannon_divergence` ("Label vs. Averaged Members Divergence"), y: `avg_member_diversity_jsd` ("Average Pairwise Member Divergence"), one point per category (annotated with the category name).
2. `micro_jsd_distributions.png`, two stacked **overlaid area charts** (one for the lowest-JSD category, one for the highest-JSD category): x-axis is the model layer, y-axis is probability density, with one filled curve for the prototype distribution `P_dist` (category label) and one for the exemplar distribution `Q_dist` (average of members).
3. `scatter_jsd_vs_entropy.png`, a **scatter plot with a linear regression line**, x: `shannon_entropy_label`, y: `jensen_shannon_divergence`, one point per category (annotated with the category name).

## Results

At AP=0.6, categories vary a lot in how well their label represents their members: JSD ranges 4x across the 17 categories, from **0.064** (animal, well aligned) to **0.248** (furniture, poorly aligned). Of the two relationships this module tests, only one holds up at this threshold: **label entropy predicts divergence** (r=-0.60, p=0.011, n=17, visible in `scatter_jsd_vs_entropy.png` as a real if noisy downward trend), while **internal member diversity does not** (r=0.32, p=0.21, n=17; `scatter_jsd_vs_member_diversity.png` shows a loose cloud with no visible trend).

### Across AP thresholds

| AP | n categories | JSD range | JSD mean | Entropy-vs-JSD (r, p) | Diversity-vs-JSD (r, p) |
|---|---|---|---|---|---|
| 0.5 | 17 | 0.035–0.131 | 0.068 | 0.26 (p=0.32, n.s.) | 0.33 (p=0.19, n.s.) |
| 0.6 | 17 | 0.064–0.248 | 0.131 | -0.60 (p=0.011) | 0.32 (p=0.21, n.s.) |
| 0.7 | 17 | 0.071–0.684 | 0.286 | -0.90 (p<0.0001) | 0.25 (p=0.34, n.s.) |
| 0.8 | 15 | 0.198–0.892 | 0.458 | -0.86 (p<0.0001) | 0.03 (p=0.90, n.s.) |
| 0.9 | 9 | 0.209–0.883 | 0.514 | -0.74 (p=0.024) | -0.26 (p=0.50, n.s.) |

Two findings emerge that are *more* informative than the single-threshold view:

**The entropy-vs-divergence relationship is not robust at lenient thresholds, it only emerges, and then strongly, in the middle-to-strict range.** At AP=0.5 it's not even in the right direction (r=+0.26, not significant). It flips to a real, significant negative relationship at AP=0.6 and becomes very strong at AP=0.7–0.8 (r=-0.90, -0.86, both p<0.0001) before weakening slightly at AP=0.9 (r=-0.74, n drops to 9 categories as some lose enough members to qualify). The AP=0.6 result reported above is a fair representative of a real effect, but understates how strong it gets at AP 0.7–0.8.

**Internal member diversity never predicts divergence, at any threshold.** r flips sign across the range (0.26, 0.32, 0.25, 0.03, -0.26) and is never close to significant (smallest p=0.19). This is a much stronger negative result than the single-threshold view suggested: it isn't a borderline "not enough power" case, it's a consistent null finding across every AP value tested, including ones with the same n=17. That hypothesis, that more internally scattered categories diverge further from their own label, has no support anywhere in this sweep.

JSD itself also grows roughly 7.5x in magnitude from AP=0.5 to AP=0.9 (mean 0.068 → 0.514), tracking the same "fewer experts, noisier/more divergent per-concept distributions" pattern seen in modules 1–3.
