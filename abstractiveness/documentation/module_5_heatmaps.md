# Module 5: Pairwise Expert-Overlap Heatmaps

## Research question

Does the expert structure form recognizable clusters, with concepts that belong to the same category sharing more expert units with each other than with concepts from other categories?

## Method

Instead of comparing each concept only to its own category (module 3), this module compares every concept to every other concept. It reshapes the expert allocation data into a binary presence matrix where rows are concepts, columns are units, and a value of 1 means that unit is an expert for that concept. From that matrix it computes, for every pair of concepts, the intersection and union of their expert-unit sets, the minimum of the two set sizes, and the resulting Jaccard and overlap percentages. The full pairwise results are stored as two square matrices.

## Outputs

### CSV tables

Both matrices are built from a binary concept-by-unit presence matrix `A` (`expert_allocation_df.pivot_table(index='concept', columns='unit', values='present', fill_value=0)`, reindexed over every concept in `concept_metadata`), from which the pairwise intersection, union, and minimum-set-size matrices are computed with matrix products (`A @ A.T`).

#### 1. jaccard_matrix.csv

**Input:** `intersection / union * 100`, i.e. the pairwise Jaccard index between every pair of concepts' expert-unit sets.

A square matrix where rows and columns are concepts and entries are Jaccard similarity percentages. The unnamed first column holds the row concept label; the diagonal is always 100.

| Column | Type | Description |
|--------|------|-------------|
| (index) | string | Concept label for the row. |
| `<concept>` (one column per concept) | float | Jaccard similarity percentage between the row concept and this column concept. |

Example (head, first 6 columns, of `AP_0.6/5_heatmaps/jaccard_matrix.csv` in `research_plots_150_revised_executor`):

| (index) | animal | alligator | frog | goldfish | iguana |
|---|---|---|---|---|---|
| animal | 100.0 | 11.04 | 14.15 | 8.05 | 9.42 |
| alligator | 11.04 | 100.0 | 9.76 | 9.95 | 19.89 |
| frog | 14.15 | 9.76 | 100.0 | 12.35 | 9.20 |
| goldfish | 8.05 | 9.95 | 12.35 | 100.0 | 10.25 |
| iguana | 9.42 | 19.89 | 9.20 | 10.25 | 100.0 |

#### 2. overlap_matrix.csv

**Input:** `intersection / min(size_a, size_b) * 100`, i.e. the pairwise overlap coefficient between every pair of concepts' expert-unit sets.

A square matrix where rows and columns are concepts and entries are overlap coefficients. Same shape and indexing as `jaccard_matrix.csv`.

| Column | Type | Description |
|--------|------|-------------|
| (index) | string | Concept label for the row. |
| `<concept>` (one column per concept) | float | Overlap coefficient percentage between the row concept and this column concept. |

Example (head, first 6 columns, of `AP_0.6/5_heatmaps/overlap_matrix.csv` in `research_plots_150_revised_executor`):

| (index) | animal | alligator | frog | goldfish | iguana |
|---|---|---|---|---|---|
| animal | 100.0 | 46.30 | 29.68 | 15.28 | 40.28 |
| alligator | 46.30 | 100.0 | 54.19 | 40.53 | 33.29 |
| frog | 29.68 | 54.19 | 100.0 | 27.10 | 51.61 |
| goldfish | 15.28 | 40.53 | 27.10 | 100.0 | 41.85 |
| iguana | 40.28 | 33.29 | 51.61 | 41.85 | 100.0 |

### Plots

Two **heatmap** images (`seaborn`-style matrix heatmap rendered via the `_plot_heatmap_with_leaders` helper, `magma` colormap), with concept labels on both axes and similarity encoded by color:

- `jaccard_heatmap.png`, encodes the values of `jaccard_matrix.csv`.
- `overlap_heatmap.png`, encodes the values of `overlap_matrix.csv`.

## Results

At AP=0.6, there is a real, measurable same-category effect, but it's small and the heatmaps themselves don't make it easy to see. Splitting all 13,366 concept pairs into "same category" (619 pairs) vs. "different category" (12,747 pairs): within-category pairs average **9.05%** Jaccard versus **5.47%** across categories (1.65x higher), and **27.4%** overlap versus **19.6%** (1.4x higher). That's a genuine, directionally consistent signal, but in absolute terms both numbers are low and the within/across distributions clearly overlap heavily.

Looking at the actual rendered images confirms this: both heatmaps are dominated by the bright diagonal (self-similarity = 100%, by construction) and a faint grid pattern of brighter rows/columns, concepts that share more experts with *everyone*, not specifically with their own category, rather than clear square blocks along the diagonal where same-category concepts should cluster. A couple of small bright patches are visible near the matrix corners, but the large majority of the 150x150 grid reads as low-level, fairly uniform noise to the eye.

### Across AP thresholds

| AP | Jaccard within / across | Jaccard ratio | Overlap within / across | Overlap ratio |
|---|---|---|---|---|
| 0.5 | 21.05% / 15.54% | 1.35x | 47.83% / 39.92% | 1.20x |
| 0.6 | 9.05% / 5.47% | 1.65x | 27.42% / 19.55% | 1.40x |
| 0.7 | 4.11% / 1.99% | 2.07x | 15.45% / 9.36% | 1.65x |
| 0.8 | 1.76% / 0.55% | 3.18x | 7.63% / 3.58% | 2.13x |
| 0.9 | 0.26% / 0.08% | 3.38x | 1.21% / 0.62% | 1.95x |

This produces a genuinely counterintuitive but consistent pattern: the *relative* within-vs-across-category effect gets **stronger** as AP tightens (1.35x → 3.38x for Jaccard), exactly the opposite of what module 3 and 4's collapsing absolute similarities might suggest. But the *absolute* numbers shrink toward zero at the same time (9.05% → 0.26%). In other words, at strict thresholds, same-category concepts really are several times more similar to each other than to other concepts, but both numbers are now so close to zero that this stronger relative signal is even less likely to be visible by eye in a heatmap than it was at AP=0.6. This sharpens the original critique: **the heatmap visualization gets less informative as a picture exactly as the underlying relative signal gets more real.** A reordered/clustered heatmap, or just reporting the within/across ratio directly as this table does, would surface the finding far better than the rendered images do at any threshold tested.
