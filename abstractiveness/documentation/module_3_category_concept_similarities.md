# Module 3: Concept-to-Category Expert Similarity

## Research question

Do the expert units associated with a concept overlap strongly with the expert units associated with the category that contains it, and would that support a hierarchical organization where category labels capture a meaningful parent structure?

## Method

The method groups the expert rows by concept and converts each concept's units into a set. For every concept-category pair in the metadata it builds the set of expert units for the concept, the set of expert units for the category label, computes their intersection and union, and derives two percentage-based similarity metrics: the Jaccard index (intersection over union, which penalizes any mismatch) and the overlap coefficient (intersection over the smaller of the two sets, which rewards the concept being a subset of the category, regardless of how much bigger the category's set is).

## Outputs

### CSV table

#### category_concept_similarity_metrics.csv

**Input:** expert-unit sets built with `expert_allocation_df.groupby("concept")["unit"].apply(set)`, compared pairwise for every `(concept, category)` row of `concept_metadata` that has a non-null `category`.

Contains one row per concept-category pair.

| Column | Type | Description |
|--------|------|-------------|
| concept | string | The concept being compared. |
| category | string | The parent category label. |
| hierarchy | string | String describing the concept-category relationship. |
| jaccard_pct | float | Percentage similarity computed with the Jaccard index. |
| overlap_pct | float | Percentage similarity computed with the overlap coefficient. |

Example (head of `AP_0.6/3_category_concept_similarities/category_concept_similarity_metrics.csv` in `research_plots_150_revised_executor`):

| concept | category | hierarchy | jaccard_pct | overlap_pct |
|---|---|---|---|---|
| alligator | animal | animal -> alligator | 11.0375 | 46.2963 |
| frog | animal | animal -> frog | 14.1538 | 29.6774 |
| goldfish | animal | animal -> goldfish | 8.0488 | 15.2778 |
| iguana | animal | animal -> iguana | 9.4156 | 40.2778 |
| leech | animal | animal -> leech | 8.2544 | 28.2407 |

### Plots

The module creates two horizontal **bar charts**:

- `jaccard_hierarchy.png`, `jaccard_pct` (x-axis) plotted against `hierarchy` (y-axis, one bar per concept-category pair), showing Jaccard similarity as a percentage for each concept-category hierarchy.
- `overlap_hierarchy.png`, `overlap_pct` (x-axis) plotted against `hierarchy` (y-axis), showing overlap coefficient as a percentage for each concept-category hierarchy.

## Results

At AP=0.6, across the 147 concept-category pairs, Jaccard similarity is consistently low: mean 6.8%, median 5.6%, and 80% of pairs fall below 10%. By the more forgiving overlap-coefficient measure (which ignores how much larger the category's expert set is), the picture improves but is still modest: mean 27.2%, median 24.7%.

So a concept's expert set is, on average, far from identical to its category's expert set (low Jaccard), but a meaningful minority of a concept's own experts are typically also experts for the category (higher overlap). The bar charts make the pair-to-pair variation visible, but on their own they don't establish whether this is more than chance; module 5's pairwise heatmaps test whether within-category pairs are systematically higher than across-category pairs, which is the sharper version of this question.

### Across AP thresholds

Both metrics collapse sharply as the AP threshold tightens, and this is the single largest threshold effect anywhere in this analysis:

| AP | n pairs | Jaccard mean / median | Overlap mean / median | % pairs with jaccard < 10% |
|---|---|---|---|---|
| 0.5 | 147 | 15.3% / 14.9% | 43.7% / 43.2% | 23% |
| 0.6 | 147 | 6.8% / 5.6% | 27.2% / 24.7% | 80% |
| 0.7 | 147 | 3.3% / 1.7% | 18.2% / 11.8% | 95% |
| 0.8 | 128 | 1.8% / 0.0% | 9.4% / 0.0% | 95% |
| 0.9 | 59 | 0.6% / 0.0% | 4.8% / 0.0% | 97% |

At AP=0.5, the median pair shares a substantial 14.9% Jaccard overlap; by AP=0.8 and 0.9, **the median concept-category pair shares zero experts at all** (median = 0.0% on both metrics). This is partly mechanical, since stricter AP keeps fewer experts per concept overall, shrinking every set and therefore every intersection, but it means the "limited but useful" similarity signal described above is really an AP=0.5–0.6 phenomenon: at the two strictest thresholds tested, more than half of all concept-category pairs have no measurable similarity whatsoever by either metric, and the analysis is effectively characterizing a small subset of pairs that still share anything at all, not the typical pair. n also shrinks at AP=0.8/0.9 because some concepts have too few retained experts to compute a set-based similarity at all.
