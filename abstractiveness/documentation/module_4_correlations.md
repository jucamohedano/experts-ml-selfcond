# Module 4: Frequency, Typicality, and Expert-Count Correlations

## Research question

Are concepts that are more frequent, more typical (per human judgment), or more category-aligned (by Jaccard similarity) also associated with more expert units or stronger category overlap?

## Method

The module merges the module-1 metadata table with the module-3 similarity results and builds five regression datasets: frequency vs. expert count, human typicality vs. expert count, frequency vs. human typicality, human typicality vs. Jaccard similarity, and frequency vs. Jaccard similarity. Each of the five regressions is run through a single shared helper (`_run_regression_panel`) that drops missing values for the relevant pair of columns, saves that cleaned subset to CSV, draws the scatter+regression plot with a Pearson-annotated title if more than two points remain, and saves the PNG, all within one call, so every panel's CSV and PNG are always written together.

## Outputs

### CSV tables

The module saves the core regression datasets used for the plots.

#### 1. frequency_vs_expert_count.csv

**Input:** `merged_metadata_df[['log_frequency', 'expert_count']]`, dropping NaNs.

| Column | Type | Description |
|--------|------|-------------|
| log_frequency | float | Base-10 logarithm of the concept's frequency. |
| expert_count | int | Number of expert rows retained for the concept (module 1). |

Example (head of `AP_0.6/4_correlations/frequency_vs_expert_count.csv` in `research_plots_150_revised_executor`):

| log_frequency | expert_count |
|---|---|
| 4.2468 | 523 |
| 3.7168 | 960 |
| 5.1446 | 231 |
| 2.4362 | 397 |
| 3.4951 | 831 |

#### 2. typicality_vs_expert_count.csv

**Input:** `merged_metadata_df[['human_typicality', 'expert_count']]`, dropping NaNs.

| Column | Type | Description |
|--------|------|-------------|
| human_typicality | float | Human-judged typicality score for the concept. |
| expert_count | int | Number of expert rows retained for the concept (module 1). |

Example (head of `AP_0.6/4_correlations/typicality_vs_expert_count.csv` in `research_plots_150_revised_executor`):

| human_typicality | expert_count |
|---|---|
| 0.58 | 523 |
| 0.459 | 960 |
| 0.378 | 397 |
| 0.347 | 831 |
| 0.696 | 105 |

#### 3. frequency_vs_typicality.csv

**Input:** `merged_metadata_df[['log_frequency', 'human_typicality']]`, dropping NaNs.

| Column | Type | Description |
|--------|------|-------------|
| log_frequency | float | Base-10 logarithm of the concept's frequency. |
| human_typicality | float | Human-judged typicality score for the concept. |

Example (head of `AP_0.6/4_correlations/frequency_vs_typicality.csv` in `research_plots_150_revised_executor`):

| log_frequency | human_typicality |
|---|---|
| 4.2468 | 0.58 |
| 3.7168 | 0.459 |
| 2.4362 | 0.378 |
| 3.4951 | 0.347 |
| 4.4427 | 0.696 |

#### 4. typicality_vs_jaccard.csv

**Input:** `merged_metadata_df` inner-joined with module 3's `category_concept_similarity_metrics.csv` on `concept`/`category`, then `[['human_typicality', 'jaccard_pct']]`, dropping NaNs.

| Column | Type | Description |
|--------|------|-------------|
| human_typicality | float | Human-judged typicality score for the concept. |
| jaccard_pct | float | Jaccard similarity percentage between the concept and its category (module 3). |

Example (head of `AP_0.6/4_correlations/typicality_vs_jaccard.csv` in `research_plots_150_revised_executor`):

| human_typicality | jaccard_pct |
|---|---|
| 0.58 | 12.4535 |
| 0.459 | 11.0375 |
| 0.378 | 2.8436 |
| 0.347 | 2.3968 |
| 0.696 | 4.7619 |

#### 5. frequency_vs_jaccard.csv

**Input:** the same merged frequency/similarity table as above, restricted to `[['log_frequency', 'jaccard_pct']]`, dropping NaNs.

| Column | Type | Description |
|--------|------|-------------|
| log_frequency | float | Base-10 logarithm of the concept's frequency. |
| jaccard_pct | float | Jaccard similarity percentage between the concept and its category (module 3). |

Example (head of `AP_0.6/4_correlations/frequency_vs_jaccard.csv` in `research_plots_150_revised_executor`):

| log_frequency | jaccard_pct |
|---|---|
| 4.2468 | 12.4535 |
| 3.7168 | 11.0375 |
| 2.4362 | 2.8436 |
| 3.4951 | 2.3968 |
| 4.4427 | 4.7619 |

#### 6. correlation_summary.csv

**Input:** one row appended for each of the five regressions above, recording the Pearson statistic computed with `scipy.stats.pearsonr` on the corresponding x/y columns.

| Column | Type | Description |
|--------|------|-------------|
| plot_name | string | Identifier of the corresponding regression/plot (matches the `.png` file stem). |
| x_variable | string | Name of the column used on the x-axis. |
| y_variable | string | Name of the column used on the y-axis. |
| pearson_r | float | Pearson correlation coefficient between x_variable and y_variable. |
| pearson_p | float | Two-sided p-value for the Pearson correlation test. |
| n_points | int | Number of (x, y) pairs used after dropping missing values. |

Example (head of `AP_0.6/4_correlations/correlation_summary.csv` in `research_plots_150_revised_executor`):

| plot_name | x_variable | y_variable | pearson_r | pearson_p | n_points |
|---|---|---|---|---|---|
| frequency_vs_expert_count | log_frequency | expert_count | -0.5521 | 1.81e-14 | 164 |
| typicality_vs_expert_count | human_typicality | expert_count | -0.2581 | 1.60e-03 | 147 |
| frequency_vs_typicality | log_frequency | human_typicality | 0.4364 | 3.29e-08 | 147 |
| typicality_vs_jaccard | human_typicality | jaccard_pct | 0.3472 | 1.64e-05 | 147 |
| frequency_vs_jaccard | log_frequency | jaccard_pct | 0.2620 | 1.35e-03 | 147 |

### Plots

Each of these is a **scatter plot with a linear regression line** (`seaborn.regplot`, including the shaded confidence band) and an annotation of the Pearson $r$/$p$ statistic in the title:

- `frequency_vs_expert_count.png`, x: `log_frequency` ("Wikipedia Frequency(Log10)"), y: `expert_count` ("Expert Count").
- `typicality_vs_expert_count.png`, x: `human_typicality` ("Human Typicality"), y: `expert_count` ("Expert Count").
- `frequency_vs_typicality.png`, x: `log_frequency` ("Wikipedia Frequency(Log10)"), y: `human_typicality` ("Human Typicality").
- `typicality_vs_jaccard.png`, x: `human_typicality` ("Human Typicality"), y: `jaccard_pct` ("Jaccard Similarity Index %").
- `frequency_vs_jaccard.png`, x: `log_frequency` ("Wikipedia Frequency(Log10)"), y: `jaccard_pct` ("Jaccard Similarity Index %").

## Results

At AP=0.6, all five correlations are statistically significant (p<0.01 in four of five, p=0.0016 in the fifth), but "significant" and "strong" are not the same thing here: with 147–164 points, even modest correlations clear the significance bar. Squaring each $r$ to get variance explained tells the more honest story:

| Relationship | r | r² (variance explained) |
|---|---|---|
| Frequency vs. Expert Count | -0.55 | 30% |
| Frequency vs. Human Typicality | 0.44 | 19% |
| Human Typicality vs. Jaccard | 0.35 | 12% |
| Human Typicality vs. Expert Count | -0.26 | 7% |
| Frequency vs. Jaccard | 0.26 | 7% |

Only the frequency-vs-expert-count relationship explains close to a third of the variance, and it's the one that's also visually obvious in `frequency_vs_expert_count.png`, a clear funnel-shaped downward trend, even though individual points still scatter by a factor of 2-3x around the line. The other four relationships explain at most ~19% of variance, and `typicality_vs_jaccard.png` makes that concrete: the points form a wide, noisy cloud with no visible trend to the naked eye, and only the fitted regression line reveals the weak r=0.35 relationship.

### Across AP thresholds

| AP | Freq vs. ExpertCount (r) | Typ vs. ExpertCount (r) | Freq vs. Typ (r) | Typ vs. Jaccard (r, p) | Freq vs. Jaccard (r, p) |
|---|---|---|---|---|---|
| 0.5 | -0.56 | -0.23 | 0.44 | 0.22 (p=0.007) | 0.12 (p=0.13, n.s.) |
| 0.6 | -0.55 | -0.26 | 0.44 | 0.35 (p=2e-5) | 0.26 (p=0.001) |
| 0.7 | -0.52 | -0.26 | 0.44 | 0.32 (p=8e-5) | 0.25 (p=0.003) |
| 0.8 | -0.47 | -0.25 | 0.43 | 0.28 (p=0.001) | 0.20 (p=0.02) |
| 0.9 | -0.44 | -0.21 | 0.42 | 0.16 (p=0.24, n.s.) | 0.20 (p=0.12, n.s.) |

Two patterns stand out. First, **frequency vs. human typicality is essentially invariant to AP** (r=0.44, 0.44, 0.44, 0.43, 0.42), which makes sense, since neither variable depends on expert allocation at all; the AP threshold only changes which concepts have a defined `expert_count`/`jaccard_pct` and therefore survive the join, not the frequency or typicality values themselves. This is a useful sanity check: relationships that don't mechanically involve expert counts should be (and are) stable across thresholds.

Second, **the two Jaccard-based relationships are the least robust finding in this module**: both lose significance entirely at AP=0.9 (typicality-vs-jaccard p=0.24, frequency-vs-jaccard p=0.12), and frequency-vs-jaccard is also not significant at the most lenient threshold, AP=0.5 (p=0.13). They're only reliably significant in the middle of the range (AP 0.6–0.8). Given that module 3 already shows the median concept-category pair has *zero* shared experts at AP=0.8–0.9, this isn't surprising: once jaccard_pct is mostly zero, there's little variance left for it to correlate with anything. The frequency-vs-expert-count and typicality-vs-expert-count relationships, by contrast, stay significant and similar in magnitude at every threshold, making them the more dependable findings in this module.
