# Module 4: Frequency, Typicality, and Expert-Count Correlations

## Research question

Are concepts that are more frequent, more typical (per human judgment), or more category-aligned (by Jaccard or overlap similarity) also associated with more expert units or stronger category overlap? Where typicality and category alignment are related, does that relationship survive controlling for word frequency, and does the model-derived Cosine Typicality (module 7) track category alignment as well as Human Typicality does? And is module 2's Shannon entropy independent of the expert count, as its group comparisons require?

## Analysis

Notation. For each concept $c$ the module works with the scalar variables assembled by modules 1–3: the log-frequency $\tilde{f}_c = \log_{10} f_c$, the Human Typicality score $t_c$, the expert count $n_c$, the Jaccard similarity to the parent category $J(c,k)$ (written $J_c$ below, since each concept has one parent), the overlap coefficient to the parent category $O_c$, and module 2's Shannon entropy $H(c)$. The base table is module 1's `expert_counts_with_metadata.csv`, inner-joined with module 3's `category_concept_similarity_metrics.csv` on (`concept`, `category`) for the similarity-based panels, and the entropy panel of subchapter 4.4 draws directly on module 2's descriptor tables.

**Analysis scopes.** This module has no expert data of its own, it correlates the scalars modules 1 to 3 and 7 produce, so it inherits their scope: every panel is recomputed per scope and written to the module folder (whole model) or to `sublayers/<rank>_<sublayer>/`. One panel is the exception. `frequency_vs_typicality` reads only metadata columns, so it is identical in every scope and is written once at the module's top level rather than copied into each sublayer folder, which also means it appears in the whole-model `correlation_summary.csv` only. The cross-scope summary of the remaining panels' correlation coefficients lands in `sublayer_comparison.csv` and `sublayer_comparison.png`. Its columns are the Pearson $r$ of typicality against Jaccard, of frequency against expert count, of Shannon entropy against expert count, and of Jaccard against Cosine Typicality, one per scope, left empty wherever that panel's coverage gate was not cleared. Module 1, subchapter 1.7 defines the scopes and the rank prefix.

### 4.1 Bivariate Pearson regression panels

**Mathematical formulation.** Six variable pairs $(x, y)$ are each tested for a linear association. For a pair with $n$ complete observations $(x_i, y_i)$ (rows with a missing value in either variable are dropped per panel), the Pearson correlation coefficient is

$$r_{xy} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2}\;\sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}} \in [-1, 1],$$

with the two-sided p-value obtained from the exact null distribution used by `scipy.stats.pearsonr` (equivalently, from the statistic $t = r\sqrt{(n-2)/(1-r^2)}$ under a $t_{n-2}$ reference). $r_{xy}$ measures only the *linear* component of the relationship, while $r_{xy}^2$ is the fraction of variance in $y$ that a linear function of $x$ would explain, which is the effect-size reading used in the Results. Each panel also fits and draws the ordinary-least-squares line $\hat{y} = \beta x + \alpha$ with $\beta = \operatorname{Cov}(x,y)/\operatorname{Var}(x)$, $\alpha = \bar{y} - \beta\bar{x}$ (rendered by `seaborn.regplot` with its confidence band). The six pairs, in terms of the symbols above:

| Panel | $x$ | $y$ | Tests whether… |
|---|---|---|---|
| frequency_vs_expert_count | $\tilde{f}_c$ | $n_c$ | more frequent words recruit more experts |
| typicality_vs_expert_count | $t_c$ | $n_c$ | more typical concepts recruit more experts |
| frequency_vs_typicality | $\tilde{f}_c$ | $t_c$ | frequent words are judged more typical (no expert data involved) |
| typicality_vs_jaccard | $t_c$ | $J_c$ | more typical concepts share more experts with their category |
| typicality_vs_overlap | $t_c$ | $O_c$ | the typicality–alignment link survives switching to the containment-based similarity (module 3's overlap coefficient), which ignores the category set's larger size |
| frequency_vs_jaccard | $\tilde{f}_c$ | $J_c$ | more frequent words share more experts with their category |
| jaccard_vs_layer_profile | $J_c$ | $S_c$ | sharing neurons with the category and allocating experts to the same depths as the category are the same thing, or two different things (module 3, subchapter 3.2) |

Every panel runs through a single shared helper (`_run_regression_panel`) that drops missing values for the relevant pair, saves the cleaned subset to CSV, draws the scatter plot and saves the PNG, all within one call, so every panel's CSV and PNG are always written together regardless of how much data survived.

**Where the missing values come from.** Module 4 never generates missingness itself, it only inherits gaps that were created upstream, and a concept can drop out of a panel for two structurally different reasons.

The first is a value that is genuinely null in the source metadata, independent of the AP threshold. `human_typicality` is null for every category-label word (`furniture`, `clothing`, and so on), since those are root nodes with no HSJ pairwise rating of their own, and it can also be null for the small number of concepts the HSJ rating procedure did not cover. `frequency` (and therefore `log_frequency`, computed only when `frequency > 0` in module 1's `save_expert_counts_metadata`) is null or non-positive for the handful of concepts absent from the Wikipedia frequency source. These gaps show up as `NaN` inside an otherwise present row, so `dropna` removes only that row from only that panel.

The second, and the dominant one as AP tightens, is a concept disappearing entirely from an upstream table because it failed a structural condition, not because a cell is `NaN`. Three such conditions feed module 4:

- `expert_count` and everything derived from it: module 1's `save_expert_counts_metadata` builds the base table with `expert_allocation_df.groupby("concept").size()` then an inner merge onto the metadata. A concept with zero expert units surviving the AP filter has no rows to group, so it is absent from the grouped table, not present with a zero. The inner merge then drops it from `merged_metadata_df` entirely, and every module 4 panel built on that table (`frequency_vs_expert_count`, `typicality_vs_expert_count`, `frequency_vs_typicality`, `shannon_entropy_vs_expert_count`) loses it along with it.
- `jaccard_pct` and `overlap_pct`: module 3 only emits a row for a concept when *both* the concept's own expert set and its parent category's expert set are non-empty (`if u_concept and u_category` in `plot_hierarchy_similarities`). Root category-label words are excluded even earlier, by `concept_metadata.dropna(subset=["category"])`, since a label is never its own child. As a consequence, if the *category label itself* loses all its experts at a strict threshold, every member concept of that category loses its Jaccard and overlap value too, not just the label.
- `global_cosine_typicality`: module 7 skips any category with fewer than two member concepts that still have expert data (`if len(valid_members) < 2: continue`), so every concept in an under-populated category is absent from module 4b's panel.

In short, `NaN` in a retained row means a rating was never collected, while a concept's total absence from a panel means it (or its category) ran out of surviving expert units at the current AP threshold. The two are handled the same way by `dropna`, but only the second one is threshold-dependent, and it drives the coverage collapse at strict thresholds.

**When a correlation is reported.** Every panel is gated on two conditions together, both defined as module constants (`MIN_ABSOLUTE_N = 30`, `MIN_COVERAGE_FRACTION = 0.75`):

$$n \ge \texttt{MIN\_ABSOLUTE\_N} \quad \text{and} \quad \frac{n}{n_{\text{total}}} \ge \texttt{MIN\_COVERAGE\_FRACTION},$$

where $n$ is the number of complete pairs after `dropna` and $n_{\text{total}}$ is a fixed count of concepts that could in principle have contributed to that panel, read once from `concept_metadata` and independent of the AP threshold: the full metadata row count for panels that only need metadata and expert counts, or the count of concepts with a defined category for the Jaccard/overlap/Cosine-Typicality panels (root labels never appear there by construction and are excluded from the denominator). Because $n_{\text{total}}$ does not move with the AP threshold, `coverage_pct` in the summary reports how much of the concept set a given panel rests on at any threshold. The gate suppresses reporting on small or non-representative subsets, since survivors at a strict AP are systematically the higher-frequency or better-populated concepts rather than a random sample.

When a panel clears both bars, the Pearson statistic and the fitted regression line are computed and drawn. When it does not, the scatter of the remaining points is drawn unfitted (titled with the coverage shortfall) and the CSV holds the cleaned data, but `pearson_r`/`pearson_p` are left empty in `correlation_summary.csv`. Every panel gets a summary row either way, with `n_points`, `n_total_relevant`, and `coverage_pct` always populated.

**Generated data structures.** One two-column CSV per panel (the cleaned $(x, y)$ data actually used) and one PNG per panel, plus one summary CSV collecting the statistics of all panels in the module.

#### 1. frequency_vs_expert_count.csv, columns `log_frequency` ($\tilde{f}_c$), `expert_count` ($n_c$)

Example (head of `AP_0.6/4_correlations/frequency_vs_expert_count.csv` in `research_plots_150_revised_executor_again`):

| log_frequency | expert_count |
|---|---|
| 4.2468 | 523 |
| 3.7168 | 960 |
| 5.1446 | 231 |
| 2.4362 | 397 |
| 3.4951 | 831 |

#### 2. typicality_vs_expert_count.csv, columns `human_typicality` ($t_c$), `expert_count` ($n_c$)

Example (head of `AP_0.6/4_correlations/typicality_vs_expert_count.csv`):

| human_typicality | expert_count |
|---|---|
| 0.58 | 523 |
| 0.459 | 960 |
| 0.378 | 397 |
| 0.347 | 831 |
| 0.696 | 105 |

#### 3. frequency_vs_typicality.csv, columns `log_frequency` ($\tilde{f}_c$), `human_typicality` ($t_c$)

Example (head of `AP_0.6/4_correlations/frequency_vs_typicality.csv`):

| log_frequency | human_typicality |
|---|---|
| 4.2468 | 0.58 |
| 3.7168 | 0.459 |
| 2.4362 | 0.378 |
| 3.4951 | 0.347 |
| 4.4427 | 0.696 |

#### 4. typicality_vs_jaccard.csv, columns `human_typicality` ($t_c$), `jaccard_pct` ($J_c$)

Example (head of `AP_0.6/4_correlations/typicality_vs_jaccard.csv`):

| human_typicality | jaccard_pct |
|---|---|
| 0.58 | 8.3994 |
| 0.459 | 4.0175 |
| 0.378 | 1.0776 |
| 0.347 | 0.4454 |
| 0.696 | 2.2727 |

#### 5. typicality_vs_overlap.csv, columns `human_typicality` ($t_c$), `overlap_pct` ($O_c$)

Example (head of `AP_0.6/4_correlations/typicality_vs_overlap.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`):

| human_typicality | overlap_pct |
|---|---|
| 0.427 | 29.5752 |
| 0.045 | 8.0065 |
| 0.323 | 45.6869 |
| 0.852 | 38.75 |
| 0.979 | 51.6484 |

#### 6. frequency_vs_jaccard.csv, columns `log_frequency` ($\tilde{f}_c$), `jaccard_pct` ($J_c$)

Example (head of `AP_0.6/4_correlations/frequency_vs_jaccard.csv`):

| log_frequency | jaccard_pct |
|---|---|
| 4.2468 | 8.3994 |
| 3.7168 | 4.0175 |
| 2.4362 | 1.0776 |
| 3.4951 | 0.4454 |
| 4.4427 | 2.2727 |

#### 7. correlation_summary.csv

One row per panel run in this module (including subchapters 4.2, 4.3, and 4.4), recording the Pearson statistic computed on the corresponding $(x, y)$ columns.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| plot_name | string | (none) | Identifier of the corresponding regression/plot (matches the `.png` file stem). |
| x_variable | string | $x$ | Name of the column used on the x-axis. |
| y_variable | string | $y$ | Name of the column used on the y-axis. |
| pearson_r | float or empty | $r_{xy}$ | Pearson correlation coefficient, left empty when the panel did not clear the coverage bar (see above). |
| pearson_p | float or empty | $p$ | Two-sided p-value, left empty under the same condition as pearson_r. |
| n_points | int | $n$ | Number of $(x, y)$ pairs used after dropping missing values. |
| n_total_relevant | int | $n_{\text{total}}$ | Fixed count of concepts that could in principle contribute to this panel (read from `concept_metadata`, independent of AP), the coverage denominator. |
| coverage_pct | float | $100 \, n / n_{\text{total}}$ | Percentage of the relevant concepts actually retained. A correlation is only computed when `n_points >= 30` and `coverage_pct >= 75`. |
| controlling_for | string | $z$ | Control variable, only populated for the partial-correlation row (4.2), and empty otherwise. |

Example (GPT-2 Richie-HSJ run, `mlp.c_fc` sublayer, AP=0.5, where every panel clears the coverage bar):

| plot_name | x_variable | y_variable | pearson_r | pearson_p | n_points | n_total_relevant | coverage_pct |
|---|---|---|---|---|---|---|---|
| frequency_vs_expert_count | log_frequency | expert_count | -0.1329 | 5.82e-02 | 204 | 204 | 100.0 |
| typicality_vs_expert_count | human_typicality | expert_count | 0.0355 | 6.21e-01 | 196 | 204 | 96.1 |
| frequency_vs_typicality | log_frequency | human_typicality | -0.0256 | 7.22e-01 | 196 | 204 | 96.1 |
| typicality_vs_jaccard | human_typicality | jaccard_pct | 0.2764 | 8.79e-05 | 196 | 196 | 100.0 |
| frequency_vs_jaccard | log_frequency | jaccard_pct | 0.0789 | 2.72e-01 | 196 | 196 | 100.0 |

At AP=0.9 in the same run, `expert_count` covers only 98 of 204 concepts (48% coverage), so `frequency_vs_expert_count`, `typicality_vs_expert_count`, `frequency_vs_typicality`, and `shannon_entropy_vs_expert_count` all report `n_points=98`, `coverage_pct=48.0`, and empty `pearson_r`/`pearson_p`, and the Jaccard-based panels do not appear at all, since module 3 returns an empty similarity table at that threshold (no concept-category pair shares any expert unit).

**Plots.** Each panel is a **scatter plot with a linear regression line** (`seaborn.regplot`, including the shaded confidence band) and the Pearson $r$/$p$ annotated in the title:

- `frequency_vs_expert_count.png`, x: `log_frequency` ($\tilde{f}_c$), y: `expert_count` ($n_c$).
- `typicality_vs_expert_count.png`, x: `human_typicality` ($t_c$), y: `expert_count` ($n_c$).
- `frequency_vs_typicality.png`, x: `log_frequency` ($\tilde{f}_c$), y: `human_typicality` ($t_c$).
- `typicality_vs_jaccard.png`, x: `human_typicality` ($t_c$), y: `jaccard_pct` ($J_c$).
- `typicality_vs_overlap.png`, x: `human_typicality` ($t_c$), y: `overlap_pct` ($O_c$).
- `frequency_vs_jaccard.png`, x: `log_frequency` ($\tilde{f}_c$), y: `jaccard_pct` ($J_c$).

### 4.2 Partial correlation: Jaccard vs. typicality, controlling for frequency

**Mathematical formulation.** If frequency were correlated with both typicality and Jaccard similarity, their pairwise correlation could be partly a frequency artifact rather than a direct typicality-alignment association. To isolate the direct component, both variables are residualized against the control $z_c = \tilde{f}_c$. For a variable $v$ regressed on $z$ by simple OLS,

$$\beta_v = \frac{\operatorname{Cov}(z, v)}{\operatorname{Var}(z)}, \qquad \alpha_v = \bar{v} - \beta_v \bar{z}, \qquad \tilde{v}_c = v_c - (\beta_v z_c + \alpha_v),$$

the residual $\tilde{v}_c$ is the part of $v_c$ that a linear function of frequency cannot explain. Applying this to both variables gives the residual vectors $\tilde{J}_c$ (`jaccard_resid`) and $\tilde{t}_c$ (`typicality_resid`), and the partial correlation is the plain Pearson correlation of the residuals:

$$r_{Jt \cdot f} = \operatorname{corr}\!\left(\tilde{J}, \tilde{t}\right),$$

which is algebraically identical to the standard partial-correlation formula $r_{Jt\cdot f} = \dfrac{r_{Jt} - r_{Jf}\, r_{tf}}{\sqrt{(1 - r_{Jf}^2)(1 - r_{tf}^2)}}$. A non-zero $r_{Jt\cdot f}$ indicates a typicality-alignment association not attributable to frequency. Missingness here is the union of the sources listed under 4.1 for `jaccard_pct` and `human_typicality`, plus `log_frequency`, and the same coverage rule applies, against the categorized-concepts denominator, since this panel needs a defined category throughout.

**Generated data structures.** One CSV, one PNG, and one row appended to `correlation_summary.csv` (with `controlling_for = log_frequency`):

- `partial_correlation_jaccard_typicality.csv`, the full working table (only rows complete in all three variables):

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Concept identifier. |
| category | string | $k$ | Parent category. |
| log_frequency | float | $z_c = \tilde{f}_c$ | Control variable. |
| jaccard_pct | float | $J_c$ | Raw Jaccard similarity to the category (module 3). |
| human_typicality | float | $t_c$ | Raw Human Typicality score. |
| jaccard_resid | float | $\tilde{J}_c$ | Jaccard residual after removing the linear frequency component. |
| typicality_resid | float | $\tilde{t}_c$ | Typicality residual after removing the linear frequency component. |

Example (head of `AP_0.6/4_correlations/partial_correlation_jaccard_typicality.csv` in `research_plots_150_revised_executor_again`):

| concept | category | log_frequency | jaccard_pct | human_typicality | jaccard_resid | typicality_resid |
|---|---|---|---|---|---|---|
| airplane | vehicle | 4.2468 | 8.3994 | 0.58 | 3.7112 | 0.0335 |
| alligator | animal | 3.7168 | 4.0175 | 0.459 | 0.2522 | -0.0336 |
| anklet | jewelry | 2.4362 | 1.0776 | 0.378 | -0.4574 | 0.0156 |
| anvil | tool | 3.4951 | 0.4454 | 0.347 | -2.9337 | -0.1231 |
| bag | container | 4.4427 | 2.2727 | 0.696 | -2.7566 | 0.1296 |

- `partial_correlation_jaccard_typicality.png`, a **scatter plot with a linear regression line**, x: `typicality_resid` ($\tilde{t}_c$, "Human Typicality (residual after removing Frequency)"), y: `jaccard_resid` ($\tilde{J}_c$, "Jaccard Similarity Index % (residual after removing Frequency)"), with the partial $r$/$p$ in the title.

### 4.3 Jaccard vs. Cosine Typicality (module 4b, run after module 7)

**Mathematical formulation.** This second pass reuses module 7's model-derived global Cosine Typicality $T^{\cos}_c$, the cosine similarity between concept $c$'s binary expert-allocation vector and its category's centroid vector (see module 7, subchapter 7.1, for the full construction), and asks whether it tracks a concept's set-based category alignment the way Human Typicality does in panel 4 of subchapter 4.1. It computes the same Pearson statistic as 4.1 on the pair

$$x = T^{\cos}_c, \qquad y = J_c,$$

over the inner join of module 3's similarity table with module 7's `global_prototype_typicality.csv` on (`concept`, `category`). Because it needs module 7's output, the executor runs it after module 7 and appends its row to the same `correlation_summary.csv` that 4.1 and 4.2 already wrote, so the two "typicality vs. Jaccard" rows (human-based and model-based) can be compared directly in one file. Beyond the Jaccard-side missingness already described under 4.1, this panel loses a concept whenever module 7 drops its entire category for having fewer than two members with expert data, and it is judged against the same categorized-concepts denominator (module 7's extra per-category minimum is not itself subtracted from that denominator, so the reported coverage is, if anything, a slight underestimate of the true eligible set, never an overestimate).

**Generated data structures.** One CSV, one PNG, and one appended summary row:

- `jaccard_vs_cosine_typicality.csv`, columns `global_cosine_typicality` ($T^{\cos}_c$), `jaccard_pct` ($J_c$).

Example (head of `AP_0.6/4_correlations/jaccard_vs_cosine_typicality.csv` in `research_plots_150_revised_executor_again`):

| global_cosine_typicality | jaccard_pct |
|---|---|
| 0.5333 | 4.0175 |
| 0.4424 | 9.7765 |
| 0.3512 | 4.7722 |
| 0.5413 | 4.2105 |
| 0.4161 | 3.4924 |

- `jaccard_vs_cosine_typicality.png`, a **scatter plot with a linear regression line**, x: `global_cosine_typicality` ($T^{\cos}_c$, "Cosine Typicality"), y: `jaccard_pct` ($J_c$, "Jaccard Similarity Index %"), with the Pearson $r$/$p$ in the title.

### 4.4 Shannon entropy vs. expert count (both abstraction levels)

**Mathematical formulation.** This panel is the empirical check behind module 2's count-dependence caveat: entropy is bounded by the expert count ($H(c) \le \log_2 \min(n_c, L)$), so if entropy and expert count were strongly correlated in the analyzed range, any entropy comparison between groups with different typical counts (category labels hold systematically fewer experts than concepts) would partly measure set size rather than layer organization. The test is the pooled Pearson correlation of subchapter 4.1's form on the pair

$$x = H(c), \qquad y = n_c,$$

over *every* word in module 2's concepts descriptor table, which contains both abstraction levels, since category labels are words with expert distributions too. The two levels are separated only for display (words are classified by membership in the category list), while $r$ and $p$ are computed on the pooled sample. A near-zero $r$ licenses reading module 2's entropy contrasts as organizational, while a strong positive $r$ would flag them as count artifacts. A word is missing here only when module 2 itself dropped it for having zero experts across every layer, so this panel's coverage denominator is the full metadata count, the same one used for the metadata-only panels of 4.1.

**Generated data structures.** One CSV, one PNG, and one row in `correlation_summary.csv`:

- `shannon_entropy_vs_expert_count.csv`, the working table:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Word identifier. |
| group | string | | "Specific Concepts" or "Broad Categories" (display split). |
| shannon_entropy | float | $H(c)$ | Entropy of the word's layer distribution (module 2). |
| experts_count | int | $n_c$ | Number of expert rows behind the distribution. |

Example (head of `AP_0.6/4_correlations/shannon_entropy_vs_expert_count.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`):

| concept | group | shannon_entropy | experts_count |
|---|---|---|---|
| accountant | Specific Concepts | 4.4946 | 3277 |
| actor | Specific Concepts | 4.5028 | 1489 |
| airplane | Specific Concepts | 4.1070 | 717 |
| apple | Specific Concepts | 4.0195 | 160 |
| apricot | Specific Concepts | 3.9250 | 1975 |

- `shannon_entropy_vs_expert_count.png`, a **scatter plot with a linear regression line** over the pooled sample, x: `shannon_entropy` ($H(c)$), y: `experts_count` ($n_c$), with concepts drawn as small dots and category labels as larger diamonds, and the pooled Pearson $r$/$p$ in the title.

### 4.5 Jaccard vs. layer-profile agreement over all word pairs

**Research question.** Modules 3 and 5 now describe a word pair in two ways: the Jaccard index, which asks which neurons the two words share, and the layer-profile similarity of module 3, subchapter 3.2, which asks whether they spread their experts over the layers in the same proportions. This subchapter asks how much those two readings actually differ. If they are near-equivalent, the second carries little beyond the first. If they diverge, then two words can share neurons without agreeing on depth, or agree on depth while sharing no neuron, and the two are complementary descriptions.

Subchapter 4.1's `jaccard_vs_layer_profile` panel answers this on the 147 concept-to-parent pairs, the population every other panel of the module uses. This subchapter answers it on **every unordered pair of the $|\mathcal{C}|$ words**, $\binom{204}{2} = 20{,}706$ pairs on the Richie-HSJ set, which is the population any downstream model consuming both quantities as pairwise features would see.

**Mathematical formulation.** Both quantities are read off the flat pair vectors in $\texttt{triu}(|\mathcal{C}|, k=1)$ order, the Jaccard index from `pair_similarity_vector` and the two layer-profile readings from `pair_layer_profile_vectors`. Two panels are drawn, one against the raw similarity $S$ and one against its null z-score $z$, because $S$ is largely a readout of expert-set size while $z$ is what survives conditioning on it.

Spearman's $\rho$ is the headline statistic here rather than Pearson's $r$, and both are reported. The relation is expected to be monotone but not linear: the Jaccard index is strongly zero-inflated, with most pairs sharing no expert at all, while the layer-profile similarity saturates toward its ceiling, so a linear coefficient understates an association that the ranks capture cleanly. Writing $d_i$ for the difference in ranks of pair $i$ under the two measures, over $n$ pairs with no ties,

$$\rho = 1 - \frac{6\sum_{i=1}^{n} d_i^2}{n(n^2 - 1)} \in [-1, 1].$$

**Coverage.** The concept-level coverage gate of subchapter 4.1 does not apply, because the population is pairs rather than concepts. Instead `n_total_relevant` holds the total number of unordered pairs and `n_points` the number finite on both axes, so `coverage_pct` keeps its meaning: the share of pairs that could in principle have contributed and did. Pairs are lost only when a word holds fewer than 2 experts, which leaves it without a usable layer profile.

**Generated data structures.** Two CSVs and two PNGs. Each CSV holds the per-pair values actually plotted, with a `same_category` flag, so the pair-level data behind the correlation is inspectable and reusable:

- `jaccard_vs_layer_profile_allpairs.csv`, columns `jaccard_pct`, `layer_profile_similarity_pct`, `same_category`.
- `jaccard_vs_layer_profile_z_allpairs.csv`, columns `jaccard_pct`, `layer_profile_z`, `same_category`.

At twenty thousand points a scatter plot is a solid block of ink, so both PNGs use the density treatment module 8 established for the same problem: a log-scaled **hexbin** split into same-category and different-category panels, each carrying a binned median, a straight OLS fit, and a LOWESS smooth. Showing the OLS line against the LOWESS smooth is the point, since it puts the true, often saturating, trend next to the straight line a naive linear correlation would draw.

- `jaccard_vs_layer_profile_allpairs.png`, x: `jaccard_pct` ($J_{cd}$), y: `layer_profile_similarity_pct` ($S(c,d)$).
- `jaccard_vs_layer_profile_z_allpairs.png`, x: `jaccard_pct` ($J_{cd}$), y: `layer_profile_z` ($z(c,d)$).

Both add a row to `correlation_summary.csv`, which for these rows also carries the `spearman_rho` and `spearman_p` columns (left empty on every concept-level panel).

## Results

*Scope note.* Every figure in this section comes from the runs that predate the whole-model refactor, so it describes the **analysis sublayer** (`mlp.c_fc` for GPT-2, `mlp.gate_proj` for Qwen3), which is now one scope among several rather than the only one. Those numbers still stand, they are reproduced byte for byte by the corresponding `sublayers/<rank>_<sublayer>/` outputs. Whole-model and other-sublayer figures land here once the sweep is re-run.

Values are from the two Richie-HSJ sublayer runs, GPT-2 on `mlp.c_fc` (`research_plots_gpt2_richie_hsj_with_sublayer_analysis`) and Qwen3-1.7B on `mlp.gate_proj` (`research_plots_qwen_richie_hsj_with_sublayer_analysis`), each with 204 concepts, 8 category labels, and 196 concepts with a defined category.

**Main correlations, AP=0.5.**

| Relationship | GPT-2 (mlp.c_fc) r, p | Qwen3 (mlp.gate_proj) r, p |
|---|---|---|
| Frequency vs. Expert Count | -0.203, p=3.6e-3 | -0.176, p=1.2e-2 |
| Human Typicality vs. Expert Count | 0.036, p=0.62 (n.s.) | 0.007, p=0.92 (n.s.) |
| Frequency vs. Human Typicality | -0.026, p=0.72 (n.s.) | -0.026, p=0.72 (n.s.) |
| Human Typicality vs. Jaccard | 0.300, p=2.0e-5 | 0.381, p=3.5e-8 |
| Human Typicality vs. Overlap | (not run for GPT-2) | 0.381, p=3.7e-8 |
| Frequency vs. Jaccard | 0.059, p=0.41 (n.s.) | -0.060, p=0.40 (n.s.) |

Human-Typicality-vs-Jaccard is positive and significant in both models (r = 0.30 to 0.38), indicating that more typical concepts share more experts with their category label. Frequency-vs-expert-count is negative and modest (r ≈ -0.20, about 4% of variance). The remaining pairs are non-significant at AP=0.5, including frequency-vs-Jaccard (r ≈ 0.06 in GPT-2, r ≈ -0.06 in Qwen3).

**Concept expert count vs category alignment.** Using the raw expert-set sizes stored by module 3 alongside each Jaccard value:

| Relationship (concept's own expert set size vs alignment) | GPT-2 AP0.5 | GPT-2 AP0.6 | Qwen3 AP0.5 | Qwen3 AP0.6 |
|---|---|---|---|---|
| concept expert count vs Jaccard | -0.280, p=6.9e-5 | -0.334, p=1.8e-6 | -0.292, p=3.3e-5 | -0.378, p=4.7e-8 |
| concept expert count vs Overlap coefficient | 0.149, p=0.037 | 0.118, p=0.10 (n.s.) | 0.022, p=0.76 (n.s.) | 0.006, p=0.94 (n.s.) |

Concept expert count and Jaccard are negatively correlated in both models (r = -0.28 to -0.38, p < 1e-4), so a larger expert set is associated with lower Jaccard. The relationship is essentially unchanged when controlling for frequency (GPT-2 -0.288 to -0.282, Qwen3 -0.321 to -0.347), so it is not frequency-mediated.

This inverse relationship is a property of the Jaccard index rather than a semantic effect. Jaccard is $J = |A \cap B| / |A \cup B|$ with $A$ the concept's expert set and $B$ the category label's, and here the category label is the smaller and roughly fixed set (labels hold fewer experts than concepts). As the concept set $A$ grows, the union in the denominator grows with it while the intersection stays capped by the small label set $B$, so Jaccard falls by dilution. The overlap coefficient $|A \cap B| / \min(|A|, |B|)$, which normalizes the set-size disparity, shows no relationship with concept count (r = 0.0 to 0.15, non-significant), confirming the effect is arithmetic. The diagnostic figure `count_vs_jaccard_overlap.png` (produced separately, not part of the standard module output) shows the downward Jaccard trend in the left column flattening under the overlap coefficient in the right column, for both models.

**Typicality-alignment link and the size effect.** The Human-Typicality-vs-Jaccard result is independent of the Jaccard size effect. The partial correlation controlling for frequency is essentially unchanged (see the sweep table), and the relationship holds under the overlap coefficient (Qwen3 typicality-vs-overlap r=0.381 versus typicality-vs-Jaccard r=0.381 at AP=0.5). The typicality-alignment association is therefore size-independent.

### Across AP thresholds

Frequency-vs-expert-count and the three null relationships, both models:

| AP | GPT-2 Freq-ExpCount | Qwen3 Freq-ExpCount | GPT-2 Freq-Jaccard | Qwen3 Freq-Jaccard | Freq-Typ (both) |
|---|---|---|---|---|---|
| 0.5 | -0.203 (p=4e-3) | -0.176 (p=1e-2) | 0.059 (p=0.41, n.s.) | -0.060 (p=0.40, n.s.) | -0.026 (p=0.72, n.s.) |
| 0.6 | -0.262 (p=2e-4) | -0.215 (p=2e-3) | 0.106 (p=0.14, n.s.) | -0.020 (p=0.78, n.s.) | -0.026 (n.s.) |
| 0.7 | -0.302 (p=1e-5) | -0.229 (p=1e-3) | 0.123 (p=0.08, n.s.) | 0.040 (p=0.58, n.s.) | -0.026 (n.s.) |
| 0.8 | -0.297 (p=3e-5) | -0.203 (p=4e-3) | 0.238 (p=1e-2, n<75%) | -0.053 (p=0.46, n.s.) | -0.044 (n.s.) |
| 0.9 | -0.225 (p=2e-2, n<75%) | -0.037 (p=0.61, n.s.) | dropped (coverage) | dropped (coverage) | dropped (coverage) |

Frequency-vs-Jaccard is non-significant at every threshold in both models. The single nominal exception, GPT-2 at AP=0.8 (r=0.238, p=0.012), rests on 110 of 196 categorized concepts (56% coverage), below the 75% reporting bar, and is not reported under the coverage rule. Frequency-vs-typicality is constant at -0.026 across all thresholds within each model, since neither variable depends on the expert data and the threshold only changes which concepts survive the join.

The typicality-alignment panels, both models:

| AP | GPT-2 Typ-Jaccard | GPT-2 partial (ctrl freq) | GPT-2 Jaccard-CosineTyp | Qwen3 Typ-Jaccard | Qwen3 partial | Qwen3 Jaccard-CosineTyp |
|---|---|---|---|---|---|---|
| 0.5 | 0.300 (2e-5) | 0.302 (1.7e-5) | 0.527 (2e-15) | 0.381 (3e-8) | 0.381 (4e-8) | 0.641 (5e-24) |
| 0.6 | 0.301 (2e-5) | 0.305 (1.4e-5) | 0.343 (8e-7) | 0.376 (6e-8) | 0.376 (6e-8) | 0.568 (4e-18) |
| 0.7 | 0.206 (4e-3) | 0.211 (3e-3) | 0.155 (0.03) | 0.329 (2e-6) | 0.331 (2e-6) | 0.400 (7e-9) |
| 0.8 | 0.056 (0.56, n<75%) | 0.123 (0.20, n<75%) | 0.231 (0.02, n<75%) | 0.342 (9e-7) | 0.341 (1e-6) | 0.031 (0.67, n.s.) |

Typicality-vs-Jaccard is significant across AP 0.5 to 0.8 in Qwen3 and across AP 0.5 to 0.7 in GPT-2 (its AP=0.8 row rests on 110 concepts, below the coverage bar). The partial correlation controlling for frequency matches the raw value at every threshold, so frequency contributes negligibly. Qwen3 shows a stronger and more stable effect than GPT-2 across thresholds. The Jaccard-vs-Cosine-Typicality panel is high at AP=0.5 (r=0.53 GPT-2, 0.64 Qwen3) and decays with threshold, weak or absent by AP 0.7 to 0.8, so the model-derived centroid typicality tracks category alignment only at the most lenient threshold.

**Overlap and entropy panels.** Human-Typicality-vs-Overlap (run for Qwen3) matches typicality-vs-Jaccard at every threshold (AP=0.5 both 0.381, AP=0.8 overlap 0.304 versus Jaccard 0.342), consistent with the typicality-alignment link being independent of Jaccard's size sensitivity. Shannon-entropy-vs-expert-count is near zero at the lenient thresholds (Qwen3 r=0.114 at AP=0.5, r=0.021 at AP=0.6) and rises to r=0.233 (p=8e-4) by AP=0.8 as counts fall toward the $\log_2 n_c$ ceiling. Module 2's entropy contrasts are therefore count-independent at AP 0.5 to 0.6 but not at strict thresholds.

### Jaccard vs layer-profile agreement (subchapter 4.5)

Whole-model scope, Qwen3 on Richie-HSJ, over all 20,706 word pairs:

| AP | pairs used | coverage | Spearman $\rho$, $J$ vs $S$ | Pearson $r$, $J$ vs $S$ | Spearman $\rho$, $J$ vs $z$ |
|---|---|---|---|---|---|
| 0.5 | 20,706 | 100% | 0.440 | 0.368 | 0.386 |
| 0.6 | 20,706 | 100% | 0.379 | 0.281 | 0.332 |
| 0.7 | 20,706 | 100% | 0.330 | 0.213 | 0.261 |
| 0.8 | 20,301 | 98% | 0.296 | 0.171 | 0.210 |
| 0.9 | 17,391 | 84% | 0.146 | 0.102 | 0.114 |

The two metrics are **positively but weakly related, and they diverge further as the AP threshold tightens**, from $\rho = 0.44$ to $\rho = 0.15$. Sharing neurons and allocating experts to the same depths are therefore largely different things, which is precisely what makes the layer-profile reading worth computing rather than a restatement of the Jaccard index. At the loosest threshold the shared rank variance is about 19 percent, at the strictest about 2 percent.

Pearson sits consistently below Spearman, by 0.07 to 0.10, which is the expected signature of the non-linearity that motivated reporting the rank statistic as the headline. The hexbin panels show its source directly. The Jaccard axis is heavily zero-inflated, so a dense column of pairs sits at $J = 0$ spanning the entire range of layer-profile similarity, from roughly 30% to 80% at AP=0.6. Those are pairs that share no expert at all, which the Jaccard index cannot tell apart, and which the layer-profile metric separates across nearly its whole range. Above $J = 0$ the LOWESS smooth flattens well below the OLS line, so the linear fit substantially overstates the association at high Jaccard.

The same-category and different-category panels differ sharply. At AP=0.6 the within-panel OLS gives $r = 0.34$ while the across-panel gives $r = 0.18$, and the across-panel LOWESS is nearly flat above $J \approx 2\%$. For pairs drawn from different categories the two metrics are close to unrelated.

Substituting $z$ for $S$ lowers the correlation slightly at every threshold, by 0.03 to 0.07, so the small shared component between the Jaccard index and the layer profile is partly a common dependence on expert-set size, and removing that dependence makes the two readings more nearly independent still.

The concept-to-parent panel of subchapter 4.1 agrees in magnitude on its much smaller population, giving Pearson $r$ of 0.44, 0.28, 0.30 and 0.19 at AP=0.5 through 0.8, on 196, 196, 196 and 167 pairs respectively. At AP=0.9 coverage falls to 23% of categorized concepts, far below the 75% floor, so the correlation is withheld and the scatter is drawn unfitted, as designed.

## Conclusions

- Human Typicality is positively associated with category alignment in both models (typicality-vs-Jaccard r = 0.30 to 0.38 at AP=0.5), significant across the lenient-to-moderate thresholds, and this association is independent of both frequency (partial correlation unchanged) and Jaccard's set-size sensitivity (matched under the overlap coefficient).
- Frequency has at most a modest negative association with expert count (r ≈ -0.20 at AP=0.5, strengthening toward mid thresholds) and no association with category alignment or typicality.
- Concept expert count is negatively associated with Jaccard alignment (r = -0.28 to -0.38), but this is an arithmetic property of the Jaccard index for a fixed smaller comparison set and vanishes under the overlap coefficient, so it does not reflect a size-dependent tendency in category alignment.
- The model-derived Cosine Typicality tracks Jaccard alignment only at the most lenient threshold and decays with AP, unlike Human Typicality.
- Shannon entropy and expert count are decorrelated at AP 0.5 to 0.6, supporting module 2's use of entropy contrasts at those thresholds, and become correlated at strict thresholds.
- The Jaccard index and layer-profile agreement are only weakly related over all word pairs (Spearman 0.44 at AP=0.5 falling to 0.15 at AP=0.9), so which neurons two words share and how they distribute those neurons over depth are largely independent descriptions. The layer-profile reading is therefore a genuine addition to the pairwise feature set rather than a restatement of the Jaccard index, and the two diverge most exactly where module 5 shows the Jaccard index losing its category signal.
