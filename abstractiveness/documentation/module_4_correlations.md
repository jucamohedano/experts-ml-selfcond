# Module 4: Frequency, Typicality, and Expert-Count Correlations

## Research question

Are concepts that are more frequent, more typical (per human judgment), or more category-aligned (by Jaccard similarity) also associated with more expert units or stronger category overlap? And where typicality and category alignment are related, does that relationship survive controlling for word frequency, and does the model-derived Cosine Typicality (module 7) track category alignment as well as Human Typicality does?

## Analysis

Notation. For each concept $c$ the module works with the scalar variables assembled by module 1 and module 3: the log-frequency $\tilde{f}_c = \log_{10} f_c$, the Human Typicality score $t_c$, the expert count $n_c$, and the Jaccard similarity to the parent category $J(c,k)$ (written $J_c$ below, since each concept has one parent). The base table is module 1's `expert_counts_with_metadata.csv`, inner-joined with module 3's `category_concept_similarity_metrics.csv` on (`concept`, `category`) for the Jaccard-based panels.

### 4.1 Bivariate Pearson regression panels

**Mathematical formulation.** Five variable pairs $(x, y)$ are each tested for a linear association. For a pair with $n$ complete observations $(x_i, y_i)$ (rows with a missing value in either variable are dropped per panel), the Pearson correlation coefficient is

$$r_{xy} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2}\;\sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}} \in [-1, 1],$$

with the two-sided p-value obtained from the exact null distribution used by `scipy.stats.pearsonr` (equivalently, from the statistic $t = r\sqrt{(n-2)/(1-r^2)}$ under a $t_{n-2}$ reference). $r_{xy}$ measures only the *linear* component of the relationship, while $r_{xy}^2$ is the fraction of variance in $y$ that a linear function of $x$ would explain, which is the effect-size reading used in the Results. Each panel also fits and draws the ordinary-least-squares line $\hat{y} = \beta x + \alpha$ with $\beta = \operatorname{Cov}(x,y)/\operatorname{Var}(x)$, $\alpha = \bar{y} - \beta\bar{x}$ (rendered by `seaborn.regplot` with its confidence band). The five pairs, in terms of the symbols above:

| Panel | $x$ | $y$ | Tests whether… |
|---|---|---|---|
| frequency_vs_expert_count | $\tilde{f}_c$ | $n_c$ | more frequent words recruit more experts |
| typicality_vs_expert_count | $t_c$ | $n_c$ | more typical concepts recruit more experts |
| frequency_vs_typicality | $\tilde{f}_c$ | $t_c$ | frequent words are judged more typical (no expert data involved) |
| typicality_vs_jaccard | $t_c$ | $J_c$ | more typical concepts share more experts with their category |
| frequency_vs_jaccard | $\tilde{f}_c$ | $J_c$ | more frequent words share more experts with their category |

Every panel runs through a single shared helper (`_run_regression_panel`) that drops missing values for the relevant pair, saves the cleaned subset to CSV, draws the scatter + regression plot with the Pearson $r$/$p$ annotated in the title if more than two points remain, and saves the PNG, all within one call, so every panel's CSV and PNG are always written together.

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

#### 5. frequency_vs_jaccard.csv, columns `log_frequency` ($\tilde{f}_c$), `jaccard_pct` ($J_c$)

Example (head of `AP_0.6/4_correlations/frequency_vs_jaccard.csv`):

| log_frequency | jaccard_pct |
|---|---|
| 4.2468 | 8.3994 |
| 3.7168 | 4.0175 |
| 2.4362 | 1.0776 |
| 3.4951 | 0.4454 |
| 4.4427 | 2.2727 |

#### 6. correlation_summary.csv

One row per panel run in this module (including subchapters 4.2 and 4.3), recording the Pearson statistic computed on the corresponding $(x, y)$ columns.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| plot_name | string | (none) | Identifier of the corresponding regression/plot (matches the `.png` file stem). |
| x_variable | string | $x$ | Name of the column used on the x-axis. |
| y_variable | string | $y$ | Name of the column used on the y-axis. |
| pearson_r | float | $r_{xy}$ | Pearson correlation coefficient between x_variable and y_variable. |
| pearson_p | float | $p$ | Two-sided p-value for the Pearson correlation test. |
| n_points | int | $n$ | Number of $(x, y)$ pairs used after dropping missing values. |
| controlling_for | string | $z$ | Control variable, only populated for the partial-correlation row (4.2), and empty otherwise. |

Example (head of `AP_0.6/4_correlations/correlation_summary.csv` in `research_plots_150_revised_executor_again`):

| plot_name | x_variable | y_variable | pearson_r | pearson_p | n_points | controlling_for |
|---|---|---|---|---|---|---|
| frequency_vs_expert_count | log_frequency | expert_count | -0.5521 | 1.81e-14 | 164 | |
| typicality_vs_expert_count | human_typicality | expert_count | -0.2581 | 1.60e-03 | 147 | |
| frequency_vs_typicality | log_frequency | human_typicality | 0.4364 | 3.29e-08 | 147 | |
| typicality_vs_jaccard | human_typicality | jaccard_pct | 0.4429 | 1.95e-08 | 147 | |
| frequency_vs_jaccard | log_frequency | jaccard_pct | 0.3706 | 3.84e-06 | 147 | |

**Plots.** Each panel is a **scatter plot with a linear regression line** (`seaborn.regplot`, including the shaded confidence band) and the Pearson $r$/$p$ annotated in the title:

- `frequency_vs_expert_count.png`, x: `log_frequency` ($\tilde{f}_c$), y: `expert_count` ($n_c$).
- `typicality_vs_expert_count.png`, x: `human_typicality` ($t_c$), y: `expert_count` ($n_c$).
- `frequency_vs_typicality.png`, x: `log_frequency` ($\tilde{f}_c$), y: `human_typicality` ($t_c$).
- `typicality_vs_jaccard.png`, x: `human_typicality` ($t_c$), y: `jaccard_pct` ($J_c$).
- `frequency_vs_jaccard.png`, x: `log_frequency` ($\tilde{f}_c$), y: `jaccard_pct` ($J_c$).

### 4.2 Partial correlation: Jaccard vs. typicality, controlling for frequency

**Mathematical formulation.** Panels 3–5 of subchapter 4.1 show that frequency correlates with *both* typicality and Jaccard similarity, so their pairwise correlation could be a frequency artifact: frequent words might be both judged more typical and better aligned with their category, without typicality and alignment being directly related. To isolate the direct component, both variables are residualized against the control $z_c = \tilde{f}_c$. For a variable $v$ regressed on $z$ by simple OLS,

$$\beta_v = \frac{\operatorname{Cov}(z, v)}{\operatorname{Var}(z)}, \qquad \alpha_v = \bar{v} - \beta_v \bar{z}, \qquad \tilde{v}_c = v_c - (\beta_v z_c + \alpha_v),$$

the residual $\tilde{v}_c$ is the part of $v_c$ that a linear function of frequency cannot explain. Applying this to both variables gives the residual vectors $\tilde{J}_c$ (`jaccard_resid`) and $\tilde{t}_c$ (`typicality_resid`), and the partial correlation is the plain Pearson correlation of the residuals:

$$r_{Jt \cdot f} = \operatorname{corr}\!\left(\tilde{J}, \tilde{t}\right),$$

which is algebraically identical to the standard partial-correlation formula $r_{Jt\cdot f} = \dfrac{r_{Jt} - r_{Jf}\, r_{tf}}{\sqrt{(1 - r_{Jf}^2)(1 - r_{tf}^2)}}$. If $r_{Jt\cdot f}$ stays significantly positive, the typicality–alignment link is not just frequency in disguise.

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

over the inner join of module 3's similarity table with module 7's `global_prototype_typicality.csv` on (`concept`, `category`). Because it needs module 7's output, the executor runs it after module 7 and appends its row to the same `correlation_summary.csv` that 4.1 and 4.2 already wrote, so the two "typicality vs. Jaccard" rows (human-based and model-based) can be compared directly in one file.

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

## Results

At AP=0.6, all five correlations are statistically significant (p<0.001 in four of five, p=0.0016 in the fifth), but "significant" and "strong" are not the same thing here: with 147–164 points, even modest correlations clear the significance bar. Squaring each $r$ to get variance explained tells the more honest story:

| Relationship | r | r² (variance explained) |
|---|---|---|
| Frequency vs. Expert Count | -0.55 | 30% |
| Human Typicality vs. Jaccard | 0.44 | 20% |
| Frequency vs. Human Typicality | 0.44 | 19% |
| Frequency vs. Jaccard | 0.37 | 14% |
| Human Typicality vs. Expert Count | -0.26 | 7% |

Only the frequency-vs-expert-count relationship explains close to a third of the variance, and it's the one that's also visually obvious in `frequency_vs_expert_count.png`, a clear funnel-shaped downward trend, even though individual points still scatter by a factor of 2-3x around the line. Typicality-vs-Jaccard sits second (r=0.44, 20% of variance, close behind frequency-vs-typicality). Even so, 20% of variance is a diffuse relationship, and the scatter in `typicality_vs_jaccard.png` remains a broad cloud around the fitted line rather than a tight trend.

The two panels added by subchapters 4.2 and 4.3 sharpen this picture at AP=0.6, in opposite directions. The **partial correlation** between Jaccard similarity and Human Typicality, controlling for frequency, is $r_{Jt\cdot f}$=0.34 (p=3.1e-5, n=147). This is smaller than the raw r=0.44, so frequency does inflate the raw relationship, but it is still clearly significant, meaning the link between typicality and alignment is *not* just frequency in disguise. **Cosine Typicality, by contrast, only marginally predicts Jaccard** (r=0.16, p=0.057, not significant), so Human Typicality, which is measured entirely independently of the expert data, tracks a concept's category alignment far better than the model's own centroid-based typicality score does at this threshold.

### Across AP thresholds

| AP | Freq vs. ExpertCount (r) | Typ vs. ExpertCount (r) | Freq vs. Typ (r) | Typ vs. Jaccard (r, p) | Freq vs. Jaccard (r, p) |
|---|---|---|---|---|---|
| 0.5 | -0.56 | -0.23 | 0.44 | 0.43 (p=4e-8) | 0.42 (p=1e-7) |
| 0.6 | -0.55 | -0.26 | 0.44 | 0.44 (p=2e-8) | 0.37 (p=4e-6) |
| 0.7 | -0.52 | -0.26 | 0.44 | 0.36 (p=7e-6) | 0.29 (p=4e-4) |
| 0.8 | -0.47 | -0.25 | 0.43 | 0.30 (p=7e-4) | 0.20 (p=0.02) |
| 0.9 | -0.44 | -0.21 | 0.42 | 0.16 (p=0.24, n.s.) | 0.21 (p=0.10, n.s.) |

Two patterns stand out. First, **frequency vs. human typicality is essentially invariant to AP** (r=0.44, 0.44, 0.44, 0.43, 0.42), which makes sense, since neither variable depends on expert allocation at all. The AP threshold only changes which concepts have a defined `expert_count` or `jaccard_pct` and therefore survive the join, not the frequency or typicality values themselves. This is a useful sanity check, as relationships that do not mechanically involve expert counts should be (and are) stable across thresholds.

Second, **the two Jaccard-based relationships are strongest at the lenient end and decay monotonically as AP tightens**, losing significance entirely only at AP=0.9 (typicality-vs-jaccard p=0.24, frequency-vs-jaccard p=0.10). Given that module 3 shows the median concept-category pair has *zero* shared experts at AP=0.8 to 0.9, the AP=0.9 collapse is not surprising, since once jaccard_pct is mostly zero there is little variance left for it to correlate with anything. Both are cleanly significant across AP 0.5 to 0.8, which makes them nearly as dependable as the expert-count relationships everywhere except the strictest threshold.

The threshold sweep for the two added panels:

| AP | Partial: Jaccard vs. Typ controlling Freq (r, p, n) | Jaccard vs. Cosine Typicality (r, p, n) |
|---|---|---|
| 0.5 | 0.31 (p=1.5e-4, 147) | 0.28 (p=5.3e-4, 147) |
| 0.6 | 0.34 (p=3.1e-5, 147) | 0.16 (p=0.057, n.s., 147) |
| 0.7 | 0.27 (p=8.1e-4, 147) | 0.09 (p=0.29, n.s., 147) |
| 0.8 | 0.24 (p=0.007, 128) | 0.07 (p=0.44, n.s., 128) |
| 0.9 | 0.07 (p=0.62, n.s., 59) | -0.04 (p=0.75, n.s., 59) |

The **partial correlation tracks the raw typicality-vs-jaccard row closely at every threshold** (significant at AP 0.5 to 0.8, gone at AP=0.9), sitting consistently several points below it, so the frequency-corrected conclusion matches the raw one wherever the raw relationship exists at all, real but modest, and gone once Jaccard variance collapses at the strictest threshold. The **Jaccard-vs-Cosine-Typicality relationship is the module's weakest**, significant only at AP=0.5 (r=0.28) and marginal to null everywhere else. The fully independent Human Typicality is a *better* predictor of a concept's category alignment than the model's own Cosine Typicality at every threshold, an instructive result given module 7's finding that Cosine Typicality does not agree with human judgments pointwise either.
