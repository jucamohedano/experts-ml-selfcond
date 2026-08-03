# Module 6: Category Prototype vs. Exemplar Divergence (JSD)

## Research question

Are categories better described as a single prototype distribution (the category-label word's own layer profile) or as the average of their member concepts' layer profiles, and does that choice matter more for some categories than others?

## Analysis

Notation. As in module 2, each item $c$ has a layer probability distribution $p_c \in [0,1]^L$ over the $L = 48$ layers, built from the shared `build_layer_probability_matrix` helper. For a category $k$ with valid members $M_k$ (members present in the matrix), the two competing representations are

$$P = p_k \;\; \text{(the label's own distribution, the \emph{prototype})}, \qquad Q = \bar{q}_k, \;\; \bar{q}_{k\ell} = \frac{1}{|M_k|}\sum_{m \in M_k} p_{m\ell} \;\; \text{(the member average, the \emph{exemplars})} .$$

Only categories whose label appears in the matrix *and* that have at least one valid member are processed. This requirement can fail wholesale at strict AP thresholds: module 1 (Results, "Zero-expert words") shows that entire words, including category labels, retain zero experts at AP ≥ 0.8, which is what shrinks this module's category count across the sweep. The module guards these degenerate cases explicitly. When *no* category qualifies it saves the (empty) CSV and skips all visualizations. The two scatter plots are also skipped when no row has both of their metrics, and the micro-distribution plot requires at least two categories with a valid JSD to contrast. This keeps a full AP sweep runnable end to end instead of crashing at the strict end.

**Analysis scopes.** This module is *order-based*, meaning it reads the layer index as a depth coordinate, so it is sensitive to which layer axis it is given. It runs on the whole model first, writing into the module folder itself, then once per sublayer type into `sublayers/<rank>_<sublayer>/`. In the whole-model scope every output is produced twice, suffixed `_by_block` (the sublayers of each block summed into one depth bin, the canonical variant and the one downstream modules consume) and `_by_layer` (the flat interleaved axis at full resolution). A sublayer scope holds one layer per block, so block aggregation is an identity relabel there and only one unsuffixed variant is written. Module 1, subchapter 1.7 defines the scopes, the rank prefix, and why the flat whole-model axis is not a depth axis. A cross-scope summary lands in `sublayer_comparison.csv` and `sublayer_comparison.png` at the module's top level. Its columns are the mean label-versus-member-average divergence, the mean within-category member diversity, and the correlation between them. Read the divergence columns against `n_experts` on the same row: a sparse sublayer gives each word a thin layer distribution, which inflates JSD for reasons unrelated to prototype structure, so the sparsest scopes score highest. Module 1, subchapter 1.7 sets out that confound. Filenames listed below are given in their unsuffixed form, which is what a sublayer scope writes, and the whole-model scope inserts the axis suffix before the extension.

### 6.1 Prototype vs. exemplar divergence

**Mathematical formulation.** The dissimilarity between the two representations is measured with the Jensen–Shannon divergence. With the mixture $M = \tfrac{1}{2}(P + Q)$ and the Kullback–Leibler divergence in base 2,

$$D_{\mathrm{KL}}(P \,\|\, M) = \sum_{\ell=1}^{L} P_\ell \log_2 \frac{P_\ell}{M_\ell},$$

the Jensen–Shannon divergence is the symmetrized average

$$\mathrm{JSD}(P \,\|\, Q) = \tfrac{1}{2} D_{\mathrm{KL}}(P \,\|\, M) + \tfrac{1}{2} D_{\mathrm{KL}}(Q \,\|\, M) \; \in [0, 1] \; \text{(bits, base 2)} .$$

`scipy.spatial.distance.jensenshannon` returns the JS *distance* $\sqrt{\mathrm{JSD}}$, so the code squares it to store the divergence itself. Unlike KL, JSD is symmetric and always finite (a layer where only one of $P, Q$ has mass contributes finitely through $M$), and with base 2 it is bounded in $[0,1]$: 0 means the label's layer profile *is* the member average, 1 means they live on disjoint layers. Each category also carries the Shannon entropy of its label distribution, $H(P) = -\sum_\ell P_\ell \log_2 P_\ell$ (module 2's Definition A), and the mean Human Typicality of its valid members, $\bar{t}_k = \frac{1}{|M_k|}\sum_{m \in M_k} t_m$.

**Generated data structures.** One CSV (shared with subchapter 6.2) and one plot:

- `dual_category_jsd.csv`, one row per category, sorted ascending by `jensen_shannon_divergence`. The raw $P$/$Q$ arrays are kept in memory for the micro plot but dropped before saving.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| category | string | $k$ | Category label. |
| member_count | int | $|M_k|$ | Number of member concepts used in the average distribution. |
| avg_human_typicality | float | $\bar{t}_k$ | Mean human typicality of the category members. |
| shannon_entropy_label | float | $H(P)$ | Entropy of the category-label distribution. |
| jensen_shannon_divergence | float | $\mathrm{JSD}(P \| Q)$ | Divergence between the label distribution and the member-average distribution. |
| avg_member_diversity_jsd | float | $\bar{D}_k$ (see 6.2) | Mean pairwise divergence among the category's own member concepts (internal diversity). |

Example (head of `AP_0.6/6_dual_category_jsd/dual_category_jsd.csv` in `research_plots_150_revised_executor_again`):

| category | member_count | avg_human_typicality | shannon_entropy_label | jensen_shannon_divergence | avg_member_diversity_jsd |
|---|---|---|---|---|---|
| animal | 10 | 0.3596 | 4.5295 | 0.0642 | 0.1055 |
| drink | 9 | 0.5021 | 4.5192 | 0.0646 | 0.1283 |
| headwear | 3 | 0.3997 | 4.6060 | 0.0650 | 0.1627 |
| hardware | 9 | 0.4602 | 4.2198 | 0.0684 | 0.1709 |
| container | 11 | 0.5622 | 4.0633 | 0.0801 | 0.1585 |

- `micro_jsd_distributions.png`, two stacked **overlaid area charts** inspecting the extremes $\arg\min_k \mathrm{JSD}$ (top) and $\arg\max_k \mathrm{JSD}$ (bottom): the x-axis is the model layer $\ell$, the y-axis is probability, with one filled curve for the prototype $P_\ell$ (category label) and one for the exemplar average $Q_\ell$ (members). This shows the actual layer-wise allocations behind the smallest and largest divergence values.

### 6.2 Internal member diversity

**Mathematical formulation.** To test whether categories whose members are internally scattered also diverge more from their own label, each category's internal diversity is defined as the average pairwise JSD among its members (computed with `scipy.spatial.distance.pdist`, then squaring the returned distances):

$$\bar{D}_k = \binom{|M_k|}{2}^{-1} \sum_{\substack{m_i, m_j \in M_k \\ i < j}} \mathrm{JSD}\!\left(p_{m_i} \,\|\, p_{m_j}\right),$$

undefined (NaN) when $|M_k| < 2$. High $\bar{D}_k$ means the members themselves disagree about which layers matter, and the hypothesis under test is a positive relationship between $\bar{D}_k$ and $\mathrm{JSD}(P\|Q)$, since internally dispersed categories should be harder for a single label profile to represent. The test statistic is the Pearson correlation across categories, $r = \operatorname{corr}\big(\mathrm{JSD}(P\|Q),\, \bar{D}\big)$, annotated on the plot.

**Generated data structures.** The `avg_member_diversity_jsd` column ($\bar{D}_k$) of `dual_category_jsd.csv` above, and one plot:

- `scatter_jsd_vs_member_diversity.png`, a **scatter plot with a linear regression line**, x: `jensen_shannon_divergence` ($\mathrm{JSD}(P\|Q)$, "Label vs. Averaged Members Divergence"), y: `avg_member_diversity_jsd` ($\bar{D}_k$, "Average Pairwise Member Divergence"), one point per category (annotated with the category name), Pearson $r$/$p$ in the title.

### 6.3 Label concentration vs. divergence

**Mathematical formulation.** The second explanatory hypothesis is that how *concentrated* a label's own allocation is predicts how far it sits from its members' average: a label with a sharp, low-entropy profile (small $H(P)$) has all its mass in few layers, so unless the members peak in exactly those layers, the mixture comparison will find large gaps. The test is the Pearson correlation across categories between the two scalars defined in 6.1:

$$r = \operatorname{corr}\big(\mathrm{JSD}(P \,\|\, Q),\; H(P)\big),$$

with a *negative* $r$ meaning more concentrated labels (lower entropy) diverge more from their member average.

**Generated data structures.** No new CSV columns (both variables already live in `dual_category_jsd.csv`), only one plot:

- `scatter_jsd_vs_entropy.png`, a **scatter plot with a linear regression line**, x: `jensen_shannon_divergence` ($\mathrm{JSD}(P\|Q)$), y: `shannon_entropy_label` ($H(P)$), one point per category (annotated with the category name), Pearson $r$/$p$ in the title.

## Results

*Scope note.* The 17-category figures below come from the 150-concept run. The Richie-HSJ tables are whole-model scope on the block axis from the corrected `_sensefix` runs, where there are only **8 categories**, so every correlation in this module rests on n=8 and none of it should be read as more than suggestive.

In the 150-concept run at AP=0.6, categories vary a lot in how well their label represents their members: JSD ranges 4x across the 17 categories, from **0.064** (animal, well aligned) to **0.248** (furniture, poorly aligned). Of the two relationships this module tests, only one holds up at that threshold: **label entropy predicts divergence** (r=-0.60, p=0.011, n=17), while **internal member diversity does not** (r=0.32, p=0.21, n=17).

### Across AP thresholds, Richie-HSJ

| Model | AP | n cat. | JSD range | JSD mean | Entropy-vs-JSD (r, p) | Diversity-vs-JSD (r, p) |
|---|---|---|---|---|---|---|
| GPT-2 | 0.5 | 8 | 0.008–0.036 | 0.019 | -0.66 (0.076) | +0.90 (0.002) |
| GPT-2 | 0.6 | 8 | 0.025–0.134 | 0.061 | -0.65 (0.084) | +0.41 (0.32) |
| GPT-2 | 0.7 | 8 | 0.054–0.546 | 0.239 | -0.88 (0.004) | -0.42 (0.31) |
| GPT-2 | 0.8 | 5 | 0.196–0.702 | 0.376 | -0.66 (0.22) | +0.04 (0.95) |
| GPT-2 | 0.9 | n/a | no output | | | |
| Qwen3 | 0.5 | 8 | 0.014–0.055 | 0.028 | -0.70 (0.055) | +0.00 (0.99) |
| Qwen3 | 0.6 | 8 | 0.031–0.113 | 0.054 | -0.61 (0.11) | +0.33 (0.42) |
| Qwen3 | 0.7 | 8 | 0.047–0.171 | 0.106 | -0.04 (0.93) | +0.44 (0.28) |
| Qwen3 | 0.8 | 8 | 0.118–0.810 | 0.373 | -0.91 (0.002) | -0.22 (0.60) |
| Qwen3 | 0.9 | 3 | 0.500–0.819 | 0.643 | -0.99 (0.084) | -0.88 (0.32) |

**Label entropy and divergence are consistently negatively related**, in the same direction as the 150-concept run and in both architectures. The coefficient is between -0.61 and -0.91 in seven of the nine usable rows, reaching significance at GPT-2 AP 0.7 (r=-0.88, p=0.004) and Qwen3 AP 0.8 (r=-0.91, p=0.002). With n=8 the test has little power, so the consistency of the sign across models and thresholds is better evidence than any individual p value. A category whose label word spreads its experts widely over depth diverges less from its member average, which is the reading the 150-run supported.

**Internal member diversity again fails to predict divergence.** The sign flips across thresholds in both models (+0.90, +0.41, -0.42, +0.04 for GPT-2, and +0.00, +0.33, +0.44, -0.22, -0.88 for Qwen3), and the only significant cell, GPT-2 at AP 0.5, is the one where the JSD range is narrowest, 0.008 to 0.036, so it rests on differences too small to be meaningful. The hypothesis that more internally scattered categories diverge further from their own label has no support here, matching the 150-run conclusion.

JSD grows about 20x in magnitude from AP=0.5 to the strictest usable threshold in both models (GPT-2 0.019 to 0.376, Qwen3 0.028 to 0.643), tracking the same "fewer experts, noisier and more divergent per-concept distributions" pattern seen in modules 1 to 3. Note also how the sample dies: GPT-2 falls to 5 categories at AP 0.8 and produces nothing at AP 0.9, while Qwen3 keeps all 8 at AP 0.8 and 3 at AP 0.9.
