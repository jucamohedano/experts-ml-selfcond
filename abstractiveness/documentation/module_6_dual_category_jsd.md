# Module 6: Category Prototype vs. Exemplar Divergence (JSD)

## Research question

Are categories better described as a single prototype distribution (the category-label word's own layer profile) or as the average of their member concepts' layer profiles, and does that choice matter more for some categories than others?

## Analysis

Notation. As in module 2, each item $c$ has a layer probability distribution $p_c \in [0,1]^L$ over the $L = 48$ layers, built from the shared `build_layer_probability_matrix` helper. For a category $k$ with valid members $M_k$ (members present in the matrix), the two competing representations are

$$P = p_k \;\; \text{(the label's own distribution, the \emph{prototype})}, \qquad Q = \bar{q}_k, \;\; \bar{q}_{k\ell} = \frac{1}{|M_k|}\sum_{m \in M_k} p_{m\ell} \;\; \text{(the member average, the \emph{exemplars})} .$$

Only categories whose label appears in the matrix *and* that have at least one valid member are processed. This requirement can fail wholesale at strict AP thresholds: module 1 (Results, "Zero-expert words") shows that entire words, including category labels, retain zero experts at AP ≥ 0.8, which is what shrinks this module's category count across the sweep. The module guards these degenerate cases explicitly. When *no* category qualifies it saves the (empty) CSV and skips all visualizations. The two scatter plots are also skipped when no row has both of their metrics, and the micro-distribution plot requires at least two categories with a valid JSD to contrast. This keeps a full AP sweep runnable end to end instead of crashing at the strict end.

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

At AP=0.6, categories vary a lot in how well their label represents their members: JSD ranges 4x across the 17 categories, from **0.064** (animal, well aligned) to **0.248** (furniture, poorly aligned). Of the two relationships this module tests, only one holds up at this threshold: **label entropy predicts divergence** (r=-0.60, p=0.011, n=17, visible in `scatter_jsd_vs_entropy.png` as a real if noisy downward trend), while **internal member diversity does not** (r=0.32, p=0.21, n=17, and `scatter_jsd_vs_member_diversity.png` shows a loose cloud with no visible trend).

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

JSD itself also grows roughly 7.5x in magnitude from AP=0.5 to AP=0.9 (mean 0.068 to 0.514), tracking the same "fewer experts, noisier and more divergent per-concept distributions" pattern seen in modules 1 to 3.
