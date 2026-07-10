# Module 2: Shannon Entropy and Peak/Average Layer Allocation

## Research question

Are specific concepts more sharply localized in a few layers than the broader semantic categories that contain them, i.e. is expert allocation more concentrated at the concept level than at the category level?

## Analysis

Notation. Let $L = 48$ be the number of model layers. The module starts from the concept-by-layer count matrix $N \in \mathbb{N}^{|\mathcal{C}| \times L}$ (built by the shared `build_layer_probability_matrix` helper, also used by module 6), where $N_{c\ell}$ counts the expert rows of item $c$ in layer $\ell$, reindexed to all $L$ columns so layers with zero surviving experts still appear. Row-normalizing gives each item's layer probability distribution

$$p_{c\ell} = \frac{N_{c\ell}}{\sum_{\ell'=1}^{L} N_{c\ell'}}, \qquad p_c = (p_{c1}, \dots, p_{cL}), \quad \sum_{\ell} p_{c\ell} = 1 .$$

For a category $k$, $M_k$ denotes its member concepts (restricted to those present in the matrix).

### 2.1 Concept-level Shannon entropy, peak layer, and average layer

**Mathematical formulation.** For each item $c$, three summary statistics of the distribution $p_c$ are computed. The Shannon entropy (via `scipy.stats.entropy`, base 2, in bits) measures how spread out the allocation is:

$$H(c) = -\sum_{\ell=1}^{L} p_{c\ell}\,\log_2 p_{c\ell},$$

with the convention $0 \log_2 0 = 0$. Entropy is minimal ($H = 0$) when all experts sit in a single layer, and maximal ($H = \log_2 L \approx 5.585$ bits) when the allocation is uniform across all 48 layers, so *lower entropy = more concentrated*. The peak layer is the mode of the distribution,

$$\ell^{\ast}_c = \arg\max_{\ell} \; p_{c\ell},$$

and the average layer is its expected value, treating the layer index as a numeric position in the network:

$$\bar{\ell}_c = \sum_{\ell=1}^{L} \ell \cdot p_{c\ell}.$$

$\ell^{\ast}_c$ says where the single strongest concentration sits, while $\bar{\ell}_c$ says where the distribution's center of mass sits, which can differ substantially when the distribution is multi-modal or skewed.

**Generated data structures.** One CSV, `shannon_entropy_concepts.csv`, with one row per item:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Concept identifier. |
| shannon_entropy | float | $H(c)$ | Entropy of the concept's layerwise expert distribution. |
| peak_layer | int | $\ell^{\ast}_c$ | Layer where the concept has its strongest expert concentration. |
| avg_layer | float | $\bar{\ell}_c$ | Expected layer value for the concept distribution. |
| experts_count | int | $n_c$ | Number of expert rows used to compute the distribution. |

Example (head of `AP_0.6/2_shannon_entropy_analysis/shannon_entropy_concepts.csv` in `research_plots_150_revised_executor_again`):

| concept | shannon_entropy | peak_layer | avg_layer | experts_count |
|---|---|---|---|---|
| airplane | 4.5767 | 47 | 25.8547 | 523 |
| alligator | 5.0540 | 15 | 26.3125 | 960 |
| animal | 4.5295 | 3 | 20.7489 | 231 |
| anklet | 4.5219 | 3 | 17.9093 | 397 |
| anvil | 4.9849 | 19 | 24.5018 | 831 |

### 2.2 Dual category definitions: label distribution vs. member average

**Mathematical formulation.** Each category $k$ is represented in two independent ways, and the same three statistics of subchapter 2.1 are computed for both:

- *Definition A (the label itself)*: the category label is a word with its own row in the matrix, so its distribution is simply $p_k$, giving $H(p_k)$, $\ell^{\ast}_k$, $\bar{\ell}_k$.
- *Definition B (average of members)*: the members' distributions are averaged element-wise,

$$\bar{q}_{k\ell} = \frac{1}{|M_k|} \sum_{m \in M_k} p_{m\ell}, \qquad \bar{q}_k = (\bar{q}_{k1}, \dots, \bar{q}_{kL}),$$

which is again a valid probability distribution (a uniform mixture of the members), giving $H(\bar{q}_k)$, $\ell^{\ast}_{\bar{q}_k}$, $\bar{\ell}_{\bar{q}_k}$. Because entropy is concave, mixing distributions can only preserve or increase entropy relative to the average of the members' entropies, so averaging tends to *wash out* concentration unless all members peak in the same layers.

To quantify whether the two definitions at least *point at the same layers*, the Pearson correlation between the two $L$-dimensional vectors is computed:

$$r_k = \operatorname{corr}(p_k, \bar{q}_k) = \frac{\sum_{\ell}(p_{k\ell} - \overline{p_k})(\bar{q}_{k\ell} - \overline{\bar{q}_k})}{\sqrt{\sum_{\ell}(p_{k\ell} - \overline{p_k})^2}\sqrt{\sum_{\ell}(\bar{q}_{k\ell} - \overline{\bar{q}_k})^2}},$$

set to 0 when either vector has zero variance. High $r_k$ means the label's allocation shape mirrors the members' average shape, even if their entropies differ.

**Generated data structures.** One CSV, `shannon_entropy_categories.csv`, with one row per category, and one plot:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| category | string | $k$ | Category label. |
| shannon_entropy | float | $H(p_k)$ | Entropy of the category-label distribution (Definition A). |
| peak_layer | int | $\ell^{\ast}_k$ | Layer of the category-label peak. |
| avg_layer | float | $\bar{\ell}_k$ | Expected layer of the category-label distribution. |
| shannon_entropy_average | float | $H(\bar{q}_k)$ | Entropy of the average member distribution (Definition B). |
| peak_layer_average | int | $\ell^{\ast}_{\bar{q}_k}$ | Peak layer of the average member distribution. |
| avg_layer_average | float | $\bar{\ell}_{\bar{q}_k}$ | Expected layer of the average member distribution. |
| pearson_correlation_distributions | float | $r_k$ | Correlation between the label distribution and the member-average distribution. |
| member_count | int | $|M_k|$ | Number of valid members contributing to the category summary. |

Example (head of `AP_0.6/2_shannon_entropy_analysis/shannon_entropy_categories.csv` in `research_plots_150_revised_executor_again`):

| category | shannon_entropy | peak_layer | avg_layer | shannon_entropy_average | peak_layer_average | avg_layer_average | pearson_correlation_distributions | member_count |
|---|---|---|---|---|---|---|---|---|
| animal | 4.5295 | 3 | 20.7489 | 4.8733 | 3 | 22.5487 | 0.8996 | 10 |
| clothing | 3.7835 | 38 | 27.6000 | 4.8622 | 3 | 23.0316 | 0.7791 | 10 |
| container | 4.0633 | 3 | 15.1250 | 4.6756 | 3 | 18.5659 | 0.9154 | 11 |
| drink | 4.5192 | 1 | 21.4049 | 4.7349 | 3 | 23.3514 | 0.7940 | 9 |
| fastener | 4.8612 | 15 | 16.9444 | 4.5435 | 3 | 18.0433 | 0.5227 | 5 |

- `category_shannon_entropies_bar.png`, a horizontal **bar chart** of `shannon_entropy` ($H(p_k)$, x-axis) per `category` ($k$, y-axis), sorted descending, ranking categories by how distributed (top) versus localized (bottom) their label's expert allocation is.

### 2.3 Concept-vs-category concentration test (Mann–Whitney U)

**Mathematical formulation.** The research question is a comparison of two samples of entropies: $\{H(c) : c \in \mathcal{C}_2\}$ (concepts) versus $\{H(p_k) : k\}$ (category labels). Because entropies are bounded and not normally distributed, a non-parametric one-sided Mann–Whitney U test is used with the alternative hypothesis "category entropies are stochastically *smaller* than concept entropies" (`alternative='less'`), i.e. categories are more concentrated. Writing $n_1, n_2$ for the two sample sizes and $R_1$ for the sum of ranks of the category sample in the pooled ranking,

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1 ,$$

and the p-value is the probability, under the null of identical distributions, of a $U$ at least as extreme in the "less" direction. A significant result means category labels' allocations are systematically more concentrated (lower $H$) than concepts', not just different in shape.

**Generated data structures.** The $U$ statistic and p-value are logged to `main.log` (not saved as CSV). The comparison is visualized in:

- `category_concept_shannon_entropies.png`, a **violin plot** (with quartile lines at Q1/median/Q3) overlaid with a **strip plot** of individual points, comparing `shannon_entropy` ($H$, y-axis) across the two groups `group_type` = Specific Concepts ($H(c)$, $c \in \mathcal{C}_2$) vs. Broad Categories ($H(p_k)$) on the x-axis.

### 2.4 Peak vs. average layer distributions across groups

**Mathematical formulation.** This analysis compares *where* in the network the three representations from 2.1–2.2 place their mass, using the per-item summary positions rather than the full distributions. Three groups $g$ are formed: Specific Concepts ($\ell^{\ast}_c$, $\bar{\ell}_c$ for $c \in \mathcal{C}_2$), Broad Categories ($\ell^{\ast}_k$, $\bar{\ell}_k$), and Broad Categories (Avg) ($\ell^{\ast}_{\bar{q}_k}$, $\bar{\ell}_{\bar{q}_k}$). For the bars, each layer $\ell$ receives the percentage of group members peaking there:

$$\text{peak\%}_g(\ell) = 100 \cdot \frac{|\{c \in g : \ell^{\ast}_c = \ell\}|}{|g|},$$

which sums to 100 within each group and makes groups of very different sizes (150 concepts vs. 17 categories) comparable. For the curves, a Gaussian kernel density estimate is fitted per group to the set $\{\bar{\ell}_c : c \in g\}$ of average-layer positions (normalized within each group, `common_norm=False`), showing where each group's centers of mass congregate along the depth axis.

**Generated data structures.** One plot (no standalone CSV, since the inputs are the `peak_layer` and `avg_layer` columns of the two CSVs above):

- `peak_average_layers.png`, a combined **bar chart + KDE overlay** on twin y-axes: bars show $\text{peak\%}_g(\ell)$ (left y-axis) against the model layer (x-axis, from `peak_layer`), while the overlaid density curves show the KDE of `avg_layer` ($\bar{\ell}$, right y-axis). Both are colored and grouped by `group_type` (Specific Concepts, Broad Categories, Broad Categories (Avg)).

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
3. **Label-vs-member-average shape correlation weakens monotonically** as AP tightens (0.83, 0.82, 0.73, 0.62, and 0.59 across AP 0.5 to 0.9). At lenient thresholds the label and its members peak in very similar places, whereas at stricter thresholds they diverge more, consistent with there being fewer, noisier experts per concept to average over.

The *relative* size of the concept-vs-category gap also grows sharply with AP. At AP=0.5 concepts and categories differ by only 0.17 bits (4.83 against 4.66), and by AP=0.9 they differ by 0.89 bits on a much smaller overall scale (1.94 against 1.05), so categories collapse toward near-total concentration faster than concepts do as the threshold strips away marginal experts.
