# Module 2: Shannon Entropy and Layer-Distribution Descriptors

## Research question

Are specific concepts more sharply localized in a few layers than the broader semantic categories that contain them, i.e. is expert allocation more concentrated at the concept level than at the category level? And beyond concentration alone: do the two abstraction levels differ in *where* along the depth axis their experts sit, in how *smooth* their depth profiles are, and in how *unambiguous* their peak layers are?

## Analysis

Notation. Let $L$ be the number of layers in the analyzed support. Module 2 receives the expert table as restricted by module 1's sublayer filter, so in the sublayer-restricted runs the support is one projection type (e.g. the 28 `mlp.gate_proj` layers of Qwen3-1.7B, or the 12 `mlp.c_fc` layers of GPT-2), while layers keep their absolute whole-model indices $\ell$ (1…196 resp. 1…48), non-contiguous within the support. The module starts from the concept-by-layer count matrix $N$ (built by the shared `build_layer_probability_matrix` helper, also used by module 6), where $N_{c\ell}$ counts the expert rows of item $c$ in layer $\ell$, reindexed to the full support so layers with zero surviving experts still appear. Row-normalizing gives each item's layer probability distribution

$$p_{c\ell} = \frac{N_{c\ell}}{\sum_{\ell'} N_{c\ell'}}, \qquad p_c = (p_{c\ell})_{\ell \in \text{support}}, \quad \sum_{\ell} p_{c\ell} = 1 .$$

For a category $k$, $M_k$ denotes its member concepts (restricted to those present in the matrix).

Two structural facts about the underlying data qualify every comparison below (both are established quantitatively in module 1, subchapter 1.1). First, **expert sets overlap across words**: a single neuron is typically an expert for several words (54.9% of expert neurons serve ≥ 2 of the 204 words in the Qwen3 run at AP=0.6, and 63.7% within `mlp.gate_proj`), so the distributions $p_c$ of different words are not built from disjoint neuron populations. Comparing two words' entropies therefore compares the *shapes* of their allocations over depth, not how they divide a fixed pool of neurons between them. Second, **expert counts differ systematically between the groups being compared**, addressed in the dedicated paragraph in subchapter 2.1 below.

### 2.1 Per-word layer-distribution descriptors

**Mathematical formulation.** For each item $c$ (every word with at least one expert, including category labels, since they are words with expert sets too), six scalar descriptors of the distribution $p_c$ are computed by the shared `layer_distribution_descriptors` helper. The Shannon entropy (via `scipy.stats.entropy`, base 2, in bits) measures how spread out the allocation is:

$$H(c) = -\sum_{\ell} p_{c\ell}\,\log_2 p_{c\ell},$$

with the convention $0 \log_2 0 = 0$. Entropy is minimal ($H = 0$) when all experts sit in a single layer, and maximal ($H = \log_2 L$, e.g. $\log_2 28 \approx 4.807$ bits on the Qwen3 `mlp.gate_proj` support) when the allocation is uniform, so *lower entropy = more concentrated*. Entropy is invariant under any permutation of the layers, so it sees concentration but is blind to depth order. Its order-sensitive complement is **Geary's C**, the adjacent-layer spatial autocorrelation of the profile in depth order:

$$C(c) = \frac{\sum_{i} \left(p_{c,\ell_{i+1}} - p_{c,\ell_i}\right)^2}{2 \sum_{i} \left(p_{c,\ell_i} - \overline{p_c}\right)^2},$$

where $\ell_1 < \ell_2 < \dots$ enumerate the support in depth order. $C \approx 1$ means no depth structure (neighbors no more alike than random), $C < 1$ a smooth/clumped profile (neighboring layers alike, mass in contiguous bands), $C > 1$ a jagged, alternating profile. It is scale-invariant (counts and shares give the same value) and NaN for degenerate inputs (fewer than 3 layers or zero variance). Two words can have identical entropy but opposite C, one broad contiguous hump versus mass scattered in isolated spikes, which is exactly the distinction the shape map of subchapter 2.5 displays.

The remaining four descriptors summarize location. The peak layer is the mode of the distribution,

$$\ell^{\ast}_c = \arg\max_{\ell} \; p_{c\ell},$$

and its reliability is quantified by the **peak dominance gap**, the percentage-point lead of the peak over the runner-up layer:

$$\gamma_c = 100\,\big(p_{(1)} - p_{(2)}\big),$$

with $p_{(1)} \ge p_{(2)}$ the two largest shares. When $\gamma_c$ is near zero the "peak layer" is effectively a tie, and its location is not a trustworthy statistic, so the plots of subchapters 2.4 and 2.7 use the threshold $\gamma < 1$ pp to mark such ambiguous peaks. The average layer is the distribution's center of mass, treating the absolute layer index as a numeric depth position:

$$\bar{\ell}_c = \sum_{\ell} \ell \cdot p_{c\ell},$$

and its trimmed variant restricts the average to the $\lceil L/4 \rceil$ most-loaded layers, renormalized:

$$\bar{\ell}^{\,25\%}_c = \frac{\sum_{\ell \in T_c} \ell \cdot p_{c\ell}}{\sum_{\ell \in T_c} p_{c\ell}}, \qquad T_c = \text{top } \lceil L/4 \rceil \text{ layers of } p_c .$$

The full average is dragged toward mid-network by the low-mass tail (a near-uniform tail pulls $\bar{\ell}_c$ toward $L/2$ regardless of where the bulk sits), while the trimmed average locates the bulk of the expertise itself. $\ell^{\ast}_c$ says where the single strongest concentration sits, $\bar{\ell}_c$ and $\bar{\ell}^{\,25\%}_c$ where the mass centers, and they can differ substantially for multi-modal or skewed profiles.

**Entropy is not independent of the expert count.** A word's entropy is bounded by its number of experts as well as by the layer count:

$$H(c) \;\le\; \log_2 \min(n_c,\, L),$$

since $n_c$ experts can occupy at most $n_c$ distinct layers. A word with many experts *can in principle spread quite evenly over the layers*, while a word with few experts *cannot produce an even distribution at all*: with $n_c < L$ a uniform allocation is impossible, and even for $n_c \gtrsim L$ the attainable shares are coarse multiples of $1/n_c$ whose sampling noise biases entropy downward. Group comparisons must therefore be read against the groups' counts. This matters concretely here because category labels systematically have *fewer* experts than concepts (Qwen3 at AP=0.6: label mean 331 experts vs. concept mean 964, and in the 150-concept run, e.g. `animal` 231 vs. member concepts often 500–1,000), so part of any "labels are more concentrated" gap could in principle be a count artifact rather than an organizational fact. Two checks bound this concern. At moderate thresholds the counts are large relative to $L$ (hundreds to thousands of experts over 28–48 layers), so the bound is slack, and module 4's dedicated entropy-vs-expert-count panel finds essentially no relationship at AP=0.6 (Pearson r = 0.02, p = 0.77, n = 204 in the Qwen3 run), so the entropy differences there are not driven by set size. At strict thresholds, however, counts collapse into the regime where the bound binds (module 1 reports minimum counts of 2 at AP=0.7 and 1 at AP ≥ 0.8 in the 150-concept run, with most surviving words under 30 experts at AP ≥ 0.8), so the across-AP entropy collapse in the Results table below is partly mechanical, and level comparisons at AP ≥ 0.8 inherit the count confound.

**Generated data structures.** One CSV, `shannon_entropy_concepts.csv`, with one row per word:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | Word identifier (concepts *and* category labels). |
| shannon_entropy | float | $H(c)$ | Entropy of the word's layerwise expert distribution. |
| gearys_c | float | $C(c)$ | Adjacent-layer autocorrelation of the profile in depth order. |
| peak_layer | int | $\ell^{\ast}_c$ | Absolute index of the layer with the strongest concentration. |
| peak_gap_pct | float | $\gamma_c$ | Percentage-point lead of the peak layer over the runner-up. |
| avg_layer | float | $\bar{\ell}_c$ | Expected (absolute) layer value over the full distribution. |
| avg_layer_top25pct | float | $\bar{\ell}^{\,25\%}_c$ | Expected layer over the top 25% most-loaded layers, renormalized. |
| experts_count | int | $n_c$ | Number of expert rows used to compute the distribution. |

Example (head of `AP_0.6/2_shannon_entropy_analysis/shannon_entropy_concepts.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`, where peak/average layers are absolute indices in 1–196, on the 28-layer `mlp.gate_proj` support):

| concept | shannon_entropy | gearys_c | peak_layer | peak_gap_pct | avg_layer | avg_layer_top25pct | experts_count |
|---|---|---|---|---|---|---|---|
| accountant | 4.4946 | 0.1866 | 166 | 0.3052 | 101.83 | 113.46 | 3277 |
| actor | 4.5028 | 0.5474 | 12 | 6.0443 | 93.02 | 75.98 | 1489 |
| airplane | 4.1070 | 0.2584 | 173 | 0.5579 | 119.88 | 152.29 | 717 |

### 2.2 Dual category definitions: label distribution vs. member average

**Mathematical formulation.** Each category $k$ is represented in two independent ways, and the same six descriptors of subchapter 2.1 are computed for both:

- *Definition A (the label itself)*: the category label is a word with its own row in the matrix, so its distribution is simply $p_k$, giving $H(p_k)$, $C(p_k)$, $\ell^{\ast}_k$, $\gamma_k$, $\bar{\ell}_k$, $\bar{\ell}^{\,25\%}_k$.
- *Definition B (average of members)*: the members' distributions are averaged element-wise,

$$\bar{q}_{k\ell} = \frac{1}{|M_k|} \sum_{m \in M_k} p_{m\ell}, \qquad \bar{q}_k = (\bar{q}_{k\ell})_{\ell},$$

which is again a valid probability distribution (a uniform mixture of the members), giving the same six descriptors, stored with the `_average` suffix. Because entropy is concave, mixing distributions can only preserve or increase entropy relative to the average of the members' entropies, so averaging tends to *wash out* concentration unless all members peak in the same layers.

To quantify whether the two definitions at least *point at the same layers*, the Pearson correlation between the two support-length vectors is computed:

$$r_k = \operatorname{corr}(p_k, \bar{q}_k),$$

set to 0 when both vectors exist but either has zero variance, and NaN when one of the two representations is missing (e.g. a label with zero retained experts), while categories missing *both* representations are skipped entirely. High $r_k$ means the label's allocation shape mirrors the members' average shape, even if their entropies differ.

**Generated data structures.** One CSV, `shannon_entropy_categories.csv`, with one row per category, and one plot:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| category | string | $k$ | Category label. |
| shannon_entropy, gearys_c, peak_layer, peak_gap_pct, avg_layer, avg_layer_top25pct | float/int | $H(p_k)$, $C(p_k)$, $\ell^{\ast}_k$, $\gamma_k$, $\bar{\ell}_k$, $\bar{\ell}^{\,25\%}_k$ | The six descriptors of the category-label distribution (Definition A). |
| shannon_entropy_average, gearys_c_average, peak_layer_average, peak_gap_pct_average, avg_layer_average, avg_layer_average_top25pct | float/int | $H(\bar{q}_k)$, $C(\bar{q}_k)$, … | The same six descriptors of the average member distribution (Definition B). |
| pearson_correlation_distributions | float | $r_k$ | Correlation between the label distribution and the member-average distribution. |
| member_count | int | $|M_k|$ | Number of valid members contributing to the category summary. |

Example (first rows of `AP_0.6/2_shannon_entropy_analysis/shannon_entropy_categories.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`, location columns abbreviated):

| category | shannon_entropy | gearys_c | peak_layer | peak_gap_pct | shannon_entropy_average | gearys_c_average | peak_layer_average | pearson_correlation_distributions | member_count |
|---|---|---|---|---|---|---|---|---|---|
| furniture | 4.1739 | 0.3904 | 47 | 0.0 | 4.3876 | 0.2708 | 19 | 0.7158 | 20 |
| clothing | 4.0994 | 0.4497 | 47 | 6.1404 | 4.2417 | 0.2587 | 19 | 0.4351 | 29 |

- `category_shannon_entropies_bar.png`, a horizontal **bar chart** of `shannon_entropy` ($H(p_k)$, x-axis) per `category` ($k$, y-axis), sorted descending, ranking categories by how distributed (top) versus localized (bottom) their label's expert allocation is.

### 2.3 Concept-vs-category concentration test (Mann–Whitney U with AUC effect size)

**Mathematical formulation.** The research question is a comparison of two samples of entropies: $\{H(c) : c \in \mathcal{C}_2\}$ (concepts) versus $\{H(p_k) : k\}$ (category labels). Because entropies are bounded and not normally distributed, a non-parametric one-sided Mann–Whitney U test is used with the alternative hypothesis "category entropies are stochastically *smaller* than concept entropies" (`alternative='less'`), i.e. categories are more concentrated. Writing $n_1, n_2$ for the two sample sizes and $R_1$ for the sum of ranks of the category sample in the pooled ranking,

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1 ,$$

and the p-value is the probability, under the null of identical distributions, of a $U$ at least as extreme in the "less" direction. Alongside the p-value, the test's own effect size is logged:

$$\mathrm{AUC} = 1 - \frac{U}{n_1 n_2} = P\big(H(\text{random category}) < H(\text{random concept})\big),$$

where 0.5 means no effect and 1.0 means every category label is more concentrated than every concept, which keeps "significant" and "large" visibly separate. A significant result means category labels' allocations are systematically more concentrated (lower $H$) than concepts', not just different in shape. Per the count-dependence paragraph of subchapter 2.1, the comparison should be read jointly with module 4's entropy-vs-expert-count panel, since the two groups differ in typical expert count.

**Generated data structures.** The $U$ statistic, p-value, and AUC are logged to `main.log` (not saved as CSV). The comparison is visualized in:

- `category_concept_shannon_entropies.png`, a **violin plot** (drawn by the shared `plot_comparison_violin` helper: quartile lines with Q1/median/Q3 text labels, violins clipped to the observed data range, and a strip plot of the raw points overlaid so small groups stay honest) comparing `shannon_entropy` ($H$, y-axis) across the two groups Specific Concepts vs. Broad Categories (x-axis).

### 2.4 Peak vs. average layer distributions across groups

**Mathematical formulation.** This analysis compares *where* in the network the three representations from 2.1–2.2 place their mass, using the per-item summary positions rather than the full distributions. Three groups $g$ are formed: Specific Concepts, Broad Categories (labels), and Broad Categories (Avg) (member averages). For the bars, each layer $\ell$ receives the percentage of group members peaking there:

$$\text{peak\%}_g(\ell) = 100 \cdot \frac{|\{c \in g : \ell^{\ast}_c = \ell\}|}{|g|},$$

which sums to 100 within each group and makes groups of very different sizes comparable. Each bar is additionally split by peak reliability: the solid segment counts members whose peak dominance gap is $\gamma_c \ge 1$ pp, and a hatched segment stacked on top counts members with an *ambiguous* peak ($\gamma_c < 1$ pp), so the histogram shows at a glance how much of the peak-location mass is actually trustworthy. For the curves, a Gaussian kernel density estimate is fitted per group to the average-layer positions (normalized within each group, `common_norm=False`), in two line styles: solid for the full-distribution average $\bar{\ell}_c$ and dashed for the trimmed average $\bar{\ell}^{\,25\%}_c$, whose comparison shows how strongly the low-mass tail pulls the centers toward mid-network. Because the sublayer-restricted support is non-contiguous in absolute layer indices, all positions are mapped to consecutive axis slots via the support's label order (fractional averages are placed between the retained layers by linear interpolation).

**Generated data structures.** One plot (no standalone CSV, since the inputs are columns of the two CSVs above):

- `peak_average_layers.png`, a combined **bar chart + KDE overlay** on twin y-axes: bars show $\text{peak\%}_g(\ell)$ (left y-axis, solid = dominant peak, hatched = ambiguous peak) against the model layer (x-axis), while the overlaid density curves show the KDE of the average layers (right y-axis, solid = full, dashed = trimmed). Both are colored by group.

### 2.5 Profile-shape map: Geary's C vs. Shannon entropy

**Mathematical formulation.** Entropy and Geary's C answer orthogonal questions about a layer profile, namely *how concentrated* (order-blind) versus *how smooth in depth* (order-sensitive), so plotting every word at coordinates $(H(c), C(c))$ maps the space of profile shapes. The four corners of the map read: concentrated & smooth (low $H$, low $C$: one compact band of layers), concentrated & jagged (low $H$, high $C$: a few isolated spikes), spread & smooth (high $H$, low $C$: broad contiguous mass), spread & jagged (high $H$, high $C$: scattered mass with no depth locality). The dashed horizontal line at $C = 1$ marks "no depth structure", and dotted lines through the middle of each axis quarter the map.

**Generated data structures.** One plot (inputs are the two CSVs of 2.1–2.2):

- `gearys_entropy_scatter.png`, a **scatter plot**: every specific concept as a small dot (category-label words are excluded from the dots, since they appear in the concepts CSV too, and would otherwise be plotted twice), every category label as a larger named diamond, the 5 most extreme concepts on each end of *both* axes annotated by name, and quadrant guides and the $C=1$ reference line drawn as described, with corner glosses naming the four shape regimes.

### 2.6 Average-layer comparison across abstraction levels

**Mathematical formulation.** The direct test of whether one abstraction level's experts sit earlier or later in the network than the other's: the violin comparison of subchapter 2.3 repeated on the depth descriptor. It uses the *trimmed* average $\bar{\ell}^{\,25\%}$ rather than the full average, because the full average is dragged toward mid-network by the low-mass tail (subchapter 2.1) and thereby compresses exactly the group differences this plot is meant to show. The full averages remain available in the CSVs and in the KDE curves of 2.4.

**Generated data structures.** One plot:

- `category_concept_avg_layer_violin.png`, the same **violin + strip** composition as 2.3, comparing `avg_layer_top25pct` ($\bar{\ell}^{\,25\%}$, y-axis) between Specific Concepts and Broad Categories.

### 2.7 Peak dominance gap ECDF

**Mathematical formulation.** The peak-reliability question of 2.4 in full distributional form: for each abstraction level, the empirical cumulative distribution of the gap,

$$\mathrm{ECDF}_g(x) = \frac{|\{c \in g : \gamma_c < x\}|}{|g|},$$

read as "the fraction of the group whose peak layer leads the runner-up by less than $x$ percentage points". The steeper the curve near zero, the less meaningful the group's peak-layer statistics are.

**Generated data structures.** One plot:

- `peak_gap_ecdf.png`, one **ECDF curve** per level of `peak_gap_pct` ($\gamma$), with the ambiguity threshold ($\gamma = 1$ pp, matching the hatching of 2.4) drawn as a dashed line and annotated with the share of each level below it.

### 2.8 Level comparisons for the location and shape descriptors

**Mathematical formulation.** The concentration test of 2.3 is complemented by the same non-parametric comparison, two-sided, applied to each of the other descriptors: peak dominance gap $\gamma$, full average layer $\bar{\ell}$, trimmed average layer $\bar{\ell}^{\,25\%}$, and Geary's C, between category labels and concepts. Each comparison reports the two-sided Mann–Whitney $U$, its p-value, and the effect size $\mathrm{AUC} = U / (n_1 n_2) = P(\text{label value} > \text{concept value})$.

**Generated data structures.** Logged to `main.log` only (four lines, one per descriptor), and the corresponding visual comparisons are 2.4–2.7.

## Results

At AP=0.6 in the GPT-2 150-concept reference run, the hypothesis holds, but only modestly in magnitude. Mean entropy for the 164 concepts is **4.49 bits** (SD 0.35) versus **4.18 bits** (SD 0.33) for the 17 category labels, so categories are lower-entropy (more concentrated) than concepts, and a Mann-Whitney U test confirms this is unlikely to be chance (U=698.5, p=3.6×10⁻⁴). But the absolute gap is small relative to the scale: the maximum possible entropy over 48 layers is log₂(48) ≈ 5.58 bits, so both groups sit well below the ceiling and only about 6% apart from each other. `category_concept_shannon_entropies.png` shows this directly: the two violins clearly overlap, with the category distribution shifted down and narrower, not cleanly separated from the concept distribution.

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

The *relative* size of the concept-vs-category gap also grows sharply with AP. At AP=0.5 concepts and categories differ by only 0.17 bits (4.83 against 4.66), and by AP=0.9 they differ by 0.89 bits on a much smaller overall scale (1.94 against 1.05), so categories collapse toward near-total concentration faster than concepts do as the threshold strips away marginal experts. Two mechanical contributions to this pattern should be kept in mind (subchapter 2.1): at strict thresholds many words' expert counts fall into the regime where $H \le \log_2 n_c$ binds, so entropies are pushed down by shrinking counts and not only by sharpening organization, and category labels reach that regime earlier because they hold fewer experts to begin with.

### The Qwen3 Richie-HSJ run at AP=0.6 (sublayer-restricted)

The new descriptors and plots are documented from `research_plots_qwen_richie_hsj_with_sublayer_analysis` (Qwen3-1.7B, 204 words = 196 concepts + 8 category labels, analysis restricted to the 28 `mlp.gate_proj` layers). The picture there is instructively *different* from the GPT-2 150-concept run:

- **The central concentration hypothesis fails in this run.** Concept entropy averages 4.16 bits (SD 0.21) versus 4.20 bits (SD 0.12) for the 8 labels, both within ~15% of the 4.81-bit ceiling of a 28-layer support, and the one-sided Mann-Whitney test rejects nothing (U=875, p=0.64, AUC=0.464, i.e. essentially a coin flip). The three-way ordering of the 150-run (label < concepts < avg-member) also breaks: here concepts are nominally the sharpest (4.16 < 4.20 < 4.29). With label counts systematically smaller than concept counts (mean 331 vs. 964) *and* a near-zero entropy-vs-count correlation in module 4 (r=0.02, p=0.77, n=204), this null is not a count artifact at this threshold, and it is instead a genuine model/dataset/sublayer difference from the GPT-2 result, and a caution against treating the 150-run finding as architecture-general.
- **Where the levels do differ is depth.** Category labels sit significantly *deeper* than concepts: full average layer p=0.0081 with AUC(label>concept)=0.776, trimmed average p=0.045 with AUC=0.710 (trimmed means: labels 135.7 vs. concepts 113.1, on the absolute 1–196 index). This is the contrast that subchapter 2.6's violin displays.
- **Peak layers are often unreliable, justifying the gap machinery.** 37% of concepts and 50% of labels have an ambiguous peak ($\gamma < 1$ pp, see the ECDF of 2.7), and the levels do not differ significantly in gap (p=0.44). The hatched segments in 2.4's histogram are correspondingly large, since many apparent "peak locations" in this run are near-ties.
- **Both levels have smooth depth profiles.** Mean Geary's C is 0.32 for concepts and 0.37 for labels (no significant level difference, p=0.29), well below the $C=1$ no-structure line: expert mass sits in contiguous depth bands rather than isolated spikes, consistent with module 1's finding that `mlp.gate_proj` is among the smoothest sublayers. The shape map of 2.5 accordingly shows the population concentrated in the "smooth" half of the map, with entropy, not smoothness, doing most of the discriminating between words.
