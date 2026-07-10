# Module 7: Cosine Typicality vs. Human Typicality

## Research question

Does the expert structure of a category recover human judgments of typicality, with concepts that humans rate as more typical of a category also more similar (by cosine similarity) to that category's expert-allocation prototype?

## Analysis

Notation. For a category $k$, let $M_k$ be its member concepts that appear in the filtered expert data. Only categories with $|M_k| \ge 2$ are processed (a centroid of one member is just that member). The model-derived score computed here is called **Cosine Typicality**, to distinguish it from the metadata-sourced **Human Typicality** $t_c$ it is compared against.

### 7.1 Global prototype Cosine Typicality

**Mathematical formulation.** Within category $k$, each member concept $c \in M_k$ is encoded as a binary allocation vector over the *(layer, unit)* feature space observed among the category's members. Let $\mathcal{F}_k = \{(\ell, u) : \text{some } m \in M_k \text{ has expert } u \text{ in layer } \ell\}$ with $d_k = |\mathcal{F}_k|$. Then

$$\mathbf{x}_c \in \{0,1\}^{d_k}, \qquad (\mathbf{x}_c)_{(\ell,u)} = \begin{cases} 1 & \text{concept } c \text{ has expert unit } u \text{ in layer } \ell \\ 0 & \text{otherwise.} \end{cases}$$

(Restricting the feature space to pairs observed within the category is harmless: coordinates that are 0 for every member would change no inner product or norm.) The category's *global prototype* is the centroid, the element-wise mean of the member vectors,

$$\boldsymbol{\mu}_k = \frac{1}{|M_k|} \sum_{m \in M_k} \mathbf{x}_m \in [0,1]^{d_k},$$

whose $(\ell, u)$ coordinate is the fraction of members that use that expert. The Cosine Typicality of member $c$ is its cosine similarity to the centroid (via `sklearn.metrics.pairwise.cosine_similarity`):

$$T^{\cos}_c = \frac{\mathbf{x}_c \cdot \boldsymbol{\mu}_k}{\lVert \mathbf{x}_c \rVert \, \lVert \boldsymbol{\mu}_k \rVert} \in [0, 1],$$

which is 1 when the concept's expert pattern points in exactly the same direction as the category average and 0 when the concept shares no experts with any other member (an all-zero $\mathbf{x}_c$ is assigned similarity 0 by convention). This is the geometric analogue of prototype theory: the centroid plays the role of the category prototype, and $T^{\cos}_c$ ranks members by how well they instantiate it. Note that $c$ itself contributes $1/|M_k|$ of the centroid, so scores are inflated for tiny categories.

**Generated data structures.** One CSV, `global_prototype_typicality.csv`, with one row per concept-category pair:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| category | string | $k$ | Category label. |
| concept | string | $c$ | Concept identifier. |
| human_typicality | float | $t_c$ | Human-judged typicality score for the concept, read from the metadata. |
| global_cosine_typicality | float | $T^{\cos}_c$ | Cosine Typicality: cosine similarity between the concept vector and the global prototype vector for the category. |

Example (head of `AP_0.6/7_typicality_analysis/global_prototype_typicality.csv` in `research_plots_150_revised_executor_again`):

| category | concept | human_typicality | global_cosine_typicality |
|---|---|---|---|
| animal | alligator | 0.459 | 0.5333 |
| animal | frog | 0.575 | 0.4424 |
| animal | goldfish | 0.506 | 0.3512 |
| animal | iguana | 0.377 | 0.5413 |
| animal | leech | 0.065 | 0.4161 |

### 7.2 Per-layer prototype Cosine Typicality and the most-typical evolution

**Mathematical formulation.** The global prototype averages over the whole network at once, so it cannot say *where* in the network a concept is or isn't prototypical. The per-layer version repeats the construction of 7.1 within each layer $\ell$: the feature space is the set of units the category's members use at that layer, each member gets a binary vector $\mathbf{x}^{(\ell)}_c$ over those units, the layer prototype is the centroid

$$\boldsymbol{\mu}^{(\ell)}_k = \frac{1}{|M_k|} \sum_{m \in M_k} \mathbf{x}^{(\ell)}_m,$$

and the layer-wise Cosine Typicality is

$$T^{\cos}_{c,\ell} = \frac{\mathbf{x}^{(\ell)}_c \cdot \boldsymbol{\mu}^{(\ell)}_k}{\lVert \mathbf{x}^{(\ell)}_c \rVert \, \lVert \boldsymbol{\mu}^{(\ell)}_k \rVert},$$

set to 0 for every member when the category has no expert units at all in layer $\ell$ (empty feature space), and 0 for any member with no experts there. From these scores, each layer's *most typical member* is extracted:

$$c^{\ast}_{k,\ell} = \arg\max_{c \in M_k} T^{\cos}_{c,\ell},$$

so tracking $c^{\ast}_{k,\ell}$ across $\ell = 1, \dots, L$ shows whether the same concept anchors the category at every depth or the "best exemplar" rotates layer by layer.

**Generated data structures.** One module-level CSV and, per category, one CSV and one plot:

- `layer_prototype_typicality.csv`, one row per concept-category-layer combination:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| category | string | $k$ | Category label. |
| concept | string | $c$ | Concept identifier. |
| layer_idx | int | $\ell$ | Numeric layer index. |
| layer_name | string | $\ell$ (label) | Formatted layer label. |
| human_typicality | float | $t_c$ | Human-judged typicality score for the concept, read from the metadata. |
| layer_cosine_typicality | float | $T^{\cos}_{c,\ell}$ | Cosine Typicality: cosine similarity between the concept vector and the prototype vector at that layer. |

Example (head of `AP_0.6/7_typicality_analysis/layer_prototype_typicality.csv` in `research_plots_150_revised_executor_again`):

| category | concept | layer_idx | layer_name | human_typicality | layer_cosine_typicality |
|---|---|---|---|---|---|
| animal | alligator | 1 | 1.L.0.attn.c_attn | 0.459 | 0.2577 |
| animal | frog | 1 | 1.L.0.attn.c_attn | 0.575 | 0.3266 |
| animal | goldfish | 1 | 1.L.0.attn.c_attn | 0.506 | 0.4092 |
| animal | iguana | 1 | 1.L.0.attn.c_attn | 0.377 | 0.5324 |
| animal | leech | 1 | 1.L.0.attn.c_attn | 0.065 | 0.3314 |

- `<category>_most_typical_per_layer.csv` (per-category folder, e.g. `7_typicality_analysis/animal/animal_most_typical_per_layer.csv`): `layer_prototype_typicality.csv` filtered to the category, with the row of maximum `layer_cosine_typicality` selected per `layer_idx` (`groupby('layer_idx')['layer_cosine_typicality'].idxmax()`, i.e. $c^{\ast}_{k,\ell}$), renamed and sorted by layer. One row per model layer:

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| layer_idx | int | $\ell$ | Numeric layer index. |
| layer_name | string | $\ell$ (label) | Formatted layer label. |
| most_typical_model_concept | string | $c^{\ast}_{k,\ell}$ | Concept with the highest Cosine Typicality to the layer prototype. |
| cosine_similarity_score | float | $T^{\cos}_{c^{\ast},\ell}$ | Cosine Typicality of that concept to the layer prototype. |
| hardcoded_human_score | float | $t_{c^{\ast}}$ | Human Typicality score of that same concept. |

Example (head of `AP_0.6/7_typicality_analysis/animal/animal_most_typical_per_layer.csv` in `research_plots_150_revised_executor_again`):

| layer_idx | layer_name | most_typical_model_concept | cosine_similarity_score | hardcoded_human_score |
|---|---|---|---|---|
| 1 | 1.L.0.attn.c_attn | iguana | 0.5324 | 0.377 |
| 2 | 2.L.0.attn.c_proj | iguana | 0.5687 | 0.377 |
| 3 | 3.L.0.mlp.c_fc | iguana | 0.5706 | 0.377 |
| 4 | 4.L.0.mlp.c_proj | iguana | 0.6686 | 0.377 |
| 5 | 5.L.1.attn.c_attn | iguana | 0.5887 | 0.377 |

- `<category>_most_typical_evolution.png`, a **lollipop/stem plot**: x-axis is the model layer (`layer_name`, $\ell$), y-axis is `cosine_similarity_score` ($T^{\cos}_{c^{\ast},\ell}$, "Cosine Typicality (Layer Prototype)"), with each point labeled by `most_typical_model_concept` ($c^{\ast}_{k,\ell}$).

### 7.3 Comparison with Human Typicality

**Mathematical formulation.** The research question is answered by correlating the model-derived and human scores within each category. Over the members of category $k$ with both scores defined,

$$r_k = \operatorname{corr}\big(t_c,\; T^{\cos}_c\big)_{c \in M_k},$$

the Pearson correlation (computed only when both variables have nonzero variance within the category), with its two-sided p-value. A positive, significant $r_k$ means members humans call typical also sit close to the category centroid in expert space. The pooled correlation over all concept-category rows, ignoring category identity, is reported in the Results as well. Note that the per-category sample sizes are small ($n = |M_k|$, typically 3 to 12), so individual $r_k$ values carry wide confidence intervals.

**Generated data structures.** Two plots per category (no new CSV, both read `global_prototype_typicality.csv`):

- `<category>_typicality_comparison_bar.png`, a vertical grouped **bar chart** of `Score` (y-axis) versus `concept` ($c$, x-axis, ordered by descending $t_c$), with bars colored by `Metric`, either "Human Typicality" ($t_c$) or "Cosine Typicality" ($T^{\cos}_c$), and titled "'{category}' Human Typicality vs. Cosine Typicality". Visual agreement would show both bar heights declining together left to right.
- `<category>_typicality_correlation_scatter.png`, a **scatter plot with a linear regression line**, x: `human_typicality` ($t_c$), y: `global_cosine_typicality` ($T^{\cos}_c$), one point per concept (annotated with the concept name) and the Pearson $r_k$/$p$ in the title.

## Results

At AP=0.6, pooled across all 147 concept-category rows, ignoring which category each concept belongs to, the correlation between Human Typicality and global Cosine Typicality is **r=-0.001, p=0.99, no relationship at all**. Per-category correlations show real heterogeneity. The categories **weapon** (r=0.88, p<0.001, n=8) and **vehicle** (r=0.74, p=0.006, n=12) are strong and positive, whereas **jewelry** (r=-0.98), **container** (r=-0.74, p=0.01), and **drink** (r=-0.72, p=0.03) are significantly *negative*. Only 9 of 17 categories have a positive r at all, and only 2 of 17 are both positive and significant.

### Across AP thresholds

| AP | Pooled r | Pooled p | n | Positive / 17 cat. | Sig. positive | Sig. negative |
|---|---|---|---|---|---|---|
| 0.5 | 0.091 | 0.27 (n.s.) | 147 | 10 | 2 | 2 |
| 0.6 | -0.001 | 0.99 (n.s.) | 147 | 9 | 2 | 3 |
| 0.7 | -0.101 | 0.22 (n.s.) | 147 | 7 | 1 | 3 |
| 0.8 | -0.149 | 0.073 (n.s.) | 146 | 5 | 1 | 2 |
| 0.9 | -0.242 | **0.011 (sig.)** | 110 | 4 | 0 | 3 |

This sweep reveals a clear and somewhat damning trend that the single-threshold view hides: the pooled correlation doesn't just hover near zero. It **moves monotonically from weakly positive to significantly negative** as AP tightens (0.091, -0.001, -0.101, -0.149, and -0.242 across AP 0.5 to 0.9). At the strictest threshold, the pooled relationship is the *opposite* of the hypothesis, and is itself statistically significant. The number of categories with any positive correlation also shrinks steadily (10, 9, 7, 5, and 4 out of 17), and by AP=0.9 **zero** categories show a significant positive relationship while 3 show significant negative ones.

Two categories are worth calling out specifically because they're consistent across every threshold tested, which is rare in this analysis:

- **`weapon` is positive and significant at every single AP from 0.5 to 0.8** (r=0.87, 0.88, 0.85, 0.79), the most reliable piece of evidence anywhere in this module that the hypothesis can hold. It's notably absent from the AP=0.9 "significant categories" list only because the per-category n shrinks too far to reach significance at that threshold, not because the relationship reverses.
- **`container` is negative and significant at every single AP from 0.5 to 0.9** (r=-0.68, -0.74, -0.81, -0.84, -0.96), an equally reliable finding in the *wrong* direction. `drink` follows a similar pattern from AP=0.6 onward (r=-0.72 to -0.79).

So the honest summary, strengthened by the threshold sweep rather than weakened by it: this module does not show that the model's expert-allocation geometry recovers human typicality judgments in general, and the strictest, most selective AP threshold makes that conclusion worse, not better. A small number of categories (weapon, and to a lesser extent vehicle) show a real, repeatable positive relationship, whereas a small number of others (container, drink) show an equally real, repeatable relationship in the opposite direction. Any claim about "the expert representation captures psychologically plausible structure" needs to be scoped to specific categories, not stated as a general result, and should note that the relationship gets worse, not better, as the expert-selection criterion is tightened.
