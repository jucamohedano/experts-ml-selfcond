# Module 7: Cosine Typicality vs. Human Typicality

## Research question

Does the expert structure of a category recover human judgments of typicality, with concepts that humans rate as more typical of a category also more similar (by cosine similarity) to that category's expert-allocation prototype?

## Analysis

Notation. For a category $k$, let $M_k$ be its member concepts that appear in the current scope's expert data. Only categories with $|M_k| \ge 2$ are processed (a centroid of one member is just that member). The model-derived score computed here is called **Cosine Typicality**, to distinguish it from the metadata-sourced **Human Typicality** $t_c$ it is compared against.

**Analysis scopes.** This module is *mixed*. The global prototype of subchapter 7.1 is set-based, since its feature space is individual (layer, unit) pairs, so it is computed once per scope and its outputs carry no axis suffix. It must be fed the scope's own flat expert rows for exactly that reason: on a block-aggregated frame the layer part of each feature is the block, so units sharing an index across different projections of one block would collapse into a single feature and inflate every prototype. The per-layer prototypes of subchapter 7.2 are order-based, so in the whole-model scope they are produced twice, suffixed `_by_block` and `_by_layer`, and once unsuffixed in a sublayer scope. Every scope writes into the module folder (whole model) or `sublayers/<rank>_<sublayer>/`, with the cross-scope summary in `sublayer_comparison.csv` and `sublayer_comparison.png`, whose columns are the Pearson correlation between global Cosine Typicality and Human Typicality pooled over all categories, its p-value, and the number of concepts carrying both scores. Filenames listed below are given in their unsuffixed form, which is what a sublayer scope writes, and the whole-model scope inserts the axis suffix before the extension. Module 1, subchapter 1.7 defines the mechanics.

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

*Scope note.* The 150-concept figures below use 17 categories and the analysis sublayer only. The Richie-HSJ tables are whole-model scope from the corrected `_sensefix` runs, 197 concepts across 8 categories, with `typicality_HSJ_pairwise` as the human measure.

**The two datasets give opposite answers, and this is the most consequential disagreement in the analysis suite.** Both are reported here rather than reconciling them, since the difference is real and its cause is not isolated to one factor.

### The 150-concept run, 17 categories, analysis sublayer

At AP=0.6, pooled across all 147 concept-category rows, the correlation between Human Typicality and global Cosine Typicality is **r=-0.001, p=0.99, no relationship at all**, and the pooled value moves monotonically from weakly positive to significantly negative as AP tightens (0.091, -0.001, -0.101, -0.149, -0.242 across AP 0.5 to 0.9), the last being significant at p=0.011. The number of categories with any positive correlation shrinks steadily (10, 9, 7, 5, 4 of 17), and by AP=0.9 zero categories are significantly positive while 3 are significantly negative. `weapon` is positive and significant at every AP from 0.5 to 0.8 (r=0.87 to 0.79), while `container` is negative and significant at every AP (r=-0.68 to -0.96).

### The Richie-HSJ runs, 8 categories, whole model

| Model | AP | n | Pooled r | p | Positive / 8 | Sig. positive | Sig. negative |
|---|---|---|---|---|---|---|---|
| GPT-2 | 0.5 | 197 | +0.248 | <0.001 | 7 | 3 | 0 |
| GPT-2 | 0.6 | 197 | +0.198 | 0.005 | 7 | 4 | 0 |
| GPT-2 | 0.7 | 197 | +0.145 | 0.043 | 7 | 2 | 0 |
| GPT-2 | 0.8 | 188 | +0.141 | 0.054 | 7 | 2 | 0 |
| GPT-2 | 0.9 | 114 | +0.055 | 0.56 | 4 | 0 | 1 |
| Qwen3 | 0.5 | 197 | +0.454 | <0.001 | 8 | 4 | 0 |
| Qwen3 | 0.6 | 197 | +0.393 | <0.001 | 7 | 6 | 0 |
| Qwen3 | 0.7 | 197 | +0.307 | <0.001 | 7 | 6 | 0 |
| Qwen3 | 0.8 | 197 | +0.211 | 0.003 | 6 | 4 | 0 |
| Qwen3 | 0.9 | 192 | +0.060 | 0.41 | 5 | 1 | 0 |

Here the pooled correlation is **positive and significant in both models across AP 0.5 to 0.8**, reaching r=0.454 for Qwen3 at AP 0.5, and there is not a single significantly negative category anywhere in the sweep except one GPT-2 cell at AP 0.9. The decay toward zero at AP 0.9 is the familiar thinning effect rather than a reversal.

Per-category correlations (AP 0.5 / 0.6 / 0.7 / 0.8, asterisk marks p<0.05):

| Category | GPT-2 | Qwen3 |
|---|---|---|
| professions | +0.68* +0.65* +0.61* +0.51* | +0.62* +0.62* +0.65* +0.65* |
| sports | +0.40* +0.46* +0.50* +0.60* | +0.34 +0.41* +0.49* +0.50* |
| vegetables | +0.12 +0.26 +0.43 +0.33 | +0.54* +0.51* +0.62* +0.47* |
| fruit | +0.31 +0.45* +0.35 +0.33 | +0.68* +0.70* +0.60* +0.41 |
| clothing | +0.50* +0.43* +0.25 +0.01 | +0.62* +0.62* +0.52* +0.18 |
| vehicles | +0.27 +0.15 +0.07 +0.09 | +0.38 +0.49* +0.55* +0.47* |
| birds | -0.07 +0.03 +0.01 +0.06 | +0.24 +0.30 +0.17 -0.00 |
| furniture | +0.02 -0.19 -0.30 -0.31 | +0.26 -0.02 -0.13 -0.19 |

`professions` and `sports` are positive in both architectures at every threshold, and `furniture` is the one category that trends negative in both, though never significantly.

### Reading the disagreement

Three things differ between the two runs at once, so the reversal cannot be attributed to any single one: the stimulus set (150 concepts across 17 categories versus Richie-HSJ's 197 across 8), the analysis scope (single sublayer versus whole model), and the human measure (`typicality` versus `typicality_HSJ_pairwise`). The Richie-HSJ result additionally benefits from the word-sense correction documented in `fixes.md`, which raised category alignment across the board.

The defensible statement is therefore narrower than either run alone suggests: **on the Richie-HSJ norms, with whole-model expert sets, model-derived cosine typicality does recover human typicality judgments at a modest but reliable level in both architectures**, and the earlier negative result should be scoped to the 150-concept dataset and its single-sublayer view rather than treated as the general finding. A direct test of which factor drives the difference would require rerunning the 150-concept dataset at whole-model scope, which has not been done.
