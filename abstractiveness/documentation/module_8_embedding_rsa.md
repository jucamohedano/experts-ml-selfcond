# Module 8: Expert Set vs Embedding Semantics (RSA)

## Research question

Does the semantic structure carried by the expert set match the semantic structure the model itself carries at a middle layer, and, going one step further, which sublayer's own expert set is the best proxy for that semantics in the first place?

The expert set is this thesis's object of study, a small set of concept-selective units recovered by thresholding Average Precision. Whether it is a faithful window onto the model's semantics, or an artifact of the selection procedure, is not answered by looking at expert sets alone. This module answers it by comparing the expert set against an independent reading of the same model, the geometry of its hidden representations. Beyond that headline question for the pipeline's chosen analysis sublayer, the module also asks whether a different sublayer's own expert set would have been a better or worse proxy, using a count-matched control so the comparison is not just a restatement of which sublayer happens to hold the most experts.

A middle layer is the reference because the probing and brain alignment literature converges on intermediate layers carrying the richest semantics, before the final layers specialise for next-token prediction. Rather than accept a layer choice on citation alone, the module sweeps every layer, so the choice is checked against this project's own data.

## Analysis

### 8.1 The two representational systems

Let $C$ be the set of level-2 concepts that carry a category, $n = |C| = 196$. Category labels are excluded because a label has no same-category peers, matching module 1's convention. The concept set is fixed from the metadata rather than from the expert frame, because at strict AP thresholds the expert frame loses concepts entirely and a set that shrank with the threshold would make the matrices incomparable across the sweep.

**Expert side.** For concept $c$, let $E_c$ be its expert set, the units $(\ell, u)$ with $\mathrm{AP}_c(\ell, u) \ge \tau$, pooled over every block $\ell$ of the sublayer under analysis (`mlp.c_fc` for GPT-2, `mlp.gate_proj` for Qwen3, when that sublayer is the one being analysed, see section 8.5 for how the other sublayers of the architecture get their own expert side). Pairwise similarity is the Jaccard index

$$S^{\mathrm{exp}}_{cd} = \frac{|E_c \cap E_d|}{|E_c \cup E_d|} \in [0, 1],$$

reusing `pair_similarity_vector`, the same function behind module 1's sublayer ranking. This matrix is fixed within an AP threshold and within a sublayer, so a given sublayer's whole sweep is read against one expert side.

**Embedding side.** For concept $c$ at layer $\ell$, let $P_c$ be its positive sentences and $a^{(\ell)}_{c,s} \in \mathbb{R}^{d_\ell}$ the max-pooled activation for sentence $s$. The concept embedding is the positive-sentence mean

$$e^{(\ell)}_c = \frac{1}{|P_c|} \sum_{s \in P_c} a^{(\ell)}_{c,s}, \qquad |P_c| = 400 .$$

These come from the response pkls, the same tensors expertise is derived from, so no new model inference is involved. They are precomputed once per model by `scripts/precompute_concept_embeddings.py`, because they do not depend on $\tau$.

Each unit is then standardised across concepts, $\tilde{e}^{(\ell)}_{c,u} = (e^{(\ell)}_{c,u} - \mu_u) / \sigma_u$ with $\mu_u, \sigma_u$ the mean and standard deviation of unit $u$ over all concepts, and similarity is the Pearson correlation between concept vectors over units,

$$S^{\mathrm{emb},\ell}_{cd} = \mathrm{corr}\big(\tilde{e}^{(\ell)}_c, \tilde{e}^{(\ell)}_d\big) .$$

The standardisation is load-bearing, not cosmetic. A mean over 400 sentences is dominated by the generic sentence structure every concept shares, so the concept signal is a small perturbation on a large common baseline. Measured on GPT-2, a raw cosine over the same vectors gives a mean pairwise similarity of 0.95 and separates categories at ROC-AUC 0.79, while the standardised correlation gives a mean of $-0.003$ and an AUC of 0.93. The raw variant is retained as a robustness row under `embedding_metric = cosine_raw`, and the standardised `correlation_zscored` is primary.

### 8.2 Second-order RSA

Both matrices describe the same $n(n-1)/2 = 19{,}110$ concept pairs, so they are compared through the correlation of their off-diagonals,

$$\rho_\ell = \mathrm{Spearman}\big(\mathrm{offdiag}(S^{\mathrm{exp}}), \, \mathrm{offdiag}(S^{\mathrm{emb},\ell})\big) .$$

Spearman rather than Pearson because the relationship is monotonic but strongly saturating, which the pairs-scatter variants show directly (`rsa_pairs_scatter_binned.png` most directly, since it turns 19,110 overplotted, zero-inflated Jaccard points into legible binned medians): Jaccard is compressed near zero for most pairs while embedding similarity spans a wide range. Both sides are similarities rather than distances, so a positive $\rho$ reads directly, concepts sharing experts also sit close in embedding space.

**Significance.** Concept pairs are not independent observations, since each concept appears in $n-1$ of them. The null therefore shuffles concepts, never pairs. For a permutation $\pi$, one matrix is re-indexed as $S_{\pi(c)\pi(d)}$ and the statistic recomputed, which preserves each matrix's internal geometry and breaks only the concept-to-concept correspondence under test. The one-sided p-value is

$$p = \frac{1 + \#\{\pi : \rho_\pi \ge \rho\}}{1 + B}, \qquad B = 9999,$$

so the smallest reportable value is $10^{-4}$. This is the same Mantel scheme module 1 uses for its category alignment, extended from a binary target to a continuous one.

Permuting rows and columns is a bijection on the off-diagonal multiset, so the ranks are computed once and permuted with the matrix rather than recomputed per shuffle, and the permuted rank vector's mean and standard deviation are invariant. Spearman under permutation therefore reduces to one dot product, which measured about six times faster than gathering the upper triangle each time.

**The p-value is not the interesting axis.** At 196 concepts every layer reaches the $10^{-4}$ floor, exactly as module 1 already reports for every sublayer. Effect size, the bootstrap interval and the noise ceiling carry the argument, and Benjamini-Hochberg across layers is reported for completeness rather than because it discriminates.

### 8.3 Noise ceiling

A layer's $\rho$ cannot exceed the reliability of its own embedding matrix, so a rising curve could mean deeper layers are simply less noisy rather than more aligned. The precompute stores positive-sentence sums split by the parity of the sentence index within each batch, giving two independent half-estimates per concept. The ceiling is their agreement,

$$r_\ell = \mathrm{Spearman}\big(\mathrm{offdiag}(S^{\mathrm{emb},\ell}_{A}), \, \mathrm{offdiag}(S^{\mathrm{emb},\ell}_{B})\big), \qquad \text{ceiling}_\ell = \frac{2 r_\ell}{1 + r_\ell},$$

with the Spearman-Brown correction applied once, because each half rests on 200 sentences rather than 400. Measured on GPT-2 the ceiling is 0.98 to 0.99 at every layer, so the embedding matrices are close to perfectly reliable and the ceiling explains none of the depth variation. `rho_over_noise_ceiling` reports $\rho_\ell / \text{ceiling}_\ell$.

### 8.4 Two confounds, and what survives them

**Circularity.** Experts at layer $\ell$ are selected by Average Precision on the very activations whose positive centroid forms the embedding at layer $\ell$, so a same-layer comparison would be partly self-referential. Pooling the expert side over all blocks makes that shared derivation one component among twelve rather than the whole comparison. AP is also a rank statistic measured against concept-specific negative sentences, while the embedding is an absolute level with no such reference, and thresholding discards all magnitude information, so the two are related but far from equivalent.

**Expert density.** This one is larger and was found empirically rather than anticipated. The pooled expert set is fixed, but it is assembled from the per-layer expert sets, and expert mass across depth is strongly U-shaped: at AP 0.5 on GPT-2, block 0 contributes 13.3% and block 11 contributes 15.1% of the pool, against 4.6% at block 6. A layer donating a larger share matches the pool partly by construction. Across the twelve `mlp.c_fc` layers the association between a layer's share and its $\rho$ is Spearman 0.97, so the raw depth curve is largely a restatement of where experts live.

The module therefore also reports a leave-one-out curve, `rho_expert_vs_embedding_holdout`, where each contributing layer is scored against a pool it has been excluded from, and all depth claims are read from that curve (see `depth_curve_column` in the summary, which records which column supplied them). Removing the self-contribution turns out to change little, 0.704 to 0.689 at block 0, with the density association still 0.92, so the mechanical self-match is not the explanation.

The explanation is in `median_cross_layer_emb_similarity`: the median correlation between the embedding matrices of different blocks is 0.93, with a range of 0.86 to 0.99. The semantic geometry barely moves across depth in this dataset. The depth variation in $\rho$ is a small modulation on top of one shared geometry, which is why no layer separates from the others and why the peak plateau spans all twelve layers.

Section 8.5 below adds a second density confound at the sublayer level, expert mass differs by orders of magnitude between sublayers as well as between layers, and the count-matched control corrects for it the same way the leave-one-out curve corrects for the within-sublayer version.

### 8.5 Per-sublayer comparison

The module loops over every sublayer of the architecture (for GPT-2, `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`, parsed from the layer names in ascending layer-index order). Each sublayer builds its own pooled expert Jaccard matrix from its own expert rows only, never borrowing another sublayer's experts, and is swept against embedding layers with the same Spearman and Mantel machinery of sections 8.1 to 8.4.

The two sweep modes differ only in which embedding layers get compared against a sublayer's own pooled expert set, not in how $\rho$ itself is computed. For the sublayer under analysis, the sweep stays the whole-network sweep this module has always run, every layer of every sublayer's embedding geometry is compared against the analysis sublayer's fixed expert set. That is why its own `embedding_rsa_layer_sweep.csv` carries a `sublayer` value that changes row to row (see the example below), and it is also why the analysis sublayer's headline peak, middle layer, and plateau, still read only from its own rows (`is_analysis_sublayer == True`), sit inside a comparison that was checked against the network's full depth rather than against a narrower slice of it, this unrestricted design is the module's original behaviour and is unchanged by this session's extension. For every other sublayer, the sweep is restricted to that sublayer's own layers only, so, for example, `attn.c_attn`'s own file sweeps `attn.c_attn`'s expert set against only `attn.c_attn`'s own layers, never against `mlp.c_fc`'s or `attn.c_proj`'s, keeping a non-analysis sublayer's own output lean and avoiding rows that would only restate context the analysis sublayer's file already provides in full.

**The count-matched control.** Sublayers differ from each other by orders of magnitude in expert count (25,549 distinct units for `mlp.c_fc` against 5,581 for `mlp.c_proj` at AP 0.5 on GPT-2), and, exactly as within a single sublayer's depth curve, $\rho$ tends to rise with expert-set density. Ranking sublayers on their raw, full-expert-set $\rho$ would therefore mostly reward whichever sublayer happens to hold the most experts rather than whichever sublayer's expert set most faithfully tracks the embedding geometry.

The control equalises expert mass before ranking. Let $R_m$ be the total number of concept-unit assignment rows in sublayer $m$'s own expert dataframe, one row per concept that is a positive expert for a given (layer, unit) pair, so a unit reused across several concepts contributes one row per concept it serves and $R_m$ can exceed $m$'s distinct-unit count (`mlp.c_proj` has $R_m = 13{,}469$ assignment rows against only 5,581 distinct units). Let

$$k = \min_m R_m$$

be the smallest such row count across the sublayers of the architecture at the current AP threshold. For each sublayer $m$, keep only its $k$ highest-AP assignment rows (by the `ap` column, pooled over every layer of $m$, not per layer), rebuild the Jaccard matrix from just those rows, and correlate it against the embedding similarity at $m$'s own middle layer,

$$\rho^{\mathrm{matched}}_m = \mathrm{Spearman}\big(\mathrm{offdiag}(S^{\mathrm{exp}}_{m, \mathrm{top}\text{-}k}), \, \mathrm{offdiag}(S^{\mathrm{emb}, \ell^{\mathrm{mid}}_m})\big),$$

the same reference embedding similarity used for $m$'s full-expert-set $\rho$. Every sublayer is now scored from the same number of expert assignments, so a gap between $\rho^{\mathrm{matched}}_m$ and the raw $\rho_m$ isolates how much of a sublayer's raw score was density rather than fidelity.

A single AP 0.5 GPT-2 verification run illustrates the confound cleanly. `mlp.c_proj`, the sublayer with the fewest assignment rows, is the one that defines $k$ (13,469), so nothing is trimmed from it and its count-matched $\rho$ equals its full $\rho$ exactly (0.2371 both ways). `mlp.c_fc`, the sublayer with the most experts, drops sharply once matched down to the same expert mass, from 0.6355 full to 0.3098 count-matched, more than half its raw advantage over the other sublayers evaporates once density is controlled for. This is the module's clearest empirical illustration of the density confound the count-matched control exists to remove.

`sublayer_comparison.png` renders this as a ranked horizontal bar chart, full $\rho$ against count-matched $\rho$ for every sublayer, expert counts annotated next to each bar, and a split-half noise-ceiling reference line, with one small-multiple depth-profile panel per sublayer underneath it (the pooled and held-out curves of that sublayer's own sweep, so the ranking at the top and each sublayer's own depth behaviour are visible together in one figure).

## Results across AP thresholds

Both models were run across the full sweep on the corrected `_sensefix` data, 197 concepts. GPT-2 headline layer `27.L.6.mlp.c_fc` (block 6 of 12, derived as `n_blocks // 2`).

| AP | $\rho$ at middle layer | 95% CI | $\rho$ leave-one-out | Peak $\rho$ | Density association | Middle minus outer |
|---|---|---|---|---|---|---|
| 0.5 | 0.641 | [0.605, 0.678] | 0.637 | 0.695 | 0.95 | -0.034 |
| 0.6 | 0.494 | [0.456, 0.532] | 0.491 | 0.543 | 0.87 | -0.021 |
| 0.7 | 0.340 | [0.304, 0.376] | 0.338 | 0.357 | 0.87 | -0.006 |
| 0.8 | 0.189 | [0.156, 0.224] | 0.185 | 0.193 | 0.43 | +0.003 |
| 0.9 | 0.067 | [0.040, 0.094] | 0.067 | 0.069 | 0.50 | +0.002 |

The noise ceiling is 0.981 at every threshold, being a property of the embedding side alone, which does not depend on $\tau$.

Qwen3-1.7B on the same dataset, headline layer `103.L.14.mlp.gate_proj` (block 14 of 28).

| AP | $\rho$ at middle layer | 95% CI | $\rho$ leave-one-out | Peak $\rho$ | Density association | Middle minus outer |
|---|---|---|---|---|---|---|
| 0.5 | 0.660 | [0.624, 0.695] | 0.660 | 0.748 | 0.66 | -0.030 |
| 0.6 | 0.553 | [0.517, 0.590] | 0.553 | 0.629 | 0.74 | -0.031 |
| 0.7 | 0.462 | [0.429, 0.496] | 0.462 | 0.517 | 0.78 | -0.023 |
| 0.8 | 0.336 | [0.305, 0.370] | 0.336 | 0.380 | 0.92 | -0.016 |
| 0.9 | 0.156 | [0.127, 0.185] | n/a | 0.171 | 0.77 | -0.006 |

Its noise ceiling is 0.979, again constant across thresholds. The leave-one-out column is unavailable at Qwen3 AP 0.9, where too few concepts retain experts for the control to be computed.

The two architectures agree on every qualitative point. Agreement is strong at permissive thresholds and decays as the threshold tightens, the middle third never beats the outer thirds, the peak plateau covers most or all of the network, and the depth curve tracks expert density. Qwen3 degrades more gracefully at the strict end, holding $\rho = 0.15$ at AP 0.9 against GPT-2's 0.07, which is consistent with its larger expert pool leaving more structure intact after thresholding. Its depth curve is also visibly flatter, spanning roughly 0.65 to 0.75 across all 28 blocks, so the depth-invariance conclusion is if anything stronger for the larger model.

**The expert set does recover the model's semantic geometry.** At AP 0.5 the two systems agree at $\rho = 0.641$ (GPT-2) and $\rho = 0.660$ (Qwen3) against ceilings of 0.981 and 0.979, from a comparison in which one side is a binary set-membership pattern and the other a continuous activation geometry. This is the module's main result and it supports the validity of the expert-set methodology. The leave-one-out control changes the value by at most 0.004, so the agreement is not carried by any single concept.

**Agreement falls steeply as the threshold tightens**, from 0.64 to 0.07 in GPT-2 and 0.66 to 0.16 in Qwen3. This is expected and is not a failure: at strict thresholds most concept pairs share no experts at all, and a nearly empty Jaccard matrix cannot track a full geometry. It does set a practical ceiling on how strict a threshold can be before the expert set stops describing the model, which is useful alongside module 1's zero-expert findings.

**The middle layer is not better than the rest.** The middle-third contrast never reaches significance at any threshold, and is slightly negative at the permissive end. This does not refute the layer-probing literature, which concerns downstream task performance rather than representational geometry, and it should not be read as evidence against it. The reason it cannot adjudicate the question is the stability measure above: with cross-layer similarity at 0.93 there is barely any depth variation for a middle-layer advantage to show up in. For this comparison the choice of layer is close to immaterial, which is itself a useful negative result, and it means the pipeline's use of one analysis sublayer costs nothing here.

**The density association weakens as the threshold tightens in GPT-2**, from 0.95 at AP 0.5 to 0.43 and 0.50 at AP 0.8 and 0.9, so the more selective the expert set, the less its depth profile is governed by raw expert counts. Qwen3 does not show this: its density association is lower to begin with (0.66) and rises rather than falls (0.92 at AP 0.8). The pattern is therefore architecture-specific and should not be stated as a general property of thresholding.

### Per-sublayer comparison (section 8.5), both models at AP 0.5

| Model | Sublayer | Expert units | $\rho$ full | $\rho$ count-matched |
|---|---|---|---|---|
| GPT-2 | mlp.c_fc | 25,251 | 0.641 | 0.308 |
| GPT-2 | attn.c_attn | 17,659 | 0.583 | 0.307 |
| GPT-2 | attn.c_proj | 6,064 | 0.563 | 0.445 |
| GPT-2 | mlp.c_proj | 5,556 | 0.241 | 0.241 |
| Qwen3 | mlp.gate_proj | 97,400 | 0.660 | 0.325 |
| Qwen3 | mlp.up_proj | 91,665 | 0.651 | 0.370 |
| Qwen3 | self_attn.o_proj | 35,521 | 0.491 | 0.360 |
| Qwen3 | self_attn.v_proj | 12,442 | 0.450 | 0.423 |
| Qwen3 | self_attn.k_proj | 10,662 | 0.443 | 0.443 |
| Qwen3 | self_attn.q_proj | 19,710 | 0.429 | 0.341 |
| Qwen3 | mlp.down_proj | 29,226 | 0.263 | 0.154 |

The count-matched column is the one to read, and it changes the ranking substantially. On raw $\rho$ the analysis sublayer leads in both models, but once every sublayer is cut to the same number of experts (`k` set by the smallest sublayer, 13,482 for GPT-2 and 32,483 for Qwen3) the lead disappears: GPT-2's `mlp.c_fc` falls from 0.641 to 0.308, below `attn.c_proj` at 0.445, and Qwen3's `mlp.gate_proj` falls from 0.660 to 0.325, below `self_attn.k_proj` at 0.443 and `v_proj` at 0.423. **Most of the analysis sublayer's apparent advantage in this module is expert density rather than representational fidelity.** The one robust conclusion that survives matching is that the FFN output projection is genuinely worst in both architectures (GPT-2 `mlp.c_proj` 0.241, Qwen3 `mlp.down_proj` 0.154), which agrees with module 1's category-alignment ranking and module 5's contrast.

Note that this does not undermine the `sublayer_filter` choice, which was made on category alignment (module 1) and expert share, not on RSA fidelity. It does mean module 8's sublayer ranking should be cited from the count-matched column only.

## Outputs

Written to `results/<output_subdir>/AP_<t>/8_embedding_rsa/` for the sublayer under analysis, and to `results/<output_subdir>/AP_<t>/8_embedding_rsa/sublayers/<sublayer_name>/` for every other sublayer of the architecture. Every sublayer, analysis or not, writes its own `embedding_rsa_layer_sweep.csv`, `embedding_rsa_summary.csv`, and `rsa_layer_sweep.png`. Module 8 is skipped with a warning, leaving the rest of the sweep intact, when the embedding cache is absent. Build it with

```
python scripts/precompute_concept_embeddings.py --config gpt2_richie_hsj
```

which takes about 5 minutes for GPT-2 and 30 minutes for Qwen3.

#### `embedding_rsa_layer_sweep.csv`

One row per (`embedding_metric`, layer) swept for that sublayer's own expert set. In the analysis sublayer's own file the sweep is unrestricted, so `sublayer` changes row to row across the whole architecture and `is_analysis_sublayer` is True only on the rows belonging to the analysis sublayer itself, with `n_experts_this_layer` and `expert_share_of_pool_pct` nonzero only there too, since the pool is built from the analysis sublayer alone. In a non-analysis sublayer's own file, the sweep is restricted to that sublayer's own layers, so `sublayer` is constant and `is_analysis_sublayer` is True on every row.

| Name | Meaning | Range or units |
|---|---|---|
| embedding_metric | Which embedding similarity was used, `correlation_zscored` (primary) or `cosine_raw` (robustness) | string |
| layer_idx | Ascending index of the layer in the full layer mapping | int |
| layer_name | Layer identifier, e.g. `3.L.0.mlp.c_fc` | string |
| block | Transformer block number the layer belongs to | int, 0-indexed |
| sublayer | Sublayer name the layer belongs to | string |
| depth_fraction | Block position normalised to network depth | float in [0, 1] |
| depth_third | Depth tercile the layer falls in | "early", "middle", or "late" |
| is_analysis_sublayer | Whether this row belongs to the sublayer under analysis | bool |
| n_concepts | Concepts used in this run | int, fixed per run (196) |
| n_expert_units | Distinct expert units in this sublayer's pooled expert set | int, fixed per sublayer |
| n_experts_this_layer | Expert rows this specific layer contributed to the pool | int, 0 unless this row is an analysis-sublayer row |
| expert_share_of_pool_pct | This layer's share of the pooled expert set | float in [0, 100], 0 unless this row is an analysis-sublayer row |
| rho_expert_vs_embedding | Spearman $\rho$ between the expert Jaccard and this layer's embedding similarity | float in [-1, 1] |
| rho_ci95_lo, rho_ci95_hi | 95% concept-bootstrap interval around rho_expert_vs_embedding | float in [-1, 1] or NaN, analysis-sublayer rows only |
| mantel_p_value | Concept-permutation p-value, primary metric only | float in [0, 1], floor 1e-4 |
| mantel_p_fdr_bh | Benjamini-Hochberg correction of mantel_p_value across the layers of one embedding_metric | float in [0, 1] |
| rho_expert_vs_embedding_holdout | rho_expert_vs_embedding recomputed with this layer's own contribution removed from the pool | float in [-1, 1] or NaN, analysis-sublayer rows only |
| rho_holdout_ci95_lo, rho_holdout_ci95_hi | 95% bootstrap interval around rho_expert_vs_embedding_holdout | float in [-1, 1] or NaN |
| mantel_p_value_holdout | Concept-permutation p-value for the holdout rho | float in [0, 1] or NaN |
| noise_ceiling_splithalf | Split-half, Spearman-Brown corrected reliability of this layer's embedding matrix | float in (0, 1] or NaN |
| rho_over_noise_ceiling | rho_expert_vs_embedding divided by noise_ceiling_splithalf | float, typically in [0, 1] |

Example (head of `embedding_rsa_layer_sweep.csv` in `8_embedding_rsa/`, GPT-2, AP 0.5, run `research_plots_gpt2_richie_hsj_with_sublayer_analysis`, this is the analysis sublayer's own file, hence `sublayer` varying row to row and `is_analysis_sublayer` True only on the `mlp.c_fc` row):

| embedding_metric | layer_idx | layer_name | block | sublayer | depth_fraction | depth_third | is_analysis_sublayer | n_concepts | n_expert_units | n_experts_this_layer | expert_share_of_pool_pct | rho_expert_vs_embedding | rho_ci95_lo | rho_ci95_hi | mantel_p_value | mantel_p_fdr_bh | rho_expert_vs_embedding_holdout | rho_holdout_ci95_lo | rho_holdout_ci95_hi | mantel_p_value_holdout | noise_ceiling_splithalf | rho_over_noise_ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| correlation_zscored | 1 | 1.L.0.attn.c_attn | 0 | attn.c_attn | 0 | early | False | 196 | 25549 | 0 | 0 | 0.6757 | nan | nan | 0.0001 | 0.0001 | nan | nan | nan | nan | 0.9803 | 0.6892 |
| correlation_zscored | 2 | 2.L.0.attn.c_proj | 0 | attn.c_proj | 0 | early | False | 196 | 25549 | 0 | 0 | 0.6857 | nan | nan | 0.0001 | 0.0001 | nan | nan | nan | nan | 0.9864 | 0.6951 |
| correlation_zscored | 3 | 3.L.0.mlp.c_fc | 0 | mlp.c_fc | 0 | early | True | 196 | 25549 | 14805 | 13.29 | 0.7042 | 0.6682 | 0.7374 | 0.0001 | 0.0001 | 0.6889 | 0.6532 | 0.7213 | 0.0001 | 0.9871 | 0.7134 |
| correlation_zscored | 4 | 4.L.0.mlp.c_proj | 0 | mlp.c_proj | 0 | early | False | 196 | 25549 | 0 | 0 | 0.6788 | nan | nan | 0.0001 | 0.0001 | nan | nan | nan | nan | 0.9847 | 0.6893 |
| correlation_zscored | 5 | 5.L.1.attn.c_attn | 1 | attn.c_attn | 0.09091 | early | False | 196 | 25549 | 0 | 0 | 0.6394 | nan | nan | 0.0001 | 0.0001 | nan | nan | nan | nan | 0.9756 | 0.6554 |

A non-analysis sublayer's own file (e.g. `8_embedding_rsa/sublayers/2_attn.c_attn/embedding_rsa_layer_sweep.csv`) instead has `sublayer` constant, always `attn.c_attn`, and `is_analysis_sublayer` True on every row, since that file only sweeps its own layers.

#### `embedding_rsa_summary.csv`

One row per sublayer's own file, the middle layer's RSA with its interval, p-value and ceiling, the peak and its bootstrap-overlap plateau, the thirds contrast, and the two diagnostics that separate a real depth effect from the density confound.

| Name | Meaning | Range or units |
|---|---|---|
| middle_layer_name, middle_layer_idx, middle_block | The layer at the block nearest to half the sublayer's depth | string, int, int |
| middle_spearman_rho | rho_expert_vs_embedding at the middle layer | float in [-1, 1] |
| middle_rho_ci_lo, middle_rho_ci_hi | 95% concept-bootstrap interval at the middle layer | float in [-1, 1] |
| middle_mantel_p | Mantel p at the middle layer | float in [0, 1] |
| middle_noise_ceiling | Split-half ceiling at the middle layer | float in (0, 1] |
| middle_rho_normalized | middle_spearman_rho divided by middle_noise_ceiling | float |
| middle_spearman_rho_loo | Leave-one-out rho at the middle layer | float in [-1, 1] |
| depth_curve_column | Which layer_sweep column depth claims (peak, plateau, thirds) were read from | string, `rho_expert_vs_embedding_holdout` or `rho_expert_vs_embedding` |
| pooled_rho_vs_expert_share_spearman | Spearman association between a layer's pool share and its pooled rho, the within-sublayer density confound | float in [-1, 1] |
| median_cross_layer_emb_similarity | Median Spearman correlation between the embedding matrices of different blocks | float in [-1, 1], near 1 means depth-invariant geometry |
| peak_layer_name, peak_block, peak_depth_frac | The layer with the highest depth-curve rho | string, int, float in [0, 1] |
| peak_spearman_rho | The depth-curve rho at the peak layer | float in [-1, 1] |
| peak_plateau_layers | Semicolon-joined names of every layer whose bootstrap interval overlaps the peak's | string |
| peak_plateau_n_layers | Count of layers in peak_plateau_layers | int |
| peak_plateau_depth_lo, peak_plateau_depth_hi | Depth-fraction span of the plateau | float in [0, 1] |
| mean_rho_early, mean_rho_middle, mean_rho_late | Mean depth-curve rho within each depth third | float in [-1, 1] |
| middle_third_minus_outer_rho | mean_rho_middle minus the mean of the outer thirds | float |
| middle_vs_outer_perm_p | Permutation p for middle_third_minus_outer_rho | float in [0, 1] |
| n_concepts, n_expert_units, n_layers_swept | Run-level counts | int |

Example (one row, GPT-2, AP 0.5, analysis sublayer `mlp.c_fc`, shown as two tables since one table at 29 columns would not be readable):

| middle_layer_name | middle_layer_idx | middle_block | middle_spearman_rho | middle_rho_ci_lo | middle_rho_ci_hi | middle_mantel_p | middle_noise_ceiling | middle_rho_normalized | middle_spearman_rho_loo | depth_curve_column |
|---|---|---|---|---|---|---|---|---|---|---|
| 27.L.6.mlp.c_fc | 27 | 6 | 0.6355 | 0.5965 | 0.6728 | 0.0001 | 0.9801 | 0.6484 | 0.6318 | rho_expert_vs_embedding_holdout |

| pooled_rho_vs_expert_share_spearman | median_cross_layer_emb_similarity | peak_layer_name | peak_block | peak_depth_frac | peak_spearman_rho | peak_plateau_n_layers | mean_rho_early | mean_rho_middle | mean_rho_late | middle_third_minus_outer_rho | middle_vs_outer_perm_p | n_concepts | n_expert_units | n_layers_swept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.965 | 0.9335 | 3.L.0.mlp.c_fc | 0 | 0 | 0.6889 | 12 | 0.6792 | 0.6358 | 0.6609 | -0.03425 | 1 | 196 | 25549 | 48 |

`peak_plateau_layers`, omitted above for width, holds the verbatim value `3.L.0.mlp.c_fc;7.L.1.mlp.c_fc;11.L.2.mlp.c_fc;15.L.3.mlp.c_fc;19.L.4.mlp.c_fc;23.L.5.mlp.c_fc;27.L.6.mlp.c_fc;31.L.7.mlp.c_fc;35.L.8.mlp.c_fc;39.L.9.mlp.c_fc;43.L.10.mlp.c_fc;47.L.11.mlp.c_fc`, a semicolon-joined list, all 12 blocks of `mlp.c_fc`, meaning every block's bootstrap interval overlaps the peak's, the entire depth curve is one indistinguishable plateau at this threshold.

#### `sublayer_comparison.csv`

Written once per AP threshold, in `8_embedding_rsa/` directly, aggregating every sublayer's own middle-layer rho and its count-matched counterpart (section 8.5). Only produced when the architecture has more than one sublayer with surviving expert data at the current threshold.

| Name | Meaning | Range or units |
|---|---|---|
| sublayer | Sublayer name | string |
| is_analysis_sublayer | Whether this row is the sublayer named by the run's `sublayer_filter` | bool |
| n_expert_units | Distinct expert units in this sublayer's pooled expert set | int |
| k_count_matched | Shared assignment-row count every sublayer is trimmed to before the matched comparison | int, the minimum across sublayers |
| rho_full | This sublayer's own middle-layer rho, full expert set | float in [-1, 1] |
| rho_count_matched | This sublayer's middle-layer rho using only its top-k_count_matched experts by AP | float in [-1, 1] |
| noise_ceiling_splithalf | Split-half ceiling at this sublayer's own middle layer | float in (0, 1] |
| rho_over_noise_ceiling | rho_full divided by noise_ceiling_splithalf | float |
| middle_layer_name | This sublayer's own middle layer | string |
| peak_layer_name | This sublayer's own peak layer | string |

Example (all rows, GPT-2, AP 0.5, only 4 sublayers so this is the full file, not just a head):

| sublayer | is_analysis_sublayer | n_expert_units | k_count_matched | rho_full | rho_count_matched | noise_ceiling_splithalf | rho_over_noise_ceiling | middle_layer_name | peak_layer_name |
|---|---|---|---|---|---|---|---|---|---|
| mlp.c_fc | True | 25549 | 13469 | 0.6355 | 0.3098 | 0.9801 | 0.6484 | 27.L.6.mlp.c_fc | 3.L.0.mlp.c_fc |
| attn.c_attn | False | 17841 | 13469 | 0.5705 | 0.3044 | 0.9813 | 0.5814 | 25.L.6.attn.c_attn | 1.L.0.attn.c_attn |
| attn.c_proj | False | 6133 | 13469 | 0.5544 | 0.4393 | 0.9897 | 0.5601 | 26.L.6.attn.c_proj | 42.L.10.attn.c_proj |
| mlp.c_proj | False | 5581 | 13469 | 0.2371 | 0.2371 | 0.9667 | 0.2452 | 28.L.6.mlp.c_proj | 20.L.4.mlp.c_proj |

`mlp.c_proj` has the smallest raw assignment-row count of the four sublayers, so it is the sublayer that defines `k_count_matched` (13,469), nothing is trimmed from it and its `rho_count_matched` equals its `rho_full` exactly (both 0.2371). `mlp.c_fc`, the sublayer with the most experts (25,549 distinct units), drops sharply from 0.6355 to 0.3098 once matched down to equal expert mass, the module's clearest illustration of the density confound the control exists to correct for.

#### `rsa_layer_sweep.png`

One per sublayer directory (the analysis sublayer's own copy in `8_embedding_rsa/`, every other sublayer's own copy under `sublayers/<rank>_<name>/`, the rank being the run-wide expert-count ranking shared with every other module, see module 1 subchapter 1.7). Spearman rho against depth for that sublayer's own layers, with the bootstrap band, the split-half noise ceiling, and the middle third of the network shaded. The pooled curve and the leave-one-out curve are drawn together so how much of the pooled shape is a layer's own self-contribution is visible directly. A grey bar series on a secondary axis shows each layer's share of the pooled expert set. The legend sits below the axes rather than overlapping the curves, and the two y-axes are labelled "(curves)" for the rho axis and "(bars)" for the expert-share axis, to disambiguate them.

#### RDM heatmaps

Written only for the sublayer under analysis, at its own middle (headline) layer, under `8_embedding_rsa/rdms/` for the raw matrix CSVs and directly under `8_embedding_rsa/` for the PNGs. The two similarity matrices, expert Jaccard and embedding similarity, are shown in the same concept order, sorted by category so agreement shows up as matching block structure, on one shared colormap (magma) so the two matrices are visually comparable despite one being expert-set Jaccard and the other embedding correlation. Two variants are written for each matrix and both are kept permanently, `_percentile` (each cell replaced by its percentile rank within that matrix's own off-diagonal distribution, so equal colour intensity means equal relative rank, comparable across the two panels) and `_minmax` (each matrix's own raw autoscale, showing true magnitude within a panel at the cost of cross-panel comparability): `rdm_expert_jaccard_percentile.png`, `rdm_expert_jaccard_minmax.png`, `rdm_embedding_similarity_percentile.png`, `rdm_embedding_similarity_minmax.png`.

#### Pairs scatter

Also written only for the sublayer under analysis, at its own middle layer, directly under `8_embedding_rsa/`. The 19,110 concept pairs behind the headline rho, in three variants, all kept permanently, since a plain hexbin on the raw, zero-inflated Jaccard is hard to read: `rsa_pairs_scatter_binned.png`, `rsa_pairs_scatter_rankrank.png`, and `rsa_pairs_scatter_hexbin.png`.

`rsa_pairs_scatter_binned.png` bins pairs by their raw shared-expert count (0, 1, 2, 3-4, 5-9, 10+), plotting the median embedding similarity and interquartile band per bin, split into same- and different-category lines, the most interpretable of the three. A straight OLS fit and a LOWESS smooth are also drawn per category, fit on the raw (shared-expert-count, embedding-similarity) pairs before binning and evaluated at each bin's representative count, then plotted at the same evenly spaced bin positions as the median line (the x-axis itself is categorical, not the real count, so the fitted curves are computed in real units and only displayed at the categorical tick positions). Because shared-expert count is heavily right-skewed, the straight OLS line stays close to flat while LOWESS tracks the median's rise, visually arguing for the module's choice of a rank-based statistic (Spearman) over a linear one.

`rsa_pairs_scatter_rankrank.png` plots the rank of expert Jaccard against the rank of embedding similarity, one hexagonal density cell per group of concept pairs, magma colormap. This is the literal relationship Spearman rho measures, a diagonal-leaning density means a positive rho, and the compressed low-rank band reflects Jaccard's zero-inflation (many tied pairs at Jaccard 0 all receive similarly low ranks).

`rsa_pairs_scatter_hexbin.png` plots raw expert Jaccard against raw embedding similarity, split into same- and different-category panels, magma colormap, with a binned-median trend, a straight OLS fit (its Pearson r annotated in the legend), and a LOWESS smooth all drawn per panel. The OLS line, extrapolated across the observed Jaccard range, rises past what the data ever reaches (embedding similarity is bounded well under the value a straight fit predicts at high Jaccard), while LOWESS stays within the observed range, the same argument as the binned variant, made on the raw rather than binned scale.

#### `sublayer_comparison.png`

Written once per AP threshold, in `8_embedding_rsa/` directly, alongside `sublayer_comparison.csv`. A ranked horizontal bar chart across the top, sublayers ordered by their full rho, full-expert-set rho and count-matched rho shown as paired bars, expert counts annotated next to each sublayer's bars, and a split-half noise-ceiling reference line. Underneath, one small-multiple depth-profile panel per sublayer, the pooled and held-out rho curves of that sublayer's own sweep, so the ranking and each sublayer's own depth behaviour are visible together in one figure.
