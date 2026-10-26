# Abstractiveness experiment documentation

This folder documents the analysis pipeline implemented in [scripts/abstractiveness_executor.py](../scripts/abstractiveness_executor.py). The executor reads the expert units produced from the response data, merges them with concept metadata, and writes a suite of CSV tables and PNG figures for each AP threshold.

## Overall workflow

The pipeline starts from the per-word `expertise/expertise.csv` files of the self-conditioning expertise stage: for every word, each model unit's Average Precision at telling sentences that contain the word from sentences that do not. Module 1's *From responses to expert sets* documents that procedure and its two consequences, neurons serving several words and words left with no experts at strict thresholds. The executor is driven by a model configuration in `MODEL_CONFIGS`, chosen on the command line with `--config`, which selects the architecture (GPT-2 with 48 layers, or Qwen3-1.7B with 196), the response and metadata files, the typicality column, the analysis sublayer of module 4, the output folder, the corpus frequency module 2 reads (`frequency_corpus`, OpenWebText for GPT-2 and FineWeb for Qwen3), the generated sentence set of module 2's probing-text baseline (`probing_sentences_dir`, under `assets/`) and the embedding cache of module 4. Four configurations exist: `gpt2_richie_hsj` and `qwen3_richie_hsj` on the Richie-HSJ dataset, `qwen3_base_richie_hsj`, which runs Qwen3-1.7B-Base, the pre-trained checkpoint of the same model, on the same sentences, and `gpt2_150` on the older 150-concept dataset, which has no probing-text baseline. For every threshold in `AP_THRESHOLDS` it:

1. Loads the expertise rows and keeps those whose AP is at or above the threshold.
2. Merges them with the concept metadata ([assets/metadata_Richie_HSJ.json](../assets/metadata_Richie_HSJ.json) for the Richie-HSJ runs), renaming its typicality field to `human_typicality` so it is never confused with the model-derived cosine typicality of module 2.
3. Standardizes layer labels to `{layer_idx}.L.{block}.{sublayer}` and builds the analysis scopes, the whole model first and then one scope per sublayer type, named by a rank frozen at the most lenient threshold.
4. Runs modules 1, 2, 3 and 5 once per scope and module 4 once, since module 4 sweeps the sublayers itself, then writes one `sublayer_comparison` table per module or section.

Two switches at the top of the executor limit the cost of a sweep. `WRITE_SUBLAYER_OUTPUTS`, off by default, lets the sublayer scopes of modules 2 and 3 compute only their comparison rows, while modules 1 and 5 always write their sublayer folders. `DETAILED_AP_THRESHOLDS`, currently 0.5, 0.6 and 0.7, restricts the costly detail to those thresholds: module 1's per-concept plots and the sublayer scopes of modules 1, 4 and 5. The module 4 embedding cache is built from the response files the first time module 4 runs without it.

The outputs are written into one `AP_<threshold>` folder per threshold under `results/<output_subdir>`, one folder per module, and for modules 1 and 2 one folder per section:

```
AP_0.5/
├── 1_expert_distribution/     1.1_layer_expert_distribution/  1.2_distribution_shape/  1.3_sublayer_informativeness/
├── 2_similarities/            2.1_category_concept_similarities/  2.2_pairwise_similarities/
│                              2.3_human_similarity_validation/  2.4_typicality/  2.5_frequency_correlations/
│                              layer_profile_measures/
├── 3_dual_category_jsd/
├── 4_embedding_rsa/
└── 5_typicality_prediction/   5.1_typicality_ranking/  5.2_pair_similarity/  5.3_concept_structure/
```

Every figure shares one style set in `utils/plotting.py`: Nimbus Sans, sentence-case titles in regular weight with the statistics on a grey line underneath, no top or right spines, a light horizontal grid, one pair of colours for every correlation panel, one pair for the two abstraction levels, `magma` for similarity heatmaps and `RdBu_r` centred at zero for standard scores.

## Module overview

- [Dataset and metadata provenance](dataset_and_metadata.md): where the human-sourced inputs come from, namely the word list, the pairwise similarity judgments, the typicality columns derived from them and the word frequencies, with coverage, caveats, the metadata schema and how to regenerate it. It stops where module 1's *From responses to expert sets* starts.
- [Module 1, expert distribution](module_1_expert_distribution.md): where along depth each word's experts sit (1.1, allocation profiles, mean and cumulative distributions, per-concept plots, peak layer, average layer and peak dominance gap), how spread they are (1.2, Shannon entropy, Geary's C, each category label against its own members and entropy against expert count), and which sublayer types carry category information (1.3). It also defines the analysis scopes and the two layer axes every other module relies on.
- [Module 2, similarities](module_2_similarities.md): how alike two words' expert sets and layer profiles are, each concept against its category label (2.1, with ExpertLens's domain statistics) and every word pair (2.2), validated against human similarity ratings next to a probing-text baseline (2.3), related to typicality through cosine typicality and every human typicality correlation (2.4), and to word frequency (2.5). It also defines the profile metric registry and the coverage rule of every correlation panel.
- [Module 3, dual category JSD](module_3_dual_category_jsd.md): category-label prototypes against member-exemplar distributions, compared with Jensen-Shannon divergence.
- [Module 4, embedding RSA](module_4_embedding_rsa.md): a second-order RSA correlating the expert-set geometry against the concept-embedding geometry of each layer, with a concept-level Mantel permutation, a split-half noise ceiling, and the density and depth-stability diagnostics needed to read the depth curve, swept over every sublayer.
- [Module 5, typicality prediction and concept structure](module_5_typicality_prediction.md): three studies over one representation. Study A predicts which of two same-category concepts humans rate as more typical and Study B the measured human similarity of a pair, both from one generated feature grid. Study C asks the unsupervised question, through dendrograms with their merge heights and through the node-link concept graph of Fedzechkina's ExpertLens Figure 4.
- [Fixes](fixes.md): logical corrections applied to the pipeline, dated.

## What the documentation covers

Each module page follows the same three-part structure:

- **Research question**: what the module is trying to answer, stated before any implementation detail.
- **Analysis**: one part per analysis, grouped by section for modules 1 and 2. Each first documents the mathematical formulation in LaTeX, defining the matrices, vectors and scalars involved and what the resulting values represent, then describes the data structures generated (every CSV with its column types, descriptions and a real example head) and the plotted variables, citing the same symbols so each column and axis traces back to its formula. Example heads use `AP_0.6` as a representative threshold and name the run they come from, in that run's folder layout.
- **Results**: what the numbers and figures show, with an "Across AP thresholds" part reporting the statistics at all five thresholds. Several findings are threshold-sensitive or model-sensitive, so the Results report effect sizes and sample sizes alongside p-values and say plainly where the evidence does not support the research question. Figures computed before a later change of method are marked as stale or provisional in place, with the change that affects them.

Three markers flag material the current trees cannot reproduce. **Not generated by the current code** marks an output, figure, diagnostic or analysis that a page still describes for the record although the current code no longer writes it, because its generator line is commented out, was removed, or belonged to a one-off check. **Legacy** marks everything taken from the older 150-concept dataset: its GPT-2 run `research_plots_150_revised_executor_again` was written by an earlier version of the code, in the folder layout of that time, and is no longer kept under `results/`, and the responses of the `gpt2_150` config are no longer under `responses/`, so none of its numbers, tables or example heads can be regenerated as the repository stands. **Earlier definition** marks a value of the reference trees whose coefficient or test the consolidation of 19 October 2026 (see *Correlations and permutation tests*) redefined, so the current code computes a different number from the same data. Where it is known, the value under the current definition follows, taken from the nopad trees at AP 0.6.

The results this documentation defines are `results/research_plots_gpt2_richie_hsj_restructured_final`, written on 2 October 2026 with modules 1 and 2 rerun on 7 October after their last changes, and `results/research_plots_qwen_base_richie_hsj_FINAL`, the full sweep of Qwen3-1.7B-Base finished on 8 October 2026, both written by the current code with the layout above for all five thresholds. Wherever a module page says Qwen3 it means that pre-trained checkpoint. The post-trained Qwen3-1.7B, whose tree is `results/research_plots_qwen_richie_hsj_restructured_final`, is not cited by the module pages, and its values appear only in `fixes.md`, in the Base against post-trained comparison and in the fix records that predate the Base checkpoint. Each tree's `AP_<t>/main.log` records its runs, and `sublayer_rank.csv` at its root names every `sublayers/` folder. Older runs appear only where a page names them: the GPT-2 150-concept reference run `research_plots_150_revised_executor_again`, a run no longer kept under `results/`, behind module 1's opening results, the 150-concept comparisons and some example heads, all marked **Legacy**, two GPT-2 runs behind module 4's example heads and module 5's restructure check, and the historical GPT-2 measurements of module 5's limitations and retired MDS pass, made before the `squash` homonyms were split.

## Layout history

The five-module layout above was reached in late September 2026 in three steps, each verified to change no number, and the explanations that only the removed code comments held survive in git history at commit 38336e6.

1. The six original modules that describe the shape and properties of the data (old modules 1 to 5 and 7) were merged into a one-module and a two-module candidate beside the untouched originals. The merged files were generated from the originals' source through the AST, so only the section runners, the split of old module 4 and the entry points were written by hand. On identical inputs, GPT-2 and Qwen3 at AP 0.5 and 0.9, every CSV of both candidates was byte identical to the original layout's. Against the August trees `research_plots_gpt2_richie_hsj_sensefix` and `research_plots_qwen_richie_hsj_restructured` every layout differed in the same files, all from inputs rather than code: the metadata had split `squash` into two homonyms, the human-similarity loader had changed its rated pairs, and the Qwen3 responses had been regenerated on 26 September.
2. The two-module layout was adopted, the originals and the one-module candidate removed, the shared code split by purpose into `modules/shared_*_measures.py`, `core/` and `utils/`, and the tests rebuilt on one template. On GPT-2 at AP 0.5 and 0.9, 1,670 of 1,670 CSVs were byte identical to the pre-restructure tree. Module 2's per-metric layer-profile outputs were then grouped under `layer_profile_measures/` and its typicality and frequency panels merged into one-row grouped figures, with 1,182 CSVs byte identical and 102 equal in value apart from added columns.
3. The analyses were regrouped by question into sections 1.1 to 1.3 and 2.1 to 2.5, old modules 6, 8 and 9 renumbered 3, 4 and 5, and every figure moved to the house style. On GPT-2 at AP 0.5 the final run matched `research_plots_gpt2_richie_hsj_restructured_final` on all 1,081 CSVs.

Results trees written before step 3 use the old folder names (`6_dual_category_jsd`, `8_embedding_rsa`, `9_typicality_prediction` and the old section folders of modules 1 and 2), and trees written before the squash key fix of [fixes](fixes.md) lack the rated pairs of the two `squash` concepts.

## Shared notation

The module pages reuse a common mathematical setup, stated here once. After AP filtering at threshold $\tau$, the retained expert data is a table of (concept, layer, unit) rows. From it the modules build:

- $\mathcal{C}$: the set of all analyzed items, split by abstraction level into broad categories $\mathcal{C}_1$ (level 1, the category-label words) and specific concepts $\mathcal{C}_2$ (level 2). For a category $k$, $M_k \subseteq \mathcal{C}_2$ denotes its member concepts.
- $L$: the number of bins of the analyzed layer axis, which depends on the scope. The whole-model scope is read on the block axis, one bin per transformer block (12 for GPT-2, 28 for Qwen3), and on the flat layer axis (48 and 196). A sublayer scope holds one layer per block, with layers keeping their absolute whole-model indices $\ell$.
- $E_c$: the set of expert units retained for item $c$, each a (layer, unit) pair, with $n_c = |E_c|$ its expert count. The sets of different words overlap, since a neuron is typically an expert for several words, and $E_c$ can be empty at strict thresholds, in which case the word is absent from every table.
- $N$: the concept-by-layer count matrix, where $N_{c\ell}$ is the number of expert rows of item $c$ in layer $\ell$, so $n_c = \sum_{\ell} N_{c\ell}$.
- $p_c$: the layer probability distribution of item $c$, $p_{c\ell} = N_{c\ell} / n_c$, with $\sum_\ell p_{c\ell} = 1$.

Each module page restates the pieces of this notation it uses, plus its own definitions.

## Correlations and permutation tests

Every correlation the pipeline reports is Spearman's coefficient, and every p-value is a two-sided permutation p. Both come from `spearman_test` in `utils/statistics.py`, which modules 1 to 5 share, with `correlation` for the coefficient alone. The consolidation of 19 October 2026, proposed in `temp_docs/correlation_methods_inventory.md`, replaced the t-based p of `scipy.stats.spearmanr`, the Pearson side columns, the category indicators of the partial correlations, the simple and pooled means over categories, the one-sided tests and the generators shared across tests.

**The coefficient.** For $n$ items with values $x_1, \dots, x_n$, let $R(x_i)$ be the average rank of $x_i$, ties sharing the mean of their ranks, and

$$z_i(x) = \frac{R(x_i) - \bar{R}}{s_R(x)}, \qquad \bar{R} = \frac{n + 1}{2},$$

its rank score, with $s_R(x)$ the population standard deviation of the ranks of $x$. Spearman's coefficient is

$$\rho(x, y) = \frac{1}{n}\sum_{i=1}^{n} z_i(x)\, z_i(y) \in [-1, 1].$$

For controls $c_1, \dots, c_m$, let $e_i(x)$ and $e_i(y)$ be the residuals of $z_i(x)$ and $z_i(y)$ after a least-squares regression on an intercept and $z(c_1), \dots, z(c_m)$. The partial coefficient is

$$\rho(x, y \mid c_1, \dots, c_m) = \frac{\sum_{i=1}^{n} e_i(x)\, e_i(y)}{\sqrt{\sum_{i=1}^{n} e_i(x)^{2}\,\sum_{i=1}^{n} e_i(y)^{2}}},$$

which with one control $c$ equals the closed form $(\rho_{xy} - \rho_{xc}\,\rho_{yc}) / \sqrt{(1 - \rho_{xc}^2)(1 - \rho_{yc}^2)}$. Category never enters as a regressor. It enters through the set of items over which the ranks are taken.

**Four modalities.** A variable is *category-relative* when its value is defined with respect to the concept's category: human typicality, cosine typicality, the Jaccard index, overlap and layer-profile agreement of a concept with its category label, and every human pair rating. Expert count, entropy, Geary's C and the other layer descriptors, and corpus frequency are properties of the concept alone. A correlation involving at least one category-relative variable is computed within categories, any other over all items (`CATEGORY_RELATIVE` in module 2 lists the columns).

- **Pooled.** The coefficient over all items, numeric controls only, tested by reassigning the values of $x$ over all items.
- **Within category.** In category $k$, the coefficient $\rho_k$ over its $n_k$ members or rated pairs, the ranks taken and any control fitted inside $k$. The reported value is

  $$\rho_W = \sum_{k=1}^{K} \frac{n_k}{N}\,\rho_k, \qquad N = \sum_{k=1}^{K} n_k,$$

  which equals $\rho$ computed on rank scores standardised within each category, so the least-squares slope of those scores, kept in each panel's CSV, is $\rho_W$. A category with fewer than $3 + m$ items, or whose $\rho_k$ is undefined because a variable is constant inside it, is left out of the sum and of $N$. The test reassigns the values of $x$ among the members of each category, every category at once, which gives one p for $\rho_W$ and one per category, and for pairs it reassigns the member labels of each category's matrix, a stratified Mantel test.
- **Matrix.** The coefficient over the $n(n-1)/2$ off-diagonal entries of two concept-by-concept matrices, tested by a Mantel test that reassigns the concept labels of one matrix, moving its rows and columns together, which keeps the dependence among pairs sharing a concept. Category alignment is this modality with the same-category indicator as the second matrix, read out as a ROC-AUC, which is an affine function of $\rho$ against the indicator and so shares its p. Relabelling needs complete pairs, so the item missing the most pairs is dropped until every remaining pair is defined.
- **Reliability ceiling.** The Spearman coefficient between two halves of the reference, averaged over the splits when there are several, corrected by the Spearman-Brown formula $\rho^{*} = 2\bar{r} / (1 + \bar{r})$, with agreements reported as $\rho / \rho^{*}$ beside $\rho$: module 4's split-half embedding ceiling and the human ceiling of section 2.3.

**One test.** Every permutation p is

$$p = \frac{1 + \#\left\{b \le B : \lvert T_b \rvert \ge \lvert T \rvert\right\}}{1 + B}, \qquad B = 9{,}999,$$

where $T$ is the observed statistic, $T_b$ its value after the $b$-th reassignment, and a statistic whose expectation under reassignment is not zero is first centred on that expectation. Only $x$, or the labels of one matrix, is reassigned, while $y$ and the controls stay in place. Each test draws from a generator of its own, `numpy.random.default_rng(0)`, and each category of a within-category test from one seeded by the category name, so a p does not depend on the tests that ran before it. When the distinct reassignments number at most $9! = 362{,}880$, as for module 3's eight categories, every one is enumerated and the p is exact. The smallest p of a drawn test is $1 / (B + 1) = 0.0001$, and an exact test reaches lower, $2/8! \approx 5 \times 10^{-5}$ for module 3's eight categories. Benjamini-Hochberg stays where a conclusion rests on a sweep of parallel tests, the layers of module 4.

**Tests that are not correlations** keep their statistics: the Mann-Whitney U of labels against members (module 1), the paired t over held-out categories that compares nested feature cells (module 5, Studies A and B), and three permutation tests with the p above, the separation of groups in a distance space and the within-category share of a link graph's edges (module 5, Study C), and the contrast of the middle third of depth against the outer thirds (module 4).

**Display.** A panel draws what its test compares, with $\rho$, its p and $n$ on the panel's grey detail line. For a pooled coefficient these are the raw values, so the axes keep the scale and the distribution of the data. For a within-category one they are the deviations $x_c - \bar{x}_k$ of each concept $c$ from the mean $\bar{x}_k$ of its category $k$, which keep every within-category order the coefficient ranks (`residuals` in `utils/statistics.py`). For a partial one they are the standardised rank residuals $e(x)$ and $e(y)$ the coefficient correlates, the ranks of each variable minus their least-squares fit on an intercept and the ranks of the controls, fitted within each category when the coefficient is within category, scaled to unit variance. A rank partial cannot be shown on raw values, since the residuals of the raw values order the points differently: on the nopad trees at AP 0.6 their within-category order agreed with the rank residuals at 0.91 to 0.96 only, and a line through them read up to 0.12 below the reported partial. The axis label adds, on a second line, "deviation from category mean", "rank residual (z-score)" or "rank residual within category (z-score)". Over the points runs the least-squares line of the drawn values (`draw_regression` in `utils/plotting.py`), a visual guide, which on a partial panel passes through the origin with slope equal to the reported coefficient. No panel draws a confidence band. A bootstrap of the points treats them as independent, while the concepts cluster by category, every member's Jaccard index is measured against the same label set and typicality comes from ratings among the members, and over the eight categories of module 3 a 95% band covers the true line only about 88% of the time even for independent points. The permutation p, which assumes only that one variable is exchangeable over the concepts or among the members of each category, carries the uncertainty instead. Each panel's CSV keeps the rank scores beside the drawn values, and their least-squares slope reproduces the reported $\rho$, $\rho_W$ or partial coefficient. The FINAL trees, and the nopad trees written before 20 October 2026 (kept as `research_plots_*_nopad_before_2026-10-20`), draw a 95% bootstrap band about every line except the pair panels of section 2.3, and the figures those nopad trees hold from 17 and 18 October 2026, the partial figures of sections 2.4 and 2.5 with category as indicator columns and section 2.3's `similarity_vs_<metric>_ranked.png`, are **Not generated by the current code**. Module 4's descriptive coefficients, its split-half ceiling, bootstrap interval, density confound and depth stability, call `scipy.stats.spearmanr`, which gives the same coefficient.
