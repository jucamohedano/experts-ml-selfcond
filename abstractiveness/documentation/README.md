# Abstractiveness experiment documentation

This folder documents the analysis pipeline implemented in [scripts/abstractiveness_executor.py](../scripts/abstractiveness_executor.py). The executor reads the expert units produced from the response data, merges them with concept metadata, and writes a suite of CSV tables and PNG figures for each AP threshold.

## Overall workflow

The pipeline starts from the per-word `expertise/expertise.csv` files of the self-conditioning expertise stage: for every word, each model unit's Average Precision at telling sentences that contain the word from sentences that do not. Module 1's *From responses to expert sets* documents that procedure and its two consequences, neurons serving several words and words left with no experts at strict thresholds. The executor is driven by a model configuration in `MODEL_CONFIGS`, chosen on the command line with `--config`, which selects the architecture (GPT-2 with 48 layers, or Qwen3-1.7B with 196), the response and metadata files, the typicality column, the analysis sublayer of module 4 and the output folder. For every threshold in `AP_THRESHOLDS` it:

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
- [Module 1, expert distribution](module_1_expert_distribution.md): where along depth each word's experts sit (1.1, allocation profiles, mean and cumulative distributions, per-concept plots, peak layer, average layer and peak dominance gap), how spread they are (1.2, Shannon entropy, Geary's C, the profile-shape map and entropy against expert count), and which sublayer types carry category information (1.3). It also defines the analysis scopes and the two layer axes every other module relies on.
- [Module 2, similarities](module_2_similarities.md): how alike two words' expert sets and layer profiles are, each concept against its category label (2.1) and every word pair (2.2), validated against human similarity ratings (2.3), related to typicality through cosine typicality and every human typicality correlation (2.4), and to word frequency (2.5). It also defines the profile metric registry and the coverage rule of every correlation panel.
- [Module 3, dual category JSD](module_3_dual_category_jsd.md): category-label prototypes against member-exemplar distributions, compared with Jensen-Shannon divergence.
- [Module 4, embedding RSA](module_4_embedding_rsa.md): a second-order RSA correlating the expert-set geometry against the concept-embedding geometry of each layer, with a concept-level Mantel permutation, a split-half noise ceiling, and the density and depth-stability diagnostics needed to read the depth curve, swept over every sublayer.
- [Module 5, typicality prediction and concept structure](module_5_typicality_prediction.md): three studies over one representation. Study A predicts which of two same-category concepts humans rate as more typical and Study B the measured human similarity of a pair, both from one generated feature grid. Study C asks the unsupervised question, through dendrograms with their merge heights and through the node-link concept graph of Fedzechkina's ExpertLens Figure 4.
- [Fixes](fixes.md): logical corrections applied to the pipeline, dated.

## What the documentation covers

Each module page follows the same three-part structure:

- **Research question**: what the module is trying to answer, stated before any implementation detail.
- **Analysis**: one part per analysis, grouped by section for modules 1 and 2. Each first documents the mathematical formulation in LaTeX, defining the matrices, vectors and scalars involved and what the resulting values represent, then describes the data structures generated (every CSV with its column types, descriptions and a real example head) and the plotted variables, citing the same symbols so each column and axis traces back to its formula. Example heads use `AP_0.6` as a representative threshold and name the run they come from, in that run's folder layout.
- **Results**: what the numbers and figures show, with an "Across AP thresholds" part reporting the statistics at all five thresholds. Several findings are threshold-sensitive or model-sensitive, so the Results report effect sizes and sample sizes alongside p-values and say plainly where the evidence does not support the research question. Figures computed before a later change of method are marked as stale or provisional in place, with the change that affects them.

The results this documentation defines are `results/research_plots_gpt2_richie_hsj_restructured_final` and `results/research_plots_qwen_richie_hsj_restructured_final`, written by the current code with the layout above for all five thresholds of both architectures. Their `AP_<t>/main.log` records the run, and `sublayer_rank.csv` at their root names every `sublayers/` folder. Results sections not yet refreshed from them say so and name the run they come from, mostly the whole-model Richie-HSJ runs `research_plots_gpt2_richie_hsj_sensefix` and `research_plots_qwen_richie_hsj_sensefix`, while some older example heads and the 150-concept comparisons come from `research_plots_150_revised_executor_again`, a run no longer kept under `results/`.

## Shared notation

The module pages reuse a common mathematical setup, stated here once. After AP filtering at threshold $\tau$, the retained expert data is a table of (concept, layer, unit) rows. From it the modules build:

- $\mathcal{C}$: the set of all analyzed items, split by abstraction level into broad categories $\mathcal{C}_1$ (level 1, the category-label words) and specific concepts $\mathcal{C}_2$ (level 2). For a category $k$, $M_k \subseteq \mathcal{C}_2$ denotes its member concepts.
- $L$: the number of bins of the analyzed layer axis, which depends on the scope. The whole-model scope is read on the block axis, one bin per transformer block (12 for GPT-2, 28 for Qwen3), and on the flat layer axis (48 and 196). A sublayer scope holds one layer per block, with layers keeping their absolute whole-model indices $\ell$.
- $E_c$: the set of expert units retained for item $c$, each a (layer, unit) pair, with $n_c = |E_c|$ its expert count. The sets of different words overlap, since a neuron is typically an expert for several words, and $E_c$ can be empty at strict thresholds, in which case the word is absent from every table.
- $N$: the concept-by-layer count matrix, where $N_{c\ell}$ is the number of expert rows of item $c$ in layer $\ell$, so $n_c = \sum_{\ell} N_{c\ell}$.
- $p_c$: the layer probability distribution of item $c$, $p_{c\ell} = N_{c\ell} / n_c$, with $\sum_\ell p_{c\ell} = 1$.

Each module page restates the pieces of this notation it uses, plus its own definitions.
