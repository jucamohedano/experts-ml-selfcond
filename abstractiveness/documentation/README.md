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

Two markers flag material the current trees cannot reproduce. **Not generated by the current code** marks an output, figure, diagnostic or analysis that a page still describes for the record although the current code no longer writes it, because its generator line is commented out, was removed, or belonged to a one-off check. **Legacy** marks everything taken from the older 150-concept dataset: its GPT-2 run `research_plots_150_revised_executor_again` was written by an earlier version of the code, in the folder layout of that time, and is no longer kept under `results/`, and the responses of the `gpt2_150` config are no longer under `responses/`, so none of its numbers, tables or example heads can be regenerated as the repository stands.

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
