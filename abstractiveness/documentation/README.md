# Abstractiveness experiment documentation

This folder documents the analysis pipeline implemented in [scripts/abstractiveness_executor.py](../scripts/abstractiveness_executor.py). The script reads expert-neuron outputs produced from the response data, merges them with concept metadata, and writes a suite of CSV tables and PNG plots for each AP threshold.

## Overall workflow

The pipeline starts from the per-word `expertise/expertise.csv` files produced by the self-conditioning expertise stage: for every word, each model unit's Average Precision (AP) at discriminating sentences that contain the word from sentences that do not (the full response-to-expertise procedure, and its consequences, namely neurons serving multiple words and words ending up with zero experts, are documented in module 1, subchapter 1.1). The executor is driven by a model configuration (`MODEL_CONFIGS`) that selects the architecture (GPT-2, 48 layers, or Qwen3-1.7B, 196 layers), the response/metadata files, the typicality column, and the analysis sublayer. Each run:

1. Loads expertise rows from the relevant result directories.
2. Filters to rows whose AP score is at or above the chosen threshold.
3. Merges the expert entries with the configured concept metadata (e.g. [assets/metadata_150.json](../assets/metadata_150.json) or [assets/metadata_Richie_HSJ.json](../assets/metadata_Richie_HSJ.json)), renaming its typicality field to `human_typicality` to distinguish it from the model-computed Cosine Typicality used in module 7.
4. Standardizes layer labels (`{layer_idx}.L.{block}.{sublayer}`) and runs module 1 on the **full** expert table: whole-model distribution views plus the sublayer-informativeness ranking that scores each projection type's category alignment.
5. Restricts the expert table to the configured analysis sublayer (`sublayer_filter`: `mlp.c_fc` for GPT-2, `mlp.gate_proj` for Qwen3, in each case the winner of module 1's ranking, and `None` disables the restriction) and runs modules 2–7 on the restricted table.

The outputs are written into one `AP_{threshold}` subfolder per AP threshold under `results/<output_subdir>`. Two reference runs are cited throughout the module pages:

- `research_plots_150_revised_executor_again`, the original GPT-2 run on the 150-concept dataset, sweeping five thresholds (`AP_0.5` … `AP_0.9`) on the whole model. The pre-existing analyses' example heads and the "Across AP thresholds" results tables come from this run.
- `research_plots_qwen_richie_hsj_with_sublayer_analysis`, the Qwen3-1.7B run on the Richie-HSJ dataset (204 words) at `AP_0.6`, produced by the current sublayer-restricted pipeline. The newer analyses (cumulative-mass plots, sublayer informativeness, the extra module 2 descriptors and plots, module 4's overlap and entropy panels, module 5's counts matrix) are documented from this run, and a GPT-2 counterpart on the same dataset (`research_plots_gpt2_richie_hsj_with_sublayer_analysis`) covers the full threshold sweep.

## Module overview

- Module 1: layer-wise expert distribution, covering data provenance (responses → expert sets), percentage allocation matrices with their raw counts, cumulative-mass plots (depth-ordered and Pareto-sorted), the sublayer-informativeness ranking (category-alignment AUC with permutation test, Geary's C), and the sublayer handoff to modules 2+.
- Module 2: layer-distribution descriptors per word, namely Shannon entropy, Geary's C, peak layer with dominance gap, and full and trimmed average layer, comparing concept-level and category-level concentration, depth, and profile shape, with the entropy-vs-expert-count dependence discussed explicitly.
- Module 3: category-concept similarity, using Jaccard and overlap coefficients between concept and category expert sets, with the raw set sizes stored beside every percentage.
- Module 4: correlation analysis, linking frequency, Human Typicality, expert count, entropy, and similarity metrics with Pearson regression plots, plus a frequency-controlled partial correlation, an entropy-vs-expert-count check for module 2, and a Jaccard-vs-Cosine-Typicality panel (module 4b, run after module 7).
- Module 5: pairwise heatmaps of expert overlap, using Jaccard and overlap matrices (plus the raw shared-expert counts matrix) across all concepts, with category-block separators.
- Module 6: dual category definitions with Jensen-Shannon divergence, comparing category-label prototypes to member-exemplar distributions.
- Module 7: empirical cosine typicality, comparing concepts to global and per-layer category prototypes.

Each module is described in its own page below.

## What the documentation covers

Each module page follows the same three-part structure:

- **Research question**: the specific question the module is trying to answer, stated before any implementation detail.
- **Analysis**: one subchapter per distinct logical analysis inside the module. Each subchapter first documents the mathematical and logical formulation in LaTeX, defining the matrices, vectors, and scalar quantities involved, the formulas applied to the data, and what the resulting values represent. It then describes the data structures generated (every CSV with its column types, descriptions, and a real example head) and the plotted variables, citing the same LaTeX-defined symbols so that each file column and plot axis can be traced back to its formula. Example heads use `AP_0.6` as a single representative threshold, since file structure and column meaning do not change across thresholds, and each example states which reference run it comes from.
- **Results**: what the actual numbers and rendered plots show at `AP_0.6`, followed by an "Across AP thresholds" subsection reporting the same statistics at all five thresholds (`AP_0.5` through `AP_0.9`) of the 150-concept sweep, and, for the newer analyses, a subsection reporting the Qwen3 Richie-HSJ run. This matters because several findings are threshold-sensitive: some hold at every threshold (e.g. module 2's category-vs-concept entropy gap in the GPT-2 run, module 7's `weapon`/`container` per-category correlations), some only emerge or strengthen at stricter thresholds (module 6's entropy-vs-divergence correlation), some lose significance entirely at the strictest threshold (module 4's Jaccard-based correlations, module 7's pooled correlation reverses sign), and some are model-sensitive (module 2's central entropy hypothesis holds in the GPT-2 150-concept run but not in the Qwen3 Richie-HSJ run). The Results sections call out effect sizes (not just p-values), sample sizes where they're small enough to matter, and an honest assessment of whether the evidence supports the research question at each threshold, including cases where it doesn't, or only partially does.

## Shared notation

The module pages reuse a common mathematical setup, stated here once. After AP filtering at threshold $\tau$, the retained expert data is a table of (concept, layer, unit) rows. From it the modules build:

- $\mathcal{C}$: the set of all analyzed items, split by abstraction level into broad categories $\mathcal{C}_1$ (level 1) and specific concepts $\mathcal{C}_2$ (level 2). For a category $k$, $M_k \subseteq \mathcal{C}_2$ denotes its member concepts.
- $L$: the number of model layers in the analyzed support, 48 for GPT-2 and 196 for Qwen3-1.7B for whole-model views. When the sublayer restriction is active, the support is that projection type's layers only (e.g. 28 `mlp.gate_proj` layers for Qwen3, 12 `mlp.c_fc` layers for GPT-2), with layers keeping their absolute whole-model indices $\ell$.
- $E_c$: the set of expert units retained for item $c$, with $n_c = |E_c|$ its expert count. The sets of different words *overlap*, since a neuron is typically an expert for several words (module 1, subchapter 1.1), and $E_c$ can be empty at strict thresholds, in which case the word is absent from all tables.
- $N$: the concept-by-layer count matrix, where $N_{c\ell}$ is the number of expert rows of item $c$ in layer $\ell$ (so $n_c = \sum_{\ell} N_{c\ell}$).
- $p_c$: the layer probability distribution of item $c$, $p_{c\ell} = N_{c\ell} / n_c$, with $\sum_\ell p_{c\ell} = 1$.

Each module page restates the pieces of this notation it uses, plus its own module-specific definitions.
