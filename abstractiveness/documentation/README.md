# Abstractiveness experiment documentation

This folder documents the analysis pipeline implemented in [scripts/abstractiveness_executor.py](../scripts/abstractiveness_executor.py). The script reads expert-neuron outputs produced from the response data, merges them with concept metadata, and writes a suite of CSV tables and PNG plots for each AP threshold.

## Overall workflow

The pipeline starts from expert-neuron CSV files stored under the response/result folders. Each run:

1. Loads expertise rows from the relevant result directories.
2. Filters to rows whose Average Precision (AP) score is at or above the chosen threshold.
3. Merges the expert entries with concept metadata from [assets/metadata_150.json](../../assets/metadata_150.json), renaming its `typicality` field to `human_typicality` to distinguish it from the model-computed Cosine Typicality used in module 7.
4. Standardizes layer labels and generates the plots and CSV reports for the seven analysis modules.

The outputs are written into one `AP_{threshold}` subfolder per AP threshold under [results/research_plots_150_revised_executor](../../results/research_plots_150_revised_executor). The current reference run sweeps five thresholds, `AP_0.5`, `AP_0.6`, `AP_0.7`, `AP_0.8`, and `AP_0.9`, and every module page's Results section reports how its findings change across that full sweep, not just at a single threshold.

## Module overview

- Module 1: layer-wise expert distribution, using percentage allocation matrices across layers and abstraction levels.
- Module 2: Shannon entropy analysis, comparing concept-level and category-level concentration with peak/average layer plots.
- Module 3: category-concept similarity, using Jaccard and overlap coefficients between concept and category expert sets.
- Module 4: correlation analysis, linking frequency, Human Typicality, expert count, and similarity metrics with Pearson regression plots.
- Module 5: pairwise heatmaps of expert overlap, using Jaccard and overlap matrices across all concepts.
- Module 6: dual category definitions with Jensen-Shannon divergence, comparing category-label prototypes to member-exemplar distributions.
- Module 7: empirical cosine typicality, comparing concepts to global and per-layer category prototypes.

Each module is described in its own page below.

## What the documentation covers

Each module page follows the same four-part structure:

- **Research question**: the specific question the module is trying to answer, stated before any implementation detail.
- **Method**: how the data is transformed and which statistical or geometric method is used (entropy, Jaccard/overlap, Jensen-Shannon divergence, cosine similarity, Pearson correlation).
- **Outputs**: every CSV (with its input, column types/descriptions, and a real example head as a table) and every plot (with the chart type and exactly which columns map to which axis/legend). Example heads use `AP_0.6` as a single representative threshold, since file structure and column meaning don't change across thresholds.
- **Results**: what the actual numbers and rendered plots show at `AP_0.6`, followed by an "Across AP thresholds" subsection reporting the same statistics at all five thresholds (`AP_0.5` through `AP_0.9`). This matters because several findings are threshold-sensitive: some hold at every threshold (e.g. module 2's category-vs-concept entropy gap, module 7's `weapon`/`container` per-category correlations), some only emerge or strengthen at stricter thresholds (module 6's entropy-vs-divergence correlation), and some lose significance entirely at the strictest threshold (module 4's Jaccard-based correlations, module 7's pooled correlation reverses sign). The Results sections call out effect sizes (not just p-values), sample sizes where they're small enough to matter, and an honest assessment of whether the evidence supports the research question at each threshold, including cases where it doesn't, or only partially does.
