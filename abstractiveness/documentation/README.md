# Abstractiveness experiment documentation

This folder documents the analysis pipeline implemented in [scripts/abstractiveness_executor.py](../scripts/abstractiveness_executor.py). The script reads expert-neuron outputs produced from the response data, merges them with concept metadata, and writes a suite of CSV tables and PNG plots for each AP threshold.

## Overall workflow

The pipeline starts from expert-neuron CSV files stored under the response/result folders. Each run:

1. Loads expertise rows from the relevant result directories.
2. Filters to rows whose Average Precision (AP) score is at or above the chosen threshold.
3. Merges the expert entries with concept metadata from [assets/metadata_150.json](../assets/metadata_150.json), renaming its `typicality` field to `human_typicality` to distinguish it from the model-computed Cosine Typicality used in module 7.
4. Standardizes layer labels and generates the plots and CSV reports for the seven analysis modules.

The outputs are written into one `AP_{threshold}` subfolder per AP threshold under [results/research_plots_150_revised_executor_again](../results/research_plots_150_revised_executor_again). The current reference run sweeps five thresholds, `AP_0.5`, `AP_0.6`, `AP_0.7`, `AP_0.8`, and `AP_0.9`, and every module page's Results section reports how its findings change across that full sweep, not just at a single threshold.

## Module overview

- Module 1: layer-wise expert distribution, using percentage allocation matrices across layers and abstraction levels.
- Module 2: Shannon entropy analysis, comparing concept-level and category-level concentration with peak/average layer plots.
- Module 3: category-concept similarity, using Jaccard and overlap coefficients between concept and category expert sets.
- Module 4: correlation analysis, linking frequency, Human Typicality, expert count, and similarity metrics with Pearson regression plots, plus a frequency-controlled partial correlation and a Jaccard-vs-Cosine-Typicality panel (module 4b, run after module 7).
- Module 5: pairwise heatmaps of expert overlap, using Jaccard and overlap matrices across all concepts.
- Module 6: dual category definitions with Jensen-Shannon divergence, comparing category-label prototypes to member-exemplar distributions.
- Module 7: empirical cosine typicality, comparing concepts to global and per-layer category prototypes.

Each module is described in its own page below.

## What the documentation covers

Each module page follows the same three-part structure:

- **Research question**: the specific question the module is trying to answer, stated before any implementation detail.
- **Analysis**: one subchapter per distinct logical analysis inside the module. Each subchapter first documents the mathematical and logical formulation in LaTeX, defining the matrices, vectors, and scalar quantities involved, the formulas applied to the data, and what the resulting values represent. It then describes the data structures generated (every CSV with its column types, descriptions, and a real example head) and the plotted variables, citing the same LaTeX-defined symbols so that each file column and plot axis can be traced back to its formula. Example heads use `AP_0.6` as a single representative threshold, since file structure and column meaning do not change across thresholds.
- **Results**: what the actual numbers and rendered plots show at `AP_0.6`, followed by an "Across AP thresholds" subsection reporting the same statistics at all five thresholds (`AP_0.5` through `AP_0.9`). This matters because several findings are threshold-sensitive: some hold at every threshold (e.g. module 2's category-vs-concept entropy gap, module 7's `weapon`/`container` per-category correlations), some only emerge or strengthen at stricter thresholds (module 6's entropy-vs-divergence correlation), and some lose significance entirely at the strictest threshold (module 4's Jaccard-based correlations, module 7's pooled correlation reverses sign). The Results sections call out effect sizes (not just p-values), sample sizes where they're small enough to matter, and an honest assessment of whether the evidence supports the research question at each threshold, including cases where it doesn't, or only partially does.

## Shared notation

The module pages reuse a common mathematical setup, stated here once. After AP filtering at threshold $\tau$, the retained expert data is a table of (concept, layer, unit) rows. From it the modules build:

- $\mathcal{C}$: the set of all analyzed items, split by abstraction level into broad categories $\mathcal{C}_1$ (level 1) and specific concepts $\mathcal{C}_2$ (level 2). For a category $k$, $M_k \subseteq \mathcal{C}_2$ denotes its member concepts.
- $L = 48$: the number of model layers, indexed $\ell \in \{1, \dots, L\}$.
- $E_c$: the set of expert units retained for item $c$, with $n_c = |E_c|$ its expert count.
- $N \in \mathbb{N}^{|\mathcal{C}| \times L}$: the concept-by-layer count matrix, where $N_{c\ell}$ is the number of expert rows of item $c$ in layer $\ell$ (so $n_c = \sum_{\ell=1}^{L} N_{c\ell}$).
- $p_c \in [0,1]^L$: the layer probability distribution of item $c$, $p_{c\ell} = N_{c\ell} / n_c$, with $\sum_\ell p_{c\ell} = 1$.

Each module page restates the pieces of this notation it uses, plus its own module-specific definitions.
