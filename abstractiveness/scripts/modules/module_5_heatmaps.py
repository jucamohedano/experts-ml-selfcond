import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from utils.helpers import (save_dataframe, expert_set_overlap_matrices, category_alignment_metrics,
                           scope_out_dir, scope_summary_row, layer_profile_metric_matrices,
                           to_block_axis, ACTIVE_PROFILE_METRICS, DEFAULT_PROFILE_METRIC,
                           profile_metric_label)
from utils.human_similarity import load_human_similarity, human_noise_ceiling
from utils.plot_helpers import (_plot_heatmap_with_leaders, build_category_color_map, fig_width_for,
                                plot_sublayer_comparison_bars)

log = logging.getLogger(__name__)

# Headline metrics for the cross-scope sublayer_comparison table. The two layer-profile
# entries are ROC-AUCs rather than contrasts because ROC-AUC is rank-based, hence immune
# to the same/different pair imbalance and directly comparable to the Jaccard one above.
# Together they answer whether the layer-profile metric carries categorical signal that
# Jaccard does not.
#
# The layer-profile entries report DEFAULT_PROFILE_METRIC alone. This table's question is
# how the SCOPES compare, one number per column per scope, and carrying seven metrics
# through it would triple its width to answer a different question. How the METRICS compare
# within one scope is its own artifact, profile_metric_comparison.csv and its plot.
SUMMARY_LABELS = {
    "category_contrast_pct": "Within minus across category Jaccard %",
    "category_roc_auc": "Category alignment ROC-AUC",
    "within_category_jaccard_pct": "Within-category Jaccard %",
    "across_category_jaccard_pct": "Across-category Jaccard %",
    "layer_profile_roc_auc": f"Category alignment ROC-AUC, layer profile ({DEFAULT_PROFILE_METRIC})",
    "layer_profile_z_roc_auc": f"Category alignment ROC-AUC, layer-profile z ({DEFAULT_PROFILE_METRIC})",
    "human_rho_jaccard": "Human similarity agreement, Jaccard",
    "human_rho_layer_profile": f"Human similarity agreement, layer profile ({DEFAULT_PROFILE_METRIC})",
}


def profile_matrix_key(metric: str) -> str:
    """Matrix key, CSV stem and heatmap stem for one registered metric's agreement."""
    return f"layer_profile_{metric}"


def profile_z_matrix_key(metric: str) -> str:
    """Matrix key, CSV stem and heatmap stem for one metric's count-matched null."""
    return f"layer_profile_{metric}_z"

# Permutations for the within-category Mantel test. Pairs sharing a concept are not
# independent, so an ordinary p-value over hundreds of pairs would be badly anticonservative.
# 999 rather than module 1's 9999 because this test runs 24 times per scope (8 categories by
# 3 metrics) rather than once, and p-value resolution of 0.001 is already finer than any
# claim made from it.
HUMAN_MANTEL_PERMUTATIONS = 999

# Alpha for the star marking a supported bar in human_similarity_by_category.png. The
# stars are descriptive, no correction is applied across the 24 tests a scope runs, so a
# single starred category is weaker evidence than the mark suggests.
HUMAN_SIGNIFICANCE_ALPHA = 0.05

# The coefficient every figure and table in 5.3 reports, named explicitly wherever it is
# shown, since "rho" alone leaves the reader to guess and the choice is not cosmetic. The
# expert metrics are bounded and heavily right-skewed, with a large point mass at exactly 0
# once the threshold tightens, and the ratings are bounded averages of ordinal judgments, so
# a Pearson coefficient would report how LINEAR the relationship is rather than how strong.
#
# Which coefficient to use and how to get a p-value for it are independent choices. The
# parametric p attached to either coefficient assumes the observations are independent, and
# these are not, so significance comes from the Mantel permutation instead (see
# HUMAN_MANTEL_PERMUTATIONS). Measured on Qwen3 at AP 0.6 for the layer profile, the
# parametric Pearson p calls 4 of 8 categories significant and the Mantel test calls 1,
# with professions at parametric p=0.0008 against Mantel p=0.169.
HUMAN_COEFFICIENT = "Spearman"

# Neighbourhood fraction for the LOWESS smooth drawn on the scatter panels. 0.4 matches the
# value plot_helpers.plot_hexbin_with_trends already uses, so the two families of scatter in
# this pipeline smooth at the same scale and can be compared by eye.
LOWESS_FRACTION = 0.4

def human_validated_metrics() -> list:
    """
    Matrices validated against the human ratings, as (key, label), Jaccard plus the
    agreement and null pair of every active profile metric.

    Generated rather than written out so registering a metric extends 5.3 with it
    automatically. Cost is linear in the list: each entry runs one Mantel permutation
    sweep per category, HUMAN_MANTEL_PERMUTATIONS draws each, which is about a second per
    entry per scope on the Richie-HSJ item set.
    """
    metrics = [("jaccard", "Jaccard")]
    for metric in ACTIVE_PROFILE_METRICS:
        label = profile_metric_label(metric)
        metrics.append((profile_matrix_key(metric), f"layer profile, {label}"))
        metrics.append((profile_z_matrix_key(metric), f"layer profile z, {label}"))
    return metrics


HUMAN_VALIDATED_METRICS = human_validated_metrics()

def plot_all_heatmaps(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, heat_dir) -> tuple:
    """
    Calculate pairwise Jaccard, Overlap and layer-profile similarity matrices for all
    concepts. Generates corresponding heatmap visualizations and saves matrix CSVs.

    The first two ask which neurons a pair shares, the layer-profile pair asks whether
    they spread their experts over the layers alike, which is invisible to a set metric.
    The layer-profile question is asked once per entry of ACTIVE_PROFILE_METRICS, each
    getting its own agreement and z matrix, CSV and heatmap, because the metrics formalize
    "same depth allocation" differently and disagree by construction. See the module 3
    docstring for what separates the three families.

    Returns (concepts, jaccard_matrix, profile_matrices) where profile_matrices maps every
    layer-profile matrix key to its array, so the caller can summarize and validate
    without recomputing.
    """
    concepts = concept_metadata['concept'].unique()

    # Color each concept's label by its category (shared map -> same colors as module 3).
    # A root category node (e.g. the "furniture" concept, whose own category is null) is
    # colored as its own category, so each block and its parent label share one color.
    color_map = build_category_color_map(concept_metadata.dropna(subset=["category"])["category"])
    concept_to_cat = concept_metadata.set_index("concept")["category"].to_dict()

    def _concept_color(concept):
        cat = concept_to_cat.get(concept)
        if (cat is None or pd.isna(cat)) and concept in color_map:
            cat = concept
        return color_map.get(cat, "#000000")

    concept_colors = [_concept_color(c) for c in concepts]
    present = set(concept_metadata.dropna(subset=["category"])["category"])
    color_legend = {cat: col for cat, col in color_map.items() if cat in present}

    # Category-block boundaries (concepts are ordered by category, each block led by its
    # root label, which counts as part of its own category).
    def _effective_category(concept):
        cat = concept_to_cat.get(concept)
        if (cat is None or pd.isna(cat)) and concept in color_map:
            cat = concept
        return cat

    effective_categories = [_effective_category(c) for c in concepts]
    category_boundaries = [i for i in range(1, len(concepts))
                           if effective_categories[i] != effective_categories[i - 1]]


    # Experts are keyed on (layer_idx, unit) pairs and held sparsely; see
    # expert_set_overlap_matrices for why the dense concept-by-expert form is avoided.
    intersection, jaccard, overlap = expert_set_overlap_matrices(expert_allocation_df, concepts)
    jaccard_matrix = jaccard * 100
    overlap_matrix = overlap * 100

    # Raw shared-expert counts behind both percentage matrices, for scale. The diagonal
    # holds each concept's own expert-set size.
    counts_df = pd.DataFrame(intersection.astype(int), index=concepts, columns=concepts)
    save_dataframe(counts_df, heat_dir / "shared_expert_counts_matrix.csv", index=True)

    # Layer-profile agreement over the same concept list module 3 and module 4 pass, which
    # the null grid requires (see layer_profile_matrices). The block axis matches modules 3
    # and 4, see check_block_axis_identity.py. One pass over every active metric, sharing
    # the profile build across them.
    by_metric = layer_profile_metric_matrices(to_block_axis(expert_allocation_df), list(concepts),
                                              ACTIVE_PROFILE_METRICS)

    # Dictionary of metrics to streamline saving and plotting. The two set-based matrices
    # first, then one agreement and one z matrix per registered profile metric. Filenames
    # carry the metric name for ALL metrics including the default, so the folder reads as
    # one family: this renames the previous layer_profile_matrix.csv / _heatmap.png pair to
    # its js_distance members, values unchanged.
    matrices = {
        "jaccard": (jaccard_matrix, "Pairwise Jaccard Similarity Index %", "magma"),
        "overlap": (overlap_matrix, "Pairwise Overlap Coefficient %", "magma"),
    }
    profile_matrices = {}
    for metric in ACTIVE_PROFILE_METRICS:
        _, similarity, z = by_metric[metric]
        label = profile_metric_label(metric)
        profile_matrices[profile_matrix_key(metric)] = similarity
        profile_matrices[profile_z_matrix_key(metric)] = z
        matrices[profile_matrix_key(metric)] = (
            similarity, f"Pairwise Layer-Profile Agreement, {label}", "magma")
        matrices[profile_z_matrix_key(metric)] = (
            z, f"Pairwise Layer-Profile Agreement vs Count-Matched Null (z), {label}", "magma")

    for name, (mtx, title, cmap) in matrices.items():
        matrix_df = pd.DataFrame(mtx, index=concepts, columns=concepts)
        save_dataframe(matrix_df, heat_dir / f"{name}_matrix.csv", index=True)
        _plot_heatmap_with_leaders(
            mtx, concepts, title, heat_dir / f"{name}_heatmap.png", cmap,
            concept_colors=concept_colors, color_legend=color_legend,
            category_boundaries=category_boundaries,
        )

    return concepts, jaccard_matrix, profile_matrices


def summarize_category_contrast(concepts, similarity_matrix: np.ndarray, concept_metadata: pd.DataFrame) -> dict:
    """
    Reduce a pairwise similarity matrix to the question the heatmaps exist to answer:
    are within-category concept pairs more similar than across-category pairs?

    Restricted to level-2 concepts carrying a category, so the eight category-label words
    (whose own category is null, and which have no same-category peers) never enter the
    pair pool. Reports both group means and their difference, plus the rank-based
    roc_auc from category_alignment_metrics, which is immune to the heavy same/different
    pair imbalance and to Jaccard's skew. Fewer than two categorized concepts leaves
    nothing to contrast, so the metrics come back NaN.

    Returns GENERIC keys (within_pct, across_pct, contrast_pct, roc_auc), which the caller
    renames per matrix.

    Concepts with no usable value are dropped whole, rather than dropping individual
    non-finite PAIRS, because category_alignment_metrics derives its same-category mask
    from the concept list and its permutation test shuffles labels across concepts, both
    of which need the pair pool to stay a complete triangle over one consistent concept
    set. This is a no-op for Jaccard, which is never NaN, and matters for the
    layer-profile matrices, where a word below MIN_PROFILE_EXPERTS is NaN throughout.
    """
    concept_to_cat = concept_metadata.set_index("concept")["category"].to_dict()
    categorized = [i for i, c in enumerate(concepts) if pd.notna(concept_to_cat.get(c))]
    empty = {"within_pct": np.nan, "across_pct": np.nan,
             "contrast_pct": np.nan, "roc_auc": np.nan}
    if len(categorized) < 2:
        return empty

    # Drop concepts carrying non-finite similarities, most-affected first, until the
    # remaining submatrix is a complete finite triangle. A word with no usable profile has
    # an entirely NaN row, so this normally converges in one step.
    keep = list(categorized)
    while len(keep) >= 2:
        block = similarity_matrix[np.ix_(keep, keep)].copy()
        np.fill_diagonal(block, 0.0)  # the z diagonal is NaN by design, never a data loss
        bad = (~np.isfinite(block)).sum(axis=1)
        if bad.max() == 0:
            break
        keep.pop(int(bad.argmax()))
    if len(keep) < 2:
        return empty

    categories = np.array([concept_to_cat[concepts[i]] for i in keep])
    sub_matrix = similarity_matrix[np.ix_(keep, keep)]
    iu = np.triu_indices(len(keep), k=1)
    pair_similarity = sub_matrix[iu]
    same = (categories[:, None] == categories[None, :])[iu]
    if not same.any() or same.all():
        return empty

    alignment = category_alignment_metrics(pair_similarity, categories)
    within, across = pair_similarity[same].mean(), pair_similarity[~same].mean()
    return {"within_pct": within, "across_pct": across,
            "contrast_pct": within - across, "roc_auc": alignment["roc_auc"]}


def _within_category_vectors(concepts: list, matrix: np.ndarray, pairs: pd.DataFrame,
                             category: str, with_experts: set | None = None) -> tuple:
    """
    Model and human vectors for one category's rated pairs, aligned elementwise, plus the
    matrix positions behind them.

    Pairs are dropped for three reasons, all of which appear only at strict thresholds.

    First, a concept can be absent from the rating file or from ``concepts`` entirely.

    Second, a cell can be NaN, which happens when a word holds fewer than 2 experts and so
    has no usable layer profile (see the coverage note in subchapter 5.2). Dropping the NaN
    cells here rather than downstream matters: Spearman would otherwise return NaN for the
    whole category silently, reporting nothing rather than reporting less.

    Third, and only when ``with_experts`` is supplied, a word can hold NO experts at all.
    That case needs its own handling because it does NOT produce a NaN. expert_set_overlap_
    matrices defines Jaccard as 0 when the union is empty, deliberately, so an empty-set
    concept does not poison the heatmap with NaNs. Read as data, though, that 0 asserts
    "these two words share none of their experts", which is a statement about two expert
    sets, and here one of them does not exist. Feeding it to the correlation would score a
    missing measurement as maximal dissimilarity. Measured on Qwen3 at AP 0.9, where 10 of
    205 concepts retain nothing, this affects 114 of 2,391 rated pairs and moves the
    per-category rho by at most 0.026, so it is a correctness fix rather than a result
    change, and it leaves every threshold below 0.9 untouched.

    The genuine zeros, both words holding experts but sharing none, are kept. They are real
    measurements. They do dominate at AP 0.9, where 2,030 of 2,391 pairs are tied at exactly
    0, which is why the Jaccard rho falls there: Spearman has almost no ordering left to read.
    """
    index = {concept: position for position, concept in enumerate(concepts)}
    subset = pairs[pairs.category == category]
    model, human, positions = [], [], []
    for word_a, word_b, rating in zip(subset.word_a, subset.word_b, subset.mean_rating):
        if word_a not in index or word_b not in index:
            continue
        if with_experts is not None and (word_a not in with_experts or word_b not in with_experts):
            continue
        value = matrix[index[word_a], index[word_b]]
        if np.isfinite(value):
            model.append(value)
            human.append(rating)
            positions.append((index[word_a], index[word_b]))
    return np.asarray(model, dtype=float), np.asarray(human, dtype=float), positions


def _mantel_within_category(model: np.ndarray, human: np.ndarray, positions: list,
                            matrix: np.ndarray, rng) -> float:
    """
    Permutation p-value that shuffles CONCEPT LABELS inside the category and rebuilds the
    model vector from the same matrix.

    Permuting labels rather than the pair vector preserves the dependence structure that makes
    pairs non-independent: every pair sharing a concept moves together, exactly as it does in
    the observed data. Shuffling the pair values directly would destroy that and return a
    p-value far too small.
    """
    if len(model) < 3:
        return float("nan")
    observed = abs(spearmanr(model, human).statistic)
    members = np.array(sorted({position for pair in positions for position in pair}))
    slot = {member: index for index, member in enumerate(members)}
    rows = np.array([slot[a] for a, _ in positions])
    columns = np.array([slot[b] for _, b in positions])

    hits, drawn = 0, 0
    for _ in range(HUMAN_MANTEL_PERMUTATIONS):
        shuffled = rng.permutation(members)
        permuted = matrix[shuffled[rows], shuffled[columns]]
        # A permutation can land on cells that are NaN for other pairs (words below the
        # 2-expert floor have no layer profile), so those draws are scored on the cells that
        # remain rather than being counted as a non-exceedance, which would deflate p.
        usable = np.isfinite(permuted)
        if usable.sum() < 3:
            continue
        drawn += 1
        if abs(spearmanr(permuted[usable], human[usable]).statistic) >= observed:
            hits += 1
    if drawn == 0:
        return float("nan")
    return (hits + 1) / (drawn + 1)


def _format_mantel_p(value: float) -> str:
    """
    Mantel p for a panel title, at the resolution the permutation count can support.

    With HUMAN_MANTEL_PERMUTATIONS draws the smallest attainable value is 1/(B+1), so a
    result at the floor is reported as "<" that bound rather than as an exact figure it
    cannot justify.
    """
    floor = 1.0 / (HUMAN_MANTEL_PERMUTATIONS + 1)
    if not np.isfinite(value):
        return "p=n/a"
    if value <= floor:
        return f"p<{floor:.3f}"
    return f"p={value:.3f}"


def plot_human_vs_expert(concepts: list, matrix: np.ndarray, metric_label: str,
                         out_path, with_experts: set | None = None,
                         mantel_p: dict | None = None) -> None:
    """
    One panel per category, expert similarity against the human rating.

    Drawn per category rather than pooled because a pooled cloud hides that categories sit at
    different mean similarity levels, and that offset would read as a correlation which is
    really a between-category difference.

    Each panel title carries the Mantel p beside its rho, taken from the table this figure
    accompanies so the two can never disagree. The permutation p is the only valid one here,
    since the pairs of a category are not independent (every concept appears in m-1 of them)
    and the parametric p attached to a Spearman coefficient would be badly anticonservative.
    A rho without it invites reading a sign that the test does not support, which is exactly
    what the near-zero layer-profile panels look like.
    """
    pairs = load_human_similarity()
    categories = sorted(pairs.category.unique())
    colors = build_category_color_map(categories)
    mantel_p = mantel_p or {}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)

    for axis, category in zip(axes.ravel(), categories):
        model, human, _ = _within_category_vectors(concepts, matrix, pairs, category,
                                                   with_experts)
        if len(model) < 3:
            axis.set_visible(False)
            continue
        axis.scatter(model, human, s=14, alpha=0.45, color=colors[category], edgecolors="none")
        rho = spearmanr(model, human).statistic
        if len(np.unique(model)) > 1:
            # A LOWESS smooth rather than a straight least-squares fit, because the reported
            # statistic is Spearman, which assumes only monotonicity. An OLS line is a
            # Pearson-shaped object: its slope tracks the LINEAR association, so a panel
            # showing one beside a Spearman rho invites reading the line as the illustration
            # of the number when the two can disagree (professions at AP 0.6 is Pearson 0.171
            # against Spearman 0.105). The smooth shows the monotone shape the coefficient
            # actually measures, and it does not flatten the bend these panels display near
            # zero, where the relationship rises steeply and then saturates.
            #
            # The fit stays decoration, so any failure is logged and skipped rather than
            # allowed to take down a sweep that has already produced its numbers. At strict
            # thresholds the surviving values can be nearly collinear or almost all tied.
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(human, model, frac=LOWESS_FRACTION, return_sorted=True)
                axis.plot(smoothed[:, 0], smoothed[:, 1], color="black", linewidth=1.6)
            except Exception as error:
                log.warning(f"    Trend smooth skipped for {category}: {error}")
        axis.set_title(
            f"{category}  {HUMAN_COEFFICIENT} rho={rho:.2f}  "
            f"Mantel {_format_mantel_p(mantel_p.get(category, float('nan')))}  n={len(model)}",
            fontsize=11)
        axis.set_xlabel(f"expert {metric_label}")
    axes[0, 0].set_ylabel("human similarity rating (1-7)")
    axes[1, 0].set_ylabel("human similarity rating (1-7)")
    fig.suptitle(
        f"Expert {metric_label} against human similarity, within-category pairs\n"
        f"{HUMAN_COEFFICIENT} rank correlation, significance by Mantel permutation of "
        f"concept labels ({HUMAN_MANTEL_PERMUTATIONS} draws)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_human_agreement_bars(table: pd.DataFrame, out_path) -> None:
    """
    Agreement as a fraction of the attainable ceiling, grouped by category.

    Plotting rho over ceiling rather than raw rho means a category where the raters disagreed
    with each other is not penalised for the model's inability to predict their noise.
    """
    per_category = table[~table.category.isin(["POOLED", "POOLED_MEAN"])]
    if per_category.empty:
        return
    metrics = list(dict.fromkeys(per_category.metric))
    categories = sorted(per_category.category.unique())
    width = 0.8 / len(metrics)
    # Width scales with the number of BARS, not of categories: one series per validated
    # matrix means Jaccard plus two per registered profile metric, so a figure sized for
    # three series draws fifteen at 0.08 inches each and nothing is readable.
    fig, axis = plt.subplots(
        figsize=(fig_width_for(len(categories) * len(metrics), 0.22, min_w=9.0), 5))

    for offset, metric in enumerate(metrics):
        sub = per_category[per_category.metric == metric].set_index("category")
        values = [sub.loc[c, "rho_over_ceiling"] if c in sub.index else np.nan
                  for c in categories]
        positions = np.arange(len(categories)) + offset * width
        axis.bar(positions, values, width, label=sub.metric_label.iloc[0])
        # A star marks a bar the Mantel test supports, so a tall bar that is really noise is
        # not read as a result. Bars are otherwise indistinguishable on significance.
        for position, category, value in zip(positions, categories, values):
            if category not in sub.index or not np.isfinite(value):
                continue
            p_value = sub.loc[category, "mantel_p"]
            if np.isfinite(p_value) and p_value < HUMAN_SIGNIFICANCE_ALPHA:
                axis.text(position, value + (0.012 if value >= 0 else -0.03), "*",
                          ha="center", va="bottom" if value >= 0 else "top", fontsize=13)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(categories)) + width * (len(metrics) - 1) / 2)
    axis.set_xticklabels(categories, rotation=30, ha="right")
    axis.set_ylabel(f"{HUMAN_COEFFICIENT} rho / noise ceiling")
    axis.set_title("Agreement with human similarity, as a fraction of what raters agree on\n"
                   f"* Mantel p < {HUMAN_SIGNIFICANCE_ALPHA}, "
                   f"{HUMAN_MANTEL_PERMUTATIONS} label permutations within the category",
                   fontsize=11)
    # Legend below the axes and in columns, since one entry per validated matrix overflows
    # a single-column box inside the plot once several profile metrics are registered. It
    # clears the rotated category labels, which occupy the strip directly under the axes.
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False,
                fontsize=9, ncol=min(len(metrics), 4))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate_against_human(concepts: list, matrices: dict, out_dir,
                           make_plots: bool = True,
                           with_experts: set | None = None) -> pd.DataFrame:
    """
    Correlate each expert similarity matrix against the human similarity ratings, per category.

    Humans rated within-category pairs only, so this is necessarily a within-category test. It
    cannot use the large across-category contrast that gives the category alignment ROC-AUC its
    size, which makes it both the harder comparison and the more meaningful one.

    Two pooled figures are reported and both belong in any summary. "POOLED" is the rho over
    all pairs at once, which lets between-category differences in mean similarity contribute.
    "POOLED_MEAN" is the unweighted mean of the per-category rhos, which does not. Quoting only
    the first would overstate the within-category agreement actually being tested.

    Every rho is also divided by that category's split-half noise ceiling, so a category where
    the raters disagreed with each other is not charged for the model's inability to predict
    noise.
    """
    pairs = load_human_similarity()
    ceilings = human_noise_ceiling()
    rng = np.random.default_rng(0)
    rows = []

    for key, label in HUMAN_VALIDATED_METRICS:
        if key not in matrices:
            continue
        matrix = matrices[key]
        pooled_model, pooled_human, per_category_rho = [], [], []

        for category in sorted(ceilings):
            model, human, positions = _within_category_vectors(concepts, matrix, pairs,
                                                               category, with_experts)
            if len(model) < 3:
                continue
            rho = float(spearmanr(model, human).statistic)
            ceiling = ceilings[category]
            rows.append({"metric": key, "metric_label": label, "category": category,
                         "coefficient": HUMAN_COEFFICIENT.lower(), "test": "mantel",
                         "n_pairs": len(model), "rho": round(rho, 4),
                         "mantel_p": round(_mantel_within_category(
                             model, human, positions, matrix, rng), 4),
                         "noise_ceiling": round(ceiling, 4),
                         "rho_over_ceiling": round(rho / ceiling, 4)})
            pooled_model.extend(model)
            pooled_human.extend(human)
            per_category_rho.append(rho)

        if per_category_rho:
            pooled = float(spearmanr(pooled_model, pooled_human).statistic)
            mean_ceiling = float(np.mean([ceilings[c] for c in sorted(ceilings)]))
            # nanmean, because at strict thresholds a category can retain too few finite
            # pairs to yield a correlation, and one such category must not turn the whole
            # summary into NaN.
            mean_rho = float(np.nanmean(per_category_rho))
            for name, value in [("POOLED", pooled), ("POOLED_MEAN", mean_rho)]:
                rows.append({"metric": key, "metric_label": label, "category": name,
                             "coefficient": HUMAN_COEFFICIENT.lower(), "test": "none",
                             "n_pairs": len(pooled_model), "rho": round(value, 4),
                             "mantel_p": float("nan"),
                             "noise_ceiling": round(mean_ceiling, 4),
                             "rho_over_ceiling": round(value / mean_ceiling, 4)})

    table = pd.DataFrame(rows)
    save_dataframe(table, out_dir / "human_similarity_validation.csv")

    if make_plots and not table.empty:
        for key, label in HUMAN_VALIDATED_METRICS:
            if key in matrices:
                per_metric = table[table.metric == key]
                plot_human_vs_expert(
                    concepts, matrices[key], label,
                    out_dir / f"human_vs_expert_similarity_{key}.png",
                    with_experts=with_experts,
                    mantel_p=dict(zip(per_metric.category, per_metric.mantel_p)))
        plot_human_agreement_bars(table, out_dir / "human_similarity_by_category.png")
    return table


# Columns of the per-scope metric comparison, as {column: axis label}. Kept to four so the
# figure stays readable at plot_sublayer_comparison_bars' 4.2 inches per panel.
METRIC_COMPARISON_LABELS = {
    "category_roc_auc": "Category alignment ROC-AUC",
    "category_roc_auc_z": "Category alignment ROC-AUC, z",
    "human_rho": "Human similarity agreement (mean over categories)",
    "human_rho_z": "Human similarity agreement, z",
}


def write_profile_metric_comparison(profile_contrasts: dict, mean_rho: dict, out_dir) -> pd.DataFrame:
    """
    Reduce every active profile metric to one row, so the metrics can be compared inside a
    scope the way sublayer_comparison compares scopes inside a module.

    Two independent readings sit side by side on purpose, and module 8 already showed they
    can disagree: the sublayer best at recovering the category partition was not the one
    best at reproducing human similarity. category_roc_auc asks whether the metric
    separates within-category pairs from across-category ones, a partition question with a
    large contrast behind it. human_rho asks whether it orders WITHIN-category pairs the
    way people do, which is the harder and more externally valid test, and it is reported
    as the unweighted mean over categories rather than the pooled value, since pooling lets
    between-category differences in mean similarity masquerade as agreement.

    The z columns restate both against the count-matched null, which is the comparison to
    trust when ranking metrics: the raw agreement scales differ by metric, deliberately, so
    a metric scoring higher on category_roc_auc than another may simply be reading
    expert-set size more strongly.
    """
    rows = []
    for metric in ACTIVE_PROFILE_METRICS:
        agreement = profile_contrasts[profile_matrix_key(metric)]
        null = profile_contrasts[profile_z_matrix_key(metric)]
        rows.append({
            "metric": metric,
            "metric_label": profile_metric_label(metric),
            "within_pct": agreement["within_pct"],
            "across_pct": agreement["across_pct"],
            "contrast_pct": agreement["contrast_pct"],
            "category_roc_auc": agreement["roc_auc"],
            "category_roc_auc_z": null["roc_auc"],
            "human_rho": mean_rho.get(profile_matrix_key(metric), float("nan")),
            "human_rho_z": mean_rho.get(profile_z_matrix_key(metric), float("nan")),
        })
    table = pd.DataFrame(rows)
    save_dataframe(table, out_dir / "profile_metric_comparison.csv")
    # plot_sublayer_comparison_bars draws its FIRST row in the reference colour, which lands
    # on DEFAULT_PROFILE_METRIC because ACTIVE_PROFILE_METRICS keeps it first. That is the
    # right row to mark, since it is the metric every summary row reports, so the title says
    # so rather than leaving the dark bar looking arbitrary.
    plot_sublayer_comparison_bars(
        table, METRIC_COMPARISON_LABELS, out_dir / "profile_metric_comparison.png",
        "Layer-profile metrics compared, pairwise concept similarity\n"
        f"dark bar is the default metric, {DEFAULT_PROFILE_METRIC}",
        scope_col="metric_label")
    return table


def execute_module_5_heatmaps(scope, concept_metadata: pd.DataFrame, heat_dir) -> dict:
    """Execute Module 5: Heatmaps.
    Generate all pairwise similarity heatmaps and CSV matrices for concepts.

    Purely set-based, so the whole-model scope is the unqualified pairwise geometry and
    each sublayer scope shows whether that same block structure survives in a single
    projection type. Returns the summary row for this module's sublayer_comparison table.
    """
    out_dir = scope_out_dir(heat_dir, scope)
    log.info(f"  [{scope.label}] Generating all pairwise heatmaps and CSV matrices "
             f"(Jaccard, Overlap & {len(ACTIVE_PROFILE_METRICS)} layer-profile metrics)...")
    concepts, jaccard_matrix, profile_matrices = plot_all_heatmaps(
        scope.expert_df, concept_metadata, out_dir)

    jaccard = summarize_category_contrast(concepts, jaccard_matrix, concept_metadata)
    profile_contrasts = {key: summarize_category_contrast(concepts, matrix, concept_metadata)
                         for key, matrix in profile_matrices.items()}

    log.info(f"  [{scope.label}] Validating expert similarity against human ratings...")
    # Concepts retaining at least one expert. A concept with none still occupies a row of
    # the matrices, where its Jaccard reads 0 by definition rather than NaN, so it has to
    # be excluded by name here or it enters the correlation as a false zero.
    with_experts = set(scope.expert_df["concept"].unique())
    human = validate_against_human(
        concepts, {"jaccard": jaccard_matrix, **profile_matrices},
        out_dir, with_experts=with_experts)
    # The unweighted mean over categories, not the pooled value, so the cross-scope summary
    # carries the conservative reading of a within-category test.
    mean_rho = ({} if human.empty else
                human[human.category == "POOLED_MEAN"].set_index("metric")["rho"].to_dict())

    write_profile_metric_comparison(profile_contrasts, mean_rho, out_dir)

    default_profile = profile_contrasts[profile_matrix_key(DEFAULT_PROFILE_METRIC)]
    default_profile_z = profile_contrasts[profile_z_matrix_key(DEFAULT_PROFILE_METRIC)]
    return scope_summary_row(
        scope,
        within_category_jaccard_pct=jaccard["within_pct"],
        across_category_jaccard_pct=jaccard["across_pct"],
        category_contrast_pct=jaccard["contrast_pct"],
        category_roc_auc=jaccard["roc_auc"],
        layer_profile_roc_auc=default_profile["roc_auc"],
        layer_profile_z_roc_auc=default_profile_z["roc_auc"],
        human_rho_jaccard=mean_rho.get("jaccard", float("nan")),
        human_rho_layer_profile=mean_rho.get(
            profile_matrix_key(DEFAULT_PROFILE_METRIC), float("nan")),
    )