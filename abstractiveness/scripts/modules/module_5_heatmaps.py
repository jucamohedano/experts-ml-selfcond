import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from utils.helpers import (save_dataframe, expert_set_overlap_matrices, category_alignment_metrics,
                           scope_out_dir, scope_summary_row, layer_profile_matrices)
from utils.human_similarity import load_human_similarity, human_noise_ceiling
from utils.plot_helpers import _plot_heatmap_with_leaders, build_category_color_map, fig_width_for

log = logging.getLogger(__name__)

# Headline metrics for the cross-scope sublayer_comparison table. The two layer-profile
# entries are ROC-AUCs rather than contrasts because ROC-AUC is rank-based, hence immune
# to the same/different pair imbalance and directly comparable to the Jaccard one above.
# Together they answer whether the layer-profile metric carries categorical signal that
# Jaccard does not.
SUMMARY_LABELS = {
    "category_contrast_pct": "Within minus across category Jaccard %",
    "category_roc_auc": "Category alignment ROC-AUC",
    "within_category_jaccard_pct": "Within-category Jaccard %",
    "across_category_jaccard_pct": "Across-category Jaccard %",
    "layer_profile_roc_auc": "Category alignment ROC-AUC, layer profile",
    "layer_profile_z_roc_auc": "Category alignment ROC-AUC, layer-profile z",
    "human_rho_jaccard": "Human similarity agreement, Jaccard",
    "human_rho_layer_profile": "Human similarity agreement, layer profile",
}

# Permutations for the within-category Mantel test. Pairs sharing a concept are not
# independent, so an ordinary p-value over hundreds of pairs would be badly anticonservative.
# 999 rather than module 1's 9999 because this test runs 24 times per scope (8 categories by
# 3 metrics) rather than once, and p-value resolution of 0.001 is already finer than any
# claim made from it.
HUMAN_MANTEL_PERMUTATIONS = 999

# Metrics validated against the human ratings, as (key, label).
HUMAN_VALIDATED_METRICS = [("jaccard", "Jaccard"), ("layer_profile", "layer profile"),
                           ("layer_profile_z", "layer profile z")]

def plot_all_heatmaps(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, heat_dir) -> tuple:
    """
    Calculate pairwise Jaccard, Overlap and layer-profile similarity matrices for all
    concepts. Generates corresponding heatmap visualizations and saves matrix CSVs.

    The first two ask which neurons a pair shares, the layer-profile pair asks whether
    they spread their experts over the layers alike, which is invisible to a set metric.
    Returns (concepts, jaccard_matrix, layer_profile_matrix, layer_profile_z_matrix) so
    the caller can summarize without recomputing.
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
    # the null grid requires (see layer_profile_matrices).
    _, profile_matrix, profile_z_matrix = layer_profile_matrices(expert_allocation_df, list(concepts))

    # Dictionary of metrics to streamline saving and plotting
    matrices = {
        "jaccard": (jaccard_matrix, "Pairwise Jaccard Similarity Index %", "magma"),
        "overlap": (overlap_matrix, "Pairwise Overlap Coefficient %", "magma"),
        "layer_profile": (profile_matrix, "Pairwise Layer-Profile Similarity %", "magma"),
        "layer_profile_z": (profile_z_matrix,
                            "Pairwise Layer-Profile Agreement vs Count-Matched Null (z)", "magma"),
    }

    for name, (mtx, title, cmap) in matrices.items():
        matrix_df = pd.DataFrame(mtx, index=concepts, columns=concepts)
        save_dataframe(matrix_df, heat_dir / f"{name}_matrix.csv", index=True)
        _plot_heatmap_with_leaders(
            mtx, concepts, title, heat_dir / f"{name}_heatmap.png", cmap,
            concept_colors=concept_colors, color_legend=color_legend,
            category_boundaries=category_boundaries,
        )

    return concepts, jaccard_matrix, profile_matrix, profile_z_matrix


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
                             category: str) -> tuple:
    """
    Model and human vectors for one category's rated pairs, aligned elementwise, plus the
    matrix positions behind them.

    Pairs are dropped for two reasons, both of which appear at strict thresholds. A concept
    can be missing from this scope's matrix entirely, when it retains no experts, and a cell
    can be NaN, when a word holds fewer than 2 experts and therefore has no usable layer
    profile (see the coverage note in subchapter 5.2). Dropping the NaN cells here rather
    than downstream matters: Spearman would otherwise return NaN for the whole category
    silently, reporting nothing rather than reporting less.
    """
    index = {concept: position for position, concept in enumerate(concepts)}
    subset = pairs[pairs.category == category]
    model, human, positions = [], [], []
    for word_a, word_b, rating in zip(subset.word_a, subset.word_b, subset.mean_rating):
        if word_a in index and word_b in index:
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


def plot_human_vs_expert(concepts: list, matrix: np.ndarray, metric_label: str,
                         out_path) -> None:
    """
    One panel per category, expert similarity against the human rating.

    Drawn per category rather than pooled because a pooled cloud hides that categories sit at
    different mean similarity levels, and that offset would read as a correlation which is
    really a between-category difference.
    """
    pairs = load_human_similarity()
    categories = sorted(pairs.category.unique())
    colors = build_category_color_map(categories)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)

    for axis, category in zip(axes.ravel(), categories):
        model, human, _ = _within_category_vectors(concepts, matrix, pairs, category)
        if len(model) < 3:
            axis.set_visible(False)
            continue
        axis.scatter(model, human, s=14, alpha=0.45, color=colors[category], edgecolors="none")
        rho = spearmanr(model, human).statistic
        if len(np.unique(model)) > 1:
            # A trend line is decoration. At strict thresholds the surviving values can be
            # nearly collinear and the least-squares solve fails to converge, which must not
            # take down a sweep that has already produced its numbers.
            try:
                slope, intercept = np.polyfit(model, human, 1)
                grid = np.linspace(model.min(), model.max(), 50)
                axis.plot(grid, slope * grid + intercept, color="black", linewidth=1.2)
            except np.linalg.LinAlgError:
                log.warning(f"    Trend line skipped for {category}, least squares did not converge.")
        axis.set_title(f"{category}  rho={rho:.2f}  n={len(model)}", fontsize=11)
        axis.set_xlabel(f"expert {metric_label}")
    axes[0, 0].set_ylabel("human similarity rating (1-7)")
    axes[1, 0].set_ylabel("human similarity rating (1-7)")
    fig.suptitle(f"Expert {metric_label} against human similarity, within-category pairs",
                 fontsize=14)
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
    fig, axis = plt.subplots(figsize=(fig_width_for(len(categories), 1.2, min_w=9.0), 5))

    for offset, metric in enumerate(metrics):
        sub = per_category[per_category.metric == metric].set_index("category")
        values = [sub.loc[c, "rho_over_ceiling"] if c in sub.index else np.nan
                  for c in categories]
        axis.bar(np.arange(len(categories)) + offset * width, values, width,
                 label=sub.metric_label.iloc[0])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(categories)) + width * (len(metrics) - 1) / 2)
    axis.set_xticklabels(categories, rotation=30, ha="right")
    axis.set_ylabel("Spearman rho / noise ceiling")
    axis.set_title("Agreement with human similarity, as a fraction of what raters agree on")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate_against_human(concepts: list, matrices: dict, out_dir,
                           make_plots: bool = True) -> pd.DataFrame:
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
            model, human, positions = _within_category_vectors(concepts, matrix, pairs, category)
            if len(model) < 3:
                continue
            rho = float(spearmanr(model, human).statistic)
            ceiling = ceilings[category]
            rows.append({"metric": key, "metric_label": label, "category": category,
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
                             "n_pairs": len(pooled_model), "rho": round(value, 4),
                             "mantel_p": float("nan"),
                             "noise_ceiling": round(mean_ceiling, 4),
                             "rho_over_ceiling": round(value / mean_ceiling, 4)})

    table = pd.DataFrame(rows)
    save_dataframe(table, out_dir / "human_similarity_validation.csv")

    if make_plots and not table.empty:
        for key, label in HUMAN_VALIDATED_METRICS:
            if key in matrices:
                plot_human_vs_expert(concepts, matrices[key], label,
                                     out_dir / f"human_vs_expert_similarity_{key}.png")
        plot_human_agreement_bars(table, out_dir / "human_similarity_by_category.png")
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
             f"(Jaccard, Overlap & layer profile)...")
    concepts, jaccard_matrix, profile_matrix, profile_z_matrix = plot_all_heatmaps(
        scope.expert_df, concept_metadata, out_dir)

    jaccard = summarize_category_contrast(concepts, jaccard_matrix, concept_metadata)
    profile = summarize_category_contrast(concepts, profile_matrix, concept_metadata)
    profile_z = summarize_category_contrast(concepts, profile_z_matrix, concept_metadata)

    log.info(f"  [{scope.label}] Validating expert similarity against human ratings...")
    human = validate_against_human(
        concepts,
        {"jaccard": jaccard_matrix, "layer_profile": profile_matrix,
         "layer_profile_z": profile_z_matrix},
        out_dir)
    # The unweighted mean over categories, not the pooled value, so the cross-scope summary
    # carries the conservative reading of a within-category test.
    mean_rho = ({} if human.empty else
                human[human.category == "POOLED_MEAN"].set_index("metric")["rho"].to_dict())

    return scope_summary_row(
        scope,
        within_category_jaccard_pct=jaccard["within_pct"],
        across_category_jaccard_pct=jaccard["across_pct"],
        category_contrast_pct=jaccard["contrast_pct"],
        category_roc_auc=jaccard["roc_auc"],
        layer_profile_roc_auc=profile["roc_auc"],
        layer_profile_z_roc_auc=profile_z["roc_auc"],
        human_rho_jaccard=mean_rho.get("jaccard", float("nan")),
        human_rho_layer_profile=mean_rho.get("layer_profile", float("nan")),
    )