"""
Module 2: similarities. How alike two words' expert sets and layer profiles are, and how that
relates to human similarity judgments, typicality and frequency. Explanations live in documentation/.

  2.1 category concept similarities   each concept against its category label
  2.2 pairwise similarities           every word pair, matrices, heatmaps, Jaccard against layer profile
  2.3 human similarity validation     pairwise matrices against the human pair ratings
  2.4 typicality                      cosine typicality and every correlation with human typicality
  2.5 frequency correlations          US and Wikipedia frequency against the expert measures
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from utils.logging_and_io import save_dataframe
from modules.shared_expert_set_measures import (expert_set_overlap_matrices,
                                                category_alignment_metrics, pair_similarity_vector)
from modules.shared_layer_profile_measures import (pair_layer_profile_vectors,
                                                   layer_profile_metric_matrices,
                                                   ACTIVE_PROFILE_METRICS, DEFAULT_PROFILE_METRIC,
                                                   SIGNED_PROFILE_METRICS, profile_metric_label,
                                                   profile_metric_value_label)
from core.analysis_scopes import to_block_axis, axis_variants, scope_section_dir, scope_summary_row
from core.expert_data_loading import expert_counts_with_metadata
from core.correlation_reporting import (panel_r_columns, run_regression_panel, run_regression_grid,
                                        regression_row, regression_grid_rows, MIN_ABSOLUTE_N,
                                        MIN_COVERAGE_FRACTION)
from core.human_similarity_ratings import load_human_similarity, human_noise_ceiling
from utils.plotting import (_plot_bar_with_leaders, _plot_heatmap_with_leaders,
                            apply_rotated_leader_labels, build_category_color_map, fig_width_for,
                            plot_hexbin_with_trends, plot_sublayer_comparison_bars, set_title, set_suptitle,
                            stats_line, draw_regression, CORRELATION_COLORS, LEVEL_COLORS, SEQUENTIAL_CMAP,
                            DIVERGING_CMAP)

log = logging.getLogger(__name__)


# Per-metric layer-profile outputs of every section, grouped under one folder per scope.
LAYER_PROFILE_DIR = "layer_profile_measures"


SECTION_DIRS = {
    "2.1": "2.1_category_concept_similarities",
    "2.2": "2.2_pairwise_similarities",
    "2.3": "2.3_human_similarity_validation",
    "2.4": "2.4_typicality",
    "2.5": "2.5_frequency_correlations",
}


# ---------------------------------------------------------------------------
# Shared by the correlation panels of every section
# ---------------------------------------------------------------------------

def _regression(corr_dir, plot_name: str, data: pd.DataFrame, x_col: str, y_col: str, x_label: str,
                y_label: str, title: str, n_total_relevant: int) -> dict:
    """One regression panel with its CSV, or only its correlation_summary row without a folder."""
    if corr_dir is None:
        return regression_row(data, x_col, y_col, plot_name, n_total_relevant)[1]
    return run_regression_panel(data, x_col, y_col, x_label, y_label, title,
                                corr_dir / f"{plot_name}.csv", corr_dir / f"{plot_name}.png", n_total_relevant)


def _write_correlation_summary(rows: list, out_dir) -> pd.DataFrame:
    """A section's correlation_summary table, saved when the section has a folder."""
    summary_df = pd.DataFrame(rows)
    if out_dir is not None:
        save_dataframe(summary_df, out_dir / "correlation_summary.csv")
    return summary_df


def _concept_parent_frame(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Expert counts and metadata joined to the concept-to-label table, empty without that table."""
    if similarity_metrics_df is None or similarity_metrics_df.empty:
        return pd.DataFrame()
    return merged_metadata_df.merge(similarity_metrics_df, on=["concept", "category"], how="inner")


# ---------------------------------------------------------------------------
# 2.1 Category concept similarities: each concept against its category label
# ---------------------------------------------------------------------------

CONCEPT_CATEGORY_SUMMARY_LABELS = {
    "jaccard_mean_pct": "Mean Jaccard %",
    "jaccard_median_pct": "Median Jaccard %",
    "overlap_mean_pct": "Mean overlap %",
    "overlap_median_pct": "Median overlap %",
    "layer_profile_mean_pct": f"Mean layer-profile similarity ({DEFAULT_PROFILE_METRIC})",
    "layer_profile_z_mean": f"Mean layer-profile z vs null ({DEFAULT_PROFILE_METRIC})",
    "r_jaccard_vs_layer_profile": "r, Jaccard vs layer profile (concept to label)",
}


def profile_value_column(metric: str) -> str:
    """Agreement column for one profile metric."""
    return f"layer_profile_{metric}"


def profile_z_column(metric: str) -> str:
    """Count-matched z column for one profile metric."""
    return f"layer_profile_{metric}_z"


def plot_hierarchy_similarities(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, sim_dir,
                                profile_dir, metrics: list = ACTIVE_PROFILE_METRICS) -> pd.DataFrame:
    """Concept-to-label Jaccard, overlap and per-metric layer-profile agreement, as CSV and bar plots."""
    pair_keyed_df = expert_allocation_df.assign(
        layer_unit=list(zip(expert_allocation_df["layer_idx"], expert_allocation_df["unit"]))
    )
    unit_sets = pair_keyed_df.groupby("concept")["layer_unit"].apply(set).to_dict()

    items = list(concept_metadata["concept"].unique())
    item_row = {item: i for i, item in enumerate(items)}
    by_metric = layer_profile_metric_matrices(to_block_axis(expert_allocation_df), items,
                                              metrics)
    profile_jsd = by_metric[DEFAULT_PROFILE_METRIC][0]

    results = []
    for _, row in concept_metadata.dropna(subset=["category"]).iterrows():
        concept, category = row["concept"], row["category"]
        u_concept = unit_sets.get(concept, set())
        u_category = unit_sets.get(category, set())

        if u_concept and u_category:
            intersection = len(u_concept.intersection(u_category))
            union = len(u_concept.union(u_category))
            len_a = len(u_concept)
            len_b = len(u_category)
            i, j = item_row.get(concept), item_row.get(category)
            has_profile = i is not None and j is not None

            entry = {
                "concept": concept,
                "category": category,
                "hierarchy": f"{category} -> {concept}",
                "jaccard_pct": (intersection / union) * 100,
                "overlap_pct": (intersection / min(len_a, len_b)) * 100,
                "shared_expert_units": intersection,
                "concept_expert_units": len_a,
                "category_expert_units": len_b,
                "layer_profile_jsd_bits": profile_jsd[i, j] if has_profile else np.nan,
            }
            for metric in metrics:
                _, similarity, z = by_metric[metric]
                entry[profile_value_column(metric)] = similarity[i, j] if has_profile else np.nan
                entry[profile_z_column(metric)] = z[i, j] if has_profile else np.nan
            results.append(entry)

    similarity_metrics_df = pd.DataFrame(results)
    if similarity_metrics_df.empty: return similarity_metrics_df

    similarity_metrics_df["layer_profile_similarity_pct"] = \
        similarity_metrics_df[profile_value_column(DEFAULT_PROFILE_METRIC)]
    similarity_metrics_df["layer_profile_z"] = \
        similarity_metrics_df[profile_z_column(DEFAULT_PROFILE_METRIC)]

    if sim_dir is None:
        return similarity_metrics_df
    save_dataframe(similarity_metrics_df, sim_dir / "category_concept_similarity_metrics.csv")

    color_map = build_category_color_map(concept_metadata.dropna(subset=["category"])["category"])
    bar_colors = [color_map[c] for c in similarity_metrics_df["category"]]
    present = set(similarity_metrics_df["category"])
    color_legend = {cat: col for cat, col in color_map.items() if cat in present}

    width = fig_width_for(len(similarity_metrics_df), 0.34, min_w=16.0)
    bar_order = list(similarity_metrics_df["hierarchy"])

    for col, title in [("jaccard_pct", "Jaccard index between each concept and its category label"),
                       ("overlap_pct", "Overlap coefficient between each concept and its category label")]:
        _plot_bar_with_leaders(
            plot_dataframe=similarity_metrics_df, x_col="hierarchy", y_col=col,
            title=title, x_label="Category label and concept", y_label="Percent",
            bar_colors=bar_colors, color_legend=color_legend, legend_title="Category",
            tick_label_colors=bar_colors,
            show_x_ticks=True, figsize=(width, 10.14), order=bar_order,
            out_path=sim_dir / f"{col.replace('_pct', '')}_hierarchy.png"
        )

    def _bar_with_zero_line(y_col: str, title: str, y_label: str, out_path) -> None:
        fig, ax = plt.subplots(figsize=(width, 10.14))
        _plot_bar_with_leaders(
            plot_dataframe=similarity_metrics_df, x_col="hierarchy", y_col=y_col,
            title=title, x_label="Category label and concept", y_label=y_label,
            bar_colors=bar_colors, color_legend=color_legend, legend_title="Category",
            tick_label_colors=bar_colors,
            show_x_ticks=True, order=bar_order, ax=ax, save=True
        )
        ax.axhline(0.0, color="#2f2f2f", linewidth=1.4, linestyle="--", zorder=5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    for metric in metrics:
        label = profile_metric_label(metric)
        signed = metric in SIGNED_PROFILE_METRICS
        value_col = profile_value_column(metric)
        title = f"Layer-profile agreement with the category label, {label}"
        if signed:
            _bar_with_zero_line(value_col, title, profile_metric_value_label(metric),
                                profile_dir / f"{value_col}_hierarchy.png")
        else:
            _plot_bar_with_leaders(
                plot_dataframe=similarity_metrics_df, x_col="hierarchy", y_col=value_col,
                title=title, x_label="Category label and concept",
                y_label=profile_metric_value_label(metric),
                bar_colors=bar_colors, color_legend=color_legend, legend_title="Category",
                tick_label_colors=bar_colors,
                show_x_ticks=True, figsize=(width, 10.14), order=bar_order,
                out_path=profile_dir / f"{value_col}_hierarchy.png"
            )

        _bar_with_zero_line(
            profile_z_column(metric),
            f"Layer-profile agreement with the category label against the count-matched null (z), {label}",
            "z (standard deviations above the null)",
            profile_dir / f"{profile_z_column(metric)}_hierarchy.png")

    write_concept_parent_metric_comparison(similarity_metrics_df, profile_dir)
    return similarity_metrics_df


CONCEPT_PARENT_METRIC_LABELS = {
    "mean_agreement": "Mean concept-to-parent agreement",
    "median_agreement": "Median concept-to-parent agreement",
    "mean_z": "Mean z vs count-matched null",
    "share_z_positive": "Share of concepts with z > 0",
}


def write_concept_parent_metric_comparison(similarity_metrics_df: pd.DataFrame, sim_dir) -> pd.DataFrame:
    """One row per profile metric summarizing concept-to-label agreement."""
    rows = []
    for metric in ACTIVE_PROFILE_METRICS:
        agreement = similarity_metrics_df[profile_value_column(metric)]
        z = similarity_metrics_df[profile_z_column(metric)]
        rows.append({
            "metric": metric,
            "metric_label": profile_metric_label(metric),
            "signed": metric in SIGNED_PROFILE_METRICS,
            "n_concepts": int(agreement.notna().sum()),
            "mean_agreement": agreement.mean(),
            "median_agreement": agreement.median(),
            "mean_z": z.mean(),
            "share_z_positive": (z > 0).sum() / z.notna().sum() if z.notna().any() else np.nan,
        })
    table = pd.DataFrame(rows)
    save_dataframe(table, sim_dir / "profile_metric_comparison.csv")
    plot_sublayer_comparison_bars(
        table, CONCEPT_PARENT_METRIC_LABELS, sim_dir / "profile_metric_comparison.png",
        "Layer-profile metrics compared, concept against its category label\n"
        f"dark bar is the default metric, {DEFAULT_PROFILE_METRIC}",
        scope_col="metric_label")
    return table


def plot_jaccard_vs_layer_profile(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame,
                                  concept_metadata: pd.DataFrame, corr_dir) -> list:
    """Jaccard against layer-profile agreement on concept-to-label pairs."""
    corr_data = _concept_parent_frame(merged_metadata_df, similarity_metrics_df)
    if corr_data.empty:
        return []
    return [_regression(
        corr_dir, "jaccard_vs_layer_profile", corr_data, "jaccard_pct", "layer_profile_similarity_pct",
        "Jaccard with category label (%)", "Layer-profile agreement with category label (%)",
        "Jaccard against layer-profile agreement, concept to label", concept_metadata['category'].notna().sum())]


def run_category_concept_similarities(scope, concept_metadata: pd.DataFrame, merged_meta: pd.DataFrame, out_dir,
                                      profile_dir, metrics: list = ACTIVE_PROFILE_METRICS) -> tuple[pd.DataFrame, dict]:
    """Concept-to-label similarities of one scope, returns the table and the summary row."""
    log.info(f"  [{scope.label}] Generating hierarchy similarities (Jaccard & Overlap)...")
    similarity_metrics_df = plot_hierarchy_similarities(scope.expert_df, concept_metadata, out_dir, profile_dir,
                                                        metrics)
    summary_df = _write_correlation_summary(
        plot_jaccard_vs_layer_profile(merged_meta, similarity_metrics_df, concept_metadata, out_dir), out_dir)

    if similarity_metrics_df.empty:
        summary = scope_summary_row(scope, n_pairs=0, **{key: float("nan") for key in CONCEPT_CATEGORY_SUMMARY_LABELS})
    else:
        summary = scope_summary_row(
            scope,
            n_pairs=len(similarity_metrics_df),
            jaccard_mean_pct=similarity_metrics_df["jaccard_pct"].mean(),
            jaccard_median_pct=similarity_metrics_df["jaccard_pct"].median(),
            overlap_mean_pct=similarity_metrics_df["overlap_pct"].mean(),
            overlap_median_pct=similarity_metrics_df["overlap_pct"].median(),
            layer_profile_mean_pct=similarity_metrics_df["layer_profile_similarity_pct"].mean(),
            layer_profile_z_mean=similarity_metrics_df["layer_profile_z"].mean(),
            **panel_r_columns(summary_df),
        )
    return similarity_metrics_df, summary


# ---------------------------------------------------------------------------
# 2.2 Pairwise similarities: every word pair
# ---------------------------------------------------------------------------

PAIRWISE_SIMILARITY_SUMMARY_LABELS = {
    "category_contrast_pct": "Within minus across category Jaccard %",
    "category_roc_auc": "Category alignment ROC-AUC",
    "within_category_jaccard_pct": "Within-category Jaccard %",
    "across_category_jaccard_pct": "Across-category Jaccard %",
    "layer_profile_roc_auc": f"Category alignment ROC-AUC, layer profile ({DEFAULT_PROFILE_METRIC})",
    "layer_profile_z_roc_auc": f"Category alignment ROC-AUC, layer-profile z ({DEFAULT_PROFILE_METRIC})",
    "r_jaccard_vs_layer_profile_allpairs": "r, Jaccard vs layer profile (all pairs)",
    "r_jaccard_vs_layer_profile_z_allpairs": "r, Jaccard vs layer-profile z (all pairs)",
}


def profile_matrix_key(metric: str) -> str:
    """Matrix key and file stem for one profile metric."""
    return f"layer_profile_{metric}"


def profile_z_matrix_key(metric: str) -> str:
    """Matrix key and file stem for one metric's count-matched z."""
    return f"layer_profile_{metric}_z"


def plot_all_heatmaps(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, heat_dir,
                      profile_dir, metrics: list = ACTIVE_PROFILE_METRICS) -> tuple:
    """Pairwise Jaccard, overlap and profile matrices as CSVs and heatmaps."""
    concepts = concept_metadata['concept'].unique()

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

    def _effective_category(concept):
        cat = concept_to_cat.get(concept)
        if (cat is None or pd.isna(cat)) and concept in color_map:
            cat = concept
        return cat

    effective_categories = [_effective_category(c) for c in concepts]
    category_boundaries = [i for i in range(1, len(concepts))
                           if effective_categories[i] != effective_categories[i - 1]]

    intersection, jaccard, overlap = expert_set_overlap_matrices(expert_allocation_df, concepts)
    jaccard_matrix = jaccard * 100
    overlap_matrix = overlap * 100

    counts_df = pd.DataFrame(intersection.astype(int), index=concepts, columns=concepts)
    if heat_dir is not None:
        save_dataframe(counts_df, heat_dir / "shared_expert_counts_matrix.csv", index=True)

    by_metric = layer_profile_metric_matrices(to_block_axis(expert_allocation_df), list(concepts),
                                              metrics)

    matrices = {
        "jaccard": (jaccard_matrix, "Pairwise Jaccard index (%)", SEQUENTIAL_CMAP, None),
        "overlap": (overlap_matrix, "Pairwise overlap coefficient (%)", SEQUENTIAL_CMAP, None),
    }
    profile_matrices = {}
    for metric in metrics:
        _, similarity, z = by_metric[metric]
        label = profile_metric_label(metric)
        profile_matrices[profile_matrix_key(metric)] = similarity
        profile_matrices[profile_z_matrix_key(metric)] = z
        matrices[profile_matrix_key(metric)] = (
            similarity, f"Pairwise layer-profile agreement, {label}", SEQUENTIAL_CMAP, None)
        matrices[profile_z_matrix_key(metric)] = (
            z, f"Pairwise layer-profile agreement against the count-matched null (z), {label}", DIVERGING_CMAP, 0.0)

    for name, (mtx, title, cmap, center) in (matrices.items() if heat_dir is not None else []):
        target = heat_dir if name in ("jaccard", "overlap") else profile_dir
        matrix_df = pd.DataFrame(mtx, index=concepts, columns=concepts)
        save_dataframe(matrix_df, target / f"{name}_matrix.csv", index=True)
        _plot_heatmap_with_leaders(
            mtx, concepts, title, target / f"{name}_heatmap.png", cmap,
            concept_colors=concept_colors, color_legend=color_legend,
            category_boundaries=category_boundaries, center=center,
        )

    return concepts, jaccard_matrix, profile_matrices


def summarize_category_contrast(concepts, similarity_matrix: np.ndarray, concept_metadata: pd.DataFrame) -> dict:
    """Within against across category similarity and its ROC-AUC."""
    concept_to_cat = concept_metadata.set_index("concept")["category"].to_dict()
    categorized = [i for i, c in enumerate(concepts) if pd.notna(concept_to_cat.get(c))]
    empty = {"within_pct": np.nan, "across_pct": np.nan,
             "contrast_pct": np.nan, "roc_auc": np.nan}
    if len(categorized) < 2:
        return empty

    keep = list(categorized)
    while len(keep) >= 2:
        block = similarity_matrix[np.ix_(keep, keep)].copy()
        np.fill_diagonal(block, 0.0)
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


def plot_all_pairs_jaccard_vs_layer_profile(scope, concept_metadata: pd.DataFrame, corr_dir) -> list:
    """Jaccard against layer-profile agreement, raw and z, over every word pair."""
    items = list(concept_metadata["concept"].unique())
    jaccard = pair_similarity_vector(scope.expert_df, items) * 100.0
    profile, profile_z = pair_layer_profile_vectors(to_block_axis(scope.expert_df), items)

    iu = np.triu_indices(len(items), k=1)
    category_of = concept_metadata.set_index("concept")["category"].to_dict()
    cats = np.array([category_of.get(c) for c in items], dtype=object)
    same = (cats[:, None] == cats[None, :])[iu]

    rows = []
    panels = [
        ("jaccard_vs_layer_profile_allpairs", profile,
         "Layer-profile similarity %", "layer_profile_similarity_pct"),
        ("jaccard_vs_layer_profile_z_allpairs", profile_z,
         "Layer-profile agreement vs null (z)", "layer_profile_z"),
    ]
    for plot_name, y, y_label, y_variable in panels:
        valid = np.isfinite(jaccard) & np.isfinite(y)
        pair_frame = pd.DataFrame({"jaccard_pct": jaccard[valid], y_variable: y[valid],
                                   "same_category": same[valid]})
        if corr_dir is not None:
            save_dataframe(pair_frame, corr_dir / f"{plot_name}.csv")

        row = {"plot_name": plot_name, "x_variable": "jaccard_pct", "y_variable": y_variable,
               "pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None,
               "n_points": int(valid.sum()), "n_total_relevant": len(iu[0]),
               "coverage_pct": round(100 * valid.sum() / len(iu[0]), 1) if len(iu[0]) else 0.0}

        if valid.sum() >= MIN_ABSOLUTE_N and np.ptp(jaccard[valid]) > 0 and np.ptp(y[valid]) > 0:
            row["pearson_r"], row["pearson_p"] = stats.pearsonr(jaccard[valid], y[valid])
            row["spearman_rho"], row["spearman_p"] = stats.spearmanr(jaccard[valid], y[valid])
            if corr_dir is not None:
                plot_hexbin_with_trends(
                    jaccard[valid], y[valid], same[valid], corr_dir / f"{plot_name}.png",
                    "Expert Jaccard %", y_label, colorbar_label="Word pairs (log)")
        else:
            log.warning(f"  Skipping {plot_name}: only {int(valid.sum())} usable pairs.")
        rows.append(row)
    return rows


def run_pairwise_similarities(scope, concept_metadata: pd.DataFrame, out_dir, profile_dir,
                              metrics: list = ACTIVE_PROFILE_METRICS) -> tuple[dict, dict]:
    """Pairwise matrices of one scope, their category contrast, and Jaccard against layer profile over all pairs."""
    log.info(f"  [{scope.label}] Generating all pairwise heatmaps and CSV matrices "
             f"(Jaccard, Overlap & {len(metrics)} layer-profile metrics)...")
    concepts, jaccard_matrix, profile_matrices = plot_all_heatmaps(
        scope.expert_df, concept_metadata, out_dir, profile_dir, metrics)

    jaccard = summarize_category_contrast(concepts, jaccard_matrix, concept_metadata)
    profile_contrasts = {key: summarize_category_contrast(concepts, matrix, concept_metadata)
                         for key, matrix in profile_matrices.items()}

    log.info(f"  [{scope.label}] Comparing Jaccard against layer-profile agreement over all word pairs...")
    summary_df = _write_correlation_summary(
        plot_all_pairs_jaccard_vs_layer_profile(scope, concept_metadata, out_dir), out_dir)

    default_profile = profile_contrasts[profile_matrix_key(DEFAULT_PROFILE_METRIC)]
    default_profile_z = profile_contrasts[profile_z_matrix_key(DEFAULT_PROFILE_METRIC)]
    matrices = {"concepts": concepts, "jaccard": jaccard_matrix, "profiles": profile_matrices,
                "contrasts": profile_contrasts}
    return matrices, scope_summary_row(
        scope,
        within_category_jaccard_pct=jaccard["within_pct"],
        across_category_jaccard_pct=jaccard["across_pct"],
        category_contrast_pct=jaccard["contrast_pct"],
        category_roc_auc=jaccard["roc_auc"],
        layer_profile_roc_auc=default_profile["roc_auc"],
        layer_profile_z_roc_auc=default_profile_z["roc_auc"],
        **panel_r_columns(summary_df),
    )


# ---------------------------------------------------------------------------
# 2.3 Human similarity validation: pairwise matrices against the human pair ratings
# ---------------------------------------------------------------------------

HUMAN_VALIDATION_SUMMARY_LABELS = {
    "human_rho_jaccard": "Human similarity agreement, Jaccard",
    "human_rho_layer_profile": f"Human similarity agreement, layer profile ({DEFAULT_PROFILE_METRIC})",
}


HUMAN_MANTEL_PERMUTATIONS = 999


HUMAN_SIGNIFICANCE_ALPHA = 0.05


HUMAN_COEFFICIENT = "Spearman"


def human_validated_metrics() -> list:
    """(key, label) of every matrix validated against human ratings."""
    metrics = [("jaccard", "Jaccard")]
    for metric in ACTIVE_PROFILE_METRICS:
        label = profile_metric_label(metric)
        metrics.append((profile_matrix_key(metric), f"layer profile, {label}"))
        metrics.append((profile_z_matrix_key(metric), f"layer profile z, {label}"))
    return metrics


HUMAN_VALIDATED_METRICS = human_validated_metrics()


def _within_category_vectors(concepts: list, matrix: np.ndarray, pairs: pd.DataFrame,
                             category: str, with_experts: set | None = None) -> tuple:
    """Model and human values of one category's rated pairs, matched on concept keys, NaN and empty sets dropped."""
    index = {concept: position for position, concept in enumerate(concepts)}
    subset = pairs[pairs.category == category]
    model, human, positions = [], [], []
    for word_a, word_b, rating in zip(subset.concept_a, subset.concept_b, subset.mean_rating):
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
    """Mantel p permuting concept labels within the category."""
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
    """Mantel p at the resolution the permutation count supports."""
    floor = 1.0 / (HUMAN_MANTEL_PERMUTATIONS + 1)
    if not np.isfinite(value):
        return "p = n/a"
    if value <= floor:
        return f"p < {floor:.3f}"
    return f"p = {value:.3f}"


def plot_human_vs_expert(concepts: list, matrix: np.ndarray, metric_label: str,
                         out_path, with_experts: set | None = None,
                         mantel_p: dict | None = None) -> None:
    """Per-category scatter of expert similarity against human rating with an isotonic fit."""
    pairs = load_human_similarity()
    categories = sorted(pairs.category.unique())
    mantel_p = mantel_p or {}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharey=True)

    for axis, category in zip(axes.ravel(), categories):
        model, human, _ = _within_category_vectors(concepts, matrix, pairs, category,
                                                   with_experts)
        if len(model) < 3:
            axis.set_visible(False)
            continue
        axis.scatter(model, human, s=12, alpha=0.5, color=CORRELATION_COLORS["points"], edgecolors="none")
        rho = spearmanr(model, human).statistic
        if len(np.unique(model)) > 1:
            try:
                from sklearn.isotonic import IsotonicRegression
                order = np.argsort(model, kind="stable")
                fitted = IsotonicRegression(increasing=bool(rho >= 0),
                                            out_of_bounds="clip").fit_transform(
                                                model[order], human[order])
                axis.plot(model[order], fitted, color=CORRELATION_COLORS["line"], linewidth=1.8)
            except Exception as error:
                log.warning(f"    Trend fit skipped for {category}: {error}")
        set_title(axis, category.capitalize(),
                  f"{HUMAN_COEFFICIENT} rho = {rho:.2f}, Mantel "
                  f"{_format_mantel_p(mantel_p.get(category, float('nan')))}, n = {len(model)}")
        axis.set_xlabel(f"Expert {metric_label}")
    axes[0, 0].set_ylabel("Human similarity rating (1 to 7)")
    axes[1, 0].set_ylabel("Human similarity rating (1 to 7)")
    set_suptitle(fig, f"Expert {metric_label} against human similarity, within-category pairs\n"
                      f"{HUMAN_COEFFICIENT} rank correlation, Mantel p from {HUMAN_MANTEL_PERMUTATIONS} "
                      f"concept-label permutations, isotonic fit")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_human_agreement_bars(table: pd.DataFrame, out_path) -> None:
    """Per-category rho over noise ceiling, starred where the Mantel test supports it."""
    per_category = table[~table.category.isin(["POOLED", "POOLED_MEAN"])]
    if per_category.empty:
        return
    metrics = list(dict.fromkeys(per_category.metric))
    categories = sorted(per_category.category.unique())
    width = 0.8 / len(metrics)
    fig, axis = plt.subplots(
        figsize=(fig_width_for(len(categories) * len(metrics), 0.22, min_w=9.0), 5))

    for offset, metric in enumerate(metrics):
        sub = per_category[per_category.metric == metric].set_index("category")
        values = [sub.loc[c, "rho_over_ceiling"] if c in sub.index else np.nan
                  for c in categories]
        positions = np.arange(len(categories)) + offset * width
        axis.bar(positions, values, width, label=sub.metric_label.iloc[0])
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
    set_title(axis, "Agreement with human similarity as a fraction of the noise ceiling",
              f"* Mantel p < {HUMAN_SIGNIFICANCE_ALPHA}, "
              f"{HUMAN_MANTEL_PERMUTATIONS} label permutations within the category")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False,
                fontsize=9, ncol=min(len(metrics), 4))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate_against_human(concepts: list, matrices: dict, out_dir,
                           make_plots: bool = True,
                           with_experts: set | None = None, plot_dir=None, mantel: bool = True) -> pd.DataFrame:
    """Per-category Spearman against human similarity, Mantel p and noise ceiling, plus pooled rows."""
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
                             model, human, positions, matrix, rng), 4) if mantel else float("nan"),
                         "noise_ceiling": round(ceiling, 4),
                         "rho_over_ceiling": round(rho / ceiling, 4)})
            pooled_model.extend(model)
            pooled_human.extend(human)
            per_category_rho.append(rho)

        if per_category_rho:
            pooled = float(spearmanr(pooled_model, pooled_human).statistic)
            mean_ceiling = float(np.mean([ceilings[c] for c in sorted(ceilings)]))
            mean_rho = float(np.nanmean(per_category_rho))
            for name, value in [("POOLED", pooled), ("POOLED_MEAN", mean_rho)]:
                rows.append({"metric": key, "metric_label": label, "category": name,
                             "coefficient": HUMAN_COEFFICIENT.lower(), "test": "none",
                             "n_pairs": len(pooled_model), "rho": round(value, 4),
                             "mantel_p": float("nan"),
                             "noise_ceiling": round(mean_ceiling, 4),
                             "rho_over_ceiling": round(value / mean_ceiling, 4)})

    table = pd.DataFrame(rows)
    if out_dir is None:
        return table
    save_dataframe(table, out_dir / "human_similarity_validation.csv")

    if make_plots and not table.empty:
        for key, label in HUMAN_VALIDATED_METRICS:
            if key in matrices:
                per_metric = table[table.metric == key]
                target = out_dir if key == "jaccard" else (plot_dir or out_dir)
                plot_human_vs_expert(
                    concepts, matrices[key], label,
                    target / f"similarity_vs_{key.removeprefix('layer_profile_')}.png",
                    with_experts=with_experts,
                    mantel_p=dict(zip(per_metric.category, per_metric.mantel_p)))
        plot_human_agreement_bars(table, out_dir / "human_similarity_by_category.png")
    return table


PAIRWISE_METRIC_LABELS = {
    "category_roc_auc": "Category alignment ROC-AUC",
    "category_roc_auc_z": "Category alignment ROC-AUC, z",
    "human_rho": "Human similarity agreement (mean over categories)",
    "human_rho_z": "Human similarity agreement, z",
}


def write_pairwise_metric_comparison(profile_contrasts: dict, mean_rho: dict, out_dir) -> pd.DataFrame:
    """One row per profile metric, category alignment and human agreement, raw and z."""
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
    plot_sublayer_comparison_bars(
        table, PAIRWISE_METRIC_LABELS, out_dir / "profile_metric_comparison.png",
        "Layer-profile metrics compared, pairwise concept similarity\n"
        f"dark bar is the default metric, {DEFAULT_PROFILE_METRIC}",
        scope_col="metric_label")
    return table


def run_human_similarity_validation(scope, matrices: dict, out_dir, metric_dir, plot_dir) -> dict:
    """Every pairwise matrix against the human pair ratings, then the profile metrics ranked on both criteria."""
    if matrices is None:
        return scope_summary_row(scope, human_rho_jaccard=float("nan"), human_rho_layer_profile=float("nan"))
    log.info(f"  [{scope.label}] Validating expert similarity against human ratings...")
    # A word with no experts reads Jaccard 0 rather than NaN, so it is excluded by name.
    with_experts = set(scope.expert_df["concept"].unique())
    human = validate_against_human(
        matrices["concepts"], {"jaccard": matrices["jaccard"], **matrices["profiles"]},
        out_dir, with_experts=with_experts, plot_dir=plot_dir, mantel=out_dir is not None)
    mean_rho = ({} if human.empty else
                human[human.category == "POOLED_MEAN"].set_index("metric")["rho"].to_dict())

    if metric_dir is not None:
        write_pairwise_metric_comparison(matrices["contrasts"], mean_rho, metric_dir)
    return scope_summary_row(
        scope,
        human_rho_jaccard=mean_rho.get("jaccard", float("nan")),
        human_rho_layer_profile=mean_rho.get(profile_matrix_key(DEFAULT_PROFILE_METRIC), float("nan")),
    )


# ---------------------------------------------------------------------------
# 2.4 Typicality: cosine typicality and every correlation with human typicality
# ---------------------------------------------------------------------------

TYPICALITY_SUMMARY_LABELS = {
    "typicality_pearson_r": "Pearson r, model vs human typicality",
    "n_typicality_concepts": "Concepts with both scores",
    "r_human_typicality_vs_jaccard": "r, human typicality vs Jaccard",
    "r_jaccard_vs_cosine_typicality": "r, Jaccard vs cosine typicality",
}


def _valid_category_members(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame):
    """Categories with at least two members holding experts."""
    present_concepts = set(expert_allocation_df['concept'].unique())
    for category in concept_metadata['category'].dropna().unique():
        cat_meta = concept_metadata[concept_metadata['category'] == category]
        valid_members = [m for m in cat_meta['concept'].tolist() if m in present_concepts]
        if len(valid_members) >= 2:
            yield category, cat_meta, valid_members


def _human_typicality(cat_meta: pd.DataFrame, concept: str) -> float:
    """Human typicality of one concept, NaN when missing."""
    scores = cat_meta[cat_meta['concept'] == concept]['human_typicality'].values
    return scores[0] if len(scores) > 0 else np.nan


def compute_global_typicality(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, typ_dir) -> pd.DataFrame:
    """Cosine of each concept to its category prototype over all (layer, unit) features."""
    allocation = expert_allocation_df.assign(present=1)
    results = []

    for category, cat_meta, valid_members in _valid_category_members(allocation, concept_metadata):
        category_allocation_df = allocation[allocation['concept'].isin(valid_members)].copy()
        category_allocation_df['layer_unit'] = (category_allocation_df['layer_name'].astype(str) + "_"
                                                + category_allocation_df['unit'].astype(str))
        global_pivot = category_allocation_df.pivot_table(index='concept', columns='layer_unit',
                                                          values='present', fill_value=0)
        global_pivot = global_pivot.reindex(valid_members, fill_value=0)
        if global_pivot.empty:
            continue

        centroid_global = global_pivot.mean(axis=0).values.reshape(1, -1)
        sims_global = cosine_similarity(global_pivot.values, centroid_global).flatten()
        for i, concept in enumerate(global_pivot.index):
            results.append({
                'category': category,
                'concept': concept,
                'human_typicality': _human_typicality(cat_meta, concept),
                'global_cosine_typicality': sims_global[i],
            })

    global_typicality_df = pd.DataFrame(results)
    if typ_dir is not None:
        save_dataframe(global_typicality_df, typ_dir / "global_prototype_typicality.csv")
    return global_typicality_df


def compute_layer_typicality(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, typ_dir,
                             suffix: str = "") -> pd.DataFrame:
    """Cosine to the category prototype rebuilt at every layer."""
    allocation = expert_allocation_df.assign(present=1)
    ordered_layers = allocation[['layer_idx', 'layer_name']].drop_duplicates().sort_values('layer_idx')
    results = []

    for category, cat_meta, valid_members in _valid_category_members(allocation, concept_metadata):
        category_allocation_df = allocation[allocation['concept'].isin(valid_members)]

        for _, row in ordered_layers.iterrows():
            layer_df_sub = category_allocation_df[category_allocation_df['layer_idx'] == row['layer_idx']]
            layer_pivot = layer_df_sub.pivot_table(index='concept', columns='unit', values='present', fill_value=0)
            layer_pivot = layer_pivot.reindex(valid_members, fill_value=0)

            if layer_pivot.shape[1] > 0:
                centroid_layer = layer_pivot.mean(axis=0).values.reshape(1, -1)
                sims_layer = cosine_similarity(layer_pivot.values, centroid_layer).flatten()
            else:
                sims_layer = np.zeros(len(valid_members))

            for i, concept in enumerate(layer_pivot.index):
                results.append({
                    'category': category,
                    'concept': concept,
                    'layer_idx': row['layer_idx'],
                    'layer_name': row['layer_name'],
                    'human_typicality': _human_typicality(cat_meta, concept),
                    'layer_cosine_typicality': sims_layer[i],
                })

    layer_typicality_df = pd.DataFrame(results)
    save_dataframe(layer_typicality_df, typ_dir / f"layer_prototype_typicality{suffix}.csv")
    return layer_typicality_df


def generate_global_typicality_reports(global_typicality_df: pd.DataFrame, typ_dir) -> None:
    """Per-category bars and scatter of human against cosine typicality."""
    for category in global_typicality_df['category'].unique():
        cat_dir = typ_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        cat_global = global_typicality_df[global_typicality_df['category'] == category].dropna(subset=['human_typicality', 'global_cosine_typicality']).copy()

        if len(cat_global) > 1:
            cat_global = cat_global.sort_values('human_typicality', ascending=False)
            melted_typicality_df = cat_global.melt(
                id_vars=['concept'],
                value_vars=['human_typicality', 'global_cosine_typicality'],
                var_name='Metric', value_name='Score'
            )
            melted_typicality_df['Metric'] = melted_typicality_df['Metric'].replace({
                'human_typicality': 'Human typicality',
                'global_cosine_typicality': 'Cosine typicality'
            })

            _plot_bar_with_leaders(
                plot_dataframe=melted_typicality_df, x_col='concept', y_col='Score',
                title=f"{category.capitalize()}, human typicality and cosine typicality",
                x_label="Concepts, ordered by human typicality", y_label="Typicality (0 to 1)", legend_title="Measure",
                hue='Metric', palette=[LEVEL_COLORS["Broad Categories"], LEVEL_COLORS["Specific Concepts"]], out_path=cat_dir / f"{category}_typicality_comparison_bar.png",
                figsize=(fig_width_for(cat_global['concept'].nunique(), 0.55, min_w=14.0), 6), show_x_ticks=True
            )

            varied = cat_global['human_typicality'].std() > 0 and cat_global['global_cosine_typicality'].std() > 0
            if varied:
                r, p = pearsonr(cat_global['human_typicality'], cat_global['global_cosine_typicality'])
                detail = stats_line("Pearson r", r, p, len(cat_global))
            else:
                detail = "insufficient variance for a correlation"
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            draw_regression(ax, cat_global, 'human_typicality', 'global_cosine_typicality', fit=varied, size=40)

            for i in range(len(cat_global)):
                ax.text(
                    cat_global['human_typicality'].iloc[i],
                    cat_global['global_cosine_typicality'].iloc[i],
                    f"  {cat_global['concept'].iloc[i]}",
                    fontsize=8, alpha=0.8, va='center'
                )
            set_title(ax, f"{category.capitalize()}, human typicality against cosine typicality", detail)
            ax.set_xlabel("Human typicality")
            ax.set_ylabel("Cosine typicality (global prototype)")
            fig.tight_layout()
            fig.savefig(cat_dir / f"{category}_typicality_correlation_scatter.png", dpi=300)
            plt.close(fig)


def generate_layer_typicality_reports(layer_typicality_df: pd.DataFrame, typ_dir, suffix: str = "") -> None:
    """Per-category most typical concept at each layer, CSV and plot."""
    for category in layer_typicality_df['category'].unique():
        cat_dir = typ_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        cat_layer = layer_typicality_df[layer_typicality_df['category'] == category].copy()
        if not cat_layer.empty:
            idx = cat_layer.groupby('layer_idx')['layer_cosine_typicality'].idxmax()
            top_per_layer = cat_layer.loc[idx, ['layer_idx', 'layer_name', 'concept', 'layer_cosine_typicality', 'human_typicality']]
            top_per_layer.rename(columns={
                'concept': 'most_typical_model_concept',
                'layer_cosine_typicality': 'cosine_similarity_score',
                'human_typicality': 'hardcoded_human_score'
            }, inplace=True)
            top_per_layer = top_per_layer.sort_values('layer_idx')
            save_dataframe(top_per_layer, cat_dir / f"{category}_most_typical_per_layer{suffix}.csv")
            plot_most_typical_concept_per_layer(top_per_layer, category, cat_dir, suffix)


def plot_most_typical_concept_per_layer(top_concepts_per_layer_df: pd.DataFrame, category: str, cat_dir, suffix: str = "") -> None:
    """Most typical concept per layer as a labelled stem plot."""
    fig_w = fig_width_for(len(top_concepts_per_layer_df), 0.28, min_w=16.0)
    fig = plt.figure(figsize=(fig_w, 8))
    plt.vlines(x=range(len(top_concepts_per_layer_df)), ymin=0, ymax=top_concepts_per_layer_df['cosine_similarity_score'],
               color='gray', alpha=0.3, linewidth=2)
    for i, row in top_concepts_per_layer_df.reset_index(drop=True).iterrows():
        plt.scatter(i, row['cosine_similarity_score'], color=LEVEL_COLORS["Broad Categories"], s=20, zorder=3)
        plt.text(
            i, row['cosine_similarity_score'] + 0.02,
            row['most_typical_model_concept'],
            color=LEVEL_COLORS["Specific Concepts"], ha='left', va='bottom',
            rotation=60, fontsize=11
        )

    set_title(plt.gca(), f"{category.capitalize()}, most typical concept per layer")
    plt.xlabel("Model layer", fontsize=14)
    plt.ylabel("Cosine typicality (layer prototype)", fontsize=14)
    apply_rotated_leader_labels(plt.gca(), list(top_concepts_per_layer_df['layer_name']), axis='x', fontsize=9)

    max_y = top_concepts_per_layer_df['cosine_similarity_score'].max()
    plt.ylim(0, max_y + 0.3)
    plt.xlim(-1, len(top_concepts_per_layer_df))

    plt.tight_layout()
    plt.savefig(cat_dir / f"{category}_most_typical_evolution{suffix}.png", dpi=300)
    plt.close(fig)


def _summarize_typicality(global_typicality_df: pd.DataFrame) -> dict:
    """Pooled Pearson r between cosine and human typicality."""
    empty = {"typicality_pearson_r": np.nan, "typicality_pearson_p": np.nan, "n_typicality_concepts": 0}
    if global_typicality_df.empty:
        return empty
    paired = global_typicality_df.dropna(subset=['human_typicality', 'global_cosine_typicality'])
    if len(paired) < 3 or paired['human_typicality'].std() == 0 or paired['global_cosine_typicality'].std() == 0:
        return {**empty, "n_typicality_concepts": len(paired)}
    r, p = pearsonr(paired['human_typicality'], paired['global_cosine_typicality'])
    return {"typicality_pearson_r": r, "typicality_pearson_p": p, "n_typicality_concepts": len(paired)}


# Grouped figures: (x column, x label, x name, title, [(y column, y label, y name, denominator)]).
HUMAN_TYPICALITY_LABEL = "Human typicality (Richie and Bhatia, pairwise)"


FREQUENCY_US_LABEL = "US frequency (SUBTLEX-US lemma, Zipf)"


FREQUENCY_WIKI_LABEL = "Wikipedia frequency (lemma, Zipf)"


def grouped_correlation_figures(n_words: int, n_categorized: int) -> list:
    """Specification of the three grouped figures, the measure on x against its covariates on y."""
    expert_count = ("expert_count", "Expert count", "expert_count", n_words)
    jaccard = ("jaccard_pct", "Jaccard % with category label", "jaccard", n_categorized)
    overlap = ("overlap_pct", "Overlap % with category label", "overlap", n_categorized)
    typicality = ("human_typicality", HUMAN_TYPICALITY_LABEL, "human_typicality", n_words)
    frequency_us = ("frequency_zipf_subtlex_us_lemma", FREQUENCY_US_LABEL, "frequency_us", n_words)
    return [
        ("human_typicality", HUMAN_TYPICALITY_LABEL, "human_typicality", "Human typicality against expert measures and frequency",
         [expert_count, jaccard, overlap, frequency_us]),
        ("frequency_zipf_subtlex_us_lemma", FREQUENCY_US_LABEL, "frequency_us",
         "US frequency (SUBTLEX-US) against expert measures and human typicality",
         [expert_count, jaccard, overlap, typicality]),
        ("frequency_zipf_wikipedia_lemma", FREQUENCY_WIKI_LABEL, "frequency_wiki",
         "Wikipedia frequency against expert measures and human typicality",
         [expert_count, jaccard, overlap, typicality]),
    ]


def plot_grouped_correlations(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame,
                              concept_metadata: pd.DataFrame, corr_dir, x_names: list) -> list:
    """The grouped figures named in x_names, each measure on x against expert count, Jaccard, overlap and the others."""
    wide = merged_metadata_df.copy()
    if similarity_metrics_df is not None and not similarity_metrics_df.empty:
        wide = wide.merge(similarity_metrics_df[["concept", "category", "jaccard_pct", "overlap_pct"]],
                          on=["concept", "category"], how="left")
    else:
        wide["jaccard_pct"], wide["overlap_pct"] = np.nan, np.nan
    n_categorized = int(concept_metadata["category"].notna().sum())
    summary_rows = []
    for x_col, x_label, x_name, title, panels in grouped_correlation_figures(len(concept_metadata), n_categorized):
        if x_name not in x_names:
            continue
        if x_col not in wide.columns:
            log.warning(f"  Skipping {x_name}_correlations: metadata has no {x_col} column.")
            continue
        panels = [panel for panel in panels if panel[0] in wide.columns]
        if corr_dir is None:
            summary_rows += regression_grid_rows(wide, x_col, x_name, panels)
        else:
            summary_rows += run_regression_grid(wide, x_col, x_label, x_name, panels, title,
                                                corr_dir / f"{x_name}_correlations.csv",
                                                corr_dir / f"{x_name}_correlations.png")
    return summary_rows


def _residualize(values: pd.Series, control: pd.Series) -> pd.Series:
    """Residuals of values after a linear regression on control."""
    slope, intercept, _, _, _ = stats.linregress(control, values)
    return values - (slope * control + intercept)


def plot_partial_correlation_jaccard_human_typicality(corr_data: pd.DataFrame, corr_dir, n_total_relevant: int) -> dict:
    """Jaccard against typicality, both residualized on log frequency."""
    clean_data = corr_data.dropna(subset=["jaccard_pct", "human_typicality", "log_frequency"]).copy()
    n = len(clean_data)
    coverage = n / n_total_relevant if n_total_relevant else 0.0
    summary_row = {
        "plot_name": "partial_correlation_jaccard_human_typicality", "x_variable": "jaccard_resid",
        "y_variable": "human_typicality_resid", "pearson_r": None, "pearson_p": None,
        "n_points": n, "n_total_relevant": n_total_relevant, "coverage_pct": round(100 * coverage, 1),
        "controlling_for": "log_frequency",
    }
    if n < MIN_ABSOLUTE_N or coverage < MIN_COVERAGE_FRACTION:
        log.warning(f"  Skipping partial_correlation_jaccard_human_typicality: n={n} covers {coverage:.0%} of "
                    f"{n_total_relevant} relevant concepts (need >={MIN_ABSOLUTE_N} and >={MIN_COVERAGE_FRACTION:.0%}).")
        return summary_row

    clean_data["jaccard_resid"] = _residualize(clean_data["jaccard_pct"], clean_data["log_frequency"])
    clean_data["human_typicality_resid"] = _residualize(clean_data["human_typicality"], clean_data["log_frequency"])
    r, p = stats.pearsonr(clean_data["jaccard_resid"], clean_data["human_typicality_resid"])
    summary_row["pearson_r"], summary_row["pearson_p"] = r, p
    if corr_dir is None:
        return summary_row
    save_dataframe(
        clean_data[["concept", "category", "log_frequency", "jaccard_pct", "human_typicality", "jaccard_resid", "human_typicality_resid"]],
        corr_dir / "partial_correlation_jaccard_human_typicality.csv"
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    draw_regression(ax, clean_data, "human_typicality_resid", "jaccard_resid")
    set_title(ax, "Jaccard against human typicality, controlling for frequency",
              stats_line("Partial r", r, p, n))
    ax.set_xlabel("Human typicality, residual after log frequency")
    ax.set_ylabel("Jaccard with category label (%), residual after log frequency")
    fig.tight_layout()
    fig.savefig(corr_dir / "partial_correlation_jaccard_human_typicality.png", dpi=300)
    plt.close(fig)
    return summary_row


def plot_partial_correlation(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame,
                             concept_metadata: pd.DataFrame, corr_dir) -> list:
    """The frequency-controlled partial correlation of Jaccard and human typicality."""
    corr_data = _concept_parent_frame(merged_metadata_df, similarity_metrics_df)
    if corr_data.empty:
        return []
    return [plot_partial_correlation_jaccard_human_typicality(corr_data, corr_dir,
                                                              concept_metadata['category'].notna().sum())]


def plot_jaccard_vs_cosine_typicality(similarity_metrics_df: pd.DataFrame, global_typicality_df: pd.DataFrame,
                                      concept_metadata: pd.DataFrame, corr_dir) -> list:
    """Jaccard against cosine typicality, the model-derived counterpart of typicality against Jaccard."""
    if (similarity_metrics_df is None or similarity_metrics_df.empty
            or global_typicality_df is None or global_typicality_df.empty):
        log.warning("  Skipping Jaccard vs. Cosine Typicality: similarity or typicality data unavailable.")
        return []

    n_total_categorized = concept_metadata['category'].notna().sum()
    merged = similarity_metrics_df.merge(global_typicality_df, on=["concept", "category"], how="inner")
    return [_regression(
        corr_dir, "jaccard_vs_cosine_typicality", merged, "global_cosine_typicality", "jaccard_pct",
        "Cosine typicality", "Jaccard with category label (%)",
        "Cosine typicality against Jaccard", n_total_categorized)]


def run_typicality(scope, concept_metadata: pd.DataFrame, merged_meta: pd.DataFrame,
                   similarity_metrics_df: pd.DataFrame, out_dir) -> dict:
    """Cosine typicality of one scope, global and per layer, and every correlation with human typicality."""
    log.info(f"  [{scope.label}] Computing Empirical Cosine Typicality...")
    typ_global_df = compute_global_typicality(scope.expert_df, concept_metadata, out_dir)
    if out_dir is not None:
        generate_global_typicality_reports(typ_global_df, out_dir)
        for suffix, axis_df, axis_label in axis_variants(scope):
            log.info(f"  [{scope.label} / {axis_label}] Computing per-layer typicality prototypes...")
            typ_layer_df = compute_layer_typicality(axis_df, concept_metadata, out_dir, suffix)
            generate_layer_typicality_reports(typ_layer_df, out_dir, suffix)

    log.info(f"  [{scope.label}] Generating human typicality correlation plots...")
    rows = plot_grouped_correlations(merged_meta, similarity_metrics_df, concept_metadata, out_dir,
                                     ["human_typicality"])
    rows += plot_partial_correlation(merged_meta, similarity_metrics_df, concept_metadata, out_dir)
    rows += plot_jaccard_vs_cosine_typicality(similarity_metrics_df, typ_global_df, concept_metadata, out_dir)
    summary_df = _write_correlation_summary(rows, out_dir)
    return scope_summary_row(scope, **_summarize_typicality(typ_global_df), **panel_r_columns(summary_df))


# ---------------------------------------------------------------------------
# 2.5 Frequency correlations
# ---------------------------------------------------------------------------

FREQUENCY_SUMMARY_LABELS = {
    "r_frequency_us_vs_expert_count": "r, US frequency vs expert count",
}


def run_frequency_correlations(scope, concept_metadata: pd.DataFrame, merged_meta: pd.DataFrame,
                               similarity_metrics_df: pd.DataFrame, out_dir) -> dict:
    """US and Wikipedia frequency against expert count, Jaccard, overlap and human typicality."""
    log.info(f"  [{scope.label}] Generating frequency correlation plots...")
    rows = plot_grouped_correlations(merged_meta, similarity_metrics_df, concept_metadata, out_dir,
                                     ["frequency_us", "frequency_wiki"])
    summary_df = _write_correlation_summary(rows, out_dir)
    return scope_summary_row(scope, **panel_r_columns(summary_df))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SECTION_SUMMARY_LABELS = {
    "2.1": CONCEPT_CATEGORY_SUMMARY_LABELS,
    "2.2": PAIRWISE_SIMILARITY_SUMMARY_LABELS,
    "2.3": HUMAN_VALIDATION_SUMMARY_LABELS,
    "2.4": TYPICALITY_SUMMARY_LABELS,
    "2.5": FREQUENCY_SUMMARY_LABELS,
}


def execute_module_2_similarities(scope, concept_metadata: pd.DataFrame, module_dir,
                                  write_outputs: bool = True) -> dict:
    """Run every section on one scope, returns {section: sublayer_comparison row}.

    With write_outputs False nothing is written and only the default profile metric is computed,
    the only one the summary rows report.
    """
    metrics = ACTIVE_PROFILE_METRICS if write_outputs else [DEFAULT_PROFILE_METRIC]

    def out(section):
        return scope_section_dir(module_dir, scope, SECTION_DIRS[section]) if write_outputs else None

    def profile(subfolder):
        return scope_section_dir(module_dir, scope, f"{LAYER_PROFILE_DIR}/{subfolder}") if write_outputs else None

    merged_meta = expert_counts_with_metadata(scope.expert_df, concept_metadata)
    similarity_metrics_df, matrices = pd.DataFrame(), None
    rows = {}
    # Each section line can be commented out alone, later sections then get empty inputs.
    similarity_metrics_df, rows["2.1"] = run_category_concept_similarities(
        scope, concept_metadata, merged_meta, out("2.1"), profile("hierarchy"), metrics)
    matrices, rows["2.2"] = run_pairwise_similarities(scope, concept_metadata, out("2.2"), profile("heatmaps"), metrics)
    rows["2.3"] = run_human_similarity_validation(scope, matrices, out("2.3"), profile("heatmaps"),
                                                  profile("similarity_correlation"))
    rows["2.4"] = run_typicality(scope, concept_metadata, merged_meta, similarity_metrics_df, out("2.4"))
    rows["2.5"] = run_frequency_correlations(scope, concept_metadata, merged_meta, similarity_metrics_df, out("2.5"))
    return rows
