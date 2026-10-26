"""
Module 2: similarities. How alike two words' expert sets and layer profiles are, and how that
relates to human similarity judgments, typicality and frequency. Explanations live in documentation/.

  2.1 category concept similarities   each concept against its category label
  2.2 pairwise similarities           every word pair, matrices, heatmaps, Jaccard against layer profile
  2.3 human similarity validation     pairwise matrices against the human pair ratings
  2.4 typicality                      cosine typicality and every correlation with human typicality
  2.5 frequency correlations          the model's corpus frequency against the expert measures
"""

import functools
import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.logging_and_io import save_dataframe
from utils.statistics import PERMUTATIONS, correlation, spearman_test
from modules.shared_expert_set_measures import (expert_presence_matrix, expert_set_overlap_matrices,
                                                category_alignment_metrics, pair_similarity_vector)
from modules.shared_layer_profile_measures import (pair_layer_profile_vectors,
                                                   layer_profile_metric_matrices,
                                                   ACTIVE_PROFILE_METRICS, DEFAULT_PROFILE_METRIC,
                                                   SIGNED_PROFILE_METRICS, profile_metric_label,
                                                   profile_metric_value_label)
from core.analysis_scopes import to_block_axis, axis_variants, scope_section_dir, scope_summary_row
from core.expert_data_loading import expert_counts_with_metadata
from core.correlation_reporting import (Panel, panel_rho_columns, run_regression_panel, run_regression_grid,
                                        regression_row, regression_grid_rows, MIN_ABSOLUTE_N)
from core.human_similarity_ratings import load_human_similarity, human_noise_ceiling
from core.probing_sentences import load_positive_documents
from utils.plotting import (_plot_bar_with_leaders, _plot_heatmap_with_leaders,
                            apply_rotated_leader_labels, build_category_color_map, fig_width_for,
                            plot_hexbin_with_trends, plot_sublayer_comparison_bars, set_title, set_suptitle,
                            stats_line, draw_regression, twin_axis, LEVEL_COLORS, SERIES_COLORS,
                            DETAIL_COLOR, SEQUENTIAL_CMAP, DIVERGING_CMAP)

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

# Variables defined relative to the concept's category: a correlation involving one is computed within categories.
CATEGORY_RELATIVE = {"human_typicality", "jaccard_pct", "overlap_pct", "layer_profile_similarity_pct",
                     "global_cosine_typicality"}


# Row of the per-category tables holding the size-weighted mean over categories.
WEIGHTED_MEAN = "WEIGHTED_MEAN"


def within_category(*columns: str) -> bool:
    """Whether a correlation of these columns is computed within categories, true when any is category-relative."""
    return any(column in CATEGORY_RELATIVE for column in columns)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Mean of the finite values weighted by their weights, NaN when none is finite."""
    finite = values.notna()
    return float((values[finite] * weights[finite]).sum() / weights[finite].sum()) if finite.any() else np.nan


def _regression(corr_dir, plot_name: str, data: pd.DataFrame, x_col: str, y_col: str, x_label: str,
                y_label: str, title: str, n_total_relevant: int) -> dict:
    """One panel with its CSV, or only its correlation_summary row without a folder, within categories when either
    variable is category-relative."""
    within = within_category(x_col, y_col)
    if corr_dir is None:
        return regression_row(data, x_col, y_col, plot_name, n_total_relevant, within)[1]
    return run_regression_panel(data, x_col, y_col, x_label, y_label, title, corr_dir / f"{plot_name}.csv",
                                corr_dir / f"{plot_name}.png", n_total_relevant, within)


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
    "rho_jaccard_vs_layer_profile": "Within-category ρ, Jaccard vs layer profile (concept to label)",
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


CATEGORY_SUMMARY_MEASURES = {"jaccard_pct": "Jaccard with the category label (%)",
                             "overlap_pct": "Overlap with the category label (%)"}


def _display_word(concept: str) -> str:
    """The word of a concept key, without the homonym suffix of the squash keys."""
    return concept.split("__")[0]


def category_similarity_summary(similarity_metrics_df: pd.DataFrame, concept_metadata: pd.DataFrame) -> pd.DataFrame:
    """Per category, quartiles and mean of the members' Jaccard and overlap with the label, and the Spearman rho of
    Jaccard against human typicality with its permutation p, the components of section 2.4's within-category rho."""
    typicality = similarity_metrics_df["concept"].map(concept_metadata.set_index("concept")["human_typicality"])
    test = spearman_test(similarity_metrics_df["jaccard_pct"], typicality, groups=similarity_metrics_df["category"])
    agreement = dict(zip(test["groups"], zip(test["rho_k"], test["p_k"], test["n_k"])))
    rows = []
    for category, group in similarity_metrics_df.groupby("category"):
        row = {"category": category, "n_members": len(group),
               "label_expert_units": int(group["category_expert_units"].iloc[0])}
        for column in CATEGORY_SUMMARY_MEASURES:
            stem, values = column.removesuffix("_pct"), group[column]
            row.update({f"{stem}_median_pct": values.median(), f"{stem}_q1_pct": values.quantile(0.25),
                        f"{stem}_q3_pct": values.quantile(0.75), f"{stem}_mean_pct": values.mean()})
        rho, p, n = agreement.get(category, (np.nan, np.nan, 0))
        row.update({"rho_jaccard_human_typicality": rho, "p_jaccard_human_typicality": p, "n_with_typicality": int(n)})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_category_similarity_summary(similarity_metrics_df: pd.DataFrame, summary: pd.DataFrame, out_dir) -> None:
    """Each member's Jaccard and overlap with its label, one row per category, with the median and interquartile range."""
    categories = list(summary["category"])
    positions = np.arange(len(categories))[::-1].astype(float)
    member_color, median_color = LEVEL_COLORS["Specific Concepts"], LEVEL_COLORS["Broad Categories"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.6 * len(categories) + 2.4), sharey=True)
    jitter = np.random.default_rng(0)
    for axis, (column, label) in zip(axes, CATEGORY_SUMMARY_MEASURES.items()):
        stem = column.removesuffix("_pct")
        for y, category in zip(positions, categories):
            values = similarity_metrics_df.loc[similarity_metrics_df["category"] == category, column].to_numpy()
            axis.scatter(values, y + jitter.uniform(-0.15, 0.15, len(values)), s=10, color=member_color,
                         alpha=0.6, edgecolors="none", zorder=2)
        axis.hlines(positions, summary[f"{stem}_q1_pct"], summary[f"{stem}_q3_pct"], color=median_color,
                    linewidth=3, zorder=3)
        axis.scatter(summary[f"{stem}_median_pct"], positions, s=55, color=median_color, edgecolors="white",
                     linewidths=0.6, zorder=4)
        axis.set_xlabel(label)
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([c.capitalize() for c in categories])
    handles = [Line2D([], [], marker="o", linestyle="none", color=member_color, alpha=0.6, label="Members"),
               Line2D([], [], color=median_color, linewidth=3, label="Interquartile range"),
               Line2D([], [], marker="o", linestyle="none", color=median_color, markersize=8, label="Median")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    set_suptitle(fig, "Each member's expert overlap with its category label, per category")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "category_concept_similarity_by_category.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_category_hierarchies(similarity_metrics_df: pd.DataFrame, summary: pd.DataFrame,
                              concept_metadata: pd.DataFrame, out_dir) -> None:
    """One appendix figure per category, each member's Jaccard with the label, members by descending human typicality."""
    typicality = concept_metadata.set_index("concept")["human_typicality"]
    color_map = build_category_color_map(concept_metadata.dropna(subset=["category"])["category"])
    target = out_dir / "by_category"
    target.mkdir(exist_ok=True)
    for _, row in summary.iterrows():
        category = row["category"]
        members = (similarity_metrics_df[similarity_metrics_df["category"] == category]
                   .assign(human_typicality=lambda frame: frame["concept"].map(typicality))
                   .sort_values(["human_typicality", "concept"], ascending=[False, True]))
        n = len(members)
        fig, ax = plt.subplots(figsize=(fig_width_for(n, 0.42, min_w=9.0), 5.6))
        ax.bar(range(n), members["jaccard_pct"], color=color_map.get(category), edgecolor="black", linewidth=0.5)
        apply_rotated_leader_labels(ax, [_display_word(c) for c in members["concept"]], fontsize=9)
        ax.set_ylabel("Jaccard with the category label (%)")
        twin = twin_axis(ax)
        twin.plot(range(n), members["human_typicality"], linestyle="none", marker="D", markersize=5,
                  color=SERIES_COLORS["primary"])
        twin.set_ylim(-0.05, 1.05)
        twin.set_ylabel("Human typicality (0 to 1 within the category)")
        rho = row["rho_jaccard_human_typicality"]
        detail = (stats_line("Spearman ρ", rho, row["p_jaccard_human_typicality"], int(row["n_with_typicality"]))
                  if np.isfinite(rho) else f"n = {n}")
        set_title(ax, f"{category.capitalize()}, each member's Jaccard with the word '{category}'",
                  f"members by descending human typicality (diamonds), {detail}")
        fig.tight_layout()
        fig.savefig(target / f"jaccard_hierarchy_{category}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


CORE_DRAWS = 2000
CORE_SIZE = 4


def _packed_presence(expert_allocation_df: pd.DataFrame, concepts: list) -> np.ndarray:
    """Each concept's expert set as one row of packed bits, rows in the order of concepts."""
    presence = expert_presence_matrix(expert_allocation_df, concepts)
    row = np.zeros(presence.shape[1], dtype=bool)
    packed = np.zeros((presence.shape[0], (presence.shape[1] + 7) // 8), dtype=np.uint8)
    for i in range(presence.shape[0]):
        row[:] = False
        row[presence.indices[presence.indptr[i]:presence.indptr[i + 1]]] = True
        packed[i] = np.packbits(row)
    return packed


def _core_statistics(bits: np.ndarray, label: np.ndarray) -> tuple:
    """Core shared by every row as a share of their union, the label's share of that core, and the core size."""
    core = np.bitwise_and.reduce(bits, axis=0)
    core_size = int(np.bitwise_count(core).sum())
    union_size = int(np.bitwise_count(np.bitwise_or.reduce(bits, axis=0)).sum())
    in_label = int(np.bitwise_count(core & label).sum())
    return (100 * core_size / union_size if union_size else np.nan,
            100 * in_label / core_size if core_size else np.nan, core_size)


def _mean_of_finite(values) -> float:
    """Mean over the finite values, NaN when there are none."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def category_core_overlap(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame,
                          n_draws: int = CORE_DRAWS, k: int = CORE_SIZE, seed: int = 0) -> pd.DataFrame:
    """ExpertLens domain statistics per category: the experts shared by k random members as a share of their union,
    the label's share of that core, the same for k members of k other categories, and the core of every member."""
    present = set(expert_allocation_df["concept"])
    members = {category: sorted(c for c in group["concept"] if c in present)
               for category, group in concept_metadata.dropna(subset=["category"]).groupby("category")}
    concepts = sorted(present)
    position = {concept: i for i, concept in enumerate(concepts)}
    bits = _packed_presence(expert_allocation_df, concepts)
    rng = np.random.default_rng(seed)
    rows = []
    for category, words in members.items():
        others = [other for other, other_words in members.items() if other != category and other_words]
        if category not in present or len(words) < k or len(others) < k:
            continue
        label = bits[position[category]]
        word_rows = np.array([position[w] for w in words])
        real = np.array([_core_statistics(bits[rng.choice(word_rows, k, replace=False)], label)
                         for _ in range(n_draws)])
        baseline = np.array([_core_statistics(bits[[position[rng.choice(members[o])]
                                                    for o in rng.choice(others, k, replace=False)]], label)
                             for _ in range(n_draws)])
        every_member = _core_statistics(bits[word_rows], label)
        rows.append({
            "category": category, "n_members": len(words), "label_experts": int(np.bitwise_count(label).sum()),
            f"core_pct_{k}": _mean_of_finite(real[:, 0]), f"core_size_{k}": real[:, 2].mean(),
            f"label_share_of_core_pct_{k}": _mean_of_finite(real[:, 1]),
            f"empty_core_rate_{k}": float((real[:, 2] == 0).mean()),
            f"random_core_pct_{k}": _mean_of_finite(baseline[:, 0]), f"random_core_size_{k}": baseline[:, 2].mean(),
            f"random_label_share_of_core_pct_{k}": _mean_of_finite(baseline[:, 1]),
            f"random_empty_core_rate_{k}": float((baseline[:, 2] == 0).mean()),
            "core_pct_all_members": every_member[0], "core_size_all_members": every_member[2],
            "label_share_of_core_pct_all_members": every_member[1],
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    means = {column: _mean_of_finite(table[column]) for column in table.columns if column != "category"}
    return pd.concat([table, pd.DataFrame([{"category": "MEAN", **means}])], ignore_index=True)


def run_category_concept_similarities(scope, concept_metadata: pd.DataFrame, merged_meta: pd.DataFrame, out_dir,
                                      profile_dir, metrics: list = ACTIVE_PROFILE_METRICS) -> tuple[pd.DataFrame, dict]:
    """Concept-to-label similarities of one scope, returns the table and the summary row."""
    log.info(f"  [{scope.label}] Generating hierarchy similarities (Jaccard & Overlap)...")
    similarity_metrics_df = plot_hierarchy_similarities(scope.expert_df, concept_metadata, out_dir, profile_dir,
                                                        metrics)
    if out_dir is not None and not similarity_metrics_df.empty:
        by_category = category_similarity_summary(similarity_metrics_df, concept_metadata)
        save_dataframe(by_category, out_dir / "category_concept_similarity_by_category.csv")
        plot_category_similarity_summary(similarity_metrics_df, by_category, out_dir)
        plot_category_hierarchies(similarity_metrics_df, by_category, concept_metadata, out_dir)
    if out_dir is not None:
        core = category_core_overlap(scope.expert_df, concept_metadata)
        if not core.empty:
            save_dataframe(core, out_dir / "category_core_overlap.csv")
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
            **panel_rho_columns(summary_df),
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
    "rho_jaccard_vs_layer_profile_allpairs": "ρ, Jaccard vs layer profile (all pairs)",
    "rho_jaccard_vs_layer_profile_z_allpairs": "ρ, Jaccard vs layer-profile z (all pairs)",
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


def summarize_category_contrast(concepts, similarity_matrix: np.ndarray, concept_metadata: pd.DataFrame,
                                n_draws: int = PERMUTATIONS) -> dict:
    """Within against across category similarity of the member pairs, its ROC-AUC with a two-sided Mantel p and the
    number of pairs, the members missing the most pairs dropped until the others form complete pairs."""
    concept_to_cat = concept_metadata.set_index("concept")["category"].to_dict()
    categorized = [i for i, c in enumerate(concepts) if pd.notna(concept_to_cat.get(c))]
    categories = np.array([concept_to_cat[concepts[i]] for i in categorized])
    pair_similarity = similarity_matrix[np.ix_(categorized, categorized)][np.triu_indices(len(categorized), k=1)]
    alignment = category_alignment_metrics(pair_similarity, categories, n_draws)
    return {"within_pct": alignment["within_mean"], "across_pct": alignment["across_mean"],
            "contrast_pct": alignment["within_mean"] - alignment["across_mean"], "roc_auc": alignment["roc_auc"],
            "mantel_p": alignment["mantel_p"], "n_pairs": alignment["n_pairs"]}


def allpairs_row(plot_name: str, jaccard: np.ndarray, y: np.ndarray, y_variable: str, n_items: int,
                 n_draws: int = PERMUTATIONS) -> tuple[dict, np.ndarray]:
    """correlation_summary row of Jaccard against a pair measure over every word pair, Spearman's rho with its
    two-sided Mantel p, and the pairs it used, once the words missing the most pairs are dropped."""
    upper = np.triu_indices(n_items, k=1)
    test = spearman_test(jaccard, y, pairs=upper, n_draws=n_draws)
    covered = test["n"] >= MIN_ABSOLUTE_N and np.isfinite(test["rho"])
    return {"plot_name": plot_name, "x_variable": "jaccard_pct", "y_variable": y_variable, "controlling_for": "",
            "ranks": "pooled", "spearman_rho": test["rho"] if covered else None,
            "spearman_p": test["p"] if covered and n_draws else None, "n_points": test["n"],
            "n_total_relevant": len(upper[0]),
            "coverage_pct": round(100 * test["n"] / len(upper[0]), 1) if len(upper[0]) else 0.0}, test["used"]


# All-pairs panels: (plot name, y label, y variable).
ALLPAIRS_PANELS = [("jaccard_vs_layer_profile_allpairs", "Layer-profile similarity %", "layer_profile_similarity_pct"),
                   ("jaccard_vs_layer_profile_z_allpairs", "Layer-profile agreement vs null (z)", "layer_profile_z")]


def plot_all_pairs_jaccard_vs_layer_profile(scope, concept_metadata: pd.DataFrame, corr_dir) -> list:
    """Jaccard against layer-profile agreement, raw and z, over every word pair, the Mantel p computed only when the
    section writes outputs."""
    items = list(concept_metadata["concept"].unique())
    jaccard = pair_similarity_vector(scope.expert_df, items) * 100.0
    profiles = pair_layer_profile_vectors(to_block_axis(scope.expert_df), items)
    category_of = concept_metadata.set_index("concept")["category"].to_dict()
    cats = np.array([category_of.get(c) for c in items], dtype=object)
    same = (cats[:, None] == cats[None, :])[np.triu_indices(len(items), k=1)]

    rows = []
    for (plot_name, y_label, y_variable), y in zip(ALLPAIRS_PANELS, profiles):
        row, used = allpairs_row(plot_name, jaccard, y, y_variable, len(items),
                                 PERMUTATIONS if corr_dir is not None else 0)
        if corr_dir is not None:
            save_dataframe(pd.DataFrame({"jaccard_pct": jaccard[used], y_variable: y[used],
                                         "same_category": same[used]}), corr_dir / f"{plot_name}.csv")
        if row["spearman_rho"] is None:
            log.warning(f"  Skipping {plot_name}: only {row['n_points']} usable pairs.")
        elif corr_dir is not None:
            plot_hexbin_with_trends(
                jaccard[used], y[used], same[used], corr_dir / f"{plot_name}.png", "Expert Jaccard %", y_label,
                colorbar_label="Word pairs (log)",
                detail=stats_line("Spearman ρ", row["spearman_rho"], row["spearman_p"], row["n_points"],
                                  p_label="Mantel p"))
        rows.append(row)
    return rows


def category_mean_matrix(concepts, jaccard_matrix: np.ndarray, concept_metadata: pd.DataFrame,
                         with_experts: set) -> pd.DataFrame:
    """Mean Jaccard (%) between the members of every two categories, self-pairs and words without experts left out."""
    category_of = concept_metadata.set_index("concept")["category"]
    categories = sorted(category_of.dropna().unique())
    rows_of = {category: [i for i, word in enumerate(concepts)
                          if category_of.get(word) == category and word in with_experts]
               for category in categories}
    table = pd.DataFrame(np.nan, index=categories, columns=categories)
    for a in categories:
        for b in categories:
            block = jaccard_matrix[np.ix_(rows_of[a], rows_of[b])]
            values = block[~np.eye(len(rows_of[a]), dtype=bool)] if a == b else block.ravel()
            table.loc[a, b] = values.mean() if values.size else np.nan
    return table


def plot_category_mean_matrix(table: pd.DataFrame, out_dir) -> None:
    """The category-mean Jaccard matrix as an annotated heatmap."""
    labels = [c.capitalize() for c in table.index]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(table.to_numpy(dtype=float), annot=True, fmt=".2f", cmap=SEQUENTIAL_CMAP, square=True,
                xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Mean Jaccard (%)"}, ax=ax)
    ax.grid(False)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels, rotation=0)
    set_title(ax, "Mean pairwise Jaccard within and between categories",
              "members only, self-pairs excluded, percent")
    fig.tight_layout()
    fig.savefig(out_dir / "jaccard_category_mean_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_category_heatmaps(concepts, jaccard_matrix: np.ndarray, concept_metadata: pd.DataFrame,
                           with_experts: set, out_dir) -> None:
    """One appendix heatmap per category, the category word first and then its members by descending human typicality."""
    meta = concept_metadata.set_index("concept")
    index = {word: i for i, word in enumerate(concepts)}
    target = out_dir / "by_category"
    target.mkdir(exist_ok=True)
    for category in sorted(meta["category"].dropna().unique()):
        members = (meta[(meta["category"] == category) & meta.index.isin(with_experts)]
                   .sort_values("human_typicality", ascending=False).index.tolist())
        has_label = category in with_experts and category in index
        words = ([category] if has_label else []) + members
        if len(words) < 2:
            continue
        block = jaccard_matrix[np.ix_([index[w] for w in words], [index[w] for w in words])].astype(float)
        np.fill_diagonal(block, np.nan)
        labels = [_display_word(w) for w in words]
        side = 0.3 * len(words) + 3.0
        fig, ax = plt.subplots(figsize=(side + 1.5, side))
        sns.heatmap(block, cmap=SEQUENTIAL_CMAP, square=True, xticklabels=labels, yticklabels=labels,
                    cbar_kws={"label": "Jaccard (%)", "shrink": 0.7}, ax=ax)
        ax.grid(False)
        ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        if has_label:
            ax.axhline(1, color="white", linewidth=1.5)
            ax.axvline(1, color="white", linewidth=1.5)
        set_title(ax, f"{category.capitalize()}, pairwise Jaccard of the category word and its members",
                  "category word first, members by descending human typicality, diagonal blank")
        fig.tight_layout()
        fig.savefig(target / f"jaccard_heatmap_{category}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def run_pairwise_similarities(scope, concept_metadata: pd.DataFrame, out_dir, profile_dir,
                              metrics: list = ACTIVE_PROFILE_METRICS) -> tuple[dict, dict]:
    """Pairwise matrices of one scope, their category contrast, and Jaccard against layer profile over all pairs."""
    log.info(f"  [{scope.label}] Generating all pairwise heatmaps and CSV matrices "
             f"(Jaccard, Overlap & {len(metrics)} layer-profile metrics)...")
    concepts, jaccard_matrix, profile_matrices = plot_all_heatmaps(
        scope.expert_df, concept_metadata, out_dir, profile_dir, metrics)
    if out_dir is not None:
        with_experts = set(scope.expert_df["concept"])
        category_means = category_mean_matrix(concepts, jaccard_matrix, concept_metadata, with_experts)
        save_dataframe(category_means, out_dir / "jaccard_category_mean_matrix.csv", index=True)
        plot_category_mean_matrix(category_means, out_dir)
        plot_category_heatmaps(concepts, jaccard_matrix, concept_metadata, with_experts, out_dir)

    contrasts = {key: summarize_category_contrast(concepts, matrix, concept_metadata,
                                                  PERMUTATIONS if out_dir is not None else 0)
                 for key, matrix in {"jaccard": jaccard_matrix, **profile_matrices}.items()}
    if out_dir is not None:
        save_dataframe(pd.DataFrame([{"matrix": key, **values} for key, values in contrasts.items()]),
                       out_dir / "category_alignment.csv")

    log.info(f"  [{scope.label}] Comparing Jaccard against layer-profile agreement over all word pairs...")
    summary_df = _write_correlation_summary(
        plot_all_pairs_jaccard_vs_layer_profile(scope, concept_metadata, out_dir), out_dir)

    jaccard = contrasts["jaccard"]
    matrices = {"concepts": concepts, "jaccard": jaccard_matrix, "profiles": profile_matrices, "contrasts": contrasts}
    return matrices, scope_summary_row(
        scope,
        within_category_jaccard_pct=jaccard["within_pct"],
        across_category_jaccard_pct=jaccard["across_pct"],
        category_contrast_pct=jaccard["contrast_pct"],
        category_roc_auc=jaccard["roc_auc"],
        layer_profile_roc_auc=contrasts[profile_matrix_key(DEFAULT_PROFILE_METRIC)]["roc_auc"],
        layer_profile_z_roc_auc=contrasts[profile_z_matrix_key(DEFAULT_PROFILE_METRIC)]["roc_auc"],
        **panel_rho_columns(summary_df),
    )


# ---------------------------------------------------------------------------
# 2.3 Human similarity validation: pairwise matrices against the human pair ratings
# ---------------------------------------------------------------------------

HUMAN_VALIDATION_SUMMARY_LABELS = {
    "human_rho_jaccard": "Human similarity agreement, Jaccard",
    "human_rho_layer_profile": f"Human similarity agreement, layer profile ({DEFAULT_PROFILE_METRIC})",
    "human_rho_text_tfidf": "Human similarity agreement, probing-text TF-IDF",
    "human_partial_rho_jaccard_given_text": "Human similarity agreement, Jaccard with the text partialled out",
}


# The text baseline: each concept's positive probing sentences as one TF-IDF document.
TEXT_METRIC = "text_tfidf"
TEXT_LABEL = "probing-text TF-IDF cosine"
TEXT_SUBJECT = "Probing-text TF-IDF cosine"


HUMAN_SIGNIFICANCE_ALPHA = 0.05


def human_validated_metrics() -> list:
    """(key, label) of every matrix validated against human ratings."""
    metrics = [("jaccard", "Jaccard")]
    for metric in ACTIVE_PROFILE_METRICS:
        label = profile_metric_label(metric)
        metrics.append((profile_matrix_key(metric), f"layer profile, {label}"))
        metrics.append((profile_z_matrix_key(metric), f"layer profile z, {label}"))
    metrics.append((TEXT_METRIC, TEXT_LABEL))
    return metrics


HUMAN_VALIDATED_METRICS = human_validated_metrics()


def _rated_pairs(concepts, with_experts: set | None = None) -> tuple:
    """The rated pairs whose two concepts are known and hold experts, with the positions of both concepts."""
    index = {concept: position for position, concept in enumerate(concepts)}
    pairs = load_human_similarity()
    known = pairs.concept_a.isin(index) & pairs.concept_b.isin(index)
    if with_experts is not None:
        known &= pairs.concept_a.isin(with_experts) & pairs.concept_b.isin(with_experts)
    rated = pairs[known].reset_index(drop=True)
    return rated, rated.concept_a.map(index).to_numpy(), rated.concept_b.map(index).to_numpy()


def human_agreement(concepts, matrix: np.ndarray, with_experts: set | None = None,
                    n_draws: int = PERMUTATIONS) -> tuple[pd.DataFrame, dict]:
    """Spearman test of a pair matrix against the human ratings within each category, the member labels of each
    category relabelled (stratified Mantel), and the rated pairs it used with their within-category rank scores."""
    rated, first, second = _rated_pairs(concepts, with_experts)
    model = matrix[first, second]
    test = spearman_test(model, rated.mean_rating, groups=rated.category, pairs=(first, second), n_draws=n_draws)
    frame = rated[["category", "concept_a", "concept_b"]].assign(
        expert_similarity=model, human_rating=rated.mean_rating, expert_rank_z=test["x_scores"],
        human_rank_z=test["y_scores"])
    return frame[test["used"]], test


def _per_group(test: dict) -> dict:
    """{group: (rho, p, n)} of a spearman_test result."""
    return dict(zip(test["groups"], zip(test["rho_k"], test["p_k"], test["n_k"])))


def plot_human_vs_expert(frame: pd.DataFrame, test: dict, metric_label: str, out_path,
                         subject: str | None = None) -> None:
    """Per-category scatter of model similarity against human rating with the least-squares line, the category's
    Spearman rho with its Mantel p, and the CSV of the pairs."""
    subject = subject or f"Expert {metric_label}"
    statistics = _per_group(test)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharey=True)
    legend = True
    for axis, category in zip(axes.ravel(), sorted(load_human_similarity().category.unique())):
        rho, p, n = statistics.get(category, (np.nan, np.nan, 0))
        if not np.isfinite(rho):
            axis.set_visible(False)
            continue
        pairs = frame[frame.category == category]
        handles = draw_regression(axis, pairs.expert_similarity, pairs.human_rating, size=12)
        if legend and handles:
            axis.legend(handles=handles, loc="upper left")
            legend = False
        set_title(axis, category.capitalize(),
                  stats_line("Spearman ρ", rho, p if np.isfinite(p) else None, int(n), p_label="Mantel p"))
        axis.set_xlabel(subject)
    for axis in axes[:, 0]:
        axis.set_ylabel("Human similarity rating (1 to 7)")
    set_suptitle(fig, f"{subject} against human similarity, within-category pairs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    save_dataframe(frame, out_path.with_suffix(".csv"))


def plot_human_agreement_bars(table: pd.DataFrame, out_path) -> None:
    """Per-category rho over noise ceiling, starred where the Mantel test supports it."""
    per_category = table[table.category != WEIGHTED_MEAN]
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
    axis.set_ylabel("Spearman ρ / noise ceiling")
    set_title(axis, "Agreement with human similarity as a fraction of the noise ceiling",
              f"* Mantel p < {HUMAN_SIGNIFICANCE_ALPHA}, {PERMUTATIONS:,} label permutations within the category")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False,
                fontsize=9, ncol=min(len(metrics), 4))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate_against_human(concepts: list, matrices: dict, out_dir, make_plots: bool = True,
                           with_experts: set | None = None, plot_dir=None, mantel: bool = True) -> pd.DataFrame:
    """Per matrix and category, Spearman rho against the human ratings with its Mantel p and the noise ceiling, and
    their pair-weighted mean over categories with a stratified Mantel p."""
    ceilings = human_noise_ceiling()
    rows = []
    for key, label in HUMAN_VALIDATED_METRICS:
        if key not in matrices:
            continue
        frame, test = human_agreement(concepts, matrices[key], with_experts, PERMUTATIONS if mantel else 0)
        if not len(test["groups"]):
            continue
        ceiling = np.array([ceilings[category] for category in test["groups"]])
        valid = np.isfinite(test["rho_k"])
        mean_ceiling = test["n_k"][valid] @ ceiling[valid] / test["n_k"][valid].sum() if valid.any() else np.nan
        for category, n, rho, p, c in zip([*test["groups"], WEIGHTED_MEAN], [*test["n_k"], test["n"]],
                                          [*test["rho_k"], test["rho"]], [*test["p_k"], test["p"]],
                                          [*ceiling, mean_ceiling]):
            rows.append({"metric": key, "metric_label": label, "category": category, "n_pairs": int(n), "rho": rho,
                         "mantel_p": p, "noise_ceiling": c, "rho_over_ceiling": rho / c})
        if make_plots and out_dir is not None:
            target = out_dir if key in ("jaccard", TEXT_METRIC) else (plot_dir or out_dir)
            plot_human_vs_expert(frame, test, label, target / f"similarity_vs_{key.removeprefix('layer_profile_')}.png",
                                 subject=TEXT_SUBJECT if key == TEXT_METRIC else None)
    table = pd.DataFrame(rows)
    if out_dir is not None:
        save_dataframe(table, out_dir / "human_similarity_validation.csv")
        if make_plots and not table.empty:
            plot_human_agreement_bars(table, out_dir / "human_similarity_by_category.png")
    return table


PAIRWISE_METRIC_LABELS = {
    "category_roc_auc": "Category alignment ROC-AUC",
    "category_roc_auc_z": "Category alignment ROC-AUC, z",
    "human_rho": "Human similarity agreement (within-category ρ)",
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


@functools.lru_cache(maxsize=2)
def text_similarity_matrix(dataset_dir: str, concepts: tuple) -> np.ndarray:
    """Cosine between the TF-IDF vectors of every two concepts' positive sentences, NaN for a concept without sentences."""
    documents = load_positive_documents(dataset_dir, concepts)
    vectors = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(documents)
    similarity = (vectors @ vectors.T).toarray()
    missing = np.array([not document for document in documents])
    similarity[missing, :] = np.nan
    similarity[:, missing] = np.nan
    return similarity


TEXT_BASELINE_BARS = {
    "rho_text_human": ("Text against humans", SERIES_COLORS["muted"]),
    "rho_jaccard_human": ("Expert Jaccard against humans", SERIES_COLORS["primary"]),
    "partial_rho_jaccard_human_given_text": ("Expert Jaccard against humans, text partialled out",
                                             SERIES_COLORS["secondary"]),
}


# Zero-order coefficients of the text baseline: column, (x, y).
TEXT_BASELINE_PAIRS = {"rho_text_human": ("text", "human"), "rho_jaccard_human": ("jaccard", "human"),
                       "rho_jaccard_text": ("jaccard", "text")}


def text_baseline_table(concepts, jaccard: np.ndarray, text: np.ndarray, with_experts: set | None,
                        n_draws: int = PERMUTATIONS) -> pd.DataFrame:
    """Per category and as their pair-weighted mean, Spearman of the text and of the expert Jaccard with the human
    ratings, of Jaccard with the text, and of Jaccard with the ratings once the text is partialled out within each
    category, the last with its permutation p."""
    rated, first, second = _rated_pairs(concepts, with_experts)
    values = {"jaccard": jaccard[first, second], "human": rated.mean_rating.to_numpy(), "text": text[first, second]}
    partial = spearman_test(values["jaccard"], values["human"], [values["text"]], rated.category, (first, second),
                            n_draws=n_draws)
    if not len(partial["groups"]):
        return pd.DataFrame()
    used, groups = partial["used"], rated.category.to_numpy()[partial["used"]]
    tests = {column: spearman_test(values[x][used], values[y][used], groups=groups, n_draws=0)
             for column, (x, y) in TEXT_BASELINE_PAIRS.items()}
    return pd.DataFrame({"category": [*partial["groups"], WEIGHTED_MEAN], "n_pairs": [*partial["n_k"], partial["n"]],
                         **{column: [*test["rho_k"], test["rho"]] for column, test in tests.items()},
                         "partial_rho_jaccard_human_given_text": [*partial["rho_k"], partial["rho"]],
                         "partial_p_jaccard_human_given_text": [*partial["p_k"], partial["p"]]})


def _category_names(table: pd.DataFrame) -> list:
    """Category labels of a per-category table, its weighted mean row included."""
    return [("weighted mean" if c == WEIGHTED_MEAN else c).capitalize() for c in table["category"]]


def plot_text_baseline(table: pd.DataFrame, out_path) -> None:
    """Per category and on average, the text's and the experts' agreement with humans, and the experts' without the text."""
    x = np.arange(len(table))
    width = 0.8 / len(TEXT_BASELINE_BARS)
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, (column, (label, color)) in enumerate(TEXT_BASELINE_BARS.items()):
        ax.bar(x + (offset - 1) * width, table[column], width, label=label, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(_category_names(table), rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Spearman ρ with the human ratings")
    set_title(ax, "Agreement with human similarity, probing text against expert sets",
              "within-category pairs, TF-IDF cosine of each word's positive sentences, "
              "partial Spearman within category")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _member_and_pair_parts(values: np.ndarray, rows: np.ndarray, columns: np.ndarray, n_members: int) -> tuple:
    """Exact split of one category's complete symmetric pair values into per-member effects, their sum per pair and
    the pair residual, whose sum over the pairs of every member is zero."""
    grand = values.mean()
    totals = np.bincount(rows, values, n_members) + np.bincount(columns, values, n_members)
    member = (totals / (n_members - 1) - grand) * (n_members - 1) / (n_members - 2)
    part = member[rows] + member[columns]
    return member, part, values - grand - part


# Decomposition columns computed over members rather than pairs, averaged with member weights.
MEMBER_COLUMNS = ("rho_members", "rho_members_given_text", "rho_typicality_centrality", "rho_centrality_label_jaccard")


def human_similarity_decomposition_table(concepts, jaccard: np.ndarray, text: np.ndarray,
                                         with_experts: set | None) -> pd.DataFrame:
    """Per category, the Spearman of Jaccard against the human ratings split exactly into a member part, which
    typicality carries, and a pair-specific part, both with the text partialled out, and typicality against the
    label Jaccard, plus their weighted mean over categories, by pairs for the pair columns and by members for the
    member columns."""
    index = {concept: position for position, concept in enumerate(concepts)}
    rated, first, second = _rated_pairs(concepts, with_experts)
    rows = []
    for category, group in rated.groupby("category"):
        a, b = first[group.index], second[group.index]
        members = np.unique(np.concatenate([a, b]))
        n_members = len(members)
        model, human, text_values = jaccard[a, b], group.mean_rating.to_numpy(), text[a, b]
        complete = (len(group) == n_members * (n_members - 1) // 2 and np.isfinite(model).all()
                    and np.isfinite(text_values).all())
        if category not in index or n_members < 4 or not complete:
            continue
        a, b = np.searchsorted(members, a), np.searchsorted(members, b)
        (j_member, j_part, j_rest), (h_member, h_part, h_rest), (t_member, _, t_rest) = (
            _member_and_pair_parts(stats.rankdata(values), a, b, n_members) for values in (model, human, text_values))
        j_centred, h_centred = j_part + j_rest, h_part + h_rest
        norm = np.sqrt((j_centred ** 2).sum() * (h_centred ** 2).sum())
        typicality = (np.bincount(a, human, n_members) + np.bincount(b, human, n_members)) / (n_members - 1)
        centrality = (np.bincount(a, model, n_members) + np.bincount(b, model, n_members)) / (n_members - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rows.append({
                "category": category, "n_members": n_members, "n_pairs": len(group),
                "rho": correlation(model, human),
                "member_part": float((j_part * h_part).sum() / norm),
                "pair_part": float((j_rest * h_rest).sum() / norm),
                "member_share_jaccard": float((j_part ** 2).sum() / (j_centred ** 2).sum()),
                "member_share_human": float((h_part ** 2).sum() / (h_centred ** 2).sum()),
                "rho_members": correlation(j_member, h_member, ranks=False),
                "rho_pair_specific": correlation(j_rest, h_rest, ranks=False),
                "rho_members_given_text": correlation(j_member, h_member, [t_member], ranks=False),
                "rho_pair_specific_given_text": correlation(j_rest, h_rest, [t_rest], ranks=False),
                "rho_typicality_centrality": correlation(typicality, centrality),
                "rho_centrality_label_jaccard": correlation(centrality, jaccard[members, index[category]]),
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    mean = {column: _weighted_mean(table[column], table["n_members" if column in MEMBER_COLUMNS else "n_pairs"])
            for column in table.columns.drop(["category", "n_members", "n_pairs"])}
    return pd.concat([table, pd.DataFrame([{"category": WEIGHTED_MEAN, "n_members": int(table["n_members"].sum()),
                                            "n_pairs": int(table["n_pairs"].sum()), **mean}])], ignore_index=True)


DECOMPOSITION_BARS = {
    "member_part": ("Member part (typicality and expert-set centrality)", SERIES_COLORS["primary"]),
    "pair_part": ("Pair-specific part", SERIES_COLORS["accent"]),
}


def plot_human_similarity_decomposition(table: pd.DataFrame, out_path) -> None:
    """Per category and on average, the member and pair-specific parts of the Jaccard against human Spearman rho."""
    x = np.arange(len(table))
    width = 0.8 / len(DECOMPOSITION_BARS)
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, (column, (label, color)) in enumerate(DECOMPOSITION_BARS.items()):
        ax.bar(x + (offset - 0.5) * width, table[column], width, label=label, color=color)
    ax.scatter(x, table["rho"], marker="D", s=36, color=DETAIL_COLOR, zorder=3, label="Total rho, sum of the two parts")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(_category_names(table), rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Contribution to Spearman ρ")
    set_title(ax, "Agreement of the expert Jaccard with human similarity, member and pair-specific parts",
              "within-category pairs on ranks, the member part is the agreement carried by typicality")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_human_similarity_validation(scope, matrices: dict, out_dir, metric_dir, plot_dir,
                                    probing_sentences_dir=None) -> dict:
    """Every pairwise matrix and the probing-text baseline against the human pair ratings, then the profile metrics
    ranked on both criteria."""
    if matrices is None:
        return scope_summary_row(scope, **{key: float("nan") for key in HUMAN_VALIDATION_SUMMARY_LABELS})
    log.info(f"  [{scope.label}] Validating expert similarity against human ratings...")
    # A word with no experts reads Jaccard 0 rather than NaN, so it is excluded by name.
    with_experts = set(scope.expert_df["concept"].unique())
    validated = {"jaccard": matrices["jaccard"], **matrices["profiles"]}
    if probing_sentences_dir is not None:
        validated[TEXT_METRIC] = text_similarity_matrix(str(probing_sentences_dir), tuple(matrices["concepts"]))
    human = validate_against_human(
        matrices["concepts"], validated, out_dir, with_experts=with_experts, plot_dir=plot_dir,
        mantel=out_dir is not None)
    mean_rho = ({} if human.empty else
                human[human.category == WEIGHTED_MEAN].set_index("metric")["rho"].to_dict())

    baseline = (text_baseline_table(matrices["concepts"], matrices["jaccard"], validated[TEXT_METRIC], with_experts,
                                    PERMUTATIONS if out_dir is not None else 0)
                if TEXT_METRIC in validated else pd.DataFrame())
    if out_dir is not None and not baseline.empty:
        save_dataframe(baseline, out_dir / "human_similarity_text_baseline.csv")
        plot_text_baseline(baseline, out_dir / "human_similarity_text_baseline.png")
        decomposition = human_similarity_decomposition_table(
            matrices["concepts"], matrices["jaccard"], validated[TEXT_METRIC], with_experts)
        if not decomposition.empty:
            save_dataframe(decomposition, out_dir / "human_similarity_decomposition.csv")
            plot_human_similarity_decomposition(decomposition, out_dir / "human_similarity_decomposition.png")
    if metric_dir is not None:
        write_pairwise_metric_comparison(matrices["contrasts"], mean_rho, metric_dir)
    return scope_summary_row(
        scope,
        human_rho_jaccard=mean_rho.get("jaccard", float("nan")),
        human_rho_layer_profile=mean_rho.get(profile_matrix_key(DEFAULT_PROFILE_METRIC), float("nan")),
        human_rho_text_tfidf=mean_rho.get(TEXT_METRIC, float("nan")),
        human_partial_rho_jaccard_given_text=(
            baseline.set_index("category").loc[WEIGHTED_MEAN, "partial_rho_jaccard_human_given_text"]
            if not baseline.empty else float("nan")),
    )


# ---------------------------------------------------------------------------
# 2.4 Typicality: cosine typicality and every correlation with human typicality
# ---------------------------------------------------------------------------

TYPICALITY_SUMMARY_LABELS = {
    "typicality_spearman_rho": "Within-category ρ, cosine vs human typicality",
    "n_typicality_concepts": "Concepts with both scores",
    "rho_human_typicality_vs_jaccard": "Within-category ρ, human typicality vs Jaccard",
    "rho_jaccard_vs_cosine_typicality": "Within-category ρ, Jaccard vs cosine typicality",
    "rho_partial_correlation_jaccard_human_typicality_no_control": "ρ over all members, Jaccard vs human typicality",
    "rho_partial_correlation_jaccard_human_typicality":
        "Partial ρ over all members, Jaccard vs human typicality given frequency",
    "rho_partial_correlation_jaccard_human_typicality_within_given_frequency":
        "Within-category partial ρ, Jaccard vs human typicality given frequency",
}


# Columns of the global cosine typicality table, also when no category qualifies.
TYPICALITY_COLUMNS = ["category", "concept", "human_typicality", "global_cosine_typicality"]


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

    global_typicality_df = pd.DataFrame(results, columns=TYPICALITY_COLUMNS)
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


def generate_global_typicality_reports(global_typicality_df: pd.DataFrame, test: dict, typ_dir) -> None:
    """Per-category bars of human and cosine typicality, and their scatter with the least-squares line and the
    category's Spearman rho."""
    statistics = _per_group(test)
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

            rho, p, n = statistics.get(category, (np.nan, np.nan, 0))
            detail = (stats_line("Spearman ρ", rho, p if np.isfinite(p) else None, int(n)) if np.isfinite(rho)
                      else "insufficient variance for a correlation")
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            handles = draw_regression(ax, cat_global['human_typicality'], cat_global['global_cosine_typicality'],
                                      fit=bool(np.isfinite(rho)), size=40)
            for x, y, concept in zip(cat_global['human_typicality'], cat_global['global_cosine_typicality'],
                                     cat_global['concept']):
                ax.text(x, y, f"  {concept}", fontsize=8, alpha=0.8, va='center')
            if handles:
                ax.legend(handles=handles, loc="best")
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


def typicality_agreement(global_typicality_df: pd.DataFrame, n_draws: int = PERMUTATIONS) -> dict:
    """Spearman test of cosine against human typicality within each category."""
    return spearman_test(global_typicality_df["human_typicality"], global_typicality_df["global_cosine_typicality"],
                         groups=global_typicality_df["category"], n_draws=n_draws)


def _summarize_typicality(test: dict) -> dict:
    """Within-category Spearman rho between cosine and human typicality, its p and the concepts it used."""
    return {"typicality_spearman_rho": test["rho"], "typicality_spearman_p": test["p"],
            "n_typicality_concepts": test["n"]}


HUMAN_TYPICALITY_LABEL = "Human typicality (Richie and Bhatia, pairwise)"


# Training-exposure frequency per corpus, (metadata column, axis label), chosen per model config.
FREQUENCY_CORPORA = {
    "openwebtext": ("frequency_zipf_openwebtext_lemma", "OpenWebText frequency (lemma, Zipf)"),
    "fineweb": ("frequency_zipf_fineweb_lemma", "FineWeb frequency (lemma, Zipf)"),
}


def _frequency_short_label(frequency_corpus: str) -> str:
    """The corpus frequency label without its unit, as used in titles and residual axes."""
    return FREQUENCY_CORPORA[frequency_corpus][1].split(" (")[0]


def grouped_correlation_figures(n_words: int, n_categorized: int, frequency_corpus: str) -> list:
    """The two grouped figures as (x column, x label, x name, title, panels), the measure on x against its
    covariates on y, each panel within categories when either variable is category-relative."""
    frequency_col, frequency_label = FREQUENCY_CORPORA[frequency_corpus]
    # Column: (label, name, coverage denominator).
    variables = {"expert_count": ("Expert count", "expert_count", n_words),
                 "jaccard_pct": ("Jaccard % with category label", "jaccard", n_categorized),
                 "overlap_pct": ("Overlap % with category label", "overlap", n_categorized),
                 "human_typicality": (HUMAN_TYPICALITY_LABEL, "human_typicality", n_words),
                 frequency_col: (frequency_label, f"frequency_{frequency_corpus}", n_words)}

    def figure(x_col: str, title: str, y_cols: list) -> tuple:
        x_label, x_name, _ = variables[x_col]
        panels = [Panel(f"{x_name}_vs_{y_name}", f"vs_{y_name}", y_label, y_col, y_label, n_total,
                        within_category(x_col, y_col))
                  for y_col in y_cols for y_label, y_name, n_total in [variables[y_col]]]
        return x_col, x_label, x_name, title, panels

    return [figure("human_typicality", "Human typicality against expert measures and frequency",
                   ["expert_count", "jaccard_pct", "overlap_pct", frequency_col]),
            figure(frequency_col,
                   f"{_frequency_short_label(frequency_corpus)} against expert measures and human typicality",
                   ["expert_count", "jaccard_pct", "overlap_pct", "human_typicality"])]


def plot_grouped_correlations(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame,
                              concept_metadata: pd.DataFrame, corr_dir, x_names: list,
                              frequency_corpus: str) -> list:
    """The grouped figures named in x_names, each measure on x against expert count, Jaccard, overlap and the others."""
    wide = merged_metadata_df.copy()
    if similarity_metrics_df is not None and not similarity_metrics_df.empty:
        wide = wide.merge(similarity_metrics_df[["concept", "category", "jaccard_pct", "overlap_pct"]],
                          on=["concept", "category"], how="left")
    else:
        wide["jaccard_pct"], wide["overlap_pct"] = np.nan, np.nan
    n_categorized = int(concept_metadata["category"].notna().sum())
    summary_rows = []
    for x_col, x_label, x_name, title, panels in grouped_correlation_figures(
            len(concept_metadata), n_categorized, frequency_corpus):
        if x_name not in x_names:
            continue
        if x_col not in wide.columns:
            log.warning(f"  Skipping {x_name}_correlations: metadata has no {x_col} column.")
            continue
        panels = [panel for panel in panels if panel.y_col in wide.columns]
        if corr_dir is None:
            summary_rows += regression_grid_rows(wide, x_col, panels)
        else:
            summary_rows += run_regression_grid(wide, x_col, x_label, panels, title,
                                                corr_dir / f"{x_name}_correlations.csv",
                                                corr_dir / f"{x_name}_correlations.png")
    return summary_rows


JACCARD_LABEL = "Jaccard with category label"


def _partial_correlation_grid(corr_data: pd.DataFrame, corr_dir, n_total_relevant: int, plot_name: str,
                              file_stem: str, x: tuple, other: tuple) -> list:
    """x against the Jaccard with its category label as one row of panels ordered by increasing rho with one CSV,
    Spearman's rho over all members and within categories, each without and given the other measure, on the members
    holding every value. Returns the correlation_summary rows but that of the within-category panel without control,
    which the grouped figure reports, x as (column, label) and other as (column, name, label)."""
    (x_col, x_label), (other_col, other_name, other_label) = x, other
    missing = [column for column in (x_col, "jaccard_pct", other_col, "category") if column not in corr_data.columns]
    if missing:
        log.warning(f"  Skipping {file_stem}: no {', '.join(missing)} column.")
        return []
    data = corr_data.dropna(subset=[x_col, "jaccard_pct", other_col, "category"])
    panel = functools.partial(Panel, y_col="jaccard_pct", y_label=JACCARD_LABEL, n_total_relevant=n_total_relevant)
    panels = [panel(f"{plot_name}_no_control", "no_control", "No control"),
              panel(plot_name, f"given_{other_name}", f"Given {other_label}", controls=(other_col,)),
              panel(None, "within", "Within category", within=True),
              panel(f"{plot_name}_within_given_{other_name}", f"within_given_{other_name}",
                    f"Within category, given {other_label}", within=True, controls=(other_col,))]
    if corr_dir is None:
        return regression_grid_rows(data, x_col, panels)
    return run_regression_grid(data, x_col, x_label, panels,
                               f"{x_label} against {JACCARD_LABEL}, panels ordered by increasing ρ",
                               corr_dir / f"{file_stem}.csv", corr_dir / f"{file_stem}.png", order_by_rho=True)


def plot_human_typicality_partial_correlations(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame,
                                               concept_metadata: pd.DataFrame, corr_dir, frequency_corpus: str) -> list:
    """Jaccard against human typicality without control and given the model's corpus frequency, over all members and
    within categories."""
    corr_data = _concept_parent_frame(merged_metadata_df, similarity_metrics_df)
    if corr_data.empty:
        return []
    return _partial_correlation_grid(
        corr_data, corr_dir, concept_metadata['category'].notna().sum(),
        "partial_correlation_jaccard_human_typicality", "human_typicality_partial_correlations",
        ("human_typicality", "Human typicality"),
        (FREQUENCY_CORPORA[frequency_corpus][0], "frequency", _frequency_short_label(frequency_corpus)))


def plot_frequency_partial_correlations(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame,
                                        concept_metadata: pd.DataFrame, corr_dir, frequency_corpus: str) -> list:
    """Jaccard against the model's corpus frequency without control and given human typicality, over all members and
    within categories."""
    corr_data = _concept_parent_frame(merged_metadata_df, similarity_metrics_df)
    if corr_data.empty:
        return []
    return _partial_correlation_grid(
        corr_data, corr_dir, concept_metadata['category'].notna().sum(),
        "partial_correlation_jaccard_frequency", f"frequency_{frequency_corpus}_partial_correlations",
        (FREQUENCY_CORPORA[frequency_corpus][0], _frequency_short_label(frequency_corpus)),
        ("human_typicality", "human_typicality", "human typicality"))


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
                   similarity_metrics_df: pd.DataFrame, out_dir, frequency_corpus: str) -> dict:
    """Cosine typicality of one scope, global and per layer, and every correlation with human typicality."""
    log.info(f"  [{scope.label}] Computing Empirical Cosine Typicality...")
    typ_global_df = compute_global_typicality(scope.expert_df, concept_metadata, out_dir)
    typicality_test = typicality_agreement(typ_global_df)
    if out_dir is not None:
        generate_global_typicality_reports(typ_global_df, typicality_test, out_dir)
        for suffix, axis_df, axis_label in axis_variants(scope):
            log.info(f"  [{scope.label} / {axis_label}] Computing per-layer typicality prototypes...")
            typ_layer_df = compute_layer_typicality(axis_df, concept_metadata, out_dir, suffix)
            generate_layer_typicality_reports(typ_layer_df, out_dir, suffix)

    log.info(f"  [{scope.label}] Generating human typicality correlation plots...")
    rows = plot_grouped_correlations(merged_meta, similarity_metrics_df, concept_metadata, out_dir,
                                     ["human_typicality"], frequency_corpus)
    rows += plot_human_typicality_partial_correlations(merged_meta, similarity_metrics_df, concept_metadata, out_dir,
                                                       frequency_corpus)
    rows += plot_jaccard_vs_cosine_typicality(similarity_metrics_df, typ_global_df, concept_metadata, out_dir)
    summary_df = _write_correlation_summary(rows, out_dir)
    return scope_summary_row(scope, **_summarize_typicality(typicality_test), **panel_rho_columns(summary_df))


# ---------------------------------------------------------------------------
# 2.5 Frequency correlations
# ---------------------------------------------------------------------------

FREQUENCY_SUMMARY_LABELS = {
    **{f"rho_frequency_{corpus}_vs_{y_name}":
       f"{'Within-category ρ' if within else 'ρ'}, {_frequency_short_label(corpus)} vs {y_label}"
       for corpus in FREQUENCY_CORPORA
       for y_name, y_label, within in (("expert_count", "expert count", False), ("jaccard", "Jaccard", True),
                                       ("overlap", "overlap", True), ("human_typicality", "human typicality", True))},
    "rho_partial_correlation_jaccard_frequency_no_control": "ρ over all members, Jaccard vs frequency",
    "rho_partial_correlation_jaccard_frequency":
        "Partial ρ over all members, Jaccard vs frequency given human typicality",
    "rho_partial_correlation_jaccard_frequency_within_given_human_typicality":
        "Within-category partial ρ, Jaccard vs frequency given human typicality",
}


def run_frequency_correlations(scope, concept_metadata: pd.DataFrame, merged_meta: pd.DataFrame,
                               similarity_metrics_df: pd.DataFrame, out_dir, frequency_corpus: str) -> dict:
    """The model's corpus frequency against expert count, Jaccard, overlap and human typicality, and Jaccard
    against frequency without control and given human typicality, over all members and within categories."""
    log.info(f"  [{scope.label}] Generating frequency correlation plots...")
    rows = plot_grouped_correlations(merged_meta, similarity_metrics_df, concept_metadata, out_dir,
                                     [f"frequency_{frequency_corpus}"], frequency_corpus)
    rows += plot_frequency_partial_correlations(merged_meta, similarity_metrics_df, concept_metadata, out_dir,
                                                frequency_corpus)
    summary_df = _write_correlation_summary(rows, out_dir)
    return scope_summary_row(scope, **panel_rho_columns(summary_df))


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
                                  write_outputs: bool = True, *, frequency_corpus: str,
                                  probing_sentences_dir=None) -> dict:
    """Run every section on one scope, returns {section: sublayer_comparison row}.

    With write_outputs False nothing is written and only the default profile metric is computed,
    the only one the summary rows report. Without probing_sentences_dir section 2.3 skips the text baseline.
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
                                                  profile("similarity_correlation"), probing_sentences_dir)
    rows["2.4"] = run_typicality(scope, concept_metadata, merged_meta, similarity_metrics_df, out("2.4"),
                                 frequency_corpus)
    rows["2.5"] = run_frequency_correlations(scope, concept_metadata, merged_meta, similarity_metrics_df, out("2.5"),
                                             frequency_corpus)
    return rows
