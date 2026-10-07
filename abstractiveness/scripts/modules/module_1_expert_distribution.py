"""
Module 1: expert distribution. Where along depth each word's experts sit, how spread they are,
and which sublayers carry category information. Explanations live in documentation/.

  1.1 layer expert distribution   distributions, cumulative mass, peak and average layer
  1.2 distribution shape          Shannon entropy, Geary's C, entropy against expert count
  1.3 sublayer informativeness    sublayers ranked by category alignment, whole model only
"""

import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import entropy
from utils.logging_and_io import save_dataframe
from modules.shared_layer_distribution_measures import (gearys_c, build_layer_probability_matrix,
                                                        PEAK_GAP_RELIABLE_PP)
from modules.shared_expert_set_measures import pair_similarity_vector, category_alignment_metrics
from core.analysis_scopes import (axis_variants, scope_section_dir, scope_summary_row, to_block_axis,
                                  sublayer_expert_counts)
from core.expert_data_loading import expert_counts_with_metadata
from core.correlation_reporting import panel_r_columns, regression_row
from utils.plotting import (_plot_bar_with_leaders, apply_rotated_leader_labels, fig_width_for,
                            plot_comparison_violin, build_category_color_map, set_title, set_suptitle,
                            stats_line, coverage_line, twin_axis, LEVEL_COLORS, CORRELATION_COLORS)

log = logging.getLogger(__name__)


SECTION_DIRS = {
    "1.1": "1.1_layer_expert_distribution",
    "1.2": "1.2_distribution_shape",
    "1.3": "1.3_sublayer_informativeness",
}


# ---------------------------------------------------------------------------
# Layer descriptors, shared by sections 1.1 and 1.2
# ---------------------------------------------------------------------------

LOCATION_DESCRIPTORS = ["peak_layer", "peak_gap_pct", "avg_layer", "avg_layer_top25pct"]
SHAPE_DESCRIPTORS = ["shannon_entropy", "gearys_c"]
AVERAGE_COLUMN_NAMES = {
    "shannon_entropy": "shannon_entropy_average",
    "gearys_c": "gearys_c_average",
    "peak_layer": "peak_layer_average",
    "peak_gap_pct": "peak_gap_pct_average",
    "avg_layer": "avg_layer_average",
    "avg_layer_top25pct": "avg_layer_average_top25pct",
}


def layer_distribution_descriptors(probs, layer_indices, top_frac: float = 0.25) -> dict:
    """Entropy, Geary's C, peak layer and gap, full and top-25% average layer of one distribution."""
    if probs is None:
        return {key: np.nan for key in
                ("shannon_entropy", "gearys_c", "peak_layer", "peak_gap_pct",
                 "avg_layer", "avg_layer_top25pct")}

    top_two = np.partition(probs, -2)[-2:] if len(probs) >= 2 else None

    n_top = int(np.ceil(top_frac * len(probs)))
    top_idx = np.argpartition(probs, -n_top)[-n_top:]
    top_mass = probs[top_idx]

    return {
        "shannon_entropy": entropy(probs, base=2),
        "gearys_c": gearys_c(probs),
        "peak_layer": layer_indices[np.argmax(probs)],
        "peak_gap_pct": (top_two.max() - top_two.min()) * 100 if top_two is not None else np.nan,
        "avg_layer": np.sum(probs * layer_indices),
        "avg_layer_top25pct": (float(np.sum(top_mass * layer_indices[top_idx]) / top_mass.sum())
                               if top_mass.sum() > 0 else np.nan),
    }


def compute_layer_descriptors(expert_allocation_df: pd.DataFrame,
                              concept_metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptor tables per word and per category (label and member average)."""
    dist_matrix, prob_matrix = build_layer_probability_matrix(expert_allocation_df)
    layer_indices = prob_matrix.columns.values

    concept_results = []
    for concept in prob_matrix.index:
        probs = prob_matrix.loc[concept].values
        if np.sum(probs) == 0:
            continue
        concept_results.append({
            "concept": concept,
            **layer_distribution_descriptors(probs, layer_indices),
            "experts_count": dist_matrix.loc[concept].sum(),
        })
    concept_entropy_df = pd.DataFrame(concept_results)

    category_results = []
    for category in concept_metadata['category'].dropna().unique():
        members = concept_metadata[concept_metadata['category'] == category]['concept'].tolist()
        valid_members = [m for m in members if m in prob_matrix.index]

        label_probs = prob_matrix.loc[category].values if category in prob_matrix.index else None
        member_avg_probs = prob_matrix.loc[valid_members].mean(axis=0).values if valid_members else None

        if label_probs is None and member_avg_probs is None:
            continue

        if (label_probs is not None and member_avg_probs is not None
                and np.std(label_probs) > 0 and np.std(member_avg_probs) > 0):
            dist_corr, _ = stats.pearsonr(label_probs, member_avg_probs)
        elif label_probs is not None and member_avg_probs is not None:
            dist_corr = 0.0
        else:
            dist_corr = np.nan

        label_desc = layer_distribution_descriptors(label_probs, layer_indices)
        member_desc = layer_distribution_descriptors(member_avg_probs, layer_indices)
        category_results.append({
            "category": category,
            **label_desc,
            **{AVERAGE_COLUMN_NAMES[k]: v for k, v in member_desc.items()},
            "pearson_correlation_distributions": dist_corr,
            "member_count": len(valid_members),
        })
    category_entropy_df = pd.DataFrame(category_results)

    return concept_entropy_df, category_entropy_df


def layer_descriptors_by_axis(scope, concept_metadata: pd.DataFrame) -> list:
    """(suffix, axis_df, axis_label, concept table, category table) per axis variant, block axis first."""
    return [(suffix, axis_df, axis_label, *compute_layer_descriptors(axis_df, concept_metadata))
            for suffix, axis_df, axis_label in axis_variants(scope)]


def save_descriptor_tables(concept_df: pd.DataFrame, category_df: pd.DataFrame, out_dir, stem: str,
                           descriptors: list, suffix: str = "", category_extra: tuple = ()) -> None:
    """Chosen descriptor columns of both tables, as <stem>_concepts and <stem>_categories CSVs."""
    concept_cols = ["concept", *descriptors, "experts_count"]
    category_cols = ["category", *descriptors, *(AVERAGE_COLUMN_NAMES[d] for d in descriptors),
                     *category_extra, "member_count"]
    save_dataframe(concept_df.reindex(columns=concept_cols), out_dir / f"{stem}_concepts{suffix}.csv")
    save_dataframe(category_df.reindex(columns=category_cols), out_dir / f"{stem}_categories{suffix}.csv")


def _log_level_comparison(name: str, category_values: pd.Series, concept_values: pd.Series) -> None:
    """Log a two-sided Mann-Whitney test with AUC, labels against concepts."""
    k, c = category_values.dropna(), concept_values.dropna()
    if len(k) == 0 or len(c) == 0:
        return
    u_stat, p_val = stats.mannwhitneyu(k, c, alternative='two-sided')
    auc = u_stat / (len(k) * len(c))
    log.info(f"  [Stats] {name} (labels vs concepts): U={u_stat:.1f}, p={p_val:.4e}, AUC(label>concept)={auc:.3f}")


# ---------------------------------------------------------------------------
# 1.1 Layer expert distribution: where along depth the experts sit
# ---------------------------------------------------------------------------

LAYER_DISTRIBUTION_SUMMARY_LABELS = {
    "n_experts": "Expert rows in scope",
    "mean_expert_count_concepts": "Mean experts per concept",
    "mean_expert_count_categories": "Mean experts per category label",
}


ABSTRACTION_COLORS = {1: LEVEL_COLORS["Broad Categories"], 2: LEVEL_COLORS["Specific Concepts"]}


def compute_layer_distributions(expert_allocation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Per-concept percentage and count matrices by layer, their per-level mean, and expert counts."""
    concept_count_matrix = pd.crosstab(
        index=[expert_allocation_df['abstraction_level'], expert_allocation_df['concept']],
        columns=expert_allocation_df['layer_name'],
    )
    concept_distribution_matrix = concept_count_matrix.div(concept_count_matrix.sum(axis=1), axis=0) * 100
    global_layer_dist = concept_distribution_matrix.groupby(level='abstraction_level').mean().reset_index()
    global_layer_dist = global_layer_dist.melt(
        id_vars='abstraction_level',
        var_name='layer_name',
        value_name='mean_expert_allocation_pct'
    )
    level_counts = (concept_count_matrix.groupby(level='abstraction_level').sum()
                    .reset_index()
                    .melt(id_vars='abstraction_level', var_name='layer_name', value_name='total_expert_count'))
    global_layer_dist = global_layer_dist.merge(level_counts, on=['abstraction_level', 'layer_name'])
    global_layer_dist['layer_name'] = pd.Categorical(
        global_layer_dist['layer_name'], categories=concept_count_matrix.columns, ordered=True)
    global_layer_dist = global_layer_dist.sort_values(['abstraction_level', 'layer_name']).reset_index(drop=True)
    global_layer_dist['cumulative_pct'] = global_layer_dist.groupby('abstraction_level')['mean_expert_allocation_pct'].cumsum()
    expert_counts = expert_allocation_df.groupby('concept').size()

    return concept_distribution_matrix, concept_count_matrix, global_layer_dist, expert_counts


def plot_global_distribution(global_layer_distribution_df: pd.DataFrame, dist_dir, suffix: str = "",
                             name: str = "mean") -> None:
    """Save the per-level mean layer distribution."""
    save_dataframe(global_layer_distribution_df, dist_dir / f"{name}_expert_layer_distribution{suffix}.csv")


def _stacked_bars(ax, layer_order: list, stack: pd.DataFrame, colors: dict) -> None:
    """Bars built from per-sublayer shares, one stacked segment per sublayer, in layer_order."""
    bottom = np.zeros(len(layer_order))
    for sublayer in stack.columns:
        values = stack[sublayer].reindex(layer_order).fillna(0.0).to_numpy()
        ax.bar(range(len(layer_order)), values, bottom=bottom, color=colors[sublayer],
               edgecolor='black', linewidth=0.5, label=sublayer)
        bottom += values


def plot_cumulative_layer_distribution(global_layer_dist: pd.DataFrame, dist_dir,
                                       suffix: str = "", title_detail: str = None, name: str = "mean",
                                       title_name: str = "Mean", stack: pd.DataFrame = None,
                                       stack_colors: dict = None, bar_label: str = "Average % of experts") -> None:
    """Depth-ordered mean distribution per level with cumulative mass curves."""
    layer_order = list(dict.fromkeys(global_layer_dist['layer_name']))
    n = len(layer_order)
    fig, ax1 = plt.subplots(figsize=(fig_width_for(n, 0.28, min_w=16.0), 10.4))
    if stack is None:
        sns.barplot(data=global_layer_dist, x='layer_name', y='mean_expert_allocation_pct',
                    hue='abstraction_level', palette=ABSTRACTION_COLORS,
                    edgecolor='black', linewidth=0.5, order=layer_order, ax=ax1)
        legend_title = "Abstraction level"
    else:
        _stacked_bars(ax1, layer_order, stack, stack_colors)
        ax1.set_xlim(-0.5, n - 0.5)
        legend_title = "Sublayer"
    apply_rotated_leader_labels(ax1, layer_order, fontsize=9)
    ax1.set_xlabel("Model layer", fontsize=14)
    ax1.set_ylabel(f"{bar_label} (bars)", fontsize=14)

    ax2 = twin_axis(ax1)
    ax2.set_xlim(ax1.get_xlim())
    for level, level_df in global_layer_dist.groupby('abstraction_level'):
        curve = level_df.set_index('layer_name')['cumulative_pct'].reindex(layer_order)
        idx50 = int(np.argmax(curve.values >= 50))
        idx90 = int(np.argmax(curve.values >= 90))
        label = (f"Cumulative, level {level}: 50% reached by {layer_order[idx50]}, "
                 f"90% by {layer_order[idx90]}")
        ax2.plot(range(n), curve.values, color=ABSTRACTION_COLORS[level], linewidth=2,
                 marker='o', markersize=3, alpha=0.9, label=label)
        for idx in (idx50, idx90):
            y = curve.values[idx]
            ax2.plot([idx, idx], [0, y], linestyle=':', linewidth=1.2,
                     color=ABSTRACTION_COLORS[level], alpha=0.8)
            ax2.plot([idx, ax2.get_xlim()[1]], [y, y], linestyle=':', linewidth=1.2,
                     color=ABSTRACTION_COLORS[level], alpha=0.8)
        ax2.scatter([idx50, idx90], [curve.values[idx50], curve.values[idx90]],
                    color=ABSTRACTION_COLORS[level], marker='D', s=70,
                    edgecolor='black', linewidth=0.8, zorder=5)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative % of experts (lines)", fontsize=14)
    ax2.grid(False)
    bar_handles, bar_labels = ax1.get_legend_handles_labels()
    line_handles, line_labels = ax2.get_legend_handles_labels()
    if ax1.get_legend():
        ax1.get_legend().remove()
    ax1.legend(bar_handles + line_handles, [f"{legend_title} {label}" for label in bar_labels] + line_labels,
               loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

    set_title(ax1, f"{title_name} expert distribution with cumulative mass", title_detail)
    plt.tight_layout()
    plt.savefig(dist_dir / f"{name}_expert_layer_distribution_cumulative{suffix}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_layer_distribution_sorted(global_layer_dist: pd.DataFrame, dist_dir,
                                              suffix: str = "", title_detail: str = None,
                                              name: str = "mean", title_name: str = None,
                                              stack: pd.DataFrame = None, stack_colors: dict = None,
                                              bar_label: str = "Average % of experts") -> None:
    """Layers sorted by expert mass per level, with the cumulative share of the top layers."""
    levels = sorted(global_layer_dist['abstraction_level'].unique())
    n = global_layer_dist['layer_name'].nunique()
    fig, axes = plt.subplots(len(levels), 1,
                             figsize=(fig_width_for(n, 0.28, min_w=16.0), 7.0 * len(levels)))
    axes = np.atleast_1d(axes)

    for ax, level in zip(axes, levels):
        level_df = (global_layer_dist[global_layer_dist['abstraction_level'] == level]
                    .sort_values('mean_expert_allocation_pct', ascending=False).reset_index(drop=True))
        sorted_cumulative = level_df['mean_expert_allocation_pct'].cumsum()
        k50 = int(np.argmax(sorted_cumulative.values >= 50))
        k90 = int(np.argmax(sorted_cumulative.values >= 90))

        if stack is None:
            ax.bar(range(len(level_df)), level_df['mean_expert_allocation_pct'],
                   color=ABSTRACTION_COLORS[level], edgecolor='black', linewidth=0.5)
        else:
            _stacked_bars(ax, level_df['layer_name'].astype(str).tolist(), stack, stack_colors)
            ax.legend(title="Sublayer", loc='upper right')
        apply_rotated_leader_labels(ax, level_df['layer_name'].astype(str).tolist(), fontsize=9)
        ax.set_ylabel(f"{bar_label} (bars)", fontsize=13)
        set_title(ax, f"Abstraction level {level}, layers sorted by expert mass",
                  f"top {k50 + 1} layers hold 50% of the mass, top {k90 + 1} of {len(level_df)} hold 90%")

        ax2 = twin_axis(ax)
        ax2.set_xlim(ax.get_xlim())
        ax2.plot(range(len(level_df)), sorted_cumulative.values, color="#444444",
                 linewidth=2, marker='o', markersize=3)
        for k in (k50, k90):
            y = sorted_cumulative.values[k]
            ax2.plot([k, k], [0, y], linestyle=':', linewidth=1.2,
                     color=ABSTRACTION_COLORS[level], alpha=0.8)
            ax2.plot([k, ax2.get_xlim()[1]], [y, y], linestyle=':', linewidth=1.2,
                     color=ABSTRACTION_COLORS[level], alpha=0.8)
        ax2.scatter([k50, k90], [sorted_cumulative.values[k50], sorted_cumulative.values[k90]],
                    color=ABSTRACTION_COLORS[level], marker='D', s=70,
                    edgecolor='black', linewidth=0.8, zorder=5)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Cumulative % of experts (line)", fontsize=13)
        ax2.grid(False)

    axes[-1].set_xlabel("Model layer, descending expert mass", fontsize=14)
    heading = f"{title_name}, cumulative expert mass concentration" if title_name else \
        "Cumulative expert mass concentration"
    set_suptitle(fig, f"{heading}, {title_detail}" if title_detail else heading, y=1.0)
    plt.tight_layout()
    plt.savefig(dist_dir / f"{name}_expert_layer_distribution_cumulative_sorted{suffix}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def sublayer_shares_by_block(expert_allocation_df: pd.DataFrame) -> pd.DataFrame:
    """Expert count per (concept, block) and sublayer, block names as to_block_axis writes them."""
    names = expert_allocation_df["layer_name"].astype(str)
    block = names.str.extract(r"^\d+\.L\.(\d+)\.")[0].astype(int)
    sublayer = names.str.extract(r"^\d+\.L\.\d+\.(.+)$")[0]
    order = list(sublayer_expert_counts(expert_allocation_df).index)
    counts = pd.crosstab([expert_allocation_df["concept"].to_numpy(), (block + 1).astype(str) + ".B." + block.astype(str)],
                         sublayer)
    return counts[[s for s in order if s in counts.columns]]


def plot_per_concept_distributions(scope, dist_dir) -> None:
    """The four cumulative plots of this section and their CSVs for every concept, on both layer axes."""
    concept_root = dist_dir / "per_concept"
    block_counts = sublayer_shares_by_block(scope.expert_df)
    colors = build_category_color_map(list(block_counts.columns))

    for suffix, axis_df, axis_label in axis_variants(scope):
        concept_pct, concept_counts, _, _ = compute_layer_distributions(axis_df)
        layer_order = list(concept_counts.columns)
        title_detail = f"{scope.label}, {axis_label}"
        for (level, concept), pct in concept_pct.iterrows():
            concept_dir = concept_root / concept
            concept_dir.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame({
                "abstraction_level": level,
                "layer_name": pd.Categorical(layer_order, categories=layer_order, ordered=True),
                "mean_expert_allocation_pct": pct.to_numpy(),
                "total_expert_count": concept_counts.loc[(level, concept)].to_numpy(),
            })
            frame["cumulative_pct"] = frame["mean_expert_allocation_pct"].cumsum()
            stack = None
            if suffix == "_by_block":
                counts = block_counts.loc[concept].reindex(layer_order).fillna(0)
                stack = counts.div(counts.to_numpy().sum()) * 100
            table = frame.rename(columns={"mean_expert_allocation_pct": "expert_allocation_pct",
                                          "total_expert_count": "expert_count"})
            if stack is not None:
                table = table.join(counts.add_prefix("experts_").reset_index(drop=True))
            save_dataframe(table, concept_dir / f"{concept}_expert_layer_distribution{suffix}.csv")
            title_name = concept.capitalize()
            plot_cumulative_layer_distribution(frame, concept_dir, suffix, title_detail, name=concept,
                                               title_name=title_name, stack=stack, stack_colors=colors,
                                               bar_label="% of experts")
            plot_cumulative_layer_distribution_sorted(frame, concept_dir, suffix, title_detail, name=concept,
                                                      title_name=title_name, stack=stack,
                                                      stack_colors=colors, bar_label="% of experts")


def plot_peak_average_distributions(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, out_dir, suffix: str = "") -> None:
    """Peak-layer histogram (hatched where ambiguous) with average-layer densities."""
    layer_labels = expert_allocation_df['layer_name'].cat.categories.tolist()

    concept_subset = concept_entropy_df[['concept', 'peak_layer', 'peak_gap_pct', 'avg_layer', 'avg_layer_top25pct']].rename(columns={'concept': 'name'}).copy()
    concept_subset['group_type'] = 'Specific Concepts'
    category_subset = category_entropy_df[['category', 'peak_layer', 'peak_gap_pct', 'avg_layer', 'avg_layer_top25pct']].rename(columns={'category': 'name'}).copy()
    category_subset['group_type'] = 'Broad Categories'
    category_avg_subset = category_entropy_df[['category', 'peak_layer_average', 'peak_gap_pct_average', 'avg_layer_average', 'avg_layer_average_top25pct']].rename(
        columns={'category': 'name', 'peak_layer_average': 'peak_layer', 'peak_gap_pct_average': 'peak_gap_pct',
                 'avg_layer_average': 'avg_layer', 'avg_layer_average_top25pct': 'avg_layer_top25pct'}).copy()
    category_avg_subset['group_type'] = 'Broad Categories (Avg)'

    plot_df = pd.concat([concept_subset, category_subset, category_avg_subset], ignore_index=True).dropna(subset=['peak_layer', 'avg_layer'])
    layer_support = [int(str(name).split('.', 1)[0]) for name in layer_labels]
    positions = np.arange(len(layer_support))
    plot_df['peak_layer_idx'] = plot_df['peak_layer'].map(dict(zip(layer_support, positions))).astype(int)
    plot_df['avg_layer_idx'] = np.interp(plot_df['avg_layer'], layer_support, positions)
    plot_df['avg_layer_top25_idx'] = np.interp(plot_df['avg_layer_top25pct'], layer_support, positions)
    plot_df['reliable_peak'] = plot_df['peak_gap_pct'] >= PEAK_GAP_RELIABLE_PP

    n_layers = len(layer_labels)
    fig, ax1 = plt.subplots(figsize=(fig_width_for(n_layers, 0.28, min_w=16.0), 6))
    palette = {**LEVEL_COLORS, "Broad Categories (Avg)": "#8A9BB5"}
    group_order = ['Specific Concepts', 'Broad Categories', 'Broad Categories (Avg)']

    bar_width = 0.8 / len(group_order)
    for i, group in enumerate(group_order):
        group_df = plot_df[plot_df['group_type'] == group]
        total = len(group_df)
        if total == 0:
            continue
        reliable_pct = (group_df[group_df['reliable_peak']].groupby('peak_layer_idx').size()
                        .reindex(range(n_layers), fill_value=0) / total * 100)
        ambiguous_pct = (group_df[~group_df['reliable_peak']].groupby('peak_layer_idx').size()
                         .reindex(range(n_layers), fill_value=0) / total * 100)
        x = np.arange(n_layers) - 0.4 + bar_width * (i + 0.5)
        ax1.bar(x, reliable_pct.values, bar_width, color=palette[group], edgecolor='black', linewidth=0.6)
        ax1.bar(x, ambiguous_pct.values, bar_width, bottom=reliable_pct.values,
                color=palette[group], edgecolor='black', linewidth=0.4, hatch='///', alpha=0.55)

    legend_handles = [Patch(facecolor=palette[g], edgecolor='black', label=g) for g in group_order]
    legend_handles.append(Patch(facecolor='white', edgecolor='black', hatch='///',
                                label=f'Ambiguous peak (gap < {PEAK_GAP_RELIABLE_PP:g} pp)'))
    ax1.legend(handles=legend_handles, title="Group", loc='upper left')

    set_title(ax1, "Peak layer and average layer by abstraction level")
    ax1.set_xlabel("Model layer")
    ax1.set_ylabel("% of group peaking in this layer (bars)")
    apply_rotated_leader_labels(ax1, layer_labels, fontsize=9)
    ax1.set_xlim(-0.5, n_layers - 0.5)

    ax2 = twin_axis(ax1)
    sns.kdeplot(
        data=plot_df, x='avg_layer_idx', hue='group_type',
        palette=palette, fill=True, alpha=0.2, linewidth=2,
        ax=ax2, legend=False, common_norm=False,
        clip=(-0.5, len(layer_labels) - 0.5)
    )
    sns.kdeplot(
        data=plot_df, x='avg_layer_top25_idx', hue='group_type',
        palette=palette, fill=False, linewidth=2, linestyle='--',
        ax=ax2, legend=False, common_norm=False,
        clip=(-0.5, len(layer_labels) - 0.5)
    )
    style_handles = [
        Line2D([0], [0], color='black', lw=2, ls='-', label='Avg layer (full distribution)'),
        Line2D([0], [0], color='black', lw=2, ls='--', label='Avg layer (top 25% most-loaded layers)'),
    ]
    ax2.legend(handles=style_handles, loc='upper center')
    ax2.set_xlim(-0.5, len(layer_labels) - 0.5)
    ax2.set_ylabel("Density of average layer (curves)")
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(out_dir / f"peak_average_layers{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_avg_layer_violin(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, out_dir, suffix: str = "") -> None:
    """Top-25% average layer violins, concepts against category labels."""
    plot_comparison_violin(
        {"Specific Concepts": concept_entropy_df['avg_layer_top25pct'],
         "Broad Categories": category_entropy_df['avg_layer_top25pct']},
        y_label="Average layer (top 25% most-loaded layers)",
        title="Average expert layer, category labels against concepts",
        out_path=out_dir / f"category_concept_avg_layer_violin{suffix}.png",
        palette=LEVEL_COLORS)


def plot_peak_gap_ecdf(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, out_dir, suffix: str = "") -> None:
    """ECDF of the peak dominance gap per abstraction level."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fractions = {}
    for label, values in (("Specific Concepts", concept_entropy_df['peak_gap_pct'].dropna()),
                          ("Broad Categories", category_entropy_df['peak_gap_pct'].dropna())):
        sns.ecdfplot(values, ax=ax, color=LEVEL_COLORS[label], linewidth=2, label=label)
        fractions[label] = (values < PEAK_GAP_RELIABLE_PP).mean()
    ax.axvline(PEAK_GAP_RELIABLE_PP, linestyle='--', linewidth=1, color='#999999')
    ax.text(0.97, 0.35,
            f"Gap < {PEAK_GAP_RELIABLE_PP:g} pp (ambiguous peak)\n"
            + "\n".join(f"{label}: {frac:.0%}" for label, frac in fractions.items()),
            transform=ax.transAxes, ha='right', va='center',
            bbox=dict(boxstyle="square,pad=0.5", facecolor="white", edgecolor="#dddddd"))
    ax.set_xlabel("Peak dominance gap (percentage points)")
    ax.set_ylabel("Fraction of words with gap below x")
    ax.legend(loc='lower right')
    set_title(ax, "Peak dominance gap by abstraction level", "empirical cumulative distribution")
    fig.tight_layout()
    fig.savefig(out_dir / f"peak_gap_ecdf{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def _summarize_expert_counts(merged_meta: pd.DataFrame, concept_df: pd.DataFrame) -> dict:
    """Mean expert count per word at each abstraction level, plus the mean concept peak gap."""
    peak_gap = concept_df['peak_gap_pct'].mean() if 'peak_gap_pct' in concept_df else np.nan
    if merged_meta.empty or 'abstraction_level' not in merged_meta.columns:
        return {"mean_expert_count_categories": np.nan, "mean_expert_count_concepts": np.nan,
                "peak_gap_mean_concepts_pct": peak_gap}
    by_level = merged_meta.groupby('abstraction_level')['expert_count'].mean()
    return {"mean_expert_count_categories": by_level.get(1, np.nan),
            "mean_expert_count_concepts": by_level.get(2, np.nan),
            "peak_gap_mean_concepts_pct": peak_gap}


def run_layer_distribution(scope, concept_metadata: pd.DataFrame, merged_meta: pd.DataFrame, descriptors: list,
                           out_dir, per_concept: bool = True) -> dict:
    """Layer expert distribution of one scope, where its experts sit, per-concept plots on the whole model only."""
    log.info(f"  [{scope.label}] Generating combined metadata counts...")
    save_dataframe(merged_meta, out_dir / "expert_counts_with_metadata.csv")

    for suffix, axis_df, axis_label, concept_df, category_df in descriptors:
        log.info(f"  [{scope.label} / {axis_label}] Computing layer distribution matrices...")
        _, _, global_dist, _ = compute_layer_distributions(axis_df)
        plot_global_distribution(global_dist, out_dir, suffix)
        title_detail = f"{scope.label}, {axis_label}"
        plot_cumulative_layer_distribution(global_dist, out_dir, suffix, title_detail)
        plot_cumulative_layer_distribution_sorted(global_dist, out_dir, suffix, title_detail)

        save_descriptor_tables(concept_df, category_df, out_dir, "layer_location", LOCATION_DESCRIPTORS, suffix,
                               category_extra=("pearson_correlation_distributions",))
        log.info("  Generating peak vs. average layer distributions plot...")
        plot_peak_average_distributions(axis_df, concept_metadata, concept_df, category_df, out_dir, suffix)
        log.info("  Generating average-layer violin plot...")
        plot_avg_layer_violin(concept_df, category_df, out_dir, suffix)
        log.info("  Generating peak dominance gap ECDF...")
        plot_peak_gap_ecdf(concept_df, category_df, out_dir, suffix)
        _log_level_comparison("Peak dominance gap", category_df['peak_gap_pct'], concept_df['peak_gap_pct'])
        _log_level_comparison("Avg layer (full)", category_df['avg_layer'], concept_df['avg_layer'])
        _log_level_comparison("Avg layer (top 25% layers)", category_df['avg_layer_top25pct'],
                              concept_df['avg_layer_top25pct'])

    if scope.is_whole_model and per_concept:
        log.info("  Generating per-concept cumulative layer distribution plots...")
        plot_per_concept_distributions(scope, out_dir)

    return scope_summary_row(scope, **_summarize_expert_counts(merged_meta, descriptors[0][3]))


# ---------------------------------------------------------------------------
# 1.2 Distribution shape: how spread the experts are
# ---------------------------------------------------------------------------

SHAPE_SUMMARY_LABELS = {
    "entropy_auc_categories_over_concepts": "AUC, label entropy over concept entropy",
    "entropy_mean_categories": "Mean entropy, category labels (bits)",
    "entropy_mean_concepts": "Mean entropy, concepts (bits)",
    "gearys_c_mean_concepts": "Mean Geary's C, concepts",
    "r_shannon_entropy_vs_expert_count": "r, entropy vs expert count",
}


def _log_entropy_test(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame) -> None:
    """Log the one-sided Mann-Whitney test that labels have lower entropy than concepts."""
    clean_concept_ent = concept_entropy_df['shannon_entropy'].dropna()
    clean_category_ent = category_entropy_df['shannon_entropy'].dropna()
    if len(clean_category_ent) > 0 and len(clean_concept_ent) > 0:
        u_stat, p_val = stats.mannwhitneyu(clean_category_ent, clean_concept_ent, alternative='less')
        auc = 1 - u_stat / (len(clean_category_ent) * len(clean_concept_ent))
        log.info(f"  [Stats] Mann-Whitney U Test (Categories < Concepts): U={u_stat:.1f}, p-value={p_val:.4e}, AUC={auc:.3f}")
        if p_val < 0.05:
            log.info("  [Stats] -> Hypothesis CONFIRMED: Categories are significantly more concentrated.")
        else:
            log.info("  [Stats] -> Hypothesis REJECTED: Difference is not statistically significant.")
    else:
        log.warning("  [Stats] Insufficient valid entropy data to perform Mann-Whitney U Test.")


def plot_shannon_entropies(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, out_dir, suffix: str = "") -> None:
    """Entropy violins, concepts against category labels."""
    plot_comparison_violin(
        {"Specific Concepts": concept_entropy_df['shannon_entropy'],
         "Broad Categories": category_entropy_df['shannon_entropy']},
        y_label="Shannon entropy (bits)",
        title="Shannon entropy of expert allocations, category labels against concepts",
        out_path=out_dir / f"category_concept_shannon_entropies{suffix}.png",
        palette=LEVEL_COLORS)


def plot_gearys_entropy_scatter(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, out_dir, suffix: str = "") -> None:
    """Shape map of every word, entropy against Geary's C."""
    category_names = set(category_entropy_df['category'])
    concepts_only = concept_entropy_df[~concept_entropy_df['concept'].isin(category_names)]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(concepts_only['shannon_entropy'], concepts_only['gearys_c'],
               color=LEVEL_COLORS["Specific Concepts"], alpha=0.55, s=35, edgecolor='black',
               linewidth=0.3, label="Specific Concepts")
    ax.scatter(category_entropy_df['shannon_entropy'], category_entropy_df['gearys_c'],
               color=LEVEL_COLORS["Broad Categories"], alpha=0.95, s=110, edgecolor='black',
               linewidth=0.8, marker='D', label="Broad Categories")
    for _, row in category_entropy_df.dropna(subset=['shannon_entropy', 'gearys_c']).iterrows():
        ax.annotate(row['category'].title(), (row['shannon_entropy'], row['gearys_c']),
                    xytext=(6, 4), textcoords='offset points', fontsize=9, fontweight='bold',
                    color=LEVEL_COLORS["Broad Categories"])
    extremes = pd.concat([
        concepts_only.nlargest(5, 'gearys_c'), concepts_only.nsmallest(5, 'gearys_c'),
        concepts_only.nlargest(5, 'shannon_entropy'), concepts_only.nsmallest(5, 'shannon_entropy'),
    ]).drop_duplicates(subset='concept')
    for _, row in extremes.iterrows():
        ax.annotate(row['concept'], (row['shannon_entropy'], row['gearys_c']),
                    xytext=(6, -8), textcoords='offset points', fontsize=8, alpha=0.8)

    x_mid = sum(ax.get_xlim()) / 2
    y_mid = sum(ax.get_ylim()) / 2
    ax.axvline(x_mid, linestyle=':', linewidth=1, color='#bbbbbb')
    ax.axhline(y_mid, linestyle=':', linewidth=1, color='#bbbbbb')
    ax.axhline(1.0, linestyle='--', linewidth=1, color='#999999')
    ax.annotate("C = 1: no depth structure", xy=(0.5, 1.0), xycoords=('axes fraction', 'data'),
                xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9, color='#777777')
    corner_props = dict(fontsize=9, style='italic', color='#888888')
    ax.text(0.02, 0.02, "concentrated & smooth", transform=ax.transAxes, ha='left', va='bottom', **corner_props)
    ax.text(0.98, 0.02, "spread & smooth", transform=ax.transAxes, ha='right', va='bottom', **corner_props)
    ax.text(0.02, 0.98, "concentrated & jagged", transform=ax.transAxes, ha='left', va='top', **corner_props)
    ax.text(0.98, 0.98, "spread & jagged", transform=ax.transAxes, ha='right', va='top', **corner_props)

    ax.set_xlabel("Shannon entropy (bits)")
    ax.set_ylabel("Geary's C (adjacent-layer autocorrelation)")
    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.07))
    set_title(ax, "Layer-profile shape map, Geary's C against Shannon entropy")
    fig.tight_layout()
    fig.savefig(out_dir / f"gearys_entropy_scatter{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_category_entropies_bar(category_entropy_df: pd.DataFrame, out_dir, suffix: str = "") -> None:
    """Entropy per category label, sorted."""
    entropy_data = category_entropy_df.dropna(subset=['shannon_entropy']).copy()
    entropy_data = entropy_data.sort_values(by='shannon_entropy', ascending=False).reset_index(drop=True)
    entropy_data['category_display'] = entropy_data['category'].str.title()
    out_path = out_dir / f"category_shannon_entropies_bar{suffix}.png"
    _plot_bar_with_leaders(
        plot_dataframe=entropy_data, x_col="shannon_entropy", y_col="category_display",
        title="Shannon entropy of expert allocations by category label", x_label="Shannon entropy (bits)",
        color=LEVEL_COLORS["Broad Categories"], orient='h', out_path=out_path, figsize=(9, 8)
    )


def entropy_expert_count_data(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame) -> pd.DataFrame:
    """Words with both entropy and expert count, tagged by abstraction level."""
    category_names = set(category_entropy_df['category']) if category_entropy_df is not None else set()
    data = concept_entropy_df.dropna(subset=['shannon_entropy', 'experts_count']).copy()
    data['group'] = np.where(data['concept'].isin(category_names), 'Broad Categories', 'Specific Concepts')
    return data


def plot_entropy_vs_expert_count(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame,
                                 corr_dir, n_total_relevant: int) -> dict:
    """Entropy against expert count over both abstraction levels."""
    data = entropy_expert_count_data(concept_entropy_df, category_entropy_df)
    save_dataframe(data[['concept', 'group', 'shannon_entropy', 'experts_count']],
                   corr_dir / "shannon_entropy_vs_expert_count.csv")
    _, summary_row = regression_row(data, 'shannon_entropy', 'experts_count',
                                    "shannon_entropy_vs_expert_count", n_total_relevant)
    n = summary_row["n_points"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for group, marker, size in (("Specific Concepts", 'o', 30), ("Broad Categories", 'D', 80)):
        group_data = data[data['group'] == group]
        ax.scatter(group_data['shannon_entropy'], group_data['experts_count'], s=size, marker=marker,
                   color=LEVEL_COLORS[group], edgecolor='white', linewidth=0.4, alpha=0.8, label=group)
    fitted = summary_row["pearson_r"] is not None
    if fitted:
        sns.regplot(data=data, x='shannon_entropy', y='experts_count', scatter=False, ax=ax, seed=0,
                    color=CORRELATION_COLORS["line"], line_kws={'linewidth': 1.8})
    set_title(ax, "Shannon entropy against expert count",
              stats_line("Pearson r", summary_row["pearson_r"], summary_row["pearson_p"], n)
              if fitted else coverage_line(n, n_total_relevant))
    ax.set_xlabel("Shannon entropy (bits)")
    ax.set_ylabel("Expert count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(corr_dir / "shannon_entropy_vs_expert_count.png", dpi=300)
    plt.close(fig)
    return summary_row


def plot_distribution_correlations(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame,
                                   concept_metadata: pd.DataFrame, corr_dir) -> pd.DataFrame:
    """Entropy against expert count. Typicality and frequency panels are in module 2, sections 2.4 and 2.5."""
    summary_rows = []
    n_total_words = len(concept_metadata)
    if concept_entropy_df is not None and not concept_entropy_df.empty:
        summary_rows.append(plot_entropy_vs_expert_count(concept_entropy_df, category_entropy_df, corr_dir, n_total_words))

    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(summary_df, corr_dir / "correlation_summary.csv")
    return summary_df


def _summarize_shape(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame) -> dict:
    """Entropy and Geary's C means plus the label-over-concept entropy AUC."""
    concept_entropy = concept_entropy_df['shannon_entropy'].dropna()
    category_entropy = category_entropy_df['shannon_entropy'].dropna()
    if concept_entropy.empty or category_entropy.empty:
        auc = np.nan
    else:
        u = stats.mannwhitneyu(category_entropy, concept_entropy, alternative='two-sided').statistic
        auc = u / (len(category_entropy) * len(concept_entropy))
    return {
        "entropy_mean_concepts": concept_entropy.mean(),
        "entropy_mean_categories": category_entropy.mean(),
        "entropy_auc_categories_over_concepts": auc,
        "gearys_c_mean_concepts": concept_entropy_df['gearys_c'].mean(),
        "gearys_c_mean_categories": category_entropy_df['gearys_c'].mean(),
    }


def run_distribution_shape(scope, concept_metadata: pd.DataFrame, descriptors: list, out_dir) -> dict:
    """How spread each word's experts are over depth, per axis variant, and entropy against expert count."""
    for suffix, _, axis_label, concept_df, category_df in descriptors:
        log.info(f"  [{scope.label} / {axis_label}] Computing and plotting Shannon entropy analysis...")
        save_descriptor_tables(concept_df, category_df, out_dir, "shannon_entropy", SHAPE_DESCRIPTORS, suffix)
        _log_entropy_test(concept_df, category_df)
        log.info("  Generating Shannon entropy violin plot...")
        plot_shannon_entropies(concept_df, category_df, out_dir, suffix)
        log.info("  Generating Geary's C vs entropy shape map...")
        plot_gearys_entropy_scatter(concept_df, category_df, out_dir, suffix)
        _log_level_comparison("Geary's C", category_df['gearys_c'], concept_df['gearys_c'])
        log.info("  Generating individual category entropy bar chart...")
        plot_category_entropies_bar(category_df, out_dir, suffix)

    concept_df, category_df = descriptors[0][3], descriptors[0][4]
    log.info(f"  [{scope.label}] Generating correlation plots...")
    summary_df = plot_distribution_correlations(concept_df, category_df, concept_metadata, out_dir)
    return scope_summary_row(scope, **_summarize_shape(concept_df, category_df), **panel_r_columns(summary_df))


# ---------------------------------------------------------------------------
# 1.3 Sublayer informativeness, whole model only
# ---------------------------------------------------------------------------

def compute_sublayer_informativeness(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame,
                                     dist_dir, n_permutations: int = 9999, seed: int = 42) -> pd.DataFrame:
    """Per (level, sublayer): category alignment, Geary's C, expert share, and block-axis distribution descriptors."""
    df = expert_allocation_df.copy()
    df["sublayer"] = df["layer_name"].astype(str).str.extract(r"^\d+\.L\.\d+\.(.+)$")[0]

    categories = concept_metadata.set_index("concept")["category"]
    level_of = concept_metadata.set_index("concept")["abstraction_level"].to_dict()
    concepts = sorted(c for c in df["concept"].unique() if pd.notna(categories.get(c)))
    concept_categories = np.array([categories[c] for c in concepts])
    rng = np.random.default_rng(seed)

    total_experts = len(df)
    rows = []
    for sublayer, sub_df in df.groupby("sublayer", sort=False):
        pair_similarity = pair_similarity_vector(sub_df, concepts)
        alignment = category_alignment_metrics(pair_similarity, concept_categories, n_permutations, rng)

        word_counts = sub_df.pivot_table(index="concept", columns="layer_idx", values="unit",
                                         aggfunc="count", fill_value=0)
        word_gearys = word_counts.apply(lambda row: gearys_c(row.values), axis=1)
        word_levels = word_counts.index.map(level_of)

        counts, probabilities = build_layer_probability_matrix(to_block_axis(sub_df))
        blocks = probabilities.columns.values
        descriptors = pd.DataFrame([layer_distribution_descriptors(probabilities.loc[word].values, blocks)
                                    for word in probabilities.index], index=probabilities.index)
        descriptors["expert_count"] = counts.sum(axis=1)
        descriptor_levels = descriptors.index.map(level_of)

        expert_levels = sub_df["concept"].map(level_of)
        for level in (1, 2):
            n_level_experts = int((expert_levels == level).sum())
            words = descriptors[descriptor_levels == level]
            rows.append({
                "abstraction_level": level,
                "sublayer": sublayer,
                "gearys_c": word_gearys[word_levels == level].mean(),
                **(alignment if level == 2 else {key: np.nan for key in alignment}),
                "share_of_all_experts_pct": 100 * n_level_experts / total_experts,
                "n_experts": n_level_experts,
                "n_words": len(words),
                "mean_entropy_bits": words["shannon_entropy"].mean(),
                "reliable_peak_pct": 100 * (words["peak_gap_pct"] >= PEAK_GAP_RELIABLE_PP).mean() if len(words) else np.nan,
                "mean_depth_top25_block": words["avg_layer_top25pct"].mean(),
                "mean_expert_count": words["expert_count"].mean(),
            })

    result = pd.DataFrame(rows)
    auc_rank = (result[result["abstraction_level"] == 2]
                .set_index("sublayer")["roc_auc"].rank(ascending=False))
    result = (result.assign(_rank=result["sublayer"].map(auc_rank))
              .sort_values(["abstraction_level", "_rank"]).drop(columns="_rank")
              .reset_index(drop=True))
    save_dataframe(result, dist_dir / "sublayer_informativeness.csv")
    plot_sublayer_informativeness(result, dist_dir)
    return result


INFORMATIVENESS_PANELS = {
    "roc_auc": "Category alignment ROC-AUC (concepts)",
    "gearys_c": "Mean Geary's C",
    "mean_entropy_bits": "Mean Shannon entropy (bits)",
    "reliable_peak_pct": f"Words with a reliable peak (%, gap >= {PEAK_GAP_RELIABLE_PP:g} pp)",
    "mean_depth_top25_block": "Mean depth, top-25% average (block)",
    "mean_expert_count": "Mean experts per word",
    "share_of_all_experts_pct": "Share of all experts (%)",
}


def plot_sublayer_informativeness(result: pd.DataFrame, dist_dir) -> None:
    """One panel per informativeness column, sublayers as rows, one bar per abstraction level."""
    sublayers = list(dict.fromkeys(result["sublayer"]))
    y = np.arange(len(sublayers))
    fig, axes = plt.subplots(1, len(INFORMATIVENESS_PANELS), sharey=True,
                             figsize=(3.4 * len(INFORMATIVENESS_PANELS), 1.2 + 0.5 * len(sublayers)))
    for axis, (column, label) in zip(axes, INFORMATIVENESS_PANELS.items()):
        for offset, level in ((-0.2, 1), (0.2, 2)):
            values = (result[result["abstraction_level"] == level].set_index("sublayer")[column]
                      .reindex(sublayers).astype(float))
            axis.barh(y + offset, values.fillna(0.0), height=0.4, color=ABSTRACTION_COLORS[level],
                      label=f"Abstraction level {level}")
        axis.set_xlabel(label, fontsize=9)
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sublayers)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    set_suptitle(fig, "Sublayer informativeness per abstraction level, sublayers in level-2 ROC-AUC order")
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, frameon=False)
    fig.savefig(dist_dir / "sublayer_informativeness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_sublayer_informativeness(scope, concept_metadata: pd.DataFrame, out_dir) -> None:
    """Sublayers ranked by category alignment and descriptor means, whole-model scope only."""
    log.info("  Computing sublayer informativeness (AUC + Mantel permutation)...")
    informativeness = compute_sublayer_informativeness(scope.expert_df, concept_metadata, out_dir)
    ranked = informativeness[(informativeness["abstraction_level"] == 2) & informativeness["roc_auc"].notna()]
    if not ranked.empty:
        top = ranked.iloc[0]
        log.info(f"  Sublayer informativeness written, top by level-2 category alignment: {top['sublayer']} "
                 f"(ROC-AUC {top['roc_auc']:.3f}, r {top['category_alignment_r']:.3f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SECTION_SUMMARY_LABELS = {
    "1.1": LAYER_DISTRIBUTION_SUMMARY_LABELS,
    "1.2": SHAPE_SUMMARY_LABELS,
}


def execute_module_1_expert_distribution(scope, concept_metadata: pd.DataFrame, module_dir,
                                         per_concept: bool = True) -> dict:
    """Run every section on one scope, returns {section: sublayer_comparison row}."""
    def out(section):
        return scope_section_dir(module_dir, scope, SECTION_DIRS[section])

    merged_meta = expert_counts_with_metadata(scope.expert_df, concept_metadata)
    descriptors = layer_descriptors_by_axis(scope, concept_metadata)
    rows = {}
    # Each section line can be commented out alone.
    rows["1.1"] = run_layer_distribution(scope, concept_metadata, merged_meta, descriptors, out("1.1"), per_concept)
    rows["1.2"] = run_distribution_shape(scope, concept_metadata, descriptors, out("1.2"))
    if scope.is_whole_model:
        run_sublayer_informativeness(scope, concept_metadata, out("1.3"))
    return rows
