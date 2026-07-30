"""
Module 1: Expert layer distributions.

Runs once per analysis scope (whole model, then one scope per sublayer).

Pipeline order (mirrored by execute_module_1_layer_expert_distribution at the bottom):
  1. save_expert_counts_metadata        -- per-concept expert counts + metadata CSV
  2. compute_layer_distributions        -- count/percentage matrices, cumulative column
  3. plot_global_distribution           -- layer distribution CSV for the scope
  4. plot_cumulative_layer_distribution(+_sorted) -- cumulative-mass plots, once per
                                           layer-axis variant (block and flat)
  5. compute_sublayer_informativeness   -- (level x sublayer) ranking: category-
                                           alignment AUC + Mantel p, Geary's C, shares.
                                           Whole-model scope only, it IS the cross-
                                           sublayer table
  6. plot_per_concept_distributions     -- one distribution plot + CSV per concept,
                                           analysis sublayer only (see the entry point)
"""
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from utils.helpers import (save_dataframe, gearys_c, pair_similarity_vector,
                           category_alignment_metrics, axis_variants, scope_out_dir,
                           scope_summary_row)
from utils.plot_helpers import _plot_bar_with_leaders, apply_rotated_leader_labels, fig_width_for

log = logging.getLogger(__name__)

# Headline metrics for the cross-scope sublayer_comparison table. The deeper per-sublayer
# ranking lives in sublayer_informativeness.csv, which this table does not duplicate.
SUMMARY_LABELS = {
    "n_experts": "Expert rows in scope",
    "mean_expert_count_concepts": "Mean experts per concept",
    "mean_expert_count_categories": "Mean experts per category label",
}

# Define a high-contrast color palette for the different abstraction levels
ABSTRACTION_COLORS = {1: "#003f5c", 2: "#ffa600"}


# ---------------------------------------------------------------------------
# 1. Metadata and layer-distribution computation
# ---------------------------------------------------------------------------

def save_expert_counts_metadata(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, dist_dir) -> pd.DataFrame:
    """Save the metadata combined with expert counts into the distribution folder.
    Merges expert counts per concept with concept metadata and saves to CSV.
    Adds log-frequency column if frequency data is available.
    """
    counts = expert_allocation_df.groupby("concept").size().reset_index(name="expert_count")
    merged = counts.merge(concept_metadata, on="concept")
    if 'frequency' in merged.columns:
        merged = merged[merged['frequency'] > 0].copy()
        merged['log_frequency'] = np.log10(merged['frequency'])
    desired_order = [
        "concept", 
        "category", 
        "abstraction_level", 
        "frequency", 
        "log_frequency",
        "human_typicality",
        "expert_count"
    ]
    final_order = [col for col in desired_order if col in merged.columns]
    merged = merged[final_order]
    # Group rows by abstraction level (all level 1 first, then level 2) for readability.
    if 'abstraction_level' in merged.columns:
        merged = merged.sort_values(['abstraction_level', 'concept']).reset_index(drop=True)
    save_dataframe(merged, dist_dir / "expert_counts_with_metadata.csv")
    return merged


def compute_layer_distributions(expert_allocation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Compute layer distribution statistics for expert allocations across concepts and abstraction levels.
    Calculates percentage distributions per concept per layer, aggregates by abstraction level, and totals per concept.
    The global view carries both the mean percentage and the raw expert count per
    (abstraction level, layer) so every reported percentage has its scale next to it.
    Returns: (concept-level pct matrix, concept-level count matrix, global averaged
    distribution by abstraction level, concept expert counts).
    """
    # 1. Build matrices of the raw expert counts and the percentage distribution per concept
    concept_count_matrix = pd.crosstab(
        index=[expert_allocation_df['abstraction_level'], expert_allocation_df['concept']],
        columns=expert_allocation_df['layer_name'],
    )
    concept_distribution_matrix = concept_count_matrix.div(concept_count_matrix.sum(axis=1), axis=0) * 100
    # 2. Average those percentages by abstraction level for the global view
    global_layer_dist = concept_distribution_matrix.groupby(level='abstraction_level').mean().reset_index()
    # 3. Melt the wide matrix back into a long format for Seaborn plotting
    global_layer_dist = global_layer_dist.melt(
        id_vars='abstraction_level',
        var_name='layer_name',
        value_name='mean_expert_allocation_pct'
    )
    # 3b. Attach the raw expert count per (abstraction level, layer) as the scale reference
    level_counts = (concept_count_matrix.groupby(level='abstraction_level').sum()
                    .reset_index()
                    .melt(id_vars='abstraction_level', var_name='layer_name', value_name='total_expert_count'))
    global_layer_dist = global_layer_dist.merge(level_counts, on=['abstraction_level', 'layer_name'])
    # 3c. Group rows by abstraction level (all level 1 first, then level 2), layer-ordered
    # within each level, so the CSV reads as two clean blocks. The categorical dtype keeps
    # the layer sort in model order (1.L.0... before 10.L.1...) rather than alphabetical.
    global_layer_dist['layer_name'] = pd.Categorical(
        global_layer_dist['layer_name'], categories=concept_count_matrix.columns, ordered=True)
    global_layer_dist = global_layer_dist.sort_values(['abstraction_level', 'layer_name']).reset_index(drop=True)
    # 3d. Cumulative allocation within each abstraction level, in model-depth order
    # (per-concept percentages sum to 100 per level, so each block ends at 100).
    global_layer_dist['cumulative_pct'] = global_layer_dist.groupby('abstraction_level')['mean_expert_allocation_pct'].cumsum()
    # 4. Pre-calculate total expert counts for the plot titles
    expert_counts = expert_allocation_df.groupby('concept').size()

    return concept_distribution_matrix, concept_count_matrix, global_layer_dist, expert_counts

# ---------------------------------------------------------------------------
# 2. Global distribution and cumulative-mass plots
# ---------------------------------------------------------------------------

def plot_global_distribution(global_layer_distribution_df: pd.DataFrame, dist_dir, suffix: str = "") -> None:
    """Save the global layer distribution CSV. (The plain bar chart was retired: the
    cumulative plots carry the same bars plus the cumulative-mass curves.)"""
    save_dataframe(global_layer_distribution_df, dist_dir / f"mean_expert_layer_distribution{suffix}.csv")

def plot_cumulative_layer_distribution(global_layer_dist: pd.DataFrame, dist_dir,
                                       suffix: str = "", title_suffix: str = "") -> None:
    """
    Depth-ordered bars (both abstraction levels side by side, as in the main plot) with
    each level's cumulative expert mass drawn as a line on a secondary 0-100% axis --
    the same bars+overlay composition as module 2's peak/average plot. Because the
    layers stay in model order, the cumulative curves rise steeply exactly where the
    distribution peaks, so each peak's contribution to the total mass stays visible.
    """
    layer_order = list(dict.fromkeys(global_layer_dist['layer_name']))
    n = len(layer_order)
    fig, ax1 = plt.subplots(figsize=(fig_width_for(n, 0.28, min_w=16.0), 10.4))
    sns.barplot(data=global_layer_dist, x='layer_name', y='mean_expert_allocation_pct',
                hue='abstraction_level', palette=ABSTRACTION_COLORS,
                edgecolor='black', linewidth=0.5, order=layer_order, ax=ax1)
    apply_rotated_leader_labels(ax1, layer_order, fontsize=9)
    ax1.set_xlabel("Model Layer", fontsize=14)
    ax1.set_ylabel("Average % of Experts (bars)", fontsize=14)
    ax1.legend(title="abstraction_level", loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_xlim(ax1.get_xlim())
    for level, level_df in global_layer_dist.groupby('abstraction_level'):
        curve = level_df.set_index('layer_name')['cumulative_pct'].reindex(layer_order)
        # First layer at which the cumulative mass crosses 50% / 90%; named in the
        # legend label and flagged with a diamond marker on the curve.
        idx50 = int(np.argmax(curve.values >= 50))
        idx90 = int(np.argmax(curve.values >= 90))
        label = (f"Cumulative (level {level}) — 50% reached by {layer_order[idx50]}; "
                 f"90% by {layer_order[idx90]}")
        ax2.plot(range(n), curve.values, color=ABSTRACTION_COLORS[level], linewidth=2,
                 marker='o', markersize=3, alpha=0.9, label=label)
        # Dotted projections from each diamond down to the layer axis and across to the
        # cumulative (right) axis, so both coordinates can be read off directly.
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
    ax2.set_ylabel("Cumulative % of Experts (lines)", fontsize=14)
    ax2.grid(False)
    ax2.legend(loc='lower right')

    ax1.set_title(f"Mean Expert Distribution with Cumulative Mass{title_suffix}", fontsize=18, pad=15)
    plt.tight_layout()
    plt.savefig(dist_dir / f"mean_expert_layer_distribution_cumulative{suffix}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_layer_distribution_sorted(global_layer_dist: pd.DataFrame, dist_dir,
                                              suffix: str = "", title_suffix: str = "") -> None:
    """
    Pareto view: layers sorted by descending expert mass, one panel per abstraction
    level (the sort order differs between levels, so they cannot share an x-axis).
    The cumulative line then shows how much of the total mass the top-k layers hold;
    a dashed 80% reference line makes the concentration readable at a glance.
    """
    levels = sorted(global_layer_dist['abstraction_level'].unique())
    n = global_layer_dist['layer_name'].nunique()
    fig, axes = plt.subplots(len(levels), 1,
                             figsize=(fig_width_for(n, 0.28, min_w=16.0), 7.0 * len(levels)))
    axes = np.atleast_1d(axes)

    for ax, level in zip(axes, levels):
        level_df = (global_layer_dist[global_layer_dist['abstraction_level'] == level]
                    .sort_values('mean_expert_allocation_pct', ascending=False).reset_index(drop=True))
        # Cumulative recomputed in SORTED order (the csv column is in depth order).
        sorted_cumulative = level_df['mean_expert_allocation_pct'].cumsum()
        # Number of top-mass layers needed to reach 50% / 90% of the total mass; stated
        # in the panel title and flagged with diamond markers on the curve.
        k50 = int(np.argmax(sorted_cumulative.values >= 50))
        k90 = int(np.argmax(sorted_cumulative.values >= 90))

        ax.bar(range(len(level_df)), level_df['mean_expert_allocation_pct'],
               color=ABSTRACTION_COLORS[level], edgecolor='black', linewidth=0.5)
        apply_rotated_leader_labels(ax, level_df['layer_name'].astype(str).tolist(), fontsize=9)
        ax.set_ylabel("Average % of Experts (bars)", fontsize=13)
        ax.set_title(
            f"Abstraction level {level} — layers sorted by expert mass "
            f"(top {k50 + 1} layers hold 50% of the mass; top {k90 + 1} of {len(level_df)} hold 90%)",
            fontsize=15)

        ax2 = ax.twinx()
        ax2.set_xlim(ax.get_xlim())
        ax2.plot(range(len(level_df)), sorted_cumulative.values, color="#444444",
                 linewidth=2, marker='o', markersize=3)
        # Dotted projections from each diamond down to the layer axis and across to the
        # cumulative (right) axis, so both coordinates can be read off directly.
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
        ax2.set_ylabel("Cumulative % of Experts (line)", fontsize=13)
        ax2.grid(False)

    axes[-1].set_xlabel("Model Layer (descending expert mass)", fontsize=14)
    fig.suptitle(f"Cumulative Expert Mass Concentration{title_suffix}", fontsize=18, y=1.0)
    plt.tight_layout()
    plt.savefig(dist_dir / f"mean_expert_layer_distribution_cumulative_sorted{suffix}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Sublayer informativeness (category alignment + Geary's C)
# ---------------------------------------------------------------------------

def compute_sublayer_informativeness(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame,
                                     dist_dir, n_permutations: int = 9999, seed: int = 42) -> pd.DataFrame:
    """
    Rank sublayer types (e.g. mlp.gate_proj) by how much category semantics their
    expert sets carry, and describe each sublayer's depth profile per abstraction level.
    Saves sublayer_informativeness.csv into dist_dir and returns the table. The AP
    threshold is inherited from the already-filtered expert_allocation_df, so each
    AP_x folder gets its own table.

    One row per (abstraction_level, sublayer); rows grouped level 1 first, sublayers
    ordered by the level-2 roc_auc ranking within each block. Columns:
      gearys_c              : mean per-word Geary's C of the expert allocation across
                              the sublayer's layers in depth order (one layer per
                              block, so free of the high/low alternation between
                              neighboring projection types in whole-model order).
      roc_auc / category_alignment_r / mantel_p : category-alignment metrics of the
                              sublayer's expert sets. Only definable for level-2
                              concepts (labels have no same-category peers), so
                              level-1 rows carry NaN there.
      share_of_all_experts_pct / n_experts : that level's experts in this sublayer
                              (shares sum to 100 across the whole table).
    """
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
        # Category alignment of the sublayer's expert sets (level-2 concepts only).
        pair_similarity = pair_similarity_vector(sub_df, concepts)
        alignment = category_alignment_metrics(pair_similarity, concept_categories, n_permutations, rng)

        # Per-word Geary's C over this sublayer's layers in depth order (pivot columns
        # are layer_idx sorted ascending; gearys_c is scale-invariant, so raw counts
        # are equivalent to normalized shares).
        word_counts = sub_df.pivot_table(index="concept", columns="layer_idx", values="unit",
                                         aggfunc="count", fill_value=0)
        word_gearys = word_counts.apply(lambda row: gearys_c(row.values), axis=1)
        word_levels = word_counts.index.map(level_of)

        expert_levels = sub_df["concept"].map(level_of)
        for level in (1, 2):
            n_level_experts = int((expert_levels == level).sum())
            rows.append({
                "abstraction_level": level,
                "sublayer": sublayer,
                "gearys_c": word_gearys[word_levels == level].mean(),
                **(alignment if level == 2 else {key: np.nan for key in alignment}),
                "share_of_all_experts_pct": 100 * n_level_experts / total_experts,
                "n_experts": n_level_experts,
            })

    result = pd.DataFrame(rows)
    # Group by abstraction level (level 1 block first), sublayers in level-2 AUC order.
    auc_rank = (result[result["abstraction_level"] == 2]
                .set_index("sublayer")["roc_auc"].rank(ascending=False))
    result = (result.assign(_rank=result["sublayer"].map(auc_rank))
              .sort_values(["abstraction_level", "_rank"]).drop(columns="_rank")
              .reset_index(drop=True))
    save_dataframe(result, dist_dir / "sublayer_informativeness.csv")
    return result


# ---------------------------------------------------------------------------
# 4. Per-sublayer and per-concept distribution plots
# ---------------------------------------------------------------------------

def plot_per_concept_distributions(concept_distribution_matrix: pd.DataFrame, concept_count_matrix: pd.DataFrame,
                                   expert_counts: pd.Series, dist_dir) -> None:
    """Generate individual distribution CSVs and charts for each concept, organized by abstraction level.
    Each CSV carries the raw expert count per layer next to the percentage."""
    concept_dir = dist_dir / "per_concept"
    # One bar per model layer, so the canvas widens with the layer count.
    n_layers = concept_distribution_matrix.shape[1]
    per_concept_width = fig_width_for(n_layers, 0.26, min_w=14.0)

    for (abs_lvl, concept), row_data in concept_distribution_matrix.iterrows():
        specific_concept_dir = concept_dir / concept
        specific_concept_dir.mkdir(parents=True, exist_ok=True)

        # Convert the row series back into a simple dataframe for Seaborn
        concept_layer_distribution_df = row_data.reset_index(name='expert_allocation_pct')
        # Both matrices come from the same crosstab, so their layer columns align positionally.
        concept_layer_distribution_df['expert_count'] = concept_count_matrix.loc[(abs_lvl, concept)].values
        total_experts = expert_counts[concept]

        save_dataframe(concept_layer_distribution_df, specific_concept_dir / f"{concept}_data.csv")

        _plot_bar_with_leaders(
            plot_dataframe=concept_layer_distribution_df, x_col="layer_name", y_col="expert_allocation_pct",
            title=f"Concept: {concept.upper()} | Abstraction Level {abs_lvl} | Total Experts: {total_experts}",
            x_label="Model Layer", y_label="% of Experts", color=ABSTRACTION_COLORS[abs_lvl],
            out_path=specific_concept_dir / f"{concept}_distribution.png", figsize=(per_concept_width, 7.8), show_x_ticks=True
        )

# ---------------------------------------------------------------------------
# 5. Module entry point
# ---------------------------------------------------------------------------

def _summarize_expert_counts(merged_meta: pd.DataFrame) -> dict:
    """Mean expert count per word at each abstraction level, the scale every share rests on."""
    if merged_meta.empty or 'abstraction_level' not in merged_meta.columns:
        return {"mean_expert_count_categories": np.nan, "mean_expert_count_concepts": np.nan}
    by_level = merged_meta.groupby('abstraction_level')['expert_count'].mean()
    return {"mean_expert_count_categories": by_level.get(1, np.nan),
            "mean_expert_count_concepts": by_level.get(2, np.nan)}


def execute_module_1_layer_expert_distribution(scope, concept_metadata: pd.DataFrame, dist_dir,
                                               analysis_sublayer_df: pd.DataFrame = None) -> tuple[pd.DataFrame, dict]:
    """
    Execute Module 1: Analyze expert layer distributions across concepts and abstraction levels.

    Per scope this writes the expert-count metadata table and the layer-distribution CSV
    plus cumulative-mass plots. The distribution reads the layer axis as depth, so the
    whole-model scope produces both entries of axis_variants: "_by_block", where the
    sublayers of each transformer block are summed into one depth bin, and "_by_layer",
    the flat interleaved axis at full resolution.

    Two outputs stay whole-model only, because replicating them per sublayer would be
    either meaningless or ruinously slow:
      sublayer_informativeness.csv is a cross-sublayer ranking by construction, so it has
        one natural home at the module's top level.
      per_concept/ is capped at the analysis sublayer passed in as analysis_sublayer_df.
        One plot per concept per sublayer per AP threshold would be 204 x 7 x 5 figures
        for Qwen3, which is both unreadable and the single slowest step in the pipeline.

    Returns (merged_meta, summary_row): the expert-count metadata table that module 4
    consumes for this scope, and this module's sublayer_comparison row.
    """
    out_dir = scope_out_dir(dist_dir, scope)

    log.info(f"  [{scope.label}] Generating combined metadata counts...")
    merged_meta = save_expert_counts_metadata(scope.expert_df, concept_metadata, out_dir)

    for suffix, axis_df, axis_label in axis_variants(scope):
        log.info(f"  [{scope.label} / {axis_label}] Computing layer distribution matrices...")
        _, _, global_dist, _ = compute_layer_distributions(axis_df)
        plot_global_distribution(global_dist, out_dir, suffix)
        title_suffix = f" — {scope.label}, {axis_label}"
        plot_cumulative_layer_distribution(global_dist, out_dir, suffix, title_suffix)
        plot_cumulative_layer_distribution_sorted(global_dist, out_dir, suffix, title_suffix)

    if scope.is_whole_model:
        log.info("  Computing sublayer informativeness (AUC + Mantel permutation)...")
        informativeness = compute_sublayer_informativeness(scope.expert_df, concept_metadata, out_dir)
        log.info("  Sublayer ranking by category alignment:\n" + informativeness.to_string(index=False))

        if analysis_sublayer_df is not None and not analysis_sublayer_df.empty:
            log.info("  Generating per-concept expert layer distribution plots (analysis sublayer only)...")
            concept_matrix, concept_counts_matrix, _, expert_counts = compute_layer_distributions(analysis_sublayer_df)
            plot_per_concept_distributions(concept_matrix, concept_counts_matrix, expert_counts, out_dir)

    return merged_meta, scope_summary_row(scope, **_summarize_expert_counts(merged_meta))
