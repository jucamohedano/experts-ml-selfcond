"""
Module 1: Expert layer distributions.

Pipeline order (mirrored by execute_module_1_layer_expert_distribution at the bottom):
  1. save_expert_counts_metadata        -- per-concept expert counts + metadata CSV
  2. compute_layer_distributions        -- count/percentage matrices, cumulative column
  3. plot_global_distribution           -- global per-layer bar chart
  4. compute_sublayer_informativeness   -- (level x sublayer) ranking: category-
                                           alignment AUC + Mantel p, Geary's C, shares
  5. plot_cumulative_layer_distribution(+_sorted) -- cumulative-mass plots for the
                                           entire model and the most informative sublayer
  6. plot_per_sublayer_distributions    -- one distribution plot + CSV per sublayer type
  7. plot_per_concept_distributions     -- one distribution plot + CSV per concept
"""
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from utils.helpers import (save_dataframe, gearys_c, filter_expert_data_to_sublayer,
                           pair_similarity_vector)
from utils.plot_helpers import _plot_bar_with_leaders, apply_rotated_leader_labels, fig_width_for

log = logging.getLogger(__name__)

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

def plot_global_distribution(global_layer_distribution_df: pd.DataFrame, dist_dir) -> None:
    """Save the global layer distribution CSV. (The plain bar chart was retired: the
    cumulative plots carry the same bars plus the cumulative-mass curves.)"""
    save_dataframe(global_layer_distribution_df, dist_dir / "mean_expert_layer_distribution.csv")

def plot_cumulative_layer_distribution(global_layer_dist: pd.DataFrame, dist_dir,
                                       filename_prefix: str = "", title_suffix: str = "") -> None:
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
    plt.savefig(dist_dir / f"{filename_prefix}mean_expert_layer_distribution_cumulative.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_layer_distribution_sorted(global_layer_dist: pd.DataFrame, dist_dir,
                                              filename_prefix: str = "", title_suffix: str = "") -> None:
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
    plt.savefig(dist_dir / f"{filename_prefix}mean_expert_layer_distribution_cumulative_sorted.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Sublayer informativeness (category alignment + Geary's C)
# ---------------------------------------------------------------------------

def _category_alignment_metrics(pair_similarity: np.ndarray, concept_categories: np.ndarray,
                                n_permutations: int, rng: np.random.Generator) -> dict:
    """
    How well expert-set similarity tracks category membership, over all concept pairs.

    roc_auc (primary): P(random same-category pair is more similar than a random
    different-category pair) -- rank-based (equals Mann-Whitney U / (n1*n2)), so immune
    to the same/different pair imbalance and to the skewed shape of Jaccard values.
    Computed via the rank-sum identity with average ranks for ties (identical to
    sklearn's roc_auc_score); ranking once and re-summing per shuffle is what keeps
    9999 permutations cheap.
    category_alignment_r (companion): Pearson correlation between the same-category
    indicator (1 if a pair shares a category, 0 otherwise) and pair_similarity itself,
    r = corr(same, pair_similarity) in [-1, 1], called point-biserial because one of
    the two inputs is binary. Positive r means same-category pairs run more similar on
    average, and r^2 is the fraction of pair_similarity's variance explained by category
    membership alone (the RSA-style linear reading of categorical-model alignment).
    Unlike the rank-based AUC, r assumes a roughly linear relationship, so a skewed
    similarity distribution or a few extreme pairs can pull r away from what the AUC's
    ranking shows, which is why the two are read together rather than interchangeably.
    mantel_p: permutation p-value for the AUC. Pairs are not independent observations
    (each concept appears in n-1 pairs), so category labels are shuffled across
    CONCEPTS (never across pairs), the indicator rebuilt, and the AUC recomputed --
    preserving the similarity geometry and category sizes while breaking only the
    concept-to-category correspondence under test. Smallest reportable value is
    1/(n_permutations + 1).

    Degenerate sublayers whose pair similarities are all identical (e.g. at strict AP
    thresholds no concept pair shares any expert) short-circuit to AUC=0.5, r=NaN,
    p=1.0: with zero similarity variation there is nothing to align, and pearsonr on a
    constant vector is undefined (would only emit a ConstantInputWarning).
    """
    n = len(concept_categories)
    iu = np.triu_indices(n, k=1)
    same = (concept_categories[:, None] == concept_categories[None, :])[iu]

    if pair_similarity.min() == pair_similarity.max():
        return {"roc_auc": 0.5, "category_alignment_r": np.nan, "mantel_p": 1.0}

    # Category sizes (hence the number of same-category pairs n1) are invariant under
    # label permutation, so ranks and the U-statistic constants are precomputed once.
    ranks = stats.rankdata(pair_similarity)
    n1 = int(same.sum())
    n0 = len(same) - n1
    rank_offset = n1 * (n1 + 1) / 2

    def rank_auc(same_mask):
        return (ranks[same_mask].sum() - rank_offset) / (n1 * n0)

    auc = rank_auc(same)
    r, _ = stats.pearsonr(same.astype(float), pair_similarity)

    n_at_least = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(concept_categories)
        same_perm = (permuted[:, None] == permuted[None, :])[iu]
        if rank_auc(same_perm) >= auc:
            n_at_least += 1
    mantel_p = (1 + n_at_least) / (1 + n_permutations)

    return {"roc_auc": auc, "category_alignment_r": r, "mantel_p": mantel_p}


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
        alignment = _category_alignment_metrics(pair_similarity, concept_categories, n_permutations, rng)

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

def plot_per_sublayer_distributions(expert_allocation_df: pd.DataFrame, dist_dir, chosen_sublayer: str = None) -> None:
    """
    Generate one mean expert distribution CSV (+ plot) per sublayer type. The chosen
    sublayer's CSV goes into the main dist_dir (its plot is covered by the cumulative
    variants there); every other sublayer's CSV and plot go into the
    sublayers_distribution/ subfolder so the main folder stays focused on the analysis
    sublayer. With no chosen sublayer, everything lands in the subfolder.
    The sublayer type is parsed from layer_name ("{idx}.L.{block}.{sublayer}"), so this
    works for any architecture. Each concept's percentages are renormalized WITHIN the
    sublayer's layers (each row sums to 100% across that projection type).
    """
    others_dir = dist_dir / "sublayers_distribution"
    others_dir.mkdir(parents=True, exist_ok=True)
    sublayer_of = expert_allocation_df['layer_name'].astype(str).str.extract(r'^\d+\.L\.\d+\.(.+)$')[0]

    # .unique() follows row order; the df is sorted by layer_idx, so sublayers come out in
    # forward-pass order (e.g. q_proj, k_proj, ..., down_proj).
    for sublayer in sublayer_of.dropna().unique():
        sub_df = expert_allocation_df[sublayer_of == sublayer].copy()
        # Keep only this sublayer's layers as categories so the x-axis has one bar per block
        # instead of every model layer.
        sub_df['layer_name'] = sub_df['layer_name'].cat.remove_unused_categories()

        _, _, sublayer_global_dist, _ = compute_layer_distributions(sub_df)
        is_chosen = sublayer == chosen_sublayer
        target_dir = dist_dir if is_chosen else others_dir
        save_dataframe(sublayer_global_dist, target_dir / f"{sublayer}_mean_expert_layer_distribution.csv")
        if is_chosen:
            continue  # the chosen sublayer's plots are the cumulative ones in dist_dir

        n_layers = sublayer_global_dist['layer_name'].nunique()
        _plot_bar_with_leaders(
            plot_dataframe=sublayer_global_dist, x_col="layer_name", y_col="mean_expert_allocation_pct",
            title=f"Mean Expert Distribution across '{sublayer}' layers",
            x_label="Model Layer", y_label="Average % of Experts (within sublayer type)",
            hue="abstraction_level", palette=ABSTRACTION_COLORS,
            out_path=others_dir / f"{sublayer}_mean_expert_layer_distribution.png",
            figsize=(fig_width_for(n_layers, 0.28, min_w=16.0), 10.4), linewidth=0.5, show_x_ticks=True
        )

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

def execute_module_1_layer_expert_distribution(formatted_expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame,
                                               dist_dir, sublayer_filter: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Module 1: Analyze expert layer distributions across concepts and abstraction levels.

    Module 1 is the only module that sees the FULL expert data: the whole-model
    distribution CSV + cumulative plots and the sublayer-informativeness ranking are
    computed on everything (the ranking is what justifies the sublayer choice). When
    sublayer_filter is set, everything meant for downstream analysis is restricted to
    that sublayer: the expert-counts metadata, the per-concept plots, the focus
    cumulative plots, and the returned expert DataFrame. Non-chosen sublayers'
    distribution files land in sublayers_distribution/.

    Returns (merged_meta, analysis_expert_df): the expert-count metadata table and the
    (possibly sublayer-filtered) expert DataFrame that modules 2+ should consume.
    """
    analysis_df = filter_expert_data_to_sublayer(formatted_expert_allocation_df, sublayer_filter)
    if sublayer_filter:
        log.info(f"  Sublayer filter active: '{sublayer_filter}' "
                 f"({len(analysis_df)} of {len(formatted_expert_allocation_df)} experts retained for modules 2+).")

    log.info("  Generating combined metadata counts...")
    merged_meta = save_expert_counts_metadata(analysis_df, concept_metadata, dist_dir)

    log.info("  Computing layer distribution matrices...")
    _, _, global_dist, _ = compute_layer_distributions(formatted_expert_allocation_df)
    plot_global_distribution(global_dist, dist_dir)

    log.info("  Computing sublayer informativeness (AUC + Mantel permutation)...")
    informativeness = compute_sublayer_informativeness(formatted_expert_allocation_df, concept_metadata, dist_dir)
    log.info("  Sublayer ranking by category alignment:\n" + informativeness.to_string(index=False))

    log.info("  Generating cumulative distribution plots (entire distribution + focus sublayer)...")
    plot_cumulative_layer_distribution(global_dist, dist_dir)
    plot_cumulative_layer_distribution_sorted(global_dist, dist_dir)
    # Focus sublayer: the configured one, falling back to the level-2 AUC winner.
    if sublayer_filter:
        focus_sublayer, focus_df = sublayer_filter, analysis_df
    else:
        level2_ranking = informativeness[informativeness["abstraction_level"] == 2]
        focus_sublayer = level2_ranking.sort_values("roc_auc", ascending=False).iloc[0]["sublayer"]
        focus_df = filter_expert_data_to_sublayer(formatted_expert_allocation_df, focus_sublayer)
    _, _, focus_global_dist, _ = compute_layer_distributions(focus_df)
    plot_cumulative_layer_distribution(focus_global_dist, dist_dir,
                                       filename_prefix=f"{focus_sublayer}_", title_suffix=f" — '{focus_sublayer}' layers only")
    plot_cumulative_layer_distribution_sorted(focus_global_dist, dist_dir,
                                              filename_prefix=f"{focus_sublayer}_", title_suffix=f" — '{focus_sublayer}' layers only")

    log.info("  Generating per-sublayer-type expert layer distribution files...")
    plot_per_sublayer_distributions(formatted_expert_allocation_df, dist_dir, chosen_sublayer=sublayer_filter)

    log.info("  Generating per-concept expert layer distribution plots...")
    concept_matrix, concept_counts_matrix, _, expert_counts = compute_layer_distributions(analysis_df)
    plot_per_concept_distributions(concept_matrix, concept_counts_matrix, expert_counts, dist_dir)

    return merged_meta, analysis_df