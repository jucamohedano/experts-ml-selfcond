"""
Module 2: Shannon entropy analysis and peak/average layer distributions.

Pipeline order (mirrored by execute_module_2_shannon_entropy at the bottom):
  1. compute_shannon_entropy         -- per-word layer-distribution descriptors -> 2 CSVs
  2. plot_peak_average_distributions -- peak histogram (hatched where the peak is an
                                        ambiguous near-tie) + average-layer KDE overview
  3. plot_shannon_entropies          -- entropy violin, categories vs concepts
  4. plot_gearys_entropy_scatter     -- profile-shape map: Geary's C vs entropy per word
  5. plot_avg_layer_violin           -- trimmed average layer, categories vs concepts
  6. plot_peak_gap_ecdf              -- ECDF of the peak dominance gap per level
  7. stats logging                   -- Mann-Whitney + AUC for gap, avg layer, Geary's C
  8. plot_category_entropies_bar     -- per-category entropy bar chart
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from scipy.stats import entropy
from utils.helpers import save_dataframe, build_layer_probability_matrix, gearys_c
from utils.plot_helpers import _plot_bar_with_leaders, apply_rotated_leader_labels, fig_width_for, plot_comparison_violin

log = logging.getLogger(__name__)

LEVEL_PALETTE = {"Specific Concepts": "#D96A5B", "Broad Categories": "#4B5A6A"}
# Below this peak-dominance gap (percentage points) a word's peak layer is treated as an
# ambiguous near-tie: hatched in the peak histogram, marked in the ECDF.
PEAK_GAP_RELIABLE_PP = 1.0


# ---------------------------------------------------------------------------
# 1. Descriptors and their computation
# ---------------------------------------------------------------------------

def layer_distribution_descriptors(probs, layer_indices, top_frac: float = 0.25) -> dict:
    """
    All scalar descriptors of one word's layer distribution, in CSV column order.

    shannon_entropy   : concentration of the distribution, blind to layer order.
    gearys_c          : depth smoothness (adjacent-layer autocorrelation); order-
                        sensitive complement to the entropy. See utils.helpers.gearys_c.
    peak_layer        : layer holding the largest share.
    peak_gap_pct      : percentage-point lead of the peak layer over the second-highest
                        one; near zero means the peak is effectively a tie.
    avg_layer         : mass-weighted mean layer over the full distribution.
    avg_layer_top25pct: mass-weighted mean layer over only the ceil(top_frac * n)
                        most-loaded layers (renormalized). The full average is dragged
                        toward mid-network by the low-mass tail; trimming locates the
                        bulk of the expertise.

    Returns NaNs throughout when probs is None (e.g. a category label absent from the
    expert data).
    """
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


def compute_shannon_entropy(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, shan_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the layer-distribution descriptors for every word and save two CSVs:
    shannon_entropy_concepts.csv (one row per word with experts) and
    shannon_entropy_categories.csv (one row per category, with the descriptors of the
    category LABEL's own distribution and, suffixed `_average`, of the mean distribution
    of its member concepts; plus the Pearson correlation between those two
    distributions). Also logs a Mann-Whitney U test (with its AUC effect size) of
    whether category labels are more concentrated (lower entropy) than concepts.
    """
    dist_matrix, prob_matrix = build_layer_probability_matrix(expert_allocation_df)
    layer_indices = prob_matrix.columns.values

    # Per-word descriptors
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
    save_dataframe(concept_entropy_df, shan_dir / "shannon_entropy_concepts.csv")

    # Per-category descriptors under two definitions: the label word itself, and the
    # average distribution of the category's member concepts.
    average_column_names = {
        "shannon_entropy": "shannon_entropy_average",
        "gearys_c": "gearys_c_average",
        "peak_layer": "peak_layer_average",
        "peak_gap_pct": "peak_gap_pct_average",
        "avg_layer": "avg_layer_average",
        "avg_layer_top25pct": "avg_layer_average_top25pct",
    }
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
            **{average_column_names[k]: v for k, v in member_desc.items()},
            "pearson_correlation_distributions": dist_corr,
            "member_count": len(valid_members),
        })
    category_entropy_df = pd.DataFrame(category_results)
    save_dataframe(category_entropy_df, shan_dir / "shannon_entropy_categories.csv")

    # Statistical test: are categories more concentrated (lower entropy) than concepts?
    clean_concept_ent = concept_entropy_df['shannon_entropy'].dropna()
    clean_category_ent = category_entropy_df['shannon_entropy'].dropna()
    if len(clean_category_ent) > 0 and len(clean_concept_ent) > 0:
        u_stat, p_val = stats.mannwhitneyu(clean_category_ent, clean_concept_ent, alternative='less')
        # Effect size of the same test: AUC = P(random category entropy < random concept
        # entropy) = 1 - U/(n1*n2). 0.5 = no effect, 1.0 = every category below every concept.
        auc = 1 - u_stat / (len(clean_category_ent) * len(clean_concept_ent))
        log.info(f"  [Stats] Mann-Whitney U Test (Categories < Concepts): U={u_stat:.1f}, p-value={p_val:.4e}, AUC={auc:.3f}")
        if p_val < 0.05:
            log.info("  [Stats] -> Hypothesis CONFIRMED: Categories are significantly more concentrated.")
        else:
            log.info("  [Stats] -> Hypothesis REJECTED: Difference is not statistically significant.")
    else:
        log.warning("  [Stats] Insufficient valid entropy data to perform Mann-Whitney U Test.")

    return concept_entropy_df, category_entropy_df


def _log_level_comparison(name: str, category_values: pd.Series, concept_values: pd.Series) -> None:
    """Mann-Whitney U (two-sided) + AUC effect size for a labels-vs-concepts metric."""
    k, c = category_values.dropna(), concept_values.dropna()
    if len(k) == 0 or len(c) == 0:
        return
    u_stat, p_val = stats.mannwhitneyu(k, c, alternative='two-sided')
    auc = u_stat / (len(k) * len(c))  # P(random category value > random concept value)
    log.info(f"  [Stats] {name} (labels vs concepts): U={u_stat:.1f}, p={p_val:.4e}, AUC(label>concept)={auc:.3f}")


# ---------------------------------------------------------------------------
# 2. Plots
# ---------------------------------------------------------------------------

def plot_peak_average_distributions(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, shan_dir) -> None:
    """
    Generate combined visualization of peak expert layers (bar plot) and average layer distributions (KDE curves).
    Compares three groups: specific concepts, broad categories, and averaged category members.
    Solid curves show the full-distribution average layer; dashed curves the trimmed
    (top 25% most-loaded layers) average, freed from the low-mass tail.
    """
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
    # Convert absolute layer_idx values to axis positions (0..n-1). Under a sublayer
    # filter the retained indices are non-contiguous (e.g. 3, 7, 11, ...), so positions
    # come from the label order, not from the index value itself; np.interp places the
    # (fractional) averages between the retained layers on the same positional axis.
    layer_support = [int(str(name).split('.', 1)[0]) for name in layer_labels]
    positions = np.arange(len(layer_support))
    plot_df['peak_layer_idx'] = plot_df['peak_layer'].map(dict(zip(layer_support, positions))).astype(int)
    plot_df['avg_layer_idx'] = np.interp(plot_df['avg_layer'], layer_support, positions)
    plot_df['avg_layer_top25_idx'] = np.interp(plot_df['avg_layer_top25pct'], layer_support, positions)
    plot_df['reliable_peak'] = plot_df['peak_gap_pct'] >= PEAK_GAP_RELIABLE_PP

    # Width scales with the layer count (matching module 1's layer plots).
    n_layers = len(layer_labels)
    fig, ax1 = plt.subplots(figsize=(fig_width_for(n_layers, 0.28, min_w=16.0), 6))
    sns.set_theme(style="whitegrid")
    palette = {**LEVEL_PALETTE, "Broad Categories (Avg)": "#ECA926"}
    group_order = ['Specific Concepts', 'Broad Categories', 'Broad Categories (Avg)']

    # Grouped bars, each split into a solid segment (dominant peak) and a hatched one
    # stacked on top (ambiguous peak: gap < PEAK_GAP_RELIABLE_PP), so the histogram
    # shows how much of the peak-location mass is actually trustworthy.
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
    ax1.legend(handles=legend_handles, title="Concept Grouping", loc='upper left', fontsize=10)

    ax1.set_title("Distribution of Peak Expert Layers vs. Average Layer Allocation", fontsize=14, pad=15)
    ax1.set_xlabel("Model Layer", fontsize=11)
    ax1.set_ylabel("% of Group Peaking in this Layer (Bars)", fontsize=11)
    apply_rotated_leader_labels(ax1, layer_labels, fontsize=9)
    ax1.set_xlim(-0.5, n_layers - 0.5)

    ax2 = ax1.twinx()
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
    ax2.legend(handles=style_handles, loc='upper center', fontsize=10, framealpha=0.9)
    ax2.set_xlim(-0.5, len(layer_labels) - 0.5)
    ax2.set_ylabel("Density of Average Layer (Curves)", fontsize=11)
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(shan_dir / "peak_average_layers.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_shannon_entropies(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, shan_dir) -> None:
    """Violin comparison of Shannon entropy between specific concepts and broad category labels."""
    plot_comparison_violin(
        {"Specific Concepts": concept_entropy_df['shannon_entropy'],
         "Broad Categories": category_entropy_df['shannon_entropy']},
        y_label="Shannon Entropy (Bits)",
        title="Shannon Entropy of Expert Allocations: Categories vs. Concepts",
        out_path=shan_dir / "category_concept_shannon_entropies.png",
        palette=LEVEL_PALETTE)


def plot_gearys_entropy_scatter(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, shan_dir) -> None:
    """
    Profile-shape map: every word plotted by Shannon entropy (concentration, order-blind)
    vs Geary's C (depth smoothness, order-sensitive). The quadrants classify layer-profile
    shapes; the dashed line at C=1 marks 'no spatial structure'. Category labels are
    enlarged and named; the most jagged and smoothest concepts are annotated.
    """
    # Category-label words also appear in the concepts CSV (they are words with experts
    # too); exclude them from the concept dots and extreme annotations, otherwise every
    # label is plotted twice and can be captioned twice.
    category_names = set(category_entropy_df['category'])
    concepts_only = concept_entropy_df[~concept_entropy_df['concept'].isin(category_names)]

    plt.figure(figsize=(11, 8))
    ax = plt.gca()
    ax.scatter(concepts_only['shannon_entropy'], concepts_only['gearys_c'],
               color=LEVEL_PALETTE["Specific Concepts"], alpha=0.55, s=35, edgecolor='black',
               linewidth=0.3, label="Specific Concepts")
    ax.scatter(category_entropy_df['shannon_entropy'], category_entropy_df['gearys_c'],
               color=LEVEL_PALETTE["Broad Categories"], alpha=0.95, s=110, edgecolor='black',
               linewidth=0.8, marker='D', label="Broad Categories")
    for _, row in category_entropy_df.dropna(subset=['shannon_entropy', 'gearys_c']).iterrows():
        ax.annotate(row['category'].title(), (row['shannon_entropy'], row['gearys_c']),
                    xytext=(6, 4), textcoords='offset points', fontsize=9, fontweight='bold',
                    color=LEVEL_PALETTE["Broad Categories"])
    # Name every concept in the extreme ranges of BOTH axes.
    extremes = pd.concat([
        concepts_only.nlargest(5, 'gearys_c'), concepts_only.nsmallest(5, 'gearys_c'),
        concepts_only.nlargest(5, 'shannon_entropy'), concepts_only.nsmallest(5, 'shannon_entropy'),
    ]).drop_duplicates(subset='concept')
    for _, row in extremes.iterrows():
        ax.annotate(row['concept'], (row['shannon_entropy'], row['gearys_c']),
                    xytext=(6, -8), textcoords='offset points', fontsize=8, alpha=0.8)

    # Two dotted lines through the middle of each axis divide the map into equal
    # quarters; the dashed C = 1 line stays as the labeled 'no depth structure' reference.
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

    ax.set_xlabel("Shannon Entropy (Bits)", fontsize=12)
    ax.set_ylabel("Geary's C (adjacent-layer autocorrelation)", fontsize=12)
    # Legend tucked into the bottom-left corner, just above the quadrant hint there.
    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.07), fontsize=10)
    plt.title("Layer-Profile Shape Map: Geary's C vs. Shannon Entropy", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(shan_dir / "gearys_entropy_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_avg_layer_violin(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, shan_dir) -> None:
    """
    Violin comparison of the average expert layer between abstraction levels, in the
    same style as the entropy violin -- the direct display of whether one level's
    experts sit earlier or later in the network. Uses the trimmed (top 25% most-loaded
    layers) average because the full-distribution average is dragged toward mid-network
    by the low-mass tail; the full version remains in the CSVs and in the peak/average
    KDE plot for reference.
    """
    plot_comparison_violin(
        {"Specific Concepts": concept_entropy_df['avg_layer_top25pct'],
         "Broad Categories": category_entropy_df['avg_layer_top25pct']},
        y_label="Average layer (top 25% most-loaded layers)",
        title="Average Expert Layer: Categories vs. Concepts",
        out_path=shan_dir / "category_concept_avg_layer_violin.png",
        palette=LEVEL_PALETTE)


def plot_peak_gap_ecdf(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, shan_dir) -> None:
    """
    ECDF of the peak dominance gap per abstraction level: at any threshold x, the curve
    height is the fraction of words whose peak layer leads the runner-up by less than x
    percentage points. The dashed line marks the ambiguity threshold used by the peak
    histogram, annotated with the share of each level below it.
    """
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    fractions = {}
    for label, values in (("Specific Concepts", concept_entropy_df['peak_gap_pct'].dropna()),
                          ("Broad Categories", category_entropy_df['peak_gap_pct'].dropna())):
        sns.ecdfplot(values, ax=ax, color=LEVEL_PALETTE[label], linewidth=2, label=label)
        fractions[label] = (values < PEAK_GAP_RELIABLE_PP).mean()
    ax.axvline(PEAK_GAP_RELIABLE_PP, linestyle='--', linewidth=1, color='#999999')
    ax.annotate(
        f"Gap < {PEAK_GAP_RELIABLE_PP:g} pp (ambiguous peak):\n"
        + "\n".join(f"{label}: {frac:.0%}" for label, frac in fractions.items()),
        xy=(PEAK_GAP_RELIABLE_PP, 0.5), xytext=(12, 0), textcoords='offset points',
        fontsize=10, va='center')
    ax.set_xlabel("Peak dominance gap (percentage points)", fontsize=12)
    ax.set_ylabel("Fraction of words with gap below x", fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    plt.title("Peak-Layer Dominance Gap: ECDF by Abstraction Level", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(shan_dir / "peak_gap_ecdf.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_category_entropies_bar(category_entropy_df: pd.DataFrame, shan_dir) -> None:
    """
    Generate bar chart of Shannon entropy per category label, sorted by entropy value.
    Shows which categories have most localized (low entropy) vs distributed (high entropy) experts.
    """
    entropy_data = category_entropy_df.dropna(subset=['shannon_entropy']).copy()
    entropy_data = entropy_data.sort_values(by='shannon_entropy', ascending=False).reset_index(drop=True)
    entropy_data['category_display'] = entropy_data['category'].str.title()
    out_path = shan_dir / "category_shannon_entropies_bar.png"
    _plot_bar_with_leaders(
        plot_dataframe=entropy_data, x_col="shannon_entropy", y_col="category_display",
        title="Shannon Entropy of Expert Allocations by Category", x_label="Shannon Entropy (Bits)",
        color="#4B5A6A", orient='h', out_path=out_path, figsize=(12, 14)
    )


# ---------------------------------------------------------------------------
# 3. Module entry point
# ---------------------------------------------------------------------------

def execute_module_2_shannon_entropy(formatted_expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, shan_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Module 2: Shannon Entropy Analysis and Peak/Average Layer Distributions.
    Computes the per-word descriptor CSVs, generates the comparative visualizations
    (in the order listed in the module docstring), and returns the two descriptor
    DataFrames for downstream modules.
    """
    log.info("  Computing and plotting Shannon entropy analysis...")
    concept_entropy_df, category_entropy_df = compute_shannon_entropy(formatted_expert_allocation_df, concept_metadata, shan_dir)
    log.info("  Generating peak vs. average layer distributions plot...")
    plot_peak_average_distributions(formatted_expert_allocation_df, concept_metadata, concept_entropy_df, category_entropy_df, shan_dir)
    log.info("  Generating Shannon entropy violin plot...")
    plot_shannon_entropies(concept_entropy_df, category_entropy_df, shan_dir)
    log.info("  Generating Geary's C vs entropy shape map...")
    plot_gearys_entropy_scatter(concept_entropy_df, category_entropy_df, shan_dir)
    log.info("  Generating average-layer violin plot...")
    plot_avg_layer_violin(concept_entropy_df, category_entropy_df, shan_dir)
    log.info("  Generating peak dominance gap ECDF...")
    plot_peak_gap_ecdf(concept_entropy_df, category_entropy_df, shan_dir)
    # Labels-vs-concepts statistics for the metrics displayed above.
    _log_level_comparison("Peak dominance gap", category_entropy_df['peak_gap_pct'], concept_entropy_df['peak_gap_pct'])
    _log_level_comparison("Avg layer (full)", category_entropy_df['avg_layer'], concept_entropy_df['avg_layer'])
    _log_level_comparison("Avg layer (top 25% layers)", category_entropy_df['avg_layer_top25pct'], concept_entropy_df['avg_layer_top25pct'])
    _log_level_comparison("Geary's C", category_entropy_df['gearys_c'], concept_entropy_df['gearys_c'])
    log.info("  Generating individual category entropy bar chart...")
    plot_category_entropies_bar(category_entropy_df, shan_dir)
    return concept_entropy_df, category_entropy_df
