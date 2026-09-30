import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon, pdist
from utils.logging_and_io import save_dataframe
from modules.shared_layer_distribution_measures import build_layer_probability_matrix
from core.analysis_scopes import axis_variants, scope_out_dir, scope_summary_row
from utils.plotting import (fig_width_for, apply_rotated_leader_labels, draw_regression, set_title, set_suptitle,
                            stats_line, LEVEL_COLORS)

log = logging.getLogger(__name__)

# Headline metrics for the cross-scope sublayer_comparison table, read off the block-axis
# variant (the one whose layer axis is a genuine depth axis, see AXIS_VARIANTS below).
SUMMARY_LABELS = {
    "mean_jensen_shannon_divergence": "Mean label vs member-average JSD",
    "mean_member_diversity_jsd": "Mean within-category member JSD",
    "jsd_vs_diversity_r": "Pearson r, divergence vs diversity",
}


def compute_dual_category_jsd(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, jsd_dir,
                              suffix: str = "") -> tuple[pd.DataFrame, list]:
    """
    Compute Jensen-Shannon Divergence between category label distributions and averaged member distributions.
    Also computes the internal diversity (average pairwise JSD) of the category members.
    Returns DataFrame with divergence metrics and layer labels for visualization.
    """
    layer_labels = expert_allocation_df['layer_name'].cat.categories.tolist()
    dist_matrix, prob_matrix = build_layer_probability_matrix(expert_allocation_df)
    
    results = []
    
    for category in concept_metadata['category'].dropna().unique():
        cat_meta = concept_metadata[concept_metadata['category'] == category]
        members = cat_meta['concept'].tolist()
        valid_members = [m for m in members if m in prob_matrix.index]
        
        if category in prob_matrix.index and valid_members:
            # P: Category Label distribution (Prototype)
            P = prob_matrix.loc[category].values
            
            # Q: Averaged Member distribution (Exemplars)
            Q = prob_matrix.loc[valid_members].mean(axis=0).values
            
            # Label vs. Centroid Divergence (JSD Math)
            js_distance = jensenshannon(P, Q, base=2)
            jsd = js_distance ** 2
            
            # Internal Category Diversity (Average Pairwise Member JSD)
            if len(valid_members) > 1:
                member_probs = prob_matrix.loc[valid_members].values
                # pdist calculates all unique pairwise combinations
                pairwise_js_dist = pdist(member_probs, metric=lambda u, v: jensenshannon(u, v, base=2))
                avg_member_jsd = np.mean(pairwise_js_dist ** 2)
            else:
                avg_member_jsd = np.nan
            
            # Additional Metrics for Visualizations
            p_ent = entropy(P, base=2)
            valid_meta = cat_meta[cat_meta['concept'].isin(valid_members)]
            avg_typ = valid_meta['human_typicality'].mean() if 'human_typicality' in valid_meta.columns else np.nan

            results.append({
                "category": category,
                "member_count": len(valid_members),
                "avg_human_typicality": avg_typ,
                "shannon_entropy_label": p_ent,
                "jensen_shannon_divergence": jsd,
                "avg_member_diversity_jsd": avg_member_jsd,
                "P_dist": P,
                "Q_dist": Q 
            })
            
    jsd_results_df = pd.DataFrame(results)
    
    if not jsd_results_df.empty:
        jsd_results_df = jsd_results_df.sort_values(by="jensen_shannon_divergence").reset_index(drop=True)
        # Save a clean CSV without the massive array columns
        if jsd_dir is not None:
            save_dataframe(jsd_results_df.drop(columns=['P_dist', 'Q_dist']), jsd_dir / f"dual_category_jsd{suffix}.csv")
    else:
        log.warning("  No categories qualified for JSD (no category-label concept has experts at this AP); skipping.")

    # Return both the dataframe and the layer labels for the X-axis of the micro plot
    return jsd_results_df, layer_labels

def plot_jsd_vs_diversity_scatter(jsd_results_df: pd.DataFrame, jsd_dir, suffix: str = "") -> None:
    """
    Plots Label-Centroid Divergence (X-axis) against Internal Category Diversity (Y-axis).
    Tests if high member dispersion forces the model to create a distinct prototype label.
    """
    # An empty results frame has no columns at all, so guard before dropna(subset=...).
    if jsd_results_df.empty:
        log.warning("  Skipping JSD vs. diversity scatter: no JSD results.")
        return
    # Drop rows missing either of the two metrics we need
    df = jsd_results_df.dropna(subset=['jensen_shannon_divergence', 'avg_member_diversity_jsd']).copy()
    if df.empty:
        log.warning("  Skipping JSD vs. diversity scatter: no rows with both metrics present.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    fitted = len(df) > 2
    detail = None
    if fitted:
        r, p = stats.pearsonr(df['jensen_shannon_divergence'], df['avg_member_diversity_jsd'])
        detail = stats_line("Pearson r", r, p, len(df))
    draw_regression(ax, df, "jensen_shannon_divergence", "avg_member_diversity_jsd", fit=fitted, size=60)
    for i in range(df.shape[0]):
        ax.annotate(df['category'].iloc[i].capitalize(),
                    (df['jensen_shannon_divergence'].iloc[i], df['avg_member_diversity_jsd'].iloc[i]),
                    xytext=(6, 3), textcoords='offset points', ha='left', fontsize=8, alpha=0.8)

    set_title(ax, "Label-to-members divergence against internal category diversity", detail)
    ax.set_xlabel("Jensen-Shannon divergence, label against member average (bits)\n"
                  "high overlap (exemplar) to low overlap (prototype)")
    ax.set_ylabel("Average pairwise member divergence (JSD)\nhigh cohesion to high internal diversity")
    fig.tight_layout()
    fig.savefig(jsd_dir / f"scatter_jsd_vs_member_diversity{suffix}.png", dpi=300)
    plt.close(fig)

def plot_jsd_micro_distributions(jsd_results_df: pd.DataFrame, layer_labels: list, jsd_dir, suffix: str = "") -> None:
    """
    Visualization 2: Overlaid Area Charts for the Lowest and Highest JSD categories.
    For each of those two categories, plots P (the category-label prototype distribution)
    against Q (the average-of-members exemplar distribution) across layers, so the layer-wise
    allocations behind the smallest and largest JSD values can be inspected directly.
    """
    # Need at least two categories with a valid JSD to contrast extremes; guard the empty
    # frame (no columns) before dropna(subset=...) and NaN JSDs before idxmin/idxmax.
    if jsd_results_df.empty: return
    valid_jsd_df = jsd_results_df.dropna(subset=['jensen_shannon_divergence'])
    if len(valid_jsd_df) < 2:
        log.warning("  Skipping JSD micro distributions: fewer than 2 categories with a valid JSD.")
        return

    # Find the extremes
    min_cat = valid_jsd_df.loc[valid_jsd_df['jensen_shannon_divergence'].idxmin()]
    max_cat = valid_jsd_df.loc[valid_jsd_df['jensen_shannon_divergence'].idxmax()]

    # Width scales with the layer count (48 for GPT-2, 196 for Qwen3) so the per-layer
    # x-axis stays legible, matching module 1's layer plots.
    fig_w = fig_width_for(len(layer_labels), 0.28, min_w=16.0)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 10), sharex=True)
    x = np.arange(len(layer_labels))
    
    targets = [
        (axes[0], min_cat, "lowest divergence"),
        (axes[1], max_cat, "highest divergence")
    ]
    
    for ax, data, subtitle in targets:
        # Trace P (Prototype / Label)
        label_color, member_color = LEVEL_COLORS["Broad Categories"], LEVEL_COLORS["Specific Concepts"]
        ax.fill_between(x, data['P_dist'], color=label_color, alpha=0.35, label="Prototype (category label)")
        ax.plot(x, data['P_dist'], color=label_color, linewidth=2)
        ax.fill_between(x, data['Q_dist'], color=member_color, alpha=0.35, label="Exemplars (member average)")
        ax.plot(x, data['Q_dist'], color=member_color, linewidth=2)

        set_title(ax, f"{data['category'].capitalize()}, {subtitle}",
                  f"Jensen-Shannon divergence = {data['jensen_shannon_divergence']:.4f}")
        ax.set_ylabel("Probability density", fontsize=12)
        ax.legend(loc="upper right")
        ax.set_xlim(0, len(layer_labels) - 1)

    # Format the shared X-axis with the shared anchored leader-label styling.
    apply_rotated_leader_labels(axes[1], layer_labels, axis='x', fontsize=9)
    axes[1].set_xlabel("Model layer", fontsize=12)
    set_suptitle(fig, "Layer allocations behind the lowest and highest Jensen-Shannon divergence", y=1.02)
    plt.tight_layout()
    plt.savefig(jsd_dir / f"micro_jsd_distributions{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_jsd_vs_entropy_scatter(jsd_results_df: pd.DataFrame, jsd_dir, suffix: str = "") -> None:
    """Visualization 3: Scatter plot checking if concentrated concepts diverge more."""
    # An empty results frame has no columns at all, so guard before dropna(subset=...).
    if jsd_results_df.empty:
        log.warning("  Skipping JSD vs. entropy scatter: no JSD results.")
        return
    divergence_entropy_data = jsd_results_df.dropna(subset=['jensen_shannon_divergence', 'shannon_entropy_label']).copy()
    if divergence_entropy_data.empty:
        log.warning("  Skipping JSD vs. entropy scatter: no rows with both metrics present.")
        return

    data = divergence_entropy_data
    fig, ax = plt.subplots(figsize=(8, 7))
    fitted = len(data) > 2
    detail = None
    if fitted:
        r, p = stats.pearsonr(data['jensen_shannon_divergence'], data['shannon_entropy_label'])
        detail = stats_line("Pearson r", r, p, len(data))
    draw_regression(ax, data, "jensen_shannon_divergence", "shannon_entropy_label", fit=fitted, size=60)
    for i in range(data.shape[0]):
        ax.annotate(data['category'].iloc[i].capitalize(),
                    (data['jensen_shannon_divergence'].iloc[i], data['shannon_entropy_label'].iloc[i]),
                    xytext=(6, 3), textcoords='offset points', ha='left', fontsize=8, alpha=0.8)

    set_title(ax, "Label-to-members divergence against label entropy", detail)
    ax.set_xlabel("Jensen-Shannon divergence, label against member average (bits)\n"
                  "high overlap (exemplar) to low overlap (prototype)")
    ax.set_ylabel("Shannon entropy of the category label (bits)\nconcentrated to uniform")
    fig.tight_layout()
    fig.savefig(jsd_dir / f"scatter_jsd_vs_entropy{suffix}.png", dpi=300)
    plt.close(fig)

def _summarize_jsd(jsd_results_df: pd.DataFrame) -> dict:
    """Mean divergence, mean internal diversity, and the correlation the scatter plots."""
    empty = {key: np.nan for key in SUMMARY_LABELS}
    if jsd_results_df.empty:
        return empty
    paired = jsd_results_df.dropna(subset=['jensen_shannon_divergence', 'avg_member_diversity_jsd'])
    r = (stats.pearsonr(paired['jensen_shannon_divergence'], paired['avg_member_diversity_jsd'])[0]
         if len(paired) > 2 else np.nan)
    return {"mean_jensen_shannon_divergence": jsd_results_df['jensen_shannon_divergence'].mean(),
            "mean_member_diversity_jsd": jsd_results_df['avg_member_diversity_jsd'].mean(),
            "jsd_vs_diversity_r": r}


def execute_module_3_dual_category_jsd(scope, concept_metadata: pd.DataFrame, jsd_dir,
                                       write_outputs: bool = True) -> tuple[pd.DataFrame, list, dict]:
    """Execute Module 3: Dual Category Definitions (JSD).
    Compute and visualize Jensen-Shannon Divergence between category prototypes and exemplars.

    JSD itself is permutation-invariant, so it stays well defined on either layer axis,
    but the micro-distribution plot reads its x-axis as depth and is unreadable over the
    whole model's interleaved layers, so the whole-model scope produces both the block
    and the flat variant (see axis_variants). Summary metrics come from the canonical
    (first) variant.

    Returns (jsd_results_df, layer_labels, summary_row) for the canonical variant. With
    write_outputs False only the canonical table is computed and nothing is written.
    """
    if not write_outputs:
        results, labels = compute_dual_category_jsd(axis_variants(scope)[0][1], concept_metadata, None)
        return results, labels, scope_summary_row(scope, **_summarize_jsd(results))
    out_dir = scope_out_dir(jsd_dir, scope)
    canonical_results, canonical_labels = pd.DataFrame(), []

    for i, (suffix, axis_df, axis_label) in enumerate(axis_variants(scope)):
        log.info(f"  [{scope.label} / {axis_label}] Computing Dual Category Definitions (JSD) "
                 f"and Internal Member Diversity...")
        jsd_results_df, jsd_layer_labels = compute_dual_category_jsd(axis_df, concept_metadata, out_dir, suffix)
        if i == 0:
            canonical_results, canonical_labels = jsd_results_df, jsd_layer_labels
        if jsd_results_df.empty:
            log.warning("  Skipping JSD visualisations: no qualifying categories at this AP threshold.")
            continue
        log.info("  Generating JSD visualisations...")
        plot_jsd_vs_diversity_scatter(jsd_results_df, out_dir, suffix)
        plot_jsd_micro_distributions(jsd_results_df, jsd_layer_labels, out_dir, suffix)
        plot_jsd_vs_entropy_scatter(jsd_results_df, out_dir, suffix)

    return canonical_results, canonical_labels, scope_summary_row(scope, **_summarize_jsd(canonical_results))
