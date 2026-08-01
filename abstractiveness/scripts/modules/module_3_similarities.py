import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.helpers import save_dataframe, scope_out_dir, scope_summary_row, layer_profile_matrices
from utils.plot_helpers import _plot_bar_with_leaders, build_category_color_map, fig_width_for

log = logging.getLogger(__name__)

# Headline metrics for the cross-scope sublayer_comparison table. The layer-profile pair
# carries no median: plot_sublayer_comparison_bars sizes at 4.2 inches per panel, so six
# is already a 25-inch figure.
SUMMARY_LABELS = {
    "jaccard_mean_pct": "Mean Jaccard %",
    "jaccard_median_pct": "Median Jaccard %",
    "overlap_mean_pct": "Mean overlap %",
    "overlap_median_pct": "Median overlap %",
    "layer_profile_mean_pct": "Mean layer-profile similarity %",
    "layer_profile_z_mean": "Mean layer-profile z vs null",
}

def plot_hierarchy_similarities(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, sim_dir) -> pd.DataFrame:
    """
    Calculate Jaccard, Overlap and layer-profile similarity metrics between concepts and
    categories. Generates bar charts for each metric and returns the results DataFrame.

    Jaccard and overlap ask WHICH neurons the two words share. Layer-profile similarity
    asks whether they spread their experts over the layers in the same proportions, which
    a concept and its category can do while sharing no neuron at all, and which every
    set-based metric therefore scores as zero. Its z companion is the reading to trust:
    raw layer-profile similarity is largely a readout of expert-set size plus the model's
    global density profile, and z is what is left after both are conditioned out.
    """
    # Key expert sets on (layer_idx, unit) pairs: the raw `unit` column is only the neuron
    # index *within* a layer, so identical indices from different layers would otherwise be
    # collapsed into one element, inflating intersections between unrelated experts.
    pair_keyed_df = expert_allocation_df.assign(
        layer_unit=list(zip(expert_allocation_df["layer_idx"], expert_allocation_df["unit"]))
    )
    unit_sets = pair_keyed_df.groupby("concept")["layer_unit"].apply(set).to_dict()

    # One matrix over EVERY word, then look up the (concept, category) cell per pair. The
    # item list must match the one modules 4 and 5 pass, since the null grid spans the
    # count range across it and a different list would yield different z for the same pair.
    items = list(concept_metadata["concept"].unique())
    item_row = {item: i for i, item in enumerate(items)}
    profile_jsd, profile_sim, profile_z = layer_profile_matrices(expert_allocation_df, items)

    results = []
    for _, row in concept_metadata.dropna(subset=["category"]).iterrows():
        concept, category = row["concept"], row["category"]
        u_concept = unit_sets.get(concept, set())  # Set A
        u_category = unit_sets.get(category, set()) # Set B

        if u_concept and u_category:
            intersection = len(u_concept.intersection(u_category))
            union = len(u_concept.union(u_category))
            len_a = len(u_concept)
            len_b = len(u_category)
            i, j = item_row.get(concept), item_row.get(category)
            has_profile = i is not None and j is not None

            results.append({
                "concept": concept,
                "category": category,
                "hierarchy": f"{category} -> {concept}",
                "jaccard_pct": (intersection / union) * 100,
                "overlap_pct": (intersection / min(len_a, len_b)) * 100,
                # Raw set sizes behind the percentages, for scale
                "shared_expert_units": intersection,
                "concept_expert_units": len_a,
                "category_expert_units": len_b,
                "layer_profile_similarity_pct": profile_sim[i, j] if has_profile else np.nan,
                "layer_profile_jsd_bits": profile_jsd[i, j] if has_profile else np.nan,
                "layer_profile_z": profile_z[i, j] if has_profile else np.nan,
            })

    similarity_metrics_df = pd.DataFrame(results)
    if similarity_metrics_df.empty: return similarity_metrics_df
    
    save_dataframe(similarity_metrics_df, sim_dir / "category_concept_similarity_metrics.csv")

    # Color each concept's bar by its category. The df rows follow concept_metadata's
    # (category-grouped) order, so bars form contiguous colored blocks; the shared color
    # map keeps a category's color identical here, in the heatmaps, and anywhere else.
    color_map = build_category_color_map(concept_metadata.dropna(subset=["category"])["category"])
    bar_colors = [color_map[c] for c in similarity_metrics_df["category"]]
    present = set(similarity_metrics_df["category"])
    color_legend = {cat: col for cat, col in color_map.items() if cat in present}

    # Width scales with the number of category->concept bars so they stay legible.
    width = fig_width_for(len(similarity_metrics_df), 0.34, min_w=16.0)
    bar_order = list(similarity_metrics_df["hierarchy"])

    # Generate the 3 distinct percentage plots
    metrics = [
        ("jaccard_pct", "Jaccard Similarity Index % (Global Equivalence)"),
        ("overlap_pct", "Overlap Coefficient % (Strict Subsetting)"),
        ("layer_profile_similarity_pct", "Layer-Profile Similarity % (Depth Allocation Agreement)"),
    ]

    for col, title in metrics:
        _plot_bar_with_leaders(
            plot_dataframe=similarity_metrics_df, x_col="hierarchy", y_col=col,
            title=title, x_label="Category->Concept", y_label="Percentage %",
            bar_colors=bar_colors, color_legend=color_legend, legend_title="Category",
            tick_label_colors=bar_colors,
            show_x_ticks=True, figsize=(width, 10.14), order=bar_order,
            out_path=sim_dir / f"{col.replace('_pct', '')}_hierarchy.png"
        )

    # The z companion is on a different scale (standard deviations, signed), so it gets its
    # own axis label and a reference line at 0, the value meaning "no more agreement than
    # two arbitrary words of these expert counts would show".
    # Drawn onto an axes we own, so the reference line can be added before saving. Passing
    # ax with save=True keeps the full-size axis labels (the helper drops to fontsize 11
    # when save is False) while suppressing its own savefig, which only fires when ax is None.
    fig, ax = plt.subplots(figsize=(width, 10.14))
    _plot_bar_with_leaders(
        plot_dataframe=similarity_metrics_df, x_col="hierarchy", y_col="layer_profile_z",
        title="Layer-Profile Agreement vs Count-Matched Null (z)",
        x_label="Category->Concept", y_label="z (standard deviations above the null)",
        bar_colors=bar_colors, color_legend=color_legend, legend_title="Category",
        tick_label_colors=bar_colors,
        show_x_ticks=True, order=bar_order, ax=ax, save=True
    )
    ax.axhline(0.0, color="#2f2f2f", linewidth=1.4, linestyle="--", zorder=5)
    fig.savefig(sim_dir / "layer_profile_z_hierarchy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return similarity_metrics_df

def execute_module_3_category_concept_similarities(scope, concept_metadata: pd.DataFrame, sim_dir) -> tuple[pd.DataFrame, dict]:
    """Execute Module 3: Category-Concept Similarities.
    Calculates Jaccard, Overlap and layer-profile similarity between concepts and their
    categories.

    Jaccard and overlap are set-based, reading the expert rows as an unordered set of
    (layer_idx, unit) pairs, so layer order is irrelevant to them. Layer-profile
    similarity reads the layer axis, but only as a set of bins to normalize over, so it
    is permutation-invariant too and needs no axis_variants split: unlike Geary's C or
    peak layer, nothing in it treats adjacent layer_idx values as adjacent depths. The
    whole-model scope therefore runs once, on the flat layer axis, which is also the axis
    Jaccard lives in and the one that retains sublayer identity.

    The whole-model scope is the unqualified answer to the module's research question, and
    each sublayer scope answers the narrower "where in the block do a concept and its label
    actually share neurons, or at least agree on depth".

    Returns (similarity_metrics_df, summary_row) where summary_row feeds this module's
    sublayer_comparison table.
    """
    out_dir = scope_out_dir(sim_dir, scope)
    log.info(f"  [{scope.label}] Generating hierarchy similarities (Jaccard & Overlap)...")
    similarity_metrics_df = plot_hierarchy_similarities(scope.expert_df, concept_metadata, out_dir)

    if similarity_metrics_df.empty:
        summary = scope_summary_row(scope, n_pairs=0, **{key: float("nan") for key in SUMMARY_LABELS})
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
        )
    return similarity_metrics_df, summary