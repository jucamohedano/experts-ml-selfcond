import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.helpers import (save_dataframe, scope_out_dir, scope_summary_row,
                           layer_profile_metric_matrices, to_block_axis,
                           ACTIVE_PROFILE_METRICS, DEFAULT_PROFILE_METRIC,
                           SIGNED_PROFILE_METRICS, profile_metric_label,
                           profile_metric_value_label)
from utils.plot_helpers import (_plot_bar_with_leaders, build_category_color_map, fig_width_for,
                                plot_sublayer_comparison_bars)

log = logging.getLogger(__name__)

# Headline metrics for the cross-scope sublayer_comparison table. The layer-profile pair
# carries no median: plot_sublayer_comparison_bars sizes at 4.2 inches per panel, so six
# is already a 25-inch figure.
#
# The two layer-profile entries report DEFAULT_PROFILE_METRIC alone, not every active
# metric. A summary row carries one number per column, and seven metrics times two columns
# would be a fourteen-panel, sixty-inch figure that answers a different question from the
# one this table exists for, which is how the SCOPES compare. How the METRICS compare
# within one scope is its own artifact, profile_metric_comparison.csv and its plot.
SUMMARY_LABELS = {
    "jaccard_mean_pct": "Mean Jaccard %",
    "jaccard_median_pct": "Median Jaccard %",
    "overlap_mean_pct": "Mean overlap %",
    "overlap_median_pct": "Median overlap %",
    "layer_profile_mean_pct": f"Mean layer-profile similarity ({DEFAULT_PROFILE_METRIC})",
    "layer_profile_z_mean": f"Mean layer-profile z vs null ({DEFAULT_PROFILE_METRIC})",
}


def profile_value_column(metric: str) -> str:
    """Similarity column this module writes for one registered metric."""
    return f"layer_profile_{metric}"


def profile_z_column(metric: str) -> str:
    """Count-matched null column this module writes for one registered metric."""
    return f"layer_profile_{metric}_z"

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

    The layer-profile question is asked once per entry of ACTIVE_PROFILE_METRICS, since
    "do these two words allocate their experts to the same depths" has no single correct
    formalization and the metrics disagree by construction: the divergence family reads
    overlap of distributions, the correlation family reads deviation from a shared
    baseline and can go negative, and Wasserstein alone charges the DISTANCE mass has to
    move, so it is the only one for which a near miss in depth beats a far one. Each gets
    its own pair of columns and its own pair of hierarchy plots, and
    profile_metric_comparison.csv reduces them to one row per metric so the choice can be
    read rather than assumed.
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
    # Block axis: one bin per transformer block, so agreement reads as depth allocation
    # and not as sublayer-type allocation, and the plug-in entropy bias is ~7x smaller
    # on Qwen3. On a single-sublayer scope this is an identity relabel. Every active
    # metric is computed in one pass, sharing the profile build across them.
    by_metric = layer_profile_metric_matrices(to_block_axis(expert_allocation_df), items,
                                              ACTIVE_PROFILE_METRICS)
    # jsd_bits is metric-independent, so it is read off any entry and reported once.
    profile_jsd = by_metric[DEFAULT_PROFILE_METRIC][0]

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

            entry = {
                "concept": concept,
                "category": category,
                "hierarchy": f"{category} -> {concept}",
                "jaccard_pct": (intersection / union) * 100,
                "overlap_pct": (intersection / min(len_a, len_b)) * 100,
                # Raw set sizes behind the percentages, for scale
                "shared_expert_units": intersection,
                "concept_expert_units": len_a,
                "category_expert_units": len_b,
                "layer_profile_jsd_bits": profile_jsd[i, j] if has_profile else np.nan,
            }
            for metric in ACTIVE_PROFILE_METRICS:
                _, similarity, z = by_metric[metric]
                entry[profile_value_column(metric)] = similarity[i, j] if has_profile else np.nan
                entry[profile_z_column(metric)] = z[i, j] if has_profile else np.nan
            results.append(entry)

    similarity_metrics_df = pd.DataFrame(results)
    if similarity_metrics_df.empty: return similarity_metrics_df

    # The default metric keeps its original column names beside the generated ones. Module
    # 4 merges on layer_profile_similarity_pct, check_design_matches_module3.py pins it
    # against module 9's profile_js_distance, and every results tree written before this
    # extension carries it, so the alias is what keeps those comparisons possible. The
    # values are the same array, read twice.
    similarity_metrics_df["layer_profile_similarity_pct"] = \
        similarity_metrics_df[profile_value_column(DEFAULT_PROFILE_METRIC)]
    similarity_metrics_df["layer_profile_z"] = \
        similarity_metrics_df[profile_z_column(DEFAULT_PROFILE_METRIC)]

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

    # The two set-based plots, unchanged.
    for col, title in [("jaccard_pct", "Jaccard Similarity Index % (Global Equivalence)"),
                       ("overlap_pct", "Overlap Coefficient % (Strict Subsetting)")]:
        _plot_bar_with_leaders(
            plot_dataframe=similarity_metrics_df, x_col="hierarchy", y_col=col,
            title=title, x_label="Category->Concept", y_label="Percentage %",
            bar_colors=bar_colors, color_legend=color_legend, legend_title="Category",
            tick_label_colors=bar_colors,
            show_x_ticks=True, figsize=(width, 10.14), order=bar_order,
            out_path=sim_dir / f"{col.replace('_pct', '')}_hierarchy.png"
        )

    def _bar_with_zero_line(y_col: str, title: str, y_label: str, out_name: str) -> None:
        """A hierarchy bar plot carrying a dashed reference line at 0.

        Drawn onto an axes we own, so the line can be added before saving. Passing ax with
        save=True keeps the full-size axis labels (the helper drops to fontsize 11 when
        save is False) while suppressing its own savefig, which only fires when ax is None.
        """
        fig, ax = plt.subplots(figsize=(width, 10.14))
        _plot_bar_with_leaders(
            plot_dataframe=similarity_metrics_df, x_col="hierarchy", y_col=y_col,
            title=title, x_label="Category->Concept", y_label=y_label,
            bar_colors=bar_colors, color_legend=color_legend, legend_title="Category",
            tick_label_colors=bar_colors,
            show_x_ticks=True, order=bar_order, ax=ax, save=True
        )
        ax.axhline(0.0, color="#2f2f2f", linewidth=1.4, linestyle="--", zorder=5)
        fig.savefig(sim_dir / out_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # One pair of plots per registered metric: the raw agreement, and its count-matched z.
    # Filenames carry the metric name for ALL metrics including the default, so the folder
    # reads as one family rather than as a privileged file plus six additions. This renames
    # the previous layer_profile_similarity_hierarchy.png and layer_profile_z_hierarchy.png
    # to the js_distance members of that family, values unchanged.
    for metric in ACTIVE_PROFILE_METRICS:
        label = profile_metric_label(metric)
        signed = metric in SIGNED_PROFILE_METRICS
        value_col = profile_value_column(metric)
        title = f"Layer-Profile Agreement, {label} (Depth Allocation Agreement)"
        # A signed metric crosses zero, and the crossing is the reading: below it the two
        # words are heavy where the other is light. An unsigned one is bounded at 0 and the
        # line would be furniture, so it is drawn only where it means something.
        if signed:
            _bar_with_zero_line(value_col, title, profile_metric_value_label(metric),
                                f"{value_col}_hierarchy.png")
        else:
            _plot_bar_with_leaders(
                plot_dataframe=similarity_metrics_df, x_col="hierarchy", y_col=value_col,
                title=title, x_label="Category->Concept",
                y_label=profile_metric_value_label(metric),
                bar_colors=bar_colors, color_legend=color_legend, legend_title="Category",
                tick_label_colors=bar_colors,
                show_x_ticks=True, figsize=(width, 10.14), order=bar_order,
                out_path=sim_dir / f"{value_col}_hierarchy.png"
            )

        # The z companion is on a different scale (standard deviations, signed) whatever the
        # metric, so it always gets its own axis label and the reference line at 0, the value
        # meaning "no more agreement than two arbitrary words of these expert counts show".
        _bar_with_zero_line(
            profile_z_column(metric),
            f"Layer-Profile Agreement vs Count-Matched Null (z), {label}",
            "z (standard deviations above the null)",
            f"{profile_z_column(metric)}_hierarchy.png")

    write_profile_metric_comparison(similarity_metrics_df, sim_dir)
    return similarity_metrics_df


# Columns of the per-scope metric comparison, as {column: axis label}. Kept to four so the
# figure stays readable at plot_sublayer_comparison_bars' 4.2 inches per panel.
METRIC_COMPARISON_LABELS = {
    "mean_agreement": "Mean concept-to-parent agreement",
    "median_agreement": "Median concept-to-parent agreement",
    "mean_z": "Mean z vs count-matched null",
    "share_z_positive": "Share of concepts with z > 0",
}


def write_profile_metric_comparison(similarity_metrics_df: pd.DataFrame, sim_dir) -> pd.DataFrame:
    """
    Reduce every active profile metric to one row, so the metrics can be compared inside a
    scope the way sublayer_comparison compares scopes inside a module.

    This is the artifact the seven-metric extension exists for. Seven metrics times two
    figures is fourteen bar charts per scope, which nobody reads end to end, and the raw
    agreement columns are not comparable across metrics anyway since the registry fixes
    orientation but deliberately not scale. The z columns ARE comparable, every one of them
    being a standard score against the same count-matched null over the same pairs, which
    is why mean_z and share_z_positive are the columns to read here and the two raw columns
    are carried only for scale.

    share_z_positive is the blunt reading: what fraction of concepts agree with their
    category label on depth MORE than two arbitrary words of those expert counts do. At
    0.5 the metric is finding nothing, since z is centred on the null by construction.
    """
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
    # plot_sublayer_comparison_bars draws its FIRST row in the reference colour, which lands
    # on DEFAULT_PROFILE_METRIC because ACTIVE_PROFILE_METRICS keeps it first. That is the
    # right row to mark, since it is the metric every summary row reports, so the title says
    # so rather than leaving the dark bar looking arbitrary.
    plot_sublayer_comparison_bars(
        table, METRIC_COMPARISON_LABELS, sim_dir / "profile_metric_comparison.png",
        "Layer-profile metrics compared, concept against its category label\n"
        f"dark bar is the default metric, {DEFAULT_PROFILE_METRIC}",
        scope_col="metric_label")
    return table

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