import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# Validated categorical hues (dataviz skill reference instance, light surface),
# ordered so consecutive slots strictly alternate cool/warm families:
# blue, orange, green, magenta, violet, yellow, aqua, red. Slots are assigned to
# categories in first-appearance order (see build_category_color_map), which matches
# the contiguous category blocks along these axes -- so axis-adjacent categories
# always land on a cool/warm pair and never on two similar warm tones (the old tail
# put salmon, pink and orange side by side).
_CATEGORY_PALETTE = [
    "#2a78d6", "#eb6834", "#008300", "#e87ba4",
    "#4a3aa7", "#eda100", "#1baf7a", "#e34948",
]
# Extra well-separated hues for datasets with >8 categories (the 150-concept set has
# 17). Beyond 8, color alone is below the CVD floor, so it relies on the spatial
# grouping already present in these plots (contiguous blocks / block-diagonal heatmap).
_CATEGORY_PALETTE_EXTRA = [
    "#00c2d1", "#a65628", "#7f7f7f", "#c8b900",
    "#f781bf", "#4daf4a", "#984ea3", "#666666", "#00807a",
]


def build_category_color_map(categories) -> dict:
    """
    Map each category to a fixed color, assigning palette slots in the order the
    categories first appear in ``categories``. Passing the metadata's (grouped)
    category column therefore paints contiguous axis blocks with adjacent,
    maximally CVD-separated slots, and produces the SAME map in every module so a
    category keeps one color across all plots. None/NaN entries are ignored.
    """
    seen = []
    for c in categories:
        if c is None or (isinstance(c, float) and pd.isna(c)):
            continue
        if c not in seen:
            seen.append(c)
    full = _CATEGORY_PALETTE + _CATEGORY_PALETTE_EXTRA
    return {c: full[i % len(full)] for i, c in enumerate(seen)}


def fig_width_for(n_items: int, per_item: float, min_w: float = 12.0, max_w: float = 120.0) -> float:
    """
    Figure width (inches) that scales with the number of plotted items (model layers
    or concepts) so dense axes stay legible, clamped to [min_w, max_w]. Keeps a small
    model (GPT-2's 48 layers) from producing a needlessly huge canvas and a large one
    (Qwen's 196 layers, ~200 concepts) from cramming labels on top of each other.
    """
    return float(min(max(n_items * per_item, min_w), max_w))


def apply_rotated_leader_labels(ax, labels, axis='x', fontsize=9, pad=2, show_leaders=True):
    """
    Style a dense categorical axis the same way across every module: short leader ticks
    with 45-degree labels anchored (rotation_mode='anchor') so each label's end — its last
    word — sits directly under/beside its tick, exactly as the heatmap leaders do. Shared by
    the bar helper and the custom matplotlib plots in modules 6 and 7 so they all match.
    """
    n = len(labels)
    if axis == 'x':
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', va='top', rotation_mode='anchor', fontsize=fontsize)
        ax.tick_params(axis='x', length=5 if show_leaders else 0, width=1, direction='out',
                       color='black', bottom=show_leaders, pad=pad)
    else:
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, rotation=0, ha='right', va='center', fontsize=fontsize)
        ax.tick_params(axis='y', length=5 if show_leaders else 0, width=1, direction='out',
                       color='black', left=show_leaders, pad=pad)


def _plot_bar_with_leaders(
    plot_dataframe, x_col, y_col,
    title, x_label=None, y_label=None, legend_title=None,
    color=None, hue=None, palette=None, orient='v', custom_tick_labels=None,
    out_path=None, figsize=(40.56, 10.14), ax=None, save=True,
    show_x_ticks=False, show_y_ticks=False, bar_colors=None, color_legend=None,
    tick_label_colors=None, **kwargs
) -> plt.Axes:
    """
    Create a bar chart with category labels on the relevant axis, optionally with leader-line
    tick marks connecting each (often rotated) label down to its bar.
    Supports both vertical and horizontal orientations, optional hue-based grouping, and color customization.
    Returns the matplotlib Axes object, and optionally saves to file if out_path is provided.

    Note: sns.set_theme(style="whitegrid") suppresses axis tick marks by default, so the
    show_x_ticks/show_y_ticks flags below explicitly re-enable them (via tick_params'
    bottom=/left=) rather than relying on the theme's defaults.

    Args:
        plot_dataframe: DataFrame containing the data to plot.
        x_col: Column name for x-axis (vertical) or bars (horizontal).
        y_col: Column name for y-axis values or bar labels (horizontal).
        title: Plot title.
        x_label, y_label: Axis labels.
        legend_title: Title for legend if hue grouping is used.
        color: Single color for all bars (if no palette/hue).
        hue: Column to group bars by color.
        palette: Color palette for hue groups.
        orient: 'v' for vertical bars, 'h' for horizontal bars.
        custom_tick_labels: Optional custom labels for axes.
        out_path: Path to save the figure (PNG at 300 dpi).
        figsize: Figure size as (width, height).
        ax: Existing axes object to plot on; creates new figure if None.
        save: If True and ax is None, saves figure; always displays if ax is not None.
        show_x_ticks: If True (orient='v'), draws leader-line tick marks under the x-axis
            category labels. Most useful when there are many categories (e.g. 48 model layers),
            where the tick mark is what visually ties a rotated label to its bar.
        show_y_ticks: If True (orient='h'), draws the equivalent leader-line tick marks to the
            left of the y-axis category labels.
        bar_colors: Optional per-bar colors (one entry per row of plot_dataframe, in x order)
            used to color each bar individually, e.g. by the concept's category. Overrides
            color/hue/palette. Pair with order=list(plot_dataframe[x_col]) so bar positions
            stay aligned with this list.
        color_legend: Optional {label: color} mapping rendered as a manual swatch legend
            (used with bar_colors, since per-bar coloring produces no automatic legend).
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax_current = plt.gca()
    else:
        ax_current = ax

    plot_args = {"edgecolor": "black", "orient": orient}
    if bar_colors is not None:
        plot_args["color"] = "#cccccc"  # placeholder; patches are recolored below
    elif hue is not None:
        plot_args["hue"] = hue
        if palette is not None: plot_args["palette"] = palette
    elif palette is not None:
        plot_args["palette"] = palette
    else:
        plot_args["color"] = color

    plot_args.update(kwargs)

    sns.barplot(data=plot_dataframe, x=x_col, y=y_col, ax=ax_current, **plot_args)

    if bar_colors is not None:
        for patch, bar_color in zip(ax_current.patches, bar_colors):
            patch.set_facecolor(bar_color)

    ax_current.set_title(title, fontsize=18 if save else 14, pad=15)

    if color_legend:
        handles = [Patch(facecolor=col, edgecolor="black", label=lab) for lab, col in color_legend.items()]
        ax_current.legend(
            handles=handles, title=legend_title or "Category", loc="upper right",
            fontsize=10, title_fontsize=12, framealpha=0.9, ncol=1 if len(handles) <= 8 else 2,
        )
    elif legend_title and ax_current.get_legend():
        ax_current.legend(title=legend_title, loc='upper right' if not save else 'best')

    if orient == 'v':
        if custom_tick_labels is not None:
            labels = custom_tick_labels
        else:
            labels = [t.get_text() for t in ax_current.get_xticklabels()]
            
        ax_current.set_xticks(ax_current.get_xticks())
        # Half-length leader ticks with anchored 45-degree labels: rotation_mode='anchor'
        # + ha='right'/va='top' pins each label's end (its last word) directly under its
        # tick, so the leader points at the label end (matches the heatmap look).
        ax_current.set_xticklabels(labels, rotation=45, ha='right', va='top', rotation_mode='anchor',
                                   fontsize=10 if save else 9)
        # Optional per-label colors (e.g. each concept label in its category's color).
        if tick_label_colors is not None:
            for tick_label, label_color in zip(ax_current.get_xticklabels(), tick_label_colors):
                tick_label.set_color(label_color)
        ax_current.tick_params(axis='x', length=5, width=1, direction='out', color='black', bottom=show_x_ticks, pad=2)
        
        if x_label: ax_current.set_xlabel(x_label, fontsize=14 if save else 11)
        else: ax_current.set_xlabel("")
        if y_label: ax_current.set_ylabel(y_label, fontsize=14 if save else 11)
        else: ax_current.set_ylabel("")
        
        if ax is None: plt.subplots_adjust(bottom=0.25)

    elif orient == 'h':
        if custom_tick_labels is not None:
            labels = custom_tick_labels
        else:
            labels = [t.get_text() for t in ax_current.get_yticklabels()]
            
        ax_current.set_yticks(ax_current.get_yticks())
        ax_current.set_yticklabels(labels, fontsize=12)
        ax_current.tick_params(axis='y', length=5, width=1, direction='out', color='black', left=show_y_ticks, pad=2)

        if x_label: ax_current.set_xlabel(x_label, fontsize=14 if save else 11)
        else: ax_current.set_xlabel("")
        if y_label: ax_current.set_ylabel(y_label, fontsize=14 if save else 11)
        else: ax_current.set_ylabel("")
                        
        if ax is None: plt.subplots_adjust(left=0.25)

    if save and ax is None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    return ax_current


def plot_comparison_violin(values_by_group: dict, y_label: str, title: str, out_path,
                           palette: dict = None, x_label: str = "Concept Grouping",
                           figsize=(10, 7)) -> None:
    """
    General violin + strip comparison of a numeric metric across named groups.

    values_by_group: {group_label: iterable of values} in display order; NaNs dropped.
    Each violin is clipped to the observed data range (cut=0), shows quartile lines
    with Q1/Median/Q3 text labels, and overlays the raw points so small groups stay
    honest (a violin over 8 points is otherwise fiction).
    """
    plot_df = pd.concat(
        [pd.DataFrame({"value": pd.Series(v).dropna(), "group": g}) for g, v in values_by_group.items()],
        ignore_index=True)
    group_order = list(values_by_group)

    plt.figure(figsize=figsize)
    ax = sns.violinplot(data=plot_df, x='group', y='value', hue='group', order=group_order,
                        legend=False, palette=palette, inner='quartile', linewidth=1.5,
                        alpha=0.8, cut=0)
    sns.stripplot(data=plot_df, x='group', y='value', order=group_order, color='black',
                  alpha=0.4, size=4, jitter=True, ax=ax)
    for i, group in enumerate(group_order):
        group_values = plot_df[plot_df['group'] == group]['value']
        if group_values.empty:
            continue
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
        for quantile, label, weight in ((0.25, 'Q1 (25%)', 'normal'), (0.50, 'Median', 'bold'), (0.75, 'Q3 (75%)', 'normal')):
            ax.text(i + 0.15, group_values.quantile(quantile), label, va='center', ha='left',
                    fontsize=9, fontweight=weight, color='#333333', bbox=bbox_props)
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_heatmap_with_leaders(matrix, concepts, title, out_path, cmap, concept_colors=None,
                               color_legend=None, category_boundaries=None) -> None:
    """
    Generate a heatmap visualization of a similarity/distance matrix with concept labels and leader lines.
    Each column is labeled with a concept name via annotated leader lines from the top edge.
    The figure size scales with the number of concepts so labels stay legible.
    Saves the resulting figure as a high-resolution PNG.

    Args:
        matrix: 2D numpy array or matrix to visualize.
        concepts: List of concept names for row/column labels.
        title: Plot title.
        out_path: Path where to save the PNG file.
        cmap: Colormap name for the heatmap.
        concept_colors: Optional per-concept colors (aligned to ``concepts``) applied to both
            the x leader lines/labels and the y tick labels, e.g. to color each concept by its
            category. Defaults to black.
        color_legend: Optional {label: color} mapping drawn as a swatch legend below the plot.
        category_boundaries: Optional cell indices where a new category block starts, drawn as
            a fully opaque 1 px white line on both axes, bold enough to stand out from the
            per-concept grid described below.
    """
    n = len(concepts)
    # The heatmap grows in BOTH dimensions with concept count, so pixels scale
    # quadratically -- use a gentler per-concept factor than the 1-D bar plots and a
    # 200-dpi export to keep the file to a sane size while still widening the cells.
    size = fig_width_for(n, 0.26, min_w=18.0, max_w=80.0)
    plt.figure(figsize=(size, size * 0.83))
    ax = sns.heatmap(matrix, xticklabels=False, yticklabels=concepts, cmap=cmap)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("")

    # Faint separator at every concept boundary, so each square is traceable to its own
    # row/column even away from a category edge, drawn thinner and more transparent than the
    # bold category-block lines below so the two stay visually distinct.
    for i in range(1, n):
        ax.axhline(i, color='white', linewidth=0.3, alpha=0.25)
        ax.axvline(i, color='white', linewidth=0.3, alpha=0.25)

    # Fully opaque 1 px separators at category-block boundaries, drawn on top of the per-concept
    # grid so the category structure reads clearly against the finer concept-level grid.
    if category_boundaries:
        for boundary in category_boundaries:
            ax.axhline(boundary, color='white', linewidth=1, alpha=1.0)
            ax.axvline(boundary, color='white', linewidth=1, alpha=1.0)

    # Color the y-axis concept labels by category.
    if concept_colors is not None:
        for lbl, col in zip(ax.get_yticklabels(), concept_colors):
            lbl.set_color(col)

    for i, concept in enumerate(concepts):
        x_pos = i + 0.5
        lead_color = concept_colors[i] if concept_colors is not None else "black"
        # Half-length leader line (5 pts) with the label just past its end, so the label
        # sits tight against the line pointing at its column.
        ax.annotate("", xy=(x_pos, n), xytext=(0, -5), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=lead_color, linewidth=1.0, shrinkA=0, shrinkB=0))
        ax.annotate(concept, xy=(x_pos, n), xytext=(2, -6), textcoords="offset points",
                    ha='right', va='top', rotation=45, fontsize=10, color=lead_color)

    if color_legend:
        handles = [Patch(facecolor=col, edgecolor="black", label=lab) for lab, col in color_legend.items()]
        ax.legend(
            handles=handles, title="Category", loc="upper center", bbox_to_anchor=(0.5, -0.05),
            ncol=min(len(handles), 8), fontsize=12, title_fontsize=14, framealpha=0.9,
        )

    plt.subplots_adjust(bottom=0.15)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()