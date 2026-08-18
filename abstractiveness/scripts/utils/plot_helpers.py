import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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


def plot_hexbin_with_trends(x, y, same, out_path, x_label: str, y_label: str,
                            colorbar_label: str = "Concept pairs (log)") -> None:
    """
    Density hexbin of tens of thousands of concept pairs, split into same-category and
    different-category panels, each carrying a binned median, a straight OLS fit, and a
    LOWESS smooth. Showing the OLS line against the LOWESS smooth is the point: it lets a
    reader see the true (often saturating) trend next to the straight line a naive linear
    correlation would draw.

    ``same`` is the boolean same-category mask over the same flat pair ordering as x and y.

    Shared by module 8's RSA scatter and module 4's all-pairs panels. statsmodels is
    imported inside the function because it costs about a second at import time and every
    module in the pipeline imports this file, while only two of them draw this plot.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    hb = None
    for ax, mask, title in [(axes[0], same, "Same category"), (axes[1], ~same, "Different category")]:
        if not mask.any():
            continue
        hb = ax.hexbin(x[mask], y[mask], gridsize=45, bins="log", cmap="magma", mincnt=1)
        order = np.argsort(x[mask])
        xb, yb = x[mask][order], y[mask][order]
        edges = np.quantile(xb, np.linspace(0, 1, 11))
        centers = 0.5 * (edges[:-1] + edges[1:])
        meds = [np.median(yb[(xb >= lo) & (xb <= hi)]) if ((xb >= lo) & (xb <= hi)).any() else np.nan
                for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(centers, meds, color="#ffa600", linewidth=2, label="Binned median")
        if xb.std() > 0:
            pearson_r = stats.pearsonr(xb, yb).statistic
            slope, intercept = np.polyfit(xb, yb, 1)
            xs_line = np.array([xb.min(), xb.max()])
            ax.plot(xs_line, intercept + slope * xs_line, color="#2f2f2f", linestyle="--",
                    linewidth=1.6, label=f"OLS (r={pearson_r:.2f})")
            smoothed = lowess(yb, xb, frac=0.4, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="#00b3b3", linestyle=":",
                    linewidth=2.2, label="LOWESS")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.legend(loc="upper left")
    axes[0].set_ylabel(y_label)
    if hb is not None:
        fig.colorbar(hb, ax=axes, label=colorbar_label)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- shared style for two-panel model-comparison figures -------------------------------
# Module 9 draws the same figure shape several times: a left panel ranking feature sets on
# one score, and a right panel resolving the winner per held-out category. They were styled
# independently and drifted apart, so the grammar lives here once and every such figure
# imports it. Subchapter 9.4's figure was the most developed of them and is the reference,
# in particular its habit of printing each bar's value next to it together with the value
# read against a fixed reference, since a bare correlation or accuracy is not interpretable
# without knowing what the attainable maximum or the chance level is.
COMPARISON_STYLE = {
    "figsize": (14, 5.4),
    "bar_height": 0.62,
    "grouped_bar_height": 0.38,
    "title_size": 12,
    "label_size": 10,
    "tick_size": 10,
    "value_size": 9,
    "legend_size": 9,
    # Inches of axes height per bar row, which is what makes a comparison panel scale with
    # its content instead of squeezing every row into the fixed "figsize" height. A row
    # carrying two series needs more, since it holds two bars plus the gap between them.
    # The values are set so a row is comfortably taller than the 10 point tick label and
    # the 9 point value annotation printed beside it.
    "row_height": 0.32,
    "grouped_row_height": 0.40,
    # Fixed overhead per panel for the title, the x axis label and the legend below it.
    "panel_margin": 2.4,
}


def comparison_panel_height(n_rows: int, grouped: bool = False, min_height: float = 4.6) -> float:
    """
    Axes height in inches for a horizontal-bar comparison panel of ``n_rows`` rows.

    The comparison figures of module 9 were written when a feature grid held six cells, so a
    fixed height was fine. The grid is now generated per registered profile metric, six cells
    times seven metrics plus the metric-free Jaccard cell, and at 36 rows a fixed height
    overlaps every tick label with the bar above it and prints the value annotations on top of
    one another. Height therefore scales with the row count, and ``min_height`` keeps a small
    panel from collapsing when only one metric is active.

    ``grouped`` selects the taller per-row allowance for a panel drawing two series per row.
    """
    per_row = COMPARISON_STYLE["grouped_row_height" if grouped else "row_height"]
    return max(min_height, COMPARISON_STYLE["panel_margin"] + per_row * max(n_rows, 1))

# Base is the reference/baseline series, accent the model under test, muted anything carried
# as a diagnostic rather than a candidate, and reference the line marking a ceiling or a
# chance level, which is deliberately the only green in the figure.
COMPARISON_COLORS = {
    "base": "#4B5A6A",
    "accent": "#D96A5B",
    "muted": "#9aa5b1",
    "reference": "#1a7f37",
    "zero": "#2f2f2f",
}


def comparison_legend(axis, ncol: int = 2, pad_inches: float = 0.5) -> None:
    """
    Legend below the axes, unframed, so it never covers a bar.

    The offset is expressed in INCHES and converted to the axes-relative units
    ``bbox_to_anchor`` wants, rather than left at a fixed fraction of the axes height. A
    comparison panel now grows with its row count, so a fixed fraction that sat tight under a
    5 inch panel drifts nearly two inches below a 14 inch one, stranding the legend in
    whitespace far from the bars it explains.
    """
    axes_height = axis.get_position().height * axis.figure.get_figheight()
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -pad_inches / max(axes_height, 0.1)),
                fontsize=COMPARISON_STYLE["legend_size"], frameon=False, ncol=ncol)


def annotate_barh_values(axis, values, formatter, pad: float = 0.012) -> None:
    """
    Print each horizontal bar's value at its tip, skipping missing ones.

    ``formatter`` takes the value and returns the label, so the caller decides how to express
    the reading against its own reference (fraction of a noise ceiling, points above chance).

    Text lands at the ROW centre. On a grouped panel, whose two series sit at
    ``row +/- height/2``, that is the gap between them rather than beside the bar being
    annotated, which is deliberate: the ranker panel overlays its per-category dots on one of
    the two half-rows, so aligning the text with that bar prints it straight through them.
    """
    for index, value in enumerate(values):
        if pd.notna(value):
            axis.text(value + pad, index, formatter(value), va="center", ha="left",
                      fontsize=COMPARISON_STYLE["value_size"])


def style_comparison_axis(axis, labels, title: str, xlabel: str, bold=()) -> None:
    """Shared axis furniture: y ticks from ``labels``, no y grid, top-down order.

    Names in ``bold`` are drawn heavier, which is how a headline feature set is marked when
    colour is already carrying another meaning such as which series a bar belongs to.
    """
    positions = np.arange(len(labels))
    axis.set_yticks(positions)
    axis.set_yticklabels([str(label).replace("_", " ") for label in labels],
                         fontsize=COMPARISON_STYLE["tick_size"])
    for tick, label in zip(axis.get_yticklabels(), labels):
        if label in bold:
            tick.set_fontweight("bold")
    axis.invert_yaxis()
    axis.set_xlabel(xlabel, fontsize=COMPARISON_STYLE["label_size"])
    axis.set_title(title, fontsize=COMPARISON_STYLE["title_size"])
    axis.grid(axis="y", visible=False)


_WHOLE_MODEL_BAR_COLOR = "#1f2d3d"
_SUBLAYER_BAR_COLOR = "#2a78d6"


def plot_sublayer_comparison_bars(comparison_df: pd.DataFrame, value_labels: dict, out_path,
                                  title: str, scope_col: str = "scope") -> None:
    """
    Small-multiple horizontal bars comparing every analysis scope on a module's headline
    metrics: one panel per entry of value_labels ({column: axis label}), one bar per row
    of comparison_df, in the frame's own order (whole model first).

    This is the artifact that makes a per-sublayer sweep readable. Seven sublayers times
    five AP thresholds is far too many folders to open by hand, so each module writes one
    of these next to its sublayers/ folder, and the per-scope folders become the evidence
    behind it rather than the thing anyone reads. The whole-model row is drawn in a
    distinct color because it is the reference the sublayers are being judged against,
    not another sublayer. Columns that are entirely NaN are skipped, so a metric that is
    undefined at a strict AP threshold drops its panel instead of drawing empty axes.
    """
    panels = [(col, label) for col, label in value_labels.items()
              if col in comparison_df.columns and comparison_df[col].notna().any()]
    if comparison_df.empty or not panels:
        return

    scopes = comparison_df[scope_col].tolist()
    y = range(len(scopes))
    colors = [_WHOLE_MODEL_BAR_COLOR if i == 0 else _SUBLAYER_BAR_COLOR for i in y]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 1.0 + 0.42 * len(scopes)),
                             sharey=True, squeeze=False)
    for ax, (col, label) in zip(axes[0], panels):
        values = comparison_df[col].astype(float)
        ax.barh(list(y), values.fillna(0.0), color=colors, height=0.66)
        ax.set_xlabel(label, fontsize=10)
        ax.grid(axis="y", visible=False)

        # Pad the axis around the data's own range rather than around zero: a panel whose
        # values are all negative (e.g. an r that is negative in every scope) would
        # otherwise be squeezed into the left edge of an axis running to +1.
        low, high = min(0.0, values.min()), max(0.0, values.max())
        span = (high - low) or 1.0
        ax.set_xlim(low - 0.04 * span, high + 0.16 * span)

        # Labels always sit OUTSIDE the bar, on the side the bar grows toward, so they
        # stay readable on the dark whole-model bar instead of being drawn over it.
        for yi, value in zip(y, values):
            if pd.isna(value):
                continue
            offset = 0.02 * span if value >= 0 else -0.02 * span
            ax.annotate(f"{value:.3g}", (value + offset, yi), fontsize=8, va="center",
                        ha="left" if value >= 0 else "right", color="#333")

    axes[0][0].set_yticks(list(y))
    axes[0][0].set_yticklabels(scopes, fontsize=9)
    axes[0][0].invert_yaxis()
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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