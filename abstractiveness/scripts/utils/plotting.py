"""
Figure helpers: category palette, labelled bars, heatmaps, violins, comparison panels.

Generic drawing code shared by every module, no analysis logic.

Explanations: none needed.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


FONT_FAMILY = "Nimbus Sans"
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 13, 11, 9, 9
DETAIL_COLOR = "#555555"
CORRELATION_COLORS = {"points": "#4C6A92", "line": "#A8322D"}
LEVEL_COLORS = {"Broad Categories": "#2E4057", "Specific Concepts": "#E0913B"}
SERIES_COLORS = {"primary": "#2E4057", "secondary": "#A8322D", "accent": "#E0913B", "highlight": "#7B4F9E",
                 "muted": "#8A8A8A", "band": "#D6E4F0"}
SEQUENTIAL_CMAP = "magma"
DIVERGING_CMAP = "RdBu_r"
DENSITY_CMAP = "Blues"


def apply_plot_style() -> None:
    """Font, sizes, spines and grid shared by every figure."""
    sns.set_theme(style="ticks", rc={
        "font.family": "sans-serif", "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
        "axes.titlesize": TITLE_SIZE, "axes.titleweight": "normal", "axes.titlepad": 10,
        "axes.labelsize": LABEL_SIZE, "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE, "legend.title_fontsize": LEGEND_SIZE + 1, "legend.frameon": True,
        "legend.framealpha": 0.92, "legend.edgecolor": "#dddddd", "legend.fancybox": False,
        "figure.titlesize": TITLE_SIZE + 1, "figure.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    })


apply_plot_style()


def _title_scale(fig) -> float:
    """Title size multiplier, growing with figure width so very wide figures keep a readable title."""
    return min(max(1.0, fig.get_figwidth() / 14.0), 1.8)


def set_title(ax, title: str, detail: str = None) -> None:
    """Sentence-case title in regular weight, with an optional smaller grey detail line below it."""
    scale = _title_scale(ax.figure)
    ax.set_title(title, fontsize=TITLE_SIZE * scale, pad=(10 + (16 if detail else 0)) * scale)
    if detail:
        ax.annotate(detail, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 7 * scale),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=(TICK_SIZE + 1) * scale, color=DETAIL_COLOR)


def set_suptitle(fig, title: str, y: float = None) -> None:
    """Figure-level title in regular weight, scaled like set_title."""
    kwargs = {} if y is None else {"y": y}
    fig.suptitle(title, fontsize=(TITLE_SIZE + 1) * _title_scale(fig), **kwargs)


def twin_axis(ax):
    """Right-hand twin axis with its spine shown and no grid of its own."""
    twin = ax.twinx()
    twin.spines["right"].set_visible(True)
    twin.grid(False)
    return twin


def draw_regression(ax, data: pd.DataFrame, x_col: str, y_col: str, fit: bool = True, scatter: bool = True,
                    size: float = 22) -> None:
    """Scatter with its linear fit and 95% band in the shared correlation colours, seeded so reruns match."""
    point_kws = {"s": size, "alpha": 0.75, "edgecolors": "white", "linewidths": 0.4}
    if not fit:
        ax.scatter(data[x_col], data[y_col], color=CORRELATION_COLORS["points"], **point_kws)
        return
    sns.regplot(data=data, x=x_col, y=y_col, ax=ax, scatter=scatter, seed=0,
                color=CORRELATION_COLORS["line"], scatter_kws={**point_kws, "color": CORRELATION_COLORS["points"]},
                line_kws={"color": CORRELATION_COLORS["line"], "linewidth": 1.8})


def stats_line(coefficient: str, value, p=None, n=None) -> str:
    """Statistics in one format everywhere, e.g. 'Pearson r = 0.32, p = 1.2e-04, n = 198'."""
    parts = [f"{coefficient} = {value:.2f}"]
    if p is not None:
        parts.append(f"p = {p:.1e}")
    if n is not None:
        parts.append(f"n = {n}")
    return ", ".join(parts)


def coverage_line(n: int, n_total: int) -> str:
    """Detail line of a panel below the coverage floors."""
    coverage = n / n_total if n_total else 0.0
    return f"insufficient coverage, n = {n} ({coverage:.0%} of {n_total})"


_CATEGORY_PALETTE = [
    "#2a78d6", "#eb6834", "#008300", "#e87ba4",
    "#4a3aa7", "#eda100", "#1baf7a", "#e34948",
]

_CATEGORY_PALETTE_EXTRA = [
    "#00c2d1", "#a65628", "#7f7f7f", "#c8b900",
    "#f781bf", "#4daf4a", "#984ea3", "#666666", "#00807a",
]


def build_category_color_map(categories) -> dict:
    """Map each category to a fixed color, assigning palette slots in the order the categories first appear in ``categories``."""
    seen = []
    for c in categories:
        if c is None or (isinstance(c, float) and pd.isna(c)):
            continue
        if c not in seen:
            seen.append(c)
    full = _CATEGORY_PALETTE + _CATEGORY_PALETTE_EXTRA
    return {c: full[i % len(full)] for i, c in enumerate(seen)}


def fig_width_for(n_items: int, per_item: float, min_w: float = 12.0, max_w: float = 120.0) -> float:
    """Figure width (inches) that scales with the number of plotted items (model layers or concepts) so dense axes stay legible, clamped to [min_w, max_w]."""
    return float(min(max(n_items * per_item, min_w), max_w))


def apply_rotated_leader_labels(ax, labels, axis='x', fontsize=9, pad=2, show_leaders=True):
    """Style a dense categorical axis the same way across every module: short leader ticks with 45-degree labels anchored (rotation_mode='anchor') so each label's end — its last word — sits directly under/beside its tick, exactly as the heatmap leaders do."""
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
    """Create a bar chart with category labels on the relevant axis, optionally with leader-line tick marks connecting each (often rotated) label down to its bar."""
    if ax is None:
        plt.figure(figsize=figsize)
        ax_current = plt.gca()
    else:
        ax_current = ax

    plot_args = {"edgecolor": "black", "orient": orient}
    if bar_colors is not None:
        plot_args["color"] = "#cccccc"
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

    set_title(ax_current, title)

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
        ax_current.set_xticklabels(labels, rotation=45, ha='right', va='top', rotation_mode='anchor',
                                   fontsize=10 if save else 9)
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
                           palette: dict = None, x_label: str = "Group",
                           figsize=(8, 6.5)) -> None:
    """General violin + strip comparison of a numeric metric across named groups."""
    plot_df = pd.concat(
        [pd.DataFrame({"value": pd.Series(v).dropna(), "group": g}) for g, v in values_by_group.items()],
        ignore_index=True)
    group_order = list(values_by_group)

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=plot_df, x='group', y='value', hue='group', order=group_order, ax=ax,
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
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    set_title(ax, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_hexbin_with_trends(x, y, same, out_path, x_label: str, y_label: str,
                            colorbar_label: str = "Concept pairs (log)") -> None:
    """Density hexbin of tens of thousands of concept pairs, split into same-category and different-category panels, each carrying a binned median, a straight OLS fit, and a LOWESS smooth."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    hb = None
    for ax, mask, title in [(axes[0], same, "Same category"), (axes[1], ~same, "Different category")]:
        if not mask.any():
            continue
        hb = ax.hexbin(x[mask], y[mask], gridsize=45, bins="log", cmap=DENSITY_CMAP, mincnt=1)
        order = np.argsort(x[mask])
        xb, yb = x[mask][order], y[mask][order]
        edges = np.quantile(xb, np.linspace(0, 1, 11))
        centers = 0.5 * (edges[:-1] + edges[1:])
        meds = [np.median(yb[(xb >= lo) & (xb <= hi)]) if ((xb >= lo) & (xb <= hi)).any() else np.nan
                for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(centers, meds, color="#1a1a1a", linewidth=1.6, marker="o", markersize=3, label="Binned median")
        if xb.std() > 0:
            pearson_r = stats.pearsonr(xb, yb).statistic
            slope, intercept = np.polyfit(xb, yb, 1)
            xs_line = np.array([xb.min(), xb.max()])
            ax.plot(xs_line, intercept + slope * xs_line, color=CORRELATION_COLORS["line"],
                    linewidth=1.8, label=f"Linear fit, Pearson r = {pearson_r:.2f}")
            smoothed = lowess(yb, xb, frac=0.4, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="#1a1a1a", linestyle=":",
                    linewidth=1.8, label="LOWESS")
        set_title(ax, title)
        ax.grid(False)
        ax.set_xlabel(x_label)
        ax.legend(loc="upper left")
    axes[0].set_ylabel(y_label)
    if hb is not None:
        fig.colorbar(hb, ax=axes, label=colorbar_label)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


COMPARISON_STYLE = {
    "figsize": (14, 5.4),
    "bar_height": 0.62,
    "grouped_bar_height": 0.38,
    "title_size": 12,
    "label_size": 10,
    "tick_size": 10,
    "value_size": 9,
    "legend_size": 9,
    "row_height": 0.32,
    "grouped_row_height": 0.40,
    "panel_margin": 2.4,
}


def comparison_panel_height(n_rows: int, grouped: bool = False, min_height: float = 4.6) -> float:
    """Axes height in inches for a horizontal-bar comparison panel of ``n_rows`` rows."""
    per_row = COMPARISON_STYLE["grouped_row_height" if grouped else "row_height"]
    return max(min_height, COMPARISON_STYLE["panel_margin"] + per_row * max(n_rows, 1))


COMPARISON_COLORS = {
    "base": SERIES_COLORS["primary"],
    "accent": SERIES_COLORS["secondary"],
    "muted": "#9aa5b1",
    "reference": "#1a7f37",
    "zero": "#2f2f2f",
}


def comparison_legend(axis, ncol: int = 2, pad_inches: float = 0.5) -> None:
    """Legend below the axes, unframed, so it never covers a bar."""
    axes_height = axis.get_position().height * axis.figure.get_figheight()
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -pad_inches / max(axes_height, 0.1)),
                fontsize=COMPARISON_STYLE["legend_size"], frameon=False, ncol=ncol)


def annotate_barh_values(axis, values, formatter, pad: float = 0.012) -> None:
    """Print each horizontal bar's value at its tip, skipping missing ones."""
    for index, value in enumerate(values):
        if pd.notna(value):
            axis.text(value + pad, index, formatter(value), va="center", ha="left",
                      fontsize=COMPARISON_STYLE["value_size"])


def style_comparison_axis(axis, labels, title: str, xlabel: str, bold=()) -> None:
    """Shared axis furniture: y ticks from ``labels``, no y grid, top-down order."""
    positions = np.arange(len(labels))
    axis.set_yticks(positions)
    axis.set_yticklabels([str(label).replace("_", " ") for label in labels],
                         fontsize=COMPARISON_STYLE["tick_size"])
    for tick, label in zip(axis.get_yticklabels(), labels):
        if label in bold:
            tick.set_fontweight("bold")
    axis.invert_yaxis()
    axis.set_xlabel(xlabel, fontsize=COMPARISON_STYLE["label_size"])
    set_title(axis, title)
    axis.grid(axis="y", visible=False)


_WHOLE_MODEL_BAR_COLOR = SERIES_COLORS["primary"]

_SUBLAYER_BAR_COLOR = "#8FA8C8"


def plot_sublayer_comparison_bars(comparison_df: pd.DataFrame, value_labels: dict, out_path,
                                  title: str, scope_col: str = "scope") -> None:
    """Small-multiple horizontal bars comparing every analysis scope on a module's headline metrics: one panel per entry of value_labels ({column: axis label}), one bar per row of comparison_df, in the frame's own order (whole model first)."""
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

        low, high = min(0.0, values.min()), max(0.0, values.max())
        span = (high - low) or 1.0
        ax.set_xlim(low - 0.04 * span, high + 0.16 * span)

        for yi, value in zip(y, values):
            if pd.isna(value):
                continue
            offset = 0.02 * span if value >= 0 else -0.02 * span
            ax.annotate(f"{value:.3g}", (value + offset, yi), fontsize=8, va="center",
                        ha="left" if value >= 0 else "right", color="#333")

    axes[0][0].set_yticks(list(y))
    axes[0][0].set_yticklabels(scopes, fontsize=9)
    axes[0][0].invert_yaxis()
    set_suptitle(fig, title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_with_leaders(matrix, concepts, title, out_path, cmap=SEQUENTIAL_CMAP, concept_colors=None,
                               color_legend=None, category_boundaries=None, center=None) -> None:
    """Heatmap of a concept-by-concept matrix with coloured labels and leader lines, diverging when centered."""
    n = len(concepts)
    size = fig_width_for(n, 0.26, min_w=18.0, max_w=80.0)
    plt.figure(figsize=(size, size * 0.83))
    ax = sns.heatmap(matrix, xticklabels=False, yticklabels=concepts, cmap=cmap, center=center)

    set_title(ax, title)
    ax.set_xlabel("")
    line_color = "white" if center is None else "#444444"

    for i in range(1, n):
        ax.axhline(i, color='white', linewidth=0.3, alpha=0.25)
        ax.axvline(i, color='white', linewidth=0.3, alpha=0.25)

    if category_boundaries:
        for boundary in category_boundaries:
            ax.axhline(boundary, color=line_color, linewidth=1, alpha=1.0)
            ax.axvline(boundary, color=line_color, linewidth=1, alpha=1.0)

    if concept_colors is not None:
        for lbl, col in zip(ax.get_yticklabels(), concept_colors):
            lbl.set_color(col)

    for i, concept in enumerate(concepts):
        x_pos = i + 0.5
        lead_color = concept_colors[i] if concept_colors is not None else "black"
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
