"""
Correlation panels, in the values' own units or on the rank residuals of a partial coefficient, and their
correlation_summary rows, gated by coverage floors.

Used by the correlation sections of modules 1 and 2.

Explanations: documentation/README.md (correlations and permutation tests), module_1_expert_distribution.md and
module_2_similarities.md.
"""

import logging
import pathlib
from typing import NamedTuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.logging_and_io import save_dataframe
from utils.plotting import draw_regression, set_title, set_suptitle, stats_line, coverage_line
from utils.statistics import PERMUTATIONS, residuals, spearman_test

log = logging.getLogger(__name__)


# A correlation is reported only above both floors, an absolute n and a share of eligible concepts.
MIN_ABSOLUTE_N = 30

MIN_COVERAGE_FRACTION = 0.75

# Panels the floors left empty since the last pop_coverage_skips, summarized once per scope by the executor.
_coverage_skips: list[tuple[int, float]] = []


def note_coverage_skip(plot_name: str, n: int, n_total_relevant: int) -> None:
    """Record a panel left empty by the coverage floors, logged per panel at DEBUG only."""
    coverage = n / n_total_relevant if n_total_relevant else 0.0
    _coverage_skips.append((n, coverage))
    log.debug(f"  Skipping correlation for {plot_name}: n={n} covers {coverage:.0%} of "
              f"{n_total_relevant} relevant concepts (need >={MIN_ABSOLUTE_N} and >={MIN_COVERAGE_FRACTION:.0%}).")


def pop_coverage_skips() -> str | None:
    """One line on the panels skipped since the last call, None when there were none."""
    if not _coverage_skips:
        return None
    ns, coverages = zip(*_coverage_skips)
    _coverage_skips.clear()
    n_range = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)} to {max(ns)}"
    coverage_range = (f"{min(coverages):.0%}" if f"{min(coverages):.0%}" == f"{max(coverages):.0%}"
                      else f"{min(coverages):.0%} to {max(coverages):.0%}")
    return (f"{len(ns)} correlations below the coverage floor left without rho (n {n_range}, "
            f"coverage {coverage_range}, need >={MIN_ABSOLUTE_N} and >={MIN_COVERAGE_FRACTION:.0%})")


def score_column(column: str) -> str:
    """Column holding a variable's rank scores in a panel's rows."""
    return f"{column}_rank_z"


def residual_column(column: str) -> str:
    """Column holding a variable's deviations from its category mean in a panel's rows."""
    return f"{column}_residual"


def shown_column(column: str, row: dict) -> str:
    """Column a panel draws: the rank residuals of a partial, the deviations from the category means within
    categories, the raw values otherwise."""
    if row["controlling_for"]:
        return score_column(column)
    return residual_column(column) if row["ranks"] == "within category" else column


def regression_row(data: pd.DataFrame, x_col: str, y_col: str, plot_name: str, n_total_relevant: int,
                   within: bool = False, controls: tuple = ()) -> tuple[pd.DataFrame, dict]:
    """A panel's rows with their rank scores and drawn deviations, and its correlation_summary row, rho and p only
    above both floors."""
    clean_data = data.dropna(subset=[x_col, y_col, *controls, *(["category"] if within else [])]).copy()
    n = len(clean_data)
    coverage = n / n_total_relevant if n_total_relevant else 0.0
    covered = n >= MIN_ABSOLUTE_N and coverage >= MIN_COVERAGE_FRACTION
    row = {"plot_name": plot_name, "x_variable": x_col, "y_variable": y_col, "controlling_for": " + ".join(controls),
           "ranks": "within category" if within else "pooled", "spearman_rho": None, "spearman_p": None,
           "n_points": n, "n_total_relevant": n_total_relevant, "coverage_pct": round(100 * coverage, 1)}
    if not covered and plot_name:
        note_coverage_skip(plot_name, n, n_total_relevant)
    adjusted = within and not controls
    if n < 3:
        return clean_data.assign(**{name(column): np.nan for column in (x_col, y_col)
                                    for name in (score_column, *([residual_column] if adjusted else []))}), row
    result = spearman_test(clean_data[x_col], clean_data[y_col], [clean_data[c] for c in controls],
                           clean_data["category"] if within else None, n_draws=PERMUTATIONS if covered else 0)
    clean_data = clean_data.assign(**{score_column(x_col): result["x_scores"],
                                      score_column(y_col): result["y_scores"]})[result["used"]]
    if adjusted:
        clean_data = clean_data.assign(**{residual_column(column): residuals(clean_data[column], (),
                                                                             clean_data["category"])
                                          for column in (x_col, y_col)})
    if covered:
        row.update(spearman_rho=result["rho"], spearman_p=result["p"], n_points=result["n"])
    return clean_data, row


def value_axis_label(label: str, row: dict) -> str:
    """Axis label of a panel, naming on a second line the residual it draws."""
    within = row["ranks"] == "within category"
    if row["controlling_for"]:
        return f"{label}\nrank residual{' within category' if within else ''} (z-score)"
    return f"{label}\ndeviation from category mean" if within else label


def panel_detail(row: dict, wrap: bool = False) -> str:
    """Detail line of a panel, its coefficient with p and n, or the coverage it lacked, with wrap p and n on a second
    line."""
    if row["spearman_rho"] is None:
        return coverage_line(row["n_points"], row["n_total_relevant"])
    coefficient = "partial Spearman ρ" if row["controlling_for"] else "Spearman ρ"
    coefficient = (f"Within-category {coefficient}" if row["ranks"] == "within category"
                   else coefficient[0].upper() + coefficient[1:])
    line = stats_line(coefficient, row["spearman_rho"], row["spearman_p"], row["n_points"])
    return line.replace(", ", "\n", 1) if wrap else line


def draw_panel(ax, clean_data: pd.DataFrame, row: dict, x_label: str, y_label: str, title: str,
               size: float = 22, legend: bool = True, wrap: bool = False) -> None:
    """One panel and its least-squares line, the line drawn only when rho is reported."""
    handles = draw_regression(ax, clean_data[shown_column(row["x_variable"], row)],
                              clean_data[shown_column(row["y_variable"], row)], fit=row["spearman_rho"] is not None,
                              size=size)
    if legend and handles:
        ax.legend(handles=handles, loc="best")
    set_title(ax, title, panel_detail(row, wrap))
    ax.set_xlabel(value_axis_label(x_label, row))
    ax.set_ylabel(value_axis_label(y_label, row))


def run_regression_panel(
    data: pd.DataFrame, x_col: str, y_col: str,
    x_label: str, y_label: str, title: str,
    csv_out_path: pathlib.Path, png_out_path: pathlib.Path, n_total_relevant: int, within: bool = False,
) -> dict:
    """One panel plus its CSV, returns the correlation_summary row (rho empty below the floors)."""
    clean_data, summary_row = regression_row(data, x_col, y_col, png_out_path.stem, n_total_relevant, within)
    identifiers = [column for column in ("concept", "category") if column in clean_data.columns]
    derived = [name(column) for name in (score_column, residual_column) for column in (x_col, y_col)
               if name(column) in clean_data.columns]
    save_dataframe(clean_data[[*identifiers, x_col, y_col, *derived]], csv_out_path)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    draw_panel(ax, clean_data, summary_row, x_label, y_label, title)
    fig.tight_layout()
    fig.savefig(png_out_path, dpi=300)
    plt.close(fig)
    return summary_row


def panel_rho_columns(summary_df: pd.DataFrame) -> dict:
    """Correlation summary flattened to {rho_<plot_name>: spearman_rho} for sublayer_comparison."""
    if summary_df.empty or "plot_name" not in summary_df.columns:
        return {}
    return {f"rho_{row['plot_name']}": row.get("spearman_rho", np.nan) for _, row in summary_df.iterrows()}


class Panel(NamedTuple):
    """A panel of a grid sharing x, its summary row name None when another figure reports the coefficient."""
    plot_name: str | None
    key: str
    title: str
    y_col: str
    y_label: str
    n_total_relevant: int
    within: bool = False
    controls: tuple = ()


def _grid_panel(data: pd.DataFrame, x_col: str, panel: Panel) -> tuple[pd.DataFrame, dict]:
    """The rows and the correlation_summary row of one grid panel."""
    return regression_row(data, x_col, panel.y_col, panel.plot_name, panel.n_total_relevant, panel.within,
                          panel.controls)


def regression_grid_rows(data: pd.DataFrame, x_col: str, panels: list) -> list:
    """The correlation_summary rows of run_regression_grid, without drawing or writing anything."""
    rows = [_grid_panel(data, x_col, panel)[1] for panel in panels]
    return [row for row in rows if row["plot_name"]]


def run_regression_grid(data: pd.DataFrame, x_col: str, x_label: str, panels: list, title: str,
                        csv_out_path: pathlib.Path, png_out_path: pathlib.Path, order_by_rho: bool = False) -> list:
    """One row of panels sharing x and one CSV of their scores, returns the named panels' correlation_summary rows."""
    values = list(dict.fromkeys([x_col, *(column for panel in panels for column in (panel.y_col, *panel.controls))]))
    table = data[[column for column in ("concept", "category") if column in data.columns] + values].copy()
    drawn = []
    for panel in panels:
        clean, row = _grid_panel(data, x_col, panel)
        for column in (x_col, panel.y_col):
            for name in (score_column, residual_column):
                if name(column) in clean.columns:
                    table.loc[clean.index, f"{name(column)}_{panel.key}"] = clean[name(column)]
        drawn.append((clean, row, panel))
    save_dataframe(table, csv_out_path)
    rows = [row for _, row, _ in drawn if row["plot_name"]]
    if order_by_rho:
        drawn.sort(key=lambda item: np.inf if pd.isna(item[1]["spearman_rho"]) else item[1]["spearman_rho"])
    fig, axes = plt.subplots(1, len(drawn), figsize=(4.6 * len(drawn), 4.8), squeeze=False)
    for index, (axis, (clean, row, panel)) in enumerate(zip(axes[0], drawn)):
        draw_panel(axis, clean, row, x_label, panel.y_label, panel.title, size=14, legend=index == 0, wrap=True)
    set_suptitle(fig, title)
    fig.tight_layout()
    fig.savefig(png_out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows
