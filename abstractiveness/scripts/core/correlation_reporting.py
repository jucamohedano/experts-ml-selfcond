"""
Scatter-with-fit panels and their correlation_summary rows, gated by coverage floors.

Used by the correlation sections of modules 1 and 2.

Explanations: documentation/module_1_expert_distribution.md and module_2_similarities.md.
"""

import logging
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from utils.logging_and_io import save_dataframe
from utils.plotting import draw_regression, set_title, set_suptitle, stats_line, coverage_line

log = logging.getLogger(__name__)


# A correlation is reported only above both floors, an absolute n and a share of eligible concepts.
MIN_ABSOLUTE_N = 30

MIN_COVERAGE_FRACTION = 0.75


def regression_row(data: pd.DataFrame, x_col: str, y_col: str, plot_name: str,
                   n_total_relevant: int) -> tuple[pd.DataFrame, dict]:
    """The rows a panel uses and its correlation_summary row, Pearson r only above both floors."""
    clean_data = data.dropna(subset=[x_col, y_col])
    n = len(clean_data)
    coverage = n / n_total_relevant if n_total_relevant else 0.0
    row = {"plot_name": plot_name, "x_variable": x_col, "y_variable": y_col,
           "pearson_r": None, "pearson_p": None, "n_points": n,
           "n_total_relevant": n_total_relevant, "coverage_pct": round(100 * coverage, 1)}
    if n >= MIN_ABSOLUTE_N and coverage >= MIN_COVERAGE_FRACTION:
        row["pearson_r"], row["pearson_p"] = stats.pearsonr(clean_data[x_col], clean_data[y_col])
    else:
        log.warning(f"  Skipping correlation for {plot_name}: n={n} covers {coverage:.0%} of "
                    f"{n_total_relevant} relevant concepts (need >={MIN_ABSOLUTE_N} and >={MIN_COVERAGE_FRACTION:.0%}).")
    return clean_data, row


def run_regression_panel(
    data: pd.DataFrame, x_col: str, y_col: str,
    x_label: str, y_label: str, title: str,
    csv_out_path: pathlib.Path, png_out_path: pathlib.Path, n_total_relevant: int,
) -> dict:
    """Scatter with Pearson fit plus its CSV, returns the correlation_summary row (r empty below the floors)."""
    plot_name = png_out_path.stem
    clean_data, summary_row = regression_row(data, x_col, y_col, plot_name, n_total_relevant)
    save_dataframe(clean_data[[x_col, y_col]], csv_out_path)
    n = summary_row["n_points"]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fitted = summary_row["pearson_r"] is not None
    draw_regression(ax, clean_data, x_col, y_col, fit=fitted)
    set_title(ax, title, stats_line("Pearson r", summary_row["pearson_r"], summary_row["pearson_p"], n)
              if fitted else coverage_line(n, n_total_relevant))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(png_out_path, dpi=300)
    plt.close(fig)
    return summary_row


def panel_r_columns(summary_df: pd.DataFrame) -> dict:
    """Correlation summary flattened to {r_<plot_name>: pearson_r} for sublayer_comparison."""
    if summary_df.empty or "plot_name" not in summary_df.columns:
        return {}
    return {f"r_{row['plot_name']}": row.get("pearson_r", np.nan) for _, row in summary_df.iterrows()}


def regression_grid_rows(data: pd.DataFrame, x_col: str, x_name: str, panels: list) -> list:
    """The correlation_summary rows of run_regression_grid, without drawing or writing anything."""
    return [regression_row(data, x_col, y_col, f"{x_name}_vs_{y_name}", n_total)[1]
            for y_col, _, y_name, n_total in panels]


def run_regression_grid(data: pd.DataFrame, x_col: str, x_label: str, x_name: str, panels: list,
                        title: str, csv_out_path: pathlib.Path, png_out_path: pathlib.Path) -> list:
    """One row of regression panels sharing an x variable, returns one correlation_summary row per panel.

    panels holds (y_col, y_label, y_name, n_total_relevant). Each panel drops only its own missing
    rows and applies the same coverage floors as run_regression_panel.
    """
    y_cols = [y_col for y_col, _, _, _ in panels]
    save_dataframe(data[["concept", "category", x_col] + y_cols], csv_out_path)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.8), squeeze=False)
    rows = []
    for axis, (y_col, y_label, y_name, n_total_relevant) in zip(axes[0], panels):
        clean, row = regression_row(data, x_col, y_col, f"{x_name}_vs_{y_name}", n_total_relevant)
        n = row["n_points"]
        fitted = row["pearson_r"] is not None
        draw_regression(axis, clean, x_col, y_col, fit=fitted, size=14)
        set_title(axis, y_label, stats_line("Pearson r", row["pearson_r"], row["pearson_p"], n)
                  if fitted else coverage_line(n, n_total_relevant))
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        rows.append(row)
    set_suptitle(fig, title)
    fig.tight_layout()
    fig.savefig(png_out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows
