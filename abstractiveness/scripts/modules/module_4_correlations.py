import logging
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils.helpers import save_dataframe

log = logging.getLogger(__name__)

def _run_regression_panel(
    data: pd.DataFrame, x_col: str, y_col: str,
    x_label: str, y_label: str, title_template: str, scatter_color: str,
    csv_out_path: pathlib.Path, png_out_path: pathlib.Path,
) -> dict | None:
    """
    Run one regression panel: drop rows missing x_col/y_col, save the cleaned (x_col, y_col)
    subset to CSV, then draw a scatter plot with a linear regression line (annotated with the
    Pearson r/p in the title) if there are more than 2 points, and save the PNG. The figure is
    always opened and saved/closed within this single call, so every panel's CSV and PNG are
    written together regardless of whether the caller's data was available.
    Returns a summary-row dict (plot_name/x_variable/y_variable/pearson_r/pearson_p/n_points)
    if n>2, else None.
    """
    plot_name = png_out_path.stem
    clean_data = data.dropna(subset=[x_col, y_col])
    save_dataframe(clean_data[[x_col, y_col]], csv_out_path)

    plt.figure(figsize=(10.4, 9.1))
    summary_row = None
    if len(clean_data) > 2:
        r, p = stats.pearsonr(clean_data[x_col], clean_data[y_col])
        summary_row = {
            "plot_name": plot_name, "x_variable": x_col, "y_variable": y_col,
            "pearson_r": r, "pearson_p": p, "n_points": len(clean_data),
        }
        sns.regplot(data=clean_data, x=x_col, y=y_col, scatter_kws={'color': scatter_color}, line_kws={'color': 'red'})
        plt.title(title_template.format(r=r, p=p), fontsize=16)
    plt.xlabel(x_label); plt.ylabel(y_label)
    plt.tight_layout(); plt.savefig(png_out_path, dpi=300); plt.close()
    return summary_row

def plot_correlations(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame, corr_dir) -> None:
    """
    Generate scatter plots with regression lines for multiple correlation analyses.
    Tests relationships between: frequency vs expert count, human typicality vs expert count,
    frequency vs human typicality, and (if available) human typicality/frequency vs Jaccard
    similarity to the concept's category. Saves plots and a summary CSV with correlation statistics.
    """
    summary_rows = []

    row = _run_regression_panel(
        merged_metadata_df, "log_frequency", "expert_count",
        "Wikipedia Frequency(Log10)", "Expert Count",
        "Frequency(Log10) vs Expert Count (r={r:.2f}, p={p:.2e})", '#444e86',
        corr_dir / "frequency_vs_expert_count.csv", corr_dir / "frequency_vs_expert_count.png")
    if row: summary_rows.append(row)

    row = _run_regression_panel(
        merged_metadata_df, "human_typicality", "expert_count",
        "Human Typicality", "Expert Count",
        "Human Typicality vs Expert Count (r={r:.2f}, p={p:.2e})", '#955196',
        corr_dir / "typicality_vs_expert_count.csv", corr_dir / "typicality_vs_expert_count.png")
    if row: summary_rows.append(row)

    row = _run_regression_panel(
        merged_metadata_df, "log_frequency", "human_typicality",
        "Wikipedia Frequency(Log10)", "Human Typicality",
        "Frequency(Log10) vs Human Typicality (r={r:.2f}, p={p:.2e})", '#2ca02c',
        corr_dir / "frequency_vs_typicality.csv", corr_dir / "frequency_vs_typicality.png")
    if row: summary_rows.append(row)

    if similarity_metrics_df is not None and not similarity_metrics_df.empty:
        # Merge the metadata with the similarity dataframe
        corr_data = merged_metadata_df.merge(similarity_metrics_df, on=["concept", "category"], how="inner")

        row = _run_regression_panel(
            corr_data, "human_typicality", "jaccard_pct",
            "Human Typicality", "Jaccard Similarity Index %",
            "Human Typicality vs Jaccard Similarity % (r={r:.2f}, p={p:.2e})", '#ff7c43',
            corr_dir / "typicality_vs_jaccard.csv", corr_dir / "typicality_vs_jaccard.png")
        if row: summary_rows.append(row)

        row = _run_regression_panel(
            corr_data, "log_frequency", "jaccard_pct",
            "Wikipedia Frequency(Log10)", "Jaccard Similarity Index %",
            "Frequency(Log10) vs Jaccard Similarity % (r={r:.2f}, p={p:.2e})", '#ffa600',
            corr_dir / "frequency_vs_jaccard.csv", corr_dir / "frequency_vs_jaccard.png")
        if row: summary_rows.append(row)

    save_dataframe(pd.DataFrame(summary_rows), corr_dir / "correlation_summary.csv")

def execute_module_4_correlations(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame, corr_dir) -> None:
    """Execute Module 4: Correlations.
    Generates scatter plots with regression analysis for relationships between frequency, human typicality, expert count, and similarity metrics.
    """
    log.info("  Generating correlation plots...")
    plot_correlations(merged_metadata_df, similarity_metrics_df, corr_dir)