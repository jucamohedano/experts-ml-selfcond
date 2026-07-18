import logging
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils.helpers import save_dataframe

log = logging.getLogger(__name__)

LEVEL_PALETTE = {"Specific Concepts": "#D96A5B", "Broad Categories": "#4B5A6A"}

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

def _residualize(values: pd.Series, control: pd.Series) -> pd.Series:
    """
    Regress `values` on `control` with simple linear regression and return the residuals,
    i.e. the part of `values` left over once whatever it shares with `control` is removed.
    """
    slope, intercept, _, _, _ = stats.linregress(control, values)
    return values - (slope * control + intercept)

def plot_partial_correlation_jaccard_typicality(corr_data: pd.DataFrame, corr_dir) -> dict | None:
    """
    Partial correlation between Jaccard similarity and Human Typicality, controlling for
    Wikipedia frequency. Both jaccard_pct and human_typicality are residualized against
    log_frequency (removing whatever each variable shares with frequency), then the two
    residual vectors are correlated directly, isolating whatever relationship between
    Jaccard similarity and typicality frequency alone cannot explain.
    """
    clean_data = corr_data.dropna(subset=["jaccard_pct", "human_typicality", "log_frequency"]).copy()
    if len(clean_data) <= 2:
        return None

    clean_data["jaccard_resid"] = _residualize(clean_data["jaccard_pct"], clean_data["log_frequency"])
    clean_data["typicality_resid"] = _residualize(clean_data["human_typicality"], clean_data["log_frequency"])
    save_dataframe(
        clean_data[["concept", "category", "log_frequency", "jaccard_pct", "human_typicality", "jaccard_resid", "typicality_resid"]],
        corr_dir / "partial_correlation_jaccard_typicality.csv"
    )

    r, p = stats.pearsonr(clean_data["jaccard_resid"], clean_data["typicality_resid"])
    summary_row = {
        "plot_name": "partial_correlation_jaccard_typicality", "x_variable": "jaccard_resid",
        "y_variable": "typicality_resid", "pearson_r": r, "pearson_p": p,
        "n_points": len(clean_data), "controlling_for": "log_frequency",
    }

    plt.figure(figsize=(10.4, 9.1))
    sns.regplot(data=clean_data, x="typicality_resid", y="jaccard_resid",
                scatter_kws={'color': '#6a3d9a'}, line_kws={'color': 'red'})
    plt.title(f"Jaccard vs. Human Typicality, controlling for Frequency (partial r={r:.2f}, p={p:.2e})", fontsize=16)
    plt.xlabel("Human Typicality (residual after removing Frequency)")
    plt.ylabel("Jaccard Similarity Index % (residual after removing Frequency)")
    plt.tight_layout()
    plt.savefig(corr_dir / "partial_correlation_jaccard_typicality.png", dpi=300)
    plt.close()
    return summary_row

def plot_correlations(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame,
                      concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, corr_dir) -> None:
    """
    Generate scatter plots with regression lines for multiple correlation analyses.
    Tests relationships between: frequency vs expert count, human typicality vs expert count,
    frequency vs human typicality, (if available) human typicality/frequency vs Jaccard and
    Overlap similarity to the concept's category, and Shannon entropy vs expert count across
    both abstraction levels. Saves plots and a summary CSV with correlation statistics.
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
            corr_data, "human_typicality", "overlap_pct",
            "Human Typicality", "Overlap Coefficient %",
            "Human Typicality vs Overlap Coefficient % (r={r:.2f}, p={p:.2e})", '#8e44ad',
            corr_dir / "typicality_vs_overlap.csv", corr_dir / "typicality_vs_overlap.png")
        if row: summary_rows.append(row)

        row = _run_regression_panel(
            corr_data, "log_frequency", "jaccard_pct",
            "Wikipedia Frequency(Log10)", "Jaccard Similarity Index %",
            "Frequency(Log10) vs Jaccard Similarity % (r={r:.2f}, p={p:.2e})", '#ffa600',
            corr_dir / "frequency_vs_jaccard.csv", corr_dir / "frequency_vs_jaccard.png")
        if row: summary_rows.append(row)

        row = plot_partial_correlation_jaccard_typicality(corr_data, corr_dir)
        if row: summary_rows.append(row)

    if concept_entropy_df is not None and not concept_entropy_df.empty:
        row = plot_entropy_vs_expert_count(concept_entropy_df, category_entropy_df, corr_dir)
        if row: summary_rows.append(row)

    save_dataframe(pd.DataFrame(summary_rows), corr_dir / "correlation_summary.csv")


def plot_entropy_vs_expert_count(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, corr_dir) -> dict | None:
    """
    Shannon entropy vs expert count for BOTH abstraction levels on one panel. The
    concepts descriptor table contains every word with experts (category labels
    included), so words are split by membership in the category list: concepts as
    small dots, category labels as larger diamonds. The regression line and Pearson
    r/p are computed over all words pooled.
    """
    category_names = set(category_entropy_df['category']) if category_entropy_df is not None else set()
    data = concept_entropy_df.dropna(subset=['shannon_entropy', 'experts_count']).copy()
    data['group'] = np.where(data['concept'].isin(category_names), 'Broad Categories', 'Specific Concepts')
    save_dataframe(data[['concept', 'group', 'shannon_entropy', 'experts_count']],
                   corr_dir / "shannon_entropy_vs_expert_count.csv")
    if len(data) <= 2:
        return None

    r, p = stats.pearsonr(data['shannon_entropy'], data['experts_count'])
    plt.figure(figsize=(10.4, 9.1))
    for group, marker, size in (("Specific Concepts", 'o', 45), ("Broad Categories", 'D', 110)):
        group_data = data[data['group'] == group]
        plt.scatter(group_data['shannon_entropy'], group_data['experts_count'], s=size, marker=marker,
                    color=LEVEL_PALETTE[group], edgecolor='black', linewidth=0.4, alpha=0.75, label=group)
    sns.regplot(data=data, x='shannon_entropy', y='experts_count', scatter=False, line_kws={'color': 'red'})
    plt.title(f"Shannon Entropy vs Expert Count (r={r:.2f}, p={p:.2e})", fontsize=16)
    plt.xlabel("Shannon Entropy (Bits)")
    plt.ylabel("Expert Count")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(corr_dir / "shannon_entropy_vs_expert_count.png", dpi=300)
    plt.close()
    return {"plot_name": "shannon_entropy_vs_expert_count", "x_variable": "shannon_entropy",
            "y_variable": "experts_count", "pearson_r": r, "pearson_p": p, "n_points": len(data)}

def execute_module_4b_jaccard_vs_cosine_typicality(similarity_metrics_df: pd.DataFrame, global_typicality_df: pd.DataFrame, corr_dir) -> None:
    """
    Second pass of module 4, run after module 7 so global_cosine_typicality is available.
    Compares Jaccard similarity against Cosine Typicality (module 7's model-derived score) to
    test whether it tracks a concept's similarity to its category about as well as Human
    Typicality already does (see typicality_vs_jaccard in correlation_summary.csv). Appends
    its result to the same correlation_summary.csv module 4 already wrote, so both rows can be
    compared directly.
    """
    if similarity_metrics_df is None or similarity_metrics_df.empty or global_typicality_df is None or global_typicality_df.empty:
        log.warning("  Skipping Jaccard vs. Cosine Typicality: similarity or typicality data unavailable.")
        return

    merged = similarity_metrics_df.merge(global_typicality_df, on=["concept", "category"], how="inner")
    row = _run_regression_panel(
        merged, "global_cosine_typicality", "jaccard_pct",
        "Cosine Typicality", "Jaccard Similarity Index %",
        "Cosine Typicality vs Jaccard Similarity % (r={r:.2f}, p={p:.2e})", '#17becf',
        corr_dir / "jaccard_vs_cosine_typicality.csv", corr_dir / "jaccard_vs_cosine_typicality.png")
    if row is None:
        return

    summary_path = corr_dir / "correlation_summary.csv"
    existing_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    save_dataframe(pd.concat([existing_summary, pd.DataFrame([row])], ignore_index=True), summary_path)

def execute_module_4_correlations(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame,
                                  concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, corr_dir) -> None:
    """Execute Module 4: Correlations.
    Generates scatter plots with regression analysis for relationships between frequency,
    human typicality, expert count, entropy, and similarity metrics.
    """
    log.info("  Generating correlation plots...")
    plot_correlations(merged_metadata_df, similarity_metrics_df, concept_entropy_df, category_entropy_df, corr_dir)