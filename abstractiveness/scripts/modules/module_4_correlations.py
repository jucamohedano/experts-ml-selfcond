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

# A correlation is only reported when BOTH hold: an absolute floor (below this, the
# standard error of r is too wide to trust regardless of how complete the data is) and
# a coverage floor (the retained rows must be the vast majority of the concepts that
# could in principle have contributed, not just "more than two"). n_total_relevant is
# a fixed count from concept_metadata (see plot_correlations), so coverage falls as
# concepts drop out at stricter AP thresholds, making silent data loss visible instead
# of just shrinking n_points with no reference point.
MIN_ABSOLUTE_N = 30
MIN_COVERAGE_FRACTION = 0.75

def _run_regression_panel(
    data: pd.DataFrame, x_col: str, y_col: str,
    x_label: str, y_label: str, title_template: str, scatter_color: str,
    csv_out_path: pathlib.Path, png_out_path: pathlib.Path, n_total_relevant: int,
) -> dict:
    """
    Run one regression panel: drop rows missing x_col/y_col, save the cleaned (x_col, y_col)
    subset to CSV, then draw a scatter plot and save the PNG. The figure is always opened and
    saved/closed within this single call, so every panel's CSV and PNG are always written
    together regardless of whether the caller's data was available.

    The Pearson statistic and the fitted regression line are only computed when the
    retained rows clear BOTH MIN_ABSOLUTE_N and MIN_COVERAGE_FRACTION of n_total_relevant
    (see module constants). Below that, a correlation from a handful of points, or from a
    small and potentially unrepresentative slice of the concept set, is not reported: the
    scatter is still drawn (unfitted) so the sparse data remains visible, titled with the
    coverage shortfall instead of a misleading r/p.

    Always returns a summary-row dict (plot_name, x_variable, y_variable, pearson_r,
    pearson_p, n_points, n_total_relevant, coverage_pct), with pearson_r/pearson_p left
    as None when the panel did not clear the reporting threshold, so every panel leaves a
    row in correlation_summary.csv explaining why, rather than silently vanishing.
    """
    plot_name = png_out_path.stem
    clean_data = data.dropna(subset=[x_col, y_col])
    save_dataframe(clean_data[[x_col, y_col]], csv_out_path)

    n = len(clean_data)
    coverage = n / n_total_relevant if n_total_relevant else 0.0
    summary_row = {
        "plot_name": plot_name, "x_variable": x_col, "y_variable": y_col,
        "pearson_r": None, "pearson_p": None, "n_points": n,
        "n_total_relevant": n_total_relevant, "coverage_pct": round(100 * coverage, 1),
    }

    plt.figure(figsize=(10.4, 9.1))
    if n >= MIN_ABSOLUTE_N and coverage >= MIN_COVERAGE_FRACTION:
        r, p = stats.pearsonr(clean_data[x_col], clean_data[y_col])
        summary_row["pearson_r"], summary_row["pearson_p"] = r, p
        sns.regplot(data=clean_data, x=x_col, y=y_col, scatter_kws={'color': scatter_color}, line_kws={'color': 'red'})
        plt.title(title_template.format(r=r, p=p), fontsize=16)
    else:
        log.warning(f"  Skipping correlation for {plot_name}: n={n} covers {coverage:.0%} of "
                    f"{n_total_relevant} relevant concepts (need >={MIN_ABSOLUTE_N} and >={MIN_COVERAGE_FRACTION:.0%}).")
        sns.scatterplot(data=clean_data, x=x_col, y=y_col, color=scatter_color)
        plt.title(f"{plot_name}: insufficient coverage (n={n}, {coverage:.0%} of {n_total_relevant})", fontsize=13)
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

def plot_partial_correlation_jaccard_typicality(corr_data: pd.DataFrame, corr_dir, n_total_relevant: int) -> dict:
    """
    Partial correlation between Jaccard similarity and Human Typicality, controlling for
    Wikipedia frequency. Both jaccard_pct and human_typicality are residualized against
    log_frequency (removing whatever each variable shares with frequency), then the two
    residual vectors are correlated directly, isolating whatever relationship between
    Jaccard similarity and typicality frequency alone cannot explain.

    Reports the partial correlation only when the retained rows clear both MIN_ABSOLUTE_N
    and MIN_COVERAGE_FRACTION of n_total_relevant (see module constants), same rule as
    _run_regression_panel; always returns a summary row.
    """
    clean_data = corr_data.dropna(subset=["jaccard_pct", "human_typicality", "log_frequency"]).copy()
    n = len(clean_data)
    coverage = n / n_total_relevant if n_total_relevant else 0.0
    summary_row = {
        "plot_name": "partial_correlation_jaccard_typicality", "x_variable": "jaccard_resid",
        "y_variable": "typicality_resid", "pearson_r": None, "pearson_p": None,
        "n_points": n, "n_total_relevant": n_total_relevant, "coverage_pct": round(100 * coverage, 1),
        "controlling_for": "log_frequency",
    }
    if n < MIN_ABSOLUTE_N or coverage < MIN_COVERAGE_FRACTION:
        log.warning(f"  Skipping partial_correlation_jaccard_typicality: n={n} covers {coverage:.0%} of "
                    f"{n_total_relevant} relevant concepts (need >={MIN_ABSOLUTE_N} and >={MIN_COVERAGE_FRACTION:.0%}).")
        return summary_row

    clean_data["jaccard_resid"] = _residualize(clean_data["jaccard_pct"], clean_data["log_frequency"])
    clean_data["typicality_resid"] = _residualize(clean_data["human_typicality"], clean_data["log_frequency"])
    save_dataframe(
        clean_data[["concept", "category", "log_frequency", "jaccard_pct", "human_typicality", "jaccard_resid", "typicality_resid"]],
        corr_dir / "partial_correlation_jaccard_typicality.csv"
    )

    r, p = stats.pearsonr(clean_data["jaccard_resid"], clean_data["typicality_resid"])
    summary_row["pearson_r"], summary_row["pearson_p"] = r, p

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
                      concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame,
                      concept_metadata: pd.DataFrame, corr_dir) -> None:
    """
    Generate scatter plots with regression lines for multiple correlation analyses.
    Tests relationships between: frequency vs expert count, human typicality vs expert count,
    frequency vs human typicality, (if available) human typicality/frequency vs Jaccard and
    Overlap similarity to the concept's category, and Shannon entropy vs expert count across
    both abstraction levels. Saves plots and a summary CSV with correlation statistics.

    Every panel is judged against a coverage denominator (n_total_relevant): the fixed count
    of concepts in concept_metadata that could in principle contribute to that panel,
    independent of the AP threshold. Panels involving Jaccard/overlap/category alignment use
    only categorized concepts (root category-label words can never get a row there, by
    construction in module 3), everything else uses the full metadata. A correlation is only
    computed and reported when the retained rows clear both an absolute floor and a coverage
    floor against this denominator (see MIN_ABSOLUTE_N / MIN_COVERAGE_FRACTION); otherwise the
    summary row records why (n_points, n_total_relevant, coverage_pct) with pearson_r/p left
    empty, so incomplete panels stay visible instead of silently vanishing from the summary.
    """
    summary_rows = []
    n_total_words = len(concept_metadata)
    n_total_categorized = concept_metadata['category'].notna().sum()

    summary_rows.append(_run_regression_panel(
        merged_metadata_df, "log_frequency", "expert_count",
        "Wikipedia Frequency(Log10)", "Expert Count",
        "Frequency(Log10) vs Expert Count (r={r:.2f}, p={p:.2e})", '#444e86',
        corr_dir / "frequency_vs_expert_count.csv", corr_dir / "frequency_vs_expert_count.png", n_total_words))

    summary_rows.append(_run_regression_panel(
        merged_metadata_df, "human_typicality", "expert_count",
        "Human Typicality", "Expert Count",
        "Human Typicality vs Expert Count (r={r:.2f}, p={p:.2e})", '#955196',
        corr_dir / "typicality_vs_expert_count.csv", corr_dir / "typicality_vs_expert_count.png", n_total_words))

    summary_rows.append(_run_regression_panel(
        merged_metadata_df, "log_frequency", "human_typicality",
        "Wikipedia Frequency(Log10)", "Human Typicality",
        "Frequency(Log10) vs Human Typicality (r={r:.2f}, p={p:.2e})", '#2ca02c',
        corr_dir / "frequency_vs_typicality.csv", corr_dir / "frequency_vs_typicality.png", n_total_words))

    if similarity_metrics_df is not None and not similarity_metrics_df.empty:
        # Merge the metadata with the similarity dataframe
        corr_data = merged_metadata_df.merge(similarity_metrics_df, on=["concept", "category"], how="inner")

        summary_rows.append(_run_regression_panel(
            corr_data, "human_typicality", "jaccard_pct",
            "Human Typicality", "Jaccard Similarity Index %",
            "Human Typicality vs Jaccard Similarity % (r={r:.2f}, p={p:.2e})", '#ff7c43',
            corr_dir / "typicality_vs_jaccard.csv", corr_dir / "typicality_vs_jaccard.png", n_total_categorized))

        summary_rows.append(_run_regression_panel(
            corr_data, "human_typicality", "overlap_pct",
            "Human Typicality", "Overlap Coefficient %",
            "Human Typicality vs Overlap Coefficient % (r={r:.2f}, p={p:.2e})", '#8e44ad',
            corr_dir / "typicality_vs_overlap.csv", corr_dir / "typicality_vs_overlap.png", n_total_categorized))

        summary_rows.append(_run_regression_panel(
            corr_data, "log_frequency", "jaccard_pct",
            "Wikipedia Frequency(Log10)", "Jaccard Similarity Index %",
            "Frequency(Log10) vs Jaccard Similarity % (r={r:.2f}, p={p:.2e})", '#ffa600',
            corr_dir / "frequency_vs_jaccard.csv", corr_dir / "frequency_vs_jaccard.png", n_total_categorized))

        summary_rows.append(plot_partial_correlation_jaccard_typicality(corr_data, corr_dir, n_total_categorized))

    if concept_entropy_df is not None and not concept_entropy_df.empty:
        summary_rows.append(plot_entropy_vs_expert_count(concept_entropy_df, category_entropy_df, corr_dir, n_total_words))

    save_dataframe(pd.DataFrame(summary_rows), corr_dir / "correlation_summary.csv")


def plot_entropy_vs_expert_count(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame,
                                 corr_dir, n_total_relevant: int) -> dict:
    """
    Shannon entropy vs expert count for BOTH abstraction levels on one panel. The
    concepts descriptor table contains every word with experts (category labels
    included), so words are split by membership in the category list: concepts as
    small dots, category labels as larger diamonds. The regression line and Pearson
    r/p are computed over all words pooled, and only reported once coverage clears
    MIN_ABSOLUTE_N / MIN_COVERAGE_FRACTION against n_total_relevant (see module constants).
    """
    category_names = set(category_entropy_df['category']) if category_entropy_df is not None else set()
    data = concept_entropy_df.dropna(subset=['shannon_entropy', 'experts_count']).copy()
    data['group'] = np.where(data['concept'].isin(category_names), 'Broad Categories', 'Specific Concepts')
    save_dataframe(data[['concept', 'group', 'shannon_entropy', 'experts_count']],
                   corr_dir / "shannon_entropy_vs_expert_count.csv")

    n = len(data)
    coverage = n / n_total_relevant if n_total_relevant else 0.0
    summary_row = {
        "plot_name": "shannon_entropy_vs_expert_count", "x_variable": "shannon_entropy",
        "y_variable": "experts_count", "pearson_r": None, "pearson_p": None,
        "n_points": n, "n_total_relevant": n_total_relevant, "coverage_pct": round(100 * coverage, 1),
    }
    plt.figure(figsize=(10.4, 9.1))
    for group, marker, size in (("Specific Concepts", 'o', 45), ("Broad Categories", 'D', 110)):
        group_data = data[data['group'] == group]
        plt.scatter(group_data['shannon_entropy'], group_data['experts_count'], s=size, marker=marker,
                    color=LEVEL_PALETTE[group], edgecolor='black', linewidth=0.4, alpha=0.75, label=group)
    if n >= MIN_ABSOLUTE_N and coverage >= MIN_COVERAGE_FRACTION:
        r, p = stats.pearsonr(data['shannon_entropy'], data['experts_count'])
        summary_row["pearson_r"], summary_row["pearson_p"] = r, p
        sns.regplot(data=data, x='shannon_entropy', y='experts_count', scatter=False, line_kws={'color': 'red'})
        plt.title(f"Shannon Entropy vs Expert Count (r={r:.2f}, p={p:.2e})", fontsize=16)
    else:
        log.warning(f"  Skipping correlation for shannon_entropy_vs_expert_count: n={n} covers {coverage:.0%} "
                    f"of {n_total_relevant} relevant concepts (need >={MIN_ABSOLUTE_N} and >={MIN_COVERAGE_FRACTION:.0%}).")
        plt.title(f"Shannon Entropy vs Expert Count: insufficient coverage (n={n}, {coverage:.0%} of {n_total_relevant})", fontsize=13)
    plt.xlabel("Shannon Entropy (Bits)")
    plt.ylabel("Expert Count")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(corr_dir / "shannon_entropy_vs_expert_count.png", dpi=300)
    plt.close()
    return summary_row

def execute_module_4b_jaccard_vs_cosine_typicality(similarity_metrics_df: pd.DataFrame, global_typicality_df: pd.DataFrame,
                                                   concept_metadata: pd.DataFrame, corr_dir) -> None:
    """
    Second pass of module 4, run after module 7 so global_cosine_typicality is available.
    Compares Jaccard similarity against Cosine Typicality (module 7's model-derived score) to
    test whether it tracks a concept's similarity to its category about as well as Human
    Typicality already does (see typicality_vs_jaccard in correlation_summary.csv). Appends
    its result to the same correlation_summary.csv module 4 already wrote, so both rows can be
    compared directly. Coverage denominator: categorized concepts, same convention as the
    other Jaccard-based panels (module 7 additionally requires >=2 valid members per category,
    which this denominator does not know about, so its coverage reads slightly conservative
    rather than overstated).
    """
    if similarity_metrics_df is None or similarity_metrics_df.empty or global_typicality_df is None or global_typicality_df.empty:
        log.warning("  Skipping Jaccard vs. Cosine Typicality: similarity or typicality data unavailable.")
        return

    n_total_categorized = concept_metadata['category'].notna().sum()
    merged = similarity_metrics_df.merge(global_typicality_df, on=["concept", "category"], how="inner")
    row = _run_regression_panel(
        merged, "global_cosine_typicality", "jaccard_pct",
        "Cosine Typicality", "Jaccard Similarity Index %",
        "Cosine Typicality vs Jaccard Similarity % (r={r:.2f}, p={p:.2e})", '#17becf',
        corr_dir / "jaccard_vs_cosine_typicality.csv", corr_dir / "jaccard_vs_cosine_typicality.png", n_total_categorized)

    summary_path = corr_dir / "correlation_summary.csv"
    existing_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    save_dataframe(pd.concat([existing_summary, pd.DataFrame([row])], ignore_index=True), summary_path)

def execute_module_4_correlations(merged_metadata_df: pd.DataFrame, similarity_metrics_df: pd.DataFrame,
                                  concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame,
                                  concept_metadata: pd.DataFrame, corr_dir) -> None:
    """Execute Module 4: Correlations.
    Generates scatter plots with regression analysis for relationships between frequency,
    human typicality, expert count, entropy, and similarity metrics.
    """
    log.info("  Generating correlation plots...")
    plot_correlations(merged_metadata_df, similarity_metrics_df, concept_entropy_df, category_entropy_df, concept_metadata, corr_dir)