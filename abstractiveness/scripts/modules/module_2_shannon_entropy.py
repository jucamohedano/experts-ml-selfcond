import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import entropy
from utils.helpers import save_dataframe, build_layer_probability_matrix
from utils.plot_helpers import _plot_bar_with_leaders, fig_width_for

log = logging.getLogger(__name__)

def compute_shannon_entropy(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, shan_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate Shannon entropy metrics for expert allocations per concept and category.
    Determines concentration of experts across layers (lower entropy = more concentrated).
    For each category, computes entropy under two definitions: the category-label distribution
    itself, and the average distribution of its member concepts; the Pearson correlation between
    these two distributions is also reported (pearson_correlation_distributions), along with a
    Mann-Whitney U test (logged, not returned) comparing concept-level vs. category-level entropy
    to test whether concepts are more sharply localized than categories.
    Returns DataFrames with entropy, peak layer, average layer, and expert count statistics for concepts and categories.
    """
    dist_matrix, prob_matrix = build_layer_probability_matrix(expert_allocation_df)
    layer_indices = prob_matrix.columns.values
    # Entropy for individual concepts
    concept_results = []
    for concept in prob_matrix.index:
        probs = prob_matrix.loc[concept].values
        if np.sum(probs) == 0: continue 
        
        ent = entropy(probs, base=2) 
        avg_layer = np.sum(probs * layer_indices)
        peak_layer = layer_indices[np.argmax(probs)]
        experts_count = dist_matrix.loc[concept].sum()
        
        concept_results.append({
            "concept": concept,
            "shannon_entropy": ent,
            "peak_layer": peak_layer,
            "avg_layer": avg_layer,
            "experts_count": experts_count
        })
    concept_entropy_df = pd.DataFrame(concept_results)
    save_dataframe(concept_entropy_df, shan_dir / "shannon_entropy_concepts.csv")
    
    # Entropy for categories based on two definitions: the category label itself and the average of its members
    category_results = []
    for category in concept_metadata['category'].dropna().unique():
        members = concept_metadata[concept_metadata['category'] == category]['concept'].tolist()
        valid_members = [m for m in members if m in prob_matrix.index]
        
        # Definition A: The Word Itself (Category Label)
        if category in prob_matrix.index:
            lbl_probs = prob_matrix.loc[category].values
            lbl_ent = entropy(lbl_probs, base=2)
            lbl_avg = np.sum(lbl_probs * layer_indices)
            lbl_peak = layer_indices[np.argmax(lbl_probs)]
        else:
            lbl_probs, lbl_ent, lbl_avg, lbl_peak = None, np.nan, np.nan, np.nan
            
        # Definition B: Average of the Category Members
        if valid_members:
            avg_probs = prob_matrix.loc[valid_members].mean(axis=0).values
            avg_ent = entropy(avg_probs, base=2)
            avg_layer_avg = np.sum(avg_probs * layer_indices)
            avg_peak_avg = layer_indices[np.argmax(avg_probs)]
        else:
            avg_probs, avg_ent, avg_layer_avg, avg_peak_avg = None, np.nan, np.nan, np.nan
            
        # Correlation between the two definitions
        if (lbl_probs is not None) and (avg_probs is not None):
            if np.std(lbl_probs) > 0 and np.std(avg_probs) > 0:
                dist_corr, _ = stats.pearsonr(lbl_probs, avg_probs)
            else:
                dist_corr = 0.0
        else:
            dist_corr = np.nan
            
        if valid_members or (lbl_probs is not None):
            category_results.append({
                "category": category,
                "shannon_entropy": lbl_ent,
                "peak_layer": lbl_peak,
                "avg_layer": lbl_avg,
                "shannon_entropy_average": avg_ent,
                "peak_layer_average": avg_peak_avg,
                "avg_layer_average": avg_layer_avg,
                "pearson_correlation_distributions": dist_corr,
                "member_count": len(valid_members)
            })
            
    category_entropy_df = pd.DataFrame(category_results)
    save_dataframe(category_entropy_df, shan_dir / "shannon_entropy_categories.csv")
    
    # Statistical Test: Mann-Whitney U Test to compare the distributions of Shannon entropy between Concepts and Categories
    clean_concept_ent = concept_entropy_df['shannon_entropy'].dropna()
    clean_category_ent = category_entropy_df['shannon_entropy'].dropna()
    
    if len(clean_category_ent) > 0 and len(clean_concept_ent) > 0:
        u_stat, p_val = stats.mannwhitneyu(clean_category_ent, clean_concept_ent, alternative='less')
        log.info(f"  [Stats] Mann-Whitney U Test (Categories < Concepts): U={u_stat:.1f}, p-value={p_val:.4e}")
        if p_val < 0.05:
            log.info("  [Stats] -> Hypothesis CONFIRMED: Categories are significantly more concentrated.")
        else:
            log.info("  [Stats] -> Hypothesis REJECTED: Difference is not statistically significant.")
    else:
        log.warning("  [Stats] Insufficient valid entropy data to perform Mann-Whitney U Test.")

    return concept_entropy_df, category_entropy_df

def plot_peak_average_distributions(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, shan_dir) -> None:
    """
    Generate combined visualization of peak expert layers (bar plot) and average layer distributions (KDE curves).
    Compares three groups: specific concepts, broad categories, and averaged category members.
    """
    layer_labels = expert_allocation_df['layer_name'].cat.categories.tolist()
    
    concept_subset = concept_entropy_df[['concept', 'peak_layer', 'avg_layer']].rename(columns={'concept': 'name'}).copy()
    concept_subset['group_type'] = 'Specific Concepts'
    category_subset = category_entropy_df[['category', 'peak_layer', 'avg_layer']].rename(columns={'category': 'name'}).copy()
    category_subset['group_type'] = 'Broad Categories'
    category_avg_subset = category_entropy_df[['category', 'peak_layer_average', 'avg_layer_average']].rename(
        columns={'category': 'name', 'peak_layer_average': 'peak_layer', 'avg_layer_average': 'avg_layer'}).copy()
    category_avg_subset['group_type'] = 'Broad Categories (Avg)'
    
    # Concatenate all three
    plot_df = pd.concat([concept_subset, category_subset, category_avg_subset], ignore_index=True).dropna(subset=['peak_layer', 'avg_layer'])
    # Convert to 0-based index for correct plotting alignment
    plot_df['peak_layer_idx'] = plot_df['peak_layer'] - 1
    plot_df['avg_layer_idx'] = plot_df['avg_layer'] - 1
    
    # Width scales with the layer count (matching module 1's layer plots) so the per-layer
    # x-axis stays legible; the anchored leader labels come from _plot_bar_with_leaders.
    fig, ax1 = plt.subplots(figsize=(fig_width_for(len(layer_labels), 0.28, min_w=16.0), 6))
    sns.set_theme(style="whitegrid")
    palette = {
        "Specific Concepts": "#D96A5B", 
        "Broad Categories": "#4B5A6A",
        "Broad Categories (Avg)": "#ECA926" 
    }

    group_counts = plot_df.groupby(['peak_layer_idx', 'group_type'], observed=False).size().reset_index(name='count')
    totals = plot_df['group_type'].value_counts()
    group_counts['percentage'] = group_counts.apply(
        lambda row: (row['count'] / totals[row['group_type']]) * 100, axis=1
    )
    _plot_bar_with_leaders(
        plot_dataframe=group_counts, x_col='peak_layer_idx', y_col='percentage',
        title="Distribution of Peak Expert Layers vs. Average Layer Allocation",
        x_label="Model Layer", y_label="% of Group Peaking in this Layer (Bars)", legend_title="Concept Grouping",
        hue='group_type', palette=palette, custom_tick_labels=layer_labels, ax=ax1, save=False,
        linewidth=0.8, alpha=0.9, order=range(len(layer_labels)), show_x_ticks=True
    )
    ax1.set_xlim(-0.5, len(layer_labels) - 0.5)
    ax2 = ax1.twinx() 
    sns.kdeplot(
        data=plot_df, x='avg_layer_idx', hue='group_type', 
        palette=palette, fill=True, alpha=0.2, linewidth=2,
        ax=ax2, legend=False, common_norm=False, 
        clip=(-0.5, len(layer_labels) - 0.5)
    )
    ax2.set_xlim(-0.5, len(layer_labels) - 0.5)
    ax2.set_ylabel("Density of Average Layer (Curves)", fontsize=11)
    ax2.grid(False) 
    
    plt.tight_layout()
    plt.savefig(shan_dir / "peak_average_layers.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def plot_shannon_entropies(concept_entropy_df: pd.DataFrame, category_entropy_df: pd.DataFrame, shan_dir) -> None:
    """
    Create violin plot comparing Shannon entropy distributions of specific concepts vs broad categories.
    Visualizes expert concentration differences with quartile annotations and individual data point overlay.
    """
    concept_subset = concept_entropy_df[['concept', 'shannon_entropy']].rename(columns={'concept': 'name'}).copy()
    concept_subset['group_type'] = 'Specific Concepts'
    category_subset = category_entropy_df[['category', 'shannon_entropy']].rename(columns={'category': 'name'}).copy()
    category_subset['group_type'] = 'Broad Categories'
    plot_df = pd.concat([concept_subset, category_subset], ignore_index=True).dropna(subset=['shannon_entropy'])
    
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    palette = {"Specific Concepts": "#D96A5B", "Broad Categories": "#4B5A6A"}
    # Create the Violin Plot
    ax = sns.violinplot(
        data=plot_df, 
        x='group_type', 
        y='shannon_entropy', 
        hue='group_type',
        legend=False,
        palette=palette,
        inner='quartile', # Draws dashed lines at the 25th, 50th (median), and 75th percentiles
        linewidth=1.5,
        alpha=0.8
    )
    # Overlay a Strip Plot for individual data points
    sns.stripplot(
        data=plot_df, 
        x='group_type', 
        y='shannon_entropy', 
        color='black', 
        alpha=0.4, 
        size=4,
        jitter=True, # Spreads the dots horizontally so they don't overlap completely
        ax=ax
    )
    # Add text labels for the quartile dotted lines
    groups = ['Specific Concepts', 'Broad Categories']
    for i, group in enumerate(groups):
        group_data = plot_df[plot_df['group_type'] == group]['shannon_entropy']
        if not group_data.empty:
            q1 = group_data.quantile(0.25)
            median = group_data.quantile(0.50)
            q3 = group_data.quantile(0.75)
            bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
            ax.text(i + 0.15, q1, 'Q1 (25%)', va='center', ha='left', fontsize=9, color='#333333', bbox=bbox_props)
            ax.text(i + 0.15, median, 'Median', va='center', ha='left', fontsize=9, fontweight='bold', color='#333333', bbox=bbox_props)
            ax.text(i + 0.15, q3, 'Q3 (75%)', va='center', ha='left', fontsize=9, color='#333333', bbox=bbox_props)
    ax.set_xlabel("Concept Grouping", fontsize=12, labelpad=10)
    ax.set_ylabel("Shannon Entropy (Bits)", fontsize=12, labelpad=10)
    
    plt.title("Shannon Entropy of Expert Allocations: Categories vs. Concepts", fontsize=14, pad=15)
    
    plt.tight_layout()
    save_path = shan_dir / "category_concept_shannon_entropies.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_category_entropies_bar(category_entropy_df: pd.DataFrame, shan_dir) -> None:
    """
    Generate bar chart of Shannon entropy per category label, sorted by entropy value.
    Shows which categories have most localized (low entropy) vs distributed (high entropy) experts.
    """
    entropy_data = category_entropy_df.dropna(subset=['shannon_entropy']).copy()
    entropy_data = entropy_data.sort_values(by='shannon_entropy', ascending=False).reset_index(drop=True)
    entropy_data['category_display'] = entropy_data['category'].str.title()
    out_path = shan_dir / "category_shannon_entropies_bar.png"
    _plot_bar_with_leaders(
        plot_dataframe=entropy_data, x_col="shannon_entropy", y_col="category_display",
        title="Shannon Entropy of Expert Allocations by Category", x_label="Shannon Entropy (Bits)",
        color="#4B5A6A", orient='h', out_path=out_path, figsize=(12, 14)
    )

def execute_module_2_shannon_entropy(formatted_expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, shan_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Module 2: Shannon Entropy Analysis and Peak/Average Layer Distributions.
    Computes entropy metrics, generates comparative visualizations, and returns entropy results for concepts and categories.
    """
    log.info("  Computing and plotting Shannon entropy analysis...")
    concept_entropy_df, category_entropy_df = compute_shannon_entropy(formatted_expert_allocation_df, concept_metadata, shan_dir)      
    log.info("  Generating peak vs. average layer distributions plot...")
    plot_peak_average_distributions(formatted_expert_allocation_df, concept_metadata, concept_entropy_df, category_entropy_df, shan_dir)
    log.info("  Generating Shannon entropy violin plots...")
    plot_shannon_entropies(concept_entropy_df, category_entropy_df, shan_dir)
    log.info("  Generating individual category entropy bar chart...")
    plot_category_entropies_bar(category_entropy_df, shan_dir)
    return concept_entropy_df, category_entropy_df