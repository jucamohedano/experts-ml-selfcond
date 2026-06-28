import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon, pdist
from utils.helpers import save_dataframe, build_layer_probability_matrix

log = logging.getLogger(__name__)

def compute_dual_category_jsd(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, jsd_dir) -> tuple[pd.DataFrame, list]:
    """
    Compute Jensen-Shannon Divergence between category label distributions and averaged member distributions.
    Also computes the internal diversity (average pairwise JSD) of the category members.
    Returns DataFrame with divergence metrics and layer labels for visualization.
    """
    layer_labels = expert_allocation_df['layer_name'].cat.categories.tolist()
    dist_matrix, prob_matrix = build_layer_probability_matrix(expert_allocation_df)
    
    results = []
    
    for category in concept_metadata['category'].dropna().unique():
        cat_meta = concept_metadata[concept_metadata['category'] == category]
        members = cat_meta['concept'].tolist()
        valid_members = [m for m in members if m in prob_matrix.index]
        
        if category in prob_matrix.index and valid_members:
            # P: Category Label distribution (Prototype)
            P = prob_matrix.loc[category].values
            
            # Q: Averaged Member distribution (Exemplars)
            Q = prob_matrix.loc[valid_members].mean(axis=0).values
            
            # Label vs. Centroid Divergence (JSD Math)
            js_distance = jensenshannon(P, Q, base=2)
            jsd = js_distance ** 2
            
            # Internal Category Diversity (Average Pairwise Member JSD)
            if len(valid_members) > 1:
                member_probs = prob_matrix.loc[valid_members].values
                # pdist calculates all unique pairwise combinations
                pairwise_js_dist = pdist(member_probs, metric=lambda u, v: jensenshannon(u, v, base=2))
                avg_member_jsd = np.mean(pairwise_js_dist ** 2)
            else:
                avg_member_jsd = np.nan
            
            # Additional Metrics for Visualizations
            p_ent = entropy(P, base=2)
            valid_meta = cat_meta[cat_meta['concept'].isin(valid_members)]
            avg_typ = valid_meta['human_typicality'].mean() if 'human_typicality' in valid_meta.columns else np.nan

            results.append({
                "category": category,
                "member_count": len(valid_members),
                "avg_human_typicality": avg_typ,
                "shannon_entropy_label": p_ent,
                "jensen_shannon_divergence": jsd,
                "avg_member_diversity_jsd": avg_member_jsd,
                "P_dist": P,
                "Q_dist": Q 
            })
            
    jsd_results_df = pd.DataFrame(results)
    
    if not jsd_results_df.empty:
        jsd_results_df = jsd_results_df.sort_values(by="jensen_shannon_divergence").reset_index(drop=True)
        
        # Save a clean CSV without the massive array columns
        clean_csv = jsd_results_df.drop(columns=['P_dist', 'Q_dist'])
    save_dataframe(clean_csv, jsd_dir / "dual_category_jsd.csv")
    # Return both the dataframe and the layer labels for the X-axis of the micro plot
    return jsd_results_df, layer_labels

def plot_jsd_vs_diversity_scatter(jsd_results_df: pd.DataFrame, jsd_dir) -> None:
    """
    Plots Label-Centroid Divergence (X-axis) against Internal Category Diversity (Y-axis).
    Tests if high member dispersion forces the model to create a distinct prototype label.
    """
    # Drop rows missing either of the two metrics we need
    df = jsd_results_df.dropna(subset=['jensen_shannon_divergence', 'avg_member_diversity_jsd']).copy()
    
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation for the title
    if len(df) > 2:
        r, p = stats.pearsonr(df['jensen_shannon_divergence'], df['avg_member_diversity_jsd'])
        title_suffix = f"(r={r:.2f}, p={p:.2e})"
    else:
        title_suffix = ""

    # regplot adds the scatter points and the linear regression line
    sns.regplot(
        data=df, 
        x="jensen_shannon_divergence", 
        y="avg_member_diversity_jsd", 
        scatter_kws={'color': '#1f77b4', 's': 80, 'edgecolor': 'black', 'alpha': 0.8}, 
        line_kws={'color':'#ff7f0e', 'linewidth': 2}
    )
    
    # Annotate each dot with its category name
    for i in range(df.shape[0]):
        plt.text(
            df['jensen_shannon_divergence'].iloc[i] + 0.005, # Slight right offset
            df['avg_member_diversity_jsd'].iloc[i], 
            df['category'].iloc[i].title(), 
            horizontalalignment='left', size='small', color='black', alpha=0.8
        )
        
    plt.title(f"Label-Centroid JSD vs. Internal Category Diversity {title_suffix}", fontsize=15, pad=15)
    plt.xlabel("Label vs. Averaged Members Divergence (JSD)\n← High Overlap (Exemplar strategy) | Low Overlap (Prototype strategy) →", fontsize=12)
    plt.ylabel("Average Pairwise Member Divergence (JSD)\n← High Cohesion | High Internal Diversity →", fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(jsd_dir / "scatter_jsd_vs_member_diversity.png", dpi=300)
    plt.close()

def plot_jsd_micro_distributions(jsd_results_df: pd.DataFrame, layer_labels: list, jsd_dir) -> None:
    """
    Visualization 2: Overlaid Area Charts for the Lowest and Highest JSD categories.
    For each of those two categories, plots P (the category-label prototype distribution)
    against Q (the average-of-members exemplar distribution) across layers, so the layer-wise
    allocations behind the smallest and largest JSD values can be inspected directly.
    """
    if len(jsd_results_df) < 2: return
    
    # Find the extremes
    min_cat = jsd_results_df.loc[jsd_results_df['jensen_shannon_divergence'].idxmin()]
    max_cat = jsd_results_df.loc[jsd_results_df['jensen_shannon_divergence'].idxmax()]
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    x = np.arange(len(layer_labels))
    
    targets = [
        (axes[0], min_cat, "Lowest Divergence (Highly Aligned)"),
        (axes[1], max_cat, "Highest Divergence (Highly Separated)")
    ]
    
    for ax, data, subtitle in targets:
        # Trace P (Prototype / Label)
        ax.fill_between(x, data['P_dist'], color="#D96A5B", alpha=0.4, label="Prototype (Category Label)")
        ax.plot(x, data['P_dist'], color="#D96A5B", linewidth=2)
        
        # Trace Q (Exemplar / Average of Members)
        ax.fill_between(x, data['Q_dist'], color="#4B5A6A", alpha=0.4, label="Exemplars (Avg. of Members)")
        ax.plot(x, data['Q_dist'], color="#4B5A6A", linewidth=2)
        
        cat_name = data['category'].title()
        jsd_val = data['jensen_shannon_divergence']
        
        ax.set_title(f"Category: '{cat_name}' | {subtitle} | JSD: {jsd_val:.4f}", fontsize=14)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.legend(loc="upper right")
        ax.set_xlim(0, len(layer_labels) - 1)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Format the shared X-axis
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=9)
    axes[1].set_xlabel("Model Layer", fontsize=12)
    
    plt.suptitle("Micro View: Layer Allocations Behind Jensen-Shannon Divergence", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(jsd_dir / "micro_jsd_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_jsd_vs_entropy_scatter(jsd_results_df: pd.DataFrame, jsd_dir) -> None:
    """Visualization 3: Scatter plot checking if concentrated concepts diverge more."""
    divergence_entropy_data = jsd_results_df.dropna(subset=['jensen_shannon_divergence', 'shannon_entropy_label']).copy()
    
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation for the title
    if len(divergence_entropy_data) > 2:
        r, p = stats.pearsonr(divergence_entropy_data['shannon_entropy_label'], divergence_entropy_data['jensen_shannon_divergence'])
        title_suffix = f"(r={r:.2f}, p={p:.2e})"
    else:
        title_suffix = ""

    # regplot automatically adds a line of best fit and confidence intervals
    sns.regplot(
        data=divergence_entropy_data, 
        x="shannon_entropy_label", 
        y="jensen_shannon_divergence", 
        scatter_kws={'color': '#2ca02c', 's': 70, 'edgecolor': 'black', 'alpha': 0.8}, 
        line_kws={'color':'#d62728', 'linewidth': 2}
    )
    
    # Add category text labels to the points so you know who is who
    for i in range(divergence_entropy_data.shape[0]):
        plt.text(
            divergence_entropy_data['shannon_entropy_label'].iloc[i] + 0.02, 
            divergence_entropy_data['jensen_shannon_divergence'].iloc[i], 
            divergence_entropy_data['category'].iloc[i], 
            horizontalalignment='left', size='small', color='black', alpha=0.7
        )
        
    plt.title(f"Label Entropy vs. Prototype/Exemplar Divergence {title_suffix}", fontsize=15, pad=15)
    plt.xlabel("Shannon Entropy of Category Label (Bits)\n← Concentrated | Uniform →", fontsize=12)
    plt.ylabel("Jensen-Shannon Divergence (Bits)\n← Aligned with Members | Diverged from Members →", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(jsd_dir / "scatter_jsd_vs_entropy.png", dpi=300)
    plt.close()

def execute_module_6_dual_category_jsd(formatted_expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, jsd_dir) -> tuple[pd.DataFrame, list]:
    """Execute Module 6: Dual Category Definitions (JSD).
    Compute and visualize Jensen-Shannon Divergence between category prototypes and exemplars.
    """
    log.info("  Computing Dual Category Definitions (JSD) and Internal Member Diversity...")
    jsd_results_df, jsd_layer_labels = compute_dual_category_jsd(formatted_expert_allocation_df, concept_metadata, jsd_dir)
    log.info("  Generating JSD visualisations...")
    plot_jsd_vs_diversity_scatter(jsd_results_df, jsd_dir)
    plot_jsd_micro_distributions(jsd_results_df, jsd_layer_labels, jsd_dir)
    plot_jsd_vs_entropy_scatter(jsd_results_df, jsd_dir)
    return jsd_results_df, jsd_layer_labels