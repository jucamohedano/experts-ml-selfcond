import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from utils.helpers import save_dataframe
from utils.plot_helpers import _plot_bar_chart

log = logging.getLogger(__name__)

def compute_empirical_typicality(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, typ_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Centroid-based cosine typicality of expert allocations.
    Calculates both a Global Prototype (all layers) and Per-Layer Prototypes.
    Returns DataFrames with cosine similarity scores for global and per-layer comparisons.
    """ 
    expert_allocation_df['present'] = 1 
    ordered_layers = expert_allocation_df[['layer_idx', 'layer_name']].drop_duplicates().sort_values('layer_idx')
    global_results = []
    layer_results = []
    
    for category in concept_metadata['category'].dropna().unique():
        cat_meta = concept_metadata[concept_metadata['category'] == category]
        members = cat_meta['concept'].tolist()
        valid_members = [m for m in members if m in expert_allocation_df['concept'].values]
        if len(valid_members) < 2:
            continue  
        category_allocation_df = expert_allocation_df[expert_allocation_df['concept'].isin(valid_members)].copy()
        
        # Global Prototype (Averaged space across ALL layers)
        category_allocation_df['layer_unit'] = category_allocation_df['layer_name'].astype(str) + "_" + category_allocation_df['unit'].astype(str)
        global_pivot = category_allocation_df.pivot_table(index='concept', columns='layer_unit', values='present', fill_value=0)
        global_pivot = global_pivot.reindex(valid_members, fill_value=0)
        
        if not global_pivot.empty:
            # Calculate Global Prototype (Centroid)
            centroid_global = global_pivot.mean(axis=0).values.reshape(1, -1)
            # Compare each member to the Prototype
            sims_global = cosine_similarity(global_pivot.values, centroid_global).flatten()
            
            for i, concept in enumerate(global_pivot.index):
                h_typ_arr = cat_meta[cat_meta['concept'] == concept]['human_typicality'].values
                h_typ = h_typ_arr[0] if len(h_typ_arr) > 0 else np.nan
                global_results.append({
                    'category': category,
                    'concept': concept,
                    'human_typicality': h_typ,
                    'global_cosine_typicality': sims_global[i]
                })

        # Per-Layer Prototype 
        for _, row in ordered_layers.iterrows():
            l_idx = row['layer_idx']
            l_name = row['layer_name']
            layer_df_sub = category_allocation_df[category_allocation_df['layer_idx'] == l_idx]
            layer_pivot = layer_df_sub.pivot_table(index='concept', columns='unit', values='present', fill_value=0)
            layer_pivot = layer_pivot.reindex(valid_members, fill_value=0)
            
            if layer_pivot.shape[1] > 0:
                centroid_layer = layer_pivot.mean(axis=0).values.reshape(1, -1)
                sims_layer = cosine_similarity(layer_pivot.values, centroid_layer).flatten()
            else:
                sims_layer = np.zeros(len(valid_members))
                
            for i, concept in enumerate(layer_pivot.index):
                h_typ_arr = cat_meta[cat_meta['concept'] == concept]['human_typicality'].values
                h_typ = h_typ_arr[0] if len(h_typ_arr) > 0 else np.nan
                layer_results.append({
                    'category': category,
                    'concept': concept,
                    'layer_idx': l_idx,
                    'layer_name': l_name,
                    'human_typicality': h_typ,
                    'layer_cosine_typicality': sims_layer[i]
                })
    global_typicality_df = pd.DataFrame(global_results)
    layer_typicality_df = pd.DataFrame(layer_results)
    save_dataframe(global_typicality_df, typ_dir / "global_prototype_typicality.csv")
    save_dataframe(layer_typicality_df, typ_dir / "layer_prototype_typicality.csv")
    return global_typicality_df, layer_typicality_df

def generate_category_typicality_reports(global_typicality_df: pd.DataFrame, layer_typicality_df: pd.DataFrame, typ_dir) -> None:
    """
    Create isolated reports for each category with double bar plots, correlation scatter plots, 
    and CSV tracking the most typical word per layer.
    """
    for category in global_typicality_df['category'].unique():
        cat_dir = typ_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        cat_global = global_typicality_df[global_typicality_df['category'] == category].dropna(subset=['human_typicality', 'global_cosine_typicality']).copy()
        
        if len(cat_global) > 1:
            # Sort by human typicality to make the bars slope downwards cleanly
            cat_global = cat_global.sort_values('human_typicality', ascending=False)
            melted_typicality_df = cat_global.melt(
                id_vars=['concept'],
                value_vars=['human_typicality', 'global_cosine_typicality'],
                var_name='Metric', value_name='Score'
            )
            melted_typicality_df['Metric'] = melted_typicality_df['Metric'].replace({
                'human_typicality': 'Human Typicality',
                'global_cosine_typicality': 'Cosine Typicality'
            })

            _plot_bar_chart(
                plot_dataframe=melted_typicality_df, x_col='concept', y_col='Score',
                title=f"'{category.title()}' Human Typicality vs. Cosine Typicality",
                x_label="Concepts (Ordered by Human Typicality)", y_label="Typicality Score (0 to 1)", legend_title="Metric",
                hue='Metric', palette=['#4B5A6A', '#D96A5B'], out_path=cat_dir / f"{category}_typicality_comparison_bar.png", 
                figsize=(14, 6)
            )
            
            # Catch constant array warnings safely
            if cat_global['human_typicality'].std() > 0 and cat_global['global_cosine_typicality'].std() > 0:
                r, p = pearsonr(cat_global['human_typicality'], cat_global['global_cosine_typicality'])
                title_str = f"'{category.title()}' Human Typicality vs. Cosine Typicality\nPearson r = {r:.2f} (p={p:.3f})"
            else:
                title_str = f"'{category.title()}' Human Typicality vs. Cosine Typicality\n(Insufficient variance for correlation)"
            sns.regplot(
                data=cat_global, x='human_typicality', y='global_cosine_typicality', 
                color='#d62728', scatter_kws={'s': 80, 'edgecolor': 'black'}
            )

            # Annotate each dot with the specific word
            for i in range(len(cat_global)):
                plt.text(
                    cat_global['human_typicality'].iloc[i],
                    cat_global['global_cosine_typicality'].iloc[i],
                    f"  {cat_global['concept'].iloc[i]}",
                    fontsize=9, alpha=0.8, va='center'
                )
            plt.title(title_str, fontsize=14)
            plt.xlabel("Human Typicality", fontsize=12)
            plt.ylabel("Cosine Typicality (Global Prototype)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(cat_dir / f"{category}_typicality_correlation_scatter.png", dpi=300)
            plt.close()

        # Most Typical Word Per Layer
        cat_layer = layer_typicality_df[layer_typicality_df['category'] == category].copy()
        if not cat_layer.empty:
            # Find the row index of the max typicality score for each layer
            idx = cat_layer.groupby('layer_idx')['layer_cosine_typicality'].idxmax()
            top_per_layer = cat_layer.loc[idx, ['layer_idx', 'layer_name', 'concept', 'layer_cosine_typicality', 'human_typicality']]
            top_per_layer.rename(columns={
                'concept': 'most_typical_model_concept',
                'layer_cosine_typicality': 'cosine_similarity_score',
                'human_typicality': 'hardcoded_human_score'
            }, inplace=True)
            top_per_layer = top_per_layer.sort_values('layer_idx')
            save_dataframe(top_per_layer, cat_dir / f"{category}_most_typical_per_layer.csv")
            plot_most_typical_concept_per_layer(top_per_layer, category, cat_dir)
            
def plot_most_typical_concept_per_layer(top_concepts_per_layer_df: pd.DataFrame, category: str, cat_dir) -> None:
    """
    Plot the evolution of the most typical concept across layers.
    Uses a stem-like visual where the concept label acts as the 'bar'.
    """
    plt.figure(figsize=(24, 8))
    plt.vlines(x=range(len(top_concepts_per_layer_df)), ymin=0, ymax=top_concepts_per_layer_df['cosine_similarity_score'], 
               color='gray', alpha=0.3, linewidth=2)
    for i, row in top_concepts_per_layer_df.reset_index(drop=True).iterrows():
        plt.scatter(i, row['cosine_similarity_score'], color='#4B5A6A', s=20, zorder=3)
        plt.text(
            i, row['cosine_similarity_score'] + 0.02, # slight vertical offset
            row['most_typical_model_concept'].title(), 
            color='#D96A5B', ha='left', va='bottom', 
            rotation=60, fontsize=12, fontweight='bold'
        )
        
    plt.title(f"'{category.title()}' Evolution: The Most Typical Concept per Layer", fontsize=18, pad=20)
    plt.xlabel("Model Layer", fontsize=14)
    plt.ylabel("Cosine Typicality (Layer Prototype)", fontsize=14)
    plt.xticks(ticks=range(len(top_concepts_per_layer_df)), labels=top_concepts_per_layer_df['layer_name'], rotation=45, ha='right', fontsize=9)
    
    # Dynamically adjust Y limit to ensure long words don't get cut off at the top
    max_y = top_concepts_per_layer_df['cosine_similarity_score'].max()
    plt.ylim(0, max_y + 0.3)
    plt.xlim(-1, len(top_concepts_per_layer_df))
    
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(cat_dir / f"{category}_most_typical_evolution.png", dpi=300)
    plt.close()

def execute_module_7_empirical_cosine_typicality(formatted_expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, typ_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Module 7: Empirical Cosine Typicality.
    Compute cosine similarity between concepts and category prototypes at global and per-layer levels.
    """
    log.info("  Computing Empirical Cosine Typicality...")
    typ_global_df, typ_layer_df = compute_empirical_typicality(formatted_expert_allocation_df, concept_metadata, typ_dir)
    generate_category_typicality_reports(typ_global_df, typ_layer_df, typ_dir)
    return typ_global_df, typ_layer_df