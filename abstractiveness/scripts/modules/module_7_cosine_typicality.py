import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from utils.helpers import save_dataframe, axis_variants, scope_out_dir, scope_summary_row
from utils.plot_helpers import _plot_bar_with_leaders, fig_width_for, apply_rotated_leader_labels

log = logging.getLogger(__name__)

# Headline metrics for the cross-scope sublayer_comparison table.
SUMMARY_LABELS = {
    "typicality_pearson_r": "Pearson r, model vs human typicality",
    "n_typicality_concepts": "Concepts with both scores",
}

def _valid_category_members(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame):
    """
    Yield (category, cat_meta, valid_members) for every category with at least two
    members holding experts at the current AP threshold. A prototype needs two members
    to be anything other than the member itself, so smaller categories are skipped.
    """
    present_concepts = set(expert_allocation_df['concept'].unique())
    for category in concept_metadata['category'].dropna().unique():
        cat_meta = concept_metadata[concept_metadata['category'] == category]
        valid_members = [m for m in cat_meta['concept'].tolist() if m in present_concepts]
        if len(valid_members) >= 2:
            yield category, cat_meta, valid_members


def _human_typicality(cat_meta: pd.DataFrame, concept: str) -> float:
    """Human typicality score for one concept, NaN when the metadata carries none."""
    scores = cat_meta[cat_meta['concept'] == concept]['human_typicality'].values
    return scores[0] if len(scores) > 0 else np.nan


def compute_global_typicality(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, typ_dir) -> pd.DataFrame:
    """
    Cosine typicality of each concept against a category prototype built from its whole
    expert set, one feature per (layer, unit) pair across the entire scope.

    Set-based: the prototype is a centroid in the space of individual neurons, so the
    layer axis never enters and this is computed once per scope rather than once per axis
    variant. It must be fed the scope's own (flat) frame for exactly that reason -- on a
    block-aggregated frame the layer_name that keys each feature is the block, so units
    carrying the same index in different sublayers of one block would collapse into a
    single feature and inflate every prototype.
    """
    allocation = expert_allocation_df.assign(present=1)
    results = []

    for category, cat_meta, valid_members in _valid_category_members(allocation, concept_metadata):
        category_allocation_df = allocation[allocation['concept'].isin(valid_members)].copy()
        category_allocation_df['layer_unit'] = (category_allocation_df['layer_name'].astype(str) + "_"
                                                + category_allocation_df['unit'].astype(str))
        global_pivot = category_allocation_df.pivot_table(index='concept', columns='layer_unit',
                                                          values='present', fill_value=0)
        global_pivot = global_pivot.reindex(valid_members, fill_value=0)
        if global_pivot.empty:
            continue

        centroid_global = global_pivot.mean(axis=0).values.reshape(1, -1)
        sims_global = cosine_similarity(global_pivot.values, centroid_global).flatten()
        for i, concept in enumerate(global_pivot.index):
            results.append({
                'category': category,
                'concept': concept,
                'human_typicality': _human_typicality(cat_meta, concept),
                'global_cosine_typicality': sims_global[i],
            })

    global_typicality_df = pd.DataFrame(results)
    save_dataframe(global_typicality_df, typ_dir / "global_prototype_typicality.csv")
    return global_typicality_df


def compute_layer_typicality(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, typ_dir,
                             suffix: str = "") -> pd.DataFrame:
    """
    Cosine typicality against a category prototype rebuilt independently at every layer
    of the given axis, which is what the most-typical-per-layer evolution reads.

    Order-based: it treats the axis as depth, so the whole-model scope runs it once per
    axis variant (block and flat) while a sublayer scope runs it once.
    """
    allocation = expert_allocation_df.assign(present=1)
    ordered_layers = allocation[['layer_idx', 'layer_name']].drop_duplicates().sort_values('layer_idx')
    results = []

    for category, cat_meta, valid_members in _valid_category_members(allocation, concept_metadata):
        category_allocation_df = allocation[allocation['concept'].isin(valid_members)]

        for _, row in ordered_layers.iterrows():
            layer_df_sub = category_allocation_df[category_allocation_df['layer_idx'] == row['layer_idx']]
            layer_pivot = layer_df_sub.pivot_table(index='concept', columns='unit', values='present', fill_value=0)
            layer_pivot = layer_pivot.reindex(valid_members, fill_value=0)

            if layer_pivot.shape[1] > 0:
                centroid_layer = layer_pivot.mean(axis=0).values.reshape(1, -1)
                sims_layer = cosine_similarity(layer_pivot.values, centroid_layer).flatten()
            else:
                sims_layer = np.zeros(len(valid_members))

            for i, concept in enumerate(layer_pivot.index):
                results.append({
                    'category': category,
                    'concept': concept,
                    'layer_idx': row['layer_idx'],
                    'layer_name': row['layer_name'],
                    'human_typicality': _human_typicality(cat_meta, concept),
                    'layer_cosine_typicality': sims_layer[i],
                })

    layer_typicality_df = pd.DataFrame(results)
    save_dataframe(layer_typicality_df, typ_dir / f"layer_prototype_typicality{suffix}.csv")
    return layer_typicality_df


def generate_global_typicality_reports(global_typicality_df: pd.DataFrame, typ_dir) -> None:
    """
    Per-category report on the global prototype: the human-vs-model bar comparison and
    the correlation scatter. Built from the axis-invariant global scores, so it is
    written once per scope rather than once per axis variant.
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

            _plot_bar_with_leaders(
                plot_dataframe=melted_typicality_df, x_col='concept', y_col='Score',
                title=f"'{category.title()}' Human Typicality vs. Cosine Typicality",
                x_label="Concepts (Ordered by Human Typicality)", y_label="Typicality Score (0 to 1)", legend_title="Metric",
                hue='Metric', palette=['#4B5A6A', '#D96A5B'], out_path=cat_dir / f"{category}_typicality_comparison_bar.png",
                figsize=(fig_width_for(cat_global['concept'].nunique(), 0.55, min_w=14.0), 6), show_x_ticks=True
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


def generate_layer_typicality_reports(layer_typicality_df: pd.DataFrame, typ_dir, suffix: str = "") -> None:
    """
    Per-category report on the per-layer prototypes: which concept is the most typical
    member at each layer, as a CSV and an evolution plot. Reads the axis as depth, so it
    is written once per axis variant.
    """
    for category in layer_typicality_df['category'].unique():
        cat_dir = typ_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

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
            save_dataframe(top_per_layer, cat_dir / f"{category}_most_typical_per_layer{suffix}.csv")
            plot_most_typical_concept_per_layer(top_per_layer, category, cat_dir, suffix)
            
def plot_most_typical_concept_per_layer(top_concepts_per_layer_df: pd.DataFrame, category: str, cat_dir, suffix: str = "") -> None:
    """
    Plot the evolution of the most typical concept across layers.
    Uses a stem-like visual where the concept label acts as the 'bar'.
    """
    # Width scales with the layer count so the per-layer x-axis stays legible.
    fig_w = fig_width_for(len(top_concepts_per_layer_df), 0.28, min_w=16.0)
    plt.figure(figsize=(fig_w, 8))
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
    apply_rotated_leader_labels(plt.gca(), list(top_concepts_per_layer_df['layer_name']), axis='x', fontsize=9)
    
    # Dynamically adjust Y limit to ensure long words don't get cut off at the top
    max_y = top_concepts_per_layer_df['cosine_similarity_score'].max()
    plt.ylim(0, max_y + 0.3)
    plt.xlim(-1, len(top_concepts_per_layer_df))
    
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(cat_dir / f"{category}_most_typical_evolution{suffix}.png", dpi=300)
    plt.close()

def _summarize_typicality(global_typicality_df: pd.DataFrame) -> dict:
    """
    Correlation between the model's global cosine typicality and the human typicality
    ratings, pooled over every category, which is the single number that says whether a
    scope's expert geometry recovers human graded category structure at all. Pooling
    across categories rather than averaging per-category r values keeps small categories
    from carrying the same weight as large ones.
    """
    empty = {"typicality_pearson_r": np.nan, "typicality_pearson_p": np.nan, "n_typicality_concepts": 0}
    if global_typicality_df.empty:
        return empty
    paired = global_typicality_df.dropna(subset=['human_typicality', 'global_cosine_typicality'])
    if len(paired) < 3 or paired['human_typicality'].std() == 0 or paired['global_cosine_typicality'].std() == 0:
        return {**empty, "n_typicality_concepts": len(paired)}
    r, p = pearsonr(paired['human_typicality'], paired['global_cosine_typicality'])
    return {"typicality_pearson_r": r, "typicality_pearson_p": p, "n_typicality_concepts": len(paired)}


def execute_module_7_empirical_cosine_typicality(scope, concept_metadata: pd.DataFrame, typ_dir) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Execute Module 7: Empirical Cosine Typicality.
    Compute cosine similarity between concepts and category prototypes at global and per-layer levels.

    The global prototype is set-based and so is computed once per scope from the scope's
    own flat frame. The per-layer prototypes read the axis as depth, so they run once per
    entry of axis_variants, giving the whole-model scope both a block and a flat reading.

    Returns (global_typicality_df, layer_typicality_df, summary_row), with the layer
    frame from the canonical axis variant.
    """
    out_dir = scope_out_dir(typ_dir, scope)
    log.info(f"  [{scope.label}] Computing Empirical Cosine Typicality...")
    typ_global_df = compute_global_typicality(scope.expert_df, concept_metadata, out_dir)
    generate_global_typicality_reports(typ_global_df, out_dir)

    canonical_layer_df = pd.DataFrame()
    for i, (suffix, axis_df, axis_label) in enumerate(axis_variants(scope)):
        log.info(f"  [{scope.label} / {axis_label}] Computing per-layer typicality prototypes...")
        typ_layer_df = compute_layer_typicality(axis_df, concept_metadata, out_dir, suffix)
        generate_layer_typicality_reports(typ_layer_df, out_dir, suffix)
        if i == 0:
            canonical_layer_df = typ_layer_df

    return typ_global_df, canonical_layer_df, scope_summary_row(scope, **_summarize_typicality(typ_global_df))