import logging
import pandas as pd
from utils.helpers import save_dataframe
from utils.plot_helpers import _plot_bar_with_leaders, build_category_color_map, fig_width_for

log = logging.getLogger(__name__)

def plot_hierarchy_similarities(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, sim_dir) -> pd.DataFrame:
    """
    Calculate Jaccard and Overlap similarity metrics between concepts and categories.
    Generates bar charts for both metrics and returns results DataFrame.
    """
    # Key expert sets on (layer_idx, unit) pairs: the raw `unit` column is only the neuron
    # index *within* a layer, so identical indices from different layers would otherwise be
    # collapsed into one element, inflating intersections between unrelated experts.
    pair_keyed_df = expert_allocation_df.assign(
        layer_unit=list(zip(expert_allocation_df["layer_idx"], expert_allocation_df["unit"]))
    )
    unit_sets = pair_keyed_df.groupby("concept")["layer_unit"].apply(set).to_dict()
    
    results = []
    for _, row in concept_metadata.dropna(subset=["category"]).iterrows():
        concept, category = row["concept"], row["category"]
        u_concept = unit_sets.get(concept, set())  # Set A
        u_category = unit_sets.get(category, set()) # Set B
        
        if u_concept and u_category:
            intersection = len(u_concept.intersection(u_category))
            union = len(u_concept.union(u_category))
            len_a = len(u_concept)
            len_b = len(u_category)
            
            results.append({
                "concept": concept,
                "category": category,
                "hierarchy": f"{category} -> {concept}",
                "jaccard_pct": (intersection / union) * 100,
                "overlap_pct": (intersection / min(len_a, len_b)) * 100,
                # Raw set sizes behind the percentages, for scale
                "shared_expert_units": intersection,
                "concept_expert_units": len_a,
                "category_expert_units": len_b,
            })
    
    similarity_metrics_df = pd.DataFrame(results)
    if similarity_metrics_df.empty: return similarity_metrics_df
    
    save_dataframe(similarity_metrics_df, sim_dir / "category_concept_similarity_metrics.csv")

    # Color each concept's bar by its category. The df rows follow concept_metadata's
    # (category-grouped) order, so bars form contiguous colored blocks; the shared color
    # map keeps a category's color identical here, in the heatmaps, and anywhere else.
    color_map = build_category_color_map(concept_metadata.dropna(subset=["category"])["category"])
    bar_colors = [color_map[c] for c in similarity_metrics_df["category"]]
    present = set(similarity_metrics_df["category"])
    color_legend = {cat: col for cat, col in color_map.items() if cat in present}

    # Width scales with the number of category->concept bars so they stay legible.
    width = fig_width_for(len(similarity_metrics_df), 0.34, min_w=16.0)
    bar_order = list(similarity_metrics_df["hierarchy"])

    # Generate the 2 distinct plots
    metrics = [
        ("jaccard_pct", "Jaccard Similarity Index % (Global Equivalence)"),
        ("overlap_pct", "Overlap Coefficient % (Strict Subsetting)"),
    ]

    for col, title in metrics:
        _plot_bar_with_leaders(
            plot_dataframe=similarity_metrics_df, x_col="hierarchy", y_col=col,
            title=title, x_label="Category->Concept", y_label="Percentage %",
            bar_colors=bar_colors, color_legend=color_legend, legend_title="Category",
            tick_label_colors=bar_colors,
            show_x_ticks=True, figsize=(width, 10.14), order=bar_order,
            out_path=sim_dir / f"{col.replace('_pct', '')}_hierarchy.png"
        )

    return similarity_metrics_df

def execute_module_3_category_concept_similarities(formatted_expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, sim_dir) -> pd.DataFrame:
    """Execute Module 3: Category-Concept Similarities.
    Calculates Jaccard and Overlap coefficients between concepts and their categories.
    """
    log.info("  Generating hierarchy similarities (Jaccard & Overlap)...")
    similarity_metrics_df = plot_hierarchy_similarities(formatted_expert_allocation_df, concept_metadata, sim_dir)
    return similarity_metrics_df