import logging
import pandas as pd
from utils.helpers import save_dataframe
from utils.plot_helpers import _plot_bar_with_leaders

log = logging.getLogger(__name__)

def plot_hierarchy_similarities(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, sim_dir) -> pd.DataFrame:
    """
    Calculate Jaccard and Overlap similarity metrics between concepts and categories.
    Generates bar charts for both metrics and returns results DataFrame.
    """
    unit_sets = expert_allocation_df.groupby("concept")["unit"].apply(set).to_dict()
    
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
                "overlap_pct": (intersection / min(len_a, len_b)) * 100
            })
    
    similarity_metrics_df = pd.DataFrame(results)
    if similarity_metrics_df.empty: return similarity_metrics_df
    
    save_dataframe(similarity_metrics_df, sim_dir / "category_concept_similarity_metrics.csv")

    # Generate the 2 distinct plots
    metrics = [
        ("jaccard_pct", "Jaccard Similarity Index % (Global Equivalence)", "#27ae60"),
        ("overlap_pct", "Overlap Coefficient % (Strict Subsetting)", "#8e44ad")
    ]
    
    for col, title, color in metrics:
        _plot_bar_with_leaders(
            plot_dataframe=similarity_metrics_df, x_col="hierarchy", y_col=col,
            title=title, y_label="Percentage %", color=color, 
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