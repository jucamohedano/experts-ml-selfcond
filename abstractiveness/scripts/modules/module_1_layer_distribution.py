import logging
import pandas as pd
import numpy as np
from utils.helpers import save_dataframe
from utils.plot_helpers import _plot_bar_chart

log = logging.getLogger(__name__)

# Define a high-contrast color palette for the different abstraction levels
ABSTRACTION_COLORS = {1: "#003f5c", 2: "#ffa600"} 

def save_expert_counts_metadata(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, dist_dir) -> pd.DataFrame:
    """Save the metadata combined with expert counts into the distribution folder.
    Merges expert counts per concept with concept metadata and saves to CSV.
    Adds log-frequency column if frequency data is available.
    """
    counts = expert_allocation_df.groupby("concept").size().reset_index(name="expert_count")
    merged = counts.merge(concept_metadata, on="concept")
    if 'frequency' in merged.columns:
        merged = merged[merged['frequency'] > 0].copy()
        merged['log_frequency'] = np.log10(merged['frequency'])
    desired_order = [
        "concept", 
        "category", 
        "abstraction_level", 
        "frequency", 
        "log_frequency",
        "human_typicality",
        "expert_count"
    ]
    final_order = [col for col in desired_order if col in merged.columns]
    merged = merged[final_order]
    save_dataframe(merged, dist_dir / "expert_counts_with_metadata.csv")
    return merged


def compute_layer_distributions(expert_allocation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Compute layer distribution statistics for expert allocations across concepts and abstraction levels.
    Calculates percentage distributions per concept per layer, aggregates by abstraction level, and totals per concept.
    Returns: (concept-level distribution matrix, global averaged distribution by abstraction level, concept expert counts).
    """
    # 1. Build a matrix of the percentage distribution per concept
    concept_distribution_matrix = pd.crosstab(
        index=[expert_allocation_df['abstraction_level'], expert_allocation_df['concept']],
        columns=expert_allocation_df['layer_name'],
        normalize='index' 
    ) * 100  
    # 2. Average those percentages by abstraction level for the global view
    global_layer_dist = concept_distribution_matrix.groupby(level='abstraction_level').mean().reset_index()
    # 3. Melt the wide matrix back into a long format for Seaborn plotting
    global_layer_dist = global_layer_dist.melt(
        id_vars='abstraction_level',
        var_name='layer_name', 
        value_name='mean_expert_allocation_pct'
    )
    # 4. Pre-calculate total expert counts for the plot titles
    expert_counts = expert_allocation_df.groupby('concept').size()
    
    return concept_distribution_matrix, global_layer_dist, expert_counts

def plot_global_distribution(global_layer_distribution_df: pd.DataFrame, dist_dir) -> None:
    """Save global layer distribution CSV and generate bar chart showing mean expert allocation across abstraction levels."""
    save_dataframe(global_layer_distribution_df, dist_dir / "mean_expert_layer_distribution.csv")

    _plot_bar_chart(
        plot_dataframe=global_layer_distribution_df, x_col="layer_name", y_col="mean_expert_allocation_pct",
        title="Mean Expert Distribution across model layers", x_label="Model Layer", y_label="Average % of Experts",
        hue="abstraction_level", palette=ABSTRACTION_COLORS, out_path=dist_dir / "mean_expert_layer_distribution.png", 
        figsize=(31.2, 10.4), linewidth=0.5
    )

def plot_per_concept_distributions(concept_distribution_matrix: pd.DataFrame, expert_counts: pd.Series, dist_dir) -> None:
    """Generate individual distribution CSVs and charts for each concept, organized by abstraction level."""
    concept_dir = dist_dir / "per_concept"
    
    for (abs_lvl, concept), row_data in concept_distribution_matrix.iterrows():
        specific_concept_dir = concept_dir / concept
        specific_concept_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert the row series back into a simple dataframe for Seaborn
        concept_layer_distribution_df = row_data.reset_index(name='expert_allocation_pct')
        total_experts = expert_counts[concept]
        
        save_dataframe(concept_layer_distribution_df, specific_concept_dir / f"{concept}_data.csv")

        _plot_bar_chart(
            plot_dataframe=concept_layer_distribution_df, x_col="layer_name", y_col="expert_allocation_pct",
            title=f"Concept: {concept.upper()} | Abstraction Level {abs_lvl} | Total Experts: {total_experts}",
            x_label="Model Layer", y_label="% of Experts", color=ABSTRACTION_COLORS[abs_lvl], 
            out_path=specific_concept_dir / f"{concept}_distribution.png", figsize=(28.6, 7.8)
        )

def execute_module_1_layer_expert_distribution(formatted_expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame, dist_dir) -> pd.DataFrame:
    """
    Execute Module 1: Analyze expert layer distributions across concepts and abstraction levels.
    Generates global distribution summaries, per-concept breakdowns, and saves all results to dist_dir.
    Returns merged metadata with expert counts.
    """
    log.info("  Generating combined metadata counts...")
    merged_meta = save_expert_counts_metadata(formatted_expert_allocation_df, concept_metadata, dist_dir)
    log.info("  Computing layer distribution matrices...")
    concept_matrix, global_dist, expert_counts = compute_layer_distributions(formatted_expert_allocation_df)
    log.info("  Generating global expert layer distribution plot...")
    plot_global_distribution(global_dist, dist_dir)
    log.info("  Generating per-concept expert layer distribution plots...")
    plot_per_concept_distributions(concept_matrix, expert_counts, dist_dir)
    return merged_meta