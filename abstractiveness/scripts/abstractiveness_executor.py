import logging
import pathlib
import pandas as pd
import numpy as np

from utils.helpers import set_folder_log, load_experts_data, init_global_layer_mapping
from modules.module_1_layer_distribution import execute_module_1_layer_expert_distribution
from modules.module_2_shannon_entropy import execute_module_2_shannon_entropy
from modules.module_3_similarities import execute_module_3_category_concept_similarities
from modules.module_4_correlations import execute_module_4_correlations, execute_module_4b_jaccard_vs_cosine_typicality
from modules.module_5_heatmaps import execute_module_5_heatmaps
from modules.module_6_jensen_shannon_divergence import execute_module_6_dual_category_jsd
from modules.module_7_cosine_typicality import execute_module_7_empirical_cosine_typicality

np.random.seed(42)

log = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Add a StreamHandler so logs still print to the console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

if __name__ == "__main__":
    REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
    RESPONSES_DIR = REPO_ROOT / "responses" / "GPT2_abstractiveness_150_responses"
    MODEL = "gpt2"
    OUTPUT_ROOT = REPO_ROOT / "results" / "research_plots_150_revised_executor_again"
    METADATA_PATH = REPO_ROOT / "assets" / "metadata_150.json"
    LAYER_MAPPING_PATH = REPO_ROOT / "assets" / "layer_mapping.csv"
    
    concept_metadata = pd.read_json(METADATA_PATH).rename(columns={"typicality": "human_typicality"})
    global_layer_mapping = init_global_layer_mapping(RESPONSES_DIR, MODEL, LAYER_MAPPING_PATH)
    
    for ap in [0.5, 0.6, 0.7, 0.8, 0.9]:
        out_path = OUTPUT_ROOT / f"AP_{ap}"
        set_folder_log(out_path)
        
        # Build strict directory structure
        layer_distribution_dir = out_path / "1_layer_expert_distribution"
        concept_dir = layer_distribution_dir / "per_concept"
        entropy_analysis_dir = out_path / "2_shannon_entropy_analysis"
        similarity_dir = out_path / "3_category_concept_similarities"
        correlations_dir = out_path / "4_correlations"
        heatmap_dir = out_path / "5_heatmaps"
        divergence_dir = out_path / "6_dual_category_jsd"
        typicality_dir = out_path / "7_typicality_analysis"
        for d in [concept_dir, entropy_analysis_dir, similarity_dir, correlations_dir, heatmap_dir, divergence_dir, typicality_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        log.info(f"Starting Refactored Analysis Suite for AP Threshold: {ap}")
        raw_expert_data = load_experts_data(RESPONSES_DIR, MODEL, ap)
        
        if not raw_expert_data.empty:
            formatted_expert_allocation_df = (
                raw_expert_data.merge(concept_metadata, on="concept")
                .merge(global_layer_mapping, on="layer")
                .sort_values('layer_idx')
                .drop(columns=['layer']))
            
            # Module 1: Layer Expert Distribution
            merged_meta = execute_module_1_layer_expert_distribution(formatted_expert_allocation_df, concept_metadata, layer_distribution_dir)
            
            # Module 2: Shannon Entropy Analysis and Peak/Average Layer Distributions
            concept_entropy_df, category_entropy_df = execute_module_2_shannon_entropy(formatted_expert_allocation_df, concept_metadata, entropy_analysis_dir)
            
            # Module 3: Category-Concept Similarities
            similarity_metrics_df = execute_module_3_category_concept_similarities(formatted_expert_allocation_df, concept_metadata, similarity_dir)
            
            # Module 4: Correlations
            execute_module_4_correlations(merged_meta, similarity_metrics_df, correlations_dir)
            
            # Module 5: Heatmaps
            execute_module_5_heatmaps(formatted_expert_allocation_df, concept_metadata, heatmap_dir)
            
            # Module 6: Dual Category Definitions (JSD)
            jsd_results_df, jsd_layer_labels = execute_module_6_dual_category_jsd(formatted_expert_allocation_df, concept_metadata, divergence_dir)
            
            # Module 7: Empirical Cosine Typicality
            global_typicality_df, layer_typicality_df = execute_module_7_empirical_cosine_typicality(formatted_expert_allocation_df, concept_metadata, typicality_dir)

            # Module 4b: Jaccard vs. Cosine Typicality (needs module 7's output, so it runs here)
            execute_module_4b_jaccard_vs_cosine_typicality(similarity_metrics_df, global_typicality_df, correlations_dir)

            log.info(f"  All analysis outputs cleanly structured in {out_path}")
        else:
            log.warning(f"No experts found for threshold {ap}. Skipping folder.")