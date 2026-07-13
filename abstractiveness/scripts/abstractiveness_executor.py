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

# ---------------------------------------------------------------------------
# Run configuration.
#
# Each entry fully describes one (model x dataset) run. To analyze a different
# model or metadata set, add/edit an entry and point ACTIVE_CONFIG at it -- no
# code below this block needs to change. Fields:
#   responses_subdir   : folder under responses/ holding that model's outputs
#   model_subdir       : the sub-folder inside responses_subdir to scan for
#                        expertise.csv (also passed to the layer-mapping loader)
#   architecture       : key of helpers.LAYER_ARCHITECTURES ("gpt2", "qwen3")
#   metadata_file      : concept metadata JSON under assets/
#   layer_mapping_file : cached layer mapping under assets/ (auto-generated if absent)
#   typicality_column  : metadata column to use as the human typicality score;
#                        it is renamed to "human_typicality" for all downstream modules
#   output_subdir      : folder under results/ to write this run's plots/tables
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "gpt2_150": {
        "responses_subdir": "GPT2_abstractiveness_150_responses",
        "model_subdir": "gpt2",
        "architecture": "gpt2",
        "metadata_file": "metadata_150.json",
        "layer_mapping_file": "layer_mapping_GPT2.csv",
        "typicality_column": "typicality",
        "output_subdir": "research_plots_150_revised_executor_again",
    },
    "qwen3_richie_hsj": {
        "responses_subdir": "Qwen3_1.7B_abstractiveness_Richie_HSJ_responses",
        "model_subdir": "Qwen",
        "architecture": "qwen3",
        "metadata_file": "metadata_Richie_HSJ.json",
        "layer_mapping_file": "layer_mapping_Richie_HSJ.csv",
        "typicality_column": "typicality_HSJ_pairwise",
        "output_subdir": "research_plots_qwen_richie_hsj",
    },
    # GPT-2 on the same Richie-HSJ dataset -- the architecture comparison against
    # qwen3_richie_hsj (same metadata + typicality column, GPT-2's 48-layer mapping).
    "gpt2_richie_hsj": {
        "responses_subdir": "GPT2_abstractiveness_Richie_HSJ_responses",
        "model_subdir": "gpt2",
        "architecture": "gpt2",
        "metadata_file": "metadata_Richie_HSJ.json",
        "layer_mapping_file": "layer_mapping_GPT2_Richie_HSJ.csv",
        "typicality_column": "typicality_HSJ_pairwise",
        "output_subdir": "research_plots_gpt2_richie_hsj",
    },
}

# Select which configuration to run.
ACTIVE_CONFIG = "qwen3_richie_hsj"

if __name__ == "__main__":
    REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
    cfg = MODEL_CONFIGS[ACTIVE_CONFIG]

    RESPONSES_DIR = REPO_ROOT / "responses" / cfg["responses_subdir"]
    MODEL = cfg["model_subdir"]
    ARCHITECTURE = cfg["architecture"]
    OUTPUT_ROOT = REPO_ROOT / "results" / cfg["output_subdir"]
    METADATA_PATH = REPO_ROOT / "assets" / cfg["metadata_file"]
    LAYER_MAPPING_PATH = REPO_ROOT / "assets" / cfg["layer_mapping_file"]
    TYPICALITY_COLUMN = cfg["typicality_column"]

    log.info(f"Running analysis with config '{ACTIVE_CONFIG}' (architecture: {ARCHITECTURE})")
    concept_metadata = pd.read_json(METADATA_PATH).rename(columns={TYPICALITY_COLUMN: "human_typicality"})
    global_layer_mapping = init_global_layer_mapping(RESPONSES_DIR, MODEL, LAYER_MAPPING_PATH, ARCHITECTURE)
    
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