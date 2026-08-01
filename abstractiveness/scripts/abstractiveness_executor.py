import logging
import pathlib
import pandas as pd
import numpy as np

from utils.helpers import (set_folder_log, load_experts_data, init_global_layer_mapping,
                           load_concept_embeddings, save_dataframe, build_analysis_scopes,
                           load_or_build_sublayer_rank, filter_expert_data_to_sublayer,
                           scope_out_dir)
from utils.plot_helpers import plot_sublayer_comparison_bars
from modules import (module_1_layer_distribution, module_2_shannon_entropy, module_3_similarities,
                     module_4_correlations, module_5_heatmaps, module_6_jensen_shannon_divergence,
                     module_7_cosine_typicality)
from modules.module_1_layer_distribution import (execute_module_1_layer_expert_distribution,
                                                 save_expert_counts_metadata)
from modules.module_2_shannon_entropy import execute_module_2_shannon_entropy
from modules.module_3_similarities import execute_module_3_category_concept_similarities
from modules.module_4_correlations import execute_module_4_correlations, execute_module_4b_jaccard_vs_cosine_typicality
from modules.module_5_heatmaps import execute_module_5_heatmaps
from modules.module_6_jensen_shannon_divergence import execute_module_6_dual_category_jsd
from modules.module_7_cosine_typicality import execute_module_7_empirical_cosine_typicality
from modules.module_8_embedding_rsa import execute_module_8_embedding_rsa

np.random.seed(42)

# AP thresholds swept per run. The most lenient one doubles as the reference threshold for
# the sublayer expert-count ranking, where every sublayer still has experts to rank.
AP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
REFERENCE_AP = min(AP_THRESHOLDS)

# Which modules to run. Narrow this while iterating on one module so a verification run
# costs seconds instead of the full sweep, e.g. {3, 4, 5}. Modules 1, 2 and 7 feed module
# 4, and disabling any of them silently drops the panels that depend on it (see the
# dependency handling in the scope loop), so a narrowed run is for verification, never for
# producing the results anyone reads. Restore to the full set before a real sweep.
ENABLED_MODULES = {3, 4, 5}

# Which columns each module's cross-scope comparison plot draws. Each module owns its own
# list so the metric names stay next to the code that computes them.
MODULE_SUMMARY_LABELS = {
    1: module_1_layer_distribution.SUMMARY_LABELS,
    2: module_2_shannon_entropy.SUMMARY_LABELS,
    3: module_3_similarities.SUMMARY_LABELS,
    4: module_4_correlations.SUMMARY_LABELS,
    5: module_5_heatmaps.SUMMARY_LABELS,
    6: module_6_jensen_shannon_divergence.SUMMARY_LABELS,
    7: module_7_cosine_typicality.SUMMARY_LABELS,
    8: {},
}

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
#   sublayer_filter    : sublayer type (e.g. "mlp.gate_proj") that modules 2+ are
#                        restricted to; module 1 still sees everything (whole-model
#                        plots + the informativeness ranking justifying this choice).
#                        Set to None to analyze all sublayers as before.
#   embedding_cache_file : concept embedding cache under assets/, consumed by module 8.
#                        Built once per model by scripts/precompute_concept_embeddings.py;
#                        module 8 skips itself when it is absent.
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "gpt2_150": {
        "responses_subdir": "GPT2_abstractiveness_150_responses",
        "model_subdir": "gpt2",
        "architecture": "gpt2",
        "metadata_file": "metadata_150.json",
        "layer_mapping_file": "layer_mapping_GPT2.csv",
        "typicality_column": "typicality",
        "output_subdir": "research_plots_150_final",
        "sublayer_filter": "mlp.c_fc",
        "embedding_cache_file": "concept_embeddings_gpt2_150.npz",
    },
    "qwen3_richie_hsj": {
        "responses_subdir": "Qwen3_1.7B_abstractiveness_Richie_HSJ_responses",
        "model_subdir": "Qwen",
        "architecture": "qwen3",
        "metadata_file": "metadata_Richie_HSJ.json",
        "layer_mapping_file": "layer_mapping_Qwen3_1-7B.csv",
        "typicality_column": "typicality_HSJ_pairwise",
        "output_subdir": "research_plots_qwen_richie_hsj_with_distribution",
        "sublayer_filter": "mlp.gate_proj",
        "embedding_cache_file": "concept_embeddings_qwen3_richie_hsj.npz",
    },
    # GPT-2 on the same Richie-HSJ dataset -- the architecture comparison against
    # qwen3_richie_hsj (same metadata + typicality column, GPT-2's 48-layer mapping).
    "gpt2_richie_hsj": {
        "responses_subdir": "GPT2_abstractiveness_Richie_HSJ_responses",
        "model_subdir": "gpt2",
        "architecture": "gpt2",
        "metadata_file": "metadata_Richie_HSJ.json",
        "layer_mapping_file": "layer_mapping_GPT2.csv",
        "typicality_column": "typicality_HSJ_pairwise",
        "output_subdir": "research_plots_gpt2_richie_hsj_with_distribution",
        "sublayer_filter": "mlp.c_fc",
        "embedding_cache_file": "concept_embeddings_gpt2_richie_hsj.npz",
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
    # Rows without a concept name (e.g. entries deactivated by renaming their key, which
    # pd.read_json turns into an all-NaN row plus a stray column) are dropped defensively.
    concept_metadata = (pd.read_json(METADATA_PATH)
                        .rename(columns={TYPICALITY_COLUMN: "human_typicality"})
                        .dropna(subset=["concept"])
                        .reset_index(drop=True))
    concept_metadata = concept_metadata.drop(
        columns=[c for c in concept_metadata.columns if c.startswith("_")], errors="ignore")
    global_layer_mapping = init_global_layer_mapping(RESPONSES_DIR, MODEL, LAYER_MAPPING_PATH, ARCHITECTURE)

    # Concept embeddings do not depend on the AP threshold, so the cache is read once
    # here rather than five times inside the sweep (it is close to a gigabyte for Qwen3).
    # A missing cache leaves module 8 to skip itself; every other module is unaffected.
    embedding_cache = load_concept_embeddings(REPO_ROOT / "assets" / cfg["embedding_cache_file"])


    # Sublayer ranking by expert count, frozen at the most lenient threshold of the sweep
    # and cached at the run root, so "1_mlp.gate_proj" names the same sublayer in every
    # AP_x folder and output paths stay comparable across the whole sweep.
    def _load_reference_expert_frame():
        reference = load_experts_data(RESPONSES_DIR, MODEL, REFERENCE_AP)
        return reference.merge(global_layer_mapping, on="layer").drop(columns=["layer"])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sublayer_rank = load_or_build_sublayer_rank(
        OUTPUT_ROOT / "sublayer_rank.csv", _load_reference_expert_frame, REFERENCE_AP)
    rank_by_sublayer = dict(zip(sublayer_rank["sublayer"], sublayer_rank["rank"]))

    for ap in AP_THRESHOLDS:
        out_path = OUTPUT_ROOT / f"AP_{ap}"
        set_folder_log(out_path)

        # Build strict directory structure. Each module folder holds its whole-model
        # analysis directly, plus a sublayers/<rank>_<name>/ subfolder per sublayer.
        module_dirs = {
            1: out_path / "1_layer_expert_distribution",
            2: out_path / "2_shannon_entropy_analysis",
            3: out_path / "3_category_concept_similarities",
            4: out_path / "4_correlations",
            5: out_path / "5_heatmaps",
            6: out_path / "6_dual_category_jsd",
            7: out_path / "7_typicality_analysis",
            8: out_path / "8_embedding_rsa",
        }
        for d in module_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        log.info(f"Starting Refactored Analysis Suite for AP Threshold: {ap}")
        raw_expert_data = load_experts_data(RESPONSES_DIR, MODEL, ap)

        if not raw_expert_data.empty:
            formatted_expert_allocation_df = (
                raw_expert_data.merge(concept_metadata, on="concept")
                .merge(global_layer_mapping, on="layer")
                .sort_values('layer_idx')
                .drop(columns=['layer']))

            # The analysis sublayer still gets one privileged use: module 1's per-concept
            # plots, which are far too many to replicate per sublayer.
            analysis_sublayer_df = filter_expert_data_to_sublayer(
                formatted_expert_allocation_df, cfg.get("sublayer_filter"))

            scopes = build_analysis_scopes(formatted_expert_allocation_df, sublayer_rank)
            log.info(f"  Analysis scopes: {', '.join(s.label for s in scopes)}")
            summaries = {module: [] for module in module_dirs}

            for scope in scopes:
                rows = {}
                empty = pd.DataFrame()

                # Module 1: Layer Expert Distribution
                if 1 in ENABLED_MODULES:
                    merged_meta, rows[1] = execute_module_1_layer_expert_distribution(
                        scope, concept_metadata, module_dirs[1], analysis_sublayer_df=analysis_sublayer_df)
                else:
                    # Module 4 needs module 1's expert-count table. Build just that, skipping
                    # the per-concept plots and the Mantel permutation sweep, which are the
                    # expensive parts and the reason for disabling the module in the first place.
                    merged_meta = (save_expert_counts_metadata(scope.expert_df, concept_metadata,
                                                               scope_out_dir(module_dirs[1], scope))
                                   if 4 in ENABLED_MODULES else empty)

                # Module 2: Shannon Entropy Analysis and Peak/Average Layer Distributions
                if 2 in ENABLED_MODULES:
                    concept_entropy_df, category_entropy_df, rows[2] = execute_module_2_shannon_entropy(
                        scope, concept_metadata, module_dirs[2])
                else:
                    # plot_correlations already guards on an empty entropy frame and drops
                    # its entropy panel with a warning.
                    concept_entropy_df, category_entropy_df = empty, empty

                # Module 3: Category-Concept Similarities
                if 3 in ENABLED_MODULES:
                    similarity_metrics_df, rows[3] = execute_module_3_category_concept_similarities(
                        scope, concept_metadata, module_dirs[3])
                else:
                    similarity_metrics_df = empty

                # Module 4: Correlations (uses module 2's entropy tables for the entropy panel)
                if 4 in ENABLED_MODULES:
                    rows[4] = execute_module_4_correlations(
                        scope, merged_meta, similarity_metrics_df, concept_entropy_df,
                        category_entropy_df, concept_metadata, module_dirs[4])

                # Module 5: Heatmaps
                if 5 in ENABLED_MODULES:
                    rows[5] = execute_module_5_heatmaps(scope, concept_metadata, module_dirs[5])

                # Module 6: Dual Category Definitions (JSD)
                if 6 in ENABLED_MODULES:
                    _, _, rows[6] = execute_module_6_dual_category_jsd(scope, concept_metadata, module_dirs[6])

                # Module 7: Empirical Cosine Typicality
                if 7 in ENABLED_MODULES:
                    global_typicality_df, _, rows[7] = execute_module_7_empirical_cosine_typicality(
                        scope, concept_metadata, module_dirs[7])
                else:
                    # execute_module_4b guards on an empty frame and returns {}.
                    global_typicality_df = empty

                # Module 4b: Jaccard vs. Cosine Typicality (needs module 7's output, so it runs here)
                if 4 in ENABLED_MODULES:
                    rows[4].update(execute_module_4b_jaccard_vs_cosine_typicality(
                        scope, similarity_metrics_df, global_typicality_df, concept_metadata, module_dirs[4]))

                for module, row in rows.items():
                    summaries[module].append(row)

            # One cross-scope comparison table per module: the readable summary of a sweep
            # that would otherwise be seven folders deep at every AP threshold.
            for module, rows in summaries.items():
                if not rows:
                    continue
                comparison = pd.DataFrame(rows)
                save_dataframe(comparison, module_dirs[module] / "sublayer_comparison.csv")
                plot_sublayer_comparison_bars(
                    comparison, MODULE_SUMMARY_LABELS[module],
                    module_dirs[module] / "sublayer_comparison.png",
                    f"Module {module}: analysis scopes compared (AP {ap})")

            # Module 8: Expert set vs embedding semantics (second-order RSA). It sweeps
            # sublayers internally, so it runs once outside the scope loop. The embedding
            # cache is AP-independent and is loaded once above, outside this loop.
            if 8 in ENABLED_MODULES:
                execute_module_8_embedding_rsa(
                    analysis_sublayer_df, concept_metadata, module_dirs[8],
                    embedding_cache, global_layer_mapping,
                    sublayer_filter=cfg.get("sublayer_filter"),
                    full_expert_df=formatted_expert_allocation_df,
                    sublayer_rank=rank_by_sublayer)

            log.info(f"  All analysis outputs cleanly structured in {out_path}")
        else:
            log.warning(f"No experts found for threshold {ap}. Skipping folder.")