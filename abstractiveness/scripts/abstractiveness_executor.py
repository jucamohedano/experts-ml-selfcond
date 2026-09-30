import argparse
import logging
import pathlib
import pandas as pd
import numpy as np

from utils.logging_and_io import set_folder_log, save_dataframe
from utils.plotting import plot_sublayer_comparison_bars
from core.expert_data_loading import (load_experts_data, init_global_layer_mapping,
                                      filter_expert_data_to_sublayer)
from core.embedding_cache import load_or_build_concept_embeddings
from core.analysis_scopes import build_analysis_scopes, load_or_build_sublayer_rank
from modules import (module_1_expert_distribution, module_2_similarities, module_3_dual_category_jsd,
                     module_5_typicality_prediction)
from modules.module_4_embedding_rsa import execute_module_4_embedding_rsa

np.random.seed(42)

# AP thresholds swept per run. The most lenient one doubles as the reference threshold for
# the sublayer expert-count ranking, where every sublayer still has experts to rank.
AP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
REFERENCE_AP = min(AP_THRESHOLDS)
# Thresholds at which module 1 draws its per-concept plots and modules 1, 4 and 5 run their
# sublayer scopes. Elsewhere they run the whole model (module 4 its analysis sublayer) only.
DETAILED_AP_THRESHOLDS = {0.5, 0.6, 0.7}

# Output folder of every module.
MODULE_DIRS = {1: "1_expert_distribution", 2: "2_similarities", 3: "3_dual_category_jsd",
               4: "4_embedding_rsa", 5: "5_typicality_prediction"}

# Which modules to run. Narrow it while iterating, e.g. {2}, and restore it before a real sweep.
ENABLED_MODULES = {1, 2, 3, 4, 5}

# Sublayer scopes of modules 2 and 3 compute only their sublayer_comparison rows. True writes
# every sublayer's full outputs under sublayers/ as before. Modules 1 and 5 always write them,
# module 4 runs its own sublayer sweep.
WRITE_SUBLAYER_OUTPUTS = False

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
#   embedding_cache_file : concept embedding cache under assets/, consumed by module 4.
#                        Built from the response pkls on the first run with module 4
#                        enabled, then reused, see core/embedding_cache.py.
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
        "output_subdir": "research_plots_qwen_richie_hsj_restructured_final",
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
        "output_subdir": "research_plots_gpt2_richie_hsj_restructured_final",
        "sublayer_filter": "mlp.c_fc",
        "embedding_cache_file": "concept_embeddings_gpt2_richie_hsj.npz",
    },
}

# Configuration run when --config is not given on the command line.
ACTIVE_CONFIG = "gpt2_richie_hsj"


def section_entries(module, module_dir: pathlib.Path, rows: dict) -> list:
    """Summary entries of a sectioned module, one (folder, labels, title, row) per section."""
    return [(module_dir / module.SECTION_DIRS[section], module.SECTION_SUMMARY_LABELS[section],
             f"Section {section}", row) for section, row in rows.items()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the abstractiveness analysis for one model configuration.")
    parser.add_argument("--config", default=ACTIVE_CONFIG, choices=sorted(MODEL_CONFIGS),
                        help=f"key of MODEL_CONFIGS to run (default {ACTIVE_CONFIG})")
    ACTIVE_CONFIG = parser.parse_args().config
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

    # Concept embeddings do not depend on the AP threshold, so the cache is read once here, and
    # built first from the response pkls when it is missing (about 5 minutes on GPT-2, 30 on Qwen3).
    embedding_cache = (load_or_build_concept_embeddings(
        REPO_ROOT / "assets" / cfg["embedding_cache_file"], RESPONSES_DIR, MODEL, LAYER_MAPPING_PATH,
        ARCHITECTURE) if 4 in ENABLED_MODULES else None)


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

        # Each module folder holds its whole-model analysis directly, plus a
        # sublayers/<rank>_<name>/ subfolder per sublayer.
        module_dirs = {module: out_path / folder for module, folder in MODULE_DIRS.items()}
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

            # The analysis sublayer is module 4's primary sublayer.
            analysis_sublayer_df = filter_expert_data_to_sublayer(
                formatted_expert_allocation_df, cfg.get("sublayer_filter"))

            scopes = build_analysis_scopes(formatted_expert_allocation_df, sublayer_rank)
            log.info(f"  Analysis scopes: {', '.join(s.label for s in scopes)}")
            # Rows of every cross-scope comparison table, keyed by the folder it is written to.
            summaries = {}
            detailed = ap in DETAILED_AP_THRESHOLDS
            for scope in scopes:
                write = scope.is_whole_model or WRITE_SUBLAYER_OUTPUTS
                full_scope = scope.is_whole_model or detailed
                entries = []
                if 1 in ENABLED_MODULES and full_scope:
                    entries += section_entries(
                        module_1_expert_distribution, module_dirs[1],
                        module_1_expert_distribution.execute_module_1_expert_distribution(
                            scope, concept_metadata, module_dirs[1], per_concept=detailed))
                if 2 in ENABLED_MODULES:
                    entries += section_entries(
                        module_2_similarities, module_dirs[2],
                        module_2_similarities.execute_module_2_similarities(
                            scope, concept_metadata, module_dirs[2], write))
                if 3 in ENABLED_MODULES:
                    _, _, row = module_3_dual_category_jsd.execute_module_3_dual_category_jsd(
                        scope, concept_metadata, module_dirs[3], write)
                    entries.append((module_dirs[3], module_3_dual_category_jsd.SUMMARY_LABELS, "Module 3", row))
                if 5 in ENABLED_MODULES and full_scope:
                    _, row = module_5_typicality_prediction.execute_module_5_typicality_prediction(
                        scope, concept_metadata, module_dirs[5], axis_sensitivity=(ap == REFERENCE_AP))
                    entries.append((module_dirs[5], module_5_typicality_prediction.SUMMARY_LABELS, "Module 5", row))

                for folder, labels, title, row in entries:
                    summaries.setdefault(folder, (labels, title, []))[2].append(row)

            # One cross-scope comparison table per module or section, the readable summary of
            # a sweep that would otherwise be seven folders deep at every AP threshold.
            for folder, (labels, title, rows) in summaries.items():
                if len(rows) < 2:
                    continue
                comparison = pd.DataFrame(rows)
                save_dataframe(comparison, folder / "sublayer_comparison.csv")
                plot_sublayer_comparison_bars(
                    comparison, labels, folder / "sublayer_comparison.png",
                    f"{title}: analysis scopes compared (AP {ap})")

            # Module 4: Expert set vs embedding semantics (second-order RSA). It sweeps
            # sublayers internally, so it runs once outside the scope loop. The embedding
            # cache is AP-independent and is loaded once above, outside this loop.
            if 4 in ENABLED_MODULES:
                execute_module_4_embedding_rsa(
                    analysis_sublayer_df, concept_metadata, module_dirs[4],
                    embedding_cache, global_layer_mapping,
                    sublayer_filter=cfg.get("sublayer_filter"),
                    full_expert_df=formatted_expert_allocation_df,
                    sublayer_rank=rank_by_sublayer, sweep_sublayers=detailed)

            log.info(f"  All analysis outputs cleanly structured in {out_path}")
        else:
            log.warning(f"No experts found for threshold {ap}. Skipping folder.")
