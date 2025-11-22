#!/usr/bin/env python3
"""
Hydra-based pipeline for unique expert analysis.
Replaces find_unique_experts.sh and find_unique_experts_parallel.sh.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib
import logging
import sys

# Import analysis scripts directly
from scripts.compute_neuron_correlations import run_correlation_analysis
from scripts.filter_unique_experts import run_filtering
from scripts.visualize_neuron_correlations import run_visualization
from scripts.analyze_expert_overlap import run_overlap_analysis
from scripts.visualize_expert_overlap import run_overlap_visualization
from scripts.compute_responses import run_response_computation
from scripts.compute_expertise import run_expertise_computation
from scripts.compute_shared_experts_stats import run_shared_experts_stats

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Starting Analysis Pipeline")
    log.info(f"Task: {cfg.task.name}")
    log.info(f"Model: {cfg.model.name}")
    log.info(f"Concept Group: {cfg.concept_group.name}")
    
    # Resolve base paths
    original_cwd = pathlib.Path(hydra.utils.get_original_cwd())
    
    # Responses base directory (inputs)
    responses_base = original_cwd / cfg.responses_base_dir
    
    # Output base is the current working directory (Hydra manages this)
    output_base = pathlib.Path.cwd()
    log.info(f"Output directory: {output_base}")
    
    if cfg.task.name == "compute_responses":
        log.info("Running response computation...")
        
        # For this task, responses_base is the output
        if not responses_base.exists():
            log.info(f"Creating responses base directory: {responses_base}")
            responses_base.mkdir(parents=True, exist_ok=True)
            
        # Data path is relative to original CWD
        data_path = original_cwd / "assets" / cfg.concept_group.name.split('_')[1] # Assuming structure: assets/custom_60
        # A better way might be to define data path in config, but for now lets infer or use a fixed path if not provided
        # The user command example showed: assets/Qwen3-30B-A3B-Instruct-2507_custom_60
        # Let's try to construct it or allow config override
        
        # Actually, looking at user command:
        # --data-path assets/Qwen3-30B-A3B-Instruct-2507_custom_60
        # This seems model specific in the example? Or just the folder name?
        # Let's assume standard asset structure: assets/{group_name}
        # Or maybe assets/{model}_{group} based on the example.
        # Let's use a config value for asset_dir if possible, or default to what was likely intended.
        # Given the previous context, let's assume assets are in `assets/{cfg.concept_group.name}` or similar.
        # The user example: assets/Qwen3-30B-A3B-Instruct-2507_custom_60
        # matches {model.generation}_{concept_group.group_dir_name}_{len(concepts)} potentially?
        # simpler: let's just assume it is `assets/{cfg.concept_group.name}` but the user example was specific.
        # Let's blindly pass the data_path if configured, or construct it.
        
        # Re-reading user request 1: --data-path assets/Qwen3-30B-A3B-Instruct-2507_custom_60
        # This matches the concept group name? 
        # In conf/concept_group/standard_60.yaml: name: openai_custom_60
        # In conf/model/qwen3_gpt2.yaml: generation: Qwen3-8B-FP8
        # The user example used a different model: Qwen3-30B... 
        # Let's assume data_path is `assets/{cfg.concept_group.name}` relative to original_cwd.
        
        data_path = original_cwd / "assets" / cfg.concept_group.name
        if not data_path.exists():
             # Fallback for the specific example user gave if name doesn't match folder
             # try constructing from model generation and group
             try:
                 alt_path = original_cwd / "assets" / f"{cfg.model.generation}_{cfg.concept_group.group_dir_name}_60"
                 if alt_path.exists():
                     data_path = alt_path
             except:
                 pass
        
        log.info(f"Using data path: {data_path}")
        
        run_response_computation(
            model_name_or_path=cfg.model.generation, # "gpt2" or path
            data_path=data_path,
            responses_path=responses_base, # Output to responses_base
            # concepts=, # If we want to filter, we could join cfg.concept_group.concepts
            model_cache=None, # Defaults
            tok_cache=None,
            seq_len=cfg.task.seq_len,
            num_per_concept=cfg.task.num_per_concept,
            inf_batch_size=cfg.task.inf_batch_size,
            device=cfg.task.device,
        )
        
    elif cfg.task.name == "compute_expertise":
        log.info("Running expertise computation...")
        
        if not responses_base.exists():
            log.error(f"Responses base directory not found: {responses_base}")
            return

        # In the user command: --root-dir .../Qwen3-30B-A3B-Instruct-2507_custom_60_responses/
        # --model-name gpt2
        # --concepts assets/.../concept_list.csv
        
        # We need to pass the concept list CSV. 
        # Similar logic for data_path as above to find concept_list.csv
        data_path = original_cwd / "assets" / cfg.concept_group.name
        # Check fallback
        if not data_path.exists():
             try:
                 alt_path = original_cwd / "assets" / f"{cfg.model.generation}_{cfg.concept_group.group_dir_name}_60"
                 if alt_path.exists():
                     data_path = alt_path
             except:
                 pass
        
        concept_list_path = data_path / "concept_list.csv"
        log.info(f"Using concept list: {concept_list_path}")
        
        run_expertise_computation(
            root_dir=responses_base,
            model_name=cfg.model.processing, # e.g. "gpt2"
            concepts=str(concept_list_path),
            k=cfg.task.k,
            show=cfg.task.show,
            skip=cfg.task.skip,
            black=cfg.task.black,
        )

    elif cfg.task.name == "unique_experts":
        if not responses_base.exists():
            log.error(f"Responses base directory not found: {responses_base}")
            return
            
        log.info(f"Concepts to process: {len(cfg.concept_group.concepts)}")
        
        # Iterate over concepts
        for concept in cfg.concept_group.concepts:
            log.info(f"Processing concept: {concept}")
            
            # Define concept-specific paths
            group_dir = cfg.concept_group.group_dir_name
            processing_model = cfg.model.processing
            
            concept_root = responses_base / processing_model / group_dir / concept
            
            # Input paths
            concept_responses_dir = concept_root / "responses"
            concept_expertise_csv = concept_root / "expertise" / "expertise.csv"
            
            # Output paths (structure mirrors inputs but under results root)
            concept_output_dir = output_base / concept
            
            # Check inputs
            if not concept_responses_dir.exists():
                log.warning(f"Responses not found for {concept}, skipping. (Expected: {concept_responses_dir})")
                continue
                
            if not concept_expertise_csv.exists():
                log.warning(f"Expertise CSV not found for {concept}, skipping. (Expected: {concept_expertise_csv})")
                continue
                
            try:
                # Step 1: Compute Correlations
                log.info(f"[{concept}] Step 1: Computing correlations...")
                run_correlation_analysis(
                    responses_dir=concept_responses_dir,
                    expertise_csv=concept_expertise_csv,
                    output_dir=concept_output_dir,
                    ap_threshold=cfg.task.ap_threshold,
                    correlation_threshold=cfg.task.correlation_threshold,
                    concept=concept,
                )
                
                # Step 2: Filter Unique Experts
                log.info(f"[{concept}] Step 2: Filtering unique experts...")
                run_filtering(
                    correlation_dir=concept_output_dir,
                    expertise_csv=concept_expertise_csv,
                    output_dir=concept_output_dir / "unique_experts",
                    correlation_threshold=cfg.task.correlation_threshold,
                    selection_strategy=cfg.task.selection_strategy,
                    concept=concept,
                    save_filtered_csv=True,
                )
                
                # Step 3: Visualize
                log.info(f"[{concept}] Step 3: Visualizing...")
                run_visualization(
                    correlation_dir=concept_output_dir,
                    output_dir=concept_output_dir / "visualizations",
                    concept=concept,
                    correlation_threshold=cfg.task.correlation_threshold,
                    max_layers=5, # Default from script
                    filtering_summary=concept_output_dir / "unique_experts" / f"{concept}_filtering_summary.json",
                )
                
                log.info(f"[{concept}] Completed successfully.")
                
            except Exception as e:
                log.error(f"[{concept}] Failed: {e}", exc_info=True)
                continue

    elif cfg.task.name == "expert_overlap":
        log.info("Running expert overlap analysis...")
        
        # Determine input directory
        if cfg.task.input_dir:
            # Use explicit input directory (resolved relative to original CWD)
            expertise_root = original_cwd / cfg.task.input_dir
            log.info(f"Using explicit input directory: {expertise_root}")
        else:
            # Default: Construct path to expertise directory (root of all concepts)
            group_dir = cfg.concept_group.group_dir_name
            processing_model = cfg.model.processing
            expertise_root = responses_base / processing_model / group_dir
            log.info(f"Using default expertise directory: {expertise_root}")
        
        # Paths for outputs
        overlap_output_dir = output_base / "expert_similarity"
        visualization_output_dir = output_base / "graph_outputs"
        
        if not expertise_root.exists():
            log.error(f"Expertise root directory not found: {expertise_root}")
            return

        try:
            # Step 1: Compute Overlap & Similarity Matrix
            log.info("Step 1: Computing expert overlap and similarity matrix...")
            run_overlap_analysis(
                output_dir=overlap_output_dir,
                threshold=cfg.task.ap_threshold,
                expertise_dir=expertise_root,
                save_expert_sets=True,
                save_matrix=True,
                pattern=cfg.task.pattern,
            )
            
            # Step 1.5: Compute Shared Experts Stats
            log.info("Step 1.5: Computing shared experts statistics...")
            shared_stats_output_dir = output_base / "shared_stats"
            
            run_shared_experts_stats(
                output_dir=shared_stats_output_dir,
                threshold=cfg.task.ap_threshold,
                expertise_dir=expertise_root,
                pattern=cfg.task.pattern,
            )
            
            # Step 2: Visualize Overlap
            log.info("Step 2: Visualizing expert overlap...")
            similarity_matrix_path = overlap_output_dir / f"similarity_matrix_tau_{cfg.task.ap_threshold:.1f}.csv"
            
            if not similarity_matrix_path.exists():
                log.error(f"Similarity matrix not found at {similarity_matrix_path}, skipping visualization.")
            else:
                # Get visualization params from config with defaults
                viz_cfg = cfg.task.get("visualization", {})
                
                run_overlap_visualization(
                    csv_path=str(similarity_matrix_path),
                    output_dir=str(visualization_output_dir),
                    low_threshold=viz_cfg.get("low_threshold", 0.2),
                    layout_mode=viz_cfg.get("layout_mode", "mds"),
                    topk=viz_cfg.get("topk", 10),
                    min_edge_width=viz_cfg.get("min_edge_width", 0.2),
                    max_edge_width=viz_cfg.get("max_edge_width", 15.0),
                    dpi=viz_cfg.get("dpi", 200),
                    fontsize=viz_cfg.get("fontsize", 9),
                    spring_k=viz_cfg.get("spring_k", 0.5),
                    hybrid_iterations=viz_cfg.get("hybrid_iterations", 100),
                    seed=viz_cfg.get("seed", 0),
                )
                
            log.info("Expert overlap analysis completed successfully.")
            
        except Exception as e:
            log.error(f"Expert overlap analysis failed: {e}", exc_info=True)

    elif cfg.task.name == "visualize_overlap":
        log.info("Running expert overlap visualization...")

        # CSV path (input)
        if cfg.task.csv_path:
            csv_path = pathlib.Path(cfg.task.csv_path)
            if not csv_path.is_absolute():
                csv_path = original_cwd / csv_path
            log.info(f"Using explicit CSV path: {csv_path}")
        else:
            # Default location logic matching 'expert_overlap' task default output
            # Assuming standard structure: results/expert_overlap/.../expert_similarity/similarity_matrix_...
            # But this is hard to guess without AP threshold.
            # Better to require csv_path OR try to find one in expected output location if run in same pipeline flow?
            # For decoupled task, user should likely provide it, OR we can default to searching in default response location?
            # Actually, similarity matrix is an OUTPUT of expert_overlap.
            # Let's error if not provided for now, or check a likely default if we knew the AP threshold.
            # Since AP threshold is not in this task config, we can't guess easily.
            # User must provide csv_path.
            log.error("csv_path must be provided for visualize_overlap task.")
            return

        # Output directory
        if cfg.task.output_dir:
             output_dir = pathlib.Path(cfg.task.output_dir)
             if not output_dir.is_absolute():
                 output_dir = original_cwd / output_dir
             log.info(f"Using explicit output directory: {output_dir}")
        else:
             # Default output to current Hydra run dir
             output_dir = output_base / "graph_outputs"
             log.info(f"Using default output directory: {output_dir}")

        if not csv_path.exists():
            log.error(f"Similarity matrix CSV not found at {csv_path}")
            return

        try:
            # Get visualization params from config with defaults
            viz_cfg = cfg.task.get("visualization", {})
            
            run_overlap_visualization(
                csv_path=str(csv_path),
                output_dir=str(output_dir),
                low_threshold=viz_cfg.get("low_threshold", 0.2),
                layout_mode=viz_cfg.get("layout_mode", "mds"),
                topk=viz_cfg.get("topk", 10),
                min_edge_width=viz_cfg.get("min_edge_width", 0.2),
                max_edge_width=viz_cfg.get("max_edge_width", 15.0),
                dpi=viz_cfg.get("dpi", 200),
                fontsize=viz_cfg.get("fontsize", 9),
                spring_k=viz_cfg.get("spring_k", 0.5),
                hybrid_iterations=viz_cfg.get("hybrid_iterations", 100),
                seed=viz_cfg.get("seed", 0),
            )
            log.info("Expert overlap visualization completed successfully.")
        except Exception as e:
            log.error(f"Expert overlap visualization failed: {e}", exc_info=True)

    else:
        log.error(f"Unknown task: {cfg.task.name}")

    log.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()

