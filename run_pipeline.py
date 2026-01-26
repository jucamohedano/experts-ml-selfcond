#!/usr/bin/env python3
"""
Hydra-based pipeline for unique expert analysis.
Replaces find_unique_experts.sh and find_unique_experts_parallel.sh.
"""

import hydra
from omegaconf import DictConfig
import pathlib
import logging

# Import analysis scripts directly
from scripts.compute_neuron_correlations import run_correlation_analysis
from scripts.filter_unique_experts import run_filtering
from scripts.visualize_neuron_correlations import run_visualization
from scripts.analyze_expert_overlap import run_overlap_analysis
from scripts.visualize_expert_overlap import run_overlap_visualization
from scripts.compute_responses import run_response_computation
from scripts.compute_expertise import run_expertise_computation
from scripts.compute_shared_experts_stats import run_shared_experts_stats
from scripts.subspace_gaze import run_subspace_gaze
from scripts.steering_validation import run_steering_validation
from scripts.word_features import extract_word_features
from scripts.compute_brain_rdm import run_brain_rdm
from scripts.compute_rsa import run_rsa

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
            model_name_or_path=cfg.model.processing, # "gpt2" or path
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

    elif cfg.task.name == "subspace_gaze":
        activations_path = pathlib.Path(cfg.task.activations_path)
        if not activations_path.is_absolute():
            activations_path = original_cwd / cfg.task.activations_path

        expertise_path = pathlib.Path(cfg.task.expertise_path)
        if not expertise_path.is_absolute():
            expertise_path = original_cwd / cfg.task.expertise_path

        run_subspace_gaze(
            activations_path=activations_path,
            expertise_path=expertise_path,
            ap_threshold=cfg.task.ap_threshold,
            layer=cfg.task.layer,
            method=cfg.task.method,
        )

    elif cfg.task.name == "steering_validation":
        vector_base_dir = pathlib.Path(cfg.task.vector_base_dir)
        if not vector_base_dir.is_absolute():
            vector_base_dir = original_cwd / cfg.task.vector_base_dir

        run_steering_validation(
            vector_base_dir=vector_base_dir,
            concept=cfg.task.get("concept", None),
            concept_types=cfg.task.concept_types,
            layer_indices=cfg.task.layer_indices,
            layer_types=cfg.task.layer_types,
            vector_suffixes=cfg.task.vector_suffixes,
            coeff_start=cfg.task.get("coeff_start", -5.0),
            coeff_end=cfg.task.get("coeff_end", 5.0),
            coeff_steps=cfg.task.get("coeff_steps", 21),
            max_new_tokens=cfg.task.max_new_tokens,
            num_samples=cfg.task.get("num_samples", 5),
            seed=cfg.task.get("seed", 42),
            output_dir=output_base
        )

    elif cfg.task.name == "word_feature_extraction":
        log.info("Running word feature extraction...")
        
        activations_path = pathlib.Path(cfg.task.activations_path)
        if not activations_path.is_absolute():
            activations_path = original_cwd / cfg.task.activations_path

        use_all_neurons = cfg.task.get("use_all_neurons", False)
        
        # Optional unique experts path
        unique_experts_path = None
        if cfg.task.get("unique_experts_path"):
            unique_experts_path = pathlib.Path(cfg.task.unique_experts_path)
            if not unique_experts_path.is_absolute():
                unique_experts_path = original_cwd / unique_experts_path
        
        # expertise_path is optional when use_all_neurons=True OR unique_experts_path is set
        expertise_path = None
        if cfg.task.expertise_path:
            expertise_path = pathlib.Path(cfg.task.expertise_path)
            if not expertise_path.is_absolute():
                expertise_path = original_cwd / cfg.task.expertise_path
        elif not use_all_neurons and not unique_experts_path:
            log.error("expertise_path is required when use_all_neurons=False and unique_experts_path is not set")
            return

        extract_word_features(
            activations_path=activations_path,
            expertise_path=expertise_path,
            ap_threshold=cfg.task.ap_threshold,
            layer=cfg.task.layer,
            use_all_neurons=use_all_neurons,
            unique_experts_path=unique_experts_path,
        )

    elif cfg.task.name == "brain_rdm":
        log.info("Running brain RDM computation (tessellation)...")

        # Handle flexible input: can be string, list, or Path
        brain_data_path = cfg.task.brain_data_path
        # Path resolution will be handled in run_brain_rdm via collect_mat_files

        run_brain_rdm(
            brain_data_path=brain_data_path,
            grid_shape=tuple(cfg.task.grid_shape),
            min_voxels=cfg.task.min_voxels,
            output_dir=output_base,
        )

    elif cfg.task.name == "rsa":
        log.info("Running RSA analysis...")
        
        # Build condition dict
        condition_rdms = {}
        
        if cfg.task.expert_rdms_path:
            expert_path = pathlib.Path(cfg.task.expert_rdms_path)
            if not expert_path.is_absolute():
                expert_path = original_cwd / expert_path
            condition_rdms["expert"] = expert_path
        
        if cfg.task.full_rdms_path:
            full_path = pathlib.Path(cfg.task.full_rdms_path)
            if not full_path.is_absolute():
                full_path = original_cwd / full_path
            condition_rdms["full"] = full_path
        
        if len(condition_rdms) < 2:
            log.error("At least 2 conditions required")
            return
        
        brain_rdms_path = pathlib.Path(cfg.task.brain_rdms_path)
        if not brain_rdms_path.is_absolute():
            brain_rdms_path = original_cwd / brain_rdms_path
        
        # Handle layers config (can be null, string, or list)
        layer_filter = cfg.task.get("layers", None)
        if layer_filter is not None and isinstance(layer_filter, str):
            layer_filter = [layer_filter]
        
        run_rsa(
            brain_rdms_path=brain_rdms_path,
            condition_rdms=condition_rdms,
            output_dir=output_base,
            layer_filter=layer_filter,
            apply_fdr=cfg.task.get("apply_fdr", True),
            alpha=cfg.task.get("alpha", 0.05),
        )

    else:
        log.error(f"Unknown task: {cfg.task.name}")

    log.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()

