#!/usr/bin/env python3
"""
Script to compute shared experts across all concepts.

This script loads expertise results for all concepts and computes:
- Number of experts shared across ALL concepts (intersection)
- Total number of unique experts across all concepts (union)  
- Percentage of shared experts relative to total network size

Uses AP threshold to define expert neurons, following the methodology from
'Analyze the Neurons, not the Embeddings' (Fedzechkina et al., 2025).

Outputs a simple summary to console and JSON file.
"""

import argparse
import pathlib
import json

import numpy as np

from selfcond.data import concept_list_to_df
from selfcond.expert_analysis import extract_experts_from_csv, load_multiple_expert_sets


def load_expertise_for_concept(concept_dir: pathlib.Path, concept: str, concept_group: str, threshold: float):
    """
    Load expertise results for a specific concept using AP threshold.

    Args:
        concept_dir: Path to the concept directory
        concept: Concept name
        concept_group: Concept group
        threshold: AP threshold for defining expert neurons

    Returns:
        ExpertSet object, or None if not found
    """
    expertise_csv = concept_dir / "expertise" / "expertise.csv"
    if not expertise_csv.exists():
        print(f"No expertise.csv found for {concept_group}/{concept}")
        return None

    try:
        expert_set = extract_experts_from_csv(expertise_csv, threshold)
        return expert_set
    except Exception as e:
        print(f"Error loading expertise for {concept_group}/{concept}: {e}")
        return None


def run_shared_experts_stats(
    output_dir: pathlib.Path,
    threshold: float,
    expertise_dir: pathlib.Path = None,
    pattern: str = "**/expertise.csv",
    root_dir: pathlib.Path = None,
    model_name: str = None,
    concepts_requested = None,
):
    """
    Compute shared experts across all concepts.

    Args:
        output_dir: Where to save the output JSON
        threshold: AP threshold for defining expert neurons
        expertise_dir: Directory containing multiple expertise.csv files (searches recursively)
        pattern: Glob pattern to find expertise.csv files if expertise_dir is used
        root_dir: Root directory with responses (legacy/structured mode)
        model_name: Model name (legacy/structured mode)
        concepts_requested: List of concepts or path to concept list (legacy/structured mode)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_sets = []

    # Mode 1: Load from directory using pattern (Flexible/Hydra mode)
    if expertise_dir:
        print(f"Loading expert sets from {expertise_dir} with pattern '{pattern}'...")
        expert_sets = load_multiple_expert_sets(
            expertise_dir,
            threshold=threshold,
            pattern=pattern
        )
    
    # Mode 2: Load from structured paths (Legacy/Script mode)
    elif root_dir and model_name and concepts_requested:
        # Load concepts
        if isinstance(concepts_requested, list):
            concept_list = concepts_requested
        else:
            concept_df = concept_list_to_df(concepts_requested)
            concept_list = [(row["group"], row["concept"]) for _, row in concept_df.iterrows()]

        print(f"Processing {len(concept_list)} concepts from structured dirs with AP threshold {threshold}...")

        for group, concept in concept_list:
            concept_dir = root_dir / model_name / group / concept
            expert_set = load_expertise_for_concept(concept_dir, concept, group, threshold)
            if expert_set is not None:
                expert_sets.append(expert_set)
            else:
                print(f"Skipping {group}/{concept} due to missing expertise")
    
    else:
        print("Error: Must provide either expertise_dir OR (root_dir, model_name, concepts_requested)")
        return

    if len(expert_sets) == 0:
        print("No concepts with expertise results found!")
        return

    print(f"Loaded expertise for {len(expert_sets)} concepts")

    # Verify all expert sets have the same neuron space
    if len(expert_sets) > 1:
        reference_neuron_ids = expert_sets[0].neuron_ids
        total_neurons = expert_sets[0].total_neurons
        
        for i, expert_set in enumerate(expert_sets[1:], 1):
            if not np.array_equal(expert_set.neuron_ids, reference_neuron_ids):
                print(f"Error: Expert set {i} has different neuron space than reference")
                return
            if expert_set.total_neurons != total_neurons:
                print(f"Error: Expert set {i} has different total neurons than reference")
                return
    else:
        total_neurons = expert_sets[0].total_neurons

    # Compute intersection (experts shared across ALL concepts)
    shared_experts_mask = expert_sets[0].experts.copy()
    for expert_set in expert_sets[1:]:
        shared_experts_mask = shared_experts_mask & expert_set.experts

    # Compute union (all unique experts across all concepts)
    union_experts_mask = expert_sets[0].experts.copy()
    for expert_set in expert_sets[1:]:
        union_experts_mask = union_experts_mask | expert_set.experts

    # Count shared and total unique experts
    shared_experts_count = int(np.sum(shared_experts_mask))
    total_unique_experts_count = int(np.sum(union_experts_mask))

    # Calculate percentages
    if total_neurons > 0:
        shared_percentage_of_network = (shared_experts_count / total_neurons) * 100
        unique_percentage_of_network = (total_unique_experts_count / total_neurons) * 100
    else:
        shared_percentage_of_network = 0.0
        unique_percentage_of_network = 0.0

    if total_unique_experts_count > 0:
        shared_percentage_of_union = (shared_experts_count / total_unique_experts_count) * 100
    else:
        shared_percentage_of_union = 0.0

    # Prepare results
    results = {
        "threshold": threshold,
        "total_concepts": len(expert_sets),
        "total_neurons_in_network": total_neurons,
        "shared_experts_count": shared_experts_count,
        "total_unique_experts": total_unique_experts_count,
        "shared_percentage_of_network": round(shared_percentage_of_network, 2),
        "shared_percentage_of_union": round(shared_percentage_of_union, 2),
        "unique_experts_percentage_of_network": round(unique_percentage_of_network, 2),
    }

    # Save to JSON
    output_file = output_dir / f"shared_experts_summary_tau_{threshold:.1f}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved shared experts summary to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("SHARED EXPERTS SUMMARY")
    print("="*60)
    print(f"AP Threshold: {threshold}")
    print(f"Total concepts analyzed: {len(expert_sets)}")
    print(f"Total neurons in network: {total_neurons:,}")
    print(f"Shared experts (across ALL concepts): {shared_experts_count:,}")
    print(f"Total unique experts: {total_unique_experts_count:,}")
    print(f"Shared experts as % of network: {shared_percentage_of_network:.2f}%")
    print(f"Shared experts as % of union: {shared_percentage_of_union:.2f}%")
    print(f"Unique experts as % of network: {unique_percentage_of_network:.2f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        prog="compute_shared_experts.py",
        description=(
            "Compute shared experts across all concepts and report intersection/union statistics. "
            "Uses AP threshold to define expert neurons following the methodology from "
            "'Analyze the Neurons, not the Embeddings' (Fedzechkina et al., 2025). "
            "Assumes expertise results exist for each concept (run compute_expertise.py first)."
        ),
    )
    parser.add_argument(
        "--root-dir",
        type=pathlib.Path,
        help="Root directory with responses. Should contain responses_dir/model/concept_group/concept/expertise",
        required=True,
    )
    parser.add_argument("--model-name", type=str, help="The model name", required=True)
    parser.add_argument("--concepts", type=str, help="Path to concept_list.csv or comma-separated concepts")
    parser.add_argument("--output-dir", type=pathlib.Path, help="Output directory for JSON (default: root_dir/model_name)", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="AP threshold for defining expert neurons (default: 0.5)",
    )

    args = parser.parse_args()

    root_dir = args.root_dir

    # Load concepts
    if not args.concepts:
        # Use the standard dataset location
        concepts_file = root_dir / "assets" / "openai_custom_60" / "concept_list.csv"
        if not concepts_file.exists():
            raise FileNotFoundError(f"No --concepts provided and {concepts_file} not found")
        concepts_requested = concepts_file
    else:
        if "," in args.concepts:
            concepts_requested = args.concepts.split(",")
        else:
            concepts_requested = pathlib.Path(args.concepts)

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = root_dir / args.model_name / "shared_experts"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_shared_experts_stats(
        output_dir=output_dir,
        threshold=args.threshold,
        root_dir=root_dir,
        model_name=args.model_name,
        concepts_requested=concepts_requested
    )
