#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import json
import pathlib
import sys

import numpy as np

from selfcond.expert_analysis import (
    extract_experts_from_csv,
    load_multiple_expert_sets,
    create_similarity_matrix,
    save_similarity_matrix,
)


def main():
    parser = argparse.ArgumentParser(
        prog="compute_expert_similarity.py",
        description=(
            "Compute expert neuron similarity between concepts using the method from "
            "'Analyze the Neurons, not the Embeddings' (Fedzechkina et al., 2025). "
            "Extracts expert neurons from expertise.csv files and computes Jaccard "
            "similarity between concept pairs."
        ),
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--expertise-csv",
        type=pathlib.Path,
        help="Path to a single expertise.csv file",
    )
    input_group.add_argument(
        "--expertise-dir",
        type=pathlib.Path,
        help="Directory containing multiple expertise.csv files (searches recursively)",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Directory where to save expert sets and similarity matrices",
        required=True,
    )
    
    # Analysis parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="AP threshold for defining expert neurons (default: 0.5)",
    )
    
    # Output options
    parser.add_argument(
        "--save-expert-sets",
        action="store_true",
        help="Save individual expert sets as .npy files",
    )
    parser.add_argument(
        "--save-similarity-matrix",
        action="store_true",
        help="Save similarity matrix as CSV file",
    )
    
    # Processing options
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/expertise.csv",
        help="Glob pattern to find expertise.csv files (default: **/expertise.csv)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.threshold < 0 or args.threshold > 1:
        print("Error: threshold must be between 0 and 1")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Expert Neuron Similarity Analysis")
    print(f"Threshold: {args.threshold}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    # Load expert sets
    expert_sets = []
    
    if args.expertise_csv:
        print(f"Processing single file: {args.expertise_csv}")
        if not args.expertise_csv.exists():
            print(f"Error: File {args.expertise_csv} does not exist")
            sys.exit(1)
        
        expert_set = extract_experts_from_csv(args.expertise_csv, args.threshold)
        expert_sets.append(expert_set)
        
    else:  # args.expertise_dir
        print(f"Processing directory: {args.expertise_dir}")
        if not args.expertise_dir.exists():
            print(f"Error: Directory {args.expertise_dir} does not exist")
            sys.exit(1)
        
        expert_sets = load_multiple_expert_sets(
            args.expertise_dir, 
            threshold=args.threshold,
            pattern=args.pattern
        )
        
        if not expert_sets:
            print("Error: No valid expert sets could be loaded")
            sys.exit(1)
    
    print(f"\nLoaded {len(expert_sets)} expert sets")
    
    # Save individual expert sets if requested
    if args.save_expert_sets:
        print("\nSaving individual expert sets...")
        expert_sets_dir = args.output_dir / "expert_sets"
        expert_sets_dir.mkdir(exist_ok=True)
        
        for expert_set in expert_sets:
            expert_set.save(expert_sets_dir)
        
        print(f"Expert sets saved to {expert_sets_dir}")
    
    # Compute and save similarity matrix if requested or if multiple concepts
    if args.save_similarity_matrix or len(expert_sets) > 1:
        print("\nComputing similarity matrix...")
        
        similarity_matrix, concept_names = create_similarity_matrix(expert_sets)
        
        # Save similarity matrix
        save_similarity_matrix(
            similarity_matrix,
            concept_names,
            args.output_dir,
            f"similarity_matrix_tau_{args.threshold:.1f}.csv"
        )
        
        # Compute similarity statistics (only for multiple concepts)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        if len(upper_triangle) > 0:
            mean_sim = float(np.mean(upper_triangle))
            std_sim = float(np.std(upper_triangle))
            min_sim = float(np.min(upper_triangle))
            max_sim = float(np.max(upper_triangle))
        else:
            # Single concept case
            mean_sim = None
            std_sim = None
            min_sim = None
            max_sim = None
        
        # Save analysis metadata
        metadata = {
            "analysis_type": "expert_neuron_similarity",
            "threshold": args.threshold,
            "num_concepts": len(expert_sets),
            "concepts": [
                {
                    "name": es.concept,
                    "group": es.concept_group,
                    "num_experts": es.num_experts,
                    "total_neurons": es.total_neurons,
                    "expert_percentage": 100 * es.num_experts / es.total_neurons
                }
                for es in expert_sets
            ],
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "min_similarity": min_sim,
            "max_similarity": max_sim,
        }
        
        metadata_file = args.output_dir / f"analysis_metadata_tau_{args.threshold:.1f}.json"
        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Analysis metadata saved to {metadata_file}")
        
        # Print summary statistics
        if len(expert_sets) > 1:
            print(f"\nSimilarity Matrix Summary:")
            print(f"  Mean similarity: {mean_sim:.3f}")
            print(f"  Std similarity: {std_sim:.3f}")
            print(f"  Min similarity: {min_sim:.3f}")
            print(f"  Max similarity: {max_sim:.3f}")
        else:
            print(f"\nSingle concept analysis - no pairwise similarities to compute.")
    
    # Print individual expert set statistics
    print(f"\nExpert Set Statistics:")
    for expert_set in expert_sets:
        print(f"  {expert_set.concept_group}/{expert_set.concept}: "
              f"{expert_set.num_experts}/{expert_set.total_neurons} "
              f"({100*expert_set.num_experts/expert_set.total_neurons:.1f}%) experts")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
