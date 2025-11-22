#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import json
import pathlib
import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExpertSet:
    """
    Represents a set of expert neurons for a specific concept.
    
    Attributes:
        concept: The concept name
        concept_group: The concept group/category
        experts: Boolean array indicating which neurons are experts
        neuron_ids: Array of unique neuron identifiers
        threshold: AP threshold used to define experts
        total_neurons: Total number of neurons in the model
    """
    concept: str
    concept_group: str
    experts: np.ndarray  # Boolean array
    neuron_ids: np.ndarray  # String array of neuron identifiers
    threshold: float
    total_neurons: int
    
    def __post_init__(self):
        """Validate the expert set after initialization."""
        assert len(self.experts) == len(self.neuron_ids), "Experts and neuron_ids must have same length"
        assert self.experts.dtype == bool, "Experts array must be boolean"
        assert np.sum(self.experts) <= self.total_neurons, "Number of experts cannot exceed total neurons"
    
    @property
    def num_experts(self) -> int:
        """Number of expert neurons."""
        return int(np.sum(self.experts))
    
    @property
    def expert_neuron_ids(self) -> np.ndarray:
        """Get the neuron IDs of expert neurons only."""
        return self.neuron_ids[self.experts]
    
    def intersection(self, other: 'ExpertSet') -> np.ndarray:
        """
        Compute intersection of expert sets.
        
        Args:
            other: Another ExpertSet
            
        Returns:
            Boolean array indicating neurons that are experts in both sets
        """
        assert len(self.neuron_ids) == len(other.neuron_ids), "Expert sets must have same neuron space"
        assert np.array_equal(self.neuron_ids, other.neuron_ids), "Expert sets must have same neuron ordering"
        return self.experts & other.experts
    
    def union(self, other: 'ExpertSet') -> np.ndarray:
        """
        Compute union of expert sets.
        
        Args:
            other: Another ExpertSet
            
        Returns:
            Boolean array indicating neurons that are experts in either set
        """
        assert len(self.neuron_ids) == len(other.neuron_ids), "Expert sets must have same neuron space"
        assert np.array_equal(self.neuron_ids, other.neuron_ids), "Expert sets must have same neuron ordering"
        return self.experts | other.experts
    
    def jaccard_similarity(self, other: 'ExpertSet') -> float:
        """
        Compute Jaccard similarity with another expert set.
        
        Args:
            other: Another ExpertSet
            
        Returns:
            Jaccard similarity coefficient J(A,B) = |A ∩ B| / |A ∪ B|
        """
        intersection = self.intersection(other)
        union = self.union(other)
        
        union_size = np.sum(union)
        if union_size == 0:
            return 0.0  # Both sets are empty
        
        intersection_size = np.sum(intersection)
        return float(intersection_size) / float(union_size)
    
    def save(self, output_path: pathlib.Path) -> None:
        """
        Save expert set to disk.
        
        Args:
            output_path: Path where to save the expert set
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save expert boolean array
        np.save(output_path / f"{self.concept}_experts.npy", self.experts)
        
        # Save neuron IDs
        np.save(output_path / f"{self.concept}_neuron_ids.npy", self.neuron_ids)
        
        # Save metadata
        metadata = {
            "concept": self.concept,
            "concept_group": self.concept_group,
            "threshold": self.threshold,
            "total_neurons": self.total_neurons,
            "num_experts": self.num_experts
        }
        
        with (output_path / f"{self.concept}_metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, output_path: pathlib.Path, concept: str) -> 'ExpertSet':
        """
        Load expert set from disk.
        
        Args:
            output_path: Path where the expert set is saved
            concept: Concept name
            
        Returns:
            Loaded ExpertSet
        """
        # Load expert boolean array
        experts = np.load(output_path / f"{concept}_experts.npy")
        
        # Load neuron IDs
        neuron_ids = np.load(output_path / f"{concept}_neuron_ids.npy")
        
        # Load metadata
        with (output_path / f"{concept}_metadata.json").open("r") as f:
            metadata = json.load(f)
        
        return cls(
            concept=metadata["concept"],
            concept_group=metadata["concept_group"],
            experts=experts,
            neuron_ids=neuron_ids,
            threshold=metadata["threshold"],
            total_neurons=metadata["total_neurons"]
        )


def create_neuron_id(layer: str, unit: int) -> str:
    """
    Create a unique neuron identifier from layer name and unit index.
    
    Args:
        layer: Layer name (e.g., 'transformer.h.0.attn.c_attn:0')
        unit: Unit index within the layer
        
    Returns:
        Unique neuron identifier string
    """
    return f"{layer}:{unit}"


def extract_experts_from_csv(
    expertise_csv_path: pathlib.Path,
    threshold: float = 0.5
) -> ExpertSet:
    """
    Extract expert neurons from an expertise.csv file.
    
    Args:
        expertise_csv_path: Path to the expertise.csv file
        threshold: AP threshold for defining expert neurons (default: 0.5)
        
    Returns:
        ExpertSet containing the expert neurons for this concept
    """
    print(f"Loading expertise data from {expertise_csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(expertise_csv_path)

    # Ensure deterministic ordering of neurons regardless of how the CSV was saved
    # This assumes 'layer' and 'unit' columns exist, which you validate later anyway
    if 'layer' in df.columns and 'unit' in df.columns:
        df = df.sort_values(by=['layer', 'unit'], ascending=[True, True])
    
    # Validate required columns
    required_columns = ['ap', 'layer', 'unit', 'concept', 'group']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Get concept information
    concept = df['concept'].iloc[0]
    concept_group = df['group'].iloc[0]
    
    # Verify all rows have the same concept
    if not (df['concept'] == concept).all():
        raise ValueError("All rows must have the same concept")
    
    # Create unique neuron identifiers
    neuron_ids = np.array([
        create_neuron_id(row['layer'], row['unit']) 
        for _, row in df.iterrows()
    ])
    
    # Apply threshold to create expert set
    experts = (df['ap'].values > threshold).astype(bool)
    
    total_neurons = len(df)
    num_experts = np.sum(experts)
    
    print(f"Concept: {concept_group}/{concept}")
    print(f"Total neurons: {total_neurons}")
    print(f"Expert neurons (AP > {threshold}): {num_experts} ({100*num_experts/total_neurons:.1f}%)")
    
    return ExpertSet(
        concept=concept,
        concept_group=concept_group,
        experts=experts,
        neuron_ids=neuron_ids,
        threshold=threshold,
        total_neurons=total_neurons
    )


def compute_jaccard_similarity(expert_set_a: ExpertSet, expert_set_b: ExpertSet) -> float:
    """
    Compute Jaccard similarity between two expert sets.
    
    Args:
        expert_set_a: First expert set
        expert_set_b: Second expert set
        
    Returns:
        Jaccard similarity coefficient
    """
    return expert_set_a.jaccard_similarity(expert_set_b)


def create_similarity_matrix(expert_sets: t.List[ExpertSet]) -> t.Tuple[np.ndarray, t.List[str]]:
    """
    Create a similarity matrix for multiple expert sets.
    
    Args:
        expert_sets: List of ExpertSet objects
        
    Returns:
        Tuple of (similarity_matrix, concept_names)
        - similarity_matrix: NxN matrix of Jaccard similarities
        - concept_names: List of concept names corresponding to matrix rows/columns
    """
    n_concepts = len(expert_sets)
    similarity_matrix = np.zeros((n_concepts, n_concepts))
    concept_names = [es.concept for es in expert_sets]
    
    print(f"Computing similarity matrix for {n_concepts} concepts...")
    
    for i in range(n_concepts):
        for j in range(n_concepts):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity is 1.0
            elif i < j:  # Only compute upper triangle
                similarity = compute_jaccard_similarity(expert_sets[i], expert_sets[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Matrix is symmetric
    
    return similarity_matrix, concept_names


def save_similarity_matrix(
    similarity_matrix: np.ndarray,
    concept_names: t.List[str],
    output_path: pathlib.Path,
    filename: str = "similarity_matrix.csv"
) -> None:
    """
    Save similarity matrix as CSV file.
    
    Args:
        similarity_matrix: NxN similarity matrix
        concept_names: List of concept names
        output_path: Directory where to save the file
        filename: Name of the output file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with concept names as index and columns
    df = pd.DataFrame(
        similarity_matrix,
        index=concept_names,
        columns=concept_names
    )
    
    # Save to CSV
    output_file = output_path / filename
    df.to_csv(output_file)
    print(f"Similarity matrix saved to {output_file}")


def load_multiple_expert_sets(
    expertise_dir: pathlib.Path,
    threshold: float = 0.5,
    pattern: str = "**/expertise.csv"
) -> t.List[ExpertSet]:
    """
    Load multiple expert sets from a directory containing expertise.csv files.
    
    Args:
        expertise_dir: Directory containing expertise.csv files
        threshold: AP threshold for defining experts
        pattern: Glob pattern to find expertise.csv files
        
    Returns:
        List of ExpertSet objects
    """
    expertise_files = list(expertise_dir.glob(pattern))
    
    if not expertise_files:
        raise ValueError(f"No expertise.csv files found in {expertise_dir} with pattern {pattern}")
    
    print(f"Found {len(expertise_files)} expertise.csv files")
    
    expert_sets = []
    for csv_file in sorted(expertise_files):
        try:
            expert_set = extract_experts_from_csv(csv_file, threshold)
            expert_sets.append(expert_set)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(expert_sets)} expert sets")
    return expert_sets
