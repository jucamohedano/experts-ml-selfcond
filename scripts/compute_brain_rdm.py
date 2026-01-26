#! /usr/bin/env python3
"""
Compute brain-region RDMs for the Mitchell 2008 dataset using 3D grid tessellation.

This implements Step 4 of the JuanProject.pdf analysis plan:
- Load fMRI data for 60 words (averaged over 6 trials)
- Tessellate the brain volume into a regular 3D grid
- For each non-empty region, compute a 60x60 RDM using 1 - Pearson correlation
"""

import json
import logging
import os
import pathlib
import pickle
from typing import Dict, List, Tuple, Union

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from selfcond.brain_data import load_brain_data, create_grid_regions

log = logging.getLogger(__name__)


def compute_brain_rdms(
    word_activations: np.ndarray,
    regions: Dict[Tuple[int, int, int], List[int]],
    min_voxels: int = 5,
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    Compute a 60x60 RDM for each brain region.

    Args:
        word_activations: Array of shape [60, V] (words x voxels).
        regions: Dict mapping region_id -> list of voxel indices.
        min_voxels: Minimum number of voxels required to compute an RDM.

    Returns:
        Dict mapping region_id -> 60x60 RDM matrix.
    """
    brain_rdms: Dict[Tuple[int, int, int], np.ndarray] = {}

    for region_id, voxel_indices in tqdm(regions.items(), desc="Computing Brain RDMs"):
        if len(voxel_indices) < min_voxels:
            continue

        region_acts = word_activations[:, voxel_indices]  # [60, n_voxels]

        # 1 - Pearson correlation across words
        # np.corrcoef over rows yields [60, 60]
        rdm = 1.0 - np.corrcoef(region_acts)
        brain_rdms[region_id] = rdm.astype(np.float32)

    return brain_rdms


def plot_rdm(rdm_matrix: np.ndarray,
             word_labels: List[str],
             output_path: pathlib.Path,
             region_id: Tuple[int, int, int],
             voxel_count: int) -> None:
    """
    Plot and save a brain RDM as a heatmap with word labels.
    
    Args:
        rdm_matrix: RDM of shape [60, 60]
        word_labels: List of 60 word names (alphabetical order)
        output_path: Path to save the plot
        region_id: Region identifier tuple (xi, yi, zi)
        voxel_count: Number of voxels in this region
    """
    # Create DataFrame for better labeling
    df_rdm = pd.DataFrame(rdm_matrix, index=word_labels, columns=word_labels)
    
    # Log statistics
    off_diag = rdm_matrix[~np.eye(rdm_matrix.shape[0], dtype=bool)]
    log.debug(f"Region {region_id} ({voxel_count} voxels): "
              f"Mean dissim={off_diag.mean():.3f}, "
              f"Max dissim={off_diag.max():.3f}, Min dissim={off_diag.min():.3f}")
    
    # Plot heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(df_rdm, cmap="viridis", vmin=0, vmax=2,
                square=True, linewidths=0.0, linecolor='white',
                cbar_kws={'label': 'Dissimilarity (1 - Pearson r)'})
    plt.title(f"Brain RDM - Region {region_id}\n({voxel_count} voxels)",
              fontsize=14, fontweight='bold')
    plt.xlabel("Words", fontsize=11)
    plt.ylabel("Words", fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save the RDM as CSV for easier inspection
    csv_path = output_path.with_suffix('.csv')
    df_rdm.to_csv(csv_path)
    log.debug(f"Saved RDM heatmap: {output_path}")
    log.debug(f"Saved RDM CSV: {csv_path}")


def run_brain_rdm(brain_data_path: Union[pathlib.Path, str],
                  grid_shape: Tuple[int, int, int],
                  min_voxels: int,
                  output_dir: pathlib.Path) -> None:
    """
    Main entry point for computing brain RDMs.
    Handles multiple subjects by computing RDMs per subject, then averaging.
    """
    log.info("Starting Brain RDM computation (tessellation)")
    log.info(f"Brain data path: {brain_data_path}")
    log.info(f"Grid shape: {grid_shape}")
    log.info(f"Min voxels per region: {min_voxels}")
    log.info(f"Output directory: {output_dir}")

    # Resolve path to absolute
    original_cwd = pathlib.Path(hydra.utils.get_original_cwd())
    brain_data_path = pathlib.Path(brain_data_path) if isinstance(brain_data_path, str) else brain_data_path
    if not brain_data_path.is_absolute():
        brain_data_path = original_cwd / brain_data_path

    if not brain_data_path.exists():
        raise FileNotFoundError(f"Brain data path not found: {brain_data_path}")

    # Collect .mat files: single file or directory
    if brain_data_path.is_file():
        mat_files = [brain_data_path]
    elif brain_data_path.is_dir():
        mat_files = sorted(brain_data_path.glob("*.mat"))
        if not mat_files:
            raise ValueError(f"No .mat files found in directory: {brain_data_path}")
    else:
        raise ValueError(f"Invalid brain_data_path: {brain_data_path} (must be a file or directory)")
    
    log.info(f"Found {len(mat_files)} .mat file(s)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each subject independently - NO averaging across subjects
    # RSA correlations should be computed per subject, then averaged (Step 5)
    all_brain_rdms = {}  # {subject_id: {region_id: RDM}}
    subject_info = {}    # {subject_id: {num_regions, num_voxels, ...}}
    word_labels = None   # Same across all subjects
    
    for mat_file in mat_files:
        subject_id = mat_file.stem  # e.g., "data-science-P1"
        log.info(f"Processing subject: {subject_id}")
        
        # Load single subject data
        word_activations, coords, word_labels = load_brain_data([mat_file])
        log.info(f"  Loaded {len(word_labels)} words, {coords.shape[0]} voxels")
        
        # Create regions for this subject
        regions = create_grid_regions(coords, tuple(grid_shape))
        log.info(f"  Created {len(regions)} regions")
        
        # Compute RDMs for this subject
        subject_rdms = compute_brain_rdms(word_activations, regions, min_voxels=min_voxels)
        log.info(f"  Computed {len(subject_rdms)} RDMs")
        
        # Store per-subject results
        all_brain_rdms[subject_id] = subject_rdms
        subject_info[subject_id] = {
            "num_voxels": int(coords.shape[0]),
            "num_regions_total": len(regions),
            "num_regions_kept": len(subject_rdms),
            "mat_file": str(mat_file),
        }
    
    total_rdms = sum(len(rdms) for rdms in all_brain_rdms.values())
    log.info(f"Completed: {len(all_brain_rdms)} subjects, {total_rdms} total RDMs")

    # Visualize sample RDMs from first subject
    sample_rdms_dir = output_dir / "sample_rdms"
    sample_rdms_dir.mkdir(exist_ok=True)
    
    first_subject_id = list(all_brain_rdms.keys())[0]
    first_subject_rdms = all_brain_rdms[first_subject_id]
    
    # Load first subject's regions for voxel counts
    first_mat = mat_files[0]
    _, first_coords, _ = load_brain_data([first_mat])
    first_regions = create_grid_regions(first_coords, tuple(grid_shape))
    
    region_voxel_counts = {rid: len(first_regions[rid]) for rid in first_subject_rdms.keys()}
    sorted_regions = sorted(region_voxel_counts.items(), key=lambda x: x[1], reverse=True)
    num_samples = min(10, len(first_subject_rdms))
    
    log.info(f"Visualizing {num_samples} sample Brain RDMs from {first_subject_id}...")
    for region_id, voxel_count in tqdm(sorted_regions[:num_samples], desc="Plotting sample RDMs"):
        rdm = first_subject_rdms[region_id]
        region_str = f"region_{region_id[0]}_{region_id[1]}_{region_id[2]}"
        plot_path = sample_rdms_dir / f"{first_subject_id}_{region_str}_rdm.png"
        plot_rdm(rdm, word_labels, plot_path, region_id, voxel_count)
    
    log.info(f"Saved {num_samples} sample RDM visualizations to {sample_rdms_dir}")

    # Save outputs
    # Save all subject RDMs in one file: {subject_id: {region_id: RDM}}
    rdms_file = output_dir / "brain_rdms.pkl"
    with rdms_file.open("wb") as f:
        pickle.dump(all_brain_rdms, f)
    log.info(f"Saved Brain RDMs to {rdms_file}")

    # Save word labels (same for all subjects)
    labels_file = output_dir / "word_labels.txt"
    with labels_file.open("w") as f:
        f.write("\n".join(word_labels))
    log.info(f"Saved word labels to {labels_file}")

    # Save per-subject info
    subject_info_file = output_dir / "subject_info.json"
    with subject_info_file.open("w") as f:
        json.dump(subject_info, f, indent=2)
    log.info(f"Saved subject info to {subject_info_file}")

    # Save metadata
    metadata = {
        "brain_data_paths": [str(f) for f in mat_files],
        "num_subjects": len(all_brain_rdms),
        "subject_ids": list(all_brain_rdms.keys()),
        "grid_shape": list(grid_shape),
        "min_voxels": min_voxels,
        "num_words": len(word_labels),
        "total_rdms": total_rdms,
        "note": "RDMs stored per-subject. RSA should correlate per-subject, then average correlations.",
    }
    metadata_file = output_dir / "brain_rdm_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata to {metadata_file}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    task_cfg = cfg.task

    # Handle brain_data_path: can be string or Path
    brain_data_path = task_cfg.brain_data_path
    
    grid_shape = tuple(task_cfg.grid_shape)
    min_voxels = int(task_cfg.min_voxels)

    # Hydra sets CWD to the run directory; use that as output_dir
    output_dir = pathlib.Path(os.getcwd())

    run_brain_rdm(
        brain_data_path=brain_data_path,
        grid_shape=grid_shape,
        min_voxels=min_voxels,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

