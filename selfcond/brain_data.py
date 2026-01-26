#
# For licensing see accompanying LICENSE file.
# Brain data utilities for Mitchell 2008 dataset.
#

import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import scipy.io


def load_brain_data(mat_paths: pathlib.Path | List[pathlib.Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load Mitchell 2008 brain data and average trials per word.
    For single subject use (multi-subject handled at RDM level).

    Args:
        mat_paths: Single file path or list with one file path.

    Returns:
        word_activations: Array of shape [60, V] with averaged responses per word.
        voxel_coords: Array of shape [V, 3] with x, y, z voxel coordinates.
        word_labels: List of 60 word strings in alphabetical order.
    """
    # Normalize input to list
    if isinstance(mat_paths, pathlib.Path):
        mat_paths = [mat_paths]
    
    if len(mat_paths) > 1:
        raise ValueError("load_brain_data expects a single file. "
                        "For multiple subjects, process separately and average RDMs.")
    
    mat_path = mat_paths[0]
    mat_data = scipy.io.loadmat(str(mat_path))

    # Voxel coordinates
    meta = mat_data["meta"][0, 0]
    coords: np.ndarray = meta["colToCoord"]  # (V, 3)

    # Trial data and metadata
    raw_data: np.ndarray = mat_data["data"]  # (360, 1), each entry (1, V)
    info: np.ndarray = mat_data["info"]  # (1, 360)

    # Group trials by word
    word_trials: Dict[str, List[np.ndarray]] = defaultdict(list)
    n_trials = raw_data.shape[0]
    for i in range(n_trials):
        word = info[0, i]["word"][0]
        voxels = raw_data[i, 0].flatten()  # (V,)
        word_trials[word].append(voxels)

    # Average per word
    word_labels = sorted(word_trials.keys())
    word_activations = np.array(
        [np.mean(word_trials[w], axis=0) for w in word_labels], dtype=np.float32
    )  # (60, V)

    return word_activations, coords.astype(np.float32), word_labels


def create_grid_regions(
    coords: np.ndarray, grid_shape: Tuple[int, int, int] = (11, 11, 9)
) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Tessellate brain into 3D grid regions using voxel coordinates.

    Args:
        coords: Array of shape [V, 3] with voxel coordinates.
        grid_shape: Number of bins along (x, y, z) axes, e.g. (11, 11, 9).

    Returns:
        regions: Dict mapping region_id (xi, yi, zi) to list of voxel indices.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape [V, 3], got {coords.shape}")

    nx, ny, nz = grid_shape

    x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), nx + 1)
    y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), ny + 1)
    z_bins = np.linspace(coords[:, 2].min(), coords[:, 2].max(), nz + 1)

    x_idx = np.digitize(coords[:, 0], x_bins)
    y_idx = np.digitize(coords[:, 1], y_bins)
    z_idx = np.digitize(coords[:, 2], z_bins)

    regions: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i, (xi, yi, zi) in enumerate(zip(x_idx, y_idx, z_idx)):
        regions[(int(xi), int(yi), int(zi))].append(i)

    return dict(regions)

