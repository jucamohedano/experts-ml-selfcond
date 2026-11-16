"""
Module 5's depth bands split the blocks into equal quarters and decompose the Jaccard index exactly.

Checks:
  1. depth_bands gives 7-7-7-7 blocks on 28 blocks and 3-3-3-3 on 12.
  2. The band intersections sum to the full intersection, and the band Jaccard indices weighted by
     each band's share of the union give back the full Jaccard index.
  3. A band where neither word holds a unit has union 0 and band Jaccard 0.
  4. The block count comes from the architecture, so a frame whose experts sit only in blocks 0 to 10
     of 28 still puts block 10 in the second band, and a sublayer scope whose empty blocks were dropped
     from the categories (filter_expert_data_to_sublayer) keeps its block count, an empty last block
     of GPT-2's mlp.c_fc included.

Usage, from scripts/:
    python tests/check_module5_depth_bands.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.module_5_typicality_prediction import (band_jaccard, block_coordinates, depth_band_overlap,
                                                    depth_bands)
from modules.shared_expert_set_measures import expert_set_overlap_matrices
from _report import run


def expert_frame(units: dict, n_blocks: int, sublayer: str = "mlp.gate_proj") -> pd.DataFrame:
    """Expert rows of a one-sublayer architecture, units given per concept as (block, unit) pairs."""
    names = [f"{b + 1}.L.{b}.{sublayer}" for b in range(n_blocks)]
    rows = [(concept, block + 1, unit, names[block]) for concept, pairs in units.items() for block, unit in pairs]
    frame = pd.DataFrame(rows, columns=["concept", "layer_idx", "unit", "layer_name"])
    frame["layer_name"] = pd.Categorical(frame["layer_name"], categories=names, ordered=True)
    return frame


def checks(failures: list) -> str:
    assert depth_bands(np.arange(28), 28).tolist() == [0] * 7 + [1] * 7 + [2] * 7 + [3] * 7
    assert depth_bands(np.arange(12), 12).tolist() == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3

    rng = np.random.default_rng(0)
    units = {"a": [(int(b), int(u)) for b, u in zip(rng.integers(0, 21, 60), rng.integers(0, 30, 60))],
             "b": [(int(b), int(u)) for b, u in zip(rng.integers(0, 28, 80), rng.integers(0, 30, 80))],
             "c": [(int(b), int(u)) for b, u in zip(rng.integers(0, 21, 40), rng.integers(0, 30, 40))]}
    df = expert_frame(units, 28)
    items = ["a", "b", "c"]
    inter, union = depth_band_overlap(df, items)
    full_inter, full_j, _ = expert_set_overlap_matrices(df, items)
    assert inter.shape == union.shape == (4, 3, 3), inter.shape
    assert np.allclose(inter.sum(axis=0), full_inter), "band intersections must sum to the full intersection"
    weighted = (band_jaccard(inter, union) * union).sum(axis=0) / union.sum(axis=0)
    assert np.allclose(weighted, full_j), "J must equal the union-weighted band Jaccard indices"
    assert union[3, 0, 2] == 0 and band_jaccard(inter, union)[3, 0, 2] == 0.0, "an empty band gives 0"

    sparse_df = expert_frame({"a": [(b, 1) for b in range(11)]}, 28)
    blocks, n_blocks = block_coordinates(sparse_df)
    assert n_blocks == 28, n_blocks
    assert set(depth_bands(blocks, n_blocks)[blocks == 10]) == {1}, "block 10 of 28 is in the second band"
    trimmed = expert_frame({"a": [(b, 1) for b in range(28) if b != 5]}, 28)
    trimmed["layer_name"] = trimmed["layer_name"].cat.remove_unused_categories()
    assert block_coordinates(trimmed)[1] == 28, "an empty interior block must not shrink the block count"
    last_empty = expert_frame({"a": [(b, 1) for b in range(11)]}, 12, "mlp.c_fc")
    last_empty["layer_name"] = last_empty["layer_name"].cat.remove_unused_categories()
    assert block_coordinates(last_empty)[1] == 12, "an empty last block must not shrink GPT-2's block count"
    return (f"quarters 7-7-7-7 and 3-3-3-3, J rebuilt from bands to {np.abs(weighted - full_j).max():.1e}, "
            f"empty band 0, block count from the categories")


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
