"""
Moving layer profiles onto the block axis changes nothing on sublayer scopes and loses no experts.

Checks:
  1. On a single-sublayer frame to_block_axis is an identity relabel, so layer_profile_matrices
     returns identical jsd, similarity and z with and without it. The fixture has 20 words so
     the count-matched z is finite and its arm of the comparison can fail.
  2. On a multi-sublayer frame block aggregation keeps every word's expert count.
  3. Module 2's profile call sites pass a block-axis frame.

Usage, from scripts/:
    python tests/check_block_axis_identity.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.analysis_scopes import to_block_axis
from modules.shared_layer_profile_measures import layer_profile_matrices
from _report import run

BLOCKS = 12
WORDS = 20
CALL_SITES = [
    ("modules/module_2_similarities.py", r"layer_profile_metric_matrices\(to_block_axis\("),
    ("modules/module_2_similarities.py", r"pair_layer_profile_vectors\(to_block_axis\("),
]


def make_frame(rows):
    """Expert rows with layer_name as a categorical, as the executor builds them."""
    df = pd.DataFrame(rows, columns=["concept", "layer_idx", "unit", "layer_name"])
    df['layer_name'] = df['layer_name'].astype('category')
    return df


def checks(failures: list) -> str:
    single_rows = []
    for word_index in range(WORDS):
        word = f"word_{word_index:02d}"
        for step in range(6 + word_index % 5):
            block = (word_index * 5 + step * 3 + step * step) % BLOCKS
            single_rows.append((word, block + 1, step + word_index, f"{block + 1}.L.{block}.mlp.c_fc"))
    single = make_frame(single_rows)
    items = [f"word_{i:02d}" for i in range(WORDS)]
    plain = layer_profile_matrices(single, items)
    wrapped = layer_profile_matrices(to_block_axis(single), items)
    for name, a, b in zip(("jsd", "sim", "z"), plain, wrapped):
        if not np.allclose(a, b, equal_nan=True):
            failures.append(f"single-sublayer identity broken in {name}")
    if not np.isfinite(plain[2]).any():
        failures.append("z is all NaN on the fixture, so the z arm is vacuous, widen the fixture")

    multi = make_frame([
        ("cat", 1, 0, "1.L.0.attn.c_attn"), ("cat", 3, 0, "3.L.0.mlp.c_fc"),
        ("dog", 2, 1, "2.L.0.attn.c_proj"), ("dog", 7, 1, "7.L.1.mlp.c_fc"),
    ])
    if not multi.groupby("concept").size().equals(to_block_axis(multi).groupby("concept").size()):
        failures.append("block aggregation changed per-word expert counts")

    scripts = Path(__file__).resolve().parents[1]
    for path, pattern in CALL_SITES:
        if not re.search(pattern, (scripts / path).read_text()):
            failures.append(f"{path}: expected a call through to_block_axis, not found")
    return f"{WORDS}-word identity fixture, {len(CALL_SITES)} call sites"


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
