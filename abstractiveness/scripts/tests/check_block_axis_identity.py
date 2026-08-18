"""Asserts the two properties the block-axis change relies on.

1. On a single-sublayer frame, to_block_axis is an identity relabel, so
   layer_profile_matrices returns identical values with and without it. This is
   what guarantees every sublayers/ scope reproduces byte for byte. The fixture is
   sized past the null floor of count_matched_z so the z output is defined and the
   z arm of that comparison can genuinely fail.
2. On a multi-sublayer frame, block aggregation preserves each word's total
   expert count, so MIN_PROFILE_EXPERTS gating and row sets cannot change.
3. The three call sites actually pass a block-axis frame.
Run from scripts/: python tests/check_block_axis_identity.py.
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.helpers import to_block_axis, layer_profile_matrices

failures = []

def make_frame(rows):
    df = pd.DataFrame(rows, columns=["concept", "layer_idx", "unit", "layer_name"])
    df['layer_name'] = df['layer_name'].astype('category')
    return df

# Synthetic single-sublayer frame: one layer per block, mimicking a sublayer scope.
#
# The word count is deliberately far larger than the three words the identity itself needs.
# count_matched_z returns all NaN below 2 * NULL_NEIGHBORS_MIN = 80 finite pair entries, and
# np.allclose(nan, nan, equal_nan=True) is true whatever the mapping does, so on a three-word
# fixture the z arm of this comparison could not fail at all. Twenty words give
# 20 * 19 / 2 = 190 pairs, comfortably past that floor, so z is a real array and the arm has
# teeth. The jsd and sim arms would pass on any size and are kept as they were.
BLOCKS = 12
WORDS = 20
single_rows = []
for word_index in range(WORDS):
    word = f"word_{word_index:02d}"
    # A deterministic per-word depth preference, so the profiles differ from one another and
    # the expert counts vary, which is what gives the count-matched null something to match
    # on and keeps the local standard deviation away from zero.
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

# The z arm is only a check if z is actually defined on this fixture. Below the null floor it
# would be all NaN and the comparison above would pass unconditionally, so the fixture size is
# asserted through its effect rather than trusted.
if not np.isfinite(plain[2]).any():
    failures.append("z is all NaN on the single-sublayer fixture, so the z arm of the "
                    "identity check is vacuous. Widen the fixture past the "
                    "2 * NULL_NEIGHBORS_MIN entry floor of count_matched_z")

# Multi-sublayer frame: counts per word must survive aggregation.
multi = make_frame([
    ("cat", 1, 0, "1.L.0.attn.c_attn"), ("cat", 3, 0, "3.L.0.mlp.c_fc"),
    ("dog", 2, 1, "2.L.0.attn.c_proj"), ("dog", 7, 1, "7.L.1.mlp.c_fc"),
])
before = multi.groupby("concept").size()
after = to_block_axis(multi).groupby("concept").size()
if not before.equals(after):
    failures.append("block aggregation changed per-word expert counts")

# The call sites wrap their frame. A source check, crude but effective.
base_path = Path(__file__).resolve().parents[1]
expectations = {
    "modules/module_3_similarities.py": r"layer_profile_metric_matrices\(to_block_axis\(",
    "modules/module_4_correlations.py": r"pair_layer_profile_vectors\(to_block_axis\(",
    "modules/module_5_heatmaps.py": r"layer_profile_metric_matrices\(to_block_axis\(",
}
for path, pattern in expectations.items():
    full_path = base_path / path
    source = open(full_path).read()
    if not re.search(pattern, source):
        failures.append(f"{path}: expected call through to_block_axis, not found")

if failures:
    print("FAIL check_block_axis_identity")
    for line in failures:
        print("  " + line)
    sys.exit(1)
print("OK check_block_axis_identity")
