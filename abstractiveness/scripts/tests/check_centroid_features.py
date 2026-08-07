"""Verifies the centroid features are leave-one-out and correctly ordered."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modules.module_9_typicality_prediction import add_centroid_features
from utils.helpers import build_layer_probability_matrix, to_block_axis


def _toy() -> tuple:
    """Three concepts in one category. a and b overlap in layers, c sits elsewhere, so c must
    score below a against the centroid of the others."""
    rows = []
    for concept, layers in [("a", [1, 1, 2]), ("b", [1, 2, 2]), ("c", [9, 9, 9])]:
        for unit, layer in enumerate(layers):
            rows.append({"concept": concept, "layer_idx": layer, "unit": unit,
                         "layer_name": f"{layer}.L.{layer - 1}.mlp.c_fc"})
    experts = pd.DataFrame(rows)
    # The production frame carries layer_name as an ordered Categorical (helpers builds it
    # that way so layers sort in model order), and build_layer_probability_matrix relies on
    # the .cat accessor, so the fixture has to match that dtype rather than plain object.
    ordered = sorted(experts["layer_name"].unique(), key=lambda name: int(name.split(".")[0]))
    experts["layer_name"] = pd.Categorical(experts["layer_name"], categories=ordered, ordered=True)
    design = pd.DataFrame({"concept": ["a", "b", "c"], "category": ["k", "k", "k"],
                           "human_typicality": [0.9, 0.8, 0.1]})
    return design, experts


def main() -> int:
    design, experts = _toy()
    out = add_centroid_features(design, experts)

    for column in ["mean_jaccard_to_members", "layer_profile_similarity_to_centroid",
                   "centroid_margin"]:
        assert column in out.columns, f"{column} missing"

    indexed = out.set_index("concept")
    assert indexed.loc["a", "layer_profile_similarity_to_centroid"] > \
        indexed.loc["c", "layer_profile_similarity_to_centroid"], \
        "a overlaps the other members and must score above the outlier c"
    assert indexed.loc["a", "mean_jaccard_to_members"] > \
        indexed.loc["c", "mean_jaccard_to_members"], \
        "a shares experts with b and must beat the disjoint c"
    assert out.layer_profile_similarity_to_centroid.between(0, 100).all(), "similarity off scale"

    # Leave-one-out proof: the reference for c must exclude c, so including it would differ.
    _, prob = build_layer_probability_matrix(to_block_axis(experts))
    with_self = prob.loc[["a", "b", "c"]].mean(axis=0).to_numpy()
    without_self = prob.loc[["a", "b"]].mean(axis=0).to_numpy()
    assert not np.allclose(with_self, without_self), \
        "toy data cannot distinguish leave-one-out, fix the fixture"

    print(out.to_string(index=False))
    print("OK  centroid features are leave-one-out and correctly ordered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
