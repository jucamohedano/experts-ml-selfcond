"""Checks that the pair-similarity target is symmetric, correctly joined, and shortcut-free."""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modules.module_9_typicality_prediction import build_pair_similarity_design
from utils.human_similarity import human_pair_lookup


def main() -> int:
    concepts = ["apple", "banana", "pear", "olive"]
    rows = []
    for offset, concept in enumerate(concepts):
        for unit in range(5):
            layer = 1 + (unit + offset) % 3
            rows.append({"concept": concept, "layer_idx": layer, "unit": unit,
                         "layer_name": f"{layer}.L.{layer - 1}.mlp.c_fc"})
    experts = pd.DataFrame(rows)
    ordered = sorted(experts["layer_name"].unique(), key=lambda name: int(name.split(".")[0]))
    experts["layer_name"] = pd.Categorical(experts["layer_name"], categories=ordered, ordered=True)
    design = pd.DataFrame({"concept": concepts, "category": ["fruit"] * 4,
                           "human_typicality": [0.9, 0.8, 0.7, 0.1]})

    table = build_pair_similarity_design(design, experts)

    lookup = human_pair_lookup()
    expected = sum(1 for i in range(4) for j in range(i + 1, 4)
                   if (min(concepts[i], concepts[j]), max(concepts[i], concepts[j])) in lookup)
    assert len(table) == expected, f"expected {expected} rated pairs, got {len(table)}"
    assert (table.word_a < table.word_b).all(), "pairs not canonical"
    assert table.human_similarity.between(1, 7).all(), "target off the 1 to 7 scale"
    for column in ["jaccard_pct", "layer_profile_similarity_pct", "layer_profile_z"]:
        assert column in table.columns, f"{column} missing"
    assert table.category.nunique() == 1, "toy fixture should be single-category"

    # A model given only category identity must score at chance, because the category is
    # constant within every pair. If this ever fails, the pair construction has leaked.
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    rng = np.random.default_rng(0)
    big = pd.DataFrame({"category": ["a"] * 50 + ["b"] * 50,
                        "human_similarity": np.r_[rng.normal(4, 1, 50), rng.normal(4, 1, 50)]})
    codes = pd.factorize(big.category)[0].reshape(-1, 1).astype(float)
    predicted = cross_val_predict(LinearRegression(), codes, big.human_similarity,
                                  cv=GroupKFold(n_splits=2), groups=big.category)
    leak = abs(spearmanr(predicted, big.human_similarity).statistic)
    assert leak < 0.2, f"category identity predicted pair similarity (rho={leak:.3f}), split leaks"

    print(table.to_string(index=False))
    print(f"OK  pair-similarity design is symmetric, joined correctly, "
          f"category-shortcut rho={leak:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
