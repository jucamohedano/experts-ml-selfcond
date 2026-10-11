"""
Module 2's domain-core table (ExpertLens's domain statistics) measures shared cores and the label's share correctly.

Checks:
  1. Bit arithmetic: on random expert sets, the core share, the label's share of the core and the core
     size of _core_statistics equal the same quantities computed with Python sets.
  2. Synthetic categories: members with identical sets, which the label shares, give a core of 100% of
     the union and a label share of 100%, a core of 5 shared units among 25 gives 20%, and a label
     holding 3 of those 5 units gives 60%. Members of four other categories with disjoint sets give an
     empty random core, and a category with fewer than four members is left out.
  3. The MEAN row averages the category rows, and the table repeats exactly under the same seed.

Usage, from scripts/:
    python tests/check_module2_category_core.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_2_similarities import _core_statistics, _packed_presence, category_core_overlap
from _report import run


def synthetic() -> tuple:
    """Expert rows and metadata for six categories, two with known cores and four with disjoint members."""
    sets = {"a": set(range(10)), **{f"a{i}": set(range(10)) for i in range(4)}}
    sets["b"] = {100, 101, 102, 900, 901}
    sets.update({f"b{i}": set(range(100, 105)) | set(range(200 + 10 * i, 205 + 10 * i)) for i in range(4)})
    for category, start in (("c", 300), ("d", 400), ("e", 500), ("f", 600)):
        sets[category] = {start + 99}
        size = 3 if category == "f" else 4
        sets.update({f"{category}{i}": {start + 10 * i, start + 10 * i + 1} for i in range(size)})
    rows = [(concept, 1, unit) for concept, units in sets.items() for unit in sorted(units)]
    frame = pd.DataFrame(rows, columns=["concept", "layer_idx", "unit"])
    meta = pd.DataFrame({"concept": list(sets),
                         "category": [None if len(c) == 1 else c[0] for c in sets],
                         "abstraction_level": [1 if len(c) == 1 else 2 for c in sets]})
    return frame, meta


def check_bits() -> str:
    rng = np.random.default_rng(0)
    sets = [set(rng.choice(60, rng.integers(5, 40), replace=False)) for _ in range(6)]
    frame = pd.DataFrame([(f"w{i}", 1, unit) for i, units in enumerate(sets) for unit in units],
                         columns=["concept", "layer_idx", "unit"])
    concepts = [f"w{i}" for i in range(6)]
    bits = _packed_presence(frame, concepts)
    for rows in ([0, 1], [0, 1, 2, 3], [2, 4, 5]):
        core, union = set.intersection(*(sets[i] for i in rows)), set.union(*(sets[i] for i in rows))
        expected = (100 * len(core) / len(union), 100 * len(core & sets[5]) / len(core) if core else np.nan, len(core))
        got = _core_statistics(bits[rows], bits[5])
        assert np.allclose(got, expected, equal_nan=True), f"rows {rows}: {got} against sets {expected}"
    return "bits: core share, label share and core size equal the set computation"


def check_categories() -> str:
    frame, meta = synthetic()
    table = category_core_overlap(frame, meta, n_draws=50).set_index("category")
    a, b = table.loc["a"], table.loc["b"]
    assert a["core_pct_4"] == 100 and a["label_share_of_core_pct_4"] == 100, f"identical sets: {a.to_dict()}"
    assert a["core_size_all_members"] == 10 and a["label_share_of_core_pct_all_members"] == 100, "all-member core"
    assert b["core_pct_4"] == 20 and b["core_size_4"] == 5, f"partial core: {b['core_pct_4']}, {b['core_size_4']}"
    assert b["label_share_of_core_pct_4"] == 60, f"label share {b['label_share_of_core_pct_4']}"
    assert a["random_core_pct_4"] == 0 and a["random_empty_core_rate_4"] == 1, "random baseline must be empty"
    assert "f" not in table.index, "a category with three members must be left out"
    return f"categories: core 100% and 20%, label share 100% and 60%, random core empty, {len(table) - 1} categories kept"


def check_mean_and_seed() -> str:
    frame, meta = synthetic()
    first = category_core_overlap(frame, meta, n_draws=50)
    second = category_core_overlap(frame, meta, n_draws=50)
    assert first.equals(second), "the table must repeat under the same seed"
    rows = first[first["category"] != "MEAN"]
    mean = first.set_index("category").loc["MEAN", "core_pct_4"]
    assert abs(mean - rows["core_pct_4"].mean()) < 1e-12, "MEAN row must average the categories"
    return "MEAN row averages the categories, repeatable"


def checks(failures: list) -> str:
    return ", ".join([check_bits(), check_categories(), check_mean_and_seed()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
