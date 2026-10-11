"""
Module 1's per-category entropy and Geary's C summary compares each label with its own members correctly.

Checks:
  1. Percentile: a label equal to the member median sits at 0.5, below every member at 0, above every member at 1.
  2. Synthetic profiles: normalised entropy lies in [0, 1], an even label profile gets entropy 1, and the summary
     reproduces scipy's entropy and the shared gearys_c on every member.
  3. Count matching: members with fewer experts than the label are left out of the matched comparison, and
     the matched values repeat exactly under the same seed.

Usage, from scripts/:
    python tests/check_module1_category_shape.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import entropy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_1_expert_distribution import category_shape_summary, label_percentile
from modules.shared_layer_distribution_measures import gearys_c
from _report import run

N_BLOCKS = 6


def synthetic() -> tuple:
    """Expert rows on a block axis for one label and four members, plus their metadata."""
    profiles = {"fruit": [5] * N_BLOCKS, "apple": [20, 10, 4, 2, 1, 3], "pear": [2, 4, 10, 10, 4, 2],
                "plum": [30, 0, 0, 0, 0, 30], "fig": [1, 1, 1, 0, 0, 0]}
    names = [f"{b + 1}.B.{b}" for b in range(N_BLOCKS)]
    rows = [(concept, names[b], b + 1, unit) for concept, counts in profiles.items()
            for b, count in enumerate(counts) for unit in range(count)]
    frame = pd.DataFrame(rows, columns=["concept", "layer_name", "layer_idx", "unit"])
    frame["layer_name"] = pd.Categorical(frame["layer_name"], categories=names, ordered=True)
    meta = pd.DataFrame({"concept": list(profiles), "category": [None, "fruit", "fruit", "fruit", "fruit"],
                         "abstraction_level": [1, 2, 2, 2, 2]})
    return frame, meta, profiles


def check_percentile() -> str:
    members = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert label_percentile(3.0, members) == 0.5, "median label must sit at 0.5"
    assert label_percentile(0.0, members) == 0.0 and label_percentile(9.0, members) == 1.0, "extremes"
    assert np.isnan(label_percentile(np.nan, members)), "undefined label must give NaN"
    return "percentile: 0.5 at the median, 0 and 1 at the extremes"


def check_summary() -> str:
    frame, meta, profiles = synthetic()
    summary, members = category_shape_summary(frame, meta, n_draws=50, seed=0)
    row = summary.iloc[0]
    assert abs(row["entropy_norm_label"] - 1.0) < 1e-12, f"even label entropy {row['entropy_norm_label']}"
    entropy_norm, geary = members["fruit"]
    assert np.all((entropy_norm >= 0) & (entropy_norm <= 1)), "normalised entropy outside [0, 1]"
    for value, concept in zip(entropy_norm, sorted(["apple", "pear", "plum", "fig"])):
        counts = np.array(profiles[concept], dtype=float)
        expected = entropy(counts / counts.sum(), base=2) / np.log2(N_BLOCKS)
        assert abs(value - expected) < 1e-12, f"{concept}: entropy {value} against scipy {expected}"
    for value, concept in zip(geary, sorted(["apple", "pear", "plum", "fig"])):
        counts = np.array(profiles[concept], dtype=float)
        assert abs(value - gearys_c(counts / counts.sum())) < 1e-12, f"{concept}: Geary's C"
    return f"summary: label entropy 1.0, member values match scipy and gearys_c, label percentile {row['entropy_norm_label_percentile']:.2f}"


def check_matching() -> str:
    frame, meta, _ = synthetic()
    first, _ = category_shape_summary(frame, meta, n_draws=50, seed=0)
    second, _ = category_shape_summary(frame, meta, n_draws=50, seed=0)
    assert int(first.iloc[0]["n_members_matched"]) == 3, "fig has fewer experts than the label and must be left out"
    assert first.equals(second), "matched values must repeat under the same seed"
    return f"matching: {int(first.iloc[0]['n_members_matched'])} of 4 members at the label's 30 experts, repeatable"


def checks(failures: list) -> str:
    return ", ".join([check_percentile(), check_summary(), check_matching()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
