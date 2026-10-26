"""
Module 5's concept-structure tests are two-sided permutation tests with generators of their own: the separation of each
group of a labelling in a distance space, and the share of a link graph's edges joining two words of one category.

Runs group_separation and graph_domain_structure on synthetic points and edges.

Checks:
  1. Each group's within and between means and their difference equal the direct computation, tight groups reach the
     permutation floor with a positive separation, groups spread apart on purpose reach it with a negative one, and
     random groups do not, every p from PERMUTATIONS reassignments.
  2. The within-category share of the edges is tested two-sided around its exact null expectation, the sum over
     categories of n_k (n_k - 1) / (N (N - 1)): edges all inside categories reach the floor, edges all across them
     get p below 0.01 where a one-sided test would give about 1, and the mean of the null lies within Monte Carlo
     error of the expectation.

Usage, from scripts/:
    python tests/check_module5_concept_structure_tests.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_5_typicality_prediction import graph_domain_structure, group_separation
from utils.statistics import PERMUTATIONS
from _report import run

FLOOR = round(1 / (PERMUTATIONS + 1), 4)


def check_separation() -> str:
    rng = np.random.default_rng(0)
    centres = np.repeat(100 * np.eye(20), 3, axis=0)
    points = centres + rng.normal(0, 0.5, centres.shape)
    distance = np.sqrt(((points[:, None] - points[None]) ** 2).sum(-1))
    tight = np.repeat(np.arange(20), 3).astype(str)
    spread = np.tile(["a", "b", "c"], 20)
    shuffled = rng.permutation(spread)

    table = group_separation(distance, tight, "group").set_index("group")
    for name in np.unique(tight):
        member = tight == name
        within = distance[np.ix_(member, member)][np.triu_indices(member.sum(), k=1)].mean()
        between = distance[np.ix_(member, ~member)].mean()
        assert abs(table.loc[name, "within_mean"] - round(within, 4)) < 1e-9 and \
            abs(table.loc[name, "between_mean"] - round(between, 4)) < 1e-9, f"group {name}: means"
    assert table.loc["ALL GROUPS", "separation"] > 0 and table.loc["ALL GROUPS", "p_permutation"] == FLOOR, \
        f"tight groups: {table.loc['ALL GROUPS'].to_dict()}"

    apart = group_separation(distance, spread, "group").set_index("group").loc["ALL GROUPS"]
    assert apart.separation < 0 and apart.p_permutation == FLOOR, \
        f"groups spread apart: separation {apart.separation}, p {apart.p_permutation}, the test must be two-sided"
    random = group_separation(distance, shuffled, "group").set_index("group").loc["ALL GROUPS"]
    assert random.p_permutation > 0.01, f"random groups: p {random.p_permutation}"
    return f"separation: tight p {FLOOR}, spread apart {apart.separation:.2f} at p {apart.p_permutation}, " \
           f"random p {random.p_permutation}"


def check_domain_structure() -> str:
    words = [f"w{i:02d}" for i in range(60)]
    category = np.repeat(["a", "b", "c"], 20)
    meta = pd.DataFrame({"concept": words, "category": category})
    expected = 3 * 20 * 19 / (60 * 59)

    def edges(pairs: list) -> pd.DataFrame:
        return pd.DataFrame({"concept_a": [words[i] for i, _ in pairs], "concept_b": [words[j] for _, j in pairs],
                             "weight": np.linspace(1, 0.1, len(pairs)), "category_a": [category[i] for i, _ in pairs],
                             "category_b": [category[j] for _, j in pairs]})

    inside = graph_domain_structure(edges([(i, i + 1) for i in range(0, 60, 2)]), meta).set_index("edge_set")
    across = graph_domain_structure(edges([(i, (i + 20) % 60) for i in range(0, 60, 2)]), meta).set_index("edge_set")
    for name, table, share, p_max in (("inside", inside, 100.0, FLOOR), ("across", across, 0.0, 0.01)):
        row = table.loc["all edges"]
        assert row.within_category_pct == share and row.p_permutation <= p_max, \
            f"edges {name} categories: {row.to_dict()}, the test must be two-sided"
        assert abs(row.null_within_category_pct - 100 * expected) < 0.06 + 5 * row.null_sd_pct / 100, \
            f"edges {name}: null mean {row.null_within_category_pct} against expectation {100 * expected:.2f}"
    edge_rows = inside.loc["all edges"], across.loc["all edges"]
    return f"edges: inside p {edge_rows[0].p_permutation}, across p {edge_rows[1].p_permutation}, null mean " \
           f"{edge_rows[1].null_within_category_pct}% against {100 * expected:.1f}%"


def checks(failures: list) -> str:
    return ", ".join([check_separation(), check_domain_structure()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
