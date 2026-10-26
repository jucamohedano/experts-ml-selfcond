"""
Module 2's split of the expert-human agreement into a member part (typicality) and a pair-specific part.

Checks:
  1. Split: on a random complete symmetric design the pair residuals sum to zero over every member, are
     orthogonal to the member part, and the member and pair parts of the ranked correlation add up to Spearman's rho.
  2. Table: on synthetic matrices over every metadata concept, every rated pair is used, each category's two parts
     add up to its rho, rho equals the text baseline's, WEIGHTED_MEAN weighs the pair columns by pairs and the member
     columns by members so its two parts still add up to its rho, and with a label column equal to each member's mean
     rating, centrality correlates with the label as it does with typicality.

Usage, from scripts/:
    python tests/check_module2_decomposition.py
"""
import json
import pathlib
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_2_similarities import (WEIGHTED_MEAN, _member_and_pair_parts, human_similarity_decomposition_table,
                                           text_baseline_table)
from core.human_similarity_ratings import load_human_similarity
from _report import run

METADATA = pathlib.Path(__file__).resolve().parents[3] / "abstractiveness/assets/metadata_Richie_HSJ.json"


def check_split() -> str:
    rng = np.random.default_rng(0)
    n_members = 12
    first, second = np.triu_indices(n_members, k=1)
    effects = rng.normal(size=n_members)
    x = effects[first] + effects[second] + rng.normal(size=len(first))
    y = effects[first] + effects[second] + rng.normal(size=len(first))
    rx, ry = rankdata(x), rankdata(y)
    _, x_part, x_rest = _member_and_pair_parts(rx, first, second, n_members)
    _, y_part, y_rest = _member_and_pair_parts(ry, first, second, n_members)
    sums = np.bincount(first, x_rest, n_members) + np.bincount(second, x_rest, n_members)
    assert np.abs(sums).max() < 1e-9, f"residuals must sum to zero per member, got {np.abs(sums).max()}"
    assert abs((x_part * y_rest).sum()) < 1e-9, "member part and pair residual must be orthogonal"
    norm = np.sqrt(((rx - rx.mean()) ** 2).sum() * ((ry - ry.mean()) ** 2).sum())
    total = (x_part * y_part).sum() / norm + (x_rest * y_rest).sum() / norm
    rho = spearmanr(x, y).statistic
    assert abs(total - rho) < 1e-12, f"parts add up to {total}, rho is {rho}"
    return f"split: parts add up to rho {rho:.3f}"


def check_table() -> str:
    concepts = [entry["concept"] for entry in json.load(METADATA.open())]
    index = {concept: i for i, concept in enumerate(concepts)}
    pairs = load_human_similarity()
    rng = np.random.default_rng(1)
    jaccard = rng.uniform(0, 10, size=(len(concepts), len(concepts)))
    for a, b, rating in zip(pairs.concept_a, pairs.concept_b, pairs.mean_rating):
        jaccard[index[a], index[b]] = jaccard[index[b], index[a]] = rating + rng.normal(0, 1)
    for category, group in pairs.groupby("category"):
        for member in set(group.concept_a) | set(group.concept_b):
            ratings = group[(group.concept_a == member) | (group.concept_b == member)].mean_rating
            jaccard[index[member], index[category]] = ratings.mean()
    text = rng.uniform(0, 1, size=jaccard.shape)
    text = (text + text.T) / 2

    table = human_similarity_decomposition_table(concepts, jaccard, text, None)
    rows = table[table.category != WEIGHTED_MEAN].set_index("category")
    rated = pairs.groupby("category").size()
    assert (rows["n_pairs"] == rated.reindex(rows.index)).all(), "every rated pair must be used"
    gap = (rows["member_part"] + rows["pair_part"] - rows["rho"]).abs().max()
    assert gap < 1e-12, f"parts miss rho by {gap}"
    baseline = text_baseline_table(concepts, jaccard, text, None, n_draws=0).set_index("category")
    assert np.allclose(rows["rho"], baseline.loc[rows.index, "rho_jaccard_human"]), "rho must match the text baseline"
    mean = table.set_index("category").loc[WEIGHTED_MEAN]
    by_pairs, by_members = rows["n_pairs"] / rows["n_pairs"].sum(), rows["n_members"] / rows["n_members"].sum()
    assert abs(mean["member_part"] - (by_pairs * rows["member_part"]).sum()) < 1e-12, "pair columns weigh by pairs"
    assert abs(mean["rho_members"] - (by_members * rows["rho_members"]).sum()) < 1e-12, \
        "member columns weigh by members"
    assert abs(mean["member_part"] + mean["pair_part"] - mean["rho"]) < 1e-12, "the weighted parts must add up to rho"
    assert np.allclose(rows["rho_centrality_label_jaccard"], rows["rho_typicality_centrality"]), \
        "a label Jaccard equal to typicality must correlate with centrality as typicality does"
    return (f"table: {int(rows['n_pairs'].sum())} rated pairs, parts add up to rho within {gap:.0e}, "
            f"member share {mean['member_part'] / mean['rho']:.2f}")


def checks(failures: list) -> str:
    return ", ".join([check_split(), check_table()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
