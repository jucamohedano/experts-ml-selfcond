"""
Module 2's text baseline: the TF-IDF similarity of the probing sentences and the partial Spearman of experts against
humans with the text partialled out.

Checks:
  1. Text similarity: two concepts with identical sentences have cosine 1, two with no shared term cosine 0,
     the matrix is symmetric with a unit diagonal, and a concept without a sentence file gets NaN.
  2. Table: on synthetic matrices over every metadata concept, every rated pair is used, each category's coefficients
     equal scipy's Spearman and the closed-form partial on that category's ranks with a permutation p, WEIGHTED_MEAN
     holds their pair-weighted means and a p for the partial, a text equal to the expert matrix leaves nothing to
     partial (NaN), and an unrelated text leaves the partial close to the plain rho.

Usage, from scripts/:
    python tests/check_module2_text_baseline.py
"""
import json
import math
import pathlib
import sys
import tempfile

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_2_similarities import WEIGHTED_MEAN, text_baseline_table, text_similarity_matrix
from core.human_similarity_ratings import load_human_similarity
from utils.statistics import PERMUTATIONS
from _report import run

METADATA = pathlib.Path(__file__).resolve().parents[3] / "abstractiveness/assets/metadata_Richie_HSJ.json"
SENTENCES = {"c1": ["The red apple grows on a tall tree.", "A red apple on a tree."],
             "c2": ["The red apple grows on a tall tree.", "A red apple on a tree."],
             "c3": ["Fast cars need engines and highways."],
             "c4": ["Engines move fast cars along highways.", "Fast cars and fast engines."]}


def check_text_similarity() -> str:
    with tempfile.TemporaryDirectory() as root:
        (pathlib.Path(root) / "custom").mkdir()
        for concept, positives in SENTENCES.items():
            data = {"sentences": {"positive": positives, "negative": ["unused"]}}
            (pathlib.Path(root) / "custom" / f"{concept}.json").write_text(json.dumps(data))
        similarity = text_similarity_matrix(root, ("c1", "c2", "c3", "c4", "c5"))
    assert abs(similarity[0, 1] - 1) < 1e-12, f"identical sentences give {similarity[0, 1]}"
    assert similarity[0, 2] == 0 and similarity[1, 3] == 0, "no shared term must give 0"
    known = similarity[:4, :4]
    assert np.allclose(known, known.T) and np.allclose(np.diag(known), 1), "symmetric with a unit diagonal"
    assert np.isnan(similarity[4]).all() and np.isnan(similarity[:, 4]).all(), "missing file must give NaN"
    return f"text: identical 1, disjoint 0, related {similarity[2, 3]:.2f}, missing NaN"


def closed_form(x, y, z) -> float:
    """First-order partial Spearman from the three pairwise coefficients."""
    xy, xz, yz = (spearmanr(a, b).statistic for a, b in ((x, y), (x, z), (y, z)))
    return (xy - xz * yz) / math.sqrt((1 - xz ** 2) * (1 - yz ** 2))


def check_table() -> str:
    concepts = [entry["concept"] for entry in json.load(METADATA.open())]
    index = {concept: i for i, concept in enumerate(concepts)}
    pairs = load_human_similarity()
    rng = np.random.default_rng(1)
    jaccard = rng.uniform(0, 10, size=(len(concepts), len(concepts)))
    for a, b, rating in zip(pairs.concept_a, pairs.concept_b, pairs.mean_rating):
        jaccard[index[a], index[b]] = jaccard[index[b], index[a]] = rating + rng.normal(0, 1)
    unrelated = rng.uniform(0, 1, size=jaccard.shape)
    unrelated = (unrelated + unrelated.T) / 2

    same = text_baseline_table(concepts, jaccard, jaccard, None, n_draws=0)
    table = text_baseline_table(concepts, jaccard, unrelated, None)
    rows = table[table.category != WEIGHTED_MEAN].set_index("category")
    rated = pairs.groupby("category").size()
    assert (rows["n_pairs"] == rated.reindex(rows.index)).all(), "every rated pair must be used"
    for category, group in pairs.groupby("category"):
        first, second = group.concept_a.map(index).to_numpy(), group.concept_b.map(index).to_numpy()
        model, human, text = jaccard[first, second], group.mean_rating.to_numpy(), unrelated[first, second]
        expected = {"rho_jaccard_human": spearmanr(model, human).statistic,
                    "rho_jaccard_text": spearmanr(model, text).statistic,
                    "rho_text_human": spearmanr(text, human).statistic,
                    "partial_rho_jaccard_human_given_text": closed_form(model, human, text)}
        for column, value in expected.items():
            assert abs(rows.loc[category, column] - value) < 1e-12, f"{category} {column}: {rows.loc[category, column]}"
    weights = rows["n_pairs"] / rows["n_pairs"].sum()
    mean = table.set_index("category").loc[WEIGHTED_MEAN]
    for column in ("rho_text_human", "rho_jaccard_human", "rho_jaccard_text", "partial_rho_jaccard_human_given_text"):
        assert abs(mean[column] - (weights * rows[column]).sum()) < 1e-12, f"WEIGHTED_MEAN {column} is not the mean"
    floor = 1 / (PERMUTATIONS + 1)
    p_values = table["partial_p_jaccard_human_given_text"]
    assert p_values.between(floor, 1).all(), f"partial p {p_values.tolist()}"
    assert same["partial_rho_jaccard_human_given_text"].isna().all(), \
        "a text equal to the expert matrix must leave no partial correlation"
    gap = (rows["partial_rho_jaccard_human_given_text"] - rows["rho_jaccard_human"]).abs().max()
    assert gap < 0.05, f"an unrelated text moved the partial by {gap:.3f}"
    partial, p = mean["partial_rho_jaccard_human_given_text"], mean["partial_p_jaccard_human_given_text"]
    return (f"table: {int(rows['n_pairs'].sum())} rated pairs, weighted partial {partial:.3f} at p {p}, "
            f"unrelated text moves rho by {gap:.3f} at most")


def checks(failures: list) -> str:
    return ", ".join([check_text_similarity(), check_table()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
