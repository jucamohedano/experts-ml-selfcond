"""
Module 2's text baseline: the TF-IDF similarity of the probing sentences and the partial Spearman of experts against humans.

Checks:
  1. Text similarity: two concepts with identical sentences have cosine 1, two with no shared term cosine 0,
     the matrix is symmetric with a unit diagonal, and a concept without a sentence file gets NaN.
  2. Partial correlation: _partial_from_rhos equals the correlation of the residuals left after regressing
     both ranked variables on the ranked control.
  3. Table: on synthetic matrices over every metadata concept, every rated pair is used, POOLED_MEAN averages
     the categories, a text equal to the expert matrix leaves nothing to partial (NaN), and an unrelated text
     leaves the partial close to the plain rho.

Usage, from scripts/:
    python tests/check_module2_text_baseline.py
"""
import json
import pathlib
import sys
import tempfile

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_2_similarities import _partial_from_rhos, text_baseline_table, text_similarity_matrix
from core.human_similarity_ratings import load_human_similarity
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


def check_partial() -> str:
    rng = np.random.default_rng(0)
    z = rng.normal(size=300)
    x, y = z + rng.normal(size=300), z + rng.normal(size=300)
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    residual = lambda values: values - np.polyval(np.polyfit(rz, values, 1), rz)
    expected = np.corrcoef(residual(rx), residual(ry))[0, 1]
    got = _partial_from_rhos(spearmanr(x, y).statistic, spearmanr(x, z).statistic, spearmanr(y, z).statistic)
    assert abs(got - expected) < 1e-12, f"partial {got} against residuals {expected}"
    return f"partial: {got:.3f} equals the residual correlation"


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

    same = text_baseline_table(concepts, jaccard, jaccard, None)
    table = text_baseline_table(concepts, jaccard, unrelated, None)
    rows = table[table.category != "POOLED_MEAN"].set_index("category")
    rated = pairs.groupby("category").size()
    assert (rows["n_pairs"] == rated.reindex(rows.index)).all(), "every rated pair must be used"
    mean = table.set_index("category").loc["POOLED_MEAN"]
    assert abs(mean["rho_jaccard_human"] - rows["rho_jaccard_human"].mean()) < 1e-12, "POOLED_MEAN must average"
    assert same[same.category != "POOLED_MEAN"]["partial_rho_jaccard_human_given_text"].isna().all(), \
        "a text equal to the expert matrix must leave no partial correlation"
    gap = (rows["partial_rho_jaccard_human_given_text"] - rows["rho_jaccard_human"]).abs().max()
    assert gap < 0.05, f"an unrelated text moved the partial by {gap:.3f}"
    return f"table: {int(rows['n_pairs'].sum())} rated pairs, unrelated text moves rho by {gap:.3f} at most"


def checks(failures: list) -> str:
    return ", ".join([check_text_similarity(), check_partial(), check_table()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
