"""
Module 2's section 2.2 tests, Jaccard against layer profile over every word pair and the category alignment of every
pairwise matrix, both Spearman's rho with a two-sided Mantel p.

Checks:
  1. All pairs: a pair vector against a noisy copy of itself gives p at the permutation floor, two unrelated pair
     vectors p above 0.05, the row carries no Pearson column, a word whose pairs are all NaN is dropped whole, and
     without permutations the row keeps its rho and has no p.
  2. Category alignment: a matrix built from the categories gives an ROC-AUC near 1 at the floor p and an unrelated
     one a p above 0.05, the within and across means are those of the member pairs alone, and without permutations
     the p is NaN and the AUC unchanged.

Usage, from scripts/:
    python tests/check_module2_allpairs_mantel.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.module_2_similarities import allpairs_row, summarize_category_contrast
from utils.statistics import PERMUTATIONS
from _report import run

N_WORDS = 40

FLOOR = 1 / (PERMUTATIONS + 1)


def symmetric(rng, n: int = N_WORDS) -> np.ndarray:
    """A random symmetric matrix over n words."""
    values = rng.uniform(0, 100, size=(n, n))
    return (values + values.T) / 2


def check_allpairs() -> str:
    rng = np.random.default_rng(3)
    upper = np.triu_indices(N_WORDS, k=1)
    x = symmetric(rng)[upper]
    related, _ = allpairs_row("related", x, x + rng.normal(0, 10, size=x.size), "profile", N_WORDS)
    assert related["spearman_p"] == FLOOR, f"noisy copy: p {related['spearman_p']}, floor {FLOOR}"
    assert "pearson_r" not in related and "pearson_p" not in related, "Pearson columns must be gone"
    unrelated, _ = allpairs_row("unrelated", x, symmetric(rng)[upper], "profile", N_WORDS)
    assert unrelated["spearman_p"] > 0.05, f"unrelated vectors: p {unrelated['spearman_p']}"

    with_gap = x + rng.normal(0, 10, size=x.size)
    with_gap[(upper[0] == 0) | (upper[1] == 0)] = np.nan
    gapped, used = allpairs_row("gapped", x, with_gap, "profile", N_WORDS)
    kept = (N_WORDS - 1) * (N_WORDS - 2) // 2
    assert gapped["n_points"] == used.sum() == kept, f"NaN word: {gapped['n_points']} pairs kept, expected {kept}"
    quick, _ = allpairs_row("quick", x, with_gap, "profile", N_WORDS, n_draws=0)
    assert quick["spearman_p"] is None and quick["spearman_rho"] == gapped["spearman_rho"], f"no permutations: {quick}"
    return (f"all pairs: related p {related['spearman_p']}, unrelated p {unrelated['spearman_p']:.3f}, "
            f"NaN word dropped whole")


def check_alignment() -> str:
    rng = np.random.default_rng(4)
    categories = np.repeat(["a", "b", "c"], 10)
    concepts = [f"w{i}" for i in range(len(categories))] + ["label"]
    meta = pd.DataFrame({"concept": concepts, "category": list(categories) + [None]})
    same = categories[:, None] == categories[None, :]
    aligned = np.full((len(concepts), len(concepts)), 50.0)
    aligned[:-1, :-1] = 20 * same + rng.normal(0, 5, size=same.shape)
    aligned = (aligned + aligned.T) / 2

    result = summarize_category_contrast(concepts, aligned, meta)
    upper = np.triu_indices(len(categories), k=1)
    pairs, same_pairs = aligned[:-1, :-1][upper], same[upper]
    assert abs(result["within_pct"] - pairs[same_pairs].mean()) < 1e-12, "within mean must use member pairs only"
    assert abs(result["across_pct"] - pairs[~same_pairs].mean()) < 1e-12, "across mean must use member pairs only"
    assert result["n_pairs"] == len(pairs), f"pairs {result['n_pairs']}"
    assert result["roc_auc"] > 0.95 and result["mantel_p"] == FLOOR, f"aligned: {result}"
    unrelated = summarize_category_contrast(concepts, symmetric(rng, len(concepts)), meta)
    assert unrelated["mantel_p"] > 0.05, f"unrelated matrix: p {unrelated['mantel_p']}"
    quick = summarize_category_contrast(concepts, aligned, meta, n_draws=0)
    assert np.isnan(quick["mantel_p"]) and quick["roc_auc"] == result["roc_auc"], f"no permutations: {quick}"
    return (f"alignment: AUC {result['roc_auc']:.3f} at p {result['mantel_p']}, unrelated AUC "
            f"{unrelated['roc_auc']:.3f} at p {unrelated['mantel_p']:.3f}")


def checks(failures: list) -> str:
    return ", ".join([check_allpairs(), check_alignment()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
