"""
The shared correlation statistics: Spearman's rho over all items or within groups, its partial form and its two-sided
permutation p, on synthetic data.

Checks:
  1. Rank scores have mean 0 and population sd 1, over all items or within each group, ties averaged, 0 when constant.
  2. Pooled: rho equals scipy's Spearman, a related pair gives p at the floor 1 / 10,000 in either direction, an
     unrelated one p above 0.05, and repeating the test after other random draws gives the same p.
  3. Partial: with one control rho equals the closed form of the three pairwise Spearman coefficients, and the
     coefficient alone (correlation) gives the same rho, or Pearson's r and its partial with ranks=False.
  4. Within groups: rho is the size-weighted mean of the per-group Spearman coefficients, equals the correlation and
     the least-squares slope of the returned scores, and each group's partial uses its own control slope.
  5. Exact: eight items give every one of the 8! - 1 other orderings and p = 2 / 8! for a perfect order, binary labels
     over twelve items give the 494 other arrangements, and permutation_p counts the observed value.
  6. Mantel: on pairs, rho equals scipy's over the pair vectors, a noisy copy gives the floor and unrelated matrices
     p above 0.05, a word with missing pairs is dropped whole, and within groups every group's p equals that of the
     group tested alone.
  7. Small groups are left out, a group whose coefficient is undefined is left out of rho and of n, a category
     alignment ROC-AUC shares its two-sided Mantel p with Spearman's rho against the same-category indicator, and a
     constant similarity gives AUC 0.5 without a p.
  8. Residuals in the values' own units: the deviation from the mean without controls, from each group's mean within
     groups, and from each group's own least-squares line on a control.

Usage, from scripts/:
    python tests/check_shared_statistics.py
"""
import math
import pathlib
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from utils.statistics import (PERMUTATIONS, correlation, permutation_p, rank_scores, reassignments, residuals,
                              spearman_test)
from modules.shared_expert_set_measures import category_alignment_metrics
from _report import run

FLOOR = 1 / (PERMUTATIONS + 1)


def closed_form(x, y, z, coefficient=lambda a, b: stats.spearmanr(a, b).statistic) -> float:
    """First-order partial correlation from the three pairwise coefficients, Spearman's by default."""
    xy, xz, yz = (coefficient(a, b) for a, b in ((x, y), (x, z), (y, z)))
    return (xy - xz * yz) / math.sqrt((1 - xz ** 2) * (1 - yz ** 2))


def check_scores() -> str:
    values = np.array([3.0, 1.0, 1.0, 7.0, 5.0, 2.0, 2.0, 2.0])
    groups = np.array(list("aaaabbbb"))
    pooled, within = rank_scores(values), rank_scores(values, groups)
    assert abs(pooled.mean()) < 1e-12 and abs(pooled.std() - 1) < 1e-12, "pooled scores not standardised"
    assert pooled[1] == pooled[2], "tied values must share a score"
    for name in "ab":
        part = within[groups == name]
        assert abs(part.mean()) < 1e-12, f"group {name} not centred"
    assert abs(within[groups == "a"].std() - 1) < 1e-12, "group a not unit sd"
    constant = rank_scores(np.array([1.0, 1.0, 1.0, 4.0, 5.0, 6.0]), np.array(list("aaabbb")))
    assert (constant[:3] == 0).all(), "a constant group must score 0"
    return "scores: standardised over all items and within groups"


def check_pooled() -> str:
    rng = np.random.default_rng(0)
    x = rng.normal(size=120)
    y = x + rng.normal(scale=0.8, size=120)
    result = spearman_test(x, y)
    assert abs(result["rho"] - stats.spearmanr(x, y).statistic) < 1e-12, "pooled rho differs from scipy"
    assert result["p"] == FLOOR, f"related pair p {result['p']}, floor {FLOOR}"
    assert spearman_test(x, -y)["p"] == FLOOR, "a negative relation must reach the floor, the test is two-sided"
    unrelated = spearman_test(x, rng.normal(size=120))
    assert unrelated["p"] > 0.05, f"unrelated p {unrelated['p']}"
    first = spearman_test(x, y[::-1])["p"]
    np.random.seed(7), np.random.default_rng(5).normal(size=1000), spearman_test(y, x)
    assert spearman_test(x, y[::-1])["p"] == first, "the same test must give the same p, whatever ran before it"
    return f"pooled: rho {result['rho']:.3f} at p {result['p']}, unrelated p {unrelated['p']:.3f}"


def check_partial() -> str:
    rng = np.random.default_rng(1)
    z = rng.normal(size=150)
    x, y = z + rng.normal(size=150), z + rng.normal(size=150)
    got = spearman_test(x, y, controls=[z], n_draws=0)["rho"]
    expected = closed_form(x, y, z)
    assert abs(got - expected) < 1e-12, f"partial {got}, closed form {expected}"
    assert correlation(x, y, [z]) == got, "the coefficient alone must equal the test's rho"
    pearson = lambda a, b: np.corrcoef(a, b)[0, 1]
    assert abs(correlation(x, y, ranks=False) - pearson(x, y)) < 1e-12, "ranks=False must give Pearson's r"
    on_values = closed_form(x, y, z, pearson)
    assert abs(correlation(x, y, [z], ranks=False) - on_values) < 1e-12, "ranks=False must give the partial r"
    return f"partial: {got:.3f} equals the closed form, partial r {on_values:.3f}"


def check_within() -> str:
    rng = np.random.default_rng(2)
    sizes = {"a": 20, "b": 35, "c": 28}
    groups = np.concatenate([[name] * size for name, size in sizes.items()])
    shift = {"a": 0.0, "b": 5.0, "c": -3.0}
    z = rng.normal(size=len(groups))
    slope = np.array([{"a": 1.5, "b": -1.0, "c": 0.2}[g] for g in groups])
    x = z * slope + rng.normal(size=len(groups)) + np.array([shift[g] for g in groups])
    y = x + 0.5 * z + rng.normal(scale=0.5, size=len(groups))
    result = spearman_test(x, y, groups=groups)
    per_group = {name: stats.spearmanr(x[groups == name], y[groups == name]).statistic for name in sizes}
    weighted = sum(sizes[name] * rho for name, rho in per_group.items()) / sum(sizes.values())
    assert abs(result["rho"] - weighted) < 1e-12, f"rho_W {result['rho']}, weighted mean {weighted}"
    assert list(result["groups"]) == sorted(sizes), f"groups {result['groups']}"
    assert np.allclose(result["rho_k"], [per_group[name] for name in sorted(sizes)]), "per-group rho differs"
    assert np.allclose(result["n_k"], [sizes[name] for name in sorted(sizes)]), "per-group n differs"
    scores = result["x_scores"], result["y_scores"]
    assert abs(np.corrcoef(*scores)[0, 1] - weighted) < 1e-12, "the scores must correlate at rho_W"
    assert abs(np.polyfit(*scores, 1)[0] - weighted) < 1e-12, "the least-squares slope of the scores must be rho_W"
    assert result["p"] == FLOOR, f"within p {result['p']}"

    partial = spearman_test(x, y, controls=[z], groups=groups, n_draws=0)
    per_group = {name: closed_form(x[groups == name], y[groups == name], z[groups == name]) for name in sizes}
    weighted = sum(sizes[name] * rho for name, rho in per_group.items()) / sum(sizes.values())
    assert abs(partial["rho"] - weighted) < 1e-12, f"within partial {partial['rho']}, per-group slopes {weighted}"
    assert np.isnan(partial["p"]), "no draws must leave p undefined"
    scores = partial["x_scores"], partial["y_scores"]
    assert abs(np.polyfit(*scores, 1)[0] - weighted) < 1e-12, "partial scores must have slope rho_W"
    return f"within: rho_W {result['rho']:.3f}, partial with per-group slopes {partial['rho']:.3f}"


def check_exact() -> str:
    draws = reassignments(np.arange(8))
    assert draws.shape == (math.factorial(8) - 1, 8), f"{draws.shape} orderings of 8 items"
    assert len({tuple(row) for row in draws}) == len(draws), "orderings repeat"
    assert not (draws == np.arange(8)).all(axis=1).any(), "the observed ordering must be left out"
    ordered = np.arange(8.0)
    result = spearman_test(ordered, ordered)
    assert abs(result["rho"] - 1) < 1e-12 and abs(result["p"] - 2 / math.factorial(8)) < 1e-15, \
        f"perfect order rho {result['rho']}, p {result['p']}"
    labels = np.array([0] * 4 + [1] * 4 + [0] * 4)
    binary = reassignments(labels)
    assert binary.shape[0] == math.comb(12, 4) - 1, f"{binary.shape[0]} arrangements of 4 among 12"
    assert len({tuple(labels[row]) for row in binary}) == len(binary), "label arrangements repeat"
    assert permutation_p(0.5, np.array([0.6, -0.7, 0.1, 0.4])) == 0.6, "p must count the observed value"
    assert np.isnan(permutation_p(np.nan, np.array([0.1, 0.2]))), "an undefined value has no p"
    return f"exact: {len(draws)} orderings, perfect order p {result['p']:.1e}, {len(binary)} binary arrangements"


def symmetric(rng, n) -> np.ndarray:
    values = rng.uniform(0, 100, size=(n, n))
    return (values + values.T) / 2


def check_mantel() -> str:
    rng = np.random.default_rng(3)
    n = 40
    rows, cols = np.triu_indices(n, k=1)
    a = symmetric(rng, n)
    x = a[rows, cols]
    y = x + rng.normal(scale=10, size=len(x))
    result = spearman_test(x, y, pairs=(rows, cols))
    assert abs(result["rho"] - stats.spearmanr(x, y).statistic) < 1e-12, "pair rho differs from scipy"
    assert result["p"] == FLOOR, f"noisy copy p {result['p']}"
    unrelated = spearman_test(x, symmetric(rng, n)[rows, cols], pairs=(rows, cols))
    assert unrelated["p"] > 0.05, f"unrelated p {unrelated['p']}"

    gapped = y.copy()
    gapped[(rows == 0) | (cols == 0)] = np.nan
    gapped[(rows == 1) & (cols == 2)] = np.nan
    result = spearman_test(x, gapped, pairs=(rows, cols))
    expected = ~((rows == 0) | (cols == 0) | (rows == 1) | (cols == 1))
    assert (result["used"] == expected).all(), "words 0 and 1 must be dropped whole, ties to the first word"
    assert abs(result["rho"] - stats.spearmanr(x[expected], y[expected]).statistic) < 1e-12, \
        "rho must be computed over the complete pairs"

    members = {"a": 12, "b": 10, "c": 15}
    first, second, group, offset = [], [], [], 0
    for name, size in members.items():
        r, c = np.triu_indices(size, k=1)
        first += list(r + offset)
        second += list(c + offset)
        group += [name] * len(r)
        offset += size
    first, second, group = map(np.array, (first, second, group))
    effects = rng.normal(size=offset)
    model = effects[first] + effects[second] + rng.normal(size=len(first))
    human = effects[first] + effects[second] + rng.normal(scale=2, size=len(first))
    joint = spearman_test(model, human, groups=group, pairs=(first, second))
    for k, name in enumerate(joint["groups"]):
        alone = group == name
        single = spearman_test(model[alone], human[alone], groups=group[alone],
                               pairs=(first[alone], second[alone]))
        assert abs(single["rho"] - joint["rho_k"][k]) < 1e-12, f"{name}: rho alone {single['rho']}"
        assert single["p"] == joint["p_k"][k], f"{name}: p alone {single['p']}, jointly {joint['p_k'][k]}"
    return (f"mantel: rho {result['rho']:.3f} over complete pairs, unrelated p {unrelated['p']:.3f}, "
            f"stratified rho_W {joint['rho']:.3f} at p {joint['p']}")


def check_small_groups_and_alignment() -> str:
    rng = np.random.default_rng(4)
    groups = np.array(["a"] * 30 + ["b"] * 2)
    x, y = rng.normal(size=32), rng.normal(size=32)
    result = spearman_test(x, y, groups=groups, n_draws=0)
    assert list(result["groups"]) == ["a"] and result["n"] == 30, f"groups {result['groups']}, n {result['n']}"
    assert not result["used"][30:].any(), "the two-item group must be left out"
    labels, constant = np.array(["a"] * 30 + ["c"] * 20), np.concatenate([y[:30], np.ones(20)])
    undefined = spearman_test(np.concatenate([x[:30], rng.normal(size=20)]), constant, groups=labels, n_draws=0)
    assert undefined["n"] == 30 and abs(undefined["rho"] - stats.spearmanr(x[:30], y[:30]).statistic) < 1e-12, \
        f"a constant group must leave rho and n: rho {undefined['rho']}, n {undefined['n']}"

    categories = np.repeat(np.arange(6), 8)
    rows, cols = np.triu_indices(len(categories), k=1)
    same = categories[rows] == categories[cols]
    similarity = rng.normal(size=len(rows)) + 0.8 * same
    alignment = category_alignment_metrics(similarity, categories, n_draws=PERMUTATIONS)
    assert abs(alignment["roc_auc"] - roc_auc_score(same, similarity)) < 1e-12, "AUC differs from sklearn"
    assert "category_alignment_r" not in alignment, "the point-biserial r is dropped"
    rho = spearman_test(same.astype(float), similarity, pairs=(rows, cols))
    assert alignment["mantel_p"] == rho["p"], f"AUC p {alignment['mantel_p']}, rho p {rho['p']}"
    reversed_alignment = category_alignment_metrics(-similarity, categories, n_draws=PERMUTATIONS)
    assert reversed_alignment["mantel_p"] == alignment["mantel_p"], "alignment p must be two-sided"
    flat = category_alignment_metrics(np.zeros(len(rows)), categories, n_draws=PERMUTATIONS)
    assert flat["roc_auc"] == 0.5 and np.isnan(flat["mantel_p"]), f"constant similarity: {flat}"
    return f"small groups left out, alignment AUC {alignment['roc_auc']:.3f} at Mantel p {alignment['mantel_p']}"


def check_residuals() -> str:
    rng = np.random.default_rng(4)
    groups = np.repeat(["a", "b"], 30)
    control = rng.normal(size=60)
    values = np.where(groups == "a", 2.0, -1.0) * control + np.where(groups == "a", 10.0, 0.0) + rng.normal(size=60)
    assert np.allclose(residuals(values), values - values.mean()), "without controls, the deviation from the mean"
    expected_within, expected_partial = np.empty(60), np.empty(60)
    for name in ("a", "b"):
        rows = groups == name
        expected_within[rows] = values[rows] - values[rows].mean()
        expected_partial[rows] = values[rows] - np.polyval(np.polyfit(control[rows], values[rows], 1), control[rows])
    assert np.allclose(residuals(values, groups=groups), expected_within), "within groups, the group mean removed"
    assert np.allclose(residuals(values, [control], groups), expected_partial), "one control line per group"
    pooled = residuals(values, [control])
    return f"residuals: per-group control lines, sd {expected_partial.std():.2f} against pooled {pooled.std():.2f}"


def checks(failures: list) -> str:
    return ", ".join([check_scores(), check_pooled(), check_partial(), check_within(), check_exact(),
                      check_mantel(), check_small_groups_and_alignment(), check_residuals()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
