"""
Module 5's pair-similarity study reports each cell's agreement with the human ratings as the pair-weighted mean of its
per-category Spearman coefficients, weights the noise ceiling alike, and compares nested cells by a paired t over the
held-out categories.

Runs run_pair_similarity on a synthetic pair table of five categories of unequal size, whose ratings follow the Jaccard
closely in some categories and loosely in others.

Checks:
  1. mean_category_spearman is the n_pairs-weighted mean of the per-category rho, which differs from their simple mean
     here, noise_ceiling_mean the weighted mean of the category ceilings, rho_over_ceiling their ratio, and
     delta_mean_rho_vs_jaccard the difference from the Jaccard cell.
  2. The pooled coefficient and the partial F columns are gone, and every nested comparison carries the two-sided paired
     t of the fuller cell's per-category rho against the reduced cell's, on the row of the fuller cell.

Usage, from scripts/:
    python tests/check_module5_pair_similarity_agreement.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import modules.module_5_typicality_prediction as m5
from _report import run

SIZES = {"birds": 40, "clothing": 60, "fruit": 80, "furniture": 100, "vehicles": 120}
SIGNAL = {"birds": 2.0, "clothing": 0.2, "fruit": 1.0, "furniture": 0.1, "vehicles": 1.5}


def make_table(seed: int = 0) -> pd.DataFrame:
    """A pair table shaped like build_pair_similarity_design's, the ratings tracking the Jaccard by category."""
    rng = np.random.default_rng(seed)
    frames = []
    for category, size in SIZES.items():
        jaccard = rng.uniform(0, 100, size)
        frames.append(pd.DataFrame({
            "category": category, "word_a": [f"{category}_a{i}" for i in range(size)],
            "word_b": [f"{category}_b{i}" for i in range(size)],
            "human_similarity": 4 + SIGNAL[category] * (jaccard - 50) / 50 + rng.normal(0, 1, size),
            "jaccard_pct": jaccard, "profile_js_distance": jaccard + rng.normal(0, 40, size),
            "profile_js_distance_z": rng.normal(0, 1, size)}))
    return pd.concat(frames, ignore_index=True)


def checks(failures: list) -> str:
    real_build_design = m5.build_pair_similarity_design
    m5.build_pair_similarity_design = lambda *args, **kwargs: make_table()
    try:
        results, per_category = m5.run_pair_similarity(pd.DataFrame(), pd.DataFrame(), ["js_distance"])
    finally:
        m5.build_pair_similarity_design = real_build_design

    rows = results.set_index(["model", "metric"])
    for key, cell in per_category.groupby(["model", "metric"]):
        row = rows.loc[key]
        weights = cell.n_pairs / cell.n_pairs.sum()
        assert abs(row.mean_category_spearman - (weights * cell.rho).sum()) < 2e-4, f"{key}: weighted rho"
        assert abs(row.noise_ceiling_mean - (weights * cell.noise_ceiling).sum()) < 2e-4, f"{key}: weighted ceiling"
        assert abs(row.rho_over_ceiling - row.mean_category_spearman / row.noise_ceiling_mean) < 5e-4, f"{key}: ratio"
    jaccard = per_category[per_category.model == "jaccard"]
    simple, weighted = jaccard.rho.mean(), rows.loc[("jaccard", "none")].mean_category_spearman
    assert abs(simple - weighted) > 0.01, f"fixture too even: simple mean {simple:.3f}, weighted {weighted:.3f}"
    richer = results[results.model != "jaccard"]
    assert np.allclose(richer.delta_mean_rho_vs_jaccard, richer.mean_category_spearman - weighted), "delta"

    stale = [c for c in results.columns if c.startswith("partial_f") or c in ("group_cv_spearman",
                                                                             "pooled_rho_over_ceiling")]
    assert not stale, f"columns that must be gone: {stale}"
    rho_k = per_category.pivot_table(index="category", columns=["model", "metric"], values="rho")
    tested = 0
    for reduced, full in m5.nested_pairs(m5.build_feature_grid(["js_distance"])):
        t, p = stats.ttest_rel(rho_k[(full["model"], full["metric"])], rho_k[(reduced["model"], reduced["metric"])])
        row = rows.loc[(full["model"], full["metric"])]
        reported = row.get(f"paired_t_vs_{reduced['model']}"), row.get(f"paired_t_p_vs_{reduced['model']}")
        assert reported[0] is not None and abs(reported[0] - t) < 1e-9 and abs(reported[1] - p) < 1e-12, \
            f"{full['model']} against {reduced['model']}: reported {reported}, expected t {t:.4f} p {p:.4f}"
        tested += 1
    return (f"{len(results)} cells, Jaccard weighted rho {weighted:.3f} against simple mean {simple:.3f}, "
            f"{tested} paired t over {rho_k.shape[0]} categories")


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
