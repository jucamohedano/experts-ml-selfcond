"""
Module 2's concept-level correlations that involve a category-relative variable are computed within categories, where
category offsets cannot enter them.

Checks, on three categories whose Jaccard levels differ by far more than the within-category spread:
  1. Section 2.1's per-category table reports each category's Spearman rho of Jaccard against human typicality with
     its permutation p, the components of section 2.4's within-category panel, whose rho is their member-weighted mean.
  2. Jaccard against layer profile (2.1) and cosine typicality against Jaccard (2.4) are within-category rows whose
     rho is the member-weighted mean of the per-category coefficients.
  3. Cosine against human typicality (2.4) reports the member-weighted mean of the per-category coefficients with a p
     in range, and each category's scatter draws its raw typicality values with the least-squares line, no band and
     the category's own rho.

Usage, from scripts/:
    python tests/check_module2_within_category.py
"""
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules import module_2_similarities as m2
from utils.statistics import PERMUTATIONS
from _figures import closed_figures, has_band, line_slope, points, same_points
from _report import run

FLOOR = 1 / (PERMUTATIONS + 1)


def synthetic(seed: int = 0) -> tuple:
    """Metadata, concept-to-label table and cosine typicality of three categories of twelve members."""
    rng = np.random.default_rng(seed)
    categories = np.repeat(["birds", "fruit", "tools"], 12)
    concepts = [f"{category}{i}" for i, category in enumerate(categories)]
    typicality = rng.uniform(0, 1, len(categories))
    offset = pd.Series(categories).map({"birds": 0.0, "fruit": 30.0, "tools": 60.0}).to_numpy()
    jaccard = 10 * typicality + offset + rng.normal(0, 3, len(categories))
    meta = pd.DataFrame({"concept": concepts, "category": categories, "human_typicality": typicality,
                         "expert_count": rng.integers(10, 100, len(categories))})
    sim = pd.DataFrame({"concept": concepts, "category": categories, "jaccard_pct": jaccard,
                        "overlap_pct": 1.2 * jaccard, "category_expert_units": 100,
                        "layer_profile_similarity_pct": 0.5 * jaccard + rng.normal(0, 3, len(categories))})
    cosine = pd.DataFrame({"category": categories, "concept": concepts, "human_typicality": typicality,
                           "global_cosine_typicality": typicality + rng.normal(0, 0.3, len(categories))})
    return meta, sim, cosine


def within_mean(frame: pd.DataFrame, x: str, y: str) -> float:
    """Member-weighted mean of the per-category Spearman coefficients."""
    return sum(spearmanr(group[x], group[y]).statistic * len(group)
               for _, group in frame.groupby("category")) / len(frame)


def check_label_table() -> str:
    meta, sim, _ = synthetic()
    joined = sim.merge(meta[["concept", "human_typicality"]], on="concept")
    table = m2.category_similarity_summary(sim, meta).set_index("category")
    for category, group in joined.groupby("category"):
        expected = spearmanr(group["jaccard_pct"], group["human_typicality"]).statistic
        assert abs(table.loc[category, "rho_jaccard_human_typicality"] - expected) < 1e-12, f"{category}: rho"
        assert FLOOR <= table.loc[category, "p_jaccard_human_typicality"] <= 1, f"{category}: p"
        assert table.loc[category, "n_with_typicality"] == len(group), f"{category}: n"
    panel = {row["plot_name"]: row for row in m2.plot_grouped_correlations(
        meta, sim, meta, None, ["human_typicality"], "openwebtext")}["human_typicality_vs_jaccard"]
    weights = table["n_with_typicality"] / table["n_with_typicality"].sum()
    weighted = (weights * table["rho_jaccard_human_typicality"]).sum()
    assert abs(panel["spearman_rho"] - weighted) < 1e-12, f"2.4 panel {panel['spearman_rho']}, components {weighted}"
    pooled = spearmanr(joined["jaccard_pct"], joined["human_typicality"]).statistic
    return f"2.1 table: per-category rho with p, weighted {weighted:.3f} where the pooled rho reads {pooled:.3f}"


def check_label_rows() -> str:
    meta, sim, cosine = synthetic()
    profile = m2.plot_jaccard_vs_layer_profile(meta, sim, meta, None)[0]
    expected = within_mean(sim, "jaccard_pct", "layer_profile_similarity_pct")
    assert profile["ranks"] == "within category" and abs(profile["spearman_rho"] - expected) < 1e-12, f"{profile}"
    typicality = m2.plot_jaccard_vs_cosine_typicality(sim, cosine, meta, None)[0]
    expected = within_mean(sim.merge(cosine, on=["concept", "category"]), "global_cosine_typicality", "jaccard_pct")
    assert typicality["ranks"] == "within category" and abs(typicality["spearman_rho"] - expected) < 1e-12, \
        f"{typicality}"
    return (f"rows: Jaccard-profile {profile['spearman_rho']:.3f}, cosine typicality-Jaccard "
            f"{typicality['spearman_rho']:.3f}, both within categories")


def check_cosine_typicality() -> str:
    _, _, cosine = synthetic()
    test = m2.typicality_agreement(cosine)
    row = m2._summarize_typicality(test)
    expected = within_mean(cosine, "human_typicality", "global_cosine_typicality")
    assert abs(row["typicality_spearman_rho"] - expected) < 1e-12, f"rho {row['typicality_spearman_rho']}"
    assert FLOOR <= row["typicality_spearman_p"] <= 1 and row["n_typicality_concepts"] == len(cosine), f"{row}"
    with tempfile.TemporaryDirectory() as folder:
        _, figures = closed_figures(m2.generate_global_typicality_reports, cosine, test, pathlib.Path(folder))
    scatters = [f for f in figures if f.axes and "against cosine typicality" in f.axes[0].get_title()]
    assert len(scatters) == cosine.category.nunique(), f"{len(scatters)} scatters"
    for figure in scatters:
        axis = figure.axes[0]
        group = cosine[cosine.category == axis.get_title().split(",")[0].lower()]
        assert same_points(points(axis), group.human_typicality, group.global_cosine_typicality), "raw values drawn"
        expected = np.polyfit(group.human_typicality, group.global_cosine_typicality, 1)[0]
        assert abs(line_slope(axis) - expected) < 1e-9 and not has_band(axis), "least-squares line without a band"
        assert axis.get_xlabel() == "Human typicality", f"x label {axis.get_xlabel()!r}"
    empty = m2._summarize_typicality(m2.typicality_agreement(cosine.iloc[:0]))
    assert np.isnan(empty["typicality_spearman_rho"]) and empty["n_typicality_concepts"] == 0, f"{empty}"
    return f"cosine typicality: weighted rho {row['typicality_spearman_rho']:.3f} at p {row['typicality_spearman_p']}"


def checks(failures: list) -> str:
    return ", ".join([check_label_table(), check_label_rows(), check_cosine_typicality()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
