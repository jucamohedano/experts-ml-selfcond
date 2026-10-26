"""
Module 2's typicality and frequency figures read the frequency of the corpus matched to the model and follow the
category rule: a correlation involving a category-relative variable is computed within categories.

Checks:
  1. Figures: per corpus two grouped figures, human typicality and that corpus's frequency, both reading the corpus's
     Zipf column and no SUBTLEX-US or Wikipedia column, every panel within categories except frequency against expert
     count, which involves no category-relative variable.
  2. Partial figures: on synthetic data in two categories whose typicality and frequency relate with opposite signs,
     each figure reports Spearman's rho over all members, the partial given the other measure over all members (the
     closed form on pooled ranks) and the partial within categories (the size-weighted mean of the closed forms on
     each category's ranks, one control slope per category), each with a two-sided permutation p and no Pearson
     column, while its within-category panel without control is the grouped figure's row, the size-weighted mean of
     the per-category coefficients.
  3. Outputs: each partial figure writes exactly one PNG and one CSV, whose rank-score columns reproduce the rho of
     all four panels as least-squares slopes.
  4. Section rows: section 2.5 computed without writing reports the corpus-named panels and the three partial rows,
     every key it reports is labelled, and a metadata without the corpus column is skipped rather than raising.
  5. Loader: expert_counts_with_metadata carries both corpus columns through to module 2.

Usage, from scripts/:
    python tests/check_module2_frequency.py
"""
import math
import pathlib
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.expert_data_loading import expert_counts_with_metadata
from modules import module_2_similarities as m2
from utils.statistics import PERMUTATIONS
from _report import run

OLD_COLUMNS = {"frequency_zipf_subtlex_us_lemma", "frequency_zipf_wikipedia_lemma", "log_frequency"}

FREQUENCY = "frequency_zipf_openwebtext_lemma"

# Per partial figure: x column, the other measure's column and name, and the grouped panel holding its within rho.
GRIDS = {"human_typicality": ("human_typicality", FREQUENCY, "frequency", "human_typicality_vs_jaccard"),
         "frequency": (FREQUENCY, "human_typicality", "human_typicality", "frequency_openwebtext_vs_jaccard")}


def synthetic(n: int = 60, seed: int = 0) -> tuple:
    """Metadata and concept-to-label table where Jaccard depends on typicality, frequency and category."""
    rng = np.random.default_rng(seed)
    category = np.where(np.arange(n) < n // 2, "a", "b")
    typicality = rng.uniform(0, 1, n)
    frequency = 3 + np.where(category == "a", 0.8, -0.8) * typicality + rng.normal(0, 0.3, n)
    jaccard = 10 * typicality + 4 * frequency + np.where(category == "a", 5.0, 0.0) + rng.normal(0, 2, n)
    meta = pd.DataFrame({"concept": [f"c{i}" for i in range(n)], "category": category, "abstraction_level": 2,
                         "expert_count": rng.integers(10, 100, n), "human_typicality": typicality,
                         FREQUENCY: frequency, "frequency_zipf_fineweb_lemma": frequency + 0.2})
    sim = pd.DataFrame({"concept": meta["concept"], "category": category, "jaccard_pct": jaccard,
                        "overlap_pct": 1.5 * jaccard})
    return meta, sim


def closed_form(x, y, z) -> float:
    """First-order partial Spearman from the three pairwise coefficients."""
    xy, xz, yz = (spearmanr(a, b).statistic for a, b in ((x, y), (x, z), (y, z)))
    return (xy - xz * yz) / math.sqrt((1 - xz ** 2) * (1 - yz ** 2))


def within_mean(frame: pd.DataFrame, coefficient) -> float:
    """Size-weighted mean over categories of a coefficient computed on one category's rows."""
    return sum(coefficient(group) * len(group) for _, group in frame.groupby("category")) / len(frame)


def check_figures() -> str:
    """Each corpus gets the typicality figure and its own frequency figure, within categories but for expert count."""
    for corpus, (column, _) in m2.FREQUENCY_CORPORA.items():
        figures = m2.grouped_correlation_figures(100, 90, corpus)
        names = [figure[2] for figure in figures]
        assert names == ["human_typicality", f"frequency_{corpus}"], f"{corpus}: figures {names}"
        typicality, frequency = figures
        assert frequency[0] == column, f"{corpus}: frequency figure reads {frequency[0]}"
        assert column in [panel.y_col for panel in typicality[4]], f"{corpus}: typicality figure misses {column}"
        used = {figure[0] for figure in figures} | {panel.y_col for figure in figures for panel in figure[4]}
        assert not used & OLD_COLUMNS, f"{corpus}: still reads {sorted(used & OLD_COLUMNS)}"
        assert all(panel.within for panel in typicality[4]), f"{corpus}: a typicality panel is pooled"
        pooled = [panel.y_col for panel in frequency[4] if not panel.within]
        assert pooled == ["expert_count"], f"{corpus}: pooled frequency panels {pooled}"
    return f"figures: {len(m2.FREQUENCY_CORPORA)} corpora, two figures each, pooled only against expert count"


def partial_rows(meta: pd.DataFrame, sim: pd.DataFrame, corr_dir=None) -> dict:
    """The rows of both partial figures, keyed by plot name."""
    rows = (m2.plot_human_typicality_partial_correlations(meta, sim, meta, corr_dir, "openwebtext")
            + m2.plot_frequency_partial_correlations(meta, sim, meta, corr_dir, "openwebtext"))
    assert len(rows) == 6, f"two figures of three rows each, got {len(rows)} rows"
    return {row["plot_name"]: row for row in rows}


def check_partials() -> str:
    """The rows equal the pooled and per-category closed forms, the within panel being the grouped figure's row."""
    meta, sim = synthetic()
    joined = meta.merge(sim, on=["concept", "category"])
    rows = partial_rows(meta, sim)
    grouped = {row["plot_name"]: row for row in m2.plot_grouped_correlations(
        meta, sim, meta, None, ["human_typicality", "frequency_openwebtext"], "openwebtext")}
    floor = 1 / (PERMUTATIONS + 1)
    for name, (x, other, other_name, grouped_name) in GRIDS.items():
        base = f"partial_correlation_jaccard_{name}"
        expected = {
            f"{base}_no_control": ("pooled", "", spearmanr(joined[x], joined["jaccard_pct"]).statistic),
            base: ("pooled", other, closed_form(joined[x], joined["jaccard_pct"], joined[other])),
            f"{base}_within_given_{other_name}": ("within category", other, within_mean(
                joined, lambda group: closed_form(group[x], group["jaccard_pct"], group[other]))),
        }
        for plot_name, (ranks, controls, value) in expected.items():
            row = rows[plot_name]
            assert abs(row["spearman_rho"] - value) < 1e-9, f"{plot_name}: rho {row['spearman_rho']:.6f}, {value:.6f}"
            assert row["ranks"] == ranks and row["controlling_for"] == controls, f"{plot_name}: {row}"
            assert floor <= row["spearman_p"] <= 1, f"{plot_name}: p {row['spearman_p']}"
            assert "pearson_r" not in row and "pearson_p" not in row, f"{plot_name}: Pearson columns must be gone"
        within = grouped[grouped_name]
        value = within_mean(joined, lambda group: spearmanr(group[x], group["jaccard_pct"]).statistic)
        assert abs(within["spearman_rho"] - value) < 1e-9, f"{grouped_name}: {within['spearman_rho']:.6f}, {value:.6f}"
        assert within["ranks"] == "within category", f"{grouped_name}: ranks {within['ranks']}"
    common = rows["partial_correlation_jaccard_human_typicality"]["spearman_rho"]
    per_category = rows["partial_correlation_jaccard_human_typicality_within_given_frequency"]["spearman_rho"]
    return (f"partials: Jaccard-typicality given frequency {common:.3f} over all members, {per_category:.3f} within "
            f"categories, every row equal to its closed form")


def check_outputs() -> str:
    """Each partial figure writes one PNG and one CSV, whose score columns reproduce every panel's rho."""
    meta, sim = synthetic()
    joined = meta.merge(sim, on=["concept", "category"])
    stems = {"human_typicality": "human_typicality_partial_correlations",
             "frequency": "frequency_openwebtext_partial_correlations"}
    with tempfile.TemporaryDirectory() as folder:
        rows = partial_rows(meta, sim, pathlib.Path(folder))
        files = sorted(path.name for path in pathlib.Path(folder).iterdir())
        assert files == sorted(f"{stem}.{ext}" for stem in stems.values() for ext in ("csv", "png")), f"files {files}"
        tables = {name: pd.read_csv(pathlib.Path(folder) / f"{stem}.csv") for name, stem in stems.items()}
    for name, (x, _, other_name, _) in GRIDS.items():
        base = f"partial_correlation_jaccard_{name}"
        expected = {"no_control": rows[f"{base}_no_control"]["spearman_rho"],
                    f"given_{other_name}": rows[base]["spearman_rho"],
                    "within": within_mean(joined, lambda group: spearmanr(group[x], group["jaccard_pct"]).statistic),
                    f"within_given_{other_name}": rows[f"{base}_within_given_{other_name}"]["spearman_rho"]}
        for key, rho in expected.items():
            table = tables[name]
            slope = np.polyfit(table[f"{x}_rank_z_{key}"], table[f"jaccard_pct_rank_z_{key}"], 1)[0]
            assert abs(slope - rho) < 1e-9, f"{stems[name]} {key}: CSV slope {slope:.6f}, rho {rho:.6f}"
    return f"outputs: {len(stems)} figures, one PNG and one CSV each, CSV columns reproduce all eight panels"


def check_section_rows() -> str:
    """Section 2.5 reports the corpus-named panels and the three partial rows, and skips a missing column."""
    meta, sim = synthetic()
    scope = SimpleNamespace(label="whole_model", is_whole_model=True, expert_df=pd.DataFrame())
    row = m2.run_frequency_correlations(scope, meta, meta, sim, None, "fineweb")
    expected = {f"rho_frequency_fineweb_vs_{y}" for y in ("expert_count", "jaccard", "overlap", "human_typicality")}
    expected |= {f"rho_partial_correlation_jaccard_frequency{suffix}"
                 for suffix in ("", "_no_control", "_within_given_human_typicality")}
    assert expected <= set(row), f"missing {sorted(expected - set(row))}"
    stale = [key for key in row if "frequency_us" in key or "frequency_wiki" in key or "_category" in key]
    assert not stale, f"stale keys {stale}"
    reported = {key for key in row if key.startswith("rho_")}
    unlabelled = reported - set(m2.FREQUENCY_SUMMARY_LABELS)
    assert not unlabelled, f"unlabelled summary keys {sorted(unlabelled)}"

    bare = meta.drop(columns=[FREQUENCY])
    assert m2.plot_human_typicality_partial_correlations(bare, sim, bare, None, "openwebtext") == [], \
        "missing column must skip"
    assert m2.plot_frequency_partial_correlations(bare, sim, bare, None, "openwebtext") == [], \
        "missing column must skip"
    assert m2.plot_grouped_correlations(bare, sim, bare, None, ["frequency_openwebtext"], "openwebtext") == [], \
        "missing column must skip the figure"
    return f"section 2.5: {len(reported)} labelled keys, a missing corpus column is skipped"


def check_loader() -> str:
    """The merged metadata module 2 reads keeps every corpus frequency column."""
    meta, _ = synthetic()
    experts = pd.DataFrame({"concept": np.repeat(meta["concept"].to_numpy(), 2)})
    merged = expert_counts_with_metadata(experts, meta.drop(columns=["expert_count"]).assign(frequency=100))
    columns = [column for column, _ in m2.FREQUENCY_CORPORA.values()]
    missing = [column for column in columns if column not in merged.columns]
    assert not missing, f"expert_counts_with_metadata drops {missing}"
    return f"loader: {len(columns)} corpus columns carried"


def checks(failures: list) -> str:
    return ", ".join([check_figures(), check_partials(), check_outputs(), check_section_rows(), check_loader()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
