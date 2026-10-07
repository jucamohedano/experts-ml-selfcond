"""
Module 2 reads the frequency of the corpus matched to the model, and partials frequency and typicality out of each other's relation with Jaccard.

Checks:
  1. Figures: per corpus exactly two grouped figures, human typicality and that corpus's frequency,
     both reading the corpus's Zipf column, with no SUBTLEX-US or Wikipedia column left in any figure.
  2. Partial correlations: on synthetic data, Jaccard against typicality controlling frequency and
     Jaccard against frequency controlling typicality both equal the closed-form first-order partial
     correlation, record their control column, and differ from each other.
  3. Section rows: section 2.5 computed without writing reports the corpus-named panels and the
     new partial, every key it reports is labelled, and a metadata without the corpus column is
     skipped rather than raising.
  4. Loader: expert_counts_with_metadata carries both corpus columns through to module 2.

Usage, from scripts/:
    python tests/check_module2_frequency.py
"""
import math
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.expert_data_loading import expert_counts_with_metadata
from modules import module_2_similarities as m2
from _report import run

OLD_COLUMNS = {"frequency_zipf_subtlex_us_lemma", "frequency_zipf_wikipedia_lemma", "log_frequency"}


def synthetic(n: int = 60, seed: int = 0) -> tuple:
    """Metadata and concept-to-label table where Jaccard depends on both typicality and frequency."""
    rng = np.random.default_rng(seed)
    typicality = rng.uniform(0, 1, n)
    frequency = 3 + 0.6 * typicality + rng.normal(0, 0.3, n)
    jaccard = 10 * typicality + 4 * frequency + rng.normal(0, 2, n)
    meta = pd.DataFrame({
        "concept": [f"c{i}" for i in range(n)],
        "category": ["a"] * (n // 2) + ["b"] * (n - n // 2),
        "abstraction_level": 2,
        "expert_count": rng.integers(10, 100, n),
        "human_typicality": typicality,
        "frequency_zipf_openwebtext_lemma": frequency,
        "frequency_zipf_fineweb_lemma": frequency + 0.2,
    })
    sim = pd.DataFrame({"concept": meta["concept"], "category": meta["category"],
                        "jaccard_pct": jaccard, "overlap_pct": 1.5 * jaccard})
    return meta, sim


def closed_form_partial(x, y, z) -> float:
    """First-order partial correlation of x and y given z, from the three pairwise correlations."""
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((x, y), (x, z), (y, z)))
    return (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def check_figures() -> str:
    """Each corpus gets the typicality figure and its own frequency figure, nothing else."""
    for corpus, (column, _) in m2.FREQUENCY_CORPORA.items():
        figures = m2.grouped_correlation_figures(100, 90, corpus)
        names = [figure[2] for figure in figures]
        assert names == ["human_typicality", f"frequency_{corpus}"], f"{corpus}: figures {names}"
        assert figures[1][0] == column, f"{corpus}: frequency figure reads {figures[1][0]}"
        typicality_panels = [panel[0] for panel in figures[0][4]]
        assert column in typicality_panels, f"{corpus}: typicality figure panels {typicality_panels}"
        used = {figure[0] for figure in figures} | {panel[0] for figure in figures for panel in figure[4]}
        assert not used & OLD_COLUMNS, f"{corpus}: still reads {sorted(used & OLD_COLUMNS)}"
    return f"figures: {len(m2.FREQUENCY_CORPORA)} corpora, two figures each on the matched column"


def check_partials() -> str:
    """Both partial correlations reproduce the closed form and are not the same number."""
    meta, sim = synthetic()
    joined = meta.merge(sim, on=["concept", "category"])
    column = "frequency_zipf_openwebtext_lemma"
    by_frequency = m2.plot_partial_correlation(meta, sim, meta, None, "openwebtext")
    by_typicality = m2.plot_partial_correlation_jaccard_frequency(meta, sim, meta, None, "openwebtext")
    assert len(by_frequency) == 1 and len(by_typicality) == 1, "each partial returns one summary row"
    typicality_row, frequency_row = by_frequency[0], by_typicality[0]
    expected_typicality = closed_form_partial(joined["jaccard_pct"], joined["human_typicality"], joined[column])
    expected_frequency = closed_form_partial(joined["jaccard_pct"], joined[column], joined["human_typicality"])
    assert abs(typicality_row["pearson_r"] - expected_typicality) < 1e-9, \
        f"Jaccard-typicality partial {typicality_row['pearson_r']:.6f}, closed form {expected_typicality:.6f}"
    assert abs(frequency_row["pearson_r"] - expected_frequency) < 1e-9, \
        f"Jaccard-frequency partial {frequency_row['pearson_r']:.6f}, closed form {expected_frequency:.6f}"
    assert typicality_row["controlling_for"] == column, f"controls for {typicality_row['controlling_for']}"
    assert frequency_row["controlling_for"] == "human_typicality", f"controls for {frequency_row['controlling_for']}"
    assert abs(typicality_row["pearson_r"] - frequency_row["pearson_r"]) > 0.05, "the two partials coincide"
    return (f"partials: Jaccard-typicality given frequency {typicality_row['pearson_r']:.3f}, "
            f"Jaccard-frequency given typicality {frequency_row['pearson_r']:.3f}, both equal the closed form")


def check_section_rows() -> str:
    """Section 2.5 reports the corpus-named panels and the new partial, and skips a missing column."""
    meta, sim = synthetic()
    scope = SimpleNamespace(label="whole_model", is_whole_model=True, expert_df=pd.DataFrame())
    row = m2.run_frequency_correlations(scope, meta, meta, sim, None, "fineweb")
    expected = {f"r_frequency_fineweb_vs_{y}" for y in ("expert_count", "jaccard", "overlap", "human_typicality")}
    expected.add("r_partial_correlation_jaccard_frequency")
    assert expected <= set(row), f"missing {sorted(expected - set(row))}"
    stale = [key for key in row if "frequency_us" in key or "frequency_wiki" in key]
    assert not stale, f"stale keys {stale}"
    reported = {key for key in row if key.startswith("r_")}
    unlabelled = reported - set(m2.FREQUENCY_SUMMARY_LABELS)
    assert not unlabelled, f"unlabelled summary keys {sorted(unlabelled)}"

    bare = meta.drop(columns=["frequency_zipf_openwebtext_lemma"])
    assert m2.plot_partial_correlation(bare, sim, bare, None, "openwebtext") == [], "missing column must skip"
    assert m2.plot_partial_correlation_jaccard_frequency(bare, sim, bare, None, "openwebtext") == [], \
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
    return ", ".join([check_figures(), check_partials(), check_section_rows(), check_loader()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
