"""
Correlates every lemma frequency column of the concept metadata with every other, over unique surface words.

Run after build_concept_metadata, its table feeds the frequency section of the documentation.

Usage, from scripts/:
    python -m core.data_preparation.compare_frequency_sources

Explanations: documentation/dataset_and_metadata.md, section 6.
"""

import itertools
import json
import pathlib

import pandas as pd
from scipy import stats

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
METADATA = REPO_ROOT / "abstractiveness/assets/metadata_Richie_HSJ.json"
OUTPUT_CSV = REPO_ROOT / "abstractiveness/results/frequency_sources/frequency_source_correlations.csv"
SOURCES = {
    "wikipedia": "frequency_zipf_wikipedia_lemma",
    "openwebtext": "frequency_zipf_openwebtext_lemma",
    "fineweb": "frequency_zipf_fineweb_lemma",
    "subtlex_us": "frequency_zipf_subtlex_us_lemma",
}


def unique_words(metadata: list) -> pd.DataFrame:
    """One row per surface word, since the two squash concepts share one frequency."""
    frame = pd.DataFrame(metadata)
    return frame.drop_duplicates(subset="word")[["word", *SOURCES.values()]].reset_index(drop=True)


def source_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Pearson r, Spearman rho and mean Zipf offset for every pair of sources, over the words both cover."""
    rows = []
    for a, b in itertools.combinations(SOURCES, 2):
        pair = frame[[SOURCES[a], SOURCES[b]]].dropna().astype(float)
        x, y = pair.iloc[:, 0], pair.iloc[:, 1]
        rows.append({
            "source_a": a,
            "source_b": b,
            "n": len(pair),
            "pearson_r": stats.pearsonr(x, y)[0],
            "spearman_rho": stats.spearmanr(x, y)[0],
            "mean_zipf_difference": float((x - y).mean()),
        })
    return pd.DataFrame(rows)


def main():
    table = source_correlations(unique_words(json.load(METADATA.open())))
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
