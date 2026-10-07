"""
Builds assets/metadata_Richie_HSJ.json from the Richie and Bhatia table 1 word list.

Composes the columns of frequency_covariates and typicality_covariates, which share one contract:
load_*() is a memoised reader of one external input, <name>_columns() returns the metadata fields
that file owns, keyed as they appear in the JSON.

Usage, from scripts/:
    python -m core.data_preparation.build_concept_metadata

Explanations: documentation/dataset_and_metadata.md, sections 1, 2, 6 and 8.
"""

import functools
import json
import pathlib
import pandas as pd
from core.data_preparation.frequency_covariates import frequency_columns
from core.data_preparation.typicality_covariates import typicality_columns

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
ASSETS_DIR = REPO_ROOT / "abstractiveness/assets"
WORDLIST_CSV = ASSETS_DIR / "Richie_and_Bhatia-HSJ/table 1 - word lists.csv"
OUTPUT_JSON = ASSETS_DIR / "metadata_Richie_HSJ.json"
SENSE_SEPARATOR = "__"


@functools.lru_cache(maxsize=1)
def load_categories() -> dict:
    """{category: [members]}, one column per category, blanks dropped, all lowercased."""
    frame = pd.read_csv(WORDLIST_CSV)
    categories = {}
    for column in frame.columns:
        members = frame[column].dropna().astype(str).str.strip().str.lower().tolist()
        categories[column.strip().lower()] = [m for m in members if m]
    return categories


def ambiguous_words(categories: dict) -> set:
    """Surface forms the word list places under more than one category."""
    members = [word for concepts in categories.values() for word in concepts]
    return {word for word in members if members.count(word) > 1}


def concept_key(word: str, category: str, ambiguous: set) -> str:
    """The join key for one concept, per-sense only where the word list forces it."""
    return f"{word}{SENSE_SEPARATOR}{category}" if word in ambiguous else word


def identity_columns(word: str, category, ambiguous: set) -> dict:
    """The fields this preparer owns: the join key, the surface form, and the position of the concept in the hierarchy."""
    return {
        "concept": concept_key(word, category, ambiguous) if category else word,
        "word": word,
        "abstraction_level": 2 if category else 1,
        "category": category,
    }


def build_metadata() -> list:
    """One row per concept: the 8 category labels at level 1, their members at level 2."""
    categories = load_categories()
    ambiguous = ambiguous_words(categories)
    if ambiguous:
        print(f"Words listed under two categories, keyed per sense: {sorted(ambiguous)}")

    rows = []
    for category, concepts in categories.items():
        for word, parent in [(category, None)] + [(w, category) for w in concepts]:
            row = identity_columns(word, parent, ambiguous)
            row.update(frequency_columns(word))
            row.update(typicality_columns(word, parent))
            rows.append(row)
    return rows


def main():
    rows = build_metadata()
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=4)
        handle.write("\n")

    n_level1 = sum(1 for row in rows if row["abstraction_level"] == 1)
    n_level2 = len(rows) - n_level1
    filled = lambda key: sum(1 for row in rows if row[key] is not None)
    print(f"Wrote {len(rows)} entries to {OUTPUT_JSON} "
          f"({n_level1} level 1, {n_level2} level 2)")
    for key in ("frequency_zipf_subtlex_us_lemma", "frequency_zipf_wikipedia_lemma",
                "frequency_zipf_openwebtext_lemma", "frequency_zipf_fineweb_lemma"):
        print(f"  {key}: {filled(key)}/{len(rows)}")
    for key in ("typicality_HSJ_pairwise", "typicality_HSJ_spam"):
        print(f"  {key}: {filled(key)}/{n_level2} level 2")


if __name__ == "__main__":
    main()
