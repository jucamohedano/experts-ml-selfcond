"""
Family-resemblance typicality from the two Richie and Bhatia Study 1 datasets.

Read by build_concept_metadata through typicality_columns().

Explanations: documentation/dataset_and_metadata.md, sections 3 to 5.
"""

import functools
import glob
import pathlib
import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
ASSETS_DIR = REPO_ROOT / "abstractiveness/assets"
RICHIE_DIR = ASSETS_DIR / "Richie_and_Bhatia-HSJ"
PAIRWISE_DIR = RICHIE_DIR / "study1_pairwise_data/data_individual_level"
SPAM_FILE = RICHIE_DIR / "study1_spam_data.csv"
SPAM_SENTINEL = "....."
SCORE_DECIMALS = 3


def minmax_normalize_by_category(raw_by_category: dict) -> dict:
    """{category: {word: raw}} to {category: {word: 0 to 1}}."""
    normalized = {}
    for category, word_scores in raw_by_category.items():
        values = list(word_scores.values())
        low, high = min(values), max(values)
        span = high - low
        normalized[category] = {
            word: (round((value - low) / span, SCORE_DECIMALS) if span > 0 else 0.5)
            for word, value in word_scores.items()
        }
    return normalized


@functools.lru_cache(maxsize=1)
def load_pairwise_typicality() -> dict:
    """{category: {word: score}} from the pairwise ratings."""
    raw_by_category = {}
    for path in sorted(glob.glob(str(PAIRWISE_DIR / "*_pairwise.csv"))):
        category = pathlib.Path(path).stem.replace("_pairwise", "").lower()
        frame = pd.read_csv(path)
        pair_columns = frame.columns[1:]

        pair_means = {}
        for column in pair_columns:
            word_a, word_b = (part.strip().lower() for part in column.split("\\"))
            pair_means[(word_a, word_b)] = float(frame[column].dropna().mean())

        partner_means = {}
        for (word_a, word_b), value in pair_means.items():
            partner_means.setdefault(word_a, []).append(value)
            partner_means.setdefault(word_b, []).append(value)

        raw_by_category[category] = {
            word: float(np.mean(values)) for word, values in partner_means.items()
        }

    return minmax_normalize_by_category(raw_by_category)


@functools.lru_cache(maxsize=1)
def load_spam_typicality() -> dict:
    """{category: {word: score}} from the spatial arrangement trials."""
    spam = pd.read_csv(SPAM_FILE)
    stimulus_columns = [column for column in spam.columns if column.startswith("Stim")]

    raw_by_category = {}
    for category, group in spam.groupby("Concept"):
        word_distances = {}
        for _, row in group.iterrows():
            items = []
            for index, column in enumerate(stimulus_columns, start=1):
                word = str(row[column]).strip()
                if word == SPAM_SENTINEL:
                    continue
                items.append((word.lower(), row[f"Object{index}XFinal"], row[f"Object{index}YFinal"]))

            coordinates = np.array([(x, y) for _, x, y in items], dtype=float)
            for index, (word, _, _) in enumerate(items):
                others = np.delete(coordinates, index, axis=0)
                distances = np.linalg.norm(others - coordinates[index], axis=1)
                word_distances.setdefault(word, []).append(float(np.mean(distances)))

        raw_by_category[category.lower()] = {
            word: -float(np.mean(values)) for word, values in word_distances.items()
        }

    return minmax_normalize_by_category(raw_by_category)


def typicality_columns(word: str, category) -> dict:
    """The typicality fields for one concept."""
    if category is None:
        return {"typicality_HSJ_pairwise": None, "typicality_HSJ_spam": None}

    return {
        "typicality_HSJ_pairwise": load_pairwise_typicality().get(category, {}).get(word),
        "typicality_HSJ_spam": load_spam_typicality().get(category, {}).get(word),
    }
