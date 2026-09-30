"""
Richie and Bhatia Study 1 pairwise similarity ratings, per pair and as split-half noise ceilings.

Used by module 2's human validation (section 2.3) and module 5's pair-similarity study.

Explanations: documentation/dataset_and_metadata.md.
"""

import functools
import json
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from utils.statistics import spearman_brown


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

DATA_DIR = (REPO_ROOT / "abstractiveness/assets/Richie_and_Bhatia-HSJ"
            / "study1_pairwise_data/data_individual_level")

METADATA_FILE = REPO_ROOT / "abstractiveness/assets/metadata_Richie_HSJ.json"

CATEGORY_FILES = {
    "birds": "Birds_pairwise.csv",
    "clothing": "Clothing_pairwise.csv",
    "fruit": "Fruit_pairwise.csv",
    "furniture": "Furniture_pairwise.csv",
    "professions": "Professions_pairwise.csv",
    "sports": "Sports_pairwise.csv",
    "vegetables": "Vegetables_pairwise.csv",
    "vehicles": "Vehicles_pairwise.csv",
}

MIN_SPLIT_OVERLAP = 10


def _parse_pair(column: str) -> tuple:
    """Column headers read "word_a \\ word_b"."""
    left, right = [part.strip() for part in column.split("\\")]
    return (left, right) if left < right else (right, left)


@functools.lru_cache(maxsize=1)
def _concept_keys() -> dict:
    """(category, word) to the concept key that pair member joins on."""
    with open(METADATA_FILE, "r", encoding="utf-8") as handle:
        return {(e["category"], e.get("word", e["concept"])): e["concept"]
                for e in json.load(handle) if e["category"]}


def concept_for(category: str, word: str) -> str:
    """The concept key for one pair member, falling back to the bare word."""
    return _concept_keys().get((category, word), word)


def _category_frame(category: str) -> tuple:
    """The raw subject-by-pair frame for one category, and its usable column names."""
    frame = pd.read_csv(DATA_DIR / CATEGORY_FILES[category], index_col=0)
    return frame, list(frame.columns)


@functools.lru_cache(maxsize=1)
def load_human_similarity() -> pd.DataFrame:
    """One row per rated within-category pair."""
    rows = []
    for category in CATEGORY_FILES:
        frame, columns = _category_frame(category)
        for column in columns:
            word_a, word_b = _parse_pair(column)
            values = frame[column].dropna()
            rows.append({"category": category, "word_a": word_a, "word_b": word_b,
                         "concept_a": concept_for(category, word_a),
                         "concept_b": concept_for(category, word_b),
                         "mean_rating": float(values.mean()), "n_raters": int(len(values))})
    return pd.DataFrame(rows)


@functools.lru_cache(maxsize=1)
def human_pair_lookup() -> dict:
    """(concept_a, concept_b) concept keys in sorted order -> mean rating, for per-pair joins."""
    pairs = load_human_similarity()
    return {tuple(sorted((a, b))): r for a, b, r in zip(pairs.concept_a, pairs.concept_b, pairs.mean_rating)}


@functools.lru_cache(maxsize=8)
def human_noise_ceiling(n_splits: int = 200, seed: int = 0) -> dict:
    """Per-category split-half reliability, Spearman-Brown corrected."""
    rng = np.random.default_rng(seed)
    ceilings = {}
    for category in CATEGORY_FILES:
        frame, columns = _category_frame(category)
        subset = frame[columns]
        rhos = []
        for _ in range(n_splits):
            order = rng.permutation(len(subset))
            first = subset.iloc[order[: len(subset) // 2]].mean()
            second = subset.iloc[order[len(subset) // 2:]].mean()
            usable = first.notna() & second.notna()
            if int(usable.sum()) > MIN_SPLIT_OVERLAP:
                rhos.append(spearmanr(first[usable], second[usable]).statistic)
        ceilings[category] = float(spearman_brown(float(np.mean(rhos))))
    return ceilings
