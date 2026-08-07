"""
Human pairwise similarity judgments from Richie and Bhatia's "SpAM on words" Study 1.

Participants rated every within-category pair on a 1 to 7 scale where HIGHER MEANS MORE
SIMILAR, the same direction as every expert-side metric in this pipeline, so a positive
correlation is the expected sign everywhere and nothing needs flipping at the call site.

Verified by inspection rather than assumed: the highest rated fruit pairs are
melon/watermelon at 6.45 and melon/honeydew at 6.26, the lowest is olive/banana at 1.58,
and in birds rooster/chicken reaches 6.44 against crow/penguin at 1.62.

Four categories (birds, clothing, professions, sports) used a half-sampled between-subjects
design, so 44 to 52 percent of the subject-by-pair cells are empty, but every pair still
carries between 19 and 39 raters. Unequal rater counts make some pair means noisier than
others, which attenuates any correlation rather than inflating it, and the split-half
ceiling below is what turns that into a number you can divide by.

The source table lists "squash" under both sports and vegetables. This dataset admits it
under vegetables only (see documentation/fixes.md), so sports pairs containing it are
dropped, leaving 2,391 pairs, which is exactly the pair count module 9's ranker already
enumerates combinatorially from the same word list.

Nothing here depends on the AP threshold or the analysis scope, so both entry points are
memoised: the executor calls them 40 times per model and the answer never changes.
"""
import functools
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.helpers import spearman_brown

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = (REPO_ROOT / "abstractiveness/assets/Richie_and_Bhatia-HSJ"
            / "study1_pairwise_data/data_individual_level")

# Metadata uses lowercase category names, the files are capitalised.
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

# (category, word) combinations present in the source files but not in this dataset.
EXCLUDED_MEMBERS = {("sports", "squash")}

# A split half with fewer overlapping pairs than this cannot support a stable correlation.
MIN_SPLIT_OVERLAP = 10


def _parse_pair(column: str) -> tuple:
    """Column headers read "word_a \\ word_b". Returned in canonical (lower, higher) order
    so a lookup never depends on which way round the header happened to be written."""
    left, right = [part.strip() for part in column.split("\\")]
    return (left, right) if left < right else (right, left)


def _category_frame(category: str) -> tuple:
    """The raw subject-by-pair frame for one category, and its usable column names."""
    frame = pd.read_csv(DATA_DIR / CATEGORY_FILES[category], index_col=0)
    keep = []
    for column in frame.columns:
        word_a, word_b = _parse_pair(column)
        if (category, word_a) in EXCLUDED_MEMBERS or (category, word_b) in EXCLUDED_MEMBERS:
            continue
        keep.append(column)
    return frame, keep


@functools.lru_cache(maxsize=1)
def load_human_similarity() -> pd.DataFrame:
    """
    One row per rated within-category pair.

    Columns: category, word_a, word_b (canonical order), mean_rating (1 to 7, higher is
    more similar, averaged over the subjects who rated that pair), n_raters.
    """
    rows = []
    for category in CATEGORY_FILES:
        frame, columns = _category_frame(category)
        for column in columns:
            word_a, word_b = _parse_pair(column)
            values = frame[column].dropna()
            rows.append({"category": category, "word_a": word_a, "word_b": word_b,
                         "mean_rating": float(values.mean()), "n_raters": int(len(values))})
    return pd.DataFrame(rows)


@functools.lru_cache(maxsize=1)
def human_pair_lookup() -> dict:
    """(word_a, word_b) in canonical order -> mean rating, for fast per-pair joins."""
    pairs = load_human_similarity()
    return {(a, b): r for a, b, r in zip(pairs.word_a, pairs.word_b, pairs.mean_rating)}


@functools.lru_cache(maxsize=8)
def human_noise_ceiling(n_splits: int = 200, seed: int = 0) -> dict:
    """
    Per-category split-half reliability, Spearman-Brown corrected.

    Subjects are split at random into halves, each half is averaged per pair, and the two
    pair vectors are correlated by Spearman over the pairs both halves cover. The mean over
    splits is then corrected from half-sample to full-sample reliability, which is the
    largest correlation any model could achieve against these ratings. Reporting a raw
    correlation without dividing by this understates the model, since part of the residual
    is disagreement between the humans themselves rather than model error.
    """
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
