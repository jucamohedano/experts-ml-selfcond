"""
Derives a family-resemblance typicality proxy for each word in metadata_Richie_HSJ.json
from the two Study 1 similarity datasets in Richie & Bhatia's "SpAM on words" materials:

  - study1_pairwise_data/data_individual_level/<Category>_pairwise.csv
    "Total-set" pairwise similarity ratings (1-7 scale, all C(n,2) pairs per category,
    one row per subject). No typicality is directly measured in these files.

  - study1_spam_data.csv
    Spatial Arrangement Method (SpAM) data: one row per subject per category, with the
    final (x, y) position each subject dropped every item at when arranging the category's
    words by similarity (closer = more similar). Unused stimulus slots (categories with
    fewer than 30 items) are padded with the sentinel item "....." at (3840, 2160), which
    are excluded here.

Neither study asked participants to rate typicality directly -- both only elicited pairwise
SIMILARITY judgments, and the published paper (Richie, White, Bhatia & Hout, 2020, "The spatial
arrangement method of measuring similarity can capture high-dimensional, semantic structures",
Proc. CogSci 42) never reduces this data to a single per-item value either: it only ever builds
the full aggregate (dis)similarity matrix (one subject-averaged value per pair) and feeds that
into cross-validated MDS to study recoverable dimensionality. Typicality here is therefore our
own derivation, using the standard "family resemblance" operationalization from categorization
research (Rosch & Mervis, 1975): an item's typicality within a category is proxied by its
average similarity to all other members of that category, i.e. the row/column mean of the same
aggregate matrix the paper itself constructs (see their Results section and Figure 1).

  - typicality_HSJ_pairwise: for each pair, the similarity rating (1-7) is first averaged over
    subjects (matching the paper's own aggregate-matrix construction, since the between-subjects
    design means different pairs have different numbers of raters for the four larger
    categories). A word's score is then the mean of its row/column in that matrix.
  - typicality_HSJ_spam: for each subject's trial, the Euclidean distance from that word's
    arranged position to every other item's position is computed, then averaged across subjects
    (equivalent to the row/column mean of the paper's aggregate SpAM distance matrix). Distance
    is inverted (closer = more typical).

Both raw scores are then min-max normalized within each category to a comparable [0, 1] scale
(1 = most similar to other category members on average = most typical; 0 = least).
"""
import glob
import json
import pathlib

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RICHIE_DIR = REPO_ROOT / "assets" / "Richie_and_Bhatia-HSJ"
PAIRWISE_DIR = RICHIE_DIR / "study1_pairwise_data" / "data_individual_level"
SPAM_FILE = RICHIE_DIR / "study1_spam_data.csv"
METADATA_FILE = REPO_ROOT / "assets" / "metadata_Richie_HSJ.json"


def minmax_normalize_by_category(raw_by_category):
    """raw_by_category: {category: {word: raw_score}} -> {category: {word: 0-1 score}}"""
    normalized = {}
    for cat, word_scores in raw_by_category.items():
        values = list(word_scores.values())
        lo, hi = min(values), max(values)
        span = hi - lo
        normalized[cat] = {
            w: (round((v - lo) / span, 3) if span > 0 else 0.5)
            for w, v in word_scores.items()
        }
    return normalized


def compute_pairwise_typicality():
    """Mean pairwise similarity rating (1-7) of each word to all other words in its
    category. Matches Richie, White, Bhatia & Hout (2020, CogSci) Sec. "Results and
    discussion": each pair's similarity is first averaged over subjects to build the
    aggregate similarity matrix (the same aggregation used for their Figure 1); a word's
    score is then the mean of its row/column in that matrix. This gives every pair equal
    weight, which matters because birds/clothing/professions/sports used a between-subjects
    half-sampled design, so different pairs have different numbers of raters."""
    raw_by_category = {}
    for f in sorted(glob.glob(str(PAIRWISE_DIR / "*_pairwise.csv"))):
        category = pathlib.Path(f).stem.replace("_pairwise", "").lower()
        df = pd.read_csv(f)
        pair_cols = df.columns[1:]  # first column is the (unlabeled) subject id

        # Aggregate similarity matrix: one subject-averaged value per pair
        pair_means = {}
        for col in pair_cols:
            a, b = (w.strip().lower() for w in col.split("\\"))
            pair_means[(a, b)] = float(df[col].dropna().mean())

        word_pair_means = {}
        for (a, b), val in pair_means.items():
            word_pair_means.setdefault(a, []).append(val)
            word_pair_means.setdefault(b, []).append(val)

        raw_by_category[category] = {w: float(np.mean(v)) for w, v in word_pair_means.items()}

    return minmax_normalize_by_category(raw_by_category)


def compute_spam_typicality():
    """Mean Euclidean distance (inverted) of each word's SpAM position to all other
    same-category items' positions, aggregated across subjects."""
    spam = pd.read_csv(SPAM_FILE)
    stim_cols = [c for c in spam.columns if c.startswith("Stim")]

    raw_distance_by_category = {}
    for category, group in spam.groupby("Concept"):
        word_distances = {}
        for _, row in group.iterrows():
            items = []
            for i, col in enumerate(stim_cols, start=1):
                word = str(row[col]).strip()
                if word == ".....":
                    continue
                items.append((word.lower(), row[f"Object{i}XFinal"], row[f"Object{i}YFinal"]))

            coords = np.array([(x, y) for _, x, y in items], dtype=float)
            for i, (word, _, _) in enumerate(items):
                other_coords = np.delete(coords, i, axis=0)
                dists = np.linalg.norm(other_coords - coords[i], axis=1)
                word_distances.setdefault(word, []).append(float(np.mean(dists)))

        mean_distance = {w: float(np.mean(v)) for w, v in word_distances.items()}
        # Invert: smaller mean distance (more central / more similar to peers) -> higher typicality
        raw_distance_by_category[category.lower()] = {w: -d for w, d in mean_distance.items()}

    return minmax_normalize_by_category(raw_distance_by_category)


def main():
    pairwise_typicality = compute_pairwise_typicality()
    spam_typicality = compute_spam_typicality()

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for entry in metadata:
        if entry["abstraction_level"] == 1:
            entry["typicality_HSJ_pairwise"] = None
            entry["typicality_HSJ_spam"] = None
            continue

        cat = entry["category"]
        word = entry["concept"]
        entry["typicality_HSJ_pairwise"] = pairwise_typicality.get(cat, {}).get(word, None)
        entry["typicality_HSJ_spam"] = spam_typicality.get(cat, {}).get(word, None)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    n_pairwise = sum(1 for e in metadata if e.get("typicality_HSJ_pairwise") is not None)
    n_spam = sum(1 for e in metadata if e.get("typicality_HSJ_spam") is not None)
    n_words = sum(1 for e in metadata if e["abstraction_level"] == 2)
    print(f"Updated {METADATA_FILE}")
    print(f"typicality_HSJ_pairwise: {n_pairwise}/{n_words} words")
    print(f"typicality_HSJ_spam: {n_spam}/{n_words} words")


if __name__ == "__main__":
    main()
