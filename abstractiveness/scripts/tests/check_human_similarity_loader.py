"""
The human pairwise similarity loader returns every rated pair, correctly keyed and scaled.

Checks:
  1. 2,418 within-category pairs with the expected count per category.
  2. Every word and concept key exists in the metadata, squash keyed per sense in both categories.
  3. Ratings on the 1 to 7 scale, 19 to 39 raters, pairs in canonical order.
  4. The lookup holds every pair and orders two anchor pairs correctly.
  5. Split-half noise ceilings exist for every category and fall in a plausible range.

Usage, from scripts/:
    python tests/check_human_similarity_loader.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.human_similarity_ratings import load_human_similarity, human_noise_ceiling, human_pair_lookup
from _report import run

METADATA = pathlib.Path(__file__).resolve().parents[3] / "abstractiveness/assets/metadata_Richie_HSJ.json"
EXPECTED_PAIRS = {"birds": 435, "clothing": 406, "fruit": 210, "furniture": 190,
                  "professions": 378, "sports": 378, "vegetables": 190, "vehicles": 231}


def checks(failures: list) -> str:
    pairs = load_human_similarity()
    meta = [r for r in json.load(METADATA.open()) if r.get("category")]
    known = {r.get("word", r["concept"]) for r in meta}
    known_keys = {r["concept"] for r in meta}

    assert len(pairs) == 2418, f"expected 2418 pairs, got {len(pairs)}"
    counts = pairs.groupby("category").size().to_dict()
    assert counts == EXPECTED_PAIRS, f"per-category counts wrong: {counts}"

    unknown = (set(pairs.word_a) | set(pairs.word_b)) - known
    assert not unknown, f"words absent from metadata: {sorted(unknown)}"
    for category, key in (("sports", "squash__sports"), ("vegetables", "squash__vegetables")):
        rows = pairs[pairs.category == category]
        assert "squash" in set(rows.word_a) | set(rows.word_b), f"squash missing from {category}"
        assert key in set(rows.concept_a) | set(rows.concept_b), f"{key} missing from {category}"
    assert "squash" not in known_keys, "squash must be keyed per sense, not bare"
    unknown_keys = (set(pairs.concept_a) | set(pairs.concept_b)) - known_keys
    assert not unknown_keys, f"concept keys absent from metadata: {sorted(unknown_keys)}"

    assert pairs.mean_rating.between(1, 7).all(), "ratings outside the 1 to 7 scale"
    assert pairs.n_raters.min() >= 19, f"min raters {pairs.n_raters.min()}, expected >= 19"
    assert pairs.n_raters.max() <= 39, f"max raters {pairs.n_raters.max()}, expected <= 39"
    assert (pairs.word_a < pairs.word_b).all(), "pairs not stored in canonical order"

    lookup = human_pair_lookup()
    assert len(lookup) == 2418, f"lookup has {len(lookup)} entries"
    assert lookup[("melon", "watermelon")] > 6.0, "melon/watermelon should be rated very similar"
    assert lookup[("banana", "olive")] < 2.0, "banana/olive should be rated very dissimilar"

    ceiling = human_noise_ceiling()
    assert set(ceiling) == set(EXPECTED_PAIRS), f"ceiling categories wrong: {sorted(ceiling)}"
    for category, value in ceiling.items():
        if not 0.75 < value < 0.99:
            failures.append(f"{category} ceiling {value:.3f} outside the plausible range")
    return (f"{len(pairs)} pairs, ceilings "
            + ", ".join(f"{c} {ceiling[c]:.3f}" for c in sorted(ceiling)))


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
