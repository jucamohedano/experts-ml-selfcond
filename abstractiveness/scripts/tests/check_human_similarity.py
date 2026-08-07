"""Structural checks on the human pairwise similarity loader. Exits non-zero on failure."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.human_similarity import load_human_similarity, human_noise_ceiling, human_pair_lookup

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
METADATA = REPO_ROOT / "abstractiveness/assets/metadata_Richie_HSJ.json"

EXPECTED_PAIRS = {"birds": 435, "clothing": 406, "fruit": 210, "furniture": 190,
                  "professions": 378, "sports": 351, "vegetables": 190, "vehicles": 231}


def main() -> int:
    pairs = load_human_similarity()
    meta = [r for r in json.load(METADATA.open()) if r.get("category")]
    known = {r["concept"] for r in meta}

    assert len(pairs) == 2391, f"expected 2391 pairs, got {len(pairs)}"
    counts = pairs.groupby("category").size().to_dict()
    assert counts == EXPECTED_PAIRS, f"per-category counts wrong: {counts}"

    unknown = (set(pairs.word_a) | set(pairs.word_b)) - known
    assert not unknown, f"words absent from metadata: {sorted(unknown)}"

    sports = pairs[pairs.category == "sports"]
    assert "squash" not in set(sports.word_a) | set(sports.word_b), "squash leaked into sports"
    vegetables = pairs[pairs.category == "vegetables"]
    assert "squash" in set(vegetables.word_a) | set(vegetables.word_b), \
        "squash missing from vegetables"

    assert pairs.mean_rating.between(1, 7).all(), "ratings outside the 1 to 7 scale"
    assert pairs.n_raters.min() >= 19, f"min raters {pairs.n_raters.min()}, expected >= 19"
    assert pairs.n_raters.max() <= 39, f"max raters {pairs.n_raters.max()}, expected <= 39"
    assert (pairs.word_a < pairs.word_b).all(), "pairs not stored in canonical order"

    lookup = human_pair_lookup()
    assert len(lookup) == 2391, f"lookup has {len(lookup)} entries"
    assert lookup[("melon", "watermelon")] > 6.0, "melon/watermelon should be rated very similar"
    assert lookup[("banana", "olive")] < 2.0, "banana/olive should be rated very dissimilar"

    ceiling = human_noise_ceiling()
    assert set(ceiling) == set(EXPECTED_PAIRS), f"ceiling categories wrong: {sorted(ceiling)}"
    for category, value in ceiling.items():
        assert 0.75 < value < 0.99, f"{category} ceiling {value:.3f} outside the plausible range"

    print(f"OK  {len(pairs)} pairs, {len(ceiling)} categories")
    for category in sorted(ceiling):
        sub = pairs[pairs.category == category]
        print(f"    {category:<12} pairs={len(sub):>4}  raters {sub.n_raters.min():>3}-{sub.n_raters.max():<3}"
              f"  ceiling={ceiling[category]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
