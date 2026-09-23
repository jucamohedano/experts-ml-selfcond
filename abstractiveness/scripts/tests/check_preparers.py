"""Structural checks on the frequency and typicality preparers. Exits non-zero on failure."""
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.preparers import frequency_preparer as freq
from utils.preparers.typicality_preparer import typicality_columns

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
METADATA = REPO_ROOT / "abstractiveness/assets/metadata_Richie_HSJ.json"

WIKIPEDIA_TOKENS = 2_474_589_909

# Concepts scored on the listed form alone, sampled across the reasons in SINGLE_FORM.
MUST_STAY_ALONE = {"shorts", "overalls", "jeans", "pants", "boots", "boxers",
                   "tennis", "boxing", "fencing", "running", "walking",
                   "asparagus", "broccoli", "corn", "furniture", "clothing"}

# Pairs that must be summed, including every irregular the suffix rules could get wrong.
MUST_MERGE = {
    "bus": {"bus", "buses"},
    "tomato": {"tomato", "tomatoes"},
    "potato": {"potato", "potatoes"},
    "fireman": {"fireman", "firemen"},
    "policeman": {"policeman", "policemen"},
    "postman": {"postman", "postmen"},
    "scarf": {"scarf", "scarves"},
    "cherry": {"cherry", "cherries"},
    "canary": {"canary", "canaries"},
    "turkey": {"turkey", "turkeys"},
    "ostrich": {"ostrich", "ostriches"},
    "dress": {"dress", "dresses"},
    "squash": {"squash", "squashes"},
    "piano": {"piano", "pianos"},
    "dove": {"dove", "doves"},
    "mirror": {"mirror", "mirrors"},
    "birds": {"bird", "birds"},
    "professions": {"profession", "professions"},
    "gloves": {"glove", "gloves"},
    "grapes": {"grape", "grapes"},
}

EXPECTED_SUBTLEX_US_GAPS = {"physiotherapist"}


def check_frequency(concepts) -> None:
    tokens = freq.wikipedia_corpus_tokens()
    assert tokens == WIKIPEDIA_TOKENS, f"corpus total {tokens:,}, expected {WIKIPEDIA_TOKENS:,}"
    offset = math.log10(1e9 / tokens)
    assert abs(offset - (-0.3935)) < 5e-5, f"Zipf offset {offset:.4f}, expected -0.3935"

    for word in sorted(MUST_STAY_ALONE):
        forms = freq.inflected_forms(word)
        assert forms == frozenset({word}), f"{word} merged into {sorted(forms)}"

    for word, expected in MUST_MERGE.items():
        forms = freq.inflected_forms(word)
        assert forms == frozenset(expected), \
            f"{word} gave {sorted(forms)}, expected {sorted(expected)}"

    # Every form that gets summed must actually occur in the corpus, which is what catches
    # a suffix rule inventing a spelling such as "asparaguses" or "pantses".
    counts = freq.load_wikipedia_counts()
    for word in concepts:
        for form in freq.inflected_forms(word):
            assert counts.get(form, 0) > 0, f"{word} sums {form!r}, absent from Wikipedia"

    columns = {word: freq.frequency_columns(word) for word in concepts}
    assert set(next(iter(columns.values()))) == {
        "frequency", "frequency_zipf_subtlex_us_lemma", "frequency_zipf_wikipedia_lemma"
    }, "frequency_columns changed shape"

    gaps = {w for w, c in columns.items() if c["frequency_zipf_subtlex_us_lemma"] is None}
    assert gaps == EXPECTED_SUBTLEX_US_GAPS, f"SUBTLEX-US gaps {sorted(gaps)}"

    for key in ("frequency", "frequency_zipf_wikipedia_lemma"):
        missing = [w for w, c in columns.items() if c[key] is None]
        assert not missing, f"{key} missing for {missing}"

    for word, column in columns.items():
        for key in ("frequency_zipf_subtlex_us_lemma", "frequency_zipf_wikipedia_lemma"):
            value = column[key]
            if value is not None:
                assert 0.5 < value < 7.5, f"{word} {key} = {value:.3f} is off the Zipf scale"

    # Summing can only add counts, so a lemma score never falls below its surface form.
    for word in concepts:
        for surface, lemma in ((freq.wikipedia_zipf, freq.wikipedia_lemma_zipf),
                               (freq.subtlex_us_zipf, freq.subtlex_us_lemma_zipf)):
            a, b = surface(word), lemma(word)
            if a is None or b is None:
                continue
            assert b >= a - 1e-9, f"{word}: lemma {b:.3f} below surface {a:.3f}"

    # The reference accessors stay usable even though they are not metadata columns.
    for word in concepts:
        assert freq.subtlex_uk_zipf(word) is not None, f"{word} missing from SUBTLEX-UK"

    merged = sum(1 for w in concepts if len(freq.inflected_forms(w)) > 1)
    print(f"OK  frequency: {len(concepts)} concepts, {merged} summed over two forms, "
          f"{len(concepts) - merged} single form, SUBTLEX-US coverage "
          f"{len(concepts) - len(gaps)}/{len(concepts)}")


def check_typicality(metadata) -> None:
    """The stored values were produced before the logic moved into utils, so reproducing
    them exactly is the regression test for the move."""
    compared = 0
    for entry in metadata:
        category = entry["category"] if entry["abstraction_level"] != 1 else None
        produced = typicality_columns(entry.get("word", entry["concept"]), category)
        for key, value in produced.items():
            assert value == entry[key], \
                f"{entry['concept']} {key}: stored {entry[key]}, produced {value}"
            compared += 1

    n_words = sum(1 for e in metadata if e["abstraction_level"] == 2)
    n_pairwise = sum(1 for e in metadata if e["typicality_HSJ_pairwise"] is not None)
    n_spam = sum(1 for e in metadata if e["typicality_HSJ_spam"] is not None)
    assert n_pairwise == n_words, f"pairwise covers {n_pairwise}/{n_words}"
    assert n_spam == n_words - 2, f"spam covers {n_spam}/{n_words}, expected {n_words - 2}"

    keys = [e["concept"] for e in metadata]
    assert len(keys) == len(set(keys)), "concept keys must be unique"
    assert {"squash__vegetables", "squash__sports"} <= set(keys), "both squash senses required"

    print(f"OK  typicality: {compared} values reproduce the stored metadata exactly, "
          f"pairwise {n_pairwise}/{n_words}, spam {n_spam}/{n_words}")


def main() -> int:
    metadata = json.load(METADATA.open())
    # Frequency is looked up by surface form, not by the per-sense concept key.
    check_frequency([entry.get("word", entry["concept"]) for entry in metadata])
    check_typicality(metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
