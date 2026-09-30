"""
The data-preparation scripts reproduce the stored metadata and follow their documented rules.

Checks:
  1. Frequency: Wikipedia token total and Zipf offset, forms kept alone or summed as listed,
     every summed form attested, column shape, SUBTLEX-US gaps, values on the Zipf scale,
     a lemma never below its surface form, SUBTLEX-UK coverage.
  2. Typicality: every stored value reproduced exactly, pairwise and SpAM coverage, unique keys
     with both squash senses.

Usage, from scripts/:
    python tests/check_data_preparation.py
"""
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.data_preparation import frequency_covariates as freq
from core.data_preparation.typicality_covariates import typicality_columns
from _report import run

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
METADATA = REPO_ROOT / "abstractiveness/assets/metadata_Richie_HSJ.json"

WIKIPEDIA_TOKENS = 2_474_589_909

MUST_STAY_ALONE = {"shorts", "overalls", "jeans", "pants", "boots", "boxers",
                   "tennis", "boxing", "fencing", "running", "walking",
                   "asparagus", "broccoli", "corn", "furniture", "clothing"}

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


def check_frequency(concepts) -> str:
    """Frequency covariates follow the summing rules and the Zipf scale."""
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

    for word in concepts:
        for surface, lemma in ((freq.wikipedia_zipf, freq.wikipedia_lemma_zipf),
                               (freq.subtlex_us_zipf, freq.subtlex_us_lemma_zipf)):
            a, b = surface(word), lemma(word)
            if a is None or b is None:
                continue
            assert b >= a - 1e-9, f"{word}: lemma {b:.3f} below surface {a:.3f}"

    for word in concepts:
        assert freq.subtlex_uk_zipf(word) is not None, f"{word} missing from SUBTLEX-UK"

    merged = sum(1 for w in concepts if len(freq.inflected_forms(w)) > 1)
    return (f"frequency: {len(concepts)} concepts, {merged} summed over two forms, "
          f"{len(concepts) - merged} single form, SUBTLEX-US coverage "
          f"{len(concepts) - len(gaps)}/{len(concepts)}")


def check_typicality(metadata) -> str:
    """Typicality covariates reproduce every stored metadata value exactly."""
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

    return (f"typicality: {compared} values reproduce the stored metadata exactly, "
          f"pairwise {n_pairwise}/{n_words}, spam {n_spam}/{n_words}")


def checks(failures: list) -> str:
    metadata = json.load(METADATA.open())
    frequency = check_frequency([entry.get("word", entry["concept"]) for entry in metadata])
    return f"{frequency}, {check_typicality(metadata)}"


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
