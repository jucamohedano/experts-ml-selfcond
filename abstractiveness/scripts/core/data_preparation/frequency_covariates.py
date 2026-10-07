"""
Word frequency covariates on the Zipf scale, from Wikipedia and the two SUBTLEX corpora.

Read by build_concept_metadata through frequency_columns(). The unsummed and British reference
values (wikipedia_zipf, surface_zipf, subtlex_us_zipf, subtlex_uk_zipf) stay public for auditing.
The OpenWebText and FineWeb count files are written by corpus_word_counts.

Explanations: documentation/dataset_and_metadata.md, section 6.
"""

import csv
import functools
import math
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
ASSETS_DIR = REPO_ROOT / "abstractiveness/assets"
WIKIPEDIA_FILE = ASSETS_DIR / "enwiki-2023-04-13.txt"
OPENWEBTEXT_FILE = ASSETS_DIR / "openwebtext-word-counts.txt"
FINEWEB_FILE = ASSETS_DIR / "fineweb-sample-10BT-word-counts.txt"
SUBTLEX_US_FILE = ASSETS_DIR / "SUBTLEX-US.txt"
SUBTLEX_UK_FILE = ASSETS_DIR / "SUBTLEX-UK.txt"
SUBTLEX_ENCODING = "latin-1"
PER_MILLION_TO_BILLION = 3
VOWELS = "aeiou"
IRREGULAR_PLURALS = {
    "fireman": "firemen",
    "policeman": "policemen",
    "postman": "postmen",
    "tomato": "tomatoes",
    "potato": "potatoes",
}
PLURAL_FORM_LISTED = {
    "birds": "bird",
    "professions": "profession",
    "sports": "sport",
    "vegetables": "vegetable",
    "vehicles": "vehicle",
    "beans": "bean",
    "gloves": "glove",
    "grapes": "grape",
    "mittens": "mitten",
    "pajamas": "pajama",
    "panties": "panty",
    "sneakers": "sneaker",
    "socks": "sock",
}
SINGLE_FORM = {
    "furniture", "clothing",
    "asparagus", "broccoli", "cauliflower", "celery", "corn", "lettuce", "spinach",
    "archery", "badminton", "ballet", "baseball", "basketball", "billiards", "boxing",
    "chess", "cycling", "fencing", "fishing", "golfing", "gymnastics", "handball",
    "hockey", "judo", "rowing", "rugby", "running", "sailing", "skiing", "soccer",
    "surfing", "swimming", "tennis", "volleyball", "walking",
    "boots", "boxers", "jeans", "overalls", "pants", "shorts",
}


def plural_of(word: str) -> str:
    """Noun plural of a singular surface form."""
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2] not in VOWELS:
        return word[:-1] + "ies"
    if word.endswith("fe"):
        return word[:-2] + "ves"
    if word.endswith("f"):
        return word[:-1] + "ves"
    return word + "s"


def inflected_forms(word: str) -> frozenset:
    """The surface forms whose counts are summed for this concept."""
    word = word.lower()
    if word in SINGLE_FORM:
        return frozenset({word})
    if word in PLURAL_FORM_LISTED:
        return frozenset({word, PLURAL_FORM_LISTED[word]})
    return frozenset({word, plural_of(word)})


def zipf_from_count(count: int, corpus_tokens: int) -> float:
    """Raw occurrence count to Zipf, given the size of the corpus it was counted in."""
    return math.log10(count / corpus_tokens * 1e9)


def zipf_from_per_million(per_million: float) -> float:
    """A rate already expressed per million words to Zipf."""
    return math.log10(per_million) + PER_MILLION_TO_BILLION


@functools.lru_cache(maxsize=None)
def load_counts(path: pathlib.Path) -> dict:
    """Lowercased surface form to raw token count, from a 'word count' file."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing, see documentation/dataset_and_metadata.md section 6")
    counts = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().rsplit(None, 1)
            if len(parts) != 2:
                continue
            word, count = parts
            try:
                counts[word.lower()] = int(count)
            except ValueError:
                continue
    return counts


@functools.lru_cache(maxsize=None)
def corpus_tokens(path: pathlib.Path) -> int:
    """Total tokens in a count file."""
    return sum(load_counts(path).values())


def load_wikipedia_counts() -> dict:
    """Lowercased surface form to raw Wikipedia token count."""
    return load_counts(WIKIPEDIA_FILE)


def wikipedia_corpus_tokens() -> int:
    """Total tokens in the Wikipedia count file."""
    return corpus_tokens(WIKIPEDIA_FILE)


@functools.lru_cache(maxsize=1)
def load_subtlex_us() -> dict:
    """Lowercased surface form to its rate per million words, from the SUBTLWF column."""
    rates = {}
    with open(SUBTLEX_US_FILE, "r", encoding=SUBTLEX_ENCODING) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                rate = float(row["SUBTLWF"])
            except (KeyError, ValueError):
                continue
            if rate > 0:
                rates[row["Word"].lower()] = rate
    return rates


@functools.lru_cache(maxsize=1)
def load_subtlex_uk() -> dict:
    """Lowercased surface form to its Zipf, from the LogFreq(Zipf) column."""
    zipfs = {}
    with open(SUBTLEX_UK_FILE, "r", encoding=SUBTLEX_ENCODING) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            spelling = row["Spelling"].lower()
            if spelling in zipfs:
                continue
            try:
                zipfs[spelling] = float(row["LogFreq(Zipf)"])
            except (KeyError, ValueError):
                zipfs[spelling] = None
    return zipfs


def surface_zipf(path: pathlib.Path, word: str):
    """Zipf of the listed surface form in a count file."""
    count = load_counts(path).get(word.lower())
    return None if not count else zipf_from_count(count, corpus_tokens(path))


def lemma_zipf(path: pathlib.Path, word: str):
    """Zipf of a count file's counts summed over inflected_forms()."""
    counts = load_counts(path)
    total = sum(counts.get(form, 0) for form in inflected_forms(word))
    return None if not total else zipf_from_count(total, corpus_tokens(path))


def wikipedia_zipf(word: str):
    """Zipf of the listed surface form in the Wikipedia dump."""
    return surface_zipf(WIKIPEDIA_FILE, word)


def wikipedia_lemma_zipf(word: str):
    """Zipf of the Wikipedia counts summed over inflected_forms()."""
    return lemma_zipf(WIKIPEDIA_FILE, word)


def openwebtext_lemma_zipf(word: str):
    """Zipf of the OpenWebText counts summed over inflected_forms(), the GPT-2 exposure proxy."""
    return lemma_zipf(OPENWEBTEXT_FILE, word)


def fineweb_lemma_zipf(word: str):
    """Zipf of the FineWeb sample counts summed over inflected_forms(), the Qwen3 exposure proxy."""
    return lemma_zipf(FINEWEB_FILE, word)


def subtlex_us_zipf(word: str):
    """Zipf of the listed surface form in SUBTLEX-US."""
    rate = load_subtlex_us().get(word.lower())
    return None if rate is None else zipf_from_per_million(rate)


def subtlex_us_lemma_zipf(word: str):
    """Zipf of the SUBTLEX-US rates summed over inflected_forms()."""
    rates = load_subtlex_us()
    total = sum(rates.get(form, 0.0) for form in inflected_forms(word))
    return None if total <= 0 else zipf_from_per_million(total)


def subtlex_uk_zipf(word: str):
    """Zipf of the listed surface form in SUBTLEX-UK."""
    return load_subtlex_uk().get(word.lower())


def wikipedia_count(word: str):
    """Raw Wikipedia token count of the listed surface form."""
    return load_wikipedia_counts().get(word.lower())


def frequency_columns(word: str) -> dict:
    """The frequency fields for one concept."""
    return {
        "frequency": wikipedia_count(word),
        "frequency_zipf_subtlex_us_lemma": subtlex_us_lemma_zipf(word),
        "frequency_zipf_wikipedia_lemma": wikipedia_lemma_zipf(word),
        "frequency_zipf_openwebtext_lemma": openwebtext_lemma_zipf(word),
        "frequency_zipf_fineweb_lemma": fineweb_lemma_zipf(word),
    }
