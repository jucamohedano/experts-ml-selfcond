"""
Counts the words of a Hugging Face parquet corpus under the rules of the Wikipedia frequency list.

Writes the count files that frequency_covariates reads for the training-exposure columns.

Usage, from scripts/:
    python -m core.data_preparation.corpus_word_counts openwebtext
    python -m core.data_preparation.corpus_word_counts fineweb

Explanations: documentation/dataset_and_metadata.md, section 6.
"""

import argparse
import collections
import concurrent.futures
import datetime
import json
import os
import pathlib
import re

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
ASSETS_DIR = REPO_ROOT / "abstractiveness/assets"
MIN_DOCUMENTS = 3
UNIT_DOCUMENTS = 100_000
BATCH_SIZE = 10_000
NORMALISE = str.maketrans("–’", "-'")
SPLIT = re.compile(r"[^\w\-']")
IS_WORD = re.compile(r"(^\w.*\w$)|(^\w$)")
HAS_DIGIT = re.compile(r".*\d.*")
CORPUS_CACHE_DIR = ASSETS_DIR / "corpora"
DEFAULT_WORKERS = max(1, min(10, (os.cpu_count() or 2) - 2))
CORPORA = {
    "openwebtext": {
        "repo_id": "Skylion007/openwebtext",
        "pattern": "plain_text/*.parquet",
        "output": "openwebtext-word-counts.txt",
    },
    "fineweb": {
        "repo_id": "HuggingFaceFW/fineweb",
        "pattern": "sample/10BT/*.parquet",
        "output": "fineweb-sample-10BT-word-counts.txt",
    },
}


def document_words(text: str) -> list:
    """The words of one document, normalised, split and filtered as gather_wordfreq.py does."""
    lowered = text.translate(NORMALISE).lower()
    return [w for w in SPLIT.split(lowered) if w and IS_WORD.match(w) and not HAS_DIGIT.match(w)]


def add_documents(texts, uses: collections.Counter, documents: collections.Counter) -> None:
    """Adds the token counts and the document counts of each text to the two counters."""
    for text in texts:
        words = document_words(text)
        uses.update(words)
        documents.update(set(words))


def work_units(paths, unit_documents: int = UNIT_DOCUMENTS) -> list:
    """(path, first row group, stop row group) ranges of about unit_documents documents each."""
    units = []
    for path in paths:
        metadata = pq.ParquetFile(path).metadata
        first, rows = 0, 0
        for index in range(metadata.num_row_groups):
            rows += metadata.row_group(index).num_rows
            if rows >= unit_documents or index == metadata.num_row_groups - 1:
                units.append((str(path), first, index + 1))
                first, rows = index + 1, 0
    return units


def count_unit(unit, batch_size: int = BATCH_SIZE) -> tuple:
    """Token counts, document counts and document total of one range of row groups."""
    path, first, stop = unit
    uses, documents, n_documents = collections.Counter(), collections.Counter(), 0
    batches = pq.ParquetFile(path).iter_batches(
        batch_size=batch_size, row_groups=list(range(first, stop)), columns=["text"])
    for batch in batches:
        texts = batch.column("text").to_pylist()
        add_documents(texts, uses, documents)
        n_documents += len(texts)
    return uses, documents, n_documents


def count_corpus(units, workers: int) -> tuple:
    """Token counts, document counts and document total over every unit, counted in parallel."""
    uses, documents, n_documents = collections.Counter(), collections.Counter(), 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for unit_uses, unit_documents, unit_n in tqdm(pool.map(count_unit, units), total=len(units), desc="units"):
            uses.update(unit_uses)
            documents.update(unit_documents)
            n_documents += unit_n
    return uses, documents, n_documents


def write_count_file(uses, documents, path: pathlib.Path, min_documents: int = MIN_DOCUMENTS) -> int:
    """Writes 'word count' lines, most frequent first, for words in min_documents documents or more."""
    kept = sorted(((w, c) for w, c in uses.items() if documents[w] >= min_documents),
                  key=lambda item: item[1], reverse=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for word, count in kept:
            handle.write(f"{word} {count}\n")
    os.replace(temporary, path)
    return sum(count for _, count in kept)


def download_corpus(name: str) -> tuple:
    """The pinned revision and the local parquet paths of one registered corpus, downloaded once."""
    spec = CORPORA[name]
    revision = HfApi().dataset_info(spec["repo_id"]).sha
    root = snapshot_download(repo_id=spec["repo_id"], repo_type="dataset", revision=revision,
                             allow_patterns=[spec["pattern"]], local_dir=CORPUS_CACHE_DIR / name)
    return revision, sorted(pathlib.Path(root).glob(spec["pattern"]))


def main():
    parser = argparse.ArgumentParser(description="Count the words of a training-exposure corpus.")
    parser.add_argument("corpus", choices=sorted(CORPORA))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    spec = CORPORA[args.corpus]
    revision, paths = download_corpus(args.corpus)
    units = work_units(paths)
    print(f"{args.corpus}: {len(paths)} files, {len(units)} units, {args.workers} workers, revision {revision}")
    uses, documents, n_documents = count_corpus(units, args.workers)

    output = ASSETS_DIR / spec["output"]
    tokens_kept = write_count_file(uses, documents, output)
    provenance = {
        "corpus": args.corpus,
        "repo_id": spec["repo_id"],
        "revision": revision,
        "files": len(paths),
        "documents": n_documents,
        "tokens": sum(uses.values()),
        "tokens_kept": tokens_kept,
        "types_kept": sum(1 for word in uses if documents[word] >= MIN_DOCUMENTS),
        "min_documents": MIN_DOCUMENTS,
        "counted_on": datetime.date.today().isoformat(),
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=4) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=4))


if __name__ == "__main__":
    main()
