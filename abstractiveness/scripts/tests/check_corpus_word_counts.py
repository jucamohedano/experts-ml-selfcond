"""
The corpus word counter applies the tokenisation and the document threshold of the Wikipedia list.

Checks:
  1. Tokenisation: typographic dash and apostrophe normalised, lowercased, split on characters
     outside letters, digits, hyphen and apostrophe, words with a digit dropped, single letters kept,
     tokens starting or ending with a hyphen dropped.
  2. Document threshold: a word repeated inside one document counts one document, words in fewer
     than three documents are dropped, the returned total sums only the kept words, the file is
     written most frequent first with no temporary file left behind.
  3. Parquet units: row-group units cover every row group once, and counting them in batches, in
     one process or in two workers, reproduces the counts of the texts taken at once.

Usage, from scripts/:
    python tests/check_corpus_word_counts.py
"""
import collections
import pathlib
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.data_preparation import corpus_word_counts as cwc
from _report import run


def check_tokenisation() -> str:
    """Tokenisation follows gather_wordfreq.py."""
    text = "Non-profit chair’s 3D – a café, -chair- CHAIR\nsnake_case"
    words = cwc.document_words(text)
    expected = ["non-profit", "chair's", "a", "café", "chair", "snake_case"]
    assert words == expected, f"tokenised {words}, expected {expected}"
    return f"tokenisation: {len(expected)} words kept from a mixed sample"


def check_document_threshold() -> str:
    """The three-document rule counts documents, not occurrences."""
    uses, documents = collections.Counter(), collections.Counter()
    cwc.add_documents(["bed bed bed", "bed lamp", "lamp", "sofa sofa", "sofa", "sofa"], uses, documents)
    assert uses["bed"] == 4 and documents["bed"] == 2, f"bed {uses['bed']} uses in {documents['bed']} documents"
    assert uses["sofa"] == 4 and documents["sofa"] == 3, f"sofa {uses['sofa']} uses in {documents['sofa']} documents"
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "counts.txt"
        total = cwc.write_count_file(uses, documents, path)
        assert total == 4, f"kept total {total}, expected 4"
        assert path.read_text(encoding="utf-8") == "sofa 4\n", f"file reads {path.read_text()!r}"
        assert not list(pathlib.Path(tmp).glob("*.tmp")), "temporary file left behind"
    return "threshold: bed dropped at 4 uses in 2 documents, sofa kept at 3 documents"


def check_parquet_units() -> str:
    """Unit and batch boundaries do not change the counts."""
    texts = [f"chair table {i % 7} sofa" if i % 2 else "Lamp, lamp’s rug" for i in range(25)]
    expected_uses, expected_documents = collections.Counter(), collections.Counter()
    cwc.add_documents(texts, expected_uses, expected_documents)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "shard.parquet"
        pq.write_table(pa.table({"id": list(range(25)), "text": texts}), path, row_group_size=4)
        units = cwc.work_units([path], unit_documents=10)
        covered = [index for _, first, stop in units for index in range(first, stop)]
        assert covered == list(range(7)), f"units cover row groups {covered}"
        uses, documents, n_documents = collections.Counter(), collections.Counter(), 0
        for unit in units:
            unit_uses, unit_documents, unit_n = cwc.count_unit(unit, batch_size=3)
            uses.update(unit_uses)
            documents.update(unit_documents)
            n_documents += unit_n
        assert (uses, documents, n_documents) == (expected_uses, expected_documents, 25), "batched counts differ"
        parallel = cwc.count_corpus(units, workers=2)
        assert parallel == (expected_uses, expected_documents, 25), "parallel counts differ"
    return f"parquet: {len(units)} units over 7 row groups match the single pass"


def checks(failures: list) -> str:
    return ", ".join([check_tokenisation(), check_document_threshold(), check_parquet_units()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
