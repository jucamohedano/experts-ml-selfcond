"""
The concept token table records each word's mid-sentence tokens, and its loader degrades to an empty frame.

Checks:
  1. concept_token_rows encodes " " + word without special tokens and writes n_tokens,
     first_token_id and the decoded pieces joined by "|".
  2. load_concept_tokens returns an empty frame without an architecture or without the file.
  3. The stored table holds 206 concepts per tokenizer, and the known shared first tokens:
     tuxedo, trolley and tangerine on " t" and pheasant, pomegranate and penguin on " p" in
     Qwen3-1.7B-Base, carpenter on " car" and dresser on " dress" in GPT-2.

Usage, from scripts/ (after python -m core.data_preparation.build_concept_tokens):
    python tests/check_concept_tokens.py
"""
import pathlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.concept_tokens import load_concept_tokens
from core.data_preparation.build_concept_tokens import concept_token_rows
from _report import run


class Stub:
    """A tokenizer that encodes a text as its length and 1, and decodes an id to its digits."""

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return [len(text), 1]

    def decode(self, ids):
        return str(ids[0])


def checks(failures: list) -> str:
    stub = Stub()
    rows = concept_token_rows(pd.DataFrame({"concept": ["squash__sports"], "word": ["squash"]}), "gpt2", stub)
    assert stub.calls == [(" squash", False)], stub.calls
    assert rows.loc[0, "n_tokens"] == 2 and rows.loc[0, "first_token_id"] == 7 and rows.loc[0, "pieces"] == "7|1", rows
    assert rows.loc[0, "architecture"] == "gpt2" and rows.loc[0, "first_piece"] == "7", rows
    assert load_concept_tokens(None).empty
    assert load_concept_tokens("qwen3", pathlib.Path("/nonexistent.csv")).empty
    qwen, gpt2 = load_concept_tokens("qwen3"), load_concept_tokens("gpt2")
    assert len(qwen) == len(gpt2) == 206, "run python -m core.data_preparation.build_concept_tokens first"
    assert qwen.loc[["tuxedo", "trolley", "tangerine"], "first_token_id"].nunique() == 1
    assert qwen.loc["tuxedo", "first_piece"] == " t", qwen.loc["tuxedo"]
    assert qwen.loc[["pheasant", "pomegranate", "penguin"], "first_token_id"].nunique() == 1
    assert gpt2.loc["carpenter", "first_piece"] == " car" and gpt2.loc["dresser", "first_piece"] == " dress"
    return "stub encoding, loader fallbacks, 206 concepts per tokenizer, known shared first tokens"


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
