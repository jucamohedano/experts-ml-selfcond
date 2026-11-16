"""
The token table of every concept word, as each model's tokenizer splits it inside a sentence.

Each word is encoded as " " + word without special tokens, the form it takes after the first
word of a sentence. Read through core/concept_tokens.py by module 5.

Usage, from scripts/ (tokenizers read offline, HF_HOME set and HF_HUB_OFFLINE=1):
    python -m core.data_preparation.build_concept_tokens

Explanations: documentation/dataset_and_metadata.md.
"""

import pandas as pd

from core.concept_tokens import TOKENS_PATH
from core.data_preparation.frequency_covariates import ASSETS_DIR

TOKENIZERS = {"gpt2": "gpt2", "qwen3": "Qwen/Qwen3-1.7B-Base"}
METADATA_PATH = ASSETS_DIR / "metadata_Richie_HSJ.json"


def concept_token_rows(metadata: pd.DataFrame, architecture: str, tokenizer) -> pd.DataFrame:
    """One row per concept with its token count, first token and decoded pieces."""
    rows = []
    for concept, word in zip(metadata["concept"], metadata["word"]):
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        pieces = [tokenizer.decode([token]) for token in ids]
        rows.append({"concept": concept, "word": word, "architecture": architecture,
                     "n_tokens": len(ids), "first_token_id": ids[0], "first_piece": pieces[0],
                     "pieces": "|".join(pieces)})
    return pd.DataFrame(rows)


def main() -> None:
    """Write the token table of both architectures."""
    from transformers import AutoTokenizer
    metadata = pd.read_json(METADATA_PATH).dropna(subset=["concept"])
    tables = [concept_token_rows(metadata, architecture, AutoTokenizer.from_pretrained(name, local_files_only=True))
              for architecture, name in TOKENIZERS.items()]
    table = pd.concat(tables, ignore_index=True).sort_values(["architecture", "concept"])
    table.to_csv(TOKENS_PATH, index=False)
    print(f"wrote {TOKENS_PATH}: {len(table)} rows")


if __name__ == "__main__":
    main()
