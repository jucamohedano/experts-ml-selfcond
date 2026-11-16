"""
Each concept word's tokens in its mid-sentence form, read from assets/concept_tokens.csv.

Used by module 5's Study C link graphs and Study D same-token analysis.

Explanations: documentation/dataset_and_metadata.md.
"""

import logging
import pathlib

import pandas as pd

log = logging.getLogger(__name__)

TOKENS_PATH = pathlib.Path(__file__).resolve().parents[2] / "assets" / "concept_tokens.csv"
TOKEN_COLUMNS = ["n_tokens", "first_token_id", "first_piece"]


def load_concept_tokens(architecture: str | None, path: pathlib.Path = TOKENS_PATH) -> pd.DataFrame:
    """The architecture's token rows indexed by concept, empty with one warning when unavailable."""
    empty = pd.DataFrame(columns=TOKEN_COLUMNS, index=pd.Index([], name="concept"))
    if architecture is None:
        log.warning("No architecture given, concept tokens skipped")
        return empty
    if not path.exists():
        log.warning(f"{path} is missing, concept tokens skipped (python -m core.data_preparation.build_concept_tokens)")
        return empty
    table = pd.read_csv(path, keep_default_na=False)
    table = table[table["architecture"] == architecture]
    if table.empty:
        log.warning(f"{path} holds no rows for architecture '{architecture}', concept tokens skipped")
        return empty
    return table.set_index("concept")[TOKEN_COLUMNS]
