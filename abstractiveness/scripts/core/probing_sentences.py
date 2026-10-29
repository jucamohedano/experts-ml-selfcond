"""
The positive probing sentences of each concept, joined into one document per concept.

Used by module 2's text baseline (section 2.3).

Explanations: documentation/module_2_similarities.md.
"""

import functools
import json
import pathlib


def _cut(text: str, spans: list) -> str:
    """The text with its deleted spans removed."""
    for start, end in reversed(spans):
        text = text[:start] + text[end:]
    return text


@functools.lru_cache(maxsize=2)
def load_positive_documents(dataset_dir: str, concepts: tuple) -> tuple:
    """Each concept's positive sentences, deleted spans cut, as one string in the order of concepts, empty when its file is missing."""
    root = pathlib.Path(dataset_dir) / "custom"
    documents = []
    for concept in concepts:
        path = root / f"{concept}.json"
        if not path.exists():
            documents.append("")
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        positives = data["sentences"]["positive"]
        spans = data.get("delete_spans", {}).get("positive", [[]] * len(positives))
        documents.append(" ".join(_cut(s, sp) for s, sp in zip(positives, spans)))
    return tuple(documents)
