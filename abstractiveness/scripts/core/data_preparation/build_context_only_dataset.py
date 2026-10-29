"""
Context-only copy of a stimulus dataset, every sentence carrying the character spans of the concept words to delete.

Positives lose their own concept's forms, negatives the target concept's forms and those of the concept
they were drawn from, with the forms of frequency_covariates.inflected_forms. selfcond deletes the tokens
the spans overlap before the forward pass, and the text baseline cuts them.

Usage, from scripts/:
    python -m core.data_preparation.build_context_only_dataset [source_dir] [output_dir]

Explanations: documentation/dataset_and_metadata.md, section 9.
"""

import argparse
import json
import pathlib
import re
import shutil

from core.data_preparation.frequency_covariates import ASSETS_DIR, inflected_forms

SOURCE_DIR = ASSETS_DIR / "Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot"
OUTPUT_DIR = ASSETS_DIR / "Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot_context_only"


def form_pattern(word: str) -> re.Pattern:
    """Case-insensitive pattern over a concept's inflected forms, not touching another letter."""
    forms = sorted(inflected_forms(word), key=len, reverse=True)
    return re.compile(r"(?<![^\W\d_])(?:" + "|".join(map(re.escape, forms)) + r")(?![^\W\d_])",
                      re.IGNORECASE)


def match_spans(text: str, patterns: list) -> list:
    """Sorted and merged [start, end] spans of every match of the patterns."""
    merged = []
    for start, end in sorted([m.start(), m.end()] for p in patterns for m in p.finditer(text)):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def context_only_concepts(concepts: dict) -> dict:
    """Each concept's data, keyed by storage key, with the delete_spans of its positives and negatives."""
    patterns = {key: form_pattern(data["word"]) for key, data in concepts.items()}
    sources = {}
    for key, data in concepts.items():
        for sentence in data["sentences"]["positive"]:
            sources.setdefault(sentence, []).append(key)
    out = {}
    for key, data in concepts.items():
        negatives = data["sentences"]["negative"]
        orphans = [s for s in negatives if s not in sources]
        if orphans:
            raise ValueError(f"{key}: {len(orphans)} negatives are no concept's positive, e.g. {orphans[0]!r}")
        out[key] = {**data, "delete_spans": {
            "positive": [match_spans(s, [patterns[key]]) for s in data["sentences"]["positive"]],
            "negative": [match_spans(s, [patterns[key]] + [patterns[c] for c in sources[s]]) for s in negatives],
        }}
    return out


def build(source_dir: pathlib.Path, output_dir: pathlib.Path) -> dict:
    """Writes the context-only copy of source_dir into output_dir and returns its concepts."""
    if output_dir.resolve() == source_dir.resolve():
        raise ValueError("the context-only copy must not overwrite its source dataset")
    concepts = {}
    for path in sorted(source_dir.glob("*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        concepts[data["concept"]] = data
    out = context_only_concepts(concepts)
    for key, data in out.items():
        path = output_dir / data["group"] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
    shutil.copyfile(source_dir / "concept_list.csv", output_dir / "concept_list.csv")
    return out


def main() -> None:
    """Builds the context-only dataset and prints how many sentences lost a word."""
    parser = argparse.ArgumentParser(description="Context-only copy of a stimulus dataset.")
    parser.add_argument("source_dir", type=pathlib.Path, nargs="?", default=SOURCE_DIR)
    parser.add_argument("output_dir", type=pathlib.Path, nargs="?", default=OUTPUT_DIR)
    args = parser.parse_args()
    out = build(args.source_dir, args.output_dir)
    for label in ("positive", "negative"):
        spans = [s for data in out.values() for s in data["delete_spans"][label]]
        print(f"{label}: {sum(map(bool, spans))} of {len(spans)} sentences lose a word")
    print(f"Wrote {len(out)} concepts to {args.output_dir}")


if __name__ == "__main__":
    main()
