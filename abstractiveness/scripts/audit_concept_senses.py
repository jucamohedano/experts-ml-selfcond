#!/usr/bin/env python3
"""
Audit the generated stimulus sentences for word-sense errors.

A concept is filed under a category (bed under furniture, canary under birds), but the
sentences are written by an LM from a WordNet gloss, and WordNet sense ordering does not
know about the category. When the wrong sense is picked, the concept's 400 positive
sentences describe a different word entirely, so every expert unit found for it, and every
downstream analysis that treats it as a category member, is about the wrong meaning. That
failure is invisible in the pipeline: the file is complete, the counts are right, only the
meaning is wrong.

The audit is a leave-one-out text check, independent of WordNet and of the generation
script, so it catches wrong senses no matter how they got there:

1. Every concept is represented by the concatenation of its positive sentences, with its
   own name and plural stripped out, so the profile is built from context rather than from
   repetition of the word itself.
2. Documents are TF-IDF vectorised, and each category gets a profile equal to the mean
   vector of its members.
3. Each concept is scored by cosine similarity against all eight category profiles, with
   itself held out of its own category's profile so it cannot vouch for itself.

A concept whose assigned category is not the best match is describing something other than
what its category-mates describe. Some hits are benign, for instance cycling matches the
vehicles profile because bicycles are genuinely its subject matter, so the audit reports
the ranking and lets a human read the evidence rather than deciding alone.

The chosen WordNet gloss is reported next to each concept, which is what turns a flagged
row into an actionable one: a mismatch plus a visibly wrong gloss points straight at the
sense selection in scripts/generate_definitions_dspy.py, while a mismatch with a correct
gloss points at the generation prompt instead.

Output columns of concept_sense_audit.csv:

- concept, category: as recorded in the concept JSON file
- best_category: the category profile the sentences actually resemble most
- rank_assigned: rank of the assigned category among all eight, 1 is correct
- sim_assigned, sim_best: cosine similarities behind that ranking
- gap: sim_best minus sim_assigned, 0 when the assigned category wins
- flagged: True when rank_assigned is above 1 and the concept is not in --ignore
- wordnet_sense, wordnet_gloss: the sense the current generator would choose today

Usage:

    python scripts/audit_concept_senses.py
    python scripts/audit_concept_senses.py --ignore skateboard,sailing,walking,cycling

Exit code is 1 when any concept is flagged, so the audit can gate a regeneration run.
"""

import argparse
import importlib.util
import json
import pathlib
import re
import sys
import typing as t

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "abstractiveness/assets/Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot/custom"
)
DEFAULT_OUT = REPO_ROOT / "abstractiveness/results/concept_sense_audit.csv"
GENERATOR_SCRIPT = REPO_ROOT / "scripts/generate_definitions_dspy.py"

# Concepts whose sentences legitimately resemble a neighbouring category, verified by
# reading them. Passed as the default for --ignore so a clean dataset exits 0.
BENIGN_NEIGHBOURS = ("skateboard", "sailing", "walking", "cycling")


def load_generator_module():
    """Imports the generation script by path, so the audit reports the sense that the
    generator would choose right now. Returns None when the import fails (for example when
    the WordNet corpus is not downloaded), in which case the gloss columns are left empty
    and the text-based audit still runs."""
    try:
        spec = importlib.util.spec_from_file_location("generate_definitions_dspy", GENERATOR_SCRIPT)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ensure_nltk_wordnet()
        return module
    except Exception as exc:
        print(f"Warning: could not load {GENERATOR_SCRIPT.name} ({exc}). "
              f"Gloss columns will be empty.")
        return None


def load_concepts(dataset_dir: pathlib.Path) -> t.List[dict]:
    """Reads every concept JSON in the dataset directory. Files without a category are
    level 1 category labels, which have no sibling set to compare against and so take part
    only in the gloss report."""
    records = []
    for path in sorted(dataset_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        positives = data.get("sentences", {}).get("positive", [])
        if not positives:
            print(f"Warning: {path.name} has no positive sentences, skipping.")
            continue
        records.append({
            "storage_key": path.stem,
            "concept": data["concept"],
            "category": data.get("category"),
            "positives": positives,
        })
    return records


def strip_concept_name(text: str, concept: str) -> str:
    """Removes the concept name and its plural from its own sentences. Without this the
    profile is dominated by the word itself, which is shared by both senses and therefore
    carries no information about which one is being described."""
    pattern = re.compile(r"\b" + re.escape(concept) + r"s?\b", re.IGNORECASE)
    return pattern.sub(" ", text)


def score_categories(records: t.List[dict]) -> pd.DataFrame:
    """Leave-one-out cosine similarity of each concept against every category profile."""
    members = [r for r in records if r["category"]]
    if not members:
        raise RuntimeError("No categorized concepts found, nothing to audit.")

    docs = [strip_concept_name(" ".join(r["positives"]), r["concept"]) for r in members]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs).toarray()

    categories = sorted({r["category"] for r in members})
    rows = []
    for i, record in enumerate(members):
        sims = {}
        for category in categories:
            others = [
                j for j, other in enumerate(members)
                if other["category"] == category and j != i
            ]
            if not others:
                continue
            profile = matrix[others].mean(axis=0, keepdims=True)
            sims[category] = float(cosine_similarity(matrix[i:i + 1], profile)[0, 0])

        ordered = sorted(sims, key=sims.get, reverse=True)
        assigned = record["category"]
        rows.append({
            "concept": record["concept"],
            "storage_key": record["storage_key"],
            "category": assigned,
            "best_category": ordered[0],
            "rank_assigned": ordered.index(assigned) + 1,
            "sim_assigned": round(sims[assigned], 4),
            "sim_best": round(sims[ordered[0]], 4),
            "gap": round(sims[ordered[0]] - sims[assigned], 4),
        })
    return pd.DataFrame(rows)


def add_gloss_columns(df: pd.DataFrame, records: t.List[dict], generator) -> pd.DataFrame:
    """Appends the sense the generator would choose today for each (concept, category)."""
    category_of = {r["storage_key"]: r["category"] for r in records}
    senses, glosses = [], []
    for _, row in df.iterrows():
        if generator is None:
            senses.append("")
            glosses.append("")
            continue
        category = category_of.get(row["storage_key"])
        synset = generator.select_wordnet_sense(row["concept"], category)
        senses.append(synset.name() if synset else "")
        glosses.append(synset.definition() if synset else "")
    df["wordnet_sense"] = senses
    df["wordnet_gloss"] = glosses
    return df


def report_level1_labels(records: t.List[dict], generator) -> None:
    """Level 1 category labels have no siblings, so they are reported rather than scored.
    They are worth printing because they fail the same way: with no category of their own,
    nothing constrains sense selection, which is how "sports" came to be generated as the
    Maine colloquial sense meaning a summer resident."""
    labels = [r for r in records if not r["category"]]
    if not labels:
        return
    print("\nLevel 1 category labels (not scored, no sibling set):")
    for record in labels:
        gloss = ""
        if generator is not None:
            synset = generator.select_wordnet_sense(record["concept"], None)
            gloss = f"{synset.name()}: {synset.definition()}" if synset else "no sense"
        print(f"  {record['concept']:<14} {gloss[:90]}")
        print(f"      first sentence: {record['positives'][0][:90]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=DEFAULT_DATASET_DIR,
                        help="Directory of concept JSON files to audit.")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT,
                        help="Where to write concept_sense_audit.csv.")
    parser.add_argument("--ignore", type=str, default=",".join(BENIGN_NEIGHBOURS),
                        help="Comma-separated concepts that may rank below first without "
                             "being flagged, for genuine semantic neighbours.")
    parser.add_argument("--top", type=int, default=20,
                        help="How many worst-ranked concepts to print.")
    args = parser.parse_args()

    ignored = {c.strip() for c in args.ignore.split(",") if c.strip()}

    records = load_concepts(args.dataset_dir)
    print(f"Loaded {len(records)} concept files from {args.dataset_dir}")

    generator = load_generator_module()
    df = score_categories(records)
    df = add_gloss_columns(df, records, generator)
    df["flagged"] = (df["rank_assigned"] > 1) & (~df["concept"].isin(ignored))
    df = df.sort_values(["rank_assigned", "gap"], ascending=[False, False])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

    suspects = df[df["rank_assigned"] > 1]
    print(f"\n{len(suspects)} concept(s) rank their assigned category below first "
          f"({int(df['flagged'].sum())} flagged after --ignore):")
    columns = ["concept", "category", "best_category", "rank_assigned", "gap",
               "flagged", "wordnet_sense"]
    print(suspects.head(args.top)[columns].to_string(index=False))

    report_level1_labels(records, generator)

    if df["flagged"].any():
        print("\nAudit failed: the concepts above describe something other than their "
              "category-mates. Read their sentences before regenerating.")
        return 1
    print("\nAudit passed: every concept resembles its own category most.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
