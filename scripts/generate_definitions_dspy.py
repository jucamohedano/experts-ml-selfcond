#!/usr/bin/env python3
"""
Generate concept datasets using DSPy framework.
Replacement for generate_assets_openai.py using DSPy's declarative approach.
"""

import argparse
import asyncio
import dataclasses
import json
import os
import pathlib
import random
import re
import typing as t

import dspy
from tqdm import tqdm

try:
    from nltk.corpus import wordnet as wn
    import nltk
except Exception:
    wn = None
    nltk = None

from nltk.corpus.reader.wordnet import Synset

@dataclasses.dataclass
class GenerationConfig:
    base_url: str
    model: str
    dataset_name: str
    concepts: t.List[str]
    positive_fact: int = 200
    positive_story: int = 200
    negatives_per_concept: int = 1000
    batch_size: int = 10
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 800
    seed: int = 1234
    api_key: t.Optional[str] = None


def load_config(path: pathlib.Path) -> GenerationConfig:
    with path.open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    return GenerationConfig(
        base_url=cfg.get("base_url", ""),
        model=cfg["model"],
        dataset_name=cfg["dataset_name"],
        concepts=list(cfg["concepts"]),
        positive_fact=int(cfg.get("positive_fact", 200)),
        positive_story=int(cfg.get("positive_story", 200)),
        negatives_per_concept=int(cfg.get("negatives_per_concept", 1000)),
        batch_size=int(cfg.get("batch_size", 10)),
        temperature=float(cfg.get("temperature", 0.9)),
        top_p=float(cfg.get("top_p", 0.95)),
        max_tokens=int(cfg.get("max_tokens", 800)),
        seed=int(cfg.get("seed", 1234)),
    )


def ensure_nltk_wordnet() -> None:
    global wn, nltk
    if wn is None or nltk is None:
        import nltk as _nltk
        nltk = _nltk
        from nltk.corpus import wordnet as _wn
        wn = _wn
    try:
        _ = wn.synsets("test")
    except LookupError:
        assert nltk is not None
        nltk.download("wordnet")
        try:
            nltk.download("omw-1.4")
        except Exception:
            pass


# Define the high-level categories (Synsets) that represent the concrete objects
# in your list (Tools, Vehicles, Furniture, Animals, Plants, Buildings).
TARGET_HYPERNYMS = {
    # Artifacts (man-made things: tools, furniture, vehicles, buildings)
    wn.synset('artifact.n.01'), 
    # Living Things (animals, insects, plants)
    wn.synset('living_thing.n.01'),
    # Body Parts (for arm, eye, foot, hand, leg)
    wn.synset('body_part.n.01')
}

def is_relevant_concept(synset: Synset) -> bool:
    """
    Checks if a synset belongs to a desired concrete, physical category 
    by traversing its hypernym (superclass) hierarchy.
    """
    # Use closure to traverse all hypernyms up the tree
    for hypernym in synset.closure(lambda s: s.hypernyms()):
        if hypernym in TARGET_HYPERNYMS:
            return True
    return False

# Crude category -> WordNet lexicographer-domain hint, used only to pick between
# multiple senses of a word that means different things in different categories
# (e.g. "squash" the vegetable vs. "squash" the sport). Not exhaustive -- it only
# needs to disambiguate the specific ambiguous words in a given word list.
CATEGORY_LEXNAME_HINTS: t.Dict[str, t.Set[str]] = {
    "sports": {"noun.act", "noun.event"},
    "vegetables": {"noun.plant", "noun.food"},
    "fruit": {"noun.plant", "noun.food"},
    "birds": {"noun.animal"},
    "clothing": {"noun.artifact"},
    "furniture": {"noun.artifact"},
    "vehicles": {"noun.artifact"},
    "professions": {"noun.person"},
}


def fetch_wordnet_gloss(concept: str, category: t.Optional[str] = None) -> str:
    """
    Fetches the definition for a concept, prioritizing senses that are concrete
    objects or natural kinds over abstract, informal, or verbal senses.

    If `category` is given and matches CATEGORY_LEXNAME_HINTS, a sense whose
    WordNet lexicographer domain matches is preferred over that default heuristic --
    this is what correctly separates e.g. "squash" the vegetable from "squash" the
    sport, which the generic concreteness heuristic below can't tell apart (both
    exist as valid noun senses; the concreteness heuristic always prefers the plant).
    """
    synsets = wn.synsets(concept, pos=wn.NOUN)

    if category:
        hints = CATEGORY_LEXNAME_HINTS.get(category.lower())
        if hints:
            for s in synsets:
                if s.lexname() in hints:
                    return s.definition()

    # 1. Check for relevant senses based on hypernyms
    relevant_synsets = [s for s in synsets if is_relevant_concept(s)]

    if relevant_synsets:
        # Prioritize the lowest index (most frequent) among the relevant senses
        best_synset = relevant_synsets[0]
        return best_synset.definition()

    # 2. Fallback: If no relevant hypernym is found, use the first available sense
    # This handles words where the concrete sense IS the first sense, or obscure words.
    if synsets:
        return synsets[0].definition()

    return f"{concept} (definition unavailable)"


def pick_article(word: str) -> str:
    return "an" if len(word) > 0 and word[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def normalize_sentence(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^\s*[\-\*\u2022]?\s*(\d+([\.)]|:))?\s*", "", s)
    s = s.strip('"\'').strip()
    s = re.sub(r"\s+", " ", s)
    return s


def unique_preserve_order(items: t.Iterable[str]) -> t.List[str]:
    seen = set()
    out: t.List[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


# ============================================================================
# DSPy Signatures
# ============================================================================

class GenerateFacts(dspy.Signature):
    """Generate 10 diverse, factual sentences about the given concept.

    Output exactly 10 sentences, one per line, with no numbering or bullets.
    Refer to the concept only by its name without specific classes, types, or names.
    If the concept name could refer to more than one thing, use the category and
    definition to write about the intended one only.
    Make sure the sentences are diverse and do not repeat.
    """

    concept: str = dspy.InputField(desc="The concept name to describe.")
    category: str = dspy.InputField(desc="The broader category this concept belongs to; "
                                          "disambiguates concept names with multiple meanings "
                                          "(e.g. 'squash' the vegetable vs. 'squash' the sport). "
                                          "May be blank for top-level category concepts.")
    article: str = dspy.InputField(desc="The article to use ('a' or 'an').")
    definition: str = dspy.InputField(desc="The WordNet definition of the concept.")
    random_seed: str = dspy.InputField(desc="Random seed to ensure diversity.")

    sentences: t.List[str] = dspy.OutputField(
        desc="10 factual sentences about the concept, one per line, no numbering."
    )


class GenerateStories(dspy.Signature):
    """Generate 10 diverse, story-based sentences about the given concept.

    Output exactly 10 sentences, one per line, with no numbering or bullets.
    Each sentence should be a short story about the concept.
    Refer to the concept only by its name without specific classes, types, or names.
    If the concept name could refer to more than one thing, use the category and
    definition to write about the intended one only.
    Make sure the sentences are diverse and do not repeat.
    """

    concept: str = dspy.InputField(desc="The concept name to describe.")
    category: str = dspy.InputField(desc="The broader category this concept belongs to; "
                                          "disambiguates concept names with multiple meanings "
                                          "(e.g. 'squash' the vegetable vs. 'squash' the sport). "
                                          "May be blank for top-level category concepts.")
    article: str = dspy.InputField(desc="The article to use ('a' or 'an').")
    definition: str = dspy.InputField(desc="The WordNet definition of the concept.")
    random_seed: str = dspy.InputField(desc="Random seed to ensure diversity.")

    sentences: t.List[str] = dspy.OutputField(
        desc="10 story sentences about the concept, one per line, no numbering."
    )


# ============================================================================
# DSPy Module
# ============================================================================

class ConceptGenerator(dspy.Module):
    """Generate positive examples for a concept using facts and stories."""
    
    def __init__(self, fact_count: int = 200, story_count: int = 200, generator: str = "predict"):
        super().__init__()
        if generator == "predict":
            self.fact_generator = dspy.Predict(GenerateFacts)
            self.story_generator = dspy.Predict(GenerateStories)
        else:
            self.fact_generator = dspy.ChainOfThought(GenerateFacts)
            self.story_generator = dspy.ChainOfThought(GenerateStories)
        self.fact_count = fact_count
        self.story_count = story_count
    
    def forward(self, concept: str, definition: str, article: str, category: str = ""):
        """Generate sentences for a concept (synchronous)."""
        positives = []

        # Generate facts
        fact_batches_needed = (self.fact_count + 9) // 10  # Ceiling division
        for _ in range(fact_batches_needed):
            try:
                result = self.fact_generator(
                    concept=concept,
                    category=category,
                    article=article,
                    definition=definition
                )
                positives.extend([sentence for sentence in result.sentences if sentence])
            except Exception as e:
                print(f"    Warning: Fact generation batch failed for {concept}: {e}")

        # Generate stories
        story_batches_needed = (self.story_count + 9) // 10  # Ceiling division
        for _ in range(story_batches_needed):
            try:
                result = self.story_generator(
                    concept=concept,
                    category=category,
                    article=article,
                    definition=definition
                )
                positives.extend([sentence for sentence in result.sentences if sentence])
            except Exception as e:
                print(f"    Warning: Story generation batch failed for {concept}: {e}")
        
        # Deduplicate and return
        positives = unique_preserve_order(positives)
        target = self.fact_count + self.story_count
        
        return dspy.Prediction(
            positives=positives[:target],
            fact_count=min(self.fact_count, len(positives)),
            story_count=min(self.story_count, len(positives) - self.fact_count) if len(positives) > self.fact_count else 0
        )
    
    async def aforward(self, concept: str, definition: str, article: str, category: str = ""):
        """Generate sentences for a concept (asynchronous)."""
        positives = []

        # Generate facts
        # --- Generate Facts ---
        facts_collected = []
        attempts = 0
        max_attempts = self.fact_count * 2  # Safety limit

        while len(facts_collected) < self.fact_count and attempts < max_attempts:
            attempts += 1
            try:
                result = await self.fact_generator.acall(
                    concept=concept,
                    category=category,
                    article=article,
                    definition=definition,
                    random_seed=str(random.random())
                )
                # Add unique valid sentences
                new_facts = [s for s in result.sentences if s]
                added_count = 0
                for fact in new_facts:
                    if fact not in facts_collected:
                        facts_collected.append(fact)
                        added_count += 1
                
                # Log progress every 5 attempts or if we added something
                if added_count > 0 or attempts % 5 == 0:
                    print(f"    [{concept}] Facts: {len(facts_collected)}/{self.fact_count} (Attempt {attempts})")
                    
            except Exception as e:
                print(f"    Warning: Fact generation attempt {attempts} failed for {concept}: {e}")
                # Optional: await asyncio.sleep(1) to be nice to API
        
        # --- Generate Stories ---
        stories_collected = []
        attempts = 0
        max_attempts = self.story_count * 2
        
        while len(stories_collected) < self.story_count and attempts < max_attempts:
            attempts += 1
            try:
                result = await self.story_generator.acall(
                    concept=concept,
                    category=category,
                    article=article,
                    definition=definition,
                    random_seed=str(random.random())
                )
                new_stories = [s for s in result.sentences if s]
                added_count = 0
                for story in new_stories:
                    if story not in stories_collected:
                        stories_collected.append(story)
                        added_count += 1

                # Log progress
                if added_count > 0 or attempts % 5 == 0:
                    print(f"    [{concept}] Stories: {len(stories_collected)}/{self.story_count} (Attempt {attempts})")

            except Exception as e:
                print(f"    Warning: Story generation attempt {attempts} failed for {concept}: {e}")
        
        # Combine and Deduplicate
        all_sentences = facts_collected[:self.fact_count] + stories_collected[:self.story_count]
        positives = unique_preserve_order(all_sentences)
        
        return dspy.Prediction(
            positives=positives,
            fact_count=len(facts_collected),
            story_count=len(stories_collected)
        )


# ============================================================================
# Async wrapper for DSPy (uses native async support)
# ============================================================================

async def generate_concept_positives_async(
    concept: str,
    storage_key: str,
    category: t.Optional[str],
    generator: ConceptGenerator,
) -> t.Tuple[str, str, t.Optional[str], t.List[str]]:
    """Generate positives for a single concept asynchronously using DSPy's native async support."""
    definition = fetch_wordnet_gloss(concept, category)
    article = pick_article(concept)

    # Use acall() to ensure DSPy async wrappers (callbacks, context, usage tracking) are applied.
    result = await generator.acall(concept=concept, definition=definition, article=article, category=category or "")

    # Verify counts
    total_target = generator.fact_count + generator.story_count
    if len(result.positives) < total_target:
        print(f"    WARNING: {concept} only generated {len(result.positives)}/{total_target} positives!")

    return concept, storage_key, category, result.positives


# ============================================================================
# Negative building with stratification
# ============================================================================

def build_negatives_stratified(
    *,
    positives_by_concept: t.Dict[str, t.List[str]],
    target_negatives: int,
    seed: int,
) -> t.Dict[str, t.List[str]]:
    """
    Build negatives stratified uniformly by source concept.
    
    For each target concept c:
      - Split the quota evenly across all other concepts.
      - Distribute any remainder round-robin.
      - Sample without replacement where possible.
    """
    rng = random.Random(seed)
    concepts = list(positives_by_concept.keys())
    negatives: t.Dict[str, t.List[str]] = {}

    for c in concepts:
        sources = [s for s in concepts if s != c]
        if not sources:
            negatives[c] = []
            continue

        k = len(sources)
        base_quota = target_negatives // k
        remainder = target_negatives % k

        # Shuffle round-robin order
        rr = sources[:]
        rng.shuffle(rr)

        # Initial per-source quotas
        quota = {s: base_quota for s in sources}
        for s in rr[:remainder]:
            quota[s] += 1

        # Select from each source
        selected: t.List[str] = []
        deficits = 0
        
        for s in sources:
            pool = positives_by_concept.get(s, [])
            q = quota[s]

            if q <= 0:
                continue

            if len(pool) >= q:
                picks = rng.sample(pool, q)
                selected.extend(picks)
            else:
                if len(pool) > 0:
                    picks = rng.sample(pool, len(pool))
                    selected.extend(picks)
                deficits += max(0, q - len(pool))

        # Borrow for any deficits
        if deficits > 0:
            global_pool = []
            selected_set = set(selected)
            for s in sources:
                for sent in positives_by_concept.get(s, []):
                    if sent not in selected_set:
                        global_pool.append(sent)

            if len(global_pool) >= deficits:
                selected.extend(rng.sample(global_pool, deficits))
            else:
                selected.extend(global_pool)
                still_need = deficits - len(global_pool)
                all_other = [sent for s in sources for sent in positives_by_concept.get(s, [])]
                if still_need > 0 and all_other:
                    selected.extend(rng.choices(all_other, k=still_need))

        if len(selected) > target_negatives:
            selected = selected[:target_negatives]

        negatives[c] = selected

    return negatives


# ============================================================================
# File I/O
# ============================================================================

def storage_key_for(concept: str, category: t.Optional[str], all_concepts: t.List[str]) -> str:
    """A bare concept name is ambiguous when the same word is used for two different
    senses in the same word list (e.g. "squash" the vegetable vs. "squash" the sport --
    both appear in the Richie & Bhatia HSJ table). When a name repeats, disambiguate the
    on-disk storage key (filename / intermediate dict key) with its category, so the two
    senses don't silently overwrite each other. The "concept" field written into every
    JSON file's content, and the value sent to the LM, stays the plain word, unchanged.
    """
    if category and all_concepts.count(concept) > 1:
        return f"{concept}__{category}"
    return concept


def load_categories_for_concepts(
    metadata_path: pathlib.Path, concepts: t.List[str]
) -> t.List[t.Optional[str]]:
    """Loads (concept, category) pairs from a metadata JSON (same format as
    metadata_Richie_HSJ.json), aligned by position with `concepts` -- both are built by
    flattening the same source list in the same order. Falls back to all-None (no
    disambiguation) if the metadata doesn't line up, rather than risk mismatched pairing.
    """
    with metadata_path.open("r", encoding="utf-8") as fp:
        entries = json.load(fp)
    meta_concepts = [e["concept"] for e in entries]
    if meta_concepts != concepts:
        print(f"Warning: {metadata_path} concepts don't match config concepts 1:1; "
              f"category-based disambiguation disabled.")
        return [None] * len(concepts)
    return [e.get("category") for e in entries]


def write_concept_json(
    *,
    dataset_dir: pathlib.Path,
    concept: str,
    storage_key: str,
    category: t.Optional[str],
    group: str,
    source: str,
    positives: t.List[str],
    negatives: t.List[str],
) -> None:
    out = {
        "concept": concept,
        "category": category,
        "group": group,
        "source": source,
        "sentences": {
            "positive": positives,
            "negative": negatives,
        },
    }
    out_path = dataset_dir / group / f"{storage_key}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, separators=(',', ':'))


def write_concept_list_csv(
    *, dataset_root: pathlib.Path, group: str, entries: t.List[t.Tuple[str, str, t.Optional[str]]]
) -> None:
    csv_path = dataset_root / "concept_list.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as fp:
        fp.write("group,concept,storage_key,category\n")
        for concept, storage_key, category in entries:
            fp.write(f"{group},{concept},{storage_key},{category or ''}\n")


def write_intermediate_positives(
    *,
    intermediate_dir: pathlib.Path,
    concept: str,
    storage_key: str,
    category: t.Optional[str],
    positives: t.List[str],
) -> None:
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    out = {"concept": concept, "storage_key": storage_key, "category": category, "positives": positives}
    with (intermediate_dir / f"{storage_key}.json").open("w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False)


def load_all_intermediate_records(
    intermediate_dir: pathlib.Path,
) -> t.Dict[str, dict]:
    """Returns the full record (concept, category, storage_key, positives) for every
    intermediate file, keyed by storage_key (falling back to concept for older files
    written before storage_key existed, so other datasets' existing .intermediate
    directories keep working unchanged)."""
    records: t.Dict[str, dict] = {}
    if not intermediate_dir.exists():
        return records

    for json_file in intermediate_dir.glob("*.json"):
        try:
            with json_file.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            key = data.get("storage_key") or data["concept"]
            records[key] = data
        except Exception as e:
            # Surfaced rather than silently swallowed: a file that fails to load looks
            # identical to "doesn't exist yet" to every caller, which previously caused
            # affected concepts to regenerate forever under --resume without ever being
            # recognized as complete (root cause: reading without encoding="utf-8" broke
            # on Windows' default cp1252 locale for any non-ASCII generated text).
            print(f"    WARNING: failed to load intermediate file {json_file.name}: {e}")
            continue
    return records


def load_env_file_if_present(env_path: pathlib.Path) -> None:
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


# ============================================================================
# Main pipeline
# ============================================================================

async def generate_positives(
    *,
    cfg: GenerationConfig,
    generator_type: str,
    entries: t.List[t.Tuple[str, str, t.Optional[str]]],
    intermediate_dir: pathlib.Path,
    max_concurrent_concepts: int,
) -> t.List[str]:
    """Phase 1: Generate positives for all concepts with high concurrency using DSPy's native async.

    `entries` is a list of (concept, storage_key, category) tuples -- storage_key is what
    the on-disk file gets named, so that concepts sharing a name across categories (e.g.
    "squash" as both a vegetable and a sport) don't overwrite each other. Returns the list
    of storage_keys that failed outright (raised an exception), for a visible end-of-run summary.
    """
    print(f"Phase 1: Generating positives for {len(entries)} concepts...")

    # Initialize DSPy generator
    generator = ConceptGenerator(
        fact_count=cfg.positive_fact,
        story_count=cfg.positive_story,
        generator=generator_type
    )

    failed_storage_keys: t.List[str] = []

    # Process in batches using DSPy's native async support
    for i in range(0, len(entries), max_concurrent_concepts):
        batch = entries[i:i + max_concurrent_concepts]

        print(f"  Processing batch {i//max_concurrent_concepts + 1}: {len(batch)} concepts...")

        # Create async tasks using DSPy's acall()
        tasks = [
            generate_concept_positives_async(concept, storage_key, category, generator)
            for concept, storage_key, category in batch
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Save results
        for result, (concept, storage_key, category) in zip(results, batch):
            if isinstance(result, BaseException):
                print(f"    ERROR generating {storage_key}: {result}")
                failed_storage_keys.append(storage_key)
                continue

            _, _, _, positives = result
            write_intermediate_positives(
                intermediate_dir=intermediate_dir,
                concept=concept,
                storage_key=storage_key,
                category=category,
                positives=positives,
            )
            print(f"    Generated {len(positives)} positives for {storage_key}")

    if failed_storage_keys:
        print(f"\nPhase 1 finished with {len(failed_storage_keys)} concept(s) that raised an "
              f"error and were NOT written to .intermediate: {failed_storage_keys}")
        print("Re-run with --resume to retry just these.")

    return failed_storage_keys


def gather_negatives(
    *,
    cfg: GenerationConfig,
    entries: t.List[t.Tuple[str, str, t.Optional[str]]],
    dataset_root: pathlib.Path,
    intermediate_dir: pathlib.Path,
) -> None:
    """Phase 2: Build stratified negatives and write final JSON files.

    `entries` is a list of (concept, storage_key, category) tuples, same as generate_positives.
    """
    print("Phase 2: Loading intermediate positives...")
    records = load_all_intermediate_records(intermediate_dir)
    positives_by_key = {key: r["positives"] for key, r in records.items()}

    if not positives_by_key:
        raise RuntimeError("No intermediate positives found. Run phase 1 first.")

    print(f"Phase 2: Building stratified negatives for {len(entries)} concepts...")
    negatives_by_key = build_negatives_stratified(
        positives_by_concept=positives_by_key,
        target_negatives=cfg.negatives_per_concept,
        seed=cfg.seed,
    )

    # Write final JSON files
    group = "custom"
    source_name = f"dspy_{cfg.model.replace('/', '_')}"

    missing = []
    for concept, storage_key, category in tqdm(entries, desc="Writing concept files"):
        positives = positives_by_key.get(storage_key, [])
        negatives = negatives_by_key.get(storage_key, [])
        if not positives:
            missing.append(storage_key)
        write_concept_json(
            dataset_dir=dataset_root,
            concept=concept,
            storage_key=storage_key,
            category=category,
            group=group,
            source=source_name,
            positives=positives,
            negatives=negatives,
        )

    print(f"  Wrote datasets for {len(entries)} concepts")
    if missing:
        print(f"  WARNING: {len(missing)} concept(s) had no intermediate positives and were "
              f"written with empty sentences: {missing}")

    # Write concept list CSV
    write_concept_list_csv(dataset_root=dataset_root, group=group, entries=entries)

    print(f"Phase 2: Complete. Dataset written to: {dataset_root}")


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate concept datasets using DSPy framework"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("../dataset_config_Qwen3-30B-A3B-Instruct-2507-abstractiveness_Richie_HSJ.json"),
        help="Path to dataset configuration JSON file",
    )
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=pathlib.Path("../abstractiveness/assets"),
        help="Root directory for generated dataset",
    )
    parser.add_argument(
        "--metadata",
        type=pathlib.Path,
        default=pathlib.Path("../abstractiveness/assets/metadata_Richie_HSJ.json"),
        help="Metadata JSON (concept+category pairs, same format as metadata_Richie_HSJ.json) "
             "used to disambiguate concept names reused across categories (e.g. \"squash\" as "
             "both a vegetable and a sport). Pass a nonexistent path to disable.",
    )
    parser.add_argument(
        "--only-concepts",
        type=str,
        default="",
        help="Comma-separated subset of concepts",
    )
    parser.add_argument(
        "--limit-concepts",
        type=int,
        default=0,
        help="Limit to first N concepts",
    )
    parser.add_argument(
        "--max-concurrent-concepts",
        type=int,
        default=10,
        help="Max concurrent concept processing",
    )
    parser.add_argument(
        "--phase",
        choices=["both", "positives", "negatives"],
        default="both",
        help="Which phase to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip concepts already present in .intermediate when running positives",
    )
    parser.add_argument(
        "--fix-intermediate",
        action="store_true",
        help="Fix mode: rebuild final datasets from all existing .intermediate files",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for LM provider (overrides config/env)",
    )
    parser.add_argument(
        "--generator-type",
        type=str,
        default="cot",
        choices=["predict", "cot"],
        help="Generator to use for DSPy",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    random.seed(cfg.seed)

    # Load environment
    load_env_file_if_present(pathlib.Path(".env"))

    # Get API key
    api_key = args.api_key or os.environ.get("VLLM_API_KEY", "placeholder")
    
    # Configure DSPy
    # Format: "provider/model" e.g., "openai/gpt-4" or "gemini/gemini-2.5-flash"
    lm = dspy.LM(cfg.model, api_key=api_key, api_base=cfg.base_url, temperature=cfg.temperature, max_tokens=cfg.max_tokens, timeout=1200)
    dspy.configure(lm=lm)
    
    print(f"Configured DSPy with model: {cfg.model}")

    # Setup paths
    dataset_root = args.dataset_root / cfg.dataset_name
    intermediate_dir = dataset_root / "custom" / ".intermediate"

    # Build (concept, storage_key, category) entries. storage_key only differs from
    # concept when the same concept name is reused across categories (e.g. "squash").
    all_concepts = cfg.concepts
    if args.metadata and args.metadata.exists():
        categories = load_categories_for_concepts(args.metadata, all_concepts)
    else:
        categories = [None] * len(all_concepts)
    storage_keys = [storage_key_for(c, cat, all_concepts) for c, cat in zip(all_concepts, categories)]
    entries = list(zip(all_concepts, storage_keys, categories))

    dupes = {sk for sk in storage_keys if storage_keys.count(sk) > 1}
    if dupes:
        print(f"Warning: these storage keys are STILL ambiguous after category "
              f"disambiguation (no --metadata, or missing category): {sorted(dupes)}")

    # Apply filters (match against either the bare concept or its storage_key)
    if args.only_concepts:
        requested = {c.strip() for c in args.only_concepts.split(",") if c.strip()}
        entries = [(c, sk, cat) for c, sk, cat in entries if c in requested or sk in requested]
    if args.limit_concepts and args.limit_concepts > 0:
        entries = entries[:args.limit_concepts]

    # Handle --fix-intermediate: Force use of all intermediate concepts
    if args.fix_intermediate:
        print("Mode: Fix/Rebuild from intermediate files. Ignoring config/filter concepts.")
        records = load_all_intermediate_records(intermediate_dir)
        entries = [
            (r["concept"], key, r.get("category"))
            for key, r in records.items()
        ]
        print(f"Found {len(entries)} concepts in .intermediate")
        args.phase = "negatives"  # Force phase to negatives/processing only

    target_positives = cfg.positive_fact + cfg.positive_story

    # Resume support: skip concepts already present in .intermediate for positives phase,
    # but only if they actually reached the target count -- a partial file (e.g. from a
    # run that hit the retry cap due to repetitive model output) is treated as NOT done,
    # so it gets regenerated rather than being silently stuck forever.
    entries_to_generate = list(entries)
    if args.phase in ["both", "positives"] and args.resume and not args.fix_intermediate:
        records = load_all_intermediate_records(intermediate_dir)
        complete = {
            key for key, r in records.items()
            if len(r.get("positives", [])) >= target_positives
        }
        if complete:
            entries_to_generate = [(c, sk, cat) for c, sk, cat in entries if sk not in complete]
            print(f"Resume: Skipping {len(entries) - len(entries_to_generate)} concepts already "
                  f"at {target_positives}/{target_positives} positives.")

    print(f"Processing {len(entries)} concepts (Generation queue: {len(entries_to_generate)})")

    # Execute phases
    generated_something = False
    if args.phase in ["both", "positives"]:
        if not entries_to_generate:
             print("No concepts to generate (all exist or empty list).")
        else:
            await generate_positives(
                cfg=cfg,
                generator_type=args.generator_type,
                entries=entries_to_generate,
                intermediate_dir=intermediate_dir,
                max_concurrent_concepts=args.max_concurrent_concepts,
            )
            generated_something = True

    # --resume's whole point is to end up with a complete, consistent dataset -- not just
    # topped-up .intermediate files -- so always rebuild the final output afterward, even
    # if --phase was only "positives".
    run_negatives = args.phase in ["both", "negatives"] or (args.resume and generated_something)
    if run_negatives:
        if args.resume and args.phase == "positives" and generated_something:
            print("Resume also rebuilding final output (Phase 2) so it reflects the newly "
                  "completed concepts, since --phase was 'positives' alone.")
        gather_negatives(
            cfg=cfg,
            entries=entries,
            dataset_root=dataset_root,
            intermediate_dir=intermediate_dir,
        )


if __name__ == "__main__":
    asyncio.run(async_main())