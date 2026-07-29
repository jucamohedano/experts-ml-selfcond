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


# High-level hypernyms marking a synset as a concrete, physical thing. Resolved lazily
# rather than at import time, because resolving a synset requires the WordNet corpus to
# already be downloaded, which ensure_nltk_wordnet() only guarantees once main() runs.
TARGET_HYPERNYM_NAMES = (
    # Artifacts (man-made things: tools, furniture, vehicles, buildings)
    'artifact.n.01',
    # Living Things (animals, insects, plants)
    'living_thing.n.01',
    # Body Parts (for arm, eye, foot, hand, leg)
    'body_part.n.01',
)

_TARGET_HYPERNYMS: t.Optional[t.Set[Synset]] = None


def target_hypernyms() -> t.Set[Synset]:
    global _TARGET_HYPERNYMS
    if _TARGET_HYPERNYMS is None:
        _TARGET_HYPERNYMS = {wn.synset(name) for name in TARGET_HYPERNYM_NAMES}
    return _TARGET_HYPERNYMS


def is_relevant_concept(synset: Synset) -> bool:
    """
    Checks if a synset belongs to a desired concrete, physical category
    by traversing its hypernym (superclass) hierarchy.
    """
    targets = target_hypernyms()
    # Use closure to traverse all hypernyms up the tree
    for hypernym in synset.closure(lambda s: s.hypernyms()):
        if hypernym in targets:
            return True
    return False


# Category -> the WordNet synsets the intended sense must descend from (or be). This is
# the primary disambiguation signal, and it is far stricter than the lexicographer-domain
# hint below: "canary" has a noun.animal sense and a noun.person sense, but only the bird
# descends from bird.n.01. Anchors are resolved lazily, for the same reason as above.
CATEGORY_ANCHOR_NAMES: t.Dict[str, t.Tuple[str, ...]] = {
    "birds": ("bird.n.01",),
    "fruit": ("edible_fruit.n.01", "fruit.n.01"),
    "vegetables": ("vegetable.n.01", "herb.n.01"),
    "clothing": ("clothing.n.01", "garment.n.01", "footwear.n.02", "accessory.n.01"),
    "furniture": ("furniture.n.01", "furnishing.n.02", "home_appliance.n.01"),
    "vehicles": ("vehicle.n.01",),
    "sports": ("sport.n.01", "athletic_game.n.01", "diversion.n.01"),
    "professions": ("person.n.01",),
}

# Crude category -> WordNet lexicographer-domain hint, used to pick between multiple
# senses when no anchor matches. Broader than the anchors and so a weaker signal, but it
# rescues concepts whose intended sense sits outside the anchor subtree: "chess" the board
# game is not under sport.n.01 (its first sense is a weed, noun.plant), while noun.act
# selects it correctly.
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

# Level 1 category labels carry no category of their own, so neither the anchors nor the
# lexname hints apply to them and the generic heuristic picks a person sense ("sports"
# resolves to sport.n.03, "(Maine colloquial) a temporary summer resident"). There are
# only eight of them, so they are pinned explicitly.
LEVEL1_LABEL_SENSES: t.Dict[str, str] = {
    "birds": "bird.n.01",
    "clothing": "clothing.n.01",
    "fruit": "fruit.n.01",
    "furniture": "furniture.n.01",
    "professions": "profession.n.02",
    "sports": "sport.n.01",
    "vegetables": "vegetable.n.01",
    "vehicles": "vehicle.n.01",
}

# Concepts where WordNet sense ordering picks a wrong-but-anchor-matching sense, so no
# amount of hierarchy walking helps. Both competing senses satisfy the category, and only
# the source word list settles which one is meant. Keyed by (concept, category).
SENSE_OVERRIDES: t.Dict[t.Tuple[str, str], str] = {
    # a soft padded bag, not shock_absorber.n.01 "a mechanical damper"
    ("cushion", "furniture"): "cushion.n.03",
    # the road vehicle, not van.n.03 "(Great Britain) a closed railroad car"
    ("van", "vehicles"): "van.n.05",
    # the motor scooter, not water_scooter.n.01 "a motorboat resembling a motor scooter"
    ("scooter", "vehicles"): "motor_scooter.n.01",
    # horse-drawn, not passenger_car.n.01 "a railcar where passengers ride"
    ("carriage", "vehicles"): "carriage.n.02",
    # the firefighter, not stoker.n.02 "a laborer who tends fires"
    ("fireman", "professions"): "fireman.n.04",
    # the clerical assistant, not secretary.n.01 "head of an administrative department"
    ("secretary", "professions"): "secretary.n.02",
    # the government office holder, not curate.n.01 "authorized to conduct religious worship"
    ("minister", "professions"): "minister.n.02",
    # riding in a sailboat, not seafaring.n.01 "the work of a sailor"
    ("sailing", "sports"): "sailing.n.02",
}


def select_wordnet_sense(concept: str, category: t.Optional[str] = None) -> t.Optional[Synset]:
    """
    Picks the WordNet noun sense of `concept` that the given `category` intends.

    Resolution order, first hit wins:

    1. SENSE_OVERRIDES, for the handful of words where two senses both satisfy the
       category and only the source word list decides between them.
    2. LEVEL1_LABEL_SENSES, when the concept is a category label and so has no category.
    3. CATEGORY_ANCHOR_NAMES, the lowest-indexed sense that is, or descends from, one of
       the category's anchor synsets.
    4. CATEGORY_LEXNAME_HINTS, the lowest-indexed sense in a matching lexicographer domain.
    5. The concreteness heuristic (is_relevant_concept), with noun.person senses excluded
       unless the category is professions. That exclusion is the core fix: person inherits
       from organism and then living_thing, so without it every noun.person sense passes
       the concreteness filter and outranks the intended one ("date" the escort beating
       "date" the fruit, "canary" the informer beating "canary" the bird).
    6. The first sense WordNet lists, for words whose concrete sense is already first.

    Returns None only when WordNet has no noun sense for the word at all.
    """
    key = category.lower() if category else None

    synsets = wn.synsets(concept, pos=wn.NOUN)
    if not synsets:
        return None

    if key:
        override = SENSE_OVERRIDES.get((concept.lower(), key))
        if override:
            return wn.synset(override)
    else:
        label_sense = LEVEL1_LABEL_SENSES.get(concept.lower())
        if label_sense:
            return wn.synset(label_sense)

    if key:
        anchor_names = CATEGORY_ANCHOR_NAMES.get(key, ())
        anchors = {wn.synset(name) for name in anchor_names}
        if anchors:
            for s in synsets:
                if s in anchors or anchors & set(s.closure(lambda x: x.hypernyms())):
                    return s

        hints = CATEGORY_LEXNAME_HINTS.get(key)
        if hints:
            for s in synsets:
                if s.lexname() in hints:
                    return s

    allow_person = key == "professions"
    relevant = [
        s for s in synsets
        if is_relevant_concept(s) and (allow_person or s.lexname() != "noun.person")
    ]
    if relevant:
        # Prioritize the lowest index (most frequent) among the relevant senses
        return relevant[0]

    return synsets[0]


def describe_sense(synset: t.Optional[Synset], concept: str) -> str:
    """
    A compact restatement of the chosen sense, given to the LM alongside the definition so
    the intended meaning is pinned by more than one phrasing. Reads e.g.
    "date, escort (a kind of participant)".
    """
    if synset is None:
        return f"{concept} (no WordNet sense available)"
    lemmas = ", ".join(dict.fromkeys(l.name().replace("_", " ") for l in synset.lemmas()))
    hypernyms = synset.hypernyms()
    if hypernyms:
        parent = hypernyms[0].lemmas()[0].name().replace("_", " ")
        return f"{lemmas} (a kind of {parent})"
    return lemmas


def fetch_wordnet_gloss(concept: str, category: t.Optional[str] = None) -> str:
    """Definition of the sense chosen by select_wordnet_sense."""
    synset = select_wordnet_sense(concept, category)
    if synset is None:
        return f"{concept} (definition unavailable)"
    return synset.definition()


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
    sense_hint: str = dspy.InputField(desc="Synonyms of the intended sense and what kind of "
                                           "thing it is, e.g. 'date (a kind of edible fruit)'. "
                                           "Write about this sense only.")
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
    sense_hint: str = dspy.InputField(desc="Synonyms of the intended sense and what kind of "
                                           "thing it is, e.g. 'date (a kind of edible fruit)'. "
                                           "Write about this sense only.")
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
    
    def forward(self, concept: str, definition: str, article: str, category: str = "",
                sense_hint: str = ""):
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
                    definition=definition,
                    sense_hint=sense_hint
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
                    definition=definition,
                    sense_hint=sense_hint
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
    
    async def aforward(self, concept: str, definition: str, article: str, category: str = "",
                       sense_hint: str = ""):
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
                    sense_hint=sense_hint,
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
                    sense_hint=sense_hint,
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
    synset = select_wordnet_sense(concept, category)
    definition = synset.definition() if synset else f"{concept} (definition unavailable)"
    sense_hint = describe_sense(synset, concept)
    article = pick_article(concept)
    print(f"    [{concept}] sense: {synset.name() if synset else 'none'} -- {definition}")

    # Use acall() to ensure DSPy async wrappers (callbacks, context, usage tracking) are applied.
    result = await generator.acall(concept=concept, definition=definition, article=article,
                                   category=category or "", sense_hint=sense_hint)

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

    Rows without a "concept" key are skipped rather than raising: the metadata file also
    carries deliberately excluded entries under underscore-prefixed keys (e.g.
    "_ignored_concept_squash_sport"), which are not part of the dataset and must not shift
    the positional alignment.
    """
    with metadata_path.open("r", encoding="utf-8") as fp:
        raw_entries = json.load(fp)
    entries = [e for e in raw_entries if isinstance(e, dict) and "concept" in e]
    skipped = len(raw_entries) - len(entries)
    if skipped:
        print(f"Note: skipped {skipped} metadata row(s) without a 'concept' key "
              f"(excluded entries).")
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
    all_entries: t.Optional[t.List[t.Tuple[str, str, t.Optional[str]]]] = None,
    source_tag: str = "",
) -> None:
    """Phase 2: Build stratified negatives and write final JSON files.

    `entries` is a list of (concept, storage_key, category) tuples, same as generate_positives.
    Only these concepts get a JSON written, which is what makes a targeted rebuild possible
    (regenerate a handful of concepts and leave every other file on disk untouched, so the
    model responses already computed from them stay valid).

    `all_entries` is the unfiltered list, used only for concept_list.csv. The CSV describes
    the whole dataset, so writing it from a filtered `entries` would silently truncate it to
    the rebuilt subset. It defaults to `entries` for a full run, where the two are the same.
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
    if source_tag:
        # Records that a file came from a later, targeted rebuild rather than the original
        # sweep, so a dataset containing both provenances stays auditable.
        source_name = f"{source_name}_{source_tag}"

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

    # Write concept list CSV, always describing the full dataset rather than the subset
    # that was just rebuilt.
    write_concept_list_csv(
        dataset_root=dataset_root, group=group, entries=all_entries if all_entries else entries
    )

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
        "--source-tag",
        type=str,
        default="",
        help="Suffix appended to the 'source' field of every concept JSON written by this "
             "run, e.g. 'sensefix2026-07'. Use it when rebuilding a subset so the mixed "
             "provenance of the dataset stays visible in the files themselves.",
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
    # Kept unfiltered: concept_list.csv must describe the whole dataset even when this run
    # only rebuilds a subset via --only-concepts.
    all_entries = list(entries)

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
        # In fix mode the intermediate files, not the config, define the dataset.
        all_entries = list(entries)
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
            all_entries=all_entries,
            source_tag=args.source_tag,
        )


if __name__ == "__main__":
    asyncio.run(async_main())