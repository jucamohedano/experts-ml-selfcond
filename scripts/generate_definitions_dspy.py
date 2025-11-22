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
    with path.open("r") as fp:
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


def fetch_wordnet_gloss(concept: str) -> str:
    ensure_nltk_wordnet()
    assert wn is not None
    synsets = wn.synsets(concept, pos=wn.NOUN)
    if not synsets:
        synsets = wn.synsets(concept)
    if not synsets:
        return f"{concept} (definition unavailable)"
    return synsets[0].definition()


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
    Make sure the sentences are diverse and do not repeat.
    """
    
    concept: str = dspy.InputField(desc="The concept name to describe.")
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
    Make sure the sentences are diverse and do not repeat.
    """
    
    concept: str = dspy.InputField(desc="The concept name to describe.")
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
    
    def __init__(self, fact_count: int = 200, story_count: int = 200):
        super().__init__()
        self.fact_generator = dspy.ChainOfThought(GenerateFacts)
        self.story_generator = dspy.ChainOfThought(GenerateStories)
        self.fact_count = fact_count
        self.story_count = story_count
    
    def forward(self, concept: str, definition: str, article: str):
        """Generate sentences for a concept (synchronous)."""
        positives = []
        
        # Generate facts
        fact_batches_needed = (self.fact_count + 9) // 10  # Ceiling division
        for _ in range(fact_batches_needed):
            try:
                result = self.fact_generator(
                    concept=concept,
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
    
    async def aforward(self, concept: str, definition: str, article: str):
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
    generator: ConceptGenerator,
) -> t.Tuple[str, t.List[str]]:
    """Generate positives for a single concept asynchronously using DSPy's native async support."""
    definition = fetch_wordnet_gloss(concept)
    article = pick_article(concept)
    
    # Use acall() to ensure DSPy async wrappers (callbacks, context, usage tracking) are applied.
    result = await generator.acall(concept=concept, definition=definition, article=article)
    
    # Verify counts
    total_target = generator.fact_count + generator.story_count
    if len(result.positives) < total_target:
        print(f"    WARNING: {concept} only generated {len(result.positives)}/{total_target} positives!")
    
    return concept, result.positives


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

def write_concept_json(
    *,
    dataset_dir: pathlib.Path,
    concept: str,
    group: str,
    source: str,
    positives: t.List[str],
    negatives: t.List[str],
) -> None:
    out = {
        "concept": concept,
        "group": group,
        "source": source,
        "sentences": {
            "positive": positives,
            "negative": negatives,
        },
    }
    out_path = dataset_dir / group / f"{concept}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fp:
        json.dump(out, fp, ensure_ascii=False, separators=(',', ':'))


def write_concept_list_csv(
    *, dataset_root: pathlib.Path, group: str, concepts: t.List[str]
) -> None:
    csv_path = dataset_root / "concept_list.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as fp:
        fp.write("group,concept\n")
        for c in concepts:
            fp.write(f"{group},{c}\n")


def write_intermediate_positives(
    *, intermediate_dir: pathlib.Path, concept: str, positives: t.List[str]
) -> None:
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    out = {"concept": concept, "positives": positives}
    with (intermediate_dir / f"{concept}.json").open("w") as fp:
        json.dump(out, fp, ensure_ascii=False)


def load_all_intermediate_positives(
    intermediate_dir: pathlib.Path,
) -> t.Dict[str, t.List[str]]:
    positives_by_concept: t.Dict[str, t.List[str]] = {}
    if not intermediate_dir.exists():
        return positives_by_concept
    
    for json_file in intermediate_dir.glob("*.json"):
        try:
            with json_file.open("r") as fp:
                data = json.load(fp)
            concept = data["concept"]
            positives = data["positives"]
            positives_by_concept[concept] = positives
        except Exception:
            continue
    return positives_by_concept


def load_env_file_if_present(env_path: pathlib.Path) -> None:
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
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
    concepts: t.List[str],
    intermediate_dir: pathlib.Path,
    max_concurrent_concepts: int,
) -> None:
    """Phase 1: Generate positives for all concepts with high concurrency using DSPy's native async."""
    print(f"Phase 1: Generating positives for {len(concepts)} concepts...")
    
    # Initialize DSPy generator
    generator = ConceptGenerator(
        fact_count=cfg.positive_fact,
        story_count=cfg.positive_story
    )
    
    # Process in batches using DSPy's native async support
    for i in range(0, len(concepts), max_concurrent_concepts):
        batch_concepts = concepts[i:i + max_concurrent_concepts]
        
        print(f"  Processing batch {i//max_concurrent_concepts + 1}: {len(batch_concepts)} concepts...")
        
        # Create async tasks using DSPy's acall()
        tasks = [
            generate_concept_positives_async(concept, generator)
            for concept in batch_concepts
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Save results
        for result in results:
            if isinstance(result, Exception):
                print(f"    ERROR: {result}")
                continue
            
            concept, positives = result
            write_intermediate_positives(
                intermediate_dir=intermediate_dir,
                concept=concept,
                positives=positives,
            )
            print(f"    Generated {len(positives)} positives for {concept}")


def gather_negatives(
    *,
    cfg: GenerationConfig,
    concepts: t.List[str],
    dataset_root: pathlib.Path,
    intermediate_dir: pathlib.Path,
) -> None:
    """Phase 2: Build stratified negatives and write final JSON files."""
    print("Phase 2: Loading intermediate positives...")
    positives_by_concept = load_all_intermediate_positives(intermediate_dir)
    
    if not positives_by_concept:
        raise RuntimeError("No intermediate positives found. Run phase 1 first.")
    
    print(f"Phase 2: Building stratified negatives for {len(concepts)} concepts...")
    negatives_by_concept = build_negatives_stratified(
        positives_by_concept=positives_by_concept,
        target_negatives=cfg.negatives_per_concept,
        seed=cfg.seed,
    )

    # Write final JSON files
    group = "custom"
    source_name = f"dspy_{cfg.model.replace('/', '_')}"
    
    for concept in tqdm(concepts, desc="Writing concept files"):
        positives = positives_by_concept.get(concept, [])
        negatives = negatives_by_concept.get(concept, [])
        write_concept_json(
            dataset_dir=dataset_root,
            concept=concept,
            group=group,
            source=source_name,
            positives=positives,
            negatives=negatives,
        )
    
    print(f"  Wrote datasets for {len(concepts)} concepts")

    # Write concept list CSV
    write_concept_list_csv(dataset_root=dataset_root, group=group, concepts=concepts)
    
    print(f"Phase 2: Complete. Dataset written to: {dataset_root}")


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate concept datasets using DSPy framework"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("dataset_config.json"),
        help="Path to dataset configuration JSON file",
    )
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=pathlib.Path("assets"),
        help="Root directory for generated dataset",
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
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    random.seed(cfg.seed)

    # Load environment
    load_env_file_if_present(pathlib.Path(".env"))

    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    
    # Configure DSPy
    # Format: "provider/model" e.g., "openai/gpt-4" or "gemini/gemini-2.5-flash"
    lm = dspy.LM(cfg.model, api_key=api_key, api_base=cfg.base_url, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    dspy.configure(lm=lm)
    
    print(f"Configured DSPy with model: {cfg.model}")

    # Setup paths
    dataset_root = args.dataset_root / cfg.dataset_name
    intermediate_dir = dataset_root / "custom" / ".intermediate"

    # Apply filters
    concepts = cfg.concepts
    if args.only_concepts:
        requested = [c.strip() for c in args.only_concepts.split(",") if c.strip()]
        concepts = [c for c in concepts if c in requested]
    if args.limit_concepts and args.limit_concepts > 0:
        concepts = concepts[:args.limit_concepts]
    
    # Handle --fix-intermediate: Force use of all intermediate concepts
    if args.fix_intermediate:
        print("Mode: Fix/Rebuild from intermediate files. Ignoring config/filter concepts.")
        existing_data = load_all_intermediate_positives(intermediate_dir)
        concepts = unique_preserve_order(existing_data.keys())
        print(f"Found {len(concepts)} concepts in .intermediate")
        args.phase = "negatives"  # Force phase to negatives/processing only

    # Resume support: skip concepts already present in .intermediate for positives phase
    concepts_to_generate = list(concepts)
    if args.phase in ["both", "positives"] and args.resume and not args.fix_intermediate:
        existing = set(load_all_intermediate_positives(intermediate_dir).keys())
        if existing:
            concepts_to_generate = [c for c in concepts if c not in existing]
            print(f"Resume: Skipping {len(concepts) - len(concepts_to_generate)} existing concepts.")

    print(f"Processing {len(concepts)} concepts (Generation queue: {len(concepts_to_generate)})")

    # Execute phases
    if args.phase in ["both", "positives"]:
        if not concepts_to_generate:
             print("No concepts to generate (all exist or empty list).")
        else:
            await generate_positives(
                cfg=cfg,
                concepts=concepts_to_generate,
                intermediate_dir=intermediate_dir,
                max_concurrent_concepts=args.max_concurrent_concepts,
            )
    
    if args.phase in ["both", "negatives"]:
        gather_negatives(
            cfg=cfg,
            concepts=concepts,
            dataset_root=dataset_root,
            intermediate_dir=intermediate_dir,
        )


if __name__ == "__main__":
    asyncio.run(async_main())