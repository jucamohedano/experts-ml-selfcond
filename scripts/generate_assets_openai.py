#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025
#

import argparse
import asyncio
import dataclasses
import json
import os
import pathlib
import random
import re
import shutil
import time
import typing as t

try:
    import openai
except Exception:  # noqa: E722
    openai = None  # type: ignore

from collections import defaultdict

try:
    from nltk.corpus import wordnet as wn
    import nltk
except Exception:  # noqa: E722 - ImportError/LookupError handling combined on purpose
    wn = None  # type: ignore
    nltk = None  # type: ignore


DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous data writer. Output exactly the number of sentences requested,"
    " one per line, with no numbering or bullets. Do not include lists or extra text."
)


FACT_PROMPT_TEMPLATE = (
    "Generate a set of 10 sentences, including as many facts as possible, about the concept"
    " {concept} as {article} {concept_pos} and defined as {definition}. Refer to the concept only as"
    " {concept} without including specific classes, types, or names of {concept}. Make sure the"
    " sentences are diverse and do not repeat."
)


STORY_PROMPT_TEMPLATE = (
    "Generate a set of 10 sentences, where each sentence is a short story about the concept"
    " {concept} as {article} {concept_pos} and defined as {definition}. Refer to the concept only as"
    " {concept} without including specific classes, types, or names of {concept}. Make sure the"
    " sentences are diverse and do not repeat."
)


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
    retries: int = 2
    timeout_s: int = 60
    seed: int = 1234


def load_config(path: pathlib.Path) -> GenerationConfig:
    with path.open("r") as fp:
        cfg = json.load(fp)
    return GenerationConfig(
        base_url=cfg["base_url"],
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
        retries=int(cfg.get("retries", 2)),
        timeout_s=int(cfg.get("timeout_s", 60)),
        seed=int(cfg.get("seed", 1234)),
    )


def ensure_nltk_wordnet() -> None:
    global wn, nltk  # type: ignore
    if wn is None or nltk is None:
        import nltk as _nltk  # lazy import

        nltk = _nltk  # type: ignore
        from nltk.corpus import wordnet as _wn

        wn = _wn  # type: ignore
    try:
        # Access a synset to check availability
        _ = wn.synsets("test")  # type: ignore
    except LookupError:
        assert nltk is not None
        nltk.download("wordnet")
        # Some environments require omw for expanded glosses
        try:
            nltk.download("omw-1.4")
        except Exception:
            pass


def pick_article(word: str) -> str:
    return "an" if len(word) > 0 and word[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def fetch_wordnet_gloss(concept: str) -> str:
    ensure_nltk_wordnet()
    assert wn is not None
    # Prefer noun sense
    synsets = wn.synsets(concept, pos=wn.NOUN)  # type: ignore
    if not synsets:
        synsets = wn.synsets(concept)
    if not synsets:
        return f"{concept} (definition unavailable)"
    return synsets[0].definition()


def normalize_sentence(raw: str) -> str:
    # Strip bullets/numbering
    s = raw.strip()
    s = re.sub(r"^\s*[\-\*\u2022]?\s*(\d+([\.)]|:))?\s*", "", s)
    # Remove surrounding quotes
    s = s.strip('"\'').strip()
    # Collapse whitespace
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


async def chat_completion_stream(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
    api_key: t.Optional[str],
) -> str:
    """Simplified streaming chat completion using OpenAI SDK."""
    if openai is None:
        raise RuntimeError("openai package is required. Please install with: pip install openai")
    
    # Configure client with custom base URL for OpenAI-compatible APIs
    client = openai.AsyncOpenAI(
        api_key=api_key or "not-needed",  # Some APIs might not require a key
        base_url=base_url.rstrip("/") + "/v1",
        timeout=timeout_s,
        max_retries=2,  # Built-in retry logic
    )
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,  # Enable streaming
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        
        # Collect streaming response
        chunks = []
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        
        return "".join(chunks)
        
    except openai.APIError as e:
        # Handle OpenAI-specific errors
        raise RuntimeError(f"OpenAI API error: {e}")
    except Exception as e:
        # Handle other errors (timeout, network, etc.)
        raise RuntimeError(f"Request failed: {e}")


def render_prompt(template: str, concept: str, concept_pos: str, definition: str) -> str:
    article = pick_article(concept)
    return template.format(
        concept=concept,
        article=article,
        concept_pos=concept_pos,
        definition=definition,
    )


def split_into_lines(text: str) -> t.List[str]:
    lines = [normalize_sentence(x) for x in text.strip().splitlines()]
    # Some models emit paragraphs; split on semicolons as a fallback
    normalized: t.List[str] = []
    for line in lines:
        if not line:
            continue
        if ";" in line and len(line) > 140:
            normalized.extend([normalize_sentence(y) for y in line.split(";")])
        else:
            normalized.append(line)
    # Keep non-empty
    normalized = [x for x in normalized if x]
    return normalized


async def batched_generate_async(
    *,
    cfg: GenerationConfig,
    concept: str,
    concept_pos: str,
    definition: str,
    template: str,
    target_count: int,
    api_key: t.Optional[str],
) -> t.List[str]:
    """Generate sentences until target count is reached, handling failures gracefully."""
    random_state = random.Random(cfg.seed)
    sentences: t.List[str] = []
    max_attempts = target_count * 3  # Safety limit to prevent infinite loops
    attempt_count = 0

    async def generate_single_batch() -> t.List[str]:
        prompt = render_prompt(template, concept, concept_pos, definition)
        
        # The OpenAI SDK handles retries internally, but we can add our own for robustness
        backoff_s = 1.0
        for attempt in range(cfg.retries + 1):
            try:
                content = await chat_completion_stream(
                    base_url=cfg.base_url,
                    model=cfg.model,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_tokens,
                    timeout_s=cfg.timeout_s,
                    api_key=api_key,
                )
                return split_into_lines(content)
            except Exception as exc:  # noqa: BLE001
                if attempt < cfg.retries:
                    await asyncio.sleep(backoff_s)
                    backoff_s *= 2.0
                else:
                    print(f"    Failed batch for {concept} ({template.split()[2]}): {exc}")
                    return []  # Return empty list instead of raising
        return []
    
    # Keep generating until we have enough sentences
    print(f"    Generating {template.split()[2]} sentences for {concept} (target: {target_count})")
    
    while len(sentences) < target_count and attempt_count < max_attempts:
        # Calculate how many more batches we might need
        remaining = target_count - len(sentences)
        estimated_batches = max(1, (remaining + cfg.batch_size - 1) // cfg.batch_size)
        
        # Generate multiple batches concurrently, but not too many to avoid overwhelming
        concurrent_batches = min(estimated_batches, 5)  # Limit concurrent requests
        
        batch_tasks = [generate_single_batch() for _ in range(concurrent_batches)]
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Process results
        new_sentences = []
        for batch_lines in batch_results:
            if batch_lines:  # Only process non-empty results
                new_sentences.extend(batch_lines)
        
        if new_sentences:
            # Normalize new sentences
            normalized_new = [normalize_sentence(s) for s in new_sentences if s]
            # Add only unique sentences
            for sent in normalized_new:
                if sent and sent not in sentences:
                    sentences.append(sent)
            
            print(f"      Progress: {len(sentences)}/{target_count} sentences")
        else:
            print(f"      No sentences generated in this batch, retrying...")
            await asyncio.sleep(2.0)  # Brief pause before retrying
        
        attempt_count += concurrent_batches
    
    if len(sentences) < target_count:
        print(f"    WARNING: Only generated {len(sentences)}/{target_count} sentences for {concept} after {attempt_count} attempts")
    
    # Shuffle for diversity and return exactly target_count (or all we have)
    random_state.shuffle(sentences)
    sentences = unique_preserve_order(sentences)
    return sentences[:target_count]


def build_negatives(
    *,
    positives_by_concept: t.Dict[str, t.List[str]],
    target_negatives: int,
    seed: int,
) -> t.Dict[str, t.List[str]]:
    concepts = list(positives_by_concept.keys())
    rng = random.Random(seed)
    negatives: t.Dict[str, t.List[str]] = {}
    for c in concepts:
        pool: t.List[str] = []
        for other in concepts:
            if other == c:
                continue
            pool.extend(positives_by_concept.get(other, []))
        if not pool:
            negatives[c] = []
            continue
        if len(pool) >= target_negatives:
            negatives[c] = rng.sample(pool, target_negatives)
        else:
            # Sample with replacement if pool too small
            negatives[c] = [rng.choice(pool) for _ in range(target_negatives)]
    return negatives


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
        # Single-line compact JSON to match existing format
        json.dump(out, fp, ensure_ascii=False, separators=(',', ':'))


def write_concept_list_csv(*, dataset_root: pathlib.Path, group: str, concepts: t.List[str]) -> None:
    csv_path = dataset_root / "concept_list.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as fp:
        fp.write("group,concept\n")
        for c in concepts:
            fp.write(f"{group},{c}\n")


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


def write_intermediate_positives(
    *,
    intermediate_dir: pathlib.Path,
    concept: str,
    positives: t.List[str],
) -> None:
    """Write intermediate positives for phase 1."""
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    out = {"concept": concept, "positives": positives}
    with (intermediate_dir / f"{concept}.json").open("w") as fp:
        json.dump(out, fp, ensure_ascii=False)


def load_all_intermediate_positives(
    intermediate_dir: pathlib.Path,
) -> t.Dict[str, t.List[str]]:
    """Load all intermediate positives for phase 2."""
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


async def generate_concept_positives(
    *,
    cfg: GenerationConfig,
    concept: str,
    api_key: t.Optional[str],
) -> t.List[str]:
    """Generate positives for a single concept."""
    definition = fetch_wordnet_gloss(concept)
    concept_pos = "noun"

    # Generate fact and story sentences concurrently
    fact_task = batched_generate_async(
        cfg=cfg,
        concept=concept,
        concept_pos=concept_pos,
        definition=definition,
        template=FACT_PROMPT_TEMPLATE,
        target_count=cfg.positive_fact,
        api_key=api_key,
    )
    
    story_task = batched_generate_async(
        cfg=cfg,
        concept=concept,
        concept_pos=concept_pos,
        definition=definition,
        template=STORY_PROMPT_TEMPLATE,
        target_count=cfg.positive_story,
        api_key=api_key,
    )
    
    fact_sentences, story_sentences = await asyncio.gather(fact_task, story_task)
    positives = unique_preserve_order(fact_sentences + story_sentences)
    print(f"  Total positives for {concept}: {len(positives)} (fact: {len(fact_sentences)}, story: {len(story_sentences)})")
    return positives


async def phase1_generate_positives(
    *,
    cfg: GenerationConfig,
    concepts: t.List[str],
    intermediate_dir: pathlib.Path,
    max_concurrent_concepts: int,
    api_key: t.Optional[str],
) -> None:
    """Phase 1: Generate positives for all concepts with high concurrency."""
    if openai is None:
        raise RuntimeError("openai package is required. Please install with: pip install openai")

    # Process concepts in batches to avoid overwhelming the server
    print(f"Phase 1: Generating positives for {len(concepts)} concepts with max {max_concurrent_concepts} concurrent concepts...")
    
    for i in range(0, len(concepts), max_concurrent_concepts):
        batch_concepts = concepts[i:i + max_concurrent_concepts]
        
        # Create tasks for this batch of concepts
        tasks = []
        for concept in batch_concepts:
            task = generate_concept_positives(
                cfg=cfg,
                concept=concept,
                api_key=api_key,
            )
            tasks.append((concept, task))
        
        # Execute this batch concurrently
        print(f"  Processing batch {i//max_concurrent_concepts + 1}: {len(batch_concepts)} concepts...")
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Save results
        for (concept, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                print(f"    ERROR for {concept}: {result}")
                continue
            
            positives = result
            write_intermediate_positives(
                intermediate_dir=intermediate_dir,
                concept=concept,
                positives=positives,
            )
            print(f"    Generated {len(positives)} positives for {concept}")


def phase2_build_negatives_and_write(
    *,
    cfg: GenerationConfig,
    concepts: t.List[str],
    dataset_root: pathlib.Path,
    intermediate_dir: pathlib.Path,
) -> None:
    """Phase 2: Build negatives and write final JSON files."""
    print("Phase 2: Loading intermediate positives...")
    positives_by_concept = load_all_intermediate_positives(intermediate_dir)
    
    if not positives_by_concept:
        raise RuntimeError("No intermediate positives found. Run phase 1 first.")
    
    print(f"Phase 2: Building negatives for {len(concepts)} concepts...")
    negatives_by_concept = build_negatives(
        positives_by_concept=positives_by_concept,
        target_negatives=cfg.negatives_per_concept,
        seed=cfg.seed,
    )

    # Write final JSON files
    group = "custom"
    source_name = "openai_compat_qwen3_8b_fp8"
    
    for concept in concepts:
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
        print(f"  Wrote {len(positives)} positives, {len(negatives)} negatives for {concept}")

    # Write concept list CSV
    write_concept_list_csv(dataset_root=dataset_root, group=group, concepts=concepts)
    
    # Clean up intermediate directory
    if intermediate_dir.exists():
        shutil.rmtree(intermediate_dir)
        print(f"  Cleaned up intermediate directory: {intermediate_dir}")
    
    print(f"Phase 2: Complete. Dataset written to: {dataset_root}")


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Generate assets using OpenAI-compatible API")
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
        help="Root directory to place the generated dataset folder",
    )
    parser.add_argument(
        "--only-concepts",
        type=str,
        default="",
        help="Comma-separated subset of concepts to generate (overrides config concepts)",
    )
    parser.add_argument(
        "--limit-concepts",
        type=int,
        default=0,
        help="Limit to the first N concepts (after filtering)",
    )
    parser.add_argument(
        "--max-concurrent-concepts",
        type=int,
        default=10,
        help="Max concepts to process concurrently (default: 10)",
    )
    parser.add_argument(
        "--phase",
        choices=["both", "positives", "negatives"],
        default="both",
        help="Run both phases, only positives, or only negatives (default: both)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg.seed)

    # Load .env if present
    load_env_file_if_present(pathlib.Path(".env"))

    dataset_root = args.dataset_root / cfg.dataset_name
    intermediate_dir = dataset_root / "custom" / ".intermediate"

    # Prefer OPENAI_API_KEY, fallback to OPEN_API_KEY per user request
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_API_KEY")

    # Apply concept subset filters from CLI
    concepts = cfg.concepts
    if args.only_concepts:
        requested = [c.strip() for c in args.only_concepts.split(",") if c.strip()]
        concepts = [c for c in concepts if c in requested]
    if args.limit_concepts and args.limit_concepts > 0:
        concepts = concepts[: args.limit_concepts]

    # Execute phases based on --phase argument
    if args.phase in ["both", "positives"]:
        await phase1_generate_positives(
            cfg=cfg,
            concepts=concepts,
            intermediate_dir=intermediate_dir,
            max_concurrent_concepts=args.max_concurrent_concepts,
            api_key=api_key,
        )
    
    if args.phase in ["both", "negatives"]:
        phase2_build_negatives_and_write(
            cfg=cfg,
            concepts=concepts,
            dataset_root=dataset_root,
            intermediate_dir=intermediate_dir,
        )


if __name__ == "__main__":
    asyncio.run(async_main())