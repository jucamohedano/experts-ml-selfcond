"""
Building and reading the per-concept, per-layer embedding cache consumed by module 4.

A concept's embedding at a layer is its mean max-pooled activation over its positive sentences, read
from the response pkls and stored as split-half sums so module 4 gets a noise ceiling. The executor
calls load_or_build_concept_embeddings once per run when module 4 is enabled, which builds the cache
into assets/ when it is missing.

Explanations: documentation/module_4_embedding_rsa.md.
"""

import json
import logging
import pathlib
import pickle
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from core.expert_data_loading import init_global_layer_mapping

log = logging.getLogger(__name__)

LABELS_FIELD = "labels"


def discover_concept_dirs(responses_dir: pathlib.Path, model_subdir: str) -> dict:
    """Map concept name to its folder."""
    concept_dirs = {}
    for responses_path in sorted((responses_dir / model_subdir).glob("**/custom/*/responses")):
        concept_root = responses_path.parent
        expertise_csv = concept_root / "expertise" / "expertise.csv"
        if not expertise_csv.exists():
            log.warning(f"No expertise.csv beside {responses_path}, skipping")
            continue
        concept = str(pd.read_csv(expertise_csv, usecols=["concept"], nrows=1)["concept"].iloc[0])
        if concept in concept_dirs:
            log.warning(f"Duplicate concept '{concept}' at {concept_root}, keeping the first")
            continue
        concept_dirs[concept] = concept_root
    return dict(sorted(concept_dirs.items()))


def layer_axis(responses_dir: pathlib.Path, model_subdir: str, mapping_path: pathlib.Path,
               architecture: str, sample_pkl: pathlib.Path) -> tuple:
    """Raw layer strings in layer_idx order, and the offset of each layer on the flat unit axis."""
    mapping = init_global_layer_mapping(responses_dir, model_subdir, mapping_path, architecture)
    layers = list(mapping.sort_values("layer_idx")["layer"])

    with sample_pkl.open("rb") as fp:
        batch = pickle.load(fp)
    missing = [l for l in layers if l not in batch]
    if missing:
        raise ValueError(f"Layer mapping lists layers absent from the responses: {missing[:5]}")

    dims = [batch[l].shape[1] for l in layers]
    offsets = np.concatenate([[0], np.cumsum(dims)]).astype(np.int64)
    return layers, offsets


def _accumulate(batch: dict, layers: list, offsets: np.ndarray,
                sum_even: np.ndarray, sum_odd: np.ndarray) -> tuple:
    """Add one batch's positive rows into the split-half accumulators."""
    positive_rows = np.flatnonzero(np.asarray(batch[LABELS_FIELD]) == 1)
    if positive_rows.size == 0:
        return 0, 0
    even, odd = positive_rows[0::2], positive_rows[1::2]
    for i, layer in enumerate(layers):
        activations = batch[layer]
        lo, hi = offsets[i], offsets[i + 1]
        if even.size:
            sum_even[lo:hi] += activations[even].sum(axis=0, dtype=np.float64)
        if odd.size:
            sum_odd[lo:hi] += activations[odd].sum(axis=0, dtype=np.float64)
    return even.size, odd.size


def embed_one_concept(concept: str, concept_root: pathlib.Path, layers: list,
                      offsets: np.ndarray, full_read: bool = False) -> dict:
    """Split-half positive sums for one concept, streaming one pkl at a time."""
    files = sorted((concept_root / "responses").glob("*.pkl"))
    if not files:
        return {"concept": concept, "error": "no response pkls"}

    total = offsets[-1]
    sum_even = np.zeros(total, dtype=np.float64)
    sum_odd = np.zeros(total, dtype=np.float64)
    n_even = n_odd = 0
    validated = True
    read_files = 0

    def load(path):
        with path.open("rb") as fp:
            return pickle.load(fp)

    if not full_read:
        stop_index = None
        for position in range(len(files) - 1, -1, -1):
            batch = load(files[position])
            read_files += 1
            labels = np.asarray(batch[LABELS_FIELD])
            if not (labels == 1).any():
                stop_index = position
                break
            e, o = _accumulate(batch, layers, offsets, sum_even, sum_odd)
            n_even += e
            n_odd += o

        if stop_index is None or stop_index == len(files) - 1:
            validated = False
        elif stop_index > 0:
            earlier = np.asarray(load(files[stop_index - 1])[LABELS_FIELD])
            read_files += 1
            if (earlier == 1).any():
                validated = False
        if (n_even + n_odd) == 0:
            validated = False

    if full_read or not validated:
        if not full_read:
            log.warning(f"{concept}: positive sentences are not a contiguous tail, re-reading in full")
        sum_even[:] = 0.0
        sum_odd[:] = 0.0
        n_even = n_odd = 0
        read_files = 0
        for path in files:
            e, o = _accumulate(load(path), layers, offsets, sum_even, sum_odd)
            read_files += 1
            n_even += e
            n_odd += o

    if (n_even + n_odd) == 0:
        return {"concept": concept, "error": "no positive sentences"}

    return {
        "concept": concept,
        "sum_even": sum_even.astype(np.float32),
        "sum_odd": sum_odd.astype(np.float32),
        "n_even": n_even,
        "n_odd": n_odd,
        "files_read": read_files,
        "n_files": len(files),
        "tail_scan_validated": bool(validated and not full_read),
    }


def build_concept_embeddings(cache_path: pathlib.Path, responses_dir: pathlib.Path, model_subdir: str,
                             mapping_path: pathlib.Path, architecture: str, n_jobs: int = 8,
                             validate: int = 0) -> None:
    """Write the cache from the response pkls: split-half positive sums per concept and layer, plus a sidecar."""
    concept_dirs = discover_concept_dirs(responses_dir, model_subdir)
    if not concept_dirs:
        raise RuntimeError(f"No concepts found under {responses_dir / model_subdir}")
    log.info(f"Building the concept embedding cache for {len(concept_dirs)} concepts from {responses_dir}")

    first_root = next(iter(concept_dirs.values()))
    sample_pkl = sorted((first_root / "responses").glob("*.pkl"))[0]
    layers, offsets = layer_axis(responses_dir, model_subdir, mapping_path, architecture, sample_pkl)
    log.info(f"{len(layers)} layers, {offsets[-1]} units total per concept")

    started = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(embed_one_concept)(concept, root, layers, offsets) for concept, root in concept_dirs.items())
    elapsed = time.time() - started

    failed = [r["concept"] for r in results if "error" in r]
    for r in results:
        if "error" in r:
            log.error(f"{r['concept']}: {r['error']}")
    results = [r for r in results if "error" not in r]
    if not results:
        raise RuntimeError("Every concept failed, refusing to write an empty cache")

    concepts = [r["concept"] for r in results]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        concepts=np.array(concepts),
        layers=np.array(layers),
        offsets=offsets,
        sum_even=np.vstack([r["sum_even"] for r in results]),
        sum_odd=np.vstack([r["sum_odd"] for r in results]),
        n_even=np.array([r["n_even"] for r in results], dtype=np.int32),
        n_odd=np.array([r["n_odd"] for r in results], dtype=np.int32),
    )
    all_validated = all(r["tail_scan_validated"] for r in results)
    sidecar = {
        "responses_dir": str(responses_dir),
        "model_subdir": model_subdir,
        "architecture": architecture,
        "n_concepts": len(concepts),
        "n_layers": len(layers),
        "total_units": int(offsets[-1]),
        "tail_scan_validated": all_validated,
        "concepts_not_validated": [r["concept"] for r in results if not r["tail_scan_validated"]],
        "positives_per_concept": {r["concept"]: r["n_even"] + r["n_odd"] for r in results},
        "files_read_per_concept": {r["concept"]: r["files_read"] for r in results},
        "failed_concepts": failed,
        "elapsed_seconds": round(elapsed, 1),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cache_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    log.info(f"Wrote {cache_path} ({cache_path.stat().st_size / 1e6:.0f} MB) for {len(concepts)} concepts "
             f"in {elapsed:.0f}s")
    if not all_validated:
        log.warning(f"Tail scan fell back to a full read for {len(sidecar['concepts_not_validated'])} concepts")
    if validate:
        _validate_against_full_read(concept_dirs, layers, offsets, results, validate)


def _validate_against_full_read(concept_dirs: dict, layers: list, offsets: np.ndarray,
                                results: list, n_check: int) -> None:
    """Re-read N random concepts in full and require bitwise agreement with the tail scan."""
    rng = np.random.default_rng(42)
    by_concept = {r["concept"]: r for r in results}
    sample = rng.choice(list(by_concept), size=min(n_check, len(by_concept)), replace=False)
    log.info(f"Validating the tail scan against a full read for: {', '.join(sample)}")

    failures = 0
    for concept in sample:
        full = embed_one_concept(concept, concept_dirs[concept], layers, offsets, full_read=True)
        tail = by_concept[concept]
        same = (np.array_equal(full["sum_even"], tail["sum_even"])
                and np.array_equal(full["sum_odd"], tail["sum_odd"])
                and full["n_even"] == tail["n_even"] and full["n_odd"] == tail["n_odd"])
        log.info(f"  {concept}: {'identical' if same else 'MISMATCH'} "
                 f"(tail read {tail['files_read']}/{tail['n_files']} files, "
                 f"{tail['n_even'] + tail['n_odd']} positives)")
        failures += 0 if same else 1

    if failures:
        raise RuntimeError(f"Tail scan disagreed with a full read for {failures} concepts")
    log.info("Tail scan validated, results are identical to a full read")


def load_concept_embeddings(cache_path: pathlib.Path) -> dict:
    """Load the concept embedding cache, None when it is missing."""
    cache_path = pathlib.Path(cache_path)
    if not cache_path.exists():
        log.warning(f"Concept embedding cache not found at {cache_path}, module 4 will be skipped.")
        return None

    data = np.load(cache_path, allow_pickle=False)
    n_even = data["n_even"].astype(np.float64)[:, None]
    n_odd = data["n_odd"].astype(np.float64)[:, None]

    sidecar = cache_path.with_suffix(".json")
    provenance = json.loads(sidecar.read_text()) if sidecar.exists() else {}

    return {
        "concepts": [str(c) for c in data["concepts"]],
        "layers": [str(l) for l in data["layers"]],
        "offsets": data["offsets"],
        "mean": (data["sum_even"] + data["sum_odd"]) / (n_even + n_odd),
        "half_a": data["sum_even"] / n_even,
        "half_b": data["sum_odd"] / n_odd,
        "provenance": provenance,
    }


def load_or_build_concept_embeddings(cache_path: pathlib.Path, responses_dir: pathlib.Path, model_subdir: str,
                                     mapping_path: pathlib.Path, architecture: str, n_jobs: int = 8) -> dict:
    """The embedding cache, built first from the response pkls when it does not exist yet."""
    cache_path = pathlib.Path(cache_path)
    if not cache_path.exists():
        log.info(f"Concept embedding cache missing at {cache_path}, building it (about 5 minutes on GPT-2, "
                 f"30 on Qwen3)...")
        build_concept_embeddings(cache_path, responses_dir, model_subdir, mapping_path, architecture, n_jobs)
    return load_concept_embeddings(cache_path)


def layer_slice(embedding_cache: dict, layer: str) -> slice:
    """Column slice of one raw layer string within the cache's flat unit axis."""
    i = embedding_cache["layers"].index(layer)
    return slice(int(embedding_cache["offsets"][i]), int(embedding_cache["offsets"][i + 1]))
