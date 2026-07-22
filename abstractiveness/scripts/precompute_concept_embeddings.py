"""
Precompute per-concept, per-layer embeddings from the cached model responses.

A concept embedding at layer L is the mean, over that concept's positive sentences,
of the max-pooled activation vector stored in the response pkls. That vector is the
same hidden-layer representation expertise is derived from, so no model inference is
needed here, only a reduction of what is already on disk.

Run once per model, OUTSIDE the AP sweep: embeddings do not depend on the AP
threshold, so recomputing them per threshold would repeat the read five times.

    python scripts/precompute_concept_embeddings.py --config gpt2_richie_hsj
    python scripts/precompute_concept_embeddings.py --config gpt2_richie_hsj --validate 5

Why this does not reuse selfcond.responses.read_responses_from_cached: that loader
materializes every layer and every file of a concept before returning (about 0.5 GB
for GPT-2 and 3.5 GB for Qwen3), so parallel workers would exhaust memory, and it
cannot skip the files that hold no positive sentences.

Two properties of the stored data make this cheap, and both are verified at runtime
rather than assumed:
  * Sentences are written negatives-first, so the positive sentences occupy a
    contiguous tail of the sorted file order (the last 50 of 175 files). Walking
    backward and stopping at the first fully-negative file reads about 29% of the
    bytes.
  * Only a running sum per layer is needed, so memory stays at one batch plus the
    accumulator regardless of how many sentences a concept has.

Output: assets/concept_embeddings_<config>.npz plus a .json sidecar with provenance.
Sums are stored per half rather than as a single mean so that module 8 can compute a
split-half noise ceiling per layer. The mean is recovered as
(sum_even + sum_odd) / (n_even + n_odd). Halves split on the parity of the row index
WITHIN each batch, which interleaves them at fine grain so that any drift across the
sentence set lands in both halves equally.
"""
import argparse
import json
import logging
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from abstractiveness_executor import MODEL_CONFIGS
from utils.helpers import init_global_layer_mapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

LABELS_FIELD = "labels"


# ---------------------------------------------------------------------------
# 1. Discovery
# ---------------------------------------------------------------------------

def discover_concept_dirs(responses_dir: pathlib.Path, model_subdir: str) -> dict:
    """
    Map concept name to its folder. Folder names are storage keys, which for
    ambiguous words carry a category suffix (``squash__vegetables``), so the concept
    identity is read from the folder's expertise.csv rather than taken from the
    folder name. Globs for the responses folder instead of assuming a fixed depth,
    because the model subdirectory nests differently per architecture
    (``gpt2/custom/<c>`` against ``Qwen/Qwen3-1.7B/custom/<c>``).
    """
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
    """
    Raw layer strings in layer_idx order, and the offset of each layer on the flat
    unit axis. The raw strings keep their ``:0`` suffix because that is what joins
    them to the ``layer`` column of layer_mapping_*.csv and to the sublayer regexes
    in LAYER_ARCHITECTURES.
    """
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


# ---------------------------------------------------------------------------
# 2. Per-concept extraction
# ---------------------------------------------------------------------------

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
    """
    Split-half positive sums for one concept, streaming one pkl at a time.

    Reads the positive tail only unless ``full_read``, and self-validates that
    assumption: the file the backward walk stops on and the first file of the run
    must both be entirely negative. On any violation the concept is re-read in full
    and the result is flagged, so a change in how sentences are ordered degrades to
    correct-but-slower rather than silently wrong.
    """
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

        # The walk must have terminated on a fully negative file that is not the
        # last file, otherwise the positives were not a contiguous tail.
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


def embed_from_expertise_csv(concept: str, concept_root: pathlib.Path, layers: list,
                             offsets: np.ndarray) -> dict:
    """
    Development-speed alternative: the ``on_p50`` column of expertise.csv is exactly
    the median over positive sentences, one row per unit, and it correlates with the
    positive mean at about 0.97 to 0.99. Reading it takes well under a second per
    concept against roughly 2 seconds of pkl reads, which makes iterating on the
    cache layout cheap. It is a median rather than a mean, so it is a robustness
    variant and not the canonical embedding.

    Both halves are set to the same vector, which makes the split-half noise ceiling
    degenerate. Module 8 reports no ceiling for caches built this way.
    """
    expertise = pd.read_csv(concept_root / "expertise" / "expertise.csv",
                            usecols=["layer", "unit", "on_p50"])
    vector = np.zeros(offsets[-1], dtype=np.float64)
    by_layer = {name: group for name, group in expertise.groupby("layer", sort=False)}
    for i, layer in enumerate(layers):
        group = by_layer.get(layer)
        if group is None:
            return {"concept": concept, "error": f"layer {layer} absent from expertise.csv"}
        vector[offsets[i]:offsets[i + 1]] = group.sort_values("unit")["on_p50"].to_numpy()
    half = (vector / 2.0).astype(np.float32)
    return {"concept": concept, "sum_even": half, "sum_odd": half, "n_even": 1, "n_odd": 1,
            "files_read": 0, "n_files": 0, "tail_scan_validated": True}


# ---------------------------------------------------------------------------
# 3. Cache assembly
# ---------------------------------------------------------------------------

def build_embedding_cache(config_key: str, out_path: pathlib.Path, source: str = "pkl_mean",
                          n_jobs: int = 8, validate: int = 0, limit: int = 0) -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cfg = MODEL_CONFIGS[config_key]
    responses_dir = repo_root / "responses" / cfg["responses_subdir"]

    concept_dirs = discover_concept_dirs(responses_dir, cfg["model_subdir"])
    if limit:
        concept_dirs = dict(list(concept_dirs.items())[:limit])
    if not concept_dirs:
        raise RuntimeError(f"No concepts found under {responses_dir / cfg['model_subdir']}")
    log.info(f"Found {len(concept_dirs)} concepts for config '{config_key}'")

    first_root = next(iter(concept_dirs.values()))
    sample_pkl = sorted((first_root / "responses").glob("*.pkl"))[0]
    layers, offsets = layer_axis(responses_dir, cfg["model_subdir"],
                                 repo_root / "assets" / cfg["layer_mapping_file"],
                                 cfg["architecture"], sample_pkl)
    log.info(f"{len(layers)} layers, {offsets[-1]} units total per concept")

    worker = embed_one_concept if source == "pkl_mean" else embed_from_expertise_csv
    started = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(worker)(concept, root, layers, offsets)
        for concept, root in concept_dirs.items())
    elapsed = time.time() - started

    failed = [r["concept"] for r in results if "error" in r]
    for r in results:
        if "error" in r:
            log.error(f"{r['concept']}: {r['error']}")
    results = [r for r in results if "error" not in r]
    if not results:
        raise RuntimeError("Every concept failed, refusing to write an empty cache")

    concepts = [r["concept"] for r in results]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
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
        "config": config_key,
        "source": source,
        "responses_subdir": cfg["responses_subdir"],
        "model_subdir": cfg["model_subdir"],
        "architecture": cfg["architecture"],
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
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))

    size_mb = out_path.stat().st_size / 1e6
    log.info(f"Wrote {out_path} ({size_mb:.0f} MB) for {len(concepts)} concepts in {elapsed:.0f}s")
    if not all_validated:
        log.warning(f"Tail scan fell back to a full read for "
                    f"{len(sidecar['concepts_not_validated'])} concepts")

    if validate and source == "pkl_mean":
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, choices=sorted(MODEL_CONFIGS),
                        help="key of MODEL_CONFIGS to build the cache for")
    parser.add_argument("--source", default="pkl_mean", choices=["pkl_mean", "expertise_p50"],
                        help="pkl_mean is canonical; expertise_p50 is the fast development path")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="concepts processed in parallel (memory scales with this)")
    parser.add_argument("--validate", type=int, default=0,
                        help="re-read N concepts in full and require identical results")
    parser.add_argument("--limit", type=int, default=0, help="only the first N concepts, for smoke tests")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="override the output path")
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    suffix = "" if args.source == "pkl_mean" else f"_{args.source}"
    out = args.out or repo_root / "assets" / f"concept_embeddings_{args.config}{suffix}.npz"

    build_embedding_cache(args.config, out, source=args.source, n_jobs=args.n_jobs,
                          validate=args.validate, limit=args.limit)
