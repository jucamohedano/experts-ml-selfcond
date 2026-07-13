"""
Integrity checks for cached responses and expertise results.

An interrupted run (or interrupted file transfer) leaves the concept it was
working on partially written on disk. Both compute_responses.py and
compute_expertise.py skip concepts whose output merely *exists*, so a partial
concept would otherwise be frozen in its broken state forever and silently
poison downstream statistics (e.g. average precision computed with a truncated
positive set).

The functions here only *check* and report; deletion of incomplete data is
decided by the callers.
"""
import json
import pathlib
import pickle
import typing as t

from selfcond.models import LABELS_FIELD


def expected_sample_count(json_file: pathlib.Path, num_per_concept: int = None) -> int:
    """
    Number of samples ConceptDataset will yield for a concept json, mirroring
    the per-label capping logic of ConceptDataset._load_data (each label
    contributes at most num_per_concept sentences).
    """
    with json_file.open("r") as fp:
        sentences = json.load(fp)["sentences"]
    total = 0
    for label in sentences:
        n = len(sentences[label])
        if num_per_concept is not None:
            n = min(n, num_per_concept)
        total += n
    return total


def check_responses_complete(
    responses_dir: pathlib.Path,
    expected_samples: int,
    batch_size: int = None,
) -> t.Tuple[bool, str]:
    """
    Verify that a concept's cached responses are complete.

    Checks: the directory holds contiguously numbered batch pkls, the final
    pkl (the one an interruption truncates) unpickles, and the total sample
    count across batches equals expected_samples. Batches are written by a
    non-shuffling DataLoader, so every file except the last holds exactly
    batch_size samples. If batch_size is None it is derived from the first
    pkl; passing the current run's batch size instead means responses cached
    with a different batch size are also reported incomplete (and thus get
    recomputed with the requested settings).

    Returns (ok, reason).
    """
    if not responses_dir.exists():
        return False, "no responses directory"
    pkls = sorted(responses_dir.glob("*.pkl"))
    if not pkls:
        return False, "no pkl files"
    n = len(pkls)
    try:
        indices = {int(p.stem) for p in pkls}
    except ValueError:
        return False, "unexpected pkl file names"
    if indices != set(range(n)):
        return False, f"non-contiguous batch numbering ({n} files)"

    if batch_size is None:
        try:
            with pkls[0].open("rb") as fp:
                batch_size = len(pickle.load(fp)[LABELS_FIELD])
        except Exception as e:  # truncated / unreadable first batch
            return False, f"cannot read first batch file {pkls[0].name}: {e}"
    elif not ((n - 1) * batch_size < expected_samples <= n * batch_size):
        return False, (
            f"{n} batches of {batch_size} cannot hold exactly {expected_samples} samples"
        )

    try:
        with pkls[-1].open("rb") as fp:
            n_last = len(pickle.load(fp)[LABELS_FIELD])
    except Exception as e:  # truncated / unreadable last batch
        return False, f"cannot read last batch file {pkls[-1].name}: {e}"

    total = (n - 1) * batch_size + n_last if n > 1 else n_last
    if total != expected_samples:
        return False, f"{total} samples found, expected {expected_samples}"
    return True, "complete"


def check_expertise_complete(
    expertise_dir: pathlib.Path,
    check_csv_rows: bool = False,
) -> t.Tuple[bool, str]:
    """
    Verify that a concept's expertise results are complete.

    ExpertiseResult.save() writes expertise.csv first and expertise_info.json
    last, so a parseable info json implies the csv write finished and a
    truncated json pinpoints an interrupted save. With check_csv_rows=True the
    csv line count is additionally compared against total_neurons from the
    info json, which also catches truncation from causes outside a run (e.g.
    an interrupted file transfer); it reads the whole csv, so reserve it for
    targeted checks rather than every skip decision.

    Returns (ok, reason).
    """
    csv_file = expertise_dir / "expertise.csv"
    info_file = expertise_dir / "expertise_info.json"
    if not csv_file.exists() or not info_file.exists():
        return False, "expertise.csv or expertise_info.json missing"
    try:
        with info_file.open("r") as fp:
            total_neurons = int(json.load(fp)["total_neurons"])
    except Exception as e:
        return False, f"expertise_info.json unreadable: {e}"
    if check_csv_rows:
        with csv_file.open("rb") as fp:
            n_rows = sum(buf.count(b"\n") for buf in iter(lambda: fp.read(1 << 20), b"")) - 1
        if n_rows != total_neurons:
            return False, f"expertise.csv has {n_rows} rows, expected {total_neurons}"
    return True, "complete"


def latest_concept_subdir(model_root: pathlib.Path, subdir: str) -> t.Optional[pathlib.Path]:
    """
    Most recently modified <model_root>/<concept_group>/<concept>/<subdir>
    directory, or None. Used to locate the concept a previous interrupted run
    was working on ('responses' or 'expertise').
    """
    latest, latest_mtime = None, -1.0
    for d in model_root.glob(f"*/*/{subdir}"):
        mtime = d.stat().st_mtime
        if mtime > latest_mtime:
            latest, latest_mtime = d, mtime
    return latest
