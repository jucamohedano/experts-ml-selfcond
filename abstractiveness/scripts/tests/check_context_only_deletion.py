"""
The context-only dataset marks every concept word form its rule names, and selfcond deletes those tokens.

Checks:
  1. Forms: the frequency rules decide the forms (irregular plurals, listed singulars, single forms),
     matching is case-insensitive and whole-word, possessives and hyphenated compounds lose the word.
  2. Rule: positives lose their own concept, negatives the target concept and the concept they were
     drawn from, other concept words stay, a negative drawn from no concept raises an error.
  3. Build: the written files keep their sentences, carry spans aligned with them, copy the concept list.
  4. Real dataset: no form of the target or of a negative's source survives the deletion, with the
     share of positives and negatives losing a word reported.
  5. Tokenizer: the tokens a span overlaps are deleted and the rest close up with attention mask 1,
     and ConceptDataset applies each label's spans in its negatives-first order.
  6. Text baseline: the positive documents of a context-only dataset lose the deleted spans.

Usage, from scripts/:
    python tests/check_context_only_deletion.py
"""
import contextlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT))
from core.data_preparation import build_context_only_dataset as ctx
from core.data_preparation.frequency_covariates import inflected_forms
from core.probing_sentences import load_positive_documents
from selfcond.data import ConceptDataset, PytorchTransformersTokenizer
from _report import run

DATASET = REPO_ROOT / "abstractiveness/assets/Qwen3-30B-A3B-Instruct-2507_abstractiveness_Richie_HSJ_cot"
PAD = 50256

FORM_CASES = [
    ("policeman", "The policemen arrived.", [[4, 13]]),
    ("boots", "Two boots, one boot.", [[4, 9]]),
    ("birds", "Birds sing, a bird listens.", [[0, 5], [14, 18]]),
    ("bus", "Two buses passed a bus.", [[4, 9], [19, 22]]),
    ("penguin", "The Penguin's beak and penguin-like gait.", [[4, 11], [23, 30]]),
    ("penguin", "Penguins swim.", [[0, 8]]),
    ("chess", "Chess960 and chess.", [[0, 5], [13, 18]]),
    ("birds", "A blackbird sat on the carpet.", []),
    ("car", "A blackbird sat on the carpet.", []),
]

A = "The penguin swam past a chair."
B = "A chair stood near the penguin pool."
C = "Birds rested on the chair."
EXPECTED_SPANS = {
    "penguin": {"positive": [[[4, 11]]], "negative": [[[2, 7], [23, 30]], [[0, 5]]]},
    "chair": {"positive": [[[2, 7]]], "negative": [[[4, 11], [24, 29]]]},
    "birds": {"positive": [[[0, 5]]], "negative": [[[4, 11]]]},
}

S1, S2 = "Penguins waddled near the chair.", "A chair by the penguin pool."
S1_IDS = [47, 13561, 1040, 266, 2860, 992, 1474, 262, 5118, 13]


def fixture() -> dict:
    """Three concepts whose negatives are each other's positives."""
    def concept(key, category, positives, negatives):
        return {"concept": key, "word": key, "category": category, "group": "custom",
                "source": "fixture", "sentences": {"positive": positives, "negative": negatives}}
    return {"penguin": concept("penguin", "birds", [A], [B, C]),
            "chair": concept("chair", "furniture", [B], [A]),
            "birds": concept("birds", None, [C], [A])}


def write_dataset(root: pathlib.Path, concepts: dict) -> None:
    """Concept files and concept list in the layout of the stimulus datasets."""
    (root / "custom").mkdir(parents=True)
    for key, data in concepts.items():
        (root / "custom" / f"{key}.json").write_text(json.dumps(data), encoding="utf-8")
    rows = "".join(f"custom,{k},{d['word']},{d['category'] or ''}\n" for k, d in concepts.items())
    (root / "concept_list.csv").write_text("group,concept,word,category\n" + rows, encoding="utf-8")


def cut(text: str, spans: list) -> str:
    """The text with its spans removed."""
    for start, end in reversed(spans):
        text = text[:start] + text[end:]
    return text


def check_forms(failures) -> str:
    """Forms follow the frequency rules and match whole words in any case."""
    for word, text, want in FORM_CASES:
        got = ctx.match_spans(text, [ctx.form_pattern(word)])
        if got != want:
            failures.append(f"{word} in {text!r}: spans {got}, expected {want}")
    return f"{len(FORM_CASES)} form cases"


def check_rule(failures) -> str:
    """Positives lose their own word, negatives the target's and their source's."""
    out = ctx.context_only_concepts(fixture())
    for key, want in EXPECTED_SPANS.items():
        got = out[key].get("delete_spans")
        if got != want:
            failures.append(f"{key}: spans {got}, expected {want}")
    stray = fixture()
    stray["chair"]["sentences"]["negative"].append("A sentence no concept wrote.")
    try:
        ctx.context_only_concepts(stray)
        failures.append("a negative drawn from no concept did not raise")
    except ValueError:
        pass
    return "rule on 3 fixture concepts"


def check_build(failures) -> str:
    """The written dataset keeps the sentences and adds aligned spans."""
    with tempfile.TemporaryDirectory() as tmp:
        src, out = pathlib.Path(tmp) / "src", pathlib.Path(tmp) / "out"
        concepts = fixture()
        write_dataset(src, concepts)
        ctx.build(src, out)
        for key, data in concepts.items():
            written = json.loads((out / "custom" / f"{key}.json").read_text(encoding="utf-8"))
            if written["sentences"] != data["sentences"]:
                failures.append(f"{key}: sentences changed")
            if written.get("delete_spans") != EXPECTED_SPANS[key]:
                failures.append(f"{key}: written spans {written.get('delete_spans')}")
        if (out / "concept_list.csv").read_text() != (src / "concept_list.csv").read_text():
            failures.append("concept_list.csv not copied")
    return "build writes 3 files"


def check_real_dataset(failures) -> str:
    """No target or source form survives on the stimulus dataset."""
    concepts = {}
    for path in sorted((DATASET / "custom").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        concepts[data["concept"]] = data
    sources = {}
    for key, data in concepts.items():
        for sentence in data["sentences"]["positive"]:
            sources.setdefault(sentence, set()).add(key)
    out = ctx.context_only_concepts(concepts)
    deleted = {"positive": [0, 0], "negative": [0, 0]}
    for key, data in out.items():
        target = inflected_forms(concepts[key]["word"])
        for label in ("positive", "negative"):
            texts, spans = data["sentences"][label], data["delete_spans"][label]
            if len(texts) != len(spans):
                failures.append(f"{key} {label}: {len(spans)} span lists for {len(texts)} sentences")
                continue
            for text, text_spans in zip(texts, spans):
                forms = set(target)
                if label == "negative":
                    for source in sources[text]:
                        forms |= inflected_forms(concepts[source]["word"])
                left = set(re.findall(r"[a-z]+", cut(text, text_spans).lower())) & forms
                if left:
                    failures.append(f"{key} {label}: {sorted(left)} left in {text!r}")
                if concepts[key]["category"]:
                    deleted[label][0] += bool(text_spans)
                    deleted[label][1] += 1
    share = {label: hit / total for label, (hit, total) in deleted.items()}
    return (f"{len(concepts)} concepts, no form left, members lose a word in {share['positive']:.1%} "
            f"of positives and {share['negative']:.1%} of negatives")


def check_tokenizer(failures) -> str:
    """Deleted tokens leave the input, ConceptDataset keeps the label order."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tokenizer = PytorchTransformersTokenizer("gpt2", cache_dir=None)
        plain = tokenizer.pre_process_sequence(S1, min_num_tokens=12)
        deleted = tokenizer.pre_process_sequence(S1, min_num_tokens=12, delete_spans=[[0, 8]])
        batch = tokenizer.preprocess_dataset([S1, S2], min_num_tokens=10,
                                             delete_spans=[[[0, 8]], [[2, 7], [15, 22]]])
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "penguin.json"
            path.write_text(json.dumps({
                "concept": "penguin", "group": "custom",
                "sentences": {"positive": [S1], "negative": [S2]},
                "delete_spans": {"positive": [[[0, 8]]], "negative": [[[2, 7], [15, 22]]]},
            }), encoding="utf-8")
            dataset = ConceptDataset(json_file=path, tokenizer=tokenizer, seq_len=12)
    if plain != {"input_ids": S1_IDS + [PAD, PAD], "attention_mask": [1] * 10 + [0, 0]}:
        failures.append(f"sentence without spans changed: {plain}")
    want = {"input_ids": S1_IDS[3:] + [PAD] * 5, "attention_mask": [1] * 7 + [0] * 5}
    if deleted != want:
        failures.append(f"sentence with a deleted span {deleted}, expected {want}")
    want_ids = [S1_IDS[3:] + [PAD] * 3, [32, 416, 262, 5933, 13] + [PAD] * 5]
    want_mask = [[1] * 7 + [0] * 3, [1] * 5 + [0] * 5]
    if batch["input_ids"] != want_ids or batch["attention_mask"] != want_mask:
        failures.append(f"batch {dict(batch)}, expected ids {want_ids} and mask {want_mask}")
    want_dataset = [[1] * 5 + [0] * 7, [1] * 7 + [0] * 5]
    if dataset.data["attention_mask"] != want_dataset or dataset.data["labels"] != [0, 1]:
        failures.append(f"ConceptDataset mask {dataset.data['attention_mask']}, labels {dataset.data['labels']}")
    return "tokenizer and ConceptDataset on GPT-2"


def check_text_baseline(failures) -> str:
    """The text baseline reads the positives without their deleted spans."""
    concepts = fixture()
    for key, data in concepts.items():
        data["delete_spans"] = EXPECTED_SPANS[key]
    with tempfile.TemporaryDirectory() as tmp:
        write_dataset(pathlib.Path(tmp), concepts)
        documents = load_positive_documents(tmp, ("penguin", "chair", "birds"))
    want = ("The  swam past a chair.", "A  stood near the penguin pool.", " rested on the chair.")
    if documents != want:
        failures.append(f"documents {documents}, expected {want}")
    return "text baseline cuts the spans"


def checks(failures) -> str:
    """Every property of the context-only deletion."""
    return ", ".join([check_forms(failures), check_rule(failures), check_build(failures),
                      check_real_dataset(failures), check_tokenizer(failures),
                      check_text_baseline(failures)])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
