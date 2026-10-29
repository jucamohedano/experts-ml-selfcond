"""
The sentence generator asks for the concept word, a varied position and 15 to 25 words, and keeps only valid sentences.

Checks:
  1. Validation: a sentence passes with one of the concept's inflected forms and 8 to 40 words, derived
     forms (surfers), pronouns and fragments fail, with the same matcher the context-only deletion uses.
  2. Signatures: facts and stories take a word_forms input and state the word, position and length rules.
  3. Collection: with a scripted predictor in place of the endpoint, only valid unique sentences are kept,
     every call carries the word forms and a fresh seed, rejections are counted by reason and kept, and a
     predictor that never complies stops at twice the target number of calls.
  4. Intermediate files: the validation record is written beside the positives.

Usage, from scripts/:
    python tests/check_generation_validation.py
"""
import asyncio
import json
import pathlib
import sys
import tempfile

import dspy

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import generate_definitions_dspy as gen
from _report import run

LONG = "The penguin " + " ".join(["walked"] * 38) + "."
VALIDATION_CASES = [
    ("surfing", "Every summer she drove to the coast to go surfing with her two closest friends.", None),
    ("surfing", "The surfers paddled out past the breakers at dawn, waiting for the perfect wave.", "no_word"),
    ("radish", "It thrives in cool climates and requires well-drained soil for optimal growth.", "no_word"),
    ("penguin", "Penguins swim fast.", "length"),
    ("penguin", "Penguins huddle together on the ice to stay warm.", None),
    ("penguin", LONG, None),
    ("penguin", LONG[:-1] + " again.", "length"),
    ("birds", "A small bird landed on the fence and sang for a while before flying away.", None),
    ("policeman", "Two policemen walked slowly along the quiet street after midnight.", None),
]
RULES = ("Every sentence must contain the concept word, in one of the forms listed in word_forms.",
         "Vary where the concept word appears in the sentence.",
         "Each sentence should be between 15 and 25 words long.")
VALID = ["Penguins huddle together on the ice to stay warm.",
         "A penguin can dive deeper than most other birds can.",
         "Young penguins stay close to their parents for many weeks."]


class ScriptedPredictor:
    """Returns scripted batches in order, then repeats the last one, and records each call's inputs."""

    def __init__(self, batches):
        self.batches, self.calls = batches, []

    async def acall(self, **inputs):
        self.calls.append(inputs)
        return dspy.Prediction(sentences=self.batches[min(len(self.calls), len(self.batches)) - 1])


def check_validation(failures) -> str:
    """Inflected forms and the word bounds decide, derived forms and pronouns fail."""
    for word, sentence, want in VALIDATION_CASES:
        got = gen.rejection_reason(sentence, gen.form_pattern(word))
        if got != want:
            failures.append(f"{word}: {sentence[:40]!r} gave {got}, expected {want}")
    return f"{len(VALIDATION_CASES)} validation cases"


def check_signatures(failures) -> str:
    """Both signatures take word_forms and state the three rules."""
    for sig in (gen.GenerateFacts, gen.GenerateStories):
        if "word_forms" not in sig.input_fields:
            failures.append(f"{sig.__name__} has no word_forms input")
        for rule in RULES:
            if rule not in sig.instructions:
                failures.append(f"{sig.__name__} instructions miss {rule!r}")
        if "word forms" not in sig.output_fields["sentences"].json_schema_extra["desc"]:
            failures.append(f"{sig.__name__} output description does not ask for the word forms")
    return "facts and stories signatures"


def check_collection(failures) -> str:
    """Only valid unique sentences are kept, rejections counted, the call cap holds."""
    generator = gen.ConceptGenerator(fact_count=3, story_count=2, generator="cot")
    generator.fact_generator = ScriptedPredictor([
        [VALID[0], "Penguins swim fast.", "It eats fish every single day of the long and cold winter."],
        [VALID[0], VALID[1], VALID[2]],
    ])
    generator.story_generator = ScriptedPredictor([["Nothing here names the bird at all in this sentence."]])
    result = asyncio.run(generator.aforward(concept="penguin", definition="d", article="a",
                                            category="birds", sense_hint="h"))
    if result.positives != VALID:
        failures.append(f"positives {result.positives}, expected {VALID}")
    facts, stories = result.validation["facts"], result.validation["stories"]
    want_facts = {"calls": 2, "received": 6, "length": 1, "no_word": 1, "duplicate": 1}
    if {k: facts[k] for k in want_facts} != want_facts:
        failures.append(f"fact stats {facts}, expected {want_facts}")
    if [r[0] for r in facts["rejected"]] != ["length", "no_word"]:
        failures.append(f"fact rejections {facts['rejected']}, expected a length and a no_word")
    if stories["calls"] != 4 or stories["no_word"] != 4:
        failures.append(f"story stats {stories}, expected the cap of 4 calls, all rejected")
    seeds = {c["random_seed"] for c in generator.fact_generator.calls}
    forms = {c["word_forms"] for c in generator.fact_generator.calls + generator.story_generator.calls}
    if len(seeds) != 2 or forms != {"penguin, penguins"}:
        failures.append(f"call inputs: seeds {seeds}, word forms {forms}")
    return "scripted predictor through aforward"


def check_intermediate(failures) -> str:
    """The validation record is stored with the positives."""
    validation = {"facts": {"calls": 1}, "stories": {"calls": 2}}
    with tempfile.TemporaryDirectory() as tmp:
        gen.write_intermediate_positives(intermediate_dir=pathlib.Path(tmp), concept="penguin",
                                         storage_key="penguin", category="birds", positives=VALID,
                                         validation=validation)
        record = json.loads((pathlib.Path(tmp) / "penguin.json").read_text(encoding="utf-8"))
    if record.get("validation") != validation or record["positives"] != VALID:
        failures.append(f"intermediate record {record}")
    return "intermediate record"


def checks(failures) -> str:
    """Every property of the generator's validation."""
    return ", ".join([check_validation(failures), check_signatures(failures),
                      check_collection(failures), check_intermediate(failures)])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
