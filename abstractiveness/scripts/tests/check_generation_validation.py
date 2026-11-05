"""
The sentence generator asks for the concept word, at most 3 of 10 sentences opening with it and 18 to 30 words, and keeps only valid sentences.

Checks:
  1. Validation: a sentence passes with one of the concept's inflected forms and 8 to 40 words, derived
     forms (surfers), pronouns and fragments fail, with the same matcher the context-only masking uses.
     Non-Latin letters, a missing final . ! or ?, a second sentence and a lowercase start fail too.
     A sentence opens with the concept when the word comes first or right after a, an or the.
  2. Signatures: facts and stories take a word_forms input and state the format, word, position and length rules,
     and only facts ask for general facts rather than events.
  3. Collection: with a scripted predictor in place of the endpoint, sentences are whitespace-normalised, only
     valid sentences unique ignoring case and across facts and stories are kept, at most a third of the target
     opening with the concept, every call carries the word forms and a fresh seed, rejections are counted by
     reason and kept, and a predictor that never complies stops at twice the target number of calls.
  4. Intermediate files: the validation record is written beside the positives.
  5. Senses: the everyday US sense of words whose WordNet anchor or order picks another (bus, pants, cashier,
     pajamas, hockey), with earlier corrections and the squash homonyms kept.

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
    ("penguin", "Penguins huddle together on the ice to stay warm", "unfinished"),
    ("penguin", "Penguins huddle together on the ice. They stay warm all winter long.", "multi"),
    ("penguin", "penguins huddle together on the ice to stay warm.", "lowercase"),
    ("penguin", "Penguins huddle together on the ice with perfect\u5f27\u7ebf.", "script"),
    ("penguin", "She whispered to her brother, \u201cThe penguin looks cold tonight.\u201d", None),
    ("penguin", "\u201cThe penguin looks cold,\u201d she whispered to her younger brother.", None),
    ("penguin", "The penguin\u2019s feathers keep it dry\u2014even during the harshest winter storms.", None),
    ("penguin", "Old Mr. Finch placed a tiny fish on the rock for the penguin each morning.", None),
    ("penguin", "Dr. Ruiz and St. Clair counted every penguin on the island that summer.", None),
]
START_CASES = [
    ("penguin", "Penguins huddle together on the ice to stay warm.", True),
    ("penguin", "The penguin waddled slowly towards the edge of the ice shelf.", True),
    ("penguin", "\u201cA penguin,\u201d she whispered, pointing at the small dark shape on the rocks.", True),
    ("accountant", "An accountant reviews the ledgers of the company every quarter.", True),
    ("penguin", "A small penguin waddled slowly towards the edge of the ice shelf.", False),
    ("penguin", "Then the penguin dove into the freezing water after a fish.", False),
    ("penguin", "Another penguin followed the first one into the freezing water.", False),
]
RULES = ("Output exactly 10 sentences.",
         "Every sentence must contain the concept word, in one of the forms listed in word_forms.",
         'At most 3 of the 10 sentences may begin with the concept word, with or without "a", "an" or "the" before it.',
         "In the other sentences, place the concept word in the middle or near the end.",
         "Each sentence must have 18 to 30 words, a main clause plus at least one clause adding context "
         "such as time, place, cause or consequence.")
FACTS_RULE = "State general facts that hold for the concept, not events involving a particular person or object."
MID = "Every winter the young penguin returns to the same rocky beach to breed."
FIRST_STORY = "The penguin slid down the icy slope while its chicks watched from the nest above."
STORY_A = "Last spring a curious child followed one penguin along the shore for hours."
STORY_B = "Every evening the old keeper counted each penguin returning from the sea."
VALID = ["Penguins huddle together on the ice to stay warm.",
         "A penguin can dive deeper than most other birds can.",
         "Young penguins stay close to their parents for many weeks."]


SENSES = [
    ("bus", "vehicles", "bus.n.01"), ("pants", "clothing", "trouser.n.01"), ("cashier", "professions", "cashier.n.02"),
    ("pajamas", "clothing", "pajama.n.02"), ("hockey", "sports", "ice_hockey.n.01"), ("van", "vehicles", "van.n.05"),
    ("date", "fruit", "date.n.08"), ("canary", "birds", "canary.n.04"), ("squash", "sports", "squash.n.03"),
    ("squash", "vegetables", "squash.n.02"), ("birds", None, "bird.n.01"), ("penguin", "birds", "penguin.n.01"),
]


class ScriptedPredictor:
    """Returns scripted batches in order, then repeats the last one, and records each call's inputs."""

    def __init__(self, batches):
        self.batches, self.calls = batches, []

    async def acall(self, **inputs):
        self.calls.append(inputs)
        return dspy.Prediction(sentences=self.batches[min(len(self.calls), len(self.batches)) - 1])


def check_validation(failures) -> str:
    """Inflected forms, word bounds and sentence form decide, derived forms and pronouns fail."""
    for word, sentence, want in VALIDATION_CASES:
        got = gen.rejection_reason(sentence, gen.form_pattern(word))
        if got != want:
            failures.append(f"{word}: {sentence[:40]!r} gave {got}, expected {want}")
    for word, sentence, want in START_CASES:
        if gen.starts_with_concept(sentence, gen.form_pattern(word)) != want:
            failures.append(f"{word}: {sentence[:40]!r} opening with the concept should be {want}")
    return f"{len(VALIDATION_CASES)} validation and {len(START_CASES)} opening cases"


def check_signatures(failures) -> str:
    """Both signatures take word_forms and state the format, word, position and length rules."""
    for sig in (gen.GenerateFacts, gen.GenerateStories):
        if "word_forms" not in sig.input_fields:
            failures.append(f"{sig.__name__} has no word_forms input")
        for rule in RULES:
            if rule not in sig.instructions:
                failures.append(f"{sig.__name__} instructions miss {rule!r}")
        if (FACTS_RULE in sig.instructions) != (sig is gen.GenerateFacts):
            failures.append(f"{sig.__name__} {'misses' if sig is gen.GenerateFacts else 'should not carry'} the general-facts rule")
        if "one per line" in sig.instructions:
            failures.append(f"{sig.__name__} instructions still ask for one sentence per line")
        desc = sig.output_fields["sentences"].json_schema_extra["desc"]
        for part in ("word forms", "18 to 30 words", "at most 3"):
            if part not in desc:
                failures.append(f"{sig.__name__} output description misses {part!r}")
    return "facts and stories signatures"


def check_collection(failures) -> str:
    """Sentences normalised, unique ignoring case and across kinds, openings capped at a third, rejections counted."""
    generator = gen.ConceptGenerator(fact_count=3, story_count=2, generator="cot")
    generator.fact_generator = ScriptedPredictor([
        [VALID[0], "Penguins swim fast.", "It eats fish every single day of the long and cold winter."],
        [VALID[0], VALID[1], "  Young penguins stay close\nto their parents for many weeks. "],
        ["Young Penguins stay close to their parents for many weeks.", MID],
    ])
    generator.story_generator = ScriptedPredictor([
        ["Nothing here names the bird at all in this sentence.", FIRST_STORY, MID, STORY_A],
        [STORY_B],
    ])
    result = asyncio.run(generator.aforward(concept="penguin", definition="d", article="a",
                                            category="birds", sense_hint="h"))
    want = [VALID[0], VALID[2], MID, STORY_A, STORY_B]
    if result.positives != want:
        failures.append(f"positives {result.positives}, expected {want}")
    facts, stories = result.validation["facts"], result.validation["stories"]
    want_facts = {"calls": 3, "received": 8, "kept": 3, "length": 1, "no_word": 1, "duplicate": 2, "word_first": 1}
    if {k: facts[k] for k in want_facts} != want_facts:
        failures.append(f"fact stats {facts}, expected {want_facts}")
    if [r[0] for r in facts["rejected"]] != ["length", "no_word", "word_first"]:
        failures.append(f"fact rejections {facts['rejected']}, expected a length, a no_word and a word_first")
    want_stories = {"calls": 2, "received": 5, "kept": 2, "no_word": 1, "word_first": 1, "duplicate": 1}
    if {k: stories[k] for k in want_stories} != want_stories:
        failures.append(f"story stats {stories}, expected {want_stories} (a story repeating a fact is a duplicate)")
    seeds = {c["random_seed"] for c in generator.fact_generator.calls}
    forms = {c["word_forms"] for c in generator.fact_generator.calls + generator.story_generator.calls}
    if len(seeds) != 3 or forms != {"penguin, penguins"}:
        failures.append(f"call inputs: seeds {seeds}, word forms {forms}")
    return "scripted predictor through aforward"


def check_call_cap(failures) -> str:
    """A predictor that never complies stops at twice the target number of calls."""
    generator = gen.ConceptGenerator(fact_count=2, story_count=1, generator="cot")
    generator.fact_generator = ScriptedPredictor([["Nothing here names the bird at all in this sentence."]])
    generator.story_generator = ScriptedPredictor([[FIRST_STORY]])
    result = asyncio.run(generator.aforward(concept="penguin", definition="d", article="a",
                                            category="birds", sense_hint="h"))
    calls = (result.validation["facts"]["calls"], result.validation["stories"]["calls"])
    if calls != (4, 2) or result.positives:
        failures.append(f"calls {calls}, positives {result.positives}, expected (4, 2) and none")
    return "call cap"


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


def check_senses(failures) -> str:
    """The intended WordNet sense of overridden, anchored and homonym concepts."""
    gen.ensure_nltk_wordnet()
    for concept, category, want in SENSES:
        got = gen.select_wordnet_sense(concept, category)
        if got is None or got.name() != want:
            failures.append(f"{concept} ({category}): sense {got.name() if got else None}, expected {want}")
    return f"{len(SENSES)} senses"


def checks(failures) -> str:
    """Every property of the generator's validation."""
    return ", ".join([check_validation(failures), check_signatures(failures),
                      check_collection(failures), check_call_cap(failures), check_intermediate(failures),
                      check_senses(failures)])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
