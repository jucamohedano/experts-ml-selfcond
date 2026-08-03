# Logical fixes

This file records logical corrections made to the analysis pipeline. It is kept short and is meant to be extended as further fixes are applied.

## Expert identity keyed on (layer, unit) pairs (modules 3 and 5)

The expertise data identifies each expert neuron by a `unit` index that is unique only within a single layer, because the same index is reused across the 48 layers. Modules 3 and 5 previously grouped experts by the bare `unit` value, which merged identically indexed neurons from different layers into one element. This raised the intersection between concepts that shared nothing more than an index coincidence, adding a roughly uniform baseline of similarity to every pair and compressing the contrast between within-category and across-category pairs. For example, the module 5 Jaccard within-versus-across ratio at AP=0.6 was about 1.4x under the old keying and 6.6x after the correction. Both modules now key expert sets on the `(layer_idx, unit)` pair, so that two experts are counted as shared only when they are the same neuron in the same layer.

## Wrong WordNet sense in the generated stimulus sentences (all modules)

Sixteen concepts were described by the wrong sense of their word. The stimulus sentences are written by an LM from a WordNet gloss, and when the gloss belongs to a different sense than the category intends, all 400 positive sentences of that concept describe another word entirely. The expert units found for it are then experts for that other meaning, and every analysis that treats the concept as a member of its category, similarity to the category label in module 3, the pairwise matrices in module 5, the prototype versus exemplar divergence in module 6, the model-derived typicality in module 7, is computing over a mislabelled item. The failure is invisible inside the pipeline, because the file is complete and the counts are correct, only the meaning is wrong.

The problem was found by reading the module 5 similarity heatmaps, where `date`, `canary` and `cuckoo` sat away from their category blocks, and confirmed by reading the sentences themselves.

### The affected concepts

| Concept | Category | Sense actually generated | Corrected sense |
|---|---|---|---|
| date | fruit | escort, a participant in a romantic outing | `date.n.08`, sweet edible fruit of the date palm |
| canary | birds | police informer | `canary.n.04`, small Old World finch |
| cuckoo | birds | a fool, someone acting foolishly | `cuckoo.n.02`, bird with pointed wings |
| sports | level 1 label | Maine colloquial, a temporary summer resident | `sport.n.01`, an active diversion requiring physical exertion |
| fencing | sports | fence, barrier material | `fencing.n.03`, the art of fighting with swords |
| trailer | vehicles | one who trails or lags behind | `trailer.n.04`, a wheeled vehicle pulled by a car or truck |
| cushion | furniture | mechanical shock damper | `cushion.n.03`, a soft bag filled with padding |
| radio | furniture | the broadcasting medium | `radio_receiver.n.01`, an electronic receiver |
| gloves | clothing | baseball fielder's glove | `glove.n.02`, handwear covering the hand and wrist |
| van | vehicles | British closed railway car | `van.n.05`, a truck with an enclosed cargo space |
| scooter | vehicles | motorboat resembling a motor scooter | `motor_scooter.n.01`, a low-powered wheeled vehicle |
| carriage | vehicles | railway passenger coach | `carriage.n.02`, a wheeled vehicle drawn by horses |
| fireman | professions | stoker, a laborer who tends fires | `fireman.n.04`, a member of a fire department |
| secretary | professions | head of a government department | `secretary.n.02`, an assistant handling clerical work |
| handball | sports | the rubber ball, not the game | `handball.n.02`, a game played in a walled court |
| minister | professions | a person conducting religious worship | `minister.n.02`, a person appointed to high government office |

The last four rows are judgment calls rather than plain errors, in the sense that the generated sense is valid English and, for `carriage` and `minister`, still satisfies the category. They are corrected because the Richie and Bhatia norms are US norms collected over everyday objects and occupations, and because `handball` as an object sits inside a category whose every other member is an activity, which breaks comparability at level 2.

### Root cause

Sense selection lives in `scripts/generate_definitions_dspy.py`. The old `fetch_wordnet_gloss` chose the lowest-indexed sense accepted by `is_relevant_concept`, which accepts any synset whose hypernym chain reaches `artifact.n.01`, `living_thing.n.01` or `body_part.n.01`. Because `person` inherits from `organism` and then from `living_thing`, every noun.person sense passes that filter, and WordNet usually lists the person sense first. That is why `date` resolved to the escort sense and `canary` to the informer sense.

A category to lexicographer-domain hint table was added later to correct exactly this, but it never took effect on the existing data: 201 of the 206 files under `custom/.intermediate/` carry no `category` key, meaning they were written by the earlier version of the script that passed no category at all. The hint mechanism was also insufficient on its own, since `sports` was regenerated after it landed and still broke, being a level 1 category label whose own category is null.

### The fix

`fetch_wordnet_gloss` is now a thin wrapper over `select_wordnet_sense`, which resolves in this order:

1. `SENSE_OVERRIDES`, keyed by (concept, category), for the words where two senses both satisfy the category and only the source word list decides between them.
2. `LEVEL1_LABEL_SENSES`, pinning the eight category labels explicitly, since they carry no category and so have nothing to constrain them.
3. `CATEGORY_ANCHOR_NAMES`, the lowest-indexed sense that is, or descends from, an anchor synset of the category, for example `bird.n.01` for birds and `vehicle.n.01` for vehicles. This is the main signal, and it is what separates the bird `canary` from the informer `canary`.
4. `CATEGORY_LEXNAME_HINTS`, the lexicographer-domain table, which rescues concepts whose intended sense sits outside the anchor subtree, for example `chess` the board game, whose first WordNet sense is a weed.
5. The old concreteness heuristic, with noun.person senses excluded unless the category is professions. This exclusion is the core correction.
6. The first sense WordNet lists, for words whose concrete sense is already first.

The chosen sense's synonyms and its immediate hypernym are also passed to the LM through a new `sense_hint` input field, so the intended meaning is pinned by more than a single gloss phrasing.

Three supporting corrections landed with it. WordNet synsets are resolved lazily rather than at import time, so the script can run before the corpus is downloaded. `load_categories_for_concepts` skips metadata rows without a `concept` key instead of raising `KeyError` on the deliberately excluded underscore-prefixed entries. `gather_negatives` takes the unfiltered entry list for the concept list CSV, so rebuilding a subset of concepts no longer truncates `concept_list.csv` to that subset.

### Detecting it again

`scripts/audit_concept_senses.py` re-runs the check that found the problem. Each concept is represented by its positive sentences with its own name stripped out, the documents are TF-IDF vectorised, and each concept is scored by cosine similarity against every category profile with itself held out. A concept whose assigned category is not the best match is describing something other than what its category-mates describe. The report writes `results/concept_sense_audit.csv` and exits non-zero on any flag, so it can gate a regeneration run.

The audit is a screen rather than a proof. It catches errors that cross a category boundary, such as a bird described as a person, but not errors within a category, such as a railway van described instead of a road van, since both talk about vehicles. For those the report prints the currently chosen gloss beside each concept, which makes the mismatch readable even when the similarity ranking is clean.

### What the correction changed

The sixteen concepts were regenerated with the corrected glosses, their negatives redrawn from the full corrected pool, and responses and expertise recomputed for those sixteen plus `squash` in both models. The other 188 concept files were never rewritten, so their cached responses stayed valid and untouched.

Two checks confirm the recompute did what it was supposed to and nothing else. First, the audit went from nine flagged concepts to zero, with the three remaining rank-two rows being the known benign neighbours (`cycling` and `sailing` resembling vehicles, `skateboard` resembling sports). Second, comparing the new module 3 table against the old one, exactly twenty-five untouched concepts moved, and all twenty-five are members of `sports`, which is correct: `sports` is a level 1 label that was itself regenerated, so every sports member's similarity to its parent necessarily changes. All 156 concepts in the other seven categories are identical to the previous run.

The per-concept effect is large for the words whose sense was grossly wrong. On GPT-2 at AP 0.5 in the `mlp.c_fc` scope, Jaccard with the category label:

| Concept | Before | After |
|---|---|---|
| fencing | 0.90 | 20.43 |
| handball | 6.03 | 15.28 |
| gloves | 1.16 | 10.09 |
| date | 0.00 | 7.32 |
| trailer | 1.24 | 7.49 |
| cushion | 1.97 | 6.15 |
| cuckoo | 0.29 | 3.77 |

`date` is the clearest case, it previously shared exactly zero expert units with `fruit`, which is what placed it outside the fruit block in the module 5 heatmap where the problem was first noticed. The four judgment calls barely moved, and two moved slightly down (`carriage` 10.67 to 7.28, `secretary` 10.52 to 9.92), which is expected: their old senses were still members of the assigned category, so there was no gross mismatch to repair.

At the corpus level the correction improves category alignment in both architectures. Within-category similarity rises while across-category similarity stays flat, which is the signature of removing mislabelled items rather than of inflating all similarities:

| Model, AP 0.5, whole model | Within | Across | Contrast | AUC |
|---|---|---|---|---|
| GPT-2 before | 6.718 | 1.032 | 5.686 | 0.9143 |
| GPT-2 after | 7.135 | 1.017 | 6.118 | 0.9328 |
| Qwen3 before | 6.538 | 1.022 | 5.516 | 0.9388 |
| Qwen3 after | 6.999 | 1.024 | 5.974 | 0.9576 |

(The GPT-2 rows are the `mlp.c_fc` scope, the only scope its previous run recorded.) On Qwen3, where a like-for-like baseline exists at every threshold, the alignment AUC improves at all five: 0.9388 to 0.9576 at AP 0.5, 0.9315 to 0.9516 at 0.6, 0.9048 to 0.9245 at 0.7, 0.7513 to 0.7675 at 0.8, and 0.5402 to 0.5460 at 0.9.

In other words the wrong-sense words were not a cosmetic blemish, they were suppressing the central result. A bird described as a police informer shares its experts with `professions`, which raises the across-category term and lowers the within-category one at the same time.

Results from the corrected data live in `results/research_plots_<model>_richie_hsj_sensefix/`, with the previous runs left in place as the before comparison.

## Squash restored to the dataset

`squash` appears twice in the Richie and Bhatia table, once under vegetables and once under sports, and the collision meant it entered neither model. Both sets of sentences were generated correctly and sit on disk as `squash__vegetables.json` and `squash__sports.json`, but the two metadata rows are held under underscore-prefixed keys that exclude them from the pipeline, so the concept is missing from the responses of both models.

The word is admitted once, under the vegetables sense, because vegetables is the smaller category, 19 member concepts against 27 for sports, so placing it there costs the smaller category less imbalance than removing it would. The sports row is dropped, and with the name no longer duplicated the storage key becomes the plain `squash`.

The rename matters more than it looks. `compute_responses.py` locates a concept's sentences at `<group>/<concept>.json`, reading the `concept` column of `concept_list.csv` rather than the `storage_key` column, so it looked for `squash.json`, found nothing, and skipped the concept without any error. That silent skip, not the metadata exclusion alone, is why `squash` is absent from the responses of both models.

The dataset side is done: `metadata_Richie_HSJ.json` and the dataset config both carry 205 concepts aligned one to one, `custom/squash.json` holds the vegetables sentences, the sports files are parked under `excluded_senses/` so their sentences cannot leak into any other concept's negative pool, and `prepare_metadata_richie_hsj.py` now carries an `EXCLUDED_SENSES` set so regenerating the metadata does not reintroduce the sports row. What remains is computing responses and expertise for `squash` in both models, together with the regeneration above.
