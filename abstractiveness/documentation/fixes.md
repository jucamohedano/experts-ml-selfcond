# Logical fixes

This file records logical corrections made to the analysis pipeline. It is kept short and is meant to be extended as further fixes are applied. Each entry keeps the module numbering of its date. In the current layout former modules 1, 2 and the entropy panel of 4 are module 1, former modules 3, 5, 7 and the rest of 4 are module 2 (3 is section 2.1, 5 is sections 2.2 and 2.3, 7 is section 2.4), and former modules 6, 8 and 9 are modules 3, 4 and 5.

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

`scripts/core/data_preparation/audit_stimulus_word_senses.py` re-runs the check that found the problem. Each concept is represented by its positive sentences with its own name stripped out, the documents are TF-IDF vectorised, and each concept is scored by cosine similarity against every category profile with itself held out. A concept whose assigned category is not the best match is describing something other than what its category-mates describe. The report writes `results/concept_sense_audit.csv` and exits non-zero on any flag, so it can gate a regeneration run.

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

(The GPT-2 rows are the `mlp.c_fc` scope, the only scope its previous run recorded, and Qwen3 is the post-trained Qwen3-1.7B, the only Qwen3 checkpoint run at the time.) On Qwen3, where a like-for-like baseline exists at every threshold, the alignment AUC improves at all five: 0.9388 to 0.9576 at AP 0.5, 0.9315 to 0.9516 at 0.6, 0.9048 to 0.9245 at 0.7, 0.7513 to 0.7675 at 0.8, and 0.5402 to 0.5460 at 0.9.

In other words the wrong-sense words were not a cosmetic blemish, they were suppressing the central result. A bird described as a police informer shares its experts with `professions`, which raises the across-category term and lowers the within-category one at the same time.

Results from the corrected data live in `results/research_plots_<model>_richie_hsj_sensefix/`, with the previous runs left in place as the before comparison.

## Squash restored to the dataset

`squash` appears twice in the Richie and Bhatia table, once under vegetables and once under sports, and the collision meant it entered neither model. Both sets of sentences were generated correctly and sit on disk as `squash__vegetables.json` and `squash__sports.json`, but the two metadata rows are held under underscore-prefixed keys that exclude them from the pipeline, so the concept is missing from the responses of both models.

The word is admitted once, under the vegetables sense, because vegetables is the smaller category, 19 member concepts against 27 for sports, so placing it there costs the smaller category less imbalance than removing it would. The sports row is dropped, and with the name no longer duplicated the storage key becomes the plain `squash`.

The rename matters more than it looks. `compute_responses.py` locates a concept's sentences at `<group>/<concept>.json`, reading the `concept` column of `concept_list.csv` rather than the `storage_key` column, so it looked for `squash.json`, found nothing, and skipped the concept without any error. That silent skip, not the metadata exclusion alone, is why `squash` is absent from the responses of both models.

The dataset side is done: `metadata_Richie_HSJ.json` and the dataset config both carry 205 concepts aligned one to one, `custom/squash.json` holds the vegetables sentences, the sports files are parked under `excluded_senses/` so their sentences cannot leak into any other concept's negative pool, and `prepare_metadata_richie_hsj.py` now carries an `EXCLUDED_SENSES` set so regenerating the metadata does not reintroduce the sports row. What remains is computing responses and expertise for `squash` in both models, together with the regeneration above.

**Superseded in September 2026 by the section below.**

## Both squash senses admitted (September 2026)

The exclusion above was reversed. Both senses are in the dataset as separate concepts, so the sports category is whole at 28 members and the stimulus set is 206 concepts over 205 distinct words.

### The concept key

A word the source table lists under two categories now gets one row per sense, keyed `word__category`. The identity moved off the surface form and onto a key, and the bare form lives in a new `word` field beside it. Anything lexical reads `word`, which is what WordNet is queried with, what the sentence generator sends to the model, and what the frequency tables are keyed on. Anything joining against an expert-side table reads `concept`. That split is what lets one spelling carry two expert sets.

The rationale recorded for the original exclusion was mistaken, and it is worth stating why, because it survived into two later drafts. It held that one sense's sentences would sit in the negative pool of the other and corrupt the Average Precision. They do sit there, under separate keys exactly as before, 5 sports sentences among the negatives of `squash__vegetables` and 4 in the other direction, and nothing is corrupted by it. A label is corrupt when it is false, and a sentence about the game genuinely is not about the vegetable, so its label of 0 is true. Such cross-sense sentences are hard negatives that favour units responding to the meaning over units responding to the shared spelling, and the effect is small, since only 1 of the 1,000 negatives drawn for the vegetable sense contains the token `squash`. The real obstacle was always identity, which is what the silent skip described above actually was.

The key is derived, never listed. `build_concept_metadata` computes the ambiguous set as the surface forms appearing under more than one category in the word list, and `storage_key_for` in `generate_definitions_dspy.py` reaches the same key independently from the config. Nothing names `squash`. A different duplicate word would be keyed the same way with no code change, which was verified by building a throwaway word list with `bass` under two categories and watching `bass__birds` and `bass__sports` appear.

`select_wordnet_sense` already resolved the two senses through the per-category anchors introduced by the sense fix above, returning `squash.n.02`, the edible fruit of a squash plant, for vegetables and `squash.n.03`, the game played in an enclosed court, for sports. No override was needed for either.

### What was removed

`EXCLUDED_SENSES`, `EXCLUDED_MEMBERS` and the `excluded_senses/` folder are gone. The sports sentence files moved into `custom/` and `custom/squash.json` became `custom/squash__vegetables.json`. Human similarity went from 2,391 rated pairs to 2,418, sports rising from 351 to the full 378, and because one word can now map to two concepts every pair row carries `concept_a` and `concept_b` beside `word_a` and `word_b`.

**Correction, 28 September 2026.** The two joins that read the ratings, module 2's human validation (formerly module 5) and module 9's pair-similarity study, still matched pairs on `word_a` and `word_b` rather than on the concept keys, so every pair of `squash__sports` and `squash__vegetables` was silently dropped and the pipeline kept using 2,372 of the 2,418 rated pairs. Both now join on `concept_a` and `concept_b`, and `tests/check_module2_human_validation.py` requires every rated pair to be used. Module 9's join, now module 5's `human_pair_lookup`, is keyed by the sorted concept keys, and `check_module5_pair_similarity_design` pins it for `squash__sports`. On the post-trained Qwen3-1.7B at AP 0.5, the only Qwen3 checkpoint run at the time, the whole-model Jaccard agreement moved from 0.5045 to 0.5219 in sports (351 to 378 pairs) and from 0.2920 to 0.2832 in vegetables (171 to 190 pairs), the layer profile from 0.0868 to 0.0923 and from 0.1140 to 0.1509, and the Jaccard mean over categories from 0.5070 to 0.5081, the other six categories being unchanged. Results trees written before the correction, the August ones included, lack those pairs.

### What the rebuild costs

Nothing was regenerated. Both sets of positives already existed, the sports set having been parked rather than deleted when the exclusion was made, so no sentence was written and the paid endpoint was never called. What had to change is the negative pool, because `build_negatives_stratified` splits each concept's quota across all other concepts, so moving from 205 to 206 changes the quota and the sampling order for every concept. That is a local rebuild from the existing intermediate files through `--fix-intermediate`, which forces the negatives-only phase and makes no LM calls. Responses and expertise then have to be recomputed for all 206 in both models, since every concept's negatives moved.

### Provenance of the `source` field

The rebuild rewrites every `custom/*.json`, and `source` is computed once per run and stamped on all of them, so the field no longer distinguishes the concepts regenerated by the sense fix above. It recorded a real distinction and this is where that record now lives instead. These seventeen files carried `dspy_openai_Qwen_Qwen3-30B-A3B-Instruct-2507_sensefix`, meaning their positives came from the sense-fix regeneration rather than the original sweep:

`canary`, `carriage`, `cuckoo`, `cushion`, `date`, `fencing`, `fireman`, `gloves`, `handball`, `minister`, `radio`, `scooter`, `secretary`, `sports`, `squash__vegetables`, `trailer`, `van`

The other 189 carried the plain `dspy_openai_Qwen_Qwen3-30B-A3B-Instruct-2507`. Do not pass `--source-tag` on the negatives rebuild. The flag exists to mark a targeted regeneration of positives, and using it here would assert that all 206 positive sets came from this run, which is false for every one of them.

See section 2 of [dataset_and_metadata.md](dataset_and_metadata.md) for the key scheme and section 8 for the schema.

### The silent skip, returning and closed

The first negatives rebuild reproduced the original failure exactly. `write_concept_list_csv` wrote the surface form into the `concept` column and the key into a separate `storage_key` column, but `selfcond.data.concept_list_to_df`, which both `compute_responses.py` and `compute_expertise.py` read the file through, takes only `group` and `concept` and then opens `<group>/<concept>.json`. Both squash rows therefore said `squash`, pointing at a `custom/squash.json` that no longer exists, and `compute_responses.py` answers a missing file with one line of output and a `return`, which is invisible inside a 206 concept loop.

The column now holds the storage key and the surface form moved to a `word` column, matching the concept files themselves. A guard was added beside it: writing the CSV first checks that every row resolves to a file and raises `FileNotFoundError` naming the offenders, because a row that does not resolve is a silent skip rather than a loud failure, and this is the second time that has cost a concept.

### A sentence labelled both positive and negative

`build_negatives_stratified` excluded the target concept from its list of SOURCES but never checked sentence identity, and that is not the same guarantee. The generator sometimes writes one sentence verbatim for two related concepts, `radish` and `turnip` both receiving "The plant belongs to the Brassicaceae family, which includes cabbage and mustard.", so drawing that line from `turnip` placed a `radish` positive into the negatives of `radish`. The same string then carried both labels, which is a direct corruption of that concept's Average Precision.

It surfaced only on the rebuild, at one sentence in one concept. The pre-rebuild dataset had zero such collisions, but by luck rather than by construction, since moving from 205 to 206 concepts changes both the per-source quota and the shuffle order and so re-rolls every draw. The sampler now filters each source pool against the target's own positives, at the per-source draw, at the deficit borrow and at the final fallback. A sentence shared between two concepts is still a legitimate negative for every OTHER concept, which is what the guard preserves.

### Recomputing after the rebuild

Every concept's negatives changed, not only squash's, because `build_negatives_stratified` splits each quota across all other concepts and the count moved from 205 to 206. Cached responses cannot be reused. `check_responses_complete` compares the sample COUNT against the expected 1,400 and nothing about the content, so a stale concept directory reports complete and is skipped rather than recomputed. The response trees of both models have to be removed by hand before the pipeline is rerun.

## Mantel p on a constant similarity vector (module 2, October 2026)

Section 2.3 tests each category's Spearman $\rho_k$ with a within-category Mantel permutation. At strict thresholds on thin sublayers every rated pair of a category can share the same model value, typically Jaccard 0, and Spearman is then undefined. The observed $|\rho_k|$ was NaN, no permuted value compares as greater or equal to NaN, so the hit count stayed at zero and the test returned its floor $1/(B+1) = 0.001$, a spurious significance for a correlation that does not exist. The row now carries NaN for both $\rho_k$ and $\hat p_k$, and every Spearman call of the section returns NaN for constant input without calling scipy, which also removes its `ConstantInputWarning` from the log. Rows with a defined $\rho_k$ are unchanged.

## Empty layers dropped at strict thresholds (module 1, October 2026)

`compute_layer_distributions` built its concept-by-layer counts with `pd.crosstab`, which keeps only the layers where at least one word has an expert. From AP 0.7 some layers have no expert for any word, GPT-2 losing 3 of its 48 layers at AP 0.8 and 9 at AP 0.9, the post-trained Qwen3-1.7B 6 of its 196 at AP 0.7 and 55 at AP 0.9, and one of its 28 blocks disappearing at AP 0.9. The `_by_layer` and `_by_block` distribution tables lost those rows and the plots closed the gaps in the depth axis. The shares themselves were right, since the dropped layers held no experts. `compute_sublayer_informativeness` had the same issue in its per-sublayer count pivot, so its `gearys_c` column compared layers that are not neighbours whenever a layer of the sublayer was empty. Both now reindex to every layer of the axis with zero counts. Nothing changes at AP 0.5 and 0.6, where no layer is empty, which an independent recomputation of every section 1 and 2 table from the raw `expertise.csv` files confirmed (`temp_docs/pilots_2026-10-05/sanity.py`).

## Qwen3 results switched to the Base checkpoint (October 2026)

On 8 October 2026 every Qwen3 number of the module pages moved from the post-trained Qwen3-1.7B (config `qwen3_richie_hsj`, tree `results/research_plots_qwen_richie_hsj_restructured_final`) to Qwen3-1.7B-Base, its pre-trained checkpoint (config `qwen3_base_richie_hsj`, tree `results/research_plots_qwen_base_richie_hsj_FINAL`). Both trees come from the same code, the same 206 words and the same generated sentences, so the tables below isolate what post-training changes. The module pages report the Base values only, and this entry, with the older entries above that were measured before the Base checkpoint existed, is the only place the post-trained values are kept. In every cell the first value is Base and the second post-trained, at whole-model scope unless a sublayer is named. The figures were extracted from the two trees on 9 October 2026, the module 1 tests recomputed with the same Mann-Whitney calls the pipeline logs, and the shared-expert counts read from the raw `expertise.csv` files.

What moves is consistent. Base keeps about a third more expert rows at every threshold (1,573,600 against 1,181,053 at AP 0.5) and more words at AP 0.9 (201 against 196), while the depth organisation is the same, the members' mean profile reaching 50% and 90% of its mass at the same blocks at AP 0.5. Expert-set alignment with the categories barely changes, the Jaccard ROC-AUC staying within 0.034 at every threshold, whereas every layer-profile reading is stronger in Base: the pairwise layer-profile ROC-AUC is 0.734 against 0.693 at AP 0.5, the member-to-name mean $z$ +0.58 against +0.38, its human agreement 0.111 against 0.047 and the profile tree's ARI in Study C 0.194 against 0.056. Agreement with human similarity rises by 0.02 to 0.04 at every threshold, and the share of it that survives the probing-text partial rises by the same amount. Post-training lowers the category alignment of the attention and down projections most, `self_attn.q_proj` falling from 0.786 to 0.698 at AP 0.6. A few readings differ between the checkpoints rather than only shifting: the standalone profile cell of Study A sits above chance in Base on the label word (52.8 against 43.1 percent), fewer category names are more spread than their members in Base (3 against 5 of 8), the name entropy against JSD of module 3 is weaker in Base at AP 0.5 and 0.7, and the profile tree of Study C keeps its category agreement at AP 0.8 in Base (0.132 against 0.002), which is why the module pages no longer describe the two geometries as answering the threshold with opposite signs.

### Module 1, expert rows and surviving words

Whole model, `expert_counts_with_metadata.csv` and the expert rows of section 2.2, of the 206 words.

| AP | expert rows | words with experts | words below 30 experts | category names with experts | mean experts per member | mean experts per category name |
|---|---|---|---|---|---|---|
| 0.5 | 1,573,600 vs 1,181,053 | 206 vs 206 | 0 vs 0 | 8 vs 8 | 7,846 vs 5,880 | 2,506 vs 2,112 |
| 0.6 | 569,820 vs 416,626 | 206 vs 206 | 0 vs 0 | 8 vs 8 | 2,854 vs 2,083 | 598 vs 516 |
| 0.7 | 227,691 vs 164,816 | 206 vs 206 | 0 vs 0 | 8 vs 8 | 1,144 vs 827 | 143 vs 126 |
| 0.8 | 82,110 vs 59,420 | 206 vs 206 | 8 vs 10 | 8 vs 8 | 414 vs 299 | 22 vs 22 |
| 0.9 | 16,740 vs 12,149 | 201 vs 196 | 92 vs 105 | 5 vs 3 | 85 vs 63 | 2 vs 2 |

### Module 1, where along depth the mass sits

Block axis, `mean_expert_layer_distribution_summary_by_block.csv`, blocks counted from 0, thirds in percent of the members' mean profile.

| AP | members reach 50% at block | members reach 90% at block | first third | middle third | last third | category names reach 50% at block |
|---|---|---|---|---|---|---|
| 0.5 | 20 vs 20 | 26 vs 26 | 39.7 vs 40.1 | 8.8 vs 6.5 | 51.5 vs 53.3 | 18 vs 18 |
| 0.6 | 19 vs 20 | 26 vs 26 | 43.9 vs 43.3 | 6.0 vs 4.2 | 50.1 vs 52.5 | 19 vs 20 |
| 0.7 | 10 vs 17 | 26 vs 26 | 49.2 vs 47.8 | 4.5 vs 3.2 | 46.2 vs 49.1 | 19 vs 20 |
| 0.8 | 7 vs 7 | 25 vs 25 | 55.3 vs 53.5 | 3.5 vs 2.5 | 41.2 vs 44.0 | 19 vs 20 |
| 0.9 | 6 vs 6 | 25 vs 25 | 61.3 vs 57.8 | 2.6 vs 1.5 | 36.1 vs 40.7 | 4 vs 3 |

### Module 1, category names against members

Mann-Whitney tests over the 8 category names against the 198 members, entropy one-sided, average layer two-sided, recomputed from the descriptor tables. The analysis sublayer is `mlp.gate_proj`.

| AP | name entropy, flat axis | member entropy, flat axis | entropy p, flat axis | entropy p, block axis | average layer p, flat axis | entropy p, analysis sublayer | average layer p, analysis sublayer |
|---|---|---|---|---|---|---|---|
| 0.5 | 6.07 vs 5.93 | 6.34 vs 6.20 | <0.001 vs <0.001 | 0.85 vs 0.89 | 0.29 vs 0.20 | 0.72 vs 0.86 | 0.039 vs 0.080 |
| 0.6 | 5.48 vs 5.32 | 6.01 vs 5.83 | <0.001 vs <0.001 | 0.49 vs 0.59 | 0.38 vs 0.68 | 0.34 vs 0.42 | 0.004 vs 0.013 |

### Module 1, peak reliability and smoothness

Ambiguous peaks are those leading the runner-up by less than 1 point, in percent of each group on the flat layer axis, Geary's C is the group mean.

| AP | ambiguous peaks, members | ambiguous peaks, category names | Geary's C, members, block axis | Geary's C, names, block axis | Geary's C, members, flat axis |
|---|---|---|---|---|---|
| 0.5 | 76 vs 71 | 100 vs 88 | 0.27 vs 0.26 | 0.39 vs 0.39 | 0.54 vs 0.54 |
| 0.6 | 57 vs 60 | 75 vs 38 | 0.36 vs 0.33 | 0.46 vs 0.47 | 0.62 vs 0.64 |

### Module 1, entropy against expert count

Pearson $r$ on the block axis, `1.2_distribution_shape/correlation_summary.csv`.

| AP | r |
|---|---|
| 0.5 | 0.252 vs 0.134 |
| 0.6 | 0.160 vs 0.013 |
| 0.7 | 0.132 vs 0.041 |
| 0.8 | 0.221 vs 0.218 |
| 0.9 | 0.467 vs 0.494 |

### Module 1, each category name against its own members

AP 0.6, block axis, `category_entropy_gearys_by_block.csv`, number of the 8 names above their members' median.

| quantity | raw | members matched to the name's expert count |
|---|---|---|
| names more spread than the median member | 3 vs 5 | 4 vs 6 |
| names rougher than the median member | 6 vs 8 | 5 vs 7 |

### Module 1, sublayer informativeness

Category alignment ROC-AUC of the members at AP 0.5 and 0.6, and each level's share of its own experts at AP 0.6 in percent, `sublayer_informativeness.csv` and its summary.

| sublayer | ROC-AUC, AP 0.5 | ROC-AUC, AP 0.6 | members' share | category names' share |
|---|---|---|---|---|
| mlp.gate_proj | 0.958 vs 0.956 | 0.951 vs 0.946 | 43.1 vs 46.2 | 65.9 vs 69.3 |
| mlp.up_proj | 0.957 vs 0.953 | 0.947 vs 0.939 | 27.1 vs 24.8 | 18.9 vs 17.3 |
| self_attn.o_proj | 0.949 vs 0.949 | 0.920 vs 0.896 | 11.0 vs 10.4 | 6.2 vs 5.3 |
| self_attn.v_proj | 0.933 vs 0.909 | 0.863 vs 0.790 | 3.9 vs 3.7 | 3.6 vs 3.4 |
| self_attn.k_proj | 0.915 vs 0.871 | 0.797 vs 0.722 | 2.7 vs 2.6 | 1.8 vs 1.6 |
| self_attn.q_proj | 0.915 vs 0.866 | 0.786 vs 0.698 | 4.3 vs 4.2 | 1.7 vs 1.5 |
| mlp.down_proj | 0.808 vs 0.751 | 0.691 vs 0.630 | 7.8 vs 8.2 | 1.9 vs 1.5 |
| all MLP sublayers |  |  | 78.1 vs 79.2 | 86.7 vs 88.1 |

### Module 1, experts shared between words

AP 0.6, from the raw `expertise.csv` files.

| quantity | value |
|---|---|
| expert rows | 569,820 vs 416,626 |
| distinct expert neurons | 214,988 vs 169,605 |
| neurons expert for two or more words, percent | 58.4 vs 55.0 |
| mean words per expert neuron | 2.65 vs 2.46 |
| the same within `mlp.gate_proj`, percent | 65.9 vs 64.0 |

### Module 2, section 2.1, each member against its category name

`category_concept_similarity_metrics.csv`, Jaccard and overlap in percent, $S$ and $z$ for the default metric on the block axis.

| AP | pairs | mean Jaccard | mean overlap | mean $S$ | mean $z$ | pairs with $z > 0$, percent |
|---|---|---|---|---|---|---|
| 0.5 | 198 vs 198 | 8.65 vs 9.02 | 29.28 vs 28.26 | 81.4 vs 79.4 | +0.58 vs +0.38 | 74 vs 64 |
| 0.6 | 198 vs 198 | 7.03 vs 7.89 | 30.01 vs 28.69 | 74.8 vs 72.8 | +0.41 vs +0.23 | 67 vs 63 |
| 0.7 | 198 vs 198 | 4.51 vs 5.36 | 22.91 vs 22.78 | 63.4 vs 61.9 | +0.20 vs +0.15 | 65 vs 62 |
| 0.8 | 198 vs 198 | 1.54 vs 1.92 | 11.72 vs 12.95 | 40.1 vs 36.8 | +0.17 vs +0.27 | 63 vs 68 |
| 0.9 | 118 vs 67 | 0.25 vs 0.18 | 2.26 vs 2.49 | 22.7 vs 22.4 | +0.59 vs +0.40 | 78 vs 79 |

### Module 2, section 2.1, the core shared by a whole category

`MEAN` rows of `category_core_overlap.csv`, in percent, and the categories whose every member shares at least one expert.

| AP | shared by four members | of which also the name's | empty cores | categories with an all-member core | neurons in those cores |
|---|---|---|---|---|---|
| 0.5 | 1.03 vs 0.97 | 73.0 vs 77.6 | 1 vs 1 | 6 vs 5 | 232 vs 177 |
| 0.6 | 0.50 vs 0.50 | 68.9 vs 74.3 | 9 vs 14 | 2 vs 1 | 3 vs 2 |
| 0.7 | 0.14 vs 0.16 | 55.8 vs 61.2 | 45 vs 50 | 0 vs 0 | 0 vs 0 |

### Module 2, section 2.2, every word pair

Whole model, `sublayer_comparison.csv` of section 2.2, Jaccard in percent.

| AP | within-category Jaccard | across-category Jaccard | ratio | Jaccard ROC-AUC | layer-profile ROC-AUC | layer-profile $z$ ROC-AUC |
|---|---|---|---|---|---|---|
| 0.5 | 7.48 vs 6.96 | 1.120 vs 0.997 | 6.7 vs 7.0 | 0.959 vs 0.958 | 0.734 vs 0.693 | 0.723 vs 0.683 |
| 0.6 | 5.15 vs 5.06 | 0.479 vs 0.428 | 10.8 vs 11.8 | 0.955 vs 0.951 | 0.725 vs 0.698 | 0.718 vs 0.696 |
| 0.7 | 3.27 vs 3.36 | 0.199 vs 0.183 | 16.4 vs 18.3 | 0.927 vs 0.921 | 0.720 vs 0.712 | 0.715 vs 0.709 |
| 0.8 | 1.68 vs 1.73 | 0.084 vs 0.081 | 19.9 vs 21.2 | 0.795 vs 0.761 | 0.696 vs 0.703 | 0.695 vs 0.707 |
| 0.9 | 0.42 vs 0.42 | 0.023 vs 0.021 | 18.5 vs 19.7 | 0.558 vs 0.546 | 0.656 vs 0.656 | 0.696 vs 0.684 |

### Module 2, section 2.2, Jaccard against layer-profile agreement over all pairs

`correlation_summary.csv` of section 2.2.

| AP | pairs | Spearman, $J$ against $S$ | Pearson, $J$ against $S$ | Spearman, $J$ against $z$ |
|---|---|---|---|---|
| 0.5 | 21,115 vs 21,115 | 0.396 vs 0.357 | 0.336 vs 0.309 | 0.427 vs 0.357 |
| 0.6 | 21,115 vs 21,115 | 0.356 vs 0.331 | 0.273 vs 0.253 | 0.363 vs 0.323 |
| 0.7 | 21,115 vs 21,115 | 0.350 vs 0.320 | 0.208 vs 0.197 | 0.315 vs 0.280 |
| 0.8 | 20,910 vs 21,115 | 0.336 vs 0.321 | 0.147 vs 0.147 | 0.241 vs 0.233 |
| 0.9 | 19,110 vs 17,955 | 0.175 vs 0.158 | 0.082 vs 0.080 | 0.122 vs 0.116 |

### Module 2, section 2.2, category-mean matrix

AP 0.6, `jaccard_category_mean_matrix.csv`, members only, in percent.

| quantity | value |
|---|---|
| smallest diagonal cell | 2.66 vs 2.48 |
| largest diagonal cell | 7.47 vs 7.87 |
| median off-diagonal cell | 0.37 vs 0.32 |
| fruit and vegetables | 2.85 vs 2.70 |
| sports and vehicles | 0.96 vs 0.94 |

### Module 2, per sublayer

AP 0.6, the `sublayer_comparison.csv` files of sections 2.2 and 2.3.

| scope | experts | Jaccard ROC-AUC | layer-profile ROC-AUC | human $\rho$, Jaccard |
|---|---|---|---|---|
| whole model | 569,820 vs 416,626 | 0.955 vs 0.951 | 0.725 vs 0.698 | 0.503 vs 0.477 |
| mlp.gate_proj | 246,860 vs 193,354 | 0.951 vs 0.946 | 0.719 vs 0.700 | 0.485 vs 0.468 |
| mlp.up_proj | 154,275 vs 103,078 | 0.947 vs 0.939 | 0.700 vs 0.671 | 0.529 vs 0.515 |
| self_attn.o_proj | 62,659 vs 42,925 | 0.920 vs 0.896 | 0.713 vs 0.696 | 0.522 vs 0.460 |
| mlp.down_proj | 44,336 vs 33,941 | 0.691 vs 0.630 | 0.596 vs 0.587 | 0.279 vs 0.230 |
| self_attn.q_proj | 24,348 vs 17,245 | 0.786 vs 0.698 | 0.642 vs 0.603 | 0.349 vs 0.245 |
| self_attn.v_proj | 21,930 vs 15,402 | 0.863 vs 0.790 | 0.660 vs 0.624 | 0.368 vs 0.331 |
| self_attn.k_proj | 15,412 vs 10,681 | 0.797 vs 0.722 | 0.640 vs 0.604 | 0.315 vs 0.283 |

### Module 2, the profile metric registry

AP 0.5, mean $z$ of the member-to-name pairs (`hierarchy/profile_metric_comparison.csv`), category ROC-AUC and human agreement of the pairwise matrices (`heatmaps/profile_metric_comparison.csv`).

| metric | member-to-name mean $z$ | category ROC-AUC | human $\rho$ |
|---|---|---|---|
| `js_distance` | +0.583 vs +0.381 | 0.7342 vs 0.6931 | 0.1109 vs 0.0471 |
| `wasserstein` | +0.377 vs +0.306 | 0.6324 vs 0.5963 | 0.0580 vs 0.0285 |
| `cosine` | +0.528 vs +0.276 | 0.7015 vs 0.6526 | 0.0890 vs 0.0388 |
| `pearson` | +0.407 vs +0.189 | 0.6736 vs 0.6445 | 0.1040 vs 0.0544 |
| `spearman` | +0.217 vs -0.001 | 0.6688 vs 0.6541 | 0.1172 vs 0.0630 |
| `js_divergence` | +0.532 vs +0.352 | 0.7342 vs 0.6931 | 0.1109 vs 0.0471 |
| `hellinger` | +0.579 vs +0.383 | 0.7345 vs 0.6936 | 0.1108 vs 0.0467 |
| `profile_jaccard` | +0.661 vs +0.425 | 0.7273 vs 0.6857 | 0.1166 vs 0.0449 |

### Module 2, section 2.3, agreement with human similarity

`human_similarity_validation.csv`, Spearman over the rated within-category pairs, the mean over the eight categories and its ratio to the noise ceiling.

| AP | pooled | mean | mean / ceiling | layer profile, mean | significant categories of 8 | rated pairs |
|---|---|---|---|---|---|---|
| 0.5 | 0.620 vs 0.594 | 0.532 vs 0.508 | 0.597 vs 0.570 | 0.111 vs 0.047 | 8 vs 8 | 2,418 vs 2,418 |
| 0.6 | 0.546 vs 0.520 | 0.503 vs 0.477 | 0.564 vs 0.535 | 0.096 vs 0.042 | 8 vs 7 | 2,418 vs 2,418 |
| 0.7 | 0.499 vs 0.482 | 0.532 vs 0.509 | 0.597 vs 0.572 | 0.100 vs 0.070 | 8 vs 8 | 2,418 vs 2,418 |
| 0.8 | 0.476 vs 0.454 | 0.529 vs 0.485 | 0.594 vs 0.544 | 0.100 vs 0.099 | 8 vs 8 | 2,418 vs 2,418 |
| 0.9 | 0.326 vs 0.309 | 0.333 vs 0.306 | 0.374 vs 0.343 | 0.139 vs 0.143 | 8 vs 7 | 2,370 vs 2,294 |

### Module 2, section 2.3, the probing-text baseline

`POOLED_MEAN` rows of `human_similarity_text_baseline.csv`.

| AP | text against humans | Jaccard against humans | Jaccard against text | Jaccard against humans, text partialled out |
|---|---|---|---|---|
| 0.5 | 0.643 vs 0.643 | 0.532 vs 0.508 | 0.710 vs 0.694 | 0.147 vs 0.128 |
| 0.6 | 0.643 vs 0.643 | 0.503 vs 0.477 | 0.676 vs 0.650 | 0.139 vs 0.123 |
| 0.7 | 0.643 vs 0.643 | 0.532 vs 0.509 | 0.671 vs 0.655 | 0.194 vs 0.172 |
| 0.8 | 0.643 vs 0.643 | 0.529 vs 0.485 | 0.578 vs 0.542 | 0.266 vs 0.231 |
| 0.9 | 0.639 vs 0.647 | 0.333 vs 0.306 | 0.316 vs 0.295 | 0.191 vs 0.168 |

### Module 2, section 2.4, cosine typicality against human typicality

`global_prototype_typicality.csv`, pooled Pearson and the per-category correlations.

| AP | members | pooled r | p | positive categories of 8 | significantly positive |
|---|---|---|---|---|---|
| 0.5 | 198 vs 198 | +0.452 vs +0.423 | <0.001 vs <0.001 | 8 vs 8 | 4 vs 4 |
| 0.6 | 198 vs 198 | +0.386 vs +0.360 | <0.001 vs <0.001 | 8 vs 7 | 5 vs 6 |
| 0.7 | 198 vs 198 | +0.288 vs +0.283 | <0.001 vs <0.001 | 7 vs 7 | 6 vs 6 |
| 0.8 | 198 vs 198 | +0.194 vs +0.191 | 0.006 vs 0.007 | 6 vs 7 | 3 vs 4 |
| 0.9 | 196 vs 193 | +0.042 vs +0.046 | 0.56 vs 0.53 | 4 vs 4 | 1 vs 1 |

### Module 2, section 2.4, typicality against category alignment

Pearson $r$, the partials controlling frequency and then frequency and category.

| AP | typicality against Jaccard | typicality against overlap | partial (frequency) | partial (frequency + category) | Jaccard against cosine typicality |
|---|---|---|---|---|---|
| 0.5 | 0.480 vs 0.439 | 0.452 vs 0.435 | 0.497 vs 0.455 | 0.458 vs 0.420 | 0.586 vs 0.633 |
| 0.6 | 0.402 vs 0.371 | 0.381 vs 0.390 | 0.413 vs 0.380 | 0.415 vs 0.397 | 0.627 vs 0.696 |
| 0.7 | 0.315 vs 0.299 | 0.372 vs 0.370 | 0.321 vs 0.304 | 0.397 vs 0.389 | 0.627 vs 0.704 |
| 0.8 | 0.346 vs 0.333 | 0.370 vs 0.361 | 0.351 vs 0.337 | 0.395 vs 0.386 | 0.381 vs 0.501 |
| 0.9 | gated (60%) vs gated (34%) | gated (60%) vs gated (34%) | gated (60%) vs gated (34%) | gated (60%) vs gated (34%) | gated (60%) vs gated (34%) |

### Module 2, sections 2.4 and 2.5, expert count and frequency

Pearson $r$ from the two `correlation_summary.csv` files, "gated" where the coverage rule withholds the panel.

| AP | frequency against expert count | typicality against expert count | frequency against typicality |
|---|---|---|---|
| 0.5 | -0.358 vs -0.384 | -0.001 vs 0.041 | 0.002 vs 0.002 |
| 0.6 | -0.376 vs -0.418 | 0.054 vs 0.076 | 0.002 vs 0.002 |
| 0.7 | -0.380 vs -0.429 | 0.087 vs 0.096 | 0.002 vs 0.002 |
| 0.8 | -0.376 vs -0.386 | 0.092 vs 0.086 | 0.002 vs 0.002 |
| 0.9 | -0.222 vs -0.204 | -0.010 vs -0.021 | -0.006 vs -0.008 |

### Module 2, section 2.5, frequency against Jaccard

FineWeb Zipf frequency against each member's Jaccard with its category name, raw and with human typicality partialled out, then with category added as a second control.

| AP | raw | partial (typicality) | partial (typicality + category) |
|---|---|---|---|
| 0.5 | 0.259 vs 0.271 | 0.294 vs 0.301 | 0.340 vs 0.344 |
| 0.6 | 0.240 vs 0.227 | 0.261 vs 0.244 | 0.296 vs 0.273 |
| 0.7 | 0.200 vs 0.180 | 0.210 vs 0.188 | 0.183 vs 0.166 |
| 0.8 | 0.179 vs 0.160 | 0.190 vs 0.169 | 0.120 vs 0.108 |

### Module 3, category name against member average

Block axis, `dual_category_jsd_by_block.csv`, Pearson $r$ over the categories.

| AP | categories | mean JSD | name entropy against JSD | member diversity against JSD |
|---|---|---|---|---|
| 0.5 | 8 vs 8 | 0.021 vs 0.027 | -0.43 vs -0.66 | -0.25 vs -0.02 |
| 0.6 | 8 vs 8 | 0.044 vs 0.055 | -0.61 vs -0.57 | +0.10 vs +0.28 |
| 0.7 | 8 vs 8 | 0.106 vs 0.119 | -0.28 vs -0.58 | +0.06 vs +0.34 |
| 0.8 | 8 vs 8 | 0.403 vs 0.375 | -0.95 vs -0.91 | -0.23 vs -0.11 |
| 0.9 | 5 vs 3 | 0.685 vs 0.624 | -0.98 vs -0.99 | +0.34 vs +0.99 |

### Module 4, expert sets against the embedding geometry

Analysis sublayer `mlp.gate_proj`, `embedding_rsa_summary.csv`, plateau in blocks of 28.

| AP | $\rho$ at the middle layer | peak $\rho$ | density association | middle minus outer | plateau |
|---|---|---|---|---|---|
| 0.5 | 0.661 vs 0.654 | 0.758 vs 0.745 | 0.77 vs 0.68 | -0.033 vs -0.030 | 17 vs 19 |
| 0.6 | 0.564 vs 0.550 | 0.646 vs 0.624 | 0.80 vs 0.73 | -0.033 vs -0.030 | 24 vs 25 |
| 0.7 | 0.467 vs 0.456 | 0.522 vs 0.511 | 0.82 vs 0.78 | -0.025 vs -0.023 | 28 vs 28 |
| 0.8 | 0.343 vs 0.333 | 0.388 vs 0.374 | 0.92 vs 0.91 | -0.020 vs -0.016 | 28 vs 28 |
| 0.9 | 0.169 vs 0.147 | 0.189 vs 0.166 | 0.82 vs 0.85 | -0.010 vs -0.008 | 28 vs 27 |

### Module 4, per sublayer

AP 0.5, `sublayer_comparison.csv` of module 4.

| sublayer | expert units | $\rho$ full | $\rho$ count-matched |
|---|---|---|---|
| mlp.gate_proj | 113,019 vs 97,581 | 0.661 vs 0.654 | 0.348 vs 0.323 |
| mlp.up_proj | 112,178 vs 91,445 | 0.653 vs 0.648 | 0.382 vs 0.364 |
| self_attn.o_proj | 40,248 vs 35,547 | 0.538 vs 0.489 | 0.417 vs 0.357 |
| self_attn.k_proj | 13,637 vs 10,598 | 0.483 vs 0.438 | 0.483 vs 0.438 |
| self_attn.q_proj | 24,880 vs 19,571 | 0.483 vs 0.420 | 0.405 vs 0.332 |
| self_attn.v_proj | 15,199 vs 12,381 | 0.442 vs 0.449 | 0.422 vs 0.424 |
| mlp.down_proj | 33,268 vs 28,980 | 0.283 vs 0.264 | 0.177 vs 0.155 |

### Module 5, Studies A and B

AP 0.5, default metric, Study A ranking accuracy in percent under the two references and Study B mean per-category Spearman.

| cell | Study A, label word | Study A, centroid | Study B |
|---|---|---|---|
| jaccard | 67.0 vs 65.8 | 65.5 vs 64.1 | 0.5318 vs 0.5081 |
| profile | 52.8 vs 43.1 | 49.7 vs 53.2 | 0.1109 vs 0.0471 |
| profile_z | 47.9 vs 41.8 | 46.6 vs 51.3 | 0.1168 vs 0.0693 |
| jaccard_profile | 67.6 vs 65.9 | 66.0 vs 67.7 | 0.5442 vs 0.5408 |
| jaccard_profile_z | 66.5 vs 65.3 | 64.8 vs 64.8 | 0.5202 vs 0.5038 |
| jaccard_profile_both | 66.7 vs 65.1 | 67.7 vs 67.7 | 0.5769 vs 0.5770 |

### Module 5, metric comparison

AP 0.5, the fullest cell `jaccard_profile_both` per registered metric.

| metric | Study A, label word | Study A, centroid | Study B |
|---|---|---|---|
| `js_distance` | 0.6675 vs 0.6505 | 0.6770 vs 0.6766 | 0.5769 vs 0.5770 |
| `wasserstein` | 0.6617 vs 0.6642 | 0.6431 vs 0.6522 | 0.5521 vs 0.5486 |
| `cosine` | 0.6696 vs 0.6439 | 0.6778 vs 0.6737 | 0.5787 vs 0.5680 |
| `pearson` | 0.6687 vs 0.6472 | 0.6497 vs 0.6716 | 0.5385 vs 0.5348 |
| `spearman` | 0.6543 vs 0.6299 | 0.6427 vs 0.6572 | 0.5245 vs 0.5113 |
| `js_divergence` | 0.6754 vs 0.6485 | 0.6493 vs 0.6704 | 0.5718 vs 0.5737 |
| `hellinger` | 0.6675 vs 0.6493 | 0.6758 vs 0.6758 | 0.5772 vs 0.5773 |
| `profile_jaccard` | 0.6638 vs 0.6559 | 0.6754 vs 0.6787 | 0.5795 vs 0.5769 |

### Module 5, Studies A and B across AP thresholds

Default metric, the `jaccard` cell beside the fullest cell, and the members Study A fits.

| AP | A, label word, Jaccard | A, label word, full | A, centroid, Jaccard | A, centroid, full | B, Jaccard | B, full | members in Study A |
|---|---|---|---|---|---|---|---|
| 0.5 | 67.0 vs 65.8 | 66.7 vs 65.1 | 65.5 vs 64.1 | 67.7 vs 67.7 | 0.532 vs 0.508 | 0.577 vs 0.577 | 198 vs 198 |
| 0.6 | 67.1 vs 66.4 | 66.0 vs 67.8 | 64.8 vs 63.9 | 66.0 vs 68.0 | 0.503 vs 0.477 | 0.542 vs 0.513 | 198 vs 198 |
| 0.7 | 68.5 vs 67.8 | 64.4 vs 64.3 | 65.2 vs 65.3 | 67.8 vs 67.9 | 0.532 vs 0.509 | 0.483 vs 0.402 | 198 vs 198 |
| 0.8 | 69.9 vs 67.5 | 67.0 vs 58.0 | 59.5 vs 60.7 | 59.8 vs 62.2 | 0.529 vs 0.485 | 0.398 vs 0.351 | 170 vs 198 |
| 0.9 | 43.0 vs 42.0 | 46.6 vs 38.1 | 34.1 vs 64.7 | 41.4 vs 51.0 | 0.248 vs 0.184 | 0.263 vs 0.181 | 50 vs 47 |

### Module 5, Study C across AP thresholds

Whole model, `dendrogram_summary.csv` and `linkgraph_summary.csv`, profile columns for the default metric.

| AP | expert-set ARI | profile ARI | profile $\eta^2$ depth | expert-set graph within-category | profile graph within-category |
|---|---|---|---|---|---|
| 0.5 | 0.772 vs 0.772 | 0.194 vs 0.056 | 0.459 vs 0.465 | 92.6 vs 91.4 | 47.1 vs 41.0 |
| 0.6 | 0.749 vs 0.718 | 0.162 vs 0.150 | 0.560 vs 0.535 | 91.6 vs 90.5 | 47.1 vs 43.7 |
| 0.7 | 0.643 vs 0.676 | 0.141 vs 0.130 | 0.308 vs 0.397 | 87.5 vs 86.7 | 45.6 vs 42.4 |
| 0.8 | 0.392 vs 0.459 | 0.132 vs 0.002 | 0.499 vs 0.090 | 74.7 vs 74.1 | 34.7 vs 36.9 |
| 0.9 | 0.000 vs 0.000 | 0.000 vs 0.001 | 0.084 vs 0.112 | 59.3 vs 55.0 | 29.5 vs 27.0 |

### Module 5, Study C spaces at AP 0.5

Every clustered space, `dendrogram_summary.csv`.

| space | cophenetic | ARI against category | $\eta^2$ depth | category silhouette |
|---|---|---|---|---|
| `js_distance` | 0.679 vs 0.669 | 0.194 vs 0.056 | 0.459 vs 0.465 | +0.015 vs -0.002 |
| `js_distance_z` | 0.642 vs 0.559 | 0.098 vs 0.121 | 0.344 vs 0.406 | -0.014 vs -0.018 |
| `wasserstein` | 0.742 vs 0.745 | 0.071 vs 0.095 | 0.887 vs 0.923 | -0.085 vs -0.115 |
| `cosine` | 0.664 vs 0.691 | 0.131 vs 0.056 | 0.603 vs 0.631 | -0.005 vs -0.047 |
| `pearson` | 0.743 vs 0.720 | 0.105 vs 0.053 | 0.592 vs 0.627 | -0.053 vs -0.072 |
| `spearman` | 0.802 vs 0.784 | 0.121 vs 0.035 | 0.294 vs 0.286 | -0.072 vs -0.075 |
| `js_divergence` | 0.615 vs 0.615 | 0.092 vs 0.109 | 0.482 vs 0.510 | +0.018 vs -0.012 |
| `hellinger` | 0.678 vs 0.650 | 0.204 vs 0.119 | 0.439 vs 0.538 | +0.015 vs -0.002 |
| `profile_jaccard` | 0.706 vs 0.686 | 0.086 vs 0.046 | 0.508 vs 0.514 | +0.009 vs -0.003 |
| `expert_set_jaccard` | 0.920 vs 0.917 | 0.772 vs 0.772 | 0.364 vs 0.346 | +0.047 vs +0.044 |
