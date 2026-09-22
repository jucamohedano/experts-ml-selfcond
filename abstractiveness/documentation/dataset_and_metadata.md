# Dataset and metadata provenance

## What this page covers

Every number the pipeline produces is model-derived except four, and those four come from outside the model: the **concept list** that defines what is studied, the **human pairwise similarity judgments** used as the external validation target, the **typicality scores** derived from those judgments, and the **word frequencies** used as a covariate and as a control. This page documents where each one came from, what was done to it, what it covers and does not cover, and how to regenerate it.

It is the counterpart of module 1, subchapter 1.1, which documents the other half of the provenance chain, namely how sentences become responses and responses become expert sets. That subchapter starts from a word list and takes it forward. This page ends where that one begins.

Two stimulus sets exist in the repository. The **Richie-HSJ** set (205 words) is the one both models are currently run on and is the subject of most of this page. The older **150-concept** set (164 words) is documented in the last section for completeness, since the older results tables cited across the module pages come from it.

## 1. Source materials at a glance

| File | What it is | Feeds |
|---|---|---|
| `assets/Richie_and_Bhatia-HSJ/table 1 - word lists.csv` | The 8 categories and their members, Table 1 of the source paper | the concept list, the prompt dataset config, category assignment |
| `assets/Richie_and_Bhatia-HSJ/study1_pairwise_data/data_individual_level/<Category>_pairwise.csv` | Study 1 total-set pairwise similarity ratings, one row per subject, one column per pair | `typicality_HSJ_pairwise`, module 5.3, module 9 Study B |
| `assets/Richie_and_Bhatia-HSJ/study1_spam_data.csv` | Study 1 Spatial Arrangement Method data, one row per subject per category with final (x, y) per item | `typicality_HSJ_spam` |
| `assets/THINGS-database/osfstorage/03_category-level/typicality53_mean-ratings.tsv` | THINGS database mean typicality ratings, keyed by (member, category) | the `typicality` column |
| `assets/enwiki-2023-04-13.txt` | English Wikipedia word count list, one `word count` pair per line | the `frequency` column |
| `assets/metadata_Richie_HSJ.json` | **Output.** One row per word with all covariates merged | every module, through `load_experts_data` |
| `dataset_config_Qwen3-30B-A3B-Instruct-2507-abstractiveness_Richie_HSJ.json` | **Output.** The same 205 words plus the sentence generation parameters | the sentence generation stage |

`assets/` is excluded by `.gitignore`, so none of these files travel with the repository and they have to be placed by hand on a fresh checkout. Of the five inputs, the Richie and Bhatia directory is present in the working tree today. The Wikipedia count file and the THINGS directory are **not**, so the `frequency` and `typicality` values currently in `metadata_Richie_HSJ.json` are the only surviving copies of what those two sources contributed, and regenerating the metadata from scratch requires fetching them again. Section 7 says what has to be recovered and in what order.

## 2. The concept list

**Source.** All 8 categories and their members come verbatim from Table 1 of Richie, White, Bhatia and Hout's "SpAM on words" materials, which is where the similarity judgments come from too. Using one source for both the stimuli and the validation target is deliberate, because it guarantees that every word studied has a human similarity rating against every other member of its category, with no partial coverage and no cross-dataset name matching.

**Structure.** The CSV is one column per category, with members listed down the column and shorter categories padded with blanks. `load_categories_from_wordlist` reads it column by column, drops the blanks, strips, and lowercases, so every word enters the pipeline as a lowercase surface form.

**Two abstraction levels.** Each category contributes two kinds of row. The category name itself becomes a **level 1** item (`abstraction_level: 1`, `category: null`), and each member becomes a **level 2** item (`abstraction_level: 2`, `category` set to its parent). The level 1 words are ordinary words as far as the rest of the pipeline is concerned, they get their own sentences and their own expert sets, and the comparison between the two levels is the central question of the project.

**Counts.**

| Category | In the source table | In the dataset | Note |
|---|---|---|---|
| furniture | 20 | 20 | |
| clothing | 29 | 29 | |
| birds | 30 | 30 | |
| vegetables | 20 | 20 | includes `squash` |
| sports | 28 | 27 | `squash` dropped |
| vehicles | 22 | 22 | |
| fruit | 21 | 21 | |
| professions | 28 | 28 | |
| **total members** | **198** | **197** | |
| plus level 1 labels | | **8** | |
| **total items** | | **205** | |

**The one edit made to the source list.** The table lists `squash` twice, once as a vegetable and once as a sport. A single surface form cannot carry two expert sets, because the sentences describing one sense would sit in the negative pool of the other and the Average Precision of every unit would be computed against a corrupted label. The word is admitted once, under **vegetables**, because vegetables is the smaller category (19 other members against 27 for sports), so the exclusion costs the smaller category less imbalance than dropping the word entirely would. The rule lives in `EXCLUDED_SENSES` in `prepare_metadata_richie_hsj.py` so that regenerating the metadata cannot reintroduce the sports row, and the same exclusion is repeated in `EXCLUDED_MEMBERS` in `utils/human_similarity.py` so that the sports pairs containing `squash` are dropped from the ratings too. The full history of this decision, including the silent skip in `compute_responses.py` that it originally surfaced, is in [fixes.md](fixes.md).

**Word senses are a separate problem from word strings.** The list gives strings, not meanings. Sentences are generated from a WordNet gloss, and WordNet sense ordering does not know which category the word was filed under, so a word can silently acquire sentences about the wrong meaning (`fencing` the sport against `fencing` the material, `date` the fruit against `date` the appointment). That failure is invisible downstream, because the file is complete and the counts are right, only the meaning is wrong. `scripts/audit_concept_senses.py` screens for it with a leave-one-out TF-IDF check against the category profile, and sixteen concepts were regenerated on the strength of it. This is documented in [fixes.md](fixes.md) rather than here, because it is a property of the generated sentences rather than of the source list.

## 3. Human pairwise similarity judgments

**What was collected.** In Study 1 of the source paper, participants rated the similarity of **within-category word pairs on a 1 to 7 scale, where higher means more similar**. That direction matches every expert-side metric in this pipeline, so a positive correlation is the expected sign everywhere and nothing is flipped at the call site. The rating is of **similarity**, not typicality. Neither study asked anyone to rate typicality directly, which is why section 5 has to derive it.

**File layout.** One CSV per category under `study1_pairwise_data/data_individual_level/`. The first column is the subject identifier, and every other column is one pair, with the header written as `word_a \ word_b`. Each cell is that subject's rating of that pair, and an empty cell means that subject was not asked about that pair.

**Design and coverage.** Four categories (fruit, furniture, vegetables, vehicles) were run total-set, so essentially every subject rated every pair. The four larger categories (birds, clothing, professions, sports) used a **half-sampled between-subjects** design, so around 44 to 52 percent of the subject-by-pair cells are empty by design rather than by attrition.

| Category | Subjects | Pairs used | Raters per pair, min | max | Empty cells |
|---|---|---|---|---|---|
| birds | 54 | 435 | 19 | 29 | 51.9% |
| clothing | 61 | 406 | 27 | 32 | 50.0% |
| fruit | 31 | 210 | 29 | 31 | 0.3% |
| furniture | 33 | 190 | 32 | 33 | 0.2% |
| professions | 67 | 378 | 32 | 39 | 43.6% |
| sports | 61 | 351 | 28 | 37 | 44.6% |
| vegetables | 30 | 190 | 29 | 30 | 0.0% |
| vehicles | 28 | 231 | 26 | 28 | 1.5% |
| **total** | | **2,391** | | | |

Pair counts are exactly $\binom{n}{2}$ for each category's membership, so the files are complete total-set designs with no pair missing. Sports shows 351 rather than $\binom{28}{2} = 378$ because the 27 pairs containing `squash` are dropped, and 2,391 is the same pair count module 9's ranker enumerates combinatorially from the same word list, which is the cheapest available check that the two sides agree on membership.

**Unequal rater counts are harmless here.** Different pairs carrying different numbers of raters makes some pair means noisier than others, and noise in a target **attenuates** a correlation rather than inflating it, so the risk runs against the hypothesis rather than for it. The noise ceiling below is what turns that attenuation into a number that can be divided out.

**How the pipeline reads it.** `utils/human_similarity.py` owns all access. `load_human_similarity()` returns one row per rated pair with columns `category`, `word_a`, `word_b`, `mean_rating`, `n_raters`, where the mean is taken over the subjects who actually rated that pair. Pair keys are put in canonical alphabetical order by `_parse_pair`, so a lookup never depends on which way round the header happened to be written, and `human_pair_lookup()` exposes them as a dict for fast per-pair joins. Nothing in the module depends on the AP threshold or the analysis scope, so both entry points are memoised, which matters because the executor calls them 40 times per model and the answer never changes.

**Sanity checks on the loaded ratings.** Verified by inspection rather than assumed. Mean ratings span 1.09 to 6.61. The highest are `gloves`/`mittens` at 6.61 and `educator`/`teacher` at 6.60, and the lowest are `chess`/`running` at 1.09 and `boxing`/`fishing` at 1.13. The scale is oriented the way the docstring claims.

**Noise ceiling.** `human_noise_ceiling(n_splits=200, seed=0)` estimates, per category, the largest correlation any model could achieve against these ratings. Subjects are split at random into halves, each half is averaged per pair, the two pair vectors are correlated by Spearman over the pairs both halves cover (splits with 10 or fewer overlapping pairs are discarded), and the mean over 200 splits is corrected from half-sample to full-sample reliability by Spearman-Brown, $\rho_{\text{full}} = 2\rho_{\text{half}} / (1 + \rho_{\text{half}})$.

| Category | Ceiling |
|---|---|
| birds | 0.836 |
| professions | 0.866 |
| sports | 0.883 |
| clothing | 0.893 |
| furniture | 0.899 |
| vegetables | 0.909 |
| fruit | 0.910 |
| vehicles | 0.935 |

Reporting a raw correlation against these ratings without stating it as a fraction of the ceiling understates the model, because part of the residual is disagreement between the humans themselves rather than model error.

**Where the ratings are used.** Module 5, subchapter 5.3 correlates each pairwise expert matrix against the ratings per category, with a Mantel test permuting concept labels within the category. Module 9, Study B regresses the ratings on symmetric pair features. Both report per-category results next to any pooled figure, for the reason recorded in [module_9](module_9_typicality_prediction.md).

## 4. Spatial Arrangement Method data

**What was collected.** The same Study 1 also ran a SpAM task, in which each subject arranged one category's words on a two-dimensional surface so that closer means more similar. This is an independent elicitation of the same underlying similarity structure, using a different response mode.

**File layout.** `study1_spam_data.csv` is 432 rows, one per subject per category, which is 54 subjects across each of the 8 categories. There are 30 stimulus slots, `Stim1` to `Stim30`, each paired with `Object<i>XFinal` and `Object<i>YFinal` giving the position the subject dropped that item at. Categories with fewer than 30 members pad the unused slots with the sentinel item `.....` parked at `(3840, 2160)`, and `compile_typicality_hsj.py` drops those rows before computing anything.

**Coverage gap.** The SpAM stimulus set is not identical to the word list. Vehicles was arranged with 20 items rather than the list's 22, because `truck` and `van` are absent from the SpAM stimuli. Those are exactly the two words carrying `typicality_HSJ_spam: null`, so the column covers 195 of 197 concepts while `typicality_HSJ_pairwise` covers all 197.

**Status.** `typicality_HSJ_spam` is computed and stored but is not the active typicality column in any current run. Replicating module 5.3 against this second elicitation is listed as open work in [CLAUDE.md](../CLAUDE.md), because agreement between two elicitation methods would strengthen the positive result and disagreement would bound it.

## 5. The typicality columns

Three typicality columns exist in the metadata, and only one of them is used by the Richie-HSJ runs. All three are per-concept scalars, null for every level 1 category label, since a root node has no typicality within itself.

### 5.1 `typicality_HSJ_pairwise`, the active column

Neither Richie study measured typicality, and the published paper never reduces the pairwise data to a single per-item value either, it only builds the aggregate similarity matrix and feeds that into cross-validated MDS. This column is therefore **our own derivation**, using the standard family-resemblance operationalization from categorization research (Rosch and Mervis, 1975): an item's typicality within a category is its average similarity to the other members of that category.

For a category $k$ with members $M_k$, let $r_s(a,b)$ be subject $s$'s rating of pair $(a,b)$ and $R_{s}(a,b)$ the set of subjects who rated it. The aggregate similarity matrix is built first, one subject-averaged value per pair,

$$\bar{r}(a,b) = \frac{1}{|R_s(a,b)|} \sum_{s \in R_s(a,b)} r_s(a,b),$$

and the raw score of a word is the mean of its row and column in that matrix,

$$\tau^{\text{raw}}_k(a) = \frac{1}{|M_k| - 1} \sum_{b \in M_k,\, b \neq a} \bar{r}(a,b).$$

Averaging over subjects **before** averaging over partners is the step that matters, and it matches the paper's own aggregate-matrix construction. It gives every pair equal weight in the word's score, which is necessary precisely because the four half-sampled categories have different numbers of raters per pair, and a naive mean over raw cells would silently weight heavily-rated pairs more.

The raw score is then min-max normalized **within each category**,

$$\tau_k(a) = \frac{\tau^{\text{raw}}_k(a) - \min_{c \in M_k} \tau^{\text{raw}}_k(c)}{\max_{c \in M_k} \tau^{\text{raw}}_k(c) - \min_{c \in M_k} \tau^{\text{raw}}_k(c)},$$

rounded to three decimals, so 1 is the most typical member of its category and 0 the least. A category whose members all scored identically would give every member 0.5, though that does not occur in this data.

The within-category normalization is the important caveat to carry downstream. The column is **ordinal within a category and not comparable across categories**, because the most typical bird and the most typical profession both score 1.0 regardless of how tightly either category holds together. Any analysis pooling this column across categories is reading a rank, not a level, which is the reason module 9's Study A is formulated as a within-category pairwise ranking rather than a regression on the raw value.

Coverage is 197 of 197 concepts.

### 5.2 `typicality_HSJ_spam`, the second elicitation

The same family-resemblance logic applied to the arrangement data. For each subject's trial, the Euclidean distance from a word's position to every other item's position in that trial is averaged, then averaged across subjects, which is the row mean of the aggregate SpAM distance matrix. Distance is negated so that closer means more typical, and the result is min-max normalized within category exactly as above. Coverage is 195 of 197, missing `truck` and `van`.

### 5.3 `typicality`, the THINGS ratings

A third and fully external source, the THINGS database's mean typicality ratings, keyed by `(member, category)`. These are **direct** typicality ratings rather than a derivation from similarity, which is what makes them worth carrying even at partial coverage.

A rating is used **only** when both the word and its Richie category map onto the matching THINGS category, through an explicit mapping, never by averaging a word's ratings across unrelated THINGS categories. The mapping is deliberately incomplete:

| Richie category | THINGS category |
|---|---|
| furniture | furniture |
| clothing | clothing |
| birds | bird |
| vegetables | vegetable |
| vehicles | vehicle |
| fruit | fruit |
| sports | none |
| professions | none |

The keyed lookup exists because a word such as `cabinet` is rated in THINGS under both `container` and `furniture`, and blending those into one cross-category average would produce a number that describes neither. Sports and professions have no THINGS equivalent at all and always resolve to null.

Coverage after the mapping is 96 of 197 concepts:

| Category | Concepts | With THINGS rating |
|---|---|---|
| furniture | 20 | 7 |
| clothing | 29 | 20 |
| birds | 30 | 16 |
| vegetables | 20 | 18 |
| sports | 27 | 0 |
| vehicles | 22 | 20 |
| fruit | 21 | 15 |
| professions | 28 | 0 |

### 5.4 Which column each run uses

`MODEL_CONFIGS` in `abstractiveness_executor.py` names the column through `typicality_column`, and `load_experts_data` renames whatever is named to `human_typicality` for every downstream module, so module code never references a source-specific name. Both Richie-HSJ configs use `typicality_HSJ_pairwise`, for its full coverage and because it was elicited on exactly these words. The older `gpt2_150` config uses `typicality`, the THINGS column. Anywhere the module pages say Human Typicality, that is the renamed column, and which source it came from is decided here.

## 6. Word frequency

**Source file.** `assets/enwiki-2023-04-13.txt`, a plain text English Wikipedia word count list dated 2023-04-13, one entry per line as a word followed by whitespace and an integer count. `load_frequencies_from_file` parses it with `rsplit(None, 1)`, lowercases the key, and discards any line whose trailing token is not an integer. Both `prepare_metadata_richie_hsj.py` and the older `scripts/prepare_metadata.py` read the same file through identical copies of that function.

**Provenance gap to close.** The file is not in the working tree and no fetch command or upstream URL is recorded anywhere in the repository, so the only surviving record of what it contained is the `frequency` column of the two metadata JSONs. The file name matches the naming convention of the published English Wikipedia word-frequency dumps, but that is an inference from the name rather than something this repository records, and it should be confirmed and written down here before anyone tries to reproduce the metadata or extend the stimulus set. Until then, treat the frequency column as reproducible only by reusing the existing JSON.

**Coverage and range.** Every one of the 205 words resolves, 197 of 197 concepts and 8 of 8 category labels, with no nulls and no zeros. Counts run from 183 (`footstool`) to 605,641 (`radio`), median 14,099, with `minister` at 582,021 and `sports` at 418,627 next below the maximum.

**How it is used.** Module 1's `save_expert_counts_metadata` maps the raw count to $\tilde{f}_c = \log_{10} f_c$, computed only when $f_c > 0$, and drops rows with a missing or non-positive count from the merged table. The log transform compresses the heavy right tail so that module 4's linear correlations are not dominated by a handful of very frequent words. Frequency appears as a predictor in its own right (does a more frequent word recruit more experts) and as a control, in module 4's frequency-partialled correlations, where the question is whether a typicality effect survives once frequency is held fixed.

**Three caveats the column carries.**

- These are **raw corpus counts**, not a per-million rate and not a Zipf value, so they are meaningful only relatively and only within this one corpus snapshot. Nothing in the pipeline compares them against an external frequency norm.
- Matching is on the **lowercased surface form** with no lemmatization, so `bird` and `birds` are different keys. Since the level 1 label words are the category names as written in the source table, several of them are plural (`birds`, `sports`, `vegetables`) while their members are singular, and the level 1 versus level 2 frequency comparison inherits that asymmetry.
- Homographs are **conflated**, because the corpus count cannot distinguish senses. `squash` carries the count of both the vegetable and the sport even though the dataset admits only the vegetable, and the same applies to every word the sense audit touched.

## 7. Regenerating the metadata

Two scripts, in this order, both run from `abstractiveness/scripts/`.

```bash
python prepare_metadata_richie_hsj.py   # writes assets/metadata_Richie_HSJ.json
python compile_typicality_hsj.py        # adds the two HSJ columns in place
```

The first reads the word list, the frequency file and the THINGS ratings and **writes** the JSON, producing the `concept`, `abstraction_level`, `frequency`, `typicality` and `category` fields. The second reads the pairwise and SpAM data and **updates the same file in place**, adding `typicality_HSJ_pairwise` and `typicality_HSJ_spam`. Running them in the other order loses the HSJ columns, since the first script rewrites the file wholesale.

Both print coverage counts on completion, and those are the check to read. Expect 205 entries from the first and `197/197` plus `195/197` from the second. Both scripts warn and continue when an input file is missing rather than failing, so a silent all-null column is the failure mode to watch for, and the printed counts are what catches it.

Neither script regenerates sentences, responses or expertise. Changing the stimulus list means regenerating all three, which is the expensive path documented in module 1, subchapter 1.1, and the dataset config must be kept aligned one to one with the metadata or `compute_responses.py` will skip the mismatched word without raising anything.

## 8. The metadata schema

`assets/metadata_Richie_HSJ.json` is a flat JSON list of 205 objects, 8 at level 1 and 197 at level 2.

| Field | Type | Null when | Meaning |
|---|---|---|---|
| `concept` | string | never | the lowercased word, the join key against every expert table |
| `abstraction_level` | int, 1 or 2 | never | 1 for a category label, 2 for a member concept |
| `frequency` | int | never in this file | raw Wikipedia count $f_c$ |
| `typicality` | float in [0, 1] | level 1, and level 2 outside the THINGS mapping | THINGS mean typicality rating |
| `category` | string | level 1 | parent category of a level 2 concept |
| `typicality_HSJ_pairwise` | float in [0, 1] | level 1 | family-resemblance score from the pairwise ratings |
| `typicality_HSJ_spam` | float in [0, 1] | level 1, and `truck`, `van` | family-resemblance score from the SpAM arrangements |

A level 1 row and a level 2 row:

```json
{
    "concept": "furniture",
    "abstraction_level": 1,
    "frequency": 44365,
    "typicality": null,
    "category": null,
    "typicality_HSJ_pairwise": null,
    "typicality_HSJ_spam": null
}
{
    "concept": "bed",
    "abstraction_level": 2,
    "frequency": 60242,
    "typicality": 0.765,
    "category": "furniture",
    "typicality_HSJ_pairwise": 0.707,
    "typicality_HSJ_spam": 0.229
}
```

Two kinds of null have to be told apart when reading any downstream table, and module 4 makes the distinction explicitly. A null **in this file** is a genuine gap in the human data and is independent of the AP threshold. A word **missing from a results table** is a word whose expert set came out empty at that threshold, which is a model-side outcome documented in module 1, subchapter 1.1. Both show up as a shrunken sample size, and only the second one moves when the threshold moves.

## 9. Known limitations

- **The typicality target is derived, not measured.** `typicality_HSJ_pairwise` is a family-resemblance proxy computed from similarity ratings, not a typicality rating. It is a defensible and standard operationalization, but any claim about typicality rests on that operationalization rather than on a direct measurement. The THINGS column is the only directly measured typicality in the repository, and it covers under half the concepts and none of two categories.
- **The derived typicality columns are within-category ordinal.** Min-max normalization per category destroys the between-category level. Pooled analyses of this column measure rank agreement, not agreement in degree.
- **The similarity ratings are within-category only.** No pair crossing two categories was ever rated, so the human target can validate the structure the model recovers inside a category and says nothing about the distance between categories.
- **The frequency source is unrecorded.** See section 6. Reproducible today only by reusing the existing JSON.
- **Frequency is a surface-form count.** No lemmatization, no sense disambiguation, and plural category labels compared against singular members.
- **One word was edited out of the source list.** `squash` under sports, for a reason internal to this pipeline rather than to the source data, which makes the sports category 27 members here against 28 in the published table.
- **The two elicitations do not cover the same words.** SpAM is missing `truck` and `van`, so a comparison of the two typicality columns is over 195 concepts, not 197.

## 10. The older 150-concept dataset

The GPT-2 runs cited as `research_plots_150_*` across the module pages use a different stimulus set, kept for continuity with the earlier results rather than for new work.

`assets/metadata_150.json` holds 164 entries, 17 level 1 categories (`animal`, `clothing`, `container`, `drink`, `fastener`, `food`, `furniture`, `game`, `hardware`, `headwear`, `jewelry`, `lighting`, `plant`, `tool`, `toy`, `vehicle`, `weapon`) and 147 level 2 concepts. The schema is the five original fields only, with no HSJ columns, since no pairwise similarity data exists for these words.

It is built by `scripts/prepare_metadata.py` from `conf/concept_group/abstractiveness_150.yaml` rather than from a word-list CSV, and it draws `frequency` from the same Wikipedia count file and `typicality` from the same THINGS ratings file. Its THINGS coverage is complete, 147 of 147, which is unsurprising given that the category vocabulary was chosen to match THINGS in the first place. That completeness is why the `gpt2_150` config can use `typicality` as its typicality column where the Richie-HSJ configs cannot.

The trade made when moving to Richie-HSJ was coverage of a direct typicality rating against availability of pairwise human similarity judgments. The 150-concept set has the former and none of the latter, so modules 5.3, 8 and 9's Study B have no human target on it at all.
