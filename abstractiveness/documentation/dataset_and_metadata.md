# Dataset and metadata provenance

## What this page covers

Every number the pipeline produces is model-derived except four, and those four come from outside the model: the **concept list** that defines what is studied, the **human pairwise similarity judgments** used as the external validation target, the **typicality scores** derived from those judgments, and the **word frequencies** used as a covariate and as a control. This page documents where each one came from, what was done to it, what it covers and does not cover, and how to regenerate it.

It is the counterpart of module 1's *From responses to expert sets*, which documents the other half of the provenance chain, namely how sentences become responses and responses become expert sets. That part starts from a word list and takes it forward. This page ends where that one begins.

Two stimulus sets exist in the repository. The **Richie-HSJ** set (206 concepts over 205 distinct words) is the one both models are currently run on and is the subject of most of this page. The older **150-concept** set (164 words) is documented in the last section for completeness, since the older results tables cited across the module pages come from it.

## 1. Source materials at a glance

| File | What it is | Feeds |
|---|---|---|
| `assets/Richie_and_Bhatia-HSJ/table 1 - word lists.csv` | The 8 categories and their members, Table 1 of the source paper | the concept list, the prompt dataset config, category assignment |
| `assets/Richie_and_Bhatia-HSJ/study1_pairwise_data/data_individual_level/<Category>_pairwise.csv` | Study 1 total-set pairwise similarity ratings, one row per subject, one column per pair | `typicality_HSJ_pairwise`, module 2 section 2.3, module 5 Study B |
| `assets/Richie_and_Bhatia-HSJ/study1_spam_data.csv` | Study 1 Spatial Arrangement Method data, one row per subject per category with final (x, y) per item | `typicality_HSJ_spam` |
| `assets/THINGS-database/osfstorage/03_category-level/typicality53_mean-ratings.tsv` | THINGS database mean typicality ratings, keyed by (member, category) | the `typicality` column of the older 150-concept set only |
| `assets/enwiki-2023-04-13.txt` | English Wikipedia word count list, one `word count` pair per line | the `frequency` column and the Wikipedia Zipf columns |
| `assets/SUBTLEX-US.txt` | SUBTLEX-US subtitle frequencies, 74,286 entries | the primary Zipf columns |
| `assets/SUBTLEX-UK.txt` | SUBTLEX-UK subtitle frequencies, 160,022 entries | the British reference Zipf column |
| `assets/openwebtext-word-counts.txt` | OpenWebText (`Skylion007/openwebtext`) word count list in the Wikipedia format, built by `corpus_word_counts`, with a `.json` provenance sidecar | `frequency_zipf_openwebtext_lemma`, the GPT-2 exposure proxy |
| `assets/fineweb-sample-10BT-word-counts.txt` | FineWeb sample-10BT (`HuggingFaceFW/fineweb`) word count list in the Wikipedia format, built by `corpus_word_counts`, with a `.json` provenance sidecar | `frequency_zipf_fineweb_lemma`, the Qwen3 exposure proxy |
| `assets/metadata_Richie_HSJ.json` | **Output.** One row per word with all covariates merged | every module, through `load_experts_data` |
| `dataset_config_Qwen3-30B-A3B-Instruct-2507-abstractiveness_Richie_HSJ.json` | **Output.** The same 206 concepts plus the sentence generation parameters | the sentence generation stage |

`assets/` is excluded by `.gitignore`, so none of these files travel with the repository and they have to be placed by hand on a fresh checkout. The Richie and Bhatia directory, the Wikipedia count file and the two SUBTLEX files are all present in the working tree today, and section 6.2 records where the Wikipedia file came from. The THINGS directory is **not** present, which no longer blocks anything here, since the Richie-HSJ metadata dropped the THINGS `typicality` column and nothing in this dataset reads it. Only the older 150-concept set in section 10 still needs that file.

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
| sports | 28 | 28 | includes `squash` |
| vehicles | 22 | 22 | |
| fruit | 21 | 21 | |
| professions | 28 | 28 | |
| **total members** | **198** | **198** | |
| plus level 1 labels | | **8** | |
| **total items** | | **206** | |

**The one word listed twice.** The table lists `squash` under both vegetables and sports. An expert set is defined per concept, so two senses need two concept identities, and one surface form cannot supply them: both sentence files would be named `squash.json`, and every table merging on `concept` would fold the senses together. Pooling both senses' sentences under one concept is no remedy, because the positive class would then describe two unrelated things and the expert set would be a mixture belonging to neither category. Both senses are therefore admitted as **separate concepts**, keyed `squash__vegetables` and `squash__sports`, each with its own sentences and its own expert set. The key is built in `core/data_preparation/build_concept_metadata.py` from `SENSE_SEPARATOR` whenever a surface form appears under more than one category, and the same key is what `storage_key_for` in `generate_definitions_dspy.py` already used for the on-disk filename, so the two now agree.

The separation runs on the **concept** field while the bare surface form stays in **`word`**. Anything lexical reads `word`, which is what WordNet is queried with, what the sentence generator sends to the model, and what the frequency tables are keyed on. Anything joining against an expert-side table reads `concept`. `select_wordnet_sense` resolves the two senses through its per-category anchors, returning `squash.n.02` (edible fruit of a squash plant) for vegetables and `squash.n.03` (a game played in an enclosed court) for sports. The earlier decision to admit only the vegetable, and the silent skip in `compute_responses.py` that it originally surfaced, is in [fixes.md](fixes.md).

**Word senses are a separate problem from word strings.** The list gives strings, not meanings. Sentences are generated from a WordNet gloss, and WordNet sense ordering does not know which category the word was filed under, so a word can silently acquire sentences about the wrong meaning (`fencing` the sport against `fencing` the material, `date` the fruit against `date` the appointment). That failure is invisible downstream, because the file is complete and the counts are right, only the meaning is wrong. `scripts/core/data_preparation/audit_stimulus_word_senses.py` screens for it with a leave-one-out TF-IDF check against the category profile, and sixteen concepts were regenerated on the strength of it. This is documented in [fixes.md](fixes.md) rather than here, because it is a property of the generated sentences rather than of the source list.

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
| sports | 61 | 378 | 28 | 37 | 44.6% |
| vegetables | 30 | 190 | 29 | 30 | 0.0% |
| vehicles | 28 | 231 | 26 | 28 | 1.5% |
| **total** | | **2,418** | | | |

Pair counts are exactly $\binom{n}{2}$ for each category's membership, so the files are complete total-set designs with no pair missing, sports included at $\binom{28}{2} = 378$ now that both squash senses are in the dataset. The total of 2,418 is the same pair count module 5's ranker enumerates combinatorially from the same word list, which is the cheapest available check that the two sides agree on membership. Because one word can map to two concepts, every row carries `concept_a` and `concept_b` beside `word_a` and `word_b`, and a join against an expert-side table must use the concept columns.

**Unequal rater counts are harmless here.** Different pairs carrying different numbers of raters makes some pair means noisier than others, and noise in a target **attenuates** a correlation rather than inflating it, so the risk runs against the hypothesis rather than for it. The noise ceiling below is what turns that attenuation into a number that can be divided out.

**How the pipeline reads it.** `core/human_similarity_ratings.py` owns all access. `load_human_similarity()` returns one row per rated pair with columns `category`, `word_a`, `word_b`, `mean_rating`, `n_raters`, where the mean is taken over the subjects who actually rated that pair. Pair keys are put in canonical alphabetical order by `_parse_pair`, so a lookup never depends on which way round the header happened to be written, and `human_pair_lookup()` exposes them as a dict for fast per-pair joins. Nothing in the module depends on the AP threshold or the analysis scope, so both entry points are memoised, which matters because the executor calls them 40 times per model and the answer never changes.

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

**Where the ratings are used.** Module 2, section 2.3, correlates each pairwise expert matrix against the ratings per category, with a Mantel test permuting concept labels within the category. Module 5, Study B, regresses the ratings on symmetric pair features. Both report per-category results next to any pooled figure, for the reason recorded in [module_5](module_5_typicality_prediction.md).

## 4. Spatial Arrangement Method data

**What was collected.** The same Study 1 also ran a SpAM task, in which each subject arranged one category's words on a two-dimensional surface so that closer means more similar. This is an independent elicitation of the same underlying similarity structure, using a different response mode.

**File layout.** `study1_spam_data.csv` is 432 rows, one per subject per category, which is 54 subjects across each of the 8 categories. There are 30 stimulus slots, `Stim1` to `Stim30`, each paired with `Object<i>XFinal` and `Object<i>YFinal` giving the position the subject dropped that item at. Categories with fewer than 30 members pad the unused slots with the sentinel item `.....` parked at `(3840, 2160)`, and `typicality_covariates.py` drops those rows before computing anything.

**Coverage gap.** The SpAM stimulus set is not identical to the word list. Vehicles was arranged with 20 items rather than the list's 22, because `truck` and `van` are absent from the SpAM stimuli. Those are exactly the two words carrying `typicality_HSJ_spam: null`, so the column covers 195 of 197 concepts while `typicality_HSJ_pairwise` covers all 197.

**Status.** `typicality_HSJ_spam` is computed and stored but is not the active typicality column in any current run. Replicating section 2.3 against this second elicitation is listed as open work in [CLAUDE.md](../CLAUDE.md), because agreement between two elicitation methods would strengthen the positive result and disagreement would bound it.

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

The within-category normalization is the important caveat to carry downstream. The column is **ordinal within a category and not comparable across categories**, because the most typical bird and the most typical profession both score 1.0 regardless of how tightly either category holds together. Any analysis pooling this column across categories is reading a rank, not a level, which is the reason module 5's Study A is formulated as a within-category pairwise ranking rather than a regression on the raw value.

Coverage is 198 of 198 concepts.

### 5.2 `typicality_HSJ_spam`, the second elicitation

The same family-resemblance logic applied to the arrangement data. For each subject's trial, the Euclidean distance from a word's position to every other item's position in that trial is averaged, then averaged across subjects, which is the row mean of the aggregate SpAM distance matrix. Distance is negated so that closer means more typical, and the result is min-max normalized within category exactly as above. Coverage is 195 of 197, missing `truck` and `van`.

### 5.3 `typicality`, the THINGS ratings, retired from this dataset

**This column is no longer written into `metadata_Richie_HSJ.json`.** It survives only in `metadata_150.json`, which the `gpt2_150` config still uses, and the rest of this subchapter describes it as it exists there. It was dropped here because both Richie-HSJ configs use `typicality_HSJ_pairwise`, so the column was carried at 96 of 198 coverage without a single reader, and its builder silently produced an all-null column whenever the THINGS file was absent, which it is.

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

`MODEL_CONFIGS` in `abstractiveness_executor.py` names the column through `typicality_column`, and `load_experts_data` renames whatever is named to `human_typicality` for every downstream module, so module code never references a source-specific name. Both Richie-HSJ configs use `typicality_HSJ_pairwise`, for its full coverage and because it was elicited on exactly these words. The older `gpt2_150` config uses `typicality`, the THINGS column, which now exists only in that set's own metadata file. Anywhere the module pages say Human Typicality, that is the renamed column, and which source it came from is decided here.

## 6. Word frequency

Five corpora are read, and they do four different jobs. SUBTLEX-US stands for the exposure of the human raters, OpenWebText and FineWeb sample-10BT stand for the pretraining exposure of GPT-2 and Qwen3 respectively, Wikipedia is the written-register reference the earlier analyses were run on, and SUBTLEX-UK is a dialect reference. The word counts of the two web corpora are produced by `scripts/core/data_preparation/corpus_word_counts.py`, and every column is computed by `scripts/core/data_preparation/frequency_covariates.py`. Neither file carries rationale of its own, both point here instead.

### 6.1 Which corpus, and why

**SUBTLEX-US is the human-side source.** 51 million tokens of American film and television subtitles over 74,286 entries, from Brysbaert and New (2009), which is the most widely used modern English frequency norm. The stimulus list and the human similarity ratings both come from Richie and Bhatia, an American study run on American participants, so the frequency covariate sitting beside those ratings should be American as well.

That choice was measured rather than assumed. Substituting the British norms moves the per-category Spearman correlation between `typicality_HSJ_pairwise` and frequency by 0.075 on average, and by 0.215 for professions, which falls from 0.276 to 0.061. The mechanism is visible word by word. British English uses `minister` overwhelmingly for a government minister rather than for clergy, at Zipf 5.44 against 4.27, it prefers solicitor and barrister to `lawyer`, and it says vet rather than `veterinarian`. No category changes the sign of its correlation under either corpus, so the qualitative conclusions are stable either way, but the magnitudes are not, and professions in particular would be misreported.

Those figures compare the two surface-form columns, the listed word alone on both sides. Repeated with singular and plural summed on both sides by the rule of section 6.4, over the same words, the mean shift is 0.080 and professions falls from 0.264 to 0.065, so the gap is a dialect effect and not an artefact of the lemma rule. Lemmatisation alone moves these correlations by only 0.024 within SUBTLEX-US and 0.016 within SUBTLEX-UK. The tagger-based lemma counts SUBTLEX-UK ships in its `DomPoSLemma` columns must not be substituted for the section 6.4 rule. They count only the dominant part of speech, which scores `cabinet`, `turkey` and `minister` as names, and they map 7 of the 205 words to another lemma, `running` to run, `walking` to walk, `dove` to dive, `boxers` to boxer, which widens the gap to 0.107.

**OpenWebText is the GPT-2 exposure proxy.** The confound the frequency panels control for is whether a model allocated more units to a word because it met that word more often during pretraining, which is model exposure and not human familiarity. GPT-2 was trained on WebText, which OpenAI never released, and the sample of it OpenAI once hosted is no longer downloadable. OpenWebText is the open reproduction of WebText, built from the URLs of Reddit submissions, deduplicated, restricted to English with FastText, cleared of near-duplicate documents by locality-sensitive hashing on 5-grams and of documents under 128 tokens, so it is the closest available stand-in for what GPT-2 read. Its published procedure states neither WebText's three-karma threshold nor WebText's removal of Wikipedia documents, so those two properties of the original are not guaranteed in the reproduction.

**FineWeb sample-10BT is the Qwen3 exposure proxy.** Qwen3 was pretrained on about 36 trillion tokens whose composition is described only by broad source category, with web text the bulk of it. FineWeb is the standard open reconstruction of modern Common Crawl pretraining data, 15 trillion tokens from 96 snapshots between 2013 and 2024, and sample-10BT is a random 10 billion token sample of it. It approximates the kind of text Qwen3 was trained on, not its exact content, since the multilingual, synthetic and distilled parts of Qwen3's training have no open counterpart.

**Wikipedia is the written-register reference.** 2.47 billion tokens. It was the pretraining proxy before the two web corpora were counted, and it is kept so that earlier results stay reproducible and the raw `frequency` column keeps its meaning. It is a weaker proxy for both models, for GPT-2 because WebText excluded Wikipedia, and for both because encyclopedic prose is a narrower register than web text. Wikipedia and SUBTLEX-US correlate at only r = 0.73 across the stimulus set, while both web corpora reach 0.86 against SUBTLEX-US, see section 6.8.

**SUBTLEX-UK is a reference column only.** 201 million tokens of BBC broadcast subtitles, from van Heuven, Mandera, Keuleers and Brysbaert (2014), the paper that introduced the Zipf scale. It is carried so that the British against American gap described above can be tabulated as the stated limitation it is. No analysis should be run on it.

### 6.2 Provenance of the count files

`assets/enwiki-2023-04-13.txt` is `results/enwiki-2023-04-13.txt` from [github.com/IlyaSemenov/wikipedia-word-frequency](https://github.com/IlyaSemenov/wikipedia-word-frequency), MIT licensed. The identification rests on four independent matches: the filename is exact and is the only `enwiki` file that repository publishes, the top twenty words appear in the same order as the repository's README reports them, the format is the same `word count` pair per line, and the minimum count in the file is 3, which is the repository's stated floor of appearing in at least three articles. The README quotes 2,747,823 types against the 2,765,377 this file holds, a gap of 0.6 percent that reflects a stale README figure rather than a different file.

The file totals **2,765,377 types over 2,474,589,909 tokens**. Three properties of the extraction carry into every number derived from it.

- **Counts are token counts, not document counts.** `the` occurs 186,631,452 times against the roughly 6.6 million articles English Wikipedia held in 2023, which settles it.
- **Words appearing in fewer than three articles are absent entirely**, rather than present with a low count. This is harmless here, since the rarest stimulus is `footstool` at 183.
- **Punctuation is stripped, unicode dashes and apostrophes are normalised, and tokens containing digits are dropped**, so these counts are not directly comparable against a raw dump.

This closes the provenance gap earlier revisions of this page flagged as open.

**The two web count files** were built on 1 October 2026 by `corpus_word_counts.py`, which downloads a corpus at the dataset commit current on that day and records it, together with every total below, in a `.json` sidecar next to the count file. The script applies exactly the rules of the Wikipedia list's own `gather_wordfreq.py`, the en dash and right single quotation mark normalised to a hyphen and an apostrophe, the text lowercased, split on every character that is not a letter, digit, underscore, hyphen or apostrophe, a token kept only if it starts and ends with a word character or is a single one, tokens containing a digit dropped, and a word kept only if it appears in at least three documents. The Zipf denominator is the sum of the kept counts, as for Wikipedia, so the three written-corpus columns share one scale and differ only in the text they were counted on.

| | OpenWebText | FineWeb sample-10BT |
|---|---|---|
| Hugging Face dataset | `Skylion007/openwebtext` | `HuggingFaceFW/fineweb`, `sample/10BT` |
| Revision | `79d93d786212f7344586290adb811d4ae6a1762c` | `9bb295ddab0e05d785b879661af7260fed5140fc` |
| Parquet files | 80, 24.19 GB | 15, 30.64 GB |
| Documents | 8,013,769 | 14,868,862 |
| Tokens counted | 6,396,110,620 | 7,508,557,048 |
| Tokens kept, the Zipf denominator | 6,378,456,703 (99.72%) | 7,485,930,592 (99.70%) |
| Types kept | 3,054,509 | 3,877,756 |

OpenWebText's document total equals the 8,013,769 its dataset card publishes, which confirms the whole corpus was read. The rarest stimuli remain well attested, `footstool` at 552 occurrences in OpenWebText and `hovercraft` the rarest in FineWeb at Zipf 2.48.

### 6.3 The Zipf scale

Every frequency column is reported as a Zipf value, which is the base ten logarithm of occurrences per billion words, so that

$$\mathrm{Zipf} = \log_{10}(f_{\text{per million}}) + 3.$$

For a raw count in a corpus of $N$ tokens this is $\log_{10}(f) + \log_{10}(10^9 / N)$, which for the Wikipedia file means $\mathrm{Zipf} = \log_{10}(f) - 0.3935$. `wikipedia_corpus_tokens()` sums the file rather than hardcoding that constant, so replacing the dump cannot silently leave the values scaled to the wrong corpus. Across the 205 stimuli the Wikipedia column runs from 1.87 for `footstool` to 5.39 for `radio`, against a conventional reading of below 3 as low frequency and above 4 as high.

One consequence is worth stating plainly, because it prevents a wasted comparison. Zipf is an affine transform of $\log_{10} f$, so every Pearson correlation, Spearman correlation, partial correlation and significance test is numerically identical to what the raw `frequency` column already produced. The scale buys interpretability and comparability with published norms, and nothing else. Only the summation in 6.4 changes a number.

### 6.4 Summing singular and plural

A concept is counted across every surface form that carries its sense, and across no form that does not. Where both the singular and the plural denote the concept, their counts are summed. Where only one form does, only that form is counted. The decision is held in three explicit tables in `frequency_covariates.py` rather than inferred, because every automatic alternative tested was wrong on this stimulus set. A part of speech tagged lemma map, for instance, both invented forms, folding a stray `bi` into `bus`, and missed real ones, dropping `mirrors` at 15,777 and `ministers` at 73,239 whenever the tagger read the plural as a verb or a surname.

**Regular inflection.** `plural_of()` applies the ordinary English suffix rules, adding `es` after a sibilant so that `bus` gives `buses` and `ostrich` gives `ostriches`, turning a consonant plus `y` into `ies` for `cherry` and `canary`, and turning a final `f` into `ves` for `scarf`.

**`IRREGULAR_PLURALS`** holds the five the rules get wrong, namely `fireman`, `policeman` and `postman`, which take `men`, and `tomato` and `potato`, which take `oes`.

**`PLURAL_FORM_LISTED`** holds the thirteen concepts the source table writes in the plural whose singular denotes the same thing, and so is summed with it. Five are the category labels `birds`, `professions`, `sports`, `vegetables` and `vehicles`, and the rest are `beans`, `gloves`, `grapes`, `mittens`, `pajamas`, `panties`, `sneakers` and `socks`.

**`SINGLE_FORM`** holds the forty-two concepts scored on the listed form alone, in three groups.

| Group | Members | Why |
|---|---|---|
| Mass nouns | `furniture`, `clothing`, `asparagus`, `broccoli`, `cauliflower`, `celery`, `corn`, `lettuce`, `spinach` | no count plural, and the generated form either does not occur or is under one percent of the singular |
| Activity nouns | all 27 sports | the activity has no count plural, and the plurals that do exist denote something else, since ballets are works and runnings are instances |
| Garments with a divergent singular | `boots`, `boxers`, `jeans`, `overalls`, `pants`, `shorts` | the singular is a different word or sense |

That last group is where an unguarded summation does the most damage, and the Wikipedia counts show the scale of it. `overall` the adverb outnumbers `overalls` the garment by 298 to 1, `short` the adjective outnumbers `shorts` by 26 to 1, `jean` as a name outnumbers `jeans` by 16 to 1, and `boxer` the fighter and dog breed outnumbers `boxers` the garment by 4.6 to 1. `boots` and `pants` are admitted to the same group on the same reasoning, since `boot` is largely a car boot or a verb and `pant` is a verb. Summing any of these would move the word further than the entire real spread of its category.

The net effect is 163 concepts summed over two forms and 42 scored on one. The step shifts the Wikipedia column by a median of 0.118 Zipf and correlates with the unsummed column at r = 0.989, so it is a correctness measure rather than one that will move a result. The largest single correction is `professions`, which the surface form undercounts badly because the source table writes the label in the plural while the word is normally used in the singular, at 3.61 Zipf against 4.31 after summing, and 9.04 per million against 0.37 in SUBTLEX-US.

### 6.5 Coverage

| Column | Coverage | Gap |
|---|---|---|
| `frequency_zipf_wikipedia` | 206 of 206 | |
| `frequency_zipf_wikipedia_lemma` | 206 of 206 | |
| `frequency_zipf_subtlex_us` | 205 of 206 | `physiotherapist` |
| `frequency_zipf_subtlex_us_lemma` | 205 of 206 | `physiotherapist` |
| `frequency_zipf_subtlex_uk` | 206 of 206 | |
| `frequency_zipf_openwebtext_lemma` | 206 of 206 | |
| `frequency_zipf_fineweb_lemma` | 206 of 206 | |

`physiotherapist` is the British term for what American subtitles render as the two words physical therapist, which a unigram list cannot hold. It resolves to null rather than being dropped from the dataset, because every module already drops rows with a missing covariate on a per analysis basis, so the null costs one point in the frequency analyses alone, taking professions to 26 there, and keeps the concept in the expert set, Average Precision, Jaccard and typicality analyses where nothing is missing.

### 6.6 How it is used

Frequency enters as a predictor in its own right, asking whether a more frequent word recruits more experts, and as a control in section 2.4's frequency-partialled correlation, asking whether a typicality effect survives once frequency is held fixed. It never touches expert extraction, so changing a frequency column cannot alter an Average Precision or an expert set, only the analyses that read it.

Module 2 reads one frequency column per run, the one matched to the model through the config's `frequency_corpus` field, `frequency_zipf_openwebtext_lemma` for GPT-2 and `frequency_zipf_fineweb_lemma` for Qwen3. It is the x variable of the section 2.5 figure, the frequency panel of section 2.4, the control of the section 2.4 partial correlation, and the variable of the section 2.5 partial correlation, which holds human typicality fixed instead. The SUBTLEX-US, Wikipedia and SUBTLEX-UK columns stay in the metadata for reference and for the source comparison of section 6.8, and no analysis reads them.

The raw `frequency` column is retained unchanged so that earlier results stay reproducible. `expert_counts_with_metadata` in `core/expert_data_loading.py` still maps it to $\log_{10} f_c$ and drops rows with a missing or non-positive count.

### 6.7 Caveats the columns carry

- **Homographs are conflated**, because a corpus count cannot separate senses. `squash__vegetables` and `squash__sports` are separate concepts with separate expert sets, but they share one Wikipedia count and one SUBTLEX rate, since no corpus frequency list distinguishes the two. The same applies to every word the sense audit touched.
- **The British column is not register matched** to this stimulus set, by 0.215 Spearman in professions, which is why it is a reference column rather than an analysis column.
- **Wikipedia is a formal register.** Corpora built from written material overestimate the frequency of formal words relative to how often a person meets them (Brysbaert, Mandera and Keuleers, 2018). This is the accepted cost of choosing a corpus that approximates pretraining exposure instead of human familiarity.
- **Corpus choice matters more than the summation.** The Wikipedia and SUBTLEX-US columns correlate at only r = 0.73, so roughly half the variance is corpus specific and a frequency-partialled result genuinely can differ between them. Reporting both is what turns that into evidence rather than an unexamined choice.

### 6.8 How the sources compare

`compare_frequency_sources.py` correlates every pair of lemma Zipf columns and writes `results/frequency_sources/frequency_source_correlations.csv`. The unit is the unique surface word, so `squash` counts once and n is 205, or 204 for every pair with SUBTLEX-US, which lacks `physiotherapist`. Because the three written-corpus columns share one scale, the mean difference is a meaningful offset as well as the correlations.

| Source a | Source b | n | Pearson r | Spearman ρ | Mean Zipf difference, a minus b |
|---|---|---|---|---|---|
| Wikipedia | OpenWebText | 205 | 0.892 | 0.891 | 0.079 |
| Wikipedia | FineWeb | 205 | 0.811 | 0.814 | −0.106 |
| Wikipedia | SUBTLEX-US | 204 | 0.727 | 0.739 | 0.045 |
| OpenWebText | FineWeb | 205 | 0.947 | 0.947 | −0.185 |
| OpenWebText | SUBTLEX-US | 204 | 0.863 | 0.875 | −0.033 |
| FineWeb | SUBTLEX-US | 204 | 0.864 | 0.877 | 0.150 |

Three readings follow. The two web corpora agree closely with each other, at r = 0.947, so the choice between them matters far less than the choice between web text and Wikipedia. Both web corpora agree with the subtitle norms much better than Wikipedia does, 0.86 against 0.73, which places encyclopedic prose as the outlier register among the four rather than speech. And FineWeb sits 0.185 Zipf above OpenWebText on these words, so the two proxies differ in level as well as in rank, which matters for any analysis that reads absolute values rather than correlations.

## 7. Regenerating the metadata

Run from `abstractiveness/scripts/`, in this order.

```bash
python -m core.data_preparation.corpus_word_counts openwebtext   # assets/openwebtext-word-counts.txt and .json
python -m core.data_preparation.corpus_word_counts fineweb       # assets/fineweb-sample-10BT-word-counts.txt and .json
python -m core.data_preparation.build_concept_metadata           # writes assets/metadata_Richie_HSJ.json
python -m core.data_preparation.compare_frequency_sources        # results/frequency_sources/frequency_source_correlations.csv
```

The two counting runs are needed only when a count file is missing or a corpus is to be recounted. They download 24.19 GB and 30.64 GB of parquet into `assets/corpora/`, kept for later recounts, and then count on CPU in about 12 and 14 minutes with the default ten workers. `--workers` lowers the parallelism if memory runs short. `build_concept_metadata` raises a `FileNotFoundError` naming the missing count file rather than writing a null column.

`build_concept_metadata` is the package's only entry point. It reads the word list, builds the `concept`, `word`, `abstraction_level` and `category` fields itself, then composes the columns the other two preparers own by calling `frequency_columns` and `typicality_columns`. All three follow the same contract, `load_*` for a memoised reader of one external input and `<name>_columns` for the metadata fields that preparer owns, so adding a covariate means writing one more `*_columns` function and one more `row.update` call.

It prints coverage on completion and those counts are the check to read. Expect 206 entries, `frequency_zipf_subtlex_us_lemma` at 205 of 206, `frequency_zipf_wikipedia_lemma`, `frequency_zipf_openwebtext_lemma` and `frequency_zipf_fineweb_lemma` at 206 of 206, `typicality_HSJ_pairwise` at 198 of 198 and `typicality_HSJ_spam` at 196 of 198. A silent all-null column is the failure mode to watch for when an input file is missing, and the printed counts are what catches it.

Neither script regenerates sentences, responses or expertise. Changing the stimulus list means regenerating all three, which is the expensive path documented in module 1's *From responses to expert sets*, and the dataset config must be kept aligned one to one with the metadata or `compute_responses.py` will skip the mismatched word without raising anything.

## 8. The metadata schema

`assets/metadata_Richie_HSJ.json` is a flat JSON list of 206 objects, 8 at level 1 and 198 at level 2.

| Field | Type | Null when | Meaning |
|---|---|---|---|
| `concept` | string | never | the join key against every expert table. The lowercased word, or `word__category` when the source table lists that word under two categories |
| `word` | string | never | the bare lowercased surface form. What WordNet, the sentence generator and the frequency tables read |
| `abstraction_level` | int, 1 or 2 | never | 1 for a category label, 2 for a member concept |
| `frequency` | int | never in this file | raw Wikipedia count $f_c$ of the listed surface form |
| `frequency_zipf_subtlex_us_lemma` | float | `physiotherapist` | SUBTLEX-US Zipf, singular and plural summed by section 6.4 |
| `frequency_zipf_wikipedia_lemma` | float | never | Wikipedia Zipf, singular and plural summed by section 6.4 |
| `frequency_zipf_openwebtext_lemma` | float | never | OpenWebText Zipf, the GPT-2 exposure proxy, summed by section 6.4 |
| `frequency_zipf_fineweb_lemma` | float | never | FineWeb sample-10BT Zipf, the Qwen3 exposure proxy, summed by section 6.4 |
| `category` | string | level 1 | parent category of a level 2 concept |
| `typicality_HSJ_pairwise` | float in [0, 1] | level 1 | family-resemblance score from the pairwise ratings |
| `typicality_HSJ_spam` | float in [0, 1] | level 1, and `truck`, `van` | family-resemblance score from the SpAM arrangements |

A level 1 row and a level 2 row:

```json
{
    "concept": "furniture",
    "word": "furniture",
    "abstraction_level": 1,
    "frequency": 44365,
    "category": null,
    "typicality_HSJ_pairwise": null,
    "typicality_HSJ_spam": null
}
{
    "concept": "bed",
    "word": "bed",
    "abstraction_level": 2,
    "frequency": 60242,
    "category": "furniture",
    "typicality_HSJ_pairwise": 0.707,
    "typicality_HSJ_spam": 0.229
}
```

Two kinds of null have to be told apart when reading any downstream table, and module 2's coverage rule makes the distinction explicitly. A null **in this file** is a genuine gap in the human data and is independent of the AP threshold. A word **missing from a results table** is a word whose expert set came out empty at that threshold, which is a model-side outcome documented in module 1's *From responses to expert sets*. Both show up as a shrunken sample size, and only the second one moves when the threshold moves.

## 9. Known limitations

- **The typicality target is derived, not measured.** `typicality_HSJ_pairwise` is a family-resemblance proxy computed from similarity ratings, not a typicality rating. It is a defensible and standard operationalization, but any claim about typicality rests on that operationalization rather than on a direct measurement. The THINGS column was the only directly measured typicality in the repository, and it has been retired from this dataset, having covered under half the concepts and none of two categories.
- **The derived typicality columns are within-category ordinal.** Min-max normalization per category destroys the between-category level. Pooled analyses of this column measure rank agreement, not agreement in degree.
- **The similarity ratings are within-category only.** No pair crossing two categories was ever rated, so the human target can validate the structure the model recovers inside a category and says nothing about the distance between categories.
- **Frequency carries no sense disambiguation.** Singular and plural are summed where both carry the concept's sense, as section 6.4 sets out, but a corpus count still cannot separate two senses of one spelling, so `squash` carries the sport as well as the vegetable.
- **The British frequency column is not register matched** to this American stimulus set, by 0.215 Spearman in professions, and is carried as a reference column only.
- **One word carries two concepts.** `squash` appears under both vegetables and sports and is admitted as two concepts sharing one surface form, so it contributes two expert sets, two typicality scores and one frequency value. Nothing else in the stimulus set is ambiguous this way.
- **The two elicitations do not cover the same words.** SpAM is missing `truck` and `van`, so a comparison of the two typicality columns is over 195 concepts, not 197.

## 10. The older 150-concept dataset

The GPT-2 runs cited as `research_plots_150_*` across the module pages use a different stimulus set, kept for continuity with the earlier results rather than for new work.

`assets/metadata_150.json` holds 164 entries, 17 level 1 categories (`animal`, `clothing`, `container`, `drink`, `fastener`, `food`, `furniture`, `game`, `hardware`, `headwear`, `jewelry`, `lighting`, `plant`, `tool`, `toy`, `vehicle`, `weapon`) and 147 level 2 concepts. The schema is the five original fields only, with no HSJ columns, since no pairwise similarity data exists for these words.

It is built by `scripts/prepare_metadata.py` from `conf/concept_group/abstractiveness_150.yaml` rather than from a word-list CSV, and it draws `frequency` from the same Wikipedia count file and `typicality` from the same THINGS ratings file. Its THINGS coverage is complete, 147 of 147, which is unsurprising given that the category vocabulary was chosen to match THINGS in the first place. That completeness is why the `gpt2_150` config can use `typicality` as its typicality column where the Richie-HSJ configs cannot.

The trade made when moving to Richie-HSJ was coverage of a direct typicality rating against availability of pairwise human similarity judgments. The 150-concept set has the former and none of the latter, so module 2's section 2.3 and module 5's Study B have no human target on it at all.
