# Module 9: Exploratory Human-Typicality Prediction

## Research question

Module 4 establishes that a concept's human typicality rating tracks how much its expert set overlaps its category label's, at Pearson $r$ between 0.31 and 0.39 depending on the AP threshold. Module 3, subchapter 3.2 adds a second description of the same concept-to-parent pair, the layer-profile agreement, which measures whether the two words allocate their experts to the same depths rather than to the same neurons.

Does that second description predict typicality better than the Jaccard index alone?

The module is deliberately small. Linear models with one or two predictors over 196 concepts, because the question is whether a second feature carries signal the first lacks, not how well typicality can be predicted in general.

## Analysis

Notation. For concept $c$ with parent category $k$, let $t_c \in [0,1]$ be the human typicality rating, $J_c = J(c,k)$ the Jaccard index of module 3, subchapter 3.1, and $S_c = S(c,k)$ and $z_c = z(c,k)$ the layer-profile similarity and its count-matched z-score from subchapter 3.2. All four come from one row of `category_concept_similarity_metrics.csv` joined to the metadata rating.

**Analysis scopes.** This module consumes module 3's table, so it runs once per scope exactly as module 3 does, whole model first and then once per sublayer type. A cross-scope summary lands in `sublayer_comparison.csv` and `sublayer_comparison.png`.

### 9.1 Feature sets and model

Five feature sets are fitted, of which the first and fourth are the comparison the module exists for. The single-feature layer-profile sets are carried because a gain over Jaccard is only interpretable next to what the new feature predicts on its own.

| Model | Predictors |
|---|---|
| jaccard | $J_c$ |
| layer_profile | $S_c$ |
| layer_profile_z | $z_c$ |
| jaccard_plus_profile | $J_c$, $S_c$ |
| jaccard_plus_profile_z | $J_c$, $z_c$ |

Each is an ordinary least-squares regression on standardized predictors,

$$\hat t_c = \beta_0 + \sum_{f} \beta_f \frac{x_{cf} - \bar{x}_f}{\sigma_f},$$

so a coefficient reads as the change in typicality per standard deviation of that feature. Least squares rather than a regularized variant because with 196 concepts and at most two predictors there is nothing to regularize, and an unpenalized fit keeps the coefficients interpretable.

**Two design points that decide whether the comparison means anything.**

*In-sample $R^2$ cannot answer the question.* Adding any predictor, including pure noise, never decreases it, so the two-feature model always appears to win. Every headline number here is therefore out of sample. `in_sample_r2` is still written to the CSV, but only as a reference value, and it must not be used for the comparison.

*All feature sets are fitted on identical rows.* $z_c$ is undefined for words below `MIN_PROFILE_EXPERTS`, so dropping missing values per model would train the Jaccard model on more concepts than the others and the $R^2$ values would no longer be comparable. The design frame drops rows missing any candidate feature, once, before any model is fitted.

**Evaluation.** Two cross-validation schemes, answering different questions.

`cv_r2_mean` and `cv_r2_sd` come from repeated 5-fold cross-validation, 20 shuffles. Within each shuffle the out-of-fold predictions are pooled and one $R^2$ is taken, so the reported spread describes how much the estimate moves with the fold split rather than the far noisier per-fold value on 39 test concepts.

`group_cv_r2` holds out one entire category at a time, eight folds. This is the harder question, whether typicality can be predicted for a category the model never trained on. Random folds let a model lean on the category-level mean of its training concepts, and category identity alone explains $R^2 = 0.21$ of typicality variance on this dataset, so the two numbers together separate within-category signal from between-category signal.

`cv_spearman` is the rank agreement between pooled out-of-fold predictions and the true ratings, reported because $R^2$ penalizes calibration errors that a ranking use would not care about.

**Nested comparison.** For each of the two headline pairs, a partial $F$ test on the added term,

$$F = \frac{(\mathrm{RSS}_{\text{reduced}} - \mathrm{RSS}_{\text{full}}) / (p_{\text{full}} - p_{\text{reduced}})}{\mathrm{RSS}_{\text{full}} / (n - p_{\text{full}} - 1)},$$

referred to $F_{p_{\text{full}} - p_{\text{reduced}},\; n - p_{\text{full}} - 1}$. This complements the cross-validated gain rather than replacing it. The $F$ test asks whether the added feature explains significantly more variance in this sample, a question about the fitted coefficient, while the cross-validated gain asks whether it helps predict concepts the model has not seen. A feature can pass one and fail the other, so both are reported.

Fewer than `MIN_TRAIN_CONCEPTS = 40` usable concepts skips the module for that scope with a warning.

**Generated data structures.**

`typicality_model_design.csv`, the exact rows fitted, one per concept, with columns `concept`, `category`, `human_typicality`, `jaccard_pct`, `layer_profile_similarity_pct`, `layer_profile_z`.

`typicality_model_comparison.csv`, one row per feature set:

| Column | Description |
|---|---|
| model, features, n_features | Identity of the feature set. |
| n_concepts | Rows fitted, identical across all models by construction. |
| cv_r2_mean, cv_r2_sd | Pooled out-of-fold $R^2$, mean and standard deviation over 20 shuffles. |
| cv_spearman | Rank correlation of out-of-fold predictions with the true ratings. |
| group_cv_r2 | Out-of-fold $R^2$ with whole categories held out. |
| in_sample_r2 | Reference only, not comparable across feature sets. |
| coef_* | Standardized coefficient per predictor. |
| delta_cv_r2_vs_jaccard | Cross-validated gain over the Jaccard-only model, on the fuller model's row. |
| partial_f, partial_f_p | Partial $F$ statistic and p-value for the added term. |

Two plots. `typicality_model_comparison.png` shows out-of-sample $R^2$ per feature set in two panels, random folds and held-out categories, with the two headline models in distinct colors. `typicality_predicted_vs_actual.png` shows out-of-fold predictions against the true rating for those two models, colored by category, with the identity line.

### 9.2 Within-category pairwise ranking

**Motivation.** Subchapter 9.1 regresses an absolute rating per concept, which is a harder task than the data supports and than the question requires. Three problems it runs into, all of which this reframing removes.

The rating is *relative to a category*, so the regression must simultaneously learn each category's level and the ordering within it. Category identity alone explains $R^2 = 0.21$ of the variance, so a model evaluated on random folds can score well by recovering the category and never learning anything about typicality. There are only 196 training examples. And the source column is `typicality_HSJ_pairwise`, meaning the human ratings were themselves derived from pairwise judgments, so an absolute-value regression is not the form the data was collected in.

The reframing asks instead: given two concepts of the *same* category, which is the more typical?

**Mathematical formulation.** For concepts $a$ and $b$ sharing category $k$, with per-concept feature vector $\varphi(\cdot)$, the model scores the ordered pair by

$$f(a, b) = \sigma\big(w^{\top}(\varphi(a) - \varphi(b))\big),$$

a logistic model on the feature *difference*, fitted with **no intercept**, predicting $\mathbb{1}[t_a > t_b]$.

Three properties follow from that construction, and each addresses one of the problems above.

*Category terms cancel algebraically.* Any component of $\varphi$ shared by all members of a category, including the parent's own layer profile, disappears from $\varphi(a) - \varphi(b)$. Writing the layer-profile feature as a difference from the parent makes this explicit, $(p_a - p_k) - (p_b - p_k) = p_a - p_b$. The category-identity shortcut is therefore unavailable by construction rather than merely controlled for.

*The model is exactly antisymmetric,* $f(b,a) = 1 - f(a,b)$. This requires both the missing intercept and scaling without centring, since either a constant term or a subtracted feature mean would let the model express a preference that does not flip when the pair is presented the other way round. Each training pair additionally enters twice, as $(a,b)$ labelled 1 and $(b,a)$ labelled 0.

*The sample grows by more than an order of magnitude.* Summing $\binom{n_k}{2}$ over the eight categories gives 2,372 within-category pairs from 196 concepts. Pairs across categories are excluded, since typicality is undefined between them, and exact ties are excluded, since they carry no direction to predict.

**Feature sets.** Four, all differenced pairwise:

| Model | $\varphi(c)$ |
|---|---|
| jaccard | $J_c$ |
| jaccard_plus_profile | $J_c$, $S_c$, $z_c$ |
| block_profile | the concept's own 28-bin block-axis layer distribution |
| jaccard_plus_block_profile | $J_c$ and that distribution |

The last two are diagnostics rather than candidates. Because pair differencing cancels the parent term, a distribution that still fails here cannot have been carrying depth information when it appeared to help a per-concept regression, and must instead have been acting as a category-identity proxy.

**Evaluation.** `GroupKFold` over categories, holding out one whole category at a time. Because every pair lies inside a single category, holding out a category removes every pair containing any of its concepts, so no concept ever appears in both halves and the model must generalize to a category it has not seen.

Three accuracies are reported. `accuracy` covers all held-out pairs. `accuracy_clear` covers only pairs whose ratings differ by at least `RANK_MARGIN = 0.10`, where the human data actually distinguishes the two concepts, since a model should not be penalized for missing a distinction the ratings barely make. `regression_accuracy` applies subchapter 9.1's per-concept ridge to exactly the same held-out pairs, ranking each pair by its two predicted ratings, so the two formulations are compared on one scale rather than accuracy against $R^2$.

**Significance.** The headline comparison carries a paired $t$ test over the eight per-category accuracies, not over pairs. Pairs are not independent observations, since thousands of them are built from a couple of hundred concepts, so a test treating them as independent would badly overstate significance. The eight held-out categories are close to independent replicates of the same comparison and are the honest unit of analysis.

Fewer than `MIN_RANK_PAIRS = 200` usable pairs skips the subchapter.

**Generated data structures.** `typicality_ranker_comparison.csv`, one row per feature set with `accuracy`, `accuracy_clear`, `regression_accuracy`, `accuracy_sd_across_categories`, `n_pairs`, and on the headline row `delta_accuracy_vs_jaccard`, `categories_improved`, `categories_tested` and `paired_t_p`. `typicality_ranker_per_category.csv`, one row per (feature set, held-out category). `typicality_ranker_comparison.png`, accuracy per feature set with the regression baseline beside it and a chance line at 50 percent, plus the per-category detail for the headline pair.

Three diagnostic figures accompany them, each answering a question the aggregate accuracy cannot.

- `ranker_category_dumbbell.png` joins the Jaccard accuracy of each held-out category to its Jaccard-plus-profile accuracy, sorted by the change. A mean that rests on one or two categories looks identical to a uniform small gain in the summary table and completely different here.
- `ranker_accuracy_by_gap.png` plots accuracy against how far apart the two human ratings are, generalising the single `RANK_MARGIN` cutoff into a curve. A model that has learned the construct is near chance where the raters were undecided and accurate where they were decisive, so a flat line would show the accuracy is not tracking the rating at all.
- `ranker_pair_errors.png` shows, per category, a concept-by-concept grid of which pairs were misranked, with concepts ordered by human typicality. Errors scattered near the diagonal mean the model confuses adjacent neighbours, which is expected. A contiguous block means the ordering is systematically inverted, which is a different problem with a different cause.

### 9.3 Measuring against the category centroid rather than the label word

**Research question.** Subchapters 9.1 and 9.2 measure each concept against its category **label word**, treating the word "bird" as the category's representative. Module 6 established that a label word behaves unlike the average of its members, so the two are not interchangeable. This subchapter measures against the **other members** instead, which is the family-resemblance formulation in its classic form: is a typical bird the one most like the word "bird", or the one most like the other birds?

**Mathematical formulation.** For concept $c$ in category $k$ with members $M_k$, every reference is built from $M_k \setminus \{c\}$. Leaving the concept out is not optional: in a twenty-member category a concept otherwise supplies five percent of the object it is scored against, the inflation is larger in smaller categories, and the feature becomes part self-similarity.

The set-based feature is the mean Jaccard to the other members,

$$\text{mean\_jaccard\_to\_members}(c) = \frac{100}{|M_k| - 1} \sum_{m \in M_k \setminus \{c\}} \frac{|E_c \cap E_m|}{|E_c \cup E_m|},$$

which is the model counterpart of how the human typicality proxy was built, since that averaged each word's row of the human similarity matrix.

The distribution-based feature is the layer-profile similarity to the leave-one-out centroid $\bar{p}_{-c} = \frac{1}{|M_k| - 1}\sum_{m \neq c} p_m$, using module 3's transform so it lands on the same scale as the label-word feature it is compared against,

$$\text{layer\_profile\_similarity\_to\_centroid}(c) = 100\left(1 - \sqrt{\mathrm{JSD}(p_c,\ \bar{p}_{-c})}\right).$$

The third feature contrasts the own centroid against the seven foreign ones,

$$\text{centroid\_margin}(c) = \text{sim}(p_c, \bar{p}_{-c}) - \frac{1}{7}\sum_{j \neq k} \text{sim}(p_c, \bar{p}_j).$$

`centroid_margin` replaces the count-matched $z$ used for pairs elsewhere in this pipeline. That construction does not transfer, because a centroid is an average of many profiles and has no expert count of its own, so there is nothing to match against. The margin controls the same nuisance, that some concepts have generically central profiles close to everything, and reads directly as "closer to its own category than to the others".

Profiles are taken on the block axis, matching both module 6's centroid and subchapter 9.2's `__profile__` feature.

**Comparability.** The label-word and member-average feature sets are run against each other in `FEATURE_SETS`, and both are fitted on identical rows through the single shared `dropna` over every candidate feature. A per-model `dropna` would give one reference more training data than the other and make the $R^2$ values incomparable.

### 9.4 Predicting human pair similarity directly

**Research question.** Subchapters 9.1 to 9.3 predict `typicality_HSJ_pairwise`, which is not a measured quantity. Richie and Bhatia elicited pairwise similarity only, and that column was derived by averaging each word's row of the resulting matrix. This subchapter predicts the ratings themselves.

**What changes, and why it is not just a new column.** The ranker predicts a **direction**, which of two concepts is more typical, from the antisymmetric difference $\phi(a) - \phi(b)$ of per-concept features. Human pair similarity is a **value** attached to the pair and is symmetric in its two members, so the features must be symmetric too. They are therefore the pair quantities $J(a,b)$, $S(a,b)$ and $z(a,b)$, module 5's matrices, not module 3's concept-to-parent table. Predicting a symmetric target from antisymmetric features would be incoherent.

This is consequently pairwise regression rather than ranking, evaluated by the Spearman correlation of predicted against actual, pooled and per category, each also divided by that category's noise ceiling.

**No shortcut is available here.** Subchapter 9.1 had to guard against the category-identity shortcut, where a feature carrying a category-level component lets the model predict typicality by learning category averages. That cannot happen on this target: the category is constant within every pair, so a model given only category identity scores at chance. The guard is nonetheless kept, since categories are still held out whole by `GroupKFold`, and the property is asserted in `tests/check_pair_similarity_target.py`.

**Generated data structures.** `pair_similarity_comparison.csv`, one row per feature set with `n_pairs`, `n_categories`, `group_cv_spearman` (pooled), `mean_category_spearman` (the headline), `noise_ceiling_mean`, `rho_over_ceiling`, `pooled_rho_over_ceiling` and `group_cv_r2`. `rho_over_ceiling` divides the **mean over categories** by the ceiling, not the pooled correlation, since dividing the pooled figure would propagate the between-category artefact described above into the normalised column as well. The pooled version is kept beside it as `pooled_rho_over_ceiling` so the discrepancy stays visible rather than being hidden. `pair_similarity_per_category.csv`, one row per (feature set, category), where `rho_over_ceiling` is that category's own correlation over its own ceiling and is unaffected.

## Results

Whole-model scope, corrected `_sensefix` runs, 197 concepts across all 8 categories.

### 9.1 Per-concept regression

| Model | AP | CV $R^2$ Jaccard | CV $R^2$ Jaccard + profile | $\Delta$ | partial $F$ p |
|---|---|---|---|---|---|
| GPT-2 | 0.5 | 0.081 | 0.162 | +0.081 | <0.001 |
| GPT-2 | 0.6 | 0.067 | 0.064 | -0.003 | 0.206 |
| GPT-2 | 0.7 | 0.026 | 0.053 | +0.027 | 0.005 |
| GPT-2 | 0.8 | -0.046 | -0.060 | -0.014 | 0.289 |
| Qwen3 | 0.5 | 0.186 | 0.188 | +0.002 | 0.142 |
| Qwen3 | 0.6 | 0.139 | 0.141 | +0.001 | 0.130 |
| Qwen3 | 0.7 | 0.090 | 0.114 | +0.024 | 0.007 |
| Qwen3 | 0.8 | 0.078 | 0.069 | -0.010 | 0.536 |

At AP 0.6 the single-feature models are `layer_profile` at $-0.019$ and `layer_profile_z` at $-0.023$ on Qwen3, both below zero, meaning each predicts less well on unseen concepts than simply returning the mean rating. Held-out-category $R^2$ tells the same story: Jaccard alone reaches $+0.057$ while adding the profile drops it to $-0.001$.

**The layer-profile feature still does not improve typicality prediction, but the picture is less clean than before.** Two cells now reach significance, GPT-2 at AP 0.5 ($\Delta$ = +0.081, p < 0.001) and both models at AP 0.7 ($\Delta$ = +0.027 and +0.024, p = 0.005 and 0.007). Three considerations argue against reading these as a real effect. They do not replicate across adjacent thresholds in the same model, GPT-2 going +0.081, -0.003, +0.027, -0.014 as the threshold rises. Eight scopes are tested per model, so nominal significance at p ≈ 0.005 in two of sixteen cells is close to what multiple testing produces on its own. And the held-out-category $R^2$, which is the honest generalization measure, does not improve in the significant cells either, GPT-2 AP 0.5 being the sole exception (-0.012 to +0.092). The pairwise formulation below, which is the better-posed test, finds nothing at any threshold.

The Jaccard baseline behaves as module 4 reports, decaying with the threshold until at AP 0.9 both models fail outright.

Worth noting how visible the in-sample trap is. The in-sample $R^2$ always rises when the layer-profile feature is added, since it must, while the out-of-sample $R^2$ generally falls. Reporting the in-sample number would have reversed the conclusion.

### 9.2 Pairwise ranking

Whole-model scope, Qwen3, AP 0.6, 2,391 within-category pairs over 8 held-out categories:

| Feature set | accuracy | accuracy, clear pairs | regression on the same pairs |
|---|---|---|---|
| jaccard | **66.6%** | 71.1% | 66.6% |
| jaccard + layer profile | 65.6% | 69.4% | 62.8% |
| block_profile (28 bins) | 53.7% | 54.6% | 52.4% |
| jaccard + block_profile | 60.1% | 62.9% | 59.8% |

GPT-2 at AP 0.5 gives the same ordering on the same 2,391 pairs: jaccard 61.5%, jaccard + profile 62.9%, block_profile 49.9%, jaccard + block_profile 58.8%.

Three readings, in order of confidence.

**The ranking formulation is the better one.** On the same features, the ranker reaches 65.6% where the regression reaches 62.8%. For Jaccard alone the two are identical at 66.6%, as they must be, since a single monotone feature induces the same ordering either way, so the gain appears exactly when several features have to be traded off. The reason is that the regression must additionally fit absolute levels, which the ratings do not really determine, while the ranker only has to get orderings right.

**Restricting to clear pairs helps by about 4 points** in every row (66.6% to 71.1% for Jaccard). Near-ties, where the two ratings differ by less than 0.10, are largely noise, and a model should not be judged on them.

**The layer profile still adds nothing.** The mean per-category gain is $+0.6$ points, only 4 of 8 categories improve, and the paired $t$ test over categories gives $p = 0.87$. On GPT-2 at AP 0.5 the corresponding figures are $+1.3$ points, 5 of 8 categories, $p = 0.13$. Pooled accuracy actually *falls* on Qwen3, from 66.6% to 65.6%, so the pooled and per-category summaries disagree in sign, which is itself a sign that the effect is not robust.

Per held-out category at AP 0.6, the spread swamps the mean difference:

| Category | jaccard | + layer profile | delta |
|---|---|---|---|
| fruit | 54.3% | 73.8% | **+19.5** |
| vehicles | 79.7% | 81.4% | +1.7 |
| clothing | 76.4% | 78.1% | +1.7 |
| vegetables | 36.8% | 40.0% | +3.2 |
| sports | 67.0% | 66.7% | -0.3 |
| furniture | 72.1% | 71.6% | -0.5 |
| birds | 60.9% | 58.6% | -2.3 |
| professions | 73.3% | 55.0% | **-18.3** |

The mean again rests almost entirely on `fruit`, and `professions` moves 18 points the other way, reproducing the earlier pattern on the complete 8-category data.

**The 28-bin profile settles the earlier question.** It scores 53.7% alone and drags Jaccard down from 66.6% to 60.1%. Since pair differencing cancels the parent profile algebraically, a distribution that fails here cannot have been contributing depth information when a per-concept regression on $p_c - p_k$ appeared to help. That earlier apparent gain was the category-identity shortcut.

`vegetables` still sits **below chance** (36.8% to 40.0%) under every feature set and both formulations, and this is now on corrected data with `squash` present, so the earlier speculation that the regeneration might explain it is ruled out. Something is systematically inverted for that category and it needs investigating on its own terms.

### 9.3 Label word against member average

Whole-model scope, cross-validated $R^2$ with the held-out-category value beside it. `pF` is the partial $F$ p-value for adding the centroid layer profile to the centroid Jaccard.

| Model | AP | label, Jaccard | members, Jaccard | members, both | pF |
|---|---|---|---|---|---|
| GPT-2 | 0.5 | +0.081 / -0.012 | +0.014 / -0.086 | +0.032 / -0.060 | 0.016 |
| GPT-2 | 0.6 | +0.067 / -0.030 | -0.005 / -0.096 | +0.002 / -0.079 | 0.047 |
| GPT-2 | 0.7 | +0.026 / -0.038 | -0.012 / -0.110 | -0.010 / -0.086 | 0.070 |
| GPT-2 | 0.8 | -0.046 / -0.181 | -0.028 / -0.345 | -0.047 / -0.418 | 0.557 |
| Qwen3 | 0.5 | +0.186 / +0.108 | +0.192 / +0.134 | **+0.209 / +0.151** | 0.009 |
| Qwen3 | 0.6 | +0.139 / +0.057 | +0.098 / -0.012 | +0.133 / +0.002 | 0.002 |
| Qwen3 | 0.7 | +0.090 / -0.012 | +0.031 / -0.071 | +0.053 / -0.070 | 0.010 |
| Qwen3 | 0.8 | +0.078 / -0.031 | -0.004 / -0.086 | -0.014 / -0.109 | 0.441 |

Two findings, and they point in different directions, so both have to be stated.

**The member average does not generally beat the label word.** It wins in exactly one cell, Qwen3 at AP 0.5, where it leads on both the ordinary cross-validated $R^2$ (0.209 against 0.186) and the stricter held-out-category value (0.151 against 0.108). At every other threshold and throughout GPT-2 the label word is the better reference, sometimes by a wide margin. The classic family-resemblance intuition, that a typical bird is the one most like the other birds, is therefore not supported as a general claim by this data. It holds only in the largest model at the most lenient threshold, which is also the setting where the centroid is estimated from the most experts and is consequently least noisy.

**The layer profile does add to the centroid reference, consistently.** The partial $F$ test clears 0.05 in five of the eight cells (Qwen3 at AP 0.5, 0.6 and 0.7, GPT-2 at AP 0.5 and 0.6), and the two failures are both at AP 0.8 where every model in the table has gone negative. This contrasts with the label-word reference, where the same feature never reaches significance. The layer profile is therefore not inert, it simply carries information about how a concept relates to the other members of its category rather than to the category's name, which is a different relation and was invisible to the earlier formulation.

The contrast is drawn in `typicality_reference_comparison.png`, four bars covering both references under both cross-validation schemes, with the partial $F$ p-value for the added layer profile printed over each bar that includes it, so the significant centroid case and the non-significant label-word case are visible side by side.

`centroid_margin` adds nothing anywhere (its own partial $F$ never approaches significance, p = 0.55 at Qwen3 AP 0.5), so the contrast against foreign centroids is not carrying signal beyond the own-centroid similarity.

### 9.4 Predicting human pair similarity

The headline column is `mean_category_spearman`, the mean of the per-category correlations, **not** the pooled `group_cv_spearman`.

That distinction is not cosmetic here. Out-of-fold predictions for a held-out category are miscalibrated in level relative to the categories the model trained on, so pooling them across categories introduces a between-category component that can be strongly negative while every within-category ordering remains correct. At Qwen3 AP 0.8 the pooled figure is -0.167 against +0.493 per category. Reading the pooled column as the result would report failure where the model in fact works.

| Model | AP | Jaccard | Jaccard + profile + z | profile alone (pooled) | n pairs |
|---|---|---|---|---|---|
| GPT-2 | 0.5 | 0.361 | 0.373 | -0.374 | 2,391 |
| GPT-2 | 0.6 | 0.341 | 0.349 | -0.438 | 2,391 |
| GPT-2 | 0.7 | 0.312 | 0.262 | -0.478 | 2,391 |
| GPT-2 | 0.8 | 0.349 | 0.150 | -0.416 | 912 |
| Qwen3 | 0.5 | 0.508 | **0.579** | -0.056 | 2,391 |
| Qwen3 | 0.6 | 0.478 | 0.516 | -0.287 | 2,391 |
| Qwen3 | 0.7 | 0.516 | 0.430 | -0.474 | 2,391 |
| Qwen3 | 0.8 | 0.493 | 0.315 | -0.438 | 1,984 |

**Expert similarity predicts human similarity with categories held out.** Qwen3 sits between 0.478 and 0.516 on Jaccard alone at every usable threshold, and GPT-2 between 0.312 and 0.361. These match subchapter 5.3's raw correlations almost exactly, which is the expected result: a monotone model of one feature preserves the within-category ordering, so cross-validating it can only reproduce what the feature already carries. The value of doing it here is that the model is fitted without ever seeing the test category.

**Adding the layer profile helps at lenient thresholds and hurts at strict ones.** On Qwen3 it improves 0.508 to 0.579 at AP 0.5 and 0.478 to 0.516 at AP 0.6, then degrades the model from AP 0.7 onward. The crossover is where profiles start being estimated from very few experts, and a noisy feature added to a working model subtracts. Alone the profile is useless or actively inverted, from -0.056 down to -0.478.

**AP 0.9 produces no output in either model**, since fewer than `MIN_PAIR_ROWS = 200` rated pairs survive once module 3's table has thinned, and the subchapter declines rather than reporting a correlation from a handful of pairs.

### Across scopes

At AP 0.6, per sublayer:

| Scope | n | CV $R^2$ J | CV $R^2$ J+prof | $\Delta$ | rank acc J | rank acc J+prof | pair sim $\rho$ |
|---|---|---|---|---|---|---|---|
| whole model | 197 | 0.139 | 0.141 | +0.001 | 66.6% | 65.6% | 0.516 |
| mlp.gate_proj | 197 | 0.163 | 0.192 | +0.029 | 67.1% | 64.8% | 0.471 |
| mlp.up_proj | 197 | 0.139 | 0.146 | +0.007 | 65.7% | 66.4% | 0.513 |
| self_attn.o_proj | 197 | 0.188 | 0.180 | -0.008 | 63.6% | 62.0% | **0.553** |
| mlp.down_proj | 147 | 0.001 | 0.019 | +0.018 | 58.2% | 51.1% | 0.242 |
| self_attn.q_proj | 197 | -0.021 | -0.010 | +0.012 | 53.7% | 46.6% | 0.252 |
| self_attn.v_proj | 197 | 0.061 | 0.072 | +0.011 | 60.1% | 60.5% | 0.330 |
| self_attn.k_proj | 197 | 0.021 | 0.029 | +0.007 | 56.9% | 53.0% | 0.268 |

The three formulations do not agree on which sublayer is best. The regression and the human pair-similarity target both favour `self_attn.o_proj` (0.188 against the whole model's 0.139, and $\rho$ 0.553 against 0.516), while the ranker favours `mlp.gate_proj` at 67.1 percent and places `o_proj` below the whole model. With eight scopes compared and no correction applied, none of these orderings should be treated as established.

Two patterns do hold across all three. Every formulation separates the FFN expansion projections and `o_proj` from the remaining attention projections, matching module 1's sublayer informativeness ordering. And `mlp.down_proj`, `self_attn.q_proj` and `self_attn.k_proj` are near-useless on every target, with the ranker at or below chance on two of them.

The `pair sim` column is worth reading against subchapter 5.3's caution: `mlp.gate_proj` leads on category alignment but is only fourth here, while `o_proj` leads on human similarity. Recovering a category partition and reproducing graded human similarity are different tasks, and this table shows the same dissociation from the model side.

## Conclusions

- Adding layer-profile agreement to the Jaccard index does not improve prediction of human typicality. This holds at every AP threshold, in both the regression and the ranking formulation, and under linear, tree-based and neural models.
- The layer-profile feature has no standalone predictive power for typicality, with a cross-validated $R^2$ near zero or negative throughout and a pairwise accuracy near chance.
- This follows directly from module 3, where the concept-to-parent $z$ was already flat at zero. The feature carries no information about the specific pairing typicality depends on.
- **The pairwise ranking formulation of 9.2 is nonetheless the better model of this target** and should be preferred regardless of features. It matches how the ratings were collected, removes category-level terms algebraically rather than by control, and beats the regression by about 3 points on identical features. Restricting to pairs with a clear rating difference adds a further 4 points.
- A per-concept regression on the profile *difference from the parent* appears to help substantially (cross-validated $R^2$ from 0.115 to 0.312 on the complete dataset), but that gain is the category-identity shortcut, not depth information. Holding out whole categories cuts it to a fraction, and the pairwise formulation, which cancels the parent term algebraically, removes it entirely. Treat any per-concept feature containing a category-level component as suspect until tested with held-out categories.
- Model capacity is not the limiting factor. A multilayer perceptron was the worst of four model classes tried, reaching $R^2$ between $-0.09$ and $-1.19$ on the higher-dimensional feature sets, because 196 concepts cannot support it. Random forests were the strongest nonlinear option and did not change any conclusion.
- The result is not evidence against the metric in general. Module 5 shows it separating same-category from different-category concept pairs at ROC-AUC 0.69 and holding that separation where the Jaccard index collapses. The findings together locate its usefulness in concept-to-concept comparisons rather than in concept-to-category-label ones.
- The natural follow-up, not implemented here, is to build the feature against the category *centroid*, the member-average layer profile of module 6, rather than against the category label word. Module 6 exists precisely because the label and the centroid behave differently, and typicality is defined relative to the category as a whole rather than to its name.
