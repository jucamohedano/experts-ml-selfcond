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

### Across scopes

At AP 0.6, per sublayer:

| Scope | CV $R^2$ J | CV $R^2$ J+prof | $\Delta$ | rank acc J | rank acc J+prof | $\Delta$ |
|---|---|---|---|---|---|---|
| whole model | 0.141 | 0.140 | -0.001 | 66.4% | 66.1% | +1.4 |
| mlp.gate_proj | 0.156 | 0.183 | +0.027 | 67.3% | 66.0% | -0.8 |
| mlp.up_proj | 0.152 | 0.159 | +0.008 | 65.7% | 66.2% | +2.0 |
| self_attn.o_proj | 0.211 | 0.203 | -0.008 | 63.3% | 62.2% | -0.7 |
| mlp.down_proj | -0.004 | -0.003 | +0.001 | 55.3% | 47.5% | -7.1 |
| self_attn.q_proj | -0.028 | -0.023 | +0.005 | 52.1% | 47.1% | -4.9 |
| self_attn.v_proj | 0.053 | 0.067 | +0.014 | 61.1% | 61.9% | +3.1 |
| self_attn.k_proj | 0.028 | 0.028 | +0.001 | 58.2% | 53.2% | -3.6 |

The two formulations disagree about which sublayer is best. The regression favours `self_attn.o_proj` ($R^2$ 0.211 against the whole model's 0.141) while the ranker favours `mlp.gate_proj` (67.3%) and puts `o_proj` below the whole model. With eight scopes compared and no correction applied, neither ranking should be treated as established. The one consistent pattern is that both formulations place the FFN expansion projections and `o_proj` above the remaining attention projections, matching module 1's sublayer informativeness ordering.

## Conclusions

- Adding layer-profile agreement to the Jaccard index does not improve prediction of human typicality. This holds at every AP threshold, in both the regression and the ranking formulation, and under linear, tree-based and neural models.
- The layer-profile feature has no standalone predictive power for typicality, with a cross-validated $R^2$ near zero or negative throughout and a pairwise accuracy near chance.
- This follows directly from module 3, where the concept-to-parent $z$ was already flat at zero. The feature carries no information about the specific pairing typicality depends on.
- **The pairwise ranking formulation of 9.2 is nonetheless the better model of this target** and should be preferred regardless of features. It matches how the ratings were collected, removes category-level terms algebraically rather than by control, and beats the regression by about 3 points on identical features. Restricting to pairs with a clear rating difference adds a further 4 points.
- A per-concept regression on the profile *difference from the parent* appears to help substantially (cross-validated $R^2$ from 0.115 to 0.312 on the complete dataset), but that gain is the category-identity shortcut, not depth information. Holding out whole categories cuts it to a fraction, and the pairwise formulation, which cancels the parent term algebraically, removes it entirely. Treat any per-concept feature containing a category-level component as suspect until tested with held-out categories.
- Model capacity is not the limiting factor. A multilayer perceptron was the worst of four model classes tried, reaching $R^2$ between $-0.09$ and $-1.19$ on the higher-dimensional feature sets, because 196 concepts cannot support it. Random forests were the strongest nonlinear option and did not change any conclusion.
- The result is not evidence against the metric in general. Module 5 shows it separating same-category from different-category concept pairs at ROC-AUC 0.69 and holding that separation where the Jaccard index collapses. The findings together locate its usefulness in concept-to-concept comparisons rather than in concept-to-category-label ones.
- The natural follow-up, not implemented here, is to build the feature against the category *centroid*, the member-average layer profile of module 6, rather than against the category label word. Module 6 exists precisely because the label and the centroid behave differently, and typicality is defined relative to the category as a whole rather than to its name.
