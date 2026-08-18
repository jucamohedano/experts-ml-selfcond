# Module 9: Typicality Prediction and Human Pair Similarity

## Research question

Modules 3 and 5 describe a word pair in two ways. The Jaccard index asks which neurons the two words share, and the layer-profile agreement asks whether the two words spread their experts over depth in the same proportions. Module 4 establishes that the first of those tracks human typicality at Pearson $r$ between 0.21 and 0.45 depending on the model and the AP threshold, 0.320 for GPT-2 and 0.452 for Qwen3 at AP 0.5, falling to 0.209 and 0.328 by AP 0.7.

Does the layer-profile agreement add predictive signal over the Jaccard index?

The module asks that one question with two human targets rather than asking four loosely related questions in four subchapters. Study A predicts which of two same-category concepts humans rate as the more typical, and Study B predicts the measured human similarity of a concept pair. Both are fitted on one generated feature grid whose six cells carry the same names and the same meaning in either study, so a reader can confirm that the same model was asked the same question on both targets. The module is deliberately small, with at most three predictors over a couple of hundred concepts, because the question is whether a second feature carries signal the first lacks rather than how well typicality can be predicted in general.

## Analysis

Notation. For a word $w$, let $E_w$ be its expert set and $n_w = |E_w|$ its expert count, and let $p_w$ be its layer profile, the row-normalized vector of expert counts per bin defined in module 3, subchapter 3.2. For a concept $c$ with parent category $k$, write $t_c$ for the human typicality rating, $J = J(c,k)$ for the Jaccard index of module 3, subchapter 3.1, $S = S(c,k)$ for the layer-profile agreement and $z = z(c,k)$ for its count-matched null score. The same three symbols apply to a concept pair $(a,b)$ in Study B and to a concept against its category centroid in Study A's second reference, and the resolution table below is the only place they mean anything different.

**Analysis scopes.** The module runs once per scope, whole model first and then once per sublayer type, and a cross-scope summary lands in `sublayer_comparison.csv` and `sublayer_comparison.png`. Unlike the earlier version it no longer consumes module 3's table. It computes every profile feature itself from `scope.expert_df`, calling the same helpers over the same item list, `concept_metadata['concept'].unique()`, so the default metric's label-word columns are identical to module 3's CSV by construction while a new profile metric needs no module 3 schema change. Module 3 does not have to be enabled for module 9 to run.

### The feature grid

One dictionary, six cells, generated per registered profile metric by `build_feature_grid(metrics)` rather than written out by hand.

| Cell | Features |
|---|---|
| `jaccard` | $J$ |
| `profile` | $S$ |
| `profile_z` | $z$ |
| `jaccard_profile` | $J$, $S$ |
| `jaccard_profile_z` | $J$, $z$ |
| `jaccard_profile_both` | $J$, $S$, $z$ |

The grid is a nested design rather than a list. The three single-feature cells establish standalone power, without which a gain over Jaccard is uninterpretable. The two two-feature cells test each addition to Jaccard separately, and the last tests both together. Every partial $F$ test in the module is therefore a well-defined nested comparison between two rows of one table.

The three symbols resolve to concrete columns per study and per reference as follows.

| Symbol | Study A, `label_word` | Study A, `member_centroid` | Study B |
|---|---|---|---|
| $J$ | `jaccard_pct` | `mean_jaccard_to_members` | `jaccard_pct` of the pair |
| $S$ | `profile_js_distance` | `centroid_profile_js_distance` | `profile_js_distance` of the pair |
| $z$ | `profile_js_distance_z` | `centroid_profile_js_distance_z` | `profile_js_distance_z` of the pair |

The `js_distance` segment names the profile metric, see the metric registry below. Every comparison CSV carries a `metric` column naming the metric a row was built on, and the `jaccard` cell, which uses no profile metric, carries `metric = "none"` and appears once rather than once per metric. With one registered metric this reduces exactly to the tables above, and with the seven registered today the same table repeats once per metric under one vocabulary.

**The identical-rows rule.** All cells within a study are fitted on identical rows, through a single `dropna` over every candidate feature of that study performed once before any model is fitted. This exists because $z$ is undefined for words below `MIN_PROFILE_EXPERTS = 2`, so a per-cell `dropna` would train the Jaccard cell on more concepts than the others and the accuracies would no longer be comparable. The scope of the rule is the study rather than the module. Study A drops over both references' columns including the centroid ones, because its two arms are compared against each other inside one figure. Study B does its real dropna somewhere else entirely, and this is easy to misread. It takes only `concept` and `category` off the design frame and recomputes every feature it fits as a symmetric PAIR quantity, so the frame-level `study_b_columns` list is `["jaccard_pct"]` alone and is a DEFENSIVE NO-OP, since the presence gate already guarantees a surviving row has a defined `jaccard_pct`. The identical-rows guarantee for Study B is enforced inside `run_pair_similarity`, by one shared dropna over the union of every PAIR column its six cells use, applied to the pair table before any cell is fitted. The list must not be "fixed" by adding the profile columns back. Those are concept-against-LABEL-WORD quantities Study B never reads, and gating on them deleted an entire category at AP 0.9, where the label word `vehicles` held one expert while all 22 of its concepts were fine, which took Study B to two categories and an empty result. The source carries the same warning at length. The two studies are never compared numerically against each other, so nothing is lost by letting their row sets differ. `check_pair_similarity_identical_rows.py` verifies the rule by counting the rows each cell actually sees.

### Study A, typicality by pairwise ranking

**Target and unit.** `typicality_HSJ_pairwise`, a derived quantity. Richie and Bhatia elicited pairwise similarity, and this column was produced by averaging each word's row of the resulting matrix. The unit is the ordered within-category pair. Summing $\binom{n_k}{2}$ over the eight categories gives 2,391 pairs from 197 concepts. Cross-category pairs are excluded, since typicality is undefined between them, and exact ties are excluded, since they carry no direction to predict.

**Model.** For concepts $a$ and $b$ sharing category $k$, with per-concept feature vector $\varphi(\cdot)$,

$$f(a, b) = \sigma\big(w^{\top}(\varphi(a) - \varphi(b))\big),$$

where $\sigma(x) = 1/(1 + e^{-x})$ is the logistic function and $w$ the fitted weight vector, a logistic model on the feature difference, fitted with no intercept, predicting $\mathbb{1}[t_a > t_b]$. Each pair enters twice, as $(a,b)$ labelled 1 and as $(b,a)$ labelled 0. Scaling uses `StandardScaler(with_mean=False)`.

Both the missing intercept and the uncentred scaling are required for the exact antisymmetry $f(b,a) = 1 - f(a,b)$, since either a constant term or a subtracted feature mean would let the model express a preference that does not flip when the pair is presented the other way round. `check_ranker_antisymmetry.py` pins this to numerical tolerance for every grid cell. One warning belongs with it, because it cost a round to find. A test fitted on the mirrored training set alone cannot see a violation, since that set is invariant under $(x, y) \mapsto (-x, 1-y)$, which forces the fitted intercept to exactly zero and the column means to exactly zero, so an intercept-bearing model passes at machine epsilon. The check therefore carries a second arm fitted on unmirrored imbalanced data, where an intercept moves the residual by seven to nine orders of magnitude above the tolerance.

**Why this form.** Any component of $\varphi$ shared by all members of a category disappears from $\varphi(a) - \varphi(b)$. Category identity alone explains $R^2 = 0.21$ of typicality variance on this dataset, so a per-concept model can score well by recovering the category and never learning anything about typicality. In the differenced form that shortcut is unavailable by construction rather than merely controlled for. The form also matches how the ratings were collected.

**The reference factor.** Every cell is fitted twice, once against the category label word and once against the leave-one-out member centroid, and each arm carries its own Jaccard baseline so no gain from a profile feature is ever credited to a change of reference. Scoring a concept against the mean of the other category members is the classic family-resemblance operationalization of typicality and is the one the psychological literature is built on, while scoring it against the category label word is the model-specific convenience. Carrying both as a factor of one study makes the comparison a main effect of the design rather than a separate chapter.

**The regression baseline column.** Every cell additionally reports `regression_accuracy`, a per-concept CROSS-VALIDATED RIDGE regression on the same features, applied to exactly the same held-out pairs and ranking each pair by its two predicted ratings. The estimator is `make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 30)))`, so the penalty strength is chosen inside the training fold rather than fixed, and the baseline is a penalised fit and not an unpenalised least squares one. This puts the two formulations on one scale, accuracy against accuracy rather than accuracy against $R^2$, and it is why the per-concept regression that used to be a subchapter of its own is no longer one.

For the Jaccard cell alone the two columns are identical, and the ridge penalty does not disturb that. Standardizing a single feature and shrinking its coefficient towards zero leaves the sign of the coefficient unchanged, so a single monotone feature induces the same ordering of predicted ratings under either fit. The table below shows the identity holding on both Jaccard rows, and it holds exactly in all sixteen per-category rows of `ranker_per_category.csv`, the eight held-out categories under each of the two references, not only in the pooled mean. Where several features are present the two formulations can diverge, and the divergence has two sources rather than one. The ranker optimizes the ordering of pairs directly while the regression optimizes squared error on the rating, and the ridge penalty additionally shrinks the regression's coefficients towards each other. The honest reading of the column is therefore that the differenced ranker beats its own regression baseline on every multi-feature cell while tying it exactly on the two Jaccard cells, and it does not on its own isolate the trading off of several features as the reason.

**Evaluation.** `GroupKFold` over categories, holding out one whole category at a time, eight folds. Because every pair lies inside a single category, holding out a category removes every pair containing any of its concepts, so no concept ever appears in both halves. Three accuracies are reported per cell, `accuracy` over all held-out pairs, `accuracy_clear` over pairs whose ratings differ by at least `RANK_MARGIN = 0.10`, and `regression_accuracy` as above.

**Significance.** A paired $t$ test against the `jaccard` cell of the same reference, taken over the eight per-category accuracies rather than over pairs. Thousands of pairs built from a couple of hundred concepts are not independent observations, so a test treating them as independent would badly overstate significance. The eight held-out categories are close to independent replicates and are the honest unit of analysis. The comparison is keyed on the triple of reference, metric and cell, so registering a second metric cannot pool two metrics' per-category accuracies into one delta.

**Guard.** Fewer than `MIN_RANK_PAIRS = 200` usable pairs skips the study for that scope.

### Study B, human pair similarity by regression

**Target.** The elicited ratings themselves, Richie and Bhatia Study 1, on a 1 to 7 scale with higher meaning more similar, 2,391 within-category pairs over eight categories with 19 to 39 raters each, loaded by `utils/human_similarity.py`. This is external validation rather than a proxy, since the target is a number humans produced for that specific pair, and category is constant within every pair, so category identity alone scores at chance and the shortcut that haunts a per-concept target is closed by construction.

**Why the model differs from Study A.** The ranker predicts a direction and is built from the antisymmetric difference $\varphi(a) - \varphi(b)$. Human pair similarity is a value attached to the pair and is symmetric in its two members, so the features must be symmetric too. They are the pair quantities $J(a,b)$, $S(a,b)$ and $z(a,b)$, which are module 5's matrices rather than module 3's concept-to-parent table. Predicting a symmetric target from antisymmetric features would be incoherent, so this is a regression and not a ranking. A pair has no parent either, so the reference factor of Study A does not arise here.

**Evaluation and the headline column.** `GroupKFold` over categories. The headline is `mean_category_spearman`, the mean of the per-category correlations, with the pooled `group_cv_spearman` reported beside it rather than in place of it. That distinction is load bearing. Out-of-fold predictions for a held-out category are miscalibrated in level relative to the categories the model trained on, so pooling them introduces a between-category component that can be strongly negative while every within-category ordering remains correct. On the run reported below the two columns differ by roughly 0.26 on the Jaccard cell alone, 0.3608 against 0.1007, and on the earlier Qwen3 AP 0.8 run the pooled figure was $-0.167$ against $+0.493$ per category. Reading the pooled column as the result would report failure where the model works.

**Noise ceilings.** Each per-category correlation is also divided by that category's split-half Spearman-Brown ceiling, which runs from 0.836 for birds to 0.935 for vehicles. `rho_over_ceiling` divides the mean over categories by the mean ceiling and never the pooled correlation, since dividing the pooled figure would propagate the between-category artefact into the normalized column. `pooled_rho_over_ceiling` is kept beside it so the discrepancy stays visible rather than hidden by the normalization.

**Nested tests.** Each cell reports a partial $F$ against every cell it extends by exactly one feature within the same metric, written into a `partial_f_vs_<reduced>` column pair. Three of the six cells have more than one reduced model, which is why the reduced cell's name is carried in the column name.

**Guards.** The study needs at least `MIN_PAIR_CONCEPTS = 3` concepts to form a within-category pair at all, and fewer than `MIN_PAIR_ROWS = 200` rated pairs after the shared dropna skips the study.

### The block axis

$S$ and $z$ are computed on the block-aggregated layer profile, 28 bins on Qwen3 and 12 on GPT-2, in place of the flat 196 and 48. Mechanically this wraps the expert frame at four call sites in modules 3, 4, 5 and 9 as `layer_profile_matrices(to_block_axis(df), items)`. Note that only the profile helper sees the aggregated frame. The Jaccard helpers keep the raw frame, because Jaccard reads neuron identity and relabelling blocks would collide units across sublayers and inflate every intersection.

Two arguments motivate the change, and neither is about depth ordering. The Jensen-Shannon divergence

$$\mathrm{JSD}(p, q) = H\!\left(\tfrac{p+q}{2}\right) - \tfrac{1}{2}\big(H(p) + H(q)\big)$$

with $H(p) = -\sum_\ell p[\ell] \log_2 p[\ell]$ the Shannon entropy in bits, sums over bins independently and never touches bin adjacency, so it is permutation invariant and neither axis measures depth in the ordered sense.

What changes is what a bin means. On the block axis a bin is a transformer block and nothing else, so $S$ reads as agreement in depth allocation. On the flat axis a bin is a block crossed with a sublayer, since `build_layer_mapping_from_layers` numbers layers as `block * n_sublayers + sublayer_position + 1` and indices 1 to 7 are all block 0 on Qwen3, so $S$ there confounds allocating deep with allocating to a particular projection type.

What also changes is estimation noise. Plug-in entropy bias on a $K$-bin histogram estimated from $n$ samples is approximately $(K-1)/(2 n \ln 2)$ bits. Moving from $K = 196$ to $K = 28$ reduces it sevenfold, and because it scales as $1/n$ it falls hardest on small expert sets, which is exactly the confound $z$ exists to remove. The flat axis is the more contaminated of the two estimators.

**Blast radius.** Applying `to_block_axis` to a single-sublayer frame is an identity relabel, since such a frame already holds exactly one layer per block. Every `sublayers/` scope is therefore numerically unchanged, which is verified empirically rather than only argued, and block aggregation preserves each word's total expert count, so `MIN_PROFILE_EXPERTS` gating and the resulting row sets are unchanged as well.

**Sensitivity.** One line re-runs `jaccard_profile` on the flat axis in both studies, at `REFERENCE_AP` and in the whole-model scope only, written to `axis_sensitivity.csv` in each study folder. Its purpose is to show that the axis choice is not load bearing, not to characterize the flat axis, so it is not swept across thresholds or scopes and it is not a grid cell.

### Centroid construction

For concept $c$ in category $k$ with members $M_k$, every reference quantity is built from $M_k \setminus \{c\}$. Leaving the concept out is not optional. In a twenty-member category a concept otherwise supplies five percent of the object it is scored against, the inflation is larger in smaller categories, and the feature becomes part self-similarity. Members below `MIN_PROFILE_EXPERTS` are excluded from the member list, so a one-expert word neither receives a profile score nor contaminates its category's centroid.

The Jaccard counterpart is the mean Jaccard to the other members,

$$J_{\text{mem}}(c) = \frac{100}{|M_k| - 1} \sum_{m \in M_k \setminus \{c\}} \frac{|E_c \cap E_m|}{|E_c \cup E_m|},$$

which is also the model counterpart of how the human typicality proxy was built, since that averaged each word's row of the human similarity matrix.

The profile counterpart is the agreement with the leave-one-out centroid $\bar{p}_{-c} = \frac{1}{|M_k| - 1}\sum_{m \neq c} p_m$, using the same transform so it lands on the same scale as $S$,

$$S_{\text{cen}}(c) = 100\left(1 - \sqrt{\mathrm{JSD}(p_c,\ \bar{p}_{-c})}\right).$$

**The count-matched score $z_{\text{cen}}$.** This is the same construction as the label-word $z$, applied to the concept-by-category agreement matrix. Build $S_{\text{cen}}[c, j]$ for every concept and every category, leave-one-out on the diagonal only. Place each entry at the coordinate $(\log n_c,\ \log N_j)$ where $N_j$ is the pooled expert count of centroid $j$, then standardize each entry against its $k$ nearest entries in that space with itself excluded, reusing the pair null's neighbourhood rule of `NULL_NEIGHBOR_FRACTION = 0.01` clipped to $[40, 300]$. The reference set is dominated by foreign entries, exactly as the label-word $z$'s reference set is dominated by unrelated word pairs, and

$$z_{\text{cen}}(c) = \frac{S_{\text{cen}}[c, k] - \mu(\text{reference})}{\sigma(\text{reference})}.$$

The coordinate is in FIXED ORDER, which is the one place this departs from the pair null. That function places a pair at $(\log \min(n_a, n_b),\ \log \max(n_a, n_b))$ because the two words of a pair are exchangeable and the coordinate has to be invariant to their order. A concept and a centroid are different kinds of object and are never exchanged, so the two counts enter in a fixed order and that order carries meaning. The coordinate construction is therefore a parameter of the shared `count_matched_z` helper rather than duplicated neighbourhood logic, and `check_centroid_features.py` pins per-row equality against a fixed-order rebuild on a fixture where the concept outweighs its centroid, so a symmetrized implementation fails visibly.

**Why `centroid_margin` was retired.** The old `centroid_margin`, the own-centroid agreement minus the mean of the seven foreign ones, is removed. Its stated justification was that the count-matched construction cannot transfer to a centroid, because a centroid is an average of many profiles and has no expert count of its own. That reading does not survive scrutiny. The binding sample size in a concept-to-centroid comparison is $n_c$ alone, since the centroid pools the experts of every other member and is never the noisier side of the comparison, so the construction transfers directly once the coordinate is written in fixed order. The margin also had no counterpart anywhere else in the pipeline and added nothing in any cell. In the output tables the replacement is the `centroid_profile_<metric>_z` column, and $z_{\text{cen}}$ is the name of the construction rather than of a column. The two quantities are only moderately concordant and $z_{\text{cen}}$ should be read as a replacement rather than a refinement, which the Limitations section states with its numbers.

### The metric registry

`PROFILE_METRICS` in `utils/helpers.py` maps a metric name to a function taking a row-normalized profile matrix $P$ of shape $(n, K)$ and an optional second matrix $Q$ of shape $(m, K)$, returning the $(n, m)$ cross agreement matrix, or the square $(n, n)$ case when $Q$ is omitted, oriented so that higher always means more similar. The cross form is what lets the centroid features run through the same registry function instead of a private similarity helper.

Orientation is the entire contract. Scale is deliberately not part of it, because every consumer either standardizes the feature or ranks it, and forcing each metric onto a percentage scale would manufacture false comparability between quantities with different geometry. `check_profile_metrics.py` iterates the registry and asserts the orientation on a fixture, so future metrics are covered without new test code.

The registry holds seven entries, `js_distance`, `wasserstein`, `cosine`, `pearson`, `spearman`, `js_divergence` and `hellinger`, all defined in module 3's subchapter 3.3, which also groups them into the divergence, correlation and order-reading families and explains why their raw scales are not comparable. `DEFAULT_PROFILE_METRIC` names `js_distance`, defined as $100(1 - \sqrt{\mathrm{JSD}})$ exactly as module 3 defines it, and `ACTIVE_PROFILE_METRICS` keeps it first so the summary row and the legacy column names always describe it. `layer_profile_matrices` takes a `metric` parameter defaulting to that name and dispatches through the registry. `ACTIVE_METRICS` in module 9 lists the metrics the grid is instantiated on and is now `ACTIVE_PROFILE_METRICS`, shared with modules 3 and 5, so one edit changes what the whole pipeline computes. Registering a metric and listing it there instantiates every cell of both studies on it and touches nothing else. The grid generates five cells per metric plus the single metric-free Jaccard cell, so at seven metrics Study A fits 36 cells per reference and Study B fits 36, against 6 and 6 with one metric, which is about 80 seconds for the five GPT-2 scopes of one AP threshold. The count-matched null is already metric-agnostic, since it standardizes whatever agreement matrix it is handed against count-matched reference entries, so every registered metric gets its $z$ column and its $z_{\text{cen}}$ column for free.

The six planned metrics of the follow-up project are Wasserstein, cosine, Pearson, Spearman, the Jensen-Shannon divergence without the square root, and Hellinger. One metric-specific fact is worth recording now. Wasserstein is the only one of the six that reads the ORDER of the bins, every other one being permutation invariant like the Jensen-Shannon distance. It is therefore meaningful only on the block axis, where adjacent bins are adjacent depths, and would be meaningless on the flat axis, where adjacent bins are different sublayer types of the same block. The block-axis decision above is what makes the metric extension coherent at all.

The metric a figure describes is never left implicit. `_display_metric` returns `SUMMARY_METRIC` whenever that metric is active, so the three diagnostic figures that can only draw one metric at a time, the per-category dumbbell, the accuracy-by-gap curve and the pair-error grid, draw exactly the metric the cross-scope summary row reports, whether a new metric is appended to `ACTIVE_METRICS` or prepended to it. The pair-similarity figure selects its rich cell on the `(model, metric)` pair for the same reason, rather than on whichever row of the grid order carries the model name first. Each of those titles or legends names the metric as soon as more than one is active, so a single-metric figure reads as it does today and a multi-metric one is unambiguous.

### Design rationale, what the deleted diagnostics established

Four raw-distribution feature sets and two subchapters were removed by the restructure, and their findings are the reason the design has the shape it has. Both numbers below come from the PRE-RESTRUCTURE flat-axis run on Qwen3 at AP 0.6 and are kept here as the historical record of why the module is built this way, not as current results. They are not comparable to the tables in the Results section, which come from a different model, a different threshold and the block axis.

**The category-identity shortcut, which is why Study A is a ranker.** A per-concept regression on the profile difference from the parent appeared to help substantially, with the cross-validated $R^2$ rising from 0.115 to 0.312. Pair differencing cancels the parent term algebraically, and in the differenced form that same distribution scored 53.7 percent alone and dragged Jaccard down from 66.6 to 60.1 percent. A distribution that fails there cannot have been carrying depth information when it appeared to help the per-concept form. The general lesson, that any per-concept feature carrying a category-level component is suspect until it is tested with whole categories held out, is why the surviving typicality study is a ranker and why the per-concept regression is kept only as a baseline column beside it.

**Resolution degrades the raw distribution rather than rescuing it, which is why no raw distribution is carried.** Raising the profile from 28 bins to 196 moved the standalone feature from 53.7 to 49.5 percent, which is chance, and the Jaccard combination from 60.1 to 52.8 percent, with a paired $t$ against Jaccard at $p = 0.048$. Since the finer axis is the one retaining sublayer identity, where module 1 locates the categorical signal, this is evidence that a concept's typicality is not written in where its experts sit even though category membership clearly is. Seven times the parameters on the same 2,391 pairs converted what little the coarse profile held into variance, so the grid carries the two scalar agreement measures and no raw distribution at all.

### The whitened profile, considered and declined

A defensible alternative confound control is the whitened profile, per-bin standardization of the layer distribution across words, which is standard in the representational similarity literature. It is recorded here as considered and DECLINED, with its reasoning, rather than silently omitted.

Per-bin standardization across words targets the SHARED-BACKGROUND confound, that every word inherits the model's global density profile so two arbitrary words already agree substantially before any word-specific structure enters. The count-matched $z$ targets the COUNT confound, that a small expert set yields a spuriously spiky profile through plug-in entropy bias. Those are different nuisances, but they compete for the same slot in the design, since each is a normalization of the same agreement quantity intended to make it interpretable on its own. Carrying both would put two normalizations of one quantity in the grid, which is exactly the sprawl of near-duplicate feature sets this restructure removed. Adopting whitening instead of $z$ would additionally mean removing $z$ from modules 3, 4 and 5, a far larger change than this module. The declined option is therefore recorded rather than built.

### Output layout

```
<scope>/
├── typicality_model_design.csv          shared design frame, both studies
├── 9.1_typicality_ranking/
│   ├── study_a_design.csv               the rows Study A actually fitted
│   ├── ranker_comparison.csv            12 rows, 6 cells x 2 references
│   ├── ranker_per_category.csv
│   ├── axis_sensitivity.csv             whole-model scope at REFERENCE_AP only
│   ├── typicality_ranker_comparison.png
│   ├── ranker_category_dumbbell.png
│   ├── ranker_accuracy_by_gap.png
│   └── ranker_pair_errors.png
├── 9.2_pair_similarity/
│   ├── study_b_design.csv               the rows Study B actually fitted
│   ├── pair_similarity_comparison.csv   6 rows
│   ├── pair_similarity_per_category.csv
│   ├── axis_sensitivity.csv
│   └── pair_similarity_comparison.png
├── sublayer_comparison.csv              cross-scope, written by the executor
└── sublayer_comparison.png
```

`typicality_model_design.csv` stays at the scope root because both studies are built from it. Its columns, in file order, are `concept`, `category`, `human_typicality`, `jaccard_pct`, then per registered metric the label-word pair `profile_<metric>` and `profile_<metric>_z`, then `mean_jaccard_to_members`, then per registered metric the centroid pair `centroid_profile_<metric>` and `centroid_profile_<metric>_z`. At seven registered metrics that is 4 fixed columns plus 4 per metric, 32 in all. With one registered metric it would be the nine columns `concept`, `category`, `human_typicality`, `jaccard_pct`, `profile_js_distance`, `profile_js_distance_z`, `mean_jaccard_to_members`, `centroid_profile_js_distance` and `centroid_profile_js_distance_z`. Each study additionally writes the post-dropna frame it fitted, so the identical-rows guarantee is inspectable rather than only asserted.

`ranker_comparison.csv` carries a `reference` column taking `label_word` or `member_centroid` and a `metric` column naming the profile metric, and its `model` column takes the six grid names. Together those three key the table, so no row is ambiguous about which model it describes. `pair_similarity_comparison.csv` carries the same `model` and `metric` columns without `reference`. Sublayer scopes carry the same structure one level down under `sublayers/<rank>_<sublayer>/`.

## Results

**Provisional.** The numbers below come from a single smoke run, GPT-2 on the Richie-HSJ dataset at AP 0.5 only, whole-model scope, written to `results/research_plots_gpt2_richie_hsj_smoke_m9/`. That run existed to verify the restructure's regression invariants rather than to characterize the module, so it covers one architecture, one threshold and no sublayer comparison worth reporting. The full multi-threshold sweep over both architectures has not been run at the time of writing, and until it has, nothing in this section should be read as a result across AP thresholds. Numbers from the pre-restructure run are deliberately NOT carried over, since the block axis, the retired `centroid_margin` and Study B's own row set all changed what those tables measured.

What the smoke run does establish is that the restructure changed the data path rather than the numbers where it promised to. Every `sublayers/` scope of modules 3, 4 and 5 reproduces the pre-change tree byte for byte on 60 of 76 files, with the 16 exceptions falling into three declared classes, eight module 4 files absent because the modules they read were disabled in that configuration, four `human_similarity_validation.csv` files that gained the `coefficient` and `test` schema columns since the baseline was written, and four `correlation_summary.csv` files, of which exactly one carries a real numeric difference of one unit in the last place on `pearson_p` while the other three differ only by the two rows the disabled modules did not write. Each of the 16 still passes full value equality on every column and row the baseline wrote, at a relative tolerance of $10^{-13}$. Module 9's self-computed default-metric columns equal module 3's own CSV on the same run at a maximum absolute difference of 0.0 over all 197 concepts and all three shared columns. The Jaccard cells match the pre-restructure run exactly, 2,391 pairs and accuracy 0.614806 in Study A and mean per-category $\rho$ 0.3608 in Study B, which is the expected result since Jaccard does not depend on the profile axis.

### Study A, pairwise typicality ranking, provisional

GPT-2, AP 0.5, whole-model scope, 197 concepts and 2,391 within-category pairs over 8 held-out categories. `delta` is the mean per-category accuracy difference against the `jaccard` cell of the same reference, `cats` counts the categories that improved out of 8, and $p$ is the paired $t$ over those 8 categories.

| Reference | Cell | accuracy | accuracy_clear | regression | delta | cats | $p$ |
|---|---|---|---|---|---|---|---|
| label_word | jaccard | 61.5% | 64.7% | 61.5% | | | |
| label_word | profile | 43.5% | 41.6% | 47.3% | $-17.3$ pp | 0/8 | 0.021 |
| label_word | profile_z | 52.4% | 52.7% | 42.4% | $-10.4$ pp | 1/8 | 0.035 |
| label_word | jaccard_profile | 63.0% | 66.2% | 61.3% | $+2.0$ pp | 5/8 | 0.244 |
| label_word | jaccard_profile_z | 61.8% | 64.7% | 60.6% | $+0.3$ pp | 4/8 | 0.617 |
| label_word | jaccard_profile_both | 62.6% | 66.2% | 62.3% | $+1.3$ pp | 6/8 | 0.411 |
| member_centroid | jaccard | 59.5% | 62.8% | 59.5% | | | |
| member_centroid | profile | 46.7% | 46.6% | 47.7% | $-12.1$ pp | 2/8 | 0.054 |
| member_centroid | profile_z | 52.2% | 53.4% | 50.3% | $-7.8$ pp | 2/8 | 0.044 |
| member_centroid | jaccard_profile | 60.7% | 63.1% | 59.6% | $+0.3$ pp | 5/8 | 0.834 |
| member_centroid | jaccard_profile_z | 58.8% | 61.8% | 58.8% | $-0.9$ pp | 2/8 | 0.212 |
| member_centroid | jaccard_profile_both | 63.4% | 68.0% | 62.7% | $+3.1$ pp | 5/8 | 0.190 |

Provisional readings, none of which should be relied on before the sweep. No profile cell reaches significance as an ADDITION to Jaccard under either reference, the largest gain being $+3.1$ points on the centroid reference at $p = 0.19$. As STANDALONE predictors the two profile features behave differently and only one of them is below chance. `profile` sits at 43.5 percent under the label word and 46.7 under the centroid, both below the 50 percent chance line, while `profile_z` sits at 52.4 and 52.2, marginally above chance on all pairs and above it on clear pairs too at 52.7 and 53.4. Both are well below the Jaccard baselines of 61.5 and 59.5, which is what their nominal significance in those rows measures, a significant DEFICIT against Jaccard rather than a gain. Restricting to clear pairs adds between 2.4 and 4.6 points on every Jaccard-bearing cell, and the ranker beats its own regression baseline on every multi-feature cell while tying it exactly on the two Jaccard cells, which is the behaviour the construction predicts.

### Study B, human pair similarity, provisional

Same run, 2,391 rated pairs over 8 categories, mean noise ceiling 0.8914.

| Cell | mean_category_spearman | pooled | rho_over_ceiling | delta vs jaccard | partial $F$ |
|---|---|---|---|---|---|
| jaccard | 0.3608 | 0.1007 | 0.4048 | | |
| profile | $-0.1222$ | $-0.4980$ | $-0.1371$ | $-0.483$ | |
| profile_z | 0.0259 | $-0.4541$ | 0.0291 | $-0.335$ | |
| jaccard_profile | 0.3735 | 0.2229 | 0.4190 | $+0.013$ | 97.8 vs jaccard, $p = 1.3\times10^{-22}$ |
| jaccard_profile_z | 0.3542 | 0.1335 | 0.3974 | $-0.007$ | 33.9 vs jaccard, $p = 6.5\times10^{-9}$ |
| jaccard_profile_both | 0.3890 | 0.2880 | 0.4364 | $+0.028$ | 64.1 vs jaccard_profile, $p = 1.8\times10^{-15}$ |

Provisional readings. The Jaccard cell alone reaches 40 percent of the human noise ceiling with whole categories held out. Adding both profile features lifts the mean per-category correlation from 0.3608 to 0.3890 and the pooled figure from 0.1007 to 0.2880, and every nested partial $F$ clears significance by a wide margin, which is expected on 2,391 pairs and is exactly why the honest reading is the per-category mean and its ceiling-normalized column rather than the $F$ statistic. Alone the profile features are useless or inverted on this target.

### Metric comparison, provisional

The seven-metric extension refits both studies on every registered metric. GPT-2, AP 0.5, whole-model scope, showing the fullest cell `jaccard_profile_both` since that is the comparison the module exists to make. The `jaccard` cell carries no profile metric and is one row for all of them, at 0.6148 in Study A under the label word, 0.5951 under the member centroid, and 0.3608 in Study B.

| Metric | Study A, label word | Study A, centroid | Study B, mean per-category $\rho$ |
|---|---|---|---|
| `js_distance` | 0.6257 | 0.6340 | 0.3890 |
| `wasserstein` | 0.6090 | 0.5847 | 0.3688 |
| `cosine` | 0.6232 | 0.6186 | 0.3754 |
| `pearson` | 0.6144 | 0.5947 | 0.3677 |
| `spearman` | 0.6131 | 0.5935 | 0.3808 |
| `js_divergence` | 0.6265 | 0.6332 | 0.3889 |
| `hellinger` | 0.6257 | 0.6324 | 0.3891 |

Three provisional readings, and none of them changes a conclusion of the module.

The default metric is not beaten in any column by a margin worth acting on. `js_divergence` edges Study A's label-word arm by 0.0008 and `hellinger` edges Study B by 0.0001, both far inside the per-category spread the paired $t$ is computed over, and no metric's paired $t$ against its own Jaccard baseline clears significance in Study A. So the flat reading of the restructure survives: the profile features add little to typicality ranking, and which profile measure is used is not what is holding that down.

The three divergence measures are close to interchangeable, agreeing to within 0.0008 on Study A's label-word arm and 0.0002 on Study B. `js_distance` and `js_divergence` are monotone transforms of one another, so they can differ at all only because these are linear models, and the fact that they barely do says the transform is not where the missing signal is.

`wasserstein` is the weakest metric on both studies and loses most under the CENTROID reference, 0.5847 against the centroid Jaccard baseline of 0.5951, which is the only cell in the table that falls below its own baseline. Reading it with module 5, where `wasserstein` was also last, the consistent answer is that what distinguishes two words' depth profiles is which blocks they occupy rather than how far apart those blocks are, so the one measure that charges transport distance gains nothing on this data.

### Axis sensitivity, provisional

The same run refits `jaccard_profile` on the flat 48-layer axis at the whole-model scope. Study A moves from 63.0 percent on the block axis to 62.3 percent on the flat axis, and Study B's mean per-category $\rho$ moves from 0.3735 to 0.3845. Both differences are small against the per-category spread, which is what the sensitivity line exists to show. It is one architecture at one threshold and it characterizes nothing beyond that.

## Limitations

**The concept-to-label-word count residual.** The count-matched null is defined over the full pair population, and on that population it works on both axes. Measured on this run over 20,910 pairs, the raw agreement $S$ correlates with the pair's smaller expert count at Spearman $+0.28$ on the flat axis and $+0.11$ on the block axis, while the standardized $z$ sits at $+0.0035$ and $+0.0029$ respectively. It does NOT remove the residual on the concept-to-label-word sub-population, where about $+0.21$ persists on BOTH axes, $+0.2100$ flat and $+0.2357$ block over 1,576 pairs. The reason is structural. A concept-to-label pair is systematically asymmetric, since the label word holds one of the largest expert sets in the population, and the pair's neighbourhood in the count coordinate is drawn from the same asymmetric family, so the null has nothing left to subtract.

Three clarifications matter for reading this correctly. The block axis left that slice-level residual essentially unchanged, moving it by $+0.026$, and on the 1,379 foreign-category rows the two axes are indistinguishable at $+0.2385$ flat and $+0.2412$ block. The apparent jump on the 197 own-category rows that Study A actually fits, from $+0.0856$ flat to $+0.2334$ block, reflects the loss of an ACCIDENTAL CANCELLATION in the flat-axis baseline on those particular rows rather than a new confound. And a bin-count mechanism is DISCONFIRMED rather than merely untested, because a binning effect would move the whole slice and the slice moved by $+0.026$.

This residual does not bias Study A's estimates. Concept expert count is itself uncorrelated with human typicality on this population, Spearman $-0.0414$ at $p = 0.56$ over 197 concepts, so a feature confounded with count has no channel through which the confound could manufacture accuracy against that target, and partialling count out shifts `profile_z`'s association with typicality by only 0.008, from $-0.0803$ to $-0.0727$. It is a limitation of describing the label-word profile $z$ as size-free, not a limitation of the study's conclusions.

**`z_cen` calibration is underpowered rather than clean.** The point estimate against concept expert count is $+0.0818$ with a 95 percent confidence interval of $[-0.059, +0.219]$ over 197 concepts, and 80 percent power to detect the stated $|\rho| < 0.15$ bar only arrives near $\rho = 0.198$ at that sample size, so the bar cannot be established here. The supportable claim is COMPARATIVE rather than absolute. On this exact frame $z_{\text{cen}}$ is measurably cleaner than the label-word $z$, Steiger's $Z = -2.19$ at $p = 0.029$ for two correlations sharing the count variable. The null is also visibly doing its job in the intended direction, since raw $S_{\text{cen}}$ carries $+0.162$ at $p = 0.023$ and standardizing it roughly halves the point estimate to a non-significant $+0.082$.

**`z_cen` is a replacement for `centroid_margin`, not a refinement of it.** The two are moderately concordant at Spearman $+0.41$, about 17 percent shared rank variance. Standardization explains only part of the gap, since the raw un-standardized $S_{\text{cen}}$ already agrees with the margin at only $+0.5145$, so the move from $+0.515$ to $+0.409$ is all the standardization accounts for and the remainder is STRUCTURAL. `centroid_margin` is contrastive, the own-centroid agreement minus the mean of seven foreign ones, while $S_{\text{cen}}$ and therefore $z_{\text{cen}}$ is non-contrastive and carries no foreign term at all. Those answer different questions, one asking whether a concept is closer to its own centroid than to the alternatives and the other asking how close it is to its own centroid full stop. The count-behaviour improvement is supported and is the headline, since the margin carries $+0.215$ at $p = 0.002$ against expert count where $z_{\text{cen}}$ carries $+0.082$, but any reliance on $z_{\text{cen}}$ inheriting the margin's previously validated behaviour is NOT established at 17 percent shared rank variance.

**The centroid null matches on pooled count, not on the number of profiles averaged.** Each entry of the centroid null is placed at $(\log n_c, \log N_j)$, where $N_j$ is the centroid's POOLED expert count. A three-member category's leave-one-out centroid averages two profiles while a twenty-member one averages nineteen, so at equal pooled count those two centroids carry different amounts of averaging noise, and the coordinate does not condition that difference out. The effect is second order, but it is a stated property of the construction rather than something to discover later.

**Study B inherits a presence gate it does not strictly need.** The design frame requires the category LABEL WORD to hold at least one expert before any of that category's concepts enter, which Study A needs and Study B's concept-to-concept features do not. The gate is conservative, so Study B under-reports the number of pairs it could have fitted rather than mis-reporting any number it does fit, and its statistical power at strict thresholds is understated rather than overstated. It was left unchanged deliberately, since removing it would mean introducing a second frame-construction path at the very end of the restructure and changing results a second time, and a stated conservative limitation serves the thesis better than a late structural change.

**`vegetables` is anomalous in both studies at once, and the pooled means hide it.** The accuracies and correlations above are means over eight held-out categories, and `vegetables` is the weakest category in both studies simultaneously. In Study A it is below the 50 percent chance line in ALL TWELVE cells, from 36.3 percent on the label-word Jaccard cell and 41.6 percent on that reference's fullest cell to 42.1 and 43.7 percent on the corresponding centroid cells, so no feature set and neither reference recovers its typicality ordering. In Study B it is the only category whose Jaccard correlation with human ratings is negative, $-0.011$, rising to only $+0.090$ on the fullest cell against a category noise ceiling of 0.909, where the next weakest category, `birds`, reaches $+0.104$ and $+0.197$. Being simultaneously the worst category against human similarity and the only one systematically inverted in the ranker points at the category itself rather than at either model, and the most likely candidate is polysemy in its member words, several of which are common non-food senses in a language model. It is recorded here as unexplained. All figures above are from `ranker_per_category.csv` and `pair_similarity_per_category.csv` of the smoke run named in the Results section, and the module draws `ranker_pair_errors.png` precisely so a systematically inverted category can be inspected rather than only averaged over.

**At strict thresholds Study A's numbers can be degenerate.** Measured at AP 0.9 on Qwen3 at the whole-model scope, Study A fits 47 concepts across only 2 categories, and `jaccard_pct` is exactly 0 for 45 of those 47 rows. Those cells must not be read as a result. The guards catch the empty case but not this one, where a study runs, writes a table and reports accuracies built almost entirely on a constant feature.
