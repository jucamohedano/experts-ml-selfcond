# Module 5: Typicality Prediction, Human Pair Similarity and Concept Structure

## Research question

Sections 2.1 and 2.2 of module 2 describe a word pair in two ways. The Jaccard index asks which neurons the two words share, and the layer-profile agreement asks whether the two words spread their experts over depth in the same proportions. Module 2's section 2.4 establishes that the first of those tracks human typicality at Pearson $r$ between 0.19 and 0.48 depending on the model and the AP threshold, 0.303 for GPT-2 and 0.480 for Qwen3 at AP 0.5, falling to 0.193 and 0.315 by AP 0.7 (**Earlier definition**, the current within-category $\rho_W$ reading 0.306 for GPT-2 and 0.432 for Qwen3 on the nopad trees at AP 0.6).

Does the layer-profile agreement add predictive signal over the Jaccard index?

The module asks that one question with two human targets rather than asking four loosely related questions in four subchapters. Study A predicts which of two same-category concepts humans rate as the more typical, and Study B predicts the measured human similarity of a concept pair. Both are fitted on one generated feature grid whose cells carry the same names and the same meaning in either study, so a reader can confirm that the same model was asked the same question on both targets. Since 10 November 2026 the grid asks the question of three kinds of depth information, the profile agreement $S$, its count-matched $z$ and the Jaccard index resolved into quarters of depth, against the text similarity of the probing sentences as a positive control, and `added_value.png` at the scope root draws every gain over $J$ with its interval. The module is deliberately small, with at most five predictors over a couple of hundred concepts, because the question is whether a second feature carries signal the first lacks rather than how well typicality can be predicted in general.

Study C then drops the target entirely and asks the **unsupervised** question, which every other module of this pipeline leaves unasked:

> Given nothing but the pairwise distances between the words' expert representations, what grouping falls out on its own, and is it the category structure the dataset is built on?

That question is not a variant of the first one. Studies A and B start from a grouping we already believe in and measure how well the representations honour it, so they can only ever report how much of an expected structure is present. Study C starts from nothing, so it can report a structure nobody asked for, and it does. It answers in exactly two views, a **dendrogram** of the layer profiles and a **node-link graph** in the style of Figure 4 of Fedzechkina's ExpertLens paper, one folder each under `5.3_concept_structure/`.

## Analysis

Notation. For a word $w$, let $E_w$ be its expert set and $n_w = |E_w|$ its expert count, and let $p_w$ be its layer profile, the row-normalized vector of expert counts per bin defined in module 2, section 2.1. For a concept $c$ with parent category $k$, write $t_c$ for the human typicality rating, $J = J(c,k)$ for the Jaccard index of module 2, section 2.1, $S = S(c,k)$ for the layer-profile agreement and $z = z(c,k)$ for its count-matched null score. The same three symbols apply to a concept pair $(a,b)$ in Study B and to a concept against its category centroid in Study A's second reference, and the resolution table below is the only place they mean anything different.

**Analysis scopes.** The module runs once per scope, whole model first and then once per sublayer type, and a cross-scope summary lands in `sublayer_comparison.csv` and `sublayer_comparison.png`. The sublayer scopes run only at the thresholds in the executor's `DETAILED_AP_THRESHOLDS` (currently 0.5, 0.6 and 0.7), elsewhere the module runs the whole model alone and writes no comparison. Unlike the earlier version it no longer consumes section 2.1's table. It computes every profile feature itself from `scope.expert_df`, calling the same helpers over the same item list, `concept_metadata['concept'].unique()`, so the default metric's label-word columns are identical to section 2.1's CSV by construction while a new profile metric needs no section 2.1 schema change. Module 2 does not have to be enabled for module 5 to run.

### The feature grid

One dictionary, nine cells, generated per registered profile metric by `build_feature_grid(metrics)` rather than written out by hand.

| Cell | Features |
|---|---|
| `jaccard` | $J$ |
| `profile` | $S$ |
| `profile_z` | $z$ |
| `jaccard_profile` | $J$, $S$ |
| `jaccard_profile_z` | $J$, $z$ |
| `jaccard_profile_both` | $J$, $S$, $z$ |
| `jaccard_bands` | $J$, $J_1$, $J_2$, $J_3$, $J_4$ |
| `text` | $s$ |
| `jaccard_text` | $J$, $s$ |

The grid is a nested design rather than a list. The single-feature cells establish standalone power, without which a gain over Jaccard is uninterpretable. The two-feature cells and `jaccard_bands` test each addition to Jaccard separately, and `jaccard_profile_both` tests $S$ and $z$ together. Every comparison in the module is therefore a well-defined nested comparison between two rows of one table. The cells without a profile metric, `jaccard`, `jaccard_bands`, `text` and `jaccard_text`, come first and once, the profile cells after them once per metric.

**Depth-resolved overlap, the role $B$.** The blocks are split into four equal quarters, blocks 0 to 6, 7 to 13, 14 to 20 and 21 to 27 in Qwen3-1.7B-Base and 0 to 2, 3 to 5, 6 to 8 and 9 to 11 in GPT-2, the finest split that divides both models into equal bands, since 28 and 12 share only the divisors 1, 2 and 4. With $E_a^{(b)}$ the units of $E_a$ that lie in quarter $b$,

$$J_b(a, c) = \frac{|E_a^{(b)} \cap E_c^{(b)}|}{|E_a^{(b)} \cup E_c^{(b)}|},$$

set to 0 when neither word holds a unit in the quarter, and the four values decompose the Jaccard index exactly,

$$J(a, c) = \sum_{b=1}^{4} w_b\, J_b(a, c), \qquad w_b = \frac{|E_a^{(b)} \cup E_c^{(b)}|}{|E_a \cup E_c|},$$

where $w_b$ is the share of the union that lies in quarter $b$. $J_b$ asks whether two words share units at a given depth, which $S$ cannot ask, since $S$ compares where two expert sets sit without asking whether they hold the same units. The block count is the architecture's, 12 or 28 (`n_blocks` of `LAYER_ARCHITECTURES`, recognised from the sublayer names), so a word whose experts all sit early, or a sublayer scope whose last blocks hold no expert at a strict threshold, keeps the model's quarters. The columns hold $100 J_b$, and `check_module5_depth_bands.py` rebuilds $J$ from the four quarters.

**Text similarity, the role $s$.** The TF-IDF cosine of the two words' probing sentences, the text control of module 2, section 2.3, computed by `modules/shared_text_measures.py`. It is not a depth feature. It enters as the positive control, a feature known to carry information the expert sets lack, so a comparison that cannot detect a gain of $\{J, s\}$ over $\{J\}$ could not detect a gain of a depth feature either. The two text cells exist only when the configuration names a probing-sentence folder, and the axis sensitivity line runs without them.

The three symbols resolve to concrete columns per study and per reference as follows.

| Symbol | Study A, `label_word` | Study A, `member_centroid` | Study B |
|---|---|---|---|
| $J$ | `jaccard_pct` | `mean_jaccard_to_members` | `jaccard_pct` of the pair |
| $S$ | `profile_js_distance` | `centroid_profile_js_distance` | `profile_js_distance` of the pair |
| $z$ | `profile_js_distance_z` | `centroid_profile_js_distance_z` | `profile_js_distance_z` of the pair |
| $J_b$ | `jaccard_band_<b>_pct` | `mean_jaccard_band_<b>_to_members` | `jaccard_band_<b>_pct` of the pair |
| $s$ | `text_to_label` | `mean_text_to_members` | `text_similarity` of the pair |

The `js_distance` segment names the profile metric, see the metric registry below, and `<b>` runs from 1 to 4. The member columns of $J_b$ and $s$ average over the same other members as `mean_jaccard_to_members`. Every comparison CSV carries a `metric` column naming the metric a row was built on, and the cells that use no profile metric carry `metric = "none"` and appear once rather than once per metric. With one registered metric this reduces exactly to the tables above, and with the eight registered today the same table repeats once per metric under one vocabulary.

**The identical-rows rule.** All cells within a study are fitted on identical rows, through a single `dropna` over every candidate feature of that study performed once before any model is fitted. This exists because $z$ is undefined for words below `MIN_PROFILE_EXPERTS = 2`, so a per-cell `dropna` would train the Jaccard cell on more concepts than the others and the accuracies would no longer be comparable. The scope of the rule is the study rather than the module. Study A drops over both references' columns including the centroid ones, because its two arms are compared against each other inside one figure. Study B does its real dropna somewhere else entirely, and this is easy to misread. It takes only `concept` and `category` off the design frame and recomputes every feature it fits as a symmetric PAIR quantity, so the frame-level `study_b_columns` list is `["jaccard_pct"]` alone and is a DEFENSIVE NO-OP, since the presence gate already guarantees a surviving row has a defined `jaccard_pct`. The identical-rows guarantee for Study B is enforced inside `run_pair_similarity`, by one shared dropna over the union of every PAIR column its six cells use, applied to the pair table before any cell is fitted. The list must not be "fixed" by adding the profile columns back. Those are concept-against-LABEL-WORD quantities Study B never reads, and gating on them deleted an entire category at AP 0.9, where the label word `vehicles` held one expert while all 22 of its concepts were fine, which took Study B to two categories and an empty result. The source carries the same warning at length. The two studies are never compared numerically against each other, so nothing is lost by letting their row sets differ. `check_module5_pair_similarity_identical_rows.py` verifies the rule by counting the rows each cell actually sees.

### Study A, typicality by pairwise ranking

**Target and unit.** `typicality_HSJ_pairwise`, a derived quantity. Richie and Bhatia elicited pairwise similarity, and this column was produced by averaging each word's row of the resulting matrix. The unit is the ordered within-category pair. Summing $\binom{n_k}{2}$ over the eight categories gives 2,418 pairs from 198 concepts. Cross-category pairs are excluded, since typicality is undefined between them, and exact ties are excluded, since they carry no direction to predict.

**Model.** For concepts $a$ and $b$ sharing category $k$, with per-concept feature vector $\varphi(\cdot)$,

$$f(a, b) = \sigma\big(w^{\top}(\varphi(a) - \varphi(b))\big),$$

where $\sigma(x) = 1/(1 + e^{-x})$ is the logistic function and $w$ the fitted weight vector, a logistic model on the feature difference, fitted with no intercept, predicting $\mathbb{1}[t_a > t_b]$. Each pair enters twice, as $(a,b)$ labelled 1 and as $(b,a)$ labelled 0. Scaling uses `StandardScaler(with_mean=False)`.

Both the missing intercept and the uncentred scaling are required for the exact antisymmetry $f(b,a) = 1 - f(a,b)$, since either a constant term or a subtracted feature mean would let the model express a preference that does not flip when the pair is presented the other way round. `check_module5_ranker_antisymmetry.py` pins this to numerical tolerance for every grid cell. One warning belongs with it, because it cost a round to find. A test fitted on the mirrored training set alone cannot see a violation, since that set is invariant under $(x, y) \mapsto (-x, 1-y)$, which forces the fitted intercept to exactly zero and the column means to exactly zero, so an intercept-bearing model passes at machine epsilon. The check therefore carries a second arm fitted on unmirrored imbalanced data, where an intercept moves the residual by seven to nine orders of magnitude above the tolerance.

**Why this form.** Any component of $\varphi$ shared by all members of a category disappears from $\varphi(a) - \varphi(b)$. Category identity alone explains $R^2 = 0.21$ of typicality variance on this dataset, so a per-concept model can score well by recovering the category and never learning anything about typicality. In the differenced form that shortcut is unavailable by construction rather than merely controlled for. The form also matches how the ratings were collected.

**The reference factor.** Every cell is fitted twice, once against the category label word and once against the leave-one-out member centroid, and each arm carries its own Jaccard baseline so no gain from a profile feature is ever credited to a change of reference. Scoring a concept against the mean of the other category members is the classic family-resemblance operationalization of typicality and is the one the psychological literature is built on, while scoring it against the category label word is the model-specific convenience. Carrying both as a factor of one study makes the comparison a main effect of the design rather than a separate chapter.

**The regression baseline column.** Every cell additionally reports `regression_accuracy`, a per-concept CROSS-VALIDATED RIDGE regression on the same features, applied to exactly the same held-out pairs and ranking each pair by its two predicted ratings. The estimator is `make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 30)))`, so the penalty strength is chosen inside the training fold rather than fixed, and the baseline is a penalised fit and not an unpenalised least squares one. This puts the two formulations on one scale, accuracy against accuracy rather than accuracy against $R^2$, and it is why the per-concept regression that used to be a subchapter of its own is no longer one.

Since 10 November 2026 the ridge is fitted and applied on within-category rank scores of the features and of the ratings, as Study B's cells are (below). The ranker only ever sees differences inside one category, and the ridge on raw values pooled over categories could lean on the category means of a feature, which tell nothing about which of two members is the more typical. Its values before that date are an **Earlier definition**.

For the Jaccard cell alone the two columns are identical, and the ridge penalty does not disturb that. Standardizing a single feature and shrinking its coefficient towards zero leaves the sign of the coefficient unchanged, so a single monotone feature induces the same ordering of predicted ratings under either fit. The table below shows the identity holding on both Jaccard rows, and it holds exactly in all sixteen per-category rows of `ranker_per_category.csv`, the eight held-out categories under each of the two references, not only in the pooled mean. Where several features are present the two formulations can diverge, and the divergence has two sources rather than one. The ranker optimizes the ordering of pairs directly while the regression optimizes squared error on the rating, and the ridge penalty additionally shrinks the regression's coefficients towards each other. The honest reading of the column is therefore that the differenced ranker beats its own regression baseline on every multi-feature cell while tying it exactly on the two Jaccard cells, and it does not on its own isolate the trading off of several features as the reason.

**Evaluation.** `GroupKFold` over categories, holding out one whole category at a time, eight folds. Because every pair lies inside a single category, holding out a category removes every pair containing any of its concepts, so no concept ever appears in both halves. Three accuracies are reported per cell, `accuracy` over all held-out pairs, `accuracy_clear` over pairs whose ratings differ by at least `RANK_MARGIN = 0.10`, and `regression_accuracy` as above.

**Significance.** A paired $t$ test against the `jaccard` cell of the same reference, taken over the eight per-category accuracies rather than over pairs. `delta_ci_low` and `delta_ci_high` bound `delta_accuracy_vs_jaccard` by the 95% $t$ interval of the same differences, $\bar d \pm t_{0.975,\,K-1}\, s_d / \sqrt{K}$, with $\bar d$ and $s_d$ the mean and standard deviation of the $K$ per-category differences and $t_{0.975,\,K-1}$ the 97.5th percentile of Student's $t$ with $K - 1$ degrees of freedom (`mean_difference_interval` in `utils/statistics.py`). Thousands of pairs built from a couple of hundred concepts are not independent observations, so a test treating them as independent would badly overstate significance. The eight held-out categories are close to independent replicates and are the honest unit of analysis. The comparison is keyed on the triple of reference, metric and cell, so registering a second metric cannot pool two metrics' per-category accuracies into one delta.

**Fold weights.** `ranker_fold_weights.csv` gives the ranker's weight of every feature in every fold, on standardised feature differences, with the reference, the cell and the metric. A weight whose sign holds in all eight folds is a stable part of the model, a sign that flips is noise, and the comparison of the two cells alone cannot tell them apart.

**Guard.** Fewer than `MIN_RANK_PAIRS = 200` usable pairs skips the study for that scope.

### Study B, human pair similarity by regression

**Target.** The elicited ratings themselves, Richie and Bhatia Study 1, on a 1 to 7 scale with higher meaning more similar, 2,418 within-category pairs over eight categories with 19 to 39 raters each, loaded by `core/human_similarity_ratings.py`. This is external validation rather than a proxy, since the target is a number humans produced for that specific pair, and category is constant within every pair, so category identity alone scores at chance and the shortcut that haunts a per-concept target is closed by construction.

**Why the model differs from Study A.** The ranker predicts a direction and is built from the antisymmetric difference $\varphi(a) - \varphi(b)$. Human pair similarity is a value attached to the pair and is symmetric in its two members, so the features must be symmetric too. They are the pair quantities $J(a,b)$, $S(a,b)$ and $z(a,b)$, which are section 2.2's matrices rather than section 2.1's concept-to-parent table. Predicting a symmetric target from antisymmetric features would be incoherent, so this is a regression and not a ranking. A pair has no parent either, so the reference factor of Study A does not arise here.

**Fitted within categories.** Since 10 November 2026 every cell is fitted on within-category rank scores, the average ranks of each feature and of the ratings inside each category, standardised to mean 0 and standard deviation 1, by `within_category_predictions`. The cells are evaluated by a within-category Spearman, so only the order of the pairs inside a category counts, while a least squares fit on raw values pooled over categories also learns how the category means of a feature relate to the category means of the ratings. For the profile agreement that relation runs the other way, the Spearman over the eight category means of $S$ and of the ratings being $-0.26$ / $-0.67$ (Qwen3-1.7B-Base / GPT-2) on the nopad trees, so the pooled fit gave $S$ a weight partly set by eight category means rather than by the pairs. The rank fit removes every category mean by construction and leaves the single-feature cells as they were, since one feature with a positive weight keeps its order inside each category. `group_cv_r2` is read on the rank scores of the ratings. The pooled fit is **not generated by the current code**, and its values are kept in `fixes.md`. `check_module5_within_category_fit.py` builds categories whose means run against the ratings and checks that the weight keeps its within-category sign in every fold.

**Evaluation and the headline column.** `GroupKFold` over categories. The headline is `mean_category_spearman`, the pair-weighted mean $\rho_W$ of the per-category Spearman correlations between the out-of-fold predictions and the ratings, over the categories with at least 10 pairs, computed by one grouped `spearman_test` (the within-category modality of the README's *Correlations and permutation tests*). For the Jaccard cell it is therefore the $\rho_W$ of section 2.3, 0.5266 / 0.3513 (Qwen3 / GPT-2) on the nopad trees at AP 0.6. The pooled correlation over all pairs, `group_cv_spearman`, and until 19 October 2026 the unweighted mean of the per-category values are **not generated by the current code**. Pooling is the wrong reading. Out-of-fold predictions for a held-out category are miscalibrated in level relative to the categories the model trained on, so pooling them introduces a between-category component that can be strongly negative while every within-category ordering remains correct. In the final trees the two columns differ by 0.29 on GPT-2's Jaccard cell alone at AP 0.5, 0.3597 against 0.0714, and on Qwen3 at AP 0.8 the pooled figure is $-0.107$ against $+0.529$ per category. `group_cv_r2`, the out-of-fold $R^2$ over all pairs, is kept as a description of fit.

**Noise ceilings.** Each per-category correlation is also divided by that category's split-half Spearman-Brown ceiling, which runs from 0.836 for birds to 0.935 for vehicles. `noise_ceiling_mean` is the pair-weighted mean ceiling over the same categories, and `rho_over_ceiling` divides $\rho_W$ by it. `pooled_rho_over_ceiling`, the pooled correlation over the same ceiling, is **not generated by the current code**.

**Nested tests.** Each cell is compared with every cell it extends by exactly one feature within the same metric by a paired $t$ test over the held-out categories, the per-category $\rho_k$ of the fuller cell against those of the reduced one, as Study A compares its cells, written into the column pair `paired_t_vs_<reduced>` and `paired_t_p_vs_<reduced>`, NaN when fewer than three categories carry both values. The test weights every category equally, while the headline $\rho_W$ and `delta_mean_rho_vs_jaccard` weight them by their rated pairs, 190 to 435, so the two can disagree in sign when the gain is small, as in 10 of the 240 comparisons on the nopad trees at AP 0.5 to 0.7, each with a pair-weighted gain within 0.006 of zero and a paired p of at least 0.73 or no test. `delta_vs_<reduced>` with `delta_ci_low_vs_<reduced>` and `delta_ci_high_vs_<reduced>` give the mean of the per-category differences that test compares, with the 95% $t$ interval of Study A, the unweighted counterpart of `delta_mean_rho_vs_jaccard`. `pair_similarity_fold_weights.csv` gives the weight of every feature in every fold, and `pair_similarity_table.csv` the pair table every cell was fitted on, after the shared dropna. Several cells have more than one reduced model, which is why the reduced cell's name is carried in the column name. Until 19 October 2026 the comparison was an in-sample partial $F$ over the pairs, the columns `partial_f_vs_<reduced>` and `partial_f_p_vs_<reduced>`, **not generated by the current code**, since the 2,418 pairs share 198 members and are not independent observations.

**Guards.** The study needs at least `MIN_PAIR_CONCEPTS = 3` concepts to form a within-category pair at all, and fewer than `MIN_PAIR_ROWS = 200` rated pairs after the shared dropna skips the study.

### Partial correlations of every feature

`ranker_partials.csv` and `pair_similarity_partials.csv` give the unique association of each feature with its target, the within-category partial Spearman of the README's *Correlations and permutation tests*. Every $J$, quarter, text and profile column, the raw and the count-matched profile of every metric, is tested under four control sets, none, $J$, $s$, and $J$ with $s$, leaving out a set that holds the feature's own role, so $J$ is read alone and given $s$, $s$ alone and given $J$, and every other feature under all four. Study A tests the columns of each reference against typicality with 9,999 draws within categories, and Study B tests the pair columns against the ratings with 9,999 Mantel draws, which permute the words inside each category (`pairs` the positions of the two words in the sorted list of the table's concepts). The partial measures what a feature holds beyond its controls without refitting a model, so it complements the cross-validated gain of the grid, and given $s$ it answers the text-controlled question for every metric without a cell per metric. The text control sets appear only with the text cells. `check_module5_inference.py` compares a row with `spearman_test` under the same draws.

### Study C, concept structure

**Target.** None. This study is fitted to nothing and predicts nothing, so it has no train and test split, no held-out category and no accuracy. Everything it reports describes the representation itself, scored against groupings it never saw. It has exactly two deliverables, one folder each.

**Input.** The scope's expert frame, re-keyed onto the block axis for the whole-model scope and left on its own axis otherwise, giving each item $c$ its layer profile $p_c$ over $B$ depth bins ($B = 12$ on GPT-2, $B = 28$ on Qwen3). Every word with a usable profile enters, category labels included. The study is self-sufficient from `scope.expert_df` and independent of the design frame studies A and B share, so it runs even at strict thresholds where those two have no rows left.

#### The dendrogram, and where its divisions appear

**How the tree is built.** Agglomerative, bottom up. Every word starts as its own cluster, and at each step the two closest clusters are merged, so a tree over $n$ words makes exactly $n - 1$ merges. The **height** of a merge is the distance between the two clusters at the moment they joined, under the linkage rule in use, and it is the only quantity the horizontal axis of the figure shows. Under `average` linkage, the winner on both architectures, that distance is

$$ d(A, B) = \frac{1}{|A|\,|B|} \sum_{a \in A} \sum_{b \in B} d(a, b), $$

the mean distance between every member of one cluster and every member of the other. Merges are made in non-decreasing height order, so the tree is drawn narrow near the leaves and wide near the root.

**When a division appears.** A division is a merge read backwards. Cutting the tree at height $h$ severs every link drawn beyond $h$, and what remains are the clusters that existed just before the first merge above $h$. Cutting for $k$ clusters is the same operation stated differently, since undoing the last $k - 1$ merges leaves $k$ groups, so the cut height sits just below the $(k-1)$-th largest merge height. The dendrogram figure draws that height as a dashed line and colors the branches by it.

That says where a division can be **taken**, not where one **exists**. For the second question the quantity is the gap between consecutive merge heights, written to `merge_heights.csv` as `height_gap`. The figure `merge_heights.png` of earlier versions is **not generated by the current code**. A large gap means the next thing the algorithm had to join was much further away than anything it had joined so far, which is the signature of a division the data actually carries. A run of small gaps means the merges there cost nothing, so the groups they produce are an artefact of having to keep merging rather than a boundary. A tree whose gaps are all small has no natural number of clusters at all, and the honest reading of it is that the words form a gradient. The final merge is excluded from every "largest gap" statement, because it joins everything into one cluster and cutting there is not a division.

**One dissimilarity per registered metric.** Study C sweeps `PROFILE_METRICS` exactly as sections 2.1 and 2.2 do, so every registered way of comparing two layer profiles gets its own tree and its own graph, and the choice of metric becomes visible rather than assumed. Every metric reports an agreement on a 0 to 100 scale with 100 meaning identical profiles, so one conversion serves all of them,

$$ d(a, b) = \frac{100 - S(a, b)}{100}. $$

For `js_distance` that is exactly $\sqrt{\mathrm{JSD}(p_a, p_b)}$, the Jensen-Shannon distance and a true metric on the simplex, so the default metric's numbers are unchanged by the sweep. The two signed metrics, `pearson` and `spearman`, run to -100 rather than 0, so theirs lands in $[0, 2]$. That is a scale difference rather than a problem, since average linkage and every rank statistic reported here are invariant to it, but it is the reason raw distances must never be compared across metrics. All eight are built in one pass by `layer_profile_metric_matrices`, which shares the profile build and the Jensen-Shannon tensor rather than recomputing them eight times.

One further space, `expert_set_jaccard`, is not a profile metric at all. Its dissimilarity is $1 - J(a,b)$ on the two words' expert SETS, the same Jaccard index sections 2.1 and 2.2 use, and it is the one space here that describes a word by **which** neurons it recruits rather than by **where along depth** it puts them. Clustering it with the identical machinery is what turns the set-versus-profile comparison into a controlled one: same words, same linkage, same cut, same scoring, only the description of a word changes. It also places more words than the profile spaces at strict thresholds, since a Jaccard is defined for a word holding a single expert while a layer profile needs `MIN_PROFILE_EXPERTS`.

One extra space, `js_distance_z`, carries the **count-matched null** of the default metric. It exists because of the confound stated in `helpers.layer_profile_matrices`: raw profile agreement is very largely a readout of expert-set **size**, correlating with the pair's smaller expert count at Spearman 0.965 on synthetic data carrying no true signal, so a tree built on the raw distance alone would recover "small-set words against large-set words" and present it as a discovery. One metric carries the control because the null is metric-agnostic by construction, so running it eight times would repeat the same check on eight monotone rescalings of the same profiles. Its scale is ordinal, the shift that makes it non-negative gives its zero no meaning, so it is read by average linkage and rank statistics only.

**Three candidate trees**, ranked by cophenetic correlation, which measures how faithfully a tree reproduces the distances it was built from and is therefore the criterion for which tree to read rather than a result. `average` and `complete` run on the distance matrix directly. Ward is **not** run on the distance matrix, because Ward's criterion is defined through Euclidean variance and a Jensen-Shannon matrix is not Euclidean. It runs instead on the square-rooted profiles $\sqrt{p_c}$, whose Euclidean distance is $\sqrt{2}$ times the Hellinger distance, which is the Euclidean space this problem legitimately has.

**One figure per deliverable.** Each metric produces exactly one dendrogram and one link graph. Every candidate tree is still scored, so the cophenetic ranking that picks one is evidenced, but only the winner is drawn, since three dendrograms per metric across eight metrics would be twenty-four near-identical figures answering a question the ranking CSV answers in one line. The drawn file is named after the winning linkage, `dendrogram_average.png`, `dendrogram_complete.png` or `dendrogram_ward_hellinger.png`, and average linkage wins in 96 of the 100 whole-model spaces of the two final trees, complete linkage in the other 4. The merge-height curve and the cluster quality curve are likewise kept as CSVs, `merge_heights.csv` and `cluster_quality_by_k.csv`, and not drawn: they are diagnostics behind the numbers quoted in the Results section rather than things to display, and each can be redrawn from its own table without rerunning anything. The cluster mean-profile plot of earlier versions is **not generated by the current code**.

**Scoring the cut.** For every $k$ from 2 to 20, the cut is scored against the structure we expect (adjusted Rand and normalized mutual information against the categories, adjusted Rand against the abstraction level), against internal quality (silhouette, with the largest cluster's share beside it so a cut isolating four outliers is not read as a two-way structure), and against the confounds we do not want, through
$$ \eta^2 = \frac{\sum_j n_j (\bar{x}_j - \bar{x})^2}{\sum_i (x_i - \bar{x})^2} $$
for $x$ in turn the log expert count, the peak bin, the depth centre of mass, the profile entropy and Geary's C. A cut with $\eta^2 = 0.8$ on entropy and an adjusted Rand of 0.1 on the categories has grouped the words by shape, not by meaning.

**How to read the three numbers.** Each answers a different question about the same tree, and each one alone misleads.

- **Cophenetic correlation** asks *is the tree lying about the distances?* A dendrogram is a lossy summary: it claims two words are as far apart as the height where their branches first join, and this correlates that claim against the real distance over all pairs. 1 means nothing was lost in the squashing, 0 means the picture is unrelated to the data. It measures faithfulness to the input distances and **not** whether those distances carry anything, so a perfectly accurate map of an empty field still scores high. That is not hypothetical here: `pearson` and `spearman` build the two most faithful trees in the sweep and the two emptiest.
- **Adjusted Rand index against the categories** asks *did the tree rediscover the categories on its own?* It cuts the tree into as many groups as there are categories, then for every pair of words asks whether the tree and the category agree about putting them together. "Adjusted" means the agreement that random grouping would produce by luck has been subtracted off, so 0 is chance and 1 is an exact match.
- **$\eta^2$ on a descriptor** asks *what did the tree group by instead?* It is the share of that descriptor's variation across words that knowing a word's cluster lets you predict: 0 means the clusters are unrelated to it, 1 means the clusters simply are that descriptor. Read on the depth centre of mass, it says how much of "where in the network this word's experts sit" the clustering has captured.

Read together they are the finding of this study in one line: an adjusted Rand of 0.115 beside an $\eta^2$ of 0.675 on depth says the clusters carry almost nothing about what a word means and a great deal about where in the network it lives.

**The tree's diagnostic.** `category_separation.csv` and `level_separation.csv` ask how well the groups we already believe in survive in the same distance space, so a low agreement between tree and categories can be attributed either to the tree or to the groups. For each group the separation is the mean distance from a member to a non-member minus the mean distance among its own members, with a two-sided $p$ value from 9,999 reassignments of the labels over the words, drawn from a generator of its own, $p = (1 + \#\{|\Delta_b| \ge |\Delta|\})/(1 + 9{,}999)$, centred at zero, the separation any group has in expectation under random labels, with floor 0.0001. The `ALL GROUPS` row averages the separation over the groups of at least two words and tests that mean on the same reassignments. A group of one word has NaN. Until 19 October 2026 the p was one-sided, from 2,000 shuffles of one generator seeded at `CLUSTER_SEED`, with floor 0.0005, and `CLUSTER_SEED` now seeds the MDS layouts of the link graphs alone. The permutation null is mandatory rather than optional here, since the groups are very unequal in size (8 label words against 198 members, categories of 20 to 32) and a plain comparison of means would reward the small groups by construction.

**Guards at strict thresholds.** A word enters the study only if `layer_profile_matrices` could build a profile for it, which needs `MIN_PROFILE_EXPERTS = 2` experts. That helper marks a word it could not place by filling its whole row **and column** with NaN, so usability must be read off the DIAGONAL: a usable word sits at distance 0 from itself, an unusable one at NaN. Testing whether a whole row is finite would be wrong, because one unusable word puts a NaN in every other word's row and collapses the usable set to nothing.

How much this matters grows quickly with the threshold. On the Qwen3 `mlp.down_proj` scope, 206 of 206 words are usable at AP 0.5, 205 of 206 at AP 0.6 (the single word `vehicles` holds one expert there) and 197 of 200 at AP 0.7. The sublayer scopes do not run at AP 0.8 and 0.9, which `DETAILED_AP_THRESHOLDS` leaves out, and there the whole model keeps 196 usable profiles among the 201 words with experts at AP 0.9. Two consequences are handled rather than assumed away. A dissimilarity space with no more usable words than the $k = 8$ cut asks for is skipped with a warning, since the cut height it would draw is not defined, and the rest of the study proceeds. A labelling with fewer than two distinct groups among the survivors, which is what the abstraction level becomes once every category-label word has dropped out, yields NaN for its separation and agreement columns rather than an exception. Every scope therefore contributes the same columns to `sublayer_comparison.csv`, with NaN where a deliverable could not run.

#### The link graphs

Figure 4 of Fedzechkina's ExpertLens paper (`temp_docs/ExpertLens paper-Fedzechkina.pdf`) draws the concept structure of the expert space as a node-link diagram, one node per concept with edge thickness carrying the Jaccard similarity of the two concepts' expert **sets**, and reads three things off it: that within-domain concepts are densely connected, that cross-domain edges are sparser, and that the cross-domain edges which do survive are meaningful, its examples being "driver" reaching "bus" and "vehicle" and "racing" joining sports to vehicles. Note that this figure is a node-link diagram and not an MDS, which is easy to misremember.

The figure is drawn here twice, on the two similarities this pipeline has, so the geometries can be read side by side:

- `expert_set_jaccard`, the quantity the paper uses, which asks **which** neurons two words share
- `layer_profile_agreement`, the quantity every profile-based module here uses, which asks **where** along depth two words put their experts

Three adaptations are needed and each is deliberate. First, the paper draws 40 concepts in 10 domains and can afford every edge, while this dataset has 206 words and 21,115 pairs, which at any global threshold is either a hairball or an empty canvas, so each node keeps its `GRAPH_NEIGHBORS = 3` strongest partners and an edge kept by either endpoint is kept once. That asymmetry is intended, since dropping an edge one concept counts among its strongest because the other does not would hide exactly the hub structure the figure exists to show. Second, node positions come from a 2-D metric MDS rather than from a force layout, which with 206 words would be unreadable and would change with the seed. Each graph is laid out by the same quantity its own edges carry, which matters more than it sounds: an earlier version, **not generated by the current code**, positioned both graphs by the layer-profile distance while the Jaccard graph's edges carried expert-set overlap, so the layout fought the edges, placing strongly connected words far apart because they happened to differ in depth. Measured on the 2-D category silhouette of the Jaccard graph of the earlier 205-word GPT-2 run, matching the layout to the edges moved it from -0.120 to +0.073, which is the difference between categories that read as clumps and categories that read as noise. MDS appears in this module for that one purpose, and nothing about stress or dimensionality is written out because the embedding is furniture here rather than a result. Third, the expert-set Jaccard is computed on the scope frame and not the depth-aggregated one, since a set is unordered and block aggregation would merge distinct units of one block into one bin and inflate every overlap.

The paper's qualitative claims are then made countable. `concept_graph_domain_structure.csv` compares the observed share of edges joining two words of one category against the share obtained when the category labels are reassigned over the words, 9,999 times from a generator of its own, which preserves the graph and the category sizes and destroys only the assignment, mirroring the resampled-domain baseline the paper uses for the same question. `null_within_category_pct` and `null_sd_pct` are the mean and standard deviation of the reassigned shares. The p is two-sided, centred at the share expected under random labels, $\sum_k n_k(n_k-1)/(N(N-1))$ with $n_k$ the words of category $k$ and $N$ their total, and floored at 0.0001. Until 19 October 2026 it was one-sided, from 2,000 shuffles seeded at `CLUSTER_SEED`. It is reported over all edges and over the strongest quartile separately, since the figure draws attention to the thick edges and a structure carried only by the weak ones would be a different claim. The edge list additionally carries `same_first_token`, whether the two words open with the same token of the model's tokenizer, read from `assets/concept_tokens.csv` (`dataset_and_metadata.md`, section 12), because a neuron that is an expert for a written FORM rather than for a meaning would join words like tuxedo and trolley across domains, and the column makes that alternative countable instead of leaving it to be assumed away. One word in two categories, squash, is never flagged, and the column is empty when the table lacks the architecture. `linkgraph_summary.csv` counts the flagged edges across categories in `cross_category_same_first_token`. Until 10 November 2026 the column was `shared_prefix`, the length of the common leading substring, counted in `cross_category_with_shared_prefix` at three or more letters, both **not generated by the current code**. It missed the effect, since most words that open with the same token share one or two letters only.

### Study D, depth-resolved overlap

Whole-model scope only, in `5.4_depth_resolved_overlap/`, run after Study B on the pairs Study B fitted and on Study A's design frame. It locates along depth, one block at a time, the overlap that follows the ratings and the overlap that follows the written form, and it predicts nothing.

`overlap_by_block.csv` has one row per block, with `quarter` counted from 1 as in the band columns, `pair_rho`, the within-category Spearman of the pairs' Jaccard index inside the block with the rated similarity, `pair_rho_given_text`, the same partial on the text similarity, empty without text, `typicality_rho`, the within-category Spearman of each member's Jaccard index with its category name inside the block with typicality, and the shared and union units of the rated pairs in the block, each with its share of the total. The coefficients are descriptive and carry no p.

Since 12 November 2026 the same file also holds a size-matched null for each block, because a block holds a small part of the expert units and a small set lowers any rank correlation by itself. `zero_share` is the share of rated pairs that share no unit inside the block. A random set takes units from the whole model in random order until it holds as many expert rows over the rated pair words as the block, and Study D draws 200 such sets per block with seed 0 (`DEPTH_NULL_DRAWS`, `DEPTH_NULL_SEED`). `random_pair_rho_mean`, `random_pair_rho_low` and `random_pair_rho_high` give the mean and the 2.5th and 97.5th percentiles of their within-category Spearman with the ratings, the `random_pair_rho_given_text_*` columns the same for the text partial, empty without text, and `random_zero_share` the mean share of rated pairs that share no unit in a random set. A block above `random_pair_rho_high` follows the ratings more than any set of its size from every depth, so the comparison reads depth apart from size. Fed the units in the order of the one-off check in `temp_docs/depth_gap_2026-11-11/`, `size_matched_null` reproduces its GPT-2 values to within $10^{-16}$. The pipeline's expert table lists the units in another order, so the same seed draws other sets, and on the v3 trees, rerun on 12 November 2026, the block values are identical while the random means differ from the one-off by up to 0.005 and the bounds by up to 0.011, the Monte Carlo error of 200 sets. A block is read as outside the range only beyond that margin. Trees written before that date lack the columns until module 5 reruns.

`shared_unit_depth.csv`, `same_token_pairs.csv` and `same_token_summary.csv` locate the units shared by member pairs. The pairs fall into four groups, pairs within one category, pairs across categories whose words open with the same token, pairs across categories that share only the first letter, and the other pairs across categories, the last group holding every pair across categories without a shared first token, the first-letter pairs included. One word in two categories belongs to no group. `shared_unit_depth.csv` counts per block the shared units of each group and every unit of the members, each with its share of the group's total. `same_token_pairs.csv` gives per same-token pair its Jaccard percentage, its percentile among the member pairs across categories, its shared units, their mean depth $\ell / (L - 1)$ for a unit in block $\ell$ of $L$, and their share in the last quarter. `same_token_summary.csv` gives per group its pairs, the pairs in which at least one word holds an expert (`n_pairs_with_experts`, the pairs the mean reads, fewer than all only where words lack experts, as on the context-only trees), the mean Jaccard percentage, its ratio to the other pairs across categories and the depth of the shared units. The three files need the token table and are skipped with one warning without it.

`overlap_by_depth.png` draws, on the top panel, `pair_rho` and its text partial per block ("This block's units", "This block's units, text partialled out") against the whole-model $\rho_W$ of $J$ and its text partial as dashed lines ("All units (whole model)", "All units, text partialled out"), which equal the `jaccard_pct` rows of `pair_similarity_partials.csv` with the controls none and text. A shaded band gives the 95% range of the random sets ("Random units of the block's size, 95% range") where the file holds it. The bottom panel draws per block the share of each group's shared units, with the members' own expert units as the reference, and the quarter boundaries are light rules, named in the x label. The labels were reworded on 11 November 2026, the values unchanged. A block holds 4.6% to 16% of the expert units in GPT-2, so the dashed lines compare a block with a set six to twenty-two times larger, see *What depth adds to the Jaccard index* for why they sit above every GPT-2 block and below some Qwen3-1.7B-Base blocks. `check_module5_depth_resolved_overlap.py` covers both tables on a synthetic model, the random sets (each reaches the block's count and loses it without its last unit, the same seed gives the same range) and the figure's labels, band and width.

### What each feature adds, the added-value figure

At the scope root, `added_value_deltas.csv` holds per held-out category the gain of $\{J, S\}$, $\{J, z\}$, $\{J, J_1..J_4\}$ and $\{J, s\}$ over the $\{J\}$ cell of the same arm, the accuracy difference under each reference of Study A and the $\rho$ difference of Study B, with $S$ and $z$ on `SUMMARY_METRIC`. `added_value.csv` gives per arm and cell the mean difference with its 95% $t$ interval, the categories improved and tested and the paired $p$, and `added_value.png` draws them, one panel per arm, the mean as a dot on its interval and the categories as light dots, $\{J, s\}$ in grey as the positive control. It is the module's headline figure. A depth feature adds value where its interval clears zero, and $\{J, s\}$ shows that the comparison can see a gain when there is one. `check_module5_added_value.py` checks the table against its per-category differences and the panels.

### Every cell under one metric per group, the selected-metric figure

`typicality_ranker_comparison.png` and `pair_similarity_comparison.png` draw every cell under every registered metric, 44 rows per arm, which is the full record but too much for a page. The eight metrics fall in three groups by what they compare (`PROFILE_METRIC_GROUPS`, see the module 2 page): the overlap of proportions block by block (Jensen-Shannon divergence and distance, Hellinger, weighted Jaccard), the pattern across blocks (cosine, Pearson, Spearman) and the distance along the network (Wasserstein, the only metric reading block order). Inside a group the metrics give close values, so one metric per group carries the comparison. Each group's representative (`REPRESENTATIVE_PROFILE_METRICS`) is its best metric by mean rank inside the group over the nine results Chapter 4 ranks the metrics on, on the two v3 trees at AP 0.6 (`temp_docs/profile_metric_groups_2026-11-11/`, a one-off): the Jensen-Shannon divergence at 2.03 against 2.56 for the weighted Jaccard index, 2.61 for the distance and 2.81 for Hellinger, cosine at 1.78 against 1.94 for Pearson and 2.28 for Spearman, and Wasserstein alone. Cosine leads in GPT-2 (1.44 against 2.22) and Pearson in Qwen3-1.7B-Base (1.67 against 2.11), so that choice is close.

At the scope root, `selected_metric_comparison.csv` holds every cell of Studies A and B that carries no profile feature or carries one of the three representatives, with the columns `study`, `reference`, `model`, `metric`, `metric_group` and `value`, the value being the ranking accuracy on the held-out categories for Study A (`accuracy` of `ranker_comparison.csv`) and $\rho_W$ for Study B (`mean_category_spearman` of `pair_similarity_comparison.csv`), 19 rows per arm. `selected_metric_comparison.png` draws it, the three arms side by side, one row per cell in the thesis notation, from $\{J\}$ to $\{J, s\}$, one marker per metric (circle for the divergence, square for cosine, triangle for Wasserstein), a diamond for the cells without $S$ or $z$, the value of $\{J\}$ as a dashed line, chance as a dotted line in Study A and a rule above the two text cells, the legend under the panels. `check_module5_metric_groups.py` checks the groups, the table against its inputs and the figure.

### The block axis

$S$ and $z$ are computed on the block-aggregated layer profile, 28 bins on Qwen3 and 12 on GPT-2, in place of the flat 196 and 48. Mechanically this wraps the expert frame at the call sites in modules 2 and 5 as `layer_profile_matrices(to_block_axis(df), items)`. Note that only the profile helper sees the aggregated frame. The Jaccard helpers keep the raw frame, because Jaccard reads neuron identity and relabelling blocks would collide units across sublayers and inflate every intersection.

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

The coordinate is in FIXED ORDER, which is the one place this departs from the pair null. That function places a pair at $(\log \min(n_a, n_b),\ \log \max(n_a, n_b))$ because the two words of a pair are exchangeable and the coordinate has to be invariant to their order. A concept and a centroid are different kinds of object and are never exchanged, so the two counts enter in a fixed order and that order carries meaning. The coordinate construction is therefore a parameter of the shared `count_matched_z` helper rather than duplicated neighbourhood logic, and `check_module5_centroid_features.py` pins per-row equality against a fixed-order rebuild on a fixture where the concept outweighs its centroid, so a symmetrized implementation fails visibly.

**Why `centroid_margin` was retired.** The old `centroid_margin`, the own-centroid agreement minus the mean of the seven foreign ones, is removed and **not generated by the current code**. Its stated justification was that the count-matched construction cannot transfer to a centroid, because a centroid is an average of many profiles and has no expert count of its own. That reading does not survive scrutiny. The binding sample size in a concept-to-centroid comparison is $n_c$ alone, since the centroid pools the experts of every other member and is never the noisier side of the comparison, so the construction transfers directly once the coordinate is written in fixed order. The margin also had no counterpart anywhere else in the pipeline and added nothing in any cell. In the output tables the replacement is the `centroid_profile_<metric>_z` column, and $z_{\text{cen}}$ is the name of the construction rather than of a column. The two quantities are only moderately concordant and $z_{\text{cen}}$ should be read as a replacement rather than a refinement, which the Limitations section states with its numbers.

### The metric registry

`PROFILE_METRICS` in `modules/shared_layer_profile_measures.py` maps a metric name to a function taking a row-normalized profile matrix $P$ of shape $(n, K)$ and an optional second matrix $Q$ of shape $(m, K)$, returning the $(n, m)$ cross agreement matrix, or the square $(n, n)$ case when $Q$ is omitted, oriented so that higher always means more similar. The cross form is what lets the centroid features run through the same registry function instead of a private similarity helper.

Orientation is the entire contract. Scale is deliberately not part of it, because every consumer either standardizes the feature or ranks it, and forcing each metric onto a percentage scale would manufacture false comparability between quantities with different geometry. `check_profile_metric_registry.py` iterates the registry and asserts the orientation on a fixture, so future metrics are covered without new test code.

The registry holds eight entries, `js_distance`, `wasserstein`, `cosine`, `pearson`, `spearman`, `js_divergence`, `hellinger` and `profile_jaccard`, all defined in module 2's profile metric registry, which also groups them into the divergence, correlation, order-reading and L1 families and explains why their raw scales are not comparable. `DEFAULT_PROFILE_METRIC` names `js_distance`, defined as $100(1 - \sqrt{\mathrm{JSD}})$ exactly as section 2.1 defines it, and `ACTIVE_PROFILE_METRICS` keeps it first. Module 5's summary row and its one-metric figures report `SUMMARY_METRIC`, the Jensen-Shannon divergence `js_divergence` of Chapter 4 since 10 November 2026 and `js_distance` before. `layer_profile_matrices` takes a `metric` parameter defaulting to that name and dispatches through the registry. `ACTIVE_METRICS` in module 5 lists the metrics the grid is instantiated on and is now `ACTIVE_PROFILE_METRICS`, shared with sections 2.1 and 2.2, so one edit changes what the whole pipeline computes. Registering a metric and listing it there instantiates every cell of both studies on it and touches nothing else. The grid generates five cells per metric plus four metric-free cells, the Jaccard index, its quarters, the text and the Jaccard index with the text, the last two only where probing sentences exist, so at eight metrics each study fits 44 cells (42 without text) per reference, against 9 with one metric. The GPT-2 v3 run of module 5 at AP 0.6, every scope with the permutation tests of the partial tables, took 4 min 28 s. The count-matched null is already metric-agnostic, since it standardizes whatever agreement matrix it is handed against count-matched reference entries, so every registered metric gets its $z$ column and its $z_{\text{cen}}$ column for free.

The six planned metrics of the follow-up project are Wasserstein, cosine, Pearson, Spearman, the Jensen-Shannon divergence without the square root, and Hellinger. One metric-specific fact is worth recording now. Wasserstein is the only one of the six that reads the ORDER of the bins, every other one being permutation invariant like the Jensen-Shannon distance. It is therefore meaningful only on the block axis, where adjacent bins are adjacent depths, and would be meaningless on the flat axis, where adjacent bins are different sublayer types of the same block. The block-axis decision above is what makes the metric extension coherent at all.

The metric a figure describes is never left implicit. `_display_metric` returns `SUMMARY_METRIC` whenever that metric is active, so the three diagnostic figures that can only draw one metric at a time, the per-category dumbbell, the accuracy-by-gap curve and the pair-error grid, draw exactly the metric the cross-scope summary row reports, whether a new metric is appended to `ACTIVE_METRICS` or prepended to it. The two cells those figures compare are `HEADLINE_CELLS`, $\{J\}$ and $\{J, S\}$ since 10 November 2026 and $\{J\}$ and $\{J, S, z\}$ before. The pair-similarity figure selects its rich cell on the `(model, metric)` pair for the same reason, rather than on whichever row of the grid order carries the model name first. Each of those titles or legends names the metric as soon as more than one is active, so a single-metric figure reads as it does today and a multi-metric one is unambiguous.

### Design rationale, what the deleted diagnostics established

**Not generated by the current code.** Four raw-distribution feature sets and two subchapters were removed by the restructure, and their findings are the reason the design has the shape it has. They were measured on the pre-restructure flat-axis run of the post-trained Qwen3-1.7B at AP 0.6, whose values these pages no longer report, so only their direction is kept here, as the historical record of why the module is built this way.

**The category-identity shortcut, which is why Study A is a ranker.** A per-concept regression on the profile difference from the parent appeared to help substantially, its cross-validated $R^2$ almost tripling. Pair differencing cancels the parent term algebraically, and in the differenced form that same distribution scored near chance alone and dragged the Jaccard ranking down by several points. A distribution that fails there cannot have been carrying depth information when it appeared to help the per-concept form. The general lesson, that any per-concept feature carrying a category-level component is suspect until it is tested with whole categories held out, is why the surviving typicality study is a ranker and why the per-concept regression is kept only as a baseline column beside it.

**Resolution degrades the raw distribution rather than rescuing it, which is why no raw distribution is carried.** Raising the profile from 28 bins to 196 moved the standalone feature to chance and lowered the Jaccard combination further, significantly below Jaccard alone. Since the finer axis is the one retaining sublayer identity, where module 1 locates the categorical signal, this is evidence that a concept's typicality is not written in where its experts sit even though category membership clearly is. Seven times the parameters on the same rated pairs converted what little the coarse profile held into variance, so the grid carries the two scalar agreement measures and no raw distribution at all.

### The whitened profile, considered and declined

A defensible alternative confound control is the whitened profile, per-bin standardization of the layer distribution across words, which is standard in the representational similarity literature. It is recorded here as considered and DECLINED, with its reasoning, rather than silently omitted, and it is **not generated by the current code**.

Per-bin standardization across words targets the SHARED-BACKGROUND confound, that every word inherits the model's global density profile so two arbitrary words already agree substantially before any word-specific structure enters. The count-matched $z$ targets the COUNT confound, that a small expert set yields a spuriously spiky profile through plug-in entropy bias. Those are different nuisances, but they compete for the same slot in the design, since each is a normalization of the same agreement quantity intended to make it interpretable on its own. Carrying both would put two normalizations of one quantity in the grid, which is exactly the sprawl of near-duplicate feature sets this restructure removed. Adopting whitening instead of $z$ would additionally mean removing $z$ from module 2, a far larger change than this module. The declined option is therefore recorded rather than built.

### Ideas considered and declined on 10 November 2026

The plan behind the depth features, the within-category fit and the intervals (`docs/superpowers/plans/2026-11-10-module5-depth-added-value.md`, its pilots in `temp_docs/module5_plan_pilots_2026-11-10/`) weighed and declined the following.

| Idea | Why it was declined |
|---|---|
| An equivalence test of the gain | Its margin would be arbitrary, and the interval already bounds the gain |
| A sign-flip permutation of the category differences | The module keeps the paired $t$ of 19 October 2026, and the interval uses the same $t$ |
| A bootstrap over concepts | Categories are the unit of replication, and a concept bootstrap ignores their clustering |
| Non-negative or ridge weights for the quarters | Five predictors on about 2,000 pairs, with fold signs stable in 8 of 8 in the pilot |
| Finer bands or smooth depth weights | Quarters are the finest equal split of both models, and the per-block curve of Study D shows full resolution |
| A fixed $J$ without the last quarter | It was chosen after seeing the data, while $\{J, J_1..J_4\}$ learns its weights out of sample |
| $\{J, S, s\}$ per metric | The partial of $S$ given $J$ and $s$ answers the same question without eight more cells per reference |
| A $J$ by $S$ interaction, boosting or additive models | No sign of signal, and the linear rank model is easier to read |
| Mixed models with category intercepts | The within-category rank fit removes the category intercepts |
| Weights fitted inside each category | About 25 members and 300 rated pairs per category make them unstable, and the category dots of `added_value.png` show the spread |
| A permutation null for the gain | The partials measure the unique association without a refit, and the interval bounds the gain |
| A correction over metrics and quarters | The headline comparisons, $\{J, S\}$ on one metric and $\{J, J_1..J_4\}$, were fixed in advance, and the other rows are descriptive |
| Variance partitioning, reweighted RSA, RankSVM, Bradley-Terry | Equivalent to the partials and the nested comparison, and the ranker already is a Bradley-Terry model on feature differences |
| Depth against concreteness or WordNet depth | The pilot found the within-category coefficient near 0, so the norms enter no code |
| Pruning the eight profile metrics | Left open, the CSVs keep all eight and the figures show the Jensen-Shannon divergence |

### Output layout

```
<scope>/
├── typicality_model_design.csv          shared design frame, both studies
├── added_value_deltas.csv               per category, each cell's gain over {J}
├── added_value.csv                      mean gain, 95% interval, paired p
├── added_value.png                      the headline figure
├── selected_metric_comparison.csv       every cell, one profile metric per group
├── selected_metric_comparison.png
├── 5.1_typicality_ranking/
│   ├── study_a_design.csv               the rows Study A actually fitted
│   ├── ranker_comparison.csv            12 rows, 6 cells x 2 references
│   ├── ranker_per_category.csv
│   ├── ranker_fold_weights.csv          ranker weight per feature and fold
│   ├── ranker_partials.csv              partial Spearman per feature and control set
│   ├── axis_sensitivity.csv             whole-model scope at REFERENCE_AP only
│   ├── typicality_ranker_comparison.png
│   ├── ranker_category_dumbbell.png
│   ├── ranker_accuracy_by_gap.png
│   └── ranker_pair_errors.png
├── 5.2_pair_similarity/
│   ├── study_b_design.csv               the rows Study B actually fitted
│   ├── pair_similarity_comparison.csv   6 rows
│   ├── pair_similarity_per_category.csv
│   ├── pair_similarity_fold_weights.csv weight per feature and fold
│   ├── pair_similarity_partials.csv     partial Spearman per feature and control set, Mantel p
│   ├── pair_similarity_table.csv        the rated pairs every cell was fitted on
│   ├── axis_sensitivity.csv
│   └── pair_similarity_comparison.png
├── 5.3_concept_structure/
│   ├── word_profile_descriptors.csv      per word, the vocabulary the clusters are read in
│   ├── depth_band_by_group.csv           early/middle/late shares per category and level
│   ├── dendrograms/
│   │   ├── dendrogram_summary.csv        one row per space, the whole comparison
│   │   ├── expert_set_jaccard/           WHICH neurons, the set geometry
│   │   │   ├── dendrogram_<linkage>.png  THE deliverable, leaves coloured by category
│   │   │   ├── linkage_cophenetic.csv    the three candidate trees ranked
│   │   │   ├── merge_heights.csv         every merge, its height and its gap
│   │   │   ├── cluster_quality_by_k.csv
│   │   │   ├── cluster_composition_k8.csv
│   │   │   ├── cluster_category_enrichment_k8.csv
│   │   │   ├── word_cluster_assignment_k8.csv
│   │   │   ├── category_separation.csv   within, between, separation, permutation p
│   │   │   └── level_separation.csv      the same for abstraction level
│   │   ├── <metric>/                     WHERE along depth, one per registered metric
│   │   └── js_distance_z/                the count-matched control
│   └── linkgraphs/
│       ├── linkgraph_summary.csv         one row per graph
│       ├── expert_set_jaccard/
│       │   ├── concept_graph.png         THE deliverable
│       │   ├── concept_graph_edges.csv   edge_type and same_first_token per edge
│       │   └── concept_graph_domain_structure.csv
│       └── <metric>/                     one per profile metric, same contents
├── 5.4_depth_resolved_overlap/           whole-model scope only
│   ├── overlap_by_block.csv              per block, rho with the ratings, the rated pairs' units and the size-matched null
│   ├── shared_unit_depth.csv             per block, the shared units of each pair group
│   ├── same_token_pairs.csv              the member pairs across categories opening with one token
│   ├── same_token_summary.csv            per pair group, Jaccard, ratio and depth of the shared units
│   └── overlap_by_depth.png
├── sublayer_comparison.csv              cross-scope, written by the executor
└── sublayer_comparison.png
```

`typicality_model_design.csv` stays at the scope root because both studies are built from it. Its columns, in file order, are `concept`, `category`, `human_typicality`, `jaccard_pct`, then per registered metric the label-word pair `profile_<metric>` and `profile_<metric>_z`, then `mean_jaccard_to_members`, then per registered metric the centroid pair `centroid_profile_<metric>` and `centroid_profile_<metric>_z`. At eight registered metrics that is 4 fixed columns plus 4 per metric, 36 in all. With one registered metric it would be the nine columns `concept`, `category`, `human_typicality`, `jaccard_pct`, `profile_js_distance`, `profile_js_distance_z`, `mean_jaccard_to_members`, `centroid_profile_js_distance` and `centroid_profile_js_distance_z`. Each study additionally writes the post-dropna frame it fitted, so the identical-rows guarantee is inspectable rather than only asserted.

`typicality_model_design.csv` additionally carries the four `jaccard_band_<b>_pct` and the four `mean_jaccard_band_<b>_to_members` columns, and `text_to_label` and `mean_text_to_members` when the configuration names a probing-sentence folder.

`ranker_comparison.csv` carries a `reference` column taking `label_word` or `member_centroid` and a `metric` column naming the profile metric, and its `model` column takes the nine grid names. Together those three key the table, so no row is ambiguous about which model it describes. `pair_similarity_comparison.csv` carries the same `model` and `metric` columns without `reference`. Sublayer scopes carry the same structure one level down under `sublayers/<rank>_<sublayer>/`.

## Results

Every figure below comes from the two final trees, `research_plots_gpt2_richie_hsj_restructured_final`, written on 2 October 2026 and reproduced by the current code, and `research_plots_qwen_base_richie_hsj_FINAL`, written by the current code on 8 October 2026, at whole-model scope unless a sublayer is named. Qwen3 means Qwen3-1.7B-Base, the pre-trained checkpoint, throughout, and its comparison with the post-trained Qwen3-1.7B is kept in `fixes.md`, the only page that reports post-trained values. At AP 0.5 Studies A and B fit all 198 members and the 2,418 rated within-category pairs over 8 held-out categories.

Before that sweep a smoke run, GPT-2 at AP 0.5 in `research_plots_gpt2_richie_hsj_smoke_m9`, checked that the restructure changed the data path rather than the numbers where it promised to. Every `sublayers/` scope of the former modules 3, 4 and 5, now sections of module 2, reproduces the pre-change tree byte for byte on 60 of 76 files, with the 16 exceptions falling into three declared classes, eight files of the former module 4 absent because the modules they read were disabled in that configuration, four `human_similarity_validation.csv` files that gained the `coefficient` and `test` schema columns since the baseline was written, and four `correlation_summary.csv` files, of which exactly one carries a real numeric difference of one unit in the last place on `pearson_p` while the other three differ only by the two rows the disabled modules did not write. Each of the 16 still passes full value equality on every column and row the baseline wrote, at a relative tolerance of $10^{-13}$. Module 5's self-computed default-metric columns equal section 2.1's own CSV on the same run at a maximum absolute difference of 0.0 over all 197 concepts and all three shared columns. The Jaccard cells match the pre-restructure run exactly, 2,391 pairs and accuracy 0.614806 in Study A and mean per-category $\rho$ 0.3608 in Study B, which is the expected result since Jaccard does not depend on the profile axis.

### Study A, pairwise typicality ranking

AP 0.5, whole-model scope, GPT-2 / Qwen3 in each cell. `delta` is the mean per-category accuracy difference against the `jaccard` cell of the same reference, `cats` counts the categories that improved out of 8, and $p$ is the paired $t$ over those 8 categories.

| Reference | Cell | accuracy | accuracy_clear | regression | delta | cats | $p$ |
|---|---|---|---|---|---|---|---|
| label_word | jaccard | 60.9 / 67.0% | 63.9 / 71.4% | 60.9 / 67.0% | | | |
| label_word | profile | 42.8 / 52.8% | 40.1 / 53.0% | 47.2 / 51.9% | $-17.2$ / $-15.7$ pp | 0 / 1 | 0.031 / 0.009 |
| label_word | profile_z | 45.1 / 47.9% | 43.1 / 45.7% | 44.6 / 49.0% | $-15.3$ / $-18.7$ pp | 0 / 1 | 0.046 / 0.023 |
| label_word | jaccard_profile | 61.4 / 67.6% | 64.2 / 72.5% | 60.3 / 68.1% | +0.8 / +0.8 pp | 4 / 5 | 0.734 / 0.208 |
| label_word | jaccard_profile_z | 59.9 / 66.5% | 62.1 / 70.9% | 58.1 / 67.0% | $-0.5$ / $-0.5$ pp | 4 / 1 | 0.624 / 0.106 |
| label_word | jaccard_profile_both | 61.9 / 66.7% | 65.2 / 71.2% | 62.4 / 68.2% | +1.3 / +0.0 pp | 5 / 4 | 0.452 / 0.984 |
| member_centroid | jaccard | 59.3 / 65.5% | 62.4 / 70.0% | 59.3 / 65.5% | | | |
| member_centroid | profile | 47.6 / 49.7% | 47.8 / 49.8% | 47.1 / 48.2% | $-10.5$ / $-15.4$ pp | 1 / 0 | 0.074 / 0.001 |
| member_centroid | profile_z | 54.6 / 46.6% | 56.1 / 45.5% | 52.9 / 49.5% | $-5.0$ / $-19.5$ pp | 2 / 0 | 0.070 / 0.000 |
| member_centroid | jaccard_profile | 59.6 / 66.0% | 62.0 / 70.2% | 58.6 / 64.7% | $-0.8$ / +0.2 pp | 5 / 4 | 0.732 / 0.854 |
| member_centroid | jaccard_profile_z | 58.7 / 64.8% | 61.6 / 68.9% | 57.9 / 64.6% | $-0.7$ / $-1.0$ pp | 1 / 4 | 0.264 / 0.211 |
| member_centroid | jaccard_profile_both | 64.1 / 67.7% | 68.2 / 72.3% | 62.9 / 66.6% | +4.6 / +1.5 pp | 6 / 4 | 0.053 / 0.583 |

No profile cell reaches significance as an ADDITION to Jaccard under either reference in either model. The closest is GPT-2's fullest cell on the centroid reference, $+4.6$ points with 6 of 8 categories improved at $p = 0.053$, and Qwen3's largest gain, $+1.5$ points on the same reference, sits at $p = 0.58$. As STANDALONE predictors the profile features sit near the 50 percent chance line, between 42.8 and 52.8 percent on the label-word reference and between 46.6 and 54.6 on the centroid reference, and every standalone row sits 5 to 20 points below its Jaccard baseline, which is what the nominal significance in most of those rows measures, a DEFICIT against Jaccard rather than a gain. Restricting to clear pairs adds between 2.2 and 4.8 points on every Jaccard-bearing cell. The ranker ties its regression baseline exactly on the two Jaccard cells, as the construction requires, and on the multi-feature cells neither is consistently ahead, the ranker leading on 8 of the 12 and the regression on 4, with every gap below 2 points.

### Study B, human pair similarity

Same runs, 2,418 rated pairs over 8 categories, mean noise ceiling 0.8912, GPT-2 / Qwen3 in each cell.

| Cell | mean_category_spearman | pooled | rho_over_ceiling | delta vs jaccard | partial $F$ |
|---|---|---|---|---|---|
| jaccard | 0.3597 / 0.5318 | 0.0714 / 0.5559 | 0.4035 / 0.5967 | | |
| profile | $-0.0926$ / 0.1109 | $-0.5077$ / $-0.0555$ | $-0.1039$ / 0.1244 | $-0.452$ / $-0.421$ |  |
| profile_z | $-0.0338$ / 0.1168 | $-0.4596$ / 0.0118 | $-0.0379$ / 0.1310 | $-0.394$ / $-0.415$ |  |
| jaccard_profile | 0.3641 / 0.5442 | 0.2007 / 0.5544 | 0.4086 / 0.6106 | +0.004 / +0.012 | 97.5 / 7.0 vs jaccard, $p = 1.4\times10^{-22}$ / $8.4\times10^{-3}$ |
| jaccard_profile_z | 0.3479 / 0.5202 | 0.1090 / 0.5474 | 0.3903 / 0.5837 | $-0.012$ / $-0.012$ | 32.1 / 11.08 vs jaccard, $p = 1.6\times10^{-8}$ / $8.8\times10^{-4}$ |
| jaccard_profile_both | 0.3801 / 0.5769 | 0.2732 / 0.5837 | 0.4265 / 0.6473 | +0.020 / +0.045 | 68.3 / 103.8 vs jaccard_profile, $p = 2.3\times10^{-16}$ / $6.7\times10^{-24}$ |

**Earlier definition.** `mean_category_spearman`, `rho_over_ceiling` and the mean noise ceiling above are unweighted means over the categories, and the `pooled` and partial $F$ columns are **not generated by the current code**. On the nopad trees at AP 0.6 with `js_distance`, Qwen3 / GPT-2, the pair-weighted $\rho_W$ reads 0.5266 / 0.3513 for the Jaccard cell and 0.5263 / 0.3368 for `jaccard_profile_both`, over a weighted ceiling of 0.884, and the paired $t$ over the held-out categories finds no cell that adds significantly to the Jaccard cell it extends, p between 0.15 and 1.0, while every Jaccard-bearing cell beats the profile cell it extends, p 0.0002 to 0.0035. The gain of the profile features read below does not survive the weighting there.

**Earlier definition.** Every Study B value of the reference trees, here and in the three sections that follow, comes from the pooled fit module 5 used until 10 November 2026, weights fitted on the raw values pooled over the training categories, and so does Study A's `regression` column. Fitted within categories (*Study B, human pair similarity by regression*), the Jaccard cell keeps its value, the profile cell alone turns from inverted to weakly positive, and every cell adding a profile feature to $J$ gains 0.016 to 0.018 on the GPT-2 v3 tree at AP 0.6, while the ranker accuracies of Study A do not move, see `fixes.md` and *What depth adds to the Jaccard index* below.

The Jaccard cell alone reaches 40 percent of the human noise ceiling in GPT-2 and 60 percent in Qwen3 with whole categories held out, the same per-category agreement module 2's section 2.3 reports, since a single monotone feature changes no rank within a category. Adding both profile features lifts the mean per-category correlation from 0.360 to 0.380 in GPT-2 and from 0.532 to 0.577 in Qwen3, the largest gain of the study, and the pooled figure from 0.071 to 0.273 and from 0.556 to 0.584. Every nested partial $F$ (**Earlier definition**) clears significance, including Qwen3's `jaccard_profile_z`, whose mean per-category correlation falls by 0.012 at $F = 11.1$. Significance is expected on 2,418 pairs, which is exactly why the honest reading is the per-category mean and its ceiling-normalized column rather than the $F$ statistic. Alone the profile features are useless or inverted on this target, between $-0.09$ and $+0.12$.

Every figure in this table sits below the 0.643 that the TF-IDF similarity of the probing sentences reaches on the same pairs, and the Jaccard cell's agreement falls to 0.10 in GPT-2 and 0.15 in Qwen3 once that text is partialled out (module 2, section 2.3). The profile gain is therefore a gain over a measure that mostly inherits the sentences, and it is not evidence of alignment beyond them.

### Metric comparison

The eight-metric extension refits both studies on every registered metric. AP 0.5, whole-model scope, showing the fullest cell `jaccard_profile_both` since that is the comparison the module exists to make, GPT-2 / Qwen3 in each cell. The `jaccard` cell carries no profile metric and is one row for all of them, at 0.6088 / 0.6696 in Study A under the label word, 0.5926 / 0.6551 under the member centroid, and 0.3597 / 0.5318 in Study B. The Study B values of this section are unweighted means over the categories (**Earlier definition**, see *Study B, human pair similarity*).

| Metric | Study A, label word | Study A, centroid | Study B, mean per-category $\rho$ |
|---|---|---|---|
| `js_distance` | 0.6191 / 0.6675 | 0.6410 / 0.6770 | 0.3801 / 0.5769 |
| `wasserstein` | 0.6158 / 0.6617 | 0.5831 / 0.6431 | 0.3545 / 0.5521 |
| `cosine` | 0.6208 / 0.6696 | 0.6261 / 0.6778 | 0.3675 / 0.5787 |
| `pearson` | 0.6030 / 0.6687 | 0.5893 / 0.6497 | 0.3605 / 0.5385 |
| `spearman` | 0.5935 / 0.6543 | 0.5910 / 0.6427 | 0.3747 / 0.5245 |
| `js_divergence` | 0.6166 / 0.6754 | 0.6402 / 0.6493 | 0.3841 / 0.5718 |
| `hellinger` | 0.6187 / 0.6675 | 0.6410 / 0.6758 | 0.3804 / 0.5772 |
| `profile_jaccard` | 0.6208 / 0.6638 | 0.6253 / 0.6754 | 0.3730 / 0.5795 |

Four readings, and none of them changes a conclusion of the module.

The default metric is never beaten by more than 0.008, the largest margin being `js_divergence` on Qwen3's label-word arm, 0.6754 against 0.6675, itself at $p = 0.50$ against its own Jaccard baseline. Two of the 32 paired tests against the Jaccard baseline clear 0.05, GPT-2's `hellinger` on the centroid arm, a gain at $p = 0.046$, and Qwen3's `spearman` on the label-word arm, a loss at $p = 0.033$, which is about what chance alone predicts. So the flat reading of the restructure survives: the profile features add little to typicality ranking, and which profile measure is used is not what is holding that down.

The three divergence measures are close to interchangeable, agreeing within 0.008 everywhere except Qwen3's centroid arm, where `js_divergence` falls 2.8 points below the other two. `js_distance` and `js_divergence` are monotone transforms of one another, so they can differ at all only because these are linear models.

The centred measures are the weakest where the target is graded. `spearman` is last on Study A's label-word arm in both models and on its centroid arm in Qwen3, and `spearman` and `pearson` are the two lowest on Study B in Qwen3, 0.5245 and 0.5385, the same pattern module 2's section 2.1 found for the concept-to-label pairing, where removing the global density baseline removes much of what the two words share.

`wasserstein` is weak throughout without being uniformly last. It is last on Study A's centroid arm in GPT-2 and second to last in Qwen3, and on Study B in GPT-2 it is the only metric below the Jaccard baseline, 0.3545 against 0.3597. Read with module 2's section 2.2, where it was also among the weakest on category alignment and human agreement, charging transport distance between blocks buys nothing reliable on this data.

### Axis sensitivity

The whole-model scope at AP 0.5 refits `jaccard_profile` on the flat layer axis, 48 layers on GPT-2 and 196 on Qwen3, for every registered metric, writing `axis_sensitivity.csv`. For the default metric the flat axis lowers Study A by 0.8 points in GPT-2 and 1.2 in Qwen3, and raises Study B's mean per-category $\rho$ from 0.364 to 0.379 and from 0.544 to 0.562. Across the eight metrics the axis moves Study A by at most 2.2 points and Study B by at most 0.021. These moves are far inside the per-category spread, a standard deviation of about 13 to 14 points across categories in Study A and of 0.12 to 0.26 in Study B, which is what the sensitivity line exists to show.

### Studies A and B across AP thresholds

Whole-model scope, default metric, the fullest cell `jaccard_profile_both` beside the `jaccard` cell. Where the two studies differ, pairs and categories are given as Study A / Study B. The two Study B columns are unweighted means over the categories (**Earlier definition**), and on the nopad trees at AP 0.6 the pair-weighted means of the two cells already tie in Qwen3 (0.5263 against 0.5266) and favour Jaccard in GPT-2 (0.3368 against 0.3513).

| Model | AP | members | pairs | categories | Study A, Jaccard | Study A, full cell | Study A, centroid Jaccard | Study A, centroid full cell | Study B, Jaccard | Study B, full cell |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-2 | 0.5 | 198 | 2,418 | 8 | 60.9% | 61.9% | 59.3% | 64.1% | 0.360 | 0.380 |
| GPT-2 | 0.6 | 198 | 2,418 | 8 | 61.3% | 62.8% | 59.8% | 64.2% | 0.339 | 0.335 |
| GPT-2 | 0.7 | 198 | 2,418 | 8 | 58.4% | 56.9% | 56.9% | 60.0% | 0.312 | 0.239 |
| GPT-2 | 0.8 | 69 | 762 | 3 | 58.4% | 64.2% | 60.9% | 57.1% | 0.358 | 0.169 |
| GPT-2 | 0.9 | | | | no study, no category label keeps an expert | | | | | |
| Qwen3 | 0.5 | 198 | 2,418 | 8 | 67.0% | 66.7% | 65.5% | 67.7% | 0.532 | 0.577 |
| Qwen3 | 0.6 | 198 | 2,418 | 8 | 67.1% | 66.0% | 64.8% | 66.0% | 0.503 | 0.542 |
| Qwen3 | 0.7 | 198 | 2,418 | 8 | 68.5% | 64.4% | 65.2% | 67.8% | 0.532 | 0.483 |
| Qwen3 | 0.8 | 170 | 2,040 / 2,418 | 7 / 8 | 69.9% | 67.0% | 59.5% | 59.8% | 0.529 | 0.398 |
| Qwen3 | 0.9 | 50 | 616 / 1,348 | 2 / 5 | 43.0% | 46.6% | 34.1% | 41.4% | 0.248 | 0.263 |

No threshold turns the profile features into a reliable addition. On the label-word arm of Study A the fullest cell's paired test stays at $p \ge 0.24$ in both models. On the centroid arm one cell clears 0.05, Qwen3 at AP 0.7, $+2.7$ points with 7 of 8 categories improved at $p = 0.007$, which does not survive a Bonferroni correction over the 16 tests of this table and is not matched at the neighbouring thresholds, $+1.0$ points at AP 0.6 and $-1.6$ at AP 0.8. The Jaccard cell of Study A holds between 67.0 and 69.9 percent through AP 0.8 in Qwen3.

The profile gain of Study B is confined to the lenient end. The fullest cell beats the Jaccard cell at AP 0.5 in both models and at AP 0.6 in Qwen3, and from AP 0.7 adding the profile features lowers the mean per-category correlation, by 0.073 in GPT-2 and 0.049 in Qwen3 at AP 0.7 and by 0.131 in Qwen3 at AP 0.8. The Qwen3 gain of 0.045 at AP 0.5 is therefore a property of the lenient thresholds rather than of the profile features in general.

GPT-2 at AP 0.8 rests on 3 categories and Qwen3 at AP 0.9 on 5 in Study B and 2 in Study A, the latter with an almost constant Jaccard feature (see Limitations), so neither row is a result. At AP 0.8 Qwen3's Study A loses `professions`, whose label holds a single expert there, so the label's count-matched $z$ is undefined and the category's 28 members leave the design frame. GPT-2 has no Study A or B at AP 0.9, since no category label keeps an expert there and the design frame requires one.

### What depth adds to the Jaccard index

The first tree written with the within-category fit is GPT-2 on the v3 sentences, `research_plots_gpt2_richie_hsj_v3`, at AP 0.6, whole model, $S$ the Jensen-Shannon divergence. Module 5 ran with it on the Qwen3-1.7B-Base v3 tree and on both v3 context-only trees on 11 November 2026, and the nopad trees keep the pooled values until it is rerun on them. Every gain below is the mean over the 8 held-out categories of a cell's accuracy or $\rho_k$ minus that of the Jaccard cell of its own arm, with its 95% $t$ interval, as `added_value.csv` writes it and `added_value.png` draws it. The Jaccard cells reach 58.8% under the category name and 58.5% under the other members in Study A, and $\rho_W = 0.342$ in Study B.

| Cell | Study A, category name | Study A, other members | Study B |
|---|---|---|---|
| $\{J, S\}$ | $-1.0$ pp $[-5.9, +3.9]$ | $-0.8$ pp $[-4.2, +2.6]$ | $+0.004$ $[-0.018, +0.026]$ |
| $\{J, z\}$ | $-0.7$ pp $[-6.4, +5.1]$ | $-0.0$ pp $[-0.9, +0.9]$ | $-0.003$ $[-0.012, +0.006]$ |
| $\{J, J_1, \ldots, J_4\}$ | $-3.1$ pp $[-9.7, +3.5]$ | $-1.8$ pp $[-3.9, +0.3]$ | $+0.007$ $[-0.009, +0.024]$ |
| $\{J, s\}$, positive control | $+9.3$ pp $[-1.0, +19.5]$, $p = 0.069$ | $+14.9$ pp $[+2.2, +27.5]$, $p = 0.027$ | $+0.300$ $[+0.092, +0.508]$, $p = 0.011$ |

No depth feature adds to the Jaccard index in GPT-2, and every interval of the three depth rows contains 0, while the text similarity of the probing sentences adds on the two arms where its interval clears 0, so the comparison is able to show a gain when one exists. The partial Spearman coefficients of `pair_similarity_partials.csv` locate what little the quarters carry. Given $J$ the last two quarters correlate with the ratings at $+0.060$ ($p = 0.046$) and $+0.089$ ($p = 0.004$), the first two at $-0.037$ and $-0.023$, and given $J$ and $s$ no quarter exceeds $|\rho| = 0.036$ ($p > 0.22$), against $\rho = 0.342$ for $J$ alone, 0.140 given the text, and 0.513 for the text given $J$. In every fold of $\{J, J_1, \ldots, J_4\}$ the four quarters take positive weights and $J$ a negative one, the weights of a collinear decomposition, $J$ being the union-weighted mean of the $J_b$, so the fold weights are not read one by one.

`selected_metric_comparison.png` shows that the choice of metric does not change this reading. Under the three representatives a cell holding $J$ moves by at most 0.014 in Study B ($\{J, S, z\}$, 0.338 to 0.352) and by at most 4.7 points in Study A (the other members, $\{J, S, z\}$, 58.9% to 63.6%), while $S$ alone spreads more, 40.1% to 46.1% under the category name and 42.1% to 51.1% under the other members, every value of it below chance but the divergence under the other members.

Study D resolves $J$ block by block (`5.4_depth_resolved_overlap/overlap_by_block.csv`). No single block reaches the agreement of the whole set. The per-block $\rho_W$ is 0.25 to 0.26 over blocks 0 to 2, falls to 0.178 in block 4 and rises to 0.31 to 0.33 over blocks 9 to 11, and given the text the blocks run from 0.080 to 0.143 against 0.140 for the whole set. Every block carries part of the agreement, the last quarter somewhat more, and no band of depth holds it alone.

The whole set sits above every block mostly because it is larger. A block holds 4.6% (block 6) to 16% (block 0) of the expert units, and inside one block 10% to 46% of the rated pairs share no unit at all, against 0.5% over the whole model, so the block's Jaccard index ties many pairs at 0 and carries more sampling noise, which lowers any rank correlation. A one-off check (`temp_docs/depth_gap_2026-11-11/`, whose random sets Study D draws itself since 12 November 2026) draws 200 random sets of units from the whole model with as many expert units as each block. On average they reach $\rho_W$ 0.26 to 0.31, already 0.03 to 0.08 below the whole set, and pooling $k$ random blocks raises $\rho_W$ from 0.266 at one block to 0.304 at two, 0.324 at four and 0.342 at all twelve, within 0.02 of random sets of the same size at every $k$. Against random sets of its own size in the pipeline's `overlap_by_block.csv`, block 9 lies above the 95% range (0.331 against 0.256 to 0.318) and blocks 0 and 4 below it by more than 0.01 (block 4 0.178 against 0.227 to 0.304), blocks 1 to 3 at its lower bound and the rest inside it, which is the depth signal the curve holds once size is accounted for. Given the text no block leaves the range. The same check on Qwen3-1.7B-Base, whose members hold about eight times more expert units, reads the other way: random sets of a block's size reach on average 0.36 to 0.51 against 0.538 for the whole model, lower only in blocks 12 to 19, which hold under 2% of the units each, so size explains little, while blocks 6 to 9 reach 0.548 to 0.569, above the range and the whole model, and blocks 0 to 4, block 13 and six of the seven last blocks fall below the range, so there the curve's shape is a depth effect. Given the text blocks 5 and 6 lie above the range and blocks 4 to 7 above the whole model (0.221 to 0.257 against 0.217). Chapter 4 reads these values in *Agreement Along Depth*.

Ten pairs of members from different categories open with the same GPT-2 token, car and carpenter, dress and dresser, pajamas, pheasant and pomegranate, rooster and rowing, tangerine, trolley and tuxedo, and vase and vulture (`same_token_pairs.csv`). They share 16.7% of their units on average, 30.1 times the 0.55% of the 17,074 other pairs across categories and 4.6 times the 3.67% of the 2,418 rated pairs within categories, and every one of them sits at or above the 99.9th percentile of the Jaccard index among pairs across categories. Pairs sharing only their first letter, 1,446 of them, share 1.18 times the other pairs. A one-off analysis with finer classes, word-bootstrap intervals and permutation p (`temp_docs/token_similarity_2026-11-11/`, both models, AP 0.5 to 0.7, intact and context-only) splits that group: first tokens of which one opens the other share 1.78 times the unrelated pairs, pairs with no piece in common but the same initial letters 1.14 times (Qwen3-1.7B-Base 1.38), rising with the number of common letters, and the same holds for words that are each one token, while a common ending adds nothing and the context-only trees remove every class. The units the same-token pairs share sit at a mean depth of 0.553 of the model, 36.3% in the last quarter, against 0.439 and 30.6% for the other pairs across categories and 0.576 and 44.8% within categories, so in GPT-2 the written form enters the expert sets about as deep as category membership does. Study C's link graphs flag these pairs in `same_first_token`.

### Study C, concept structure

This section is reported from the whole-model scope of the two final trees at AP 0.5, `research_plots_gpt2_richie_hsj_restructured_final` and `research_plots_qwen_base_richie_hsj_FINAL`, covering all 206 words, every one of which carries a usable profile. GPT-2 holds 217,149 expert rows over 12 blocks, Qwen3-1.7B-Base 1,573,600 over 28.

**The headline: what a word is described BY decides what the tree finds.** Ten spaces are clustered with identical machinery over the same 206 words. One describes a word by which neurons it recruits, the rest by where along depth it puts them. The trees they produce are not in the same league:

| space | cophenetic | ARI vs category | $\eta^2$ depth | category silhouette |
|---|---|---|---|---|
| **expert_set_jaccard** (GPT-2) | **0.914** | **0.682** | 0.212 | **+0.036** |
| `js_distance`, the default profile metric (GPT-2) | 0.677 | 0.115 | 0.693 | $-0.062$ |
| **expert_set_jaccard** (Qwen3) | **0.920** | **0.772** | 0.364 | **+0.047** |
| `js_distance`, the default profile metric (Qwen3) | 0.679 | 0.194 | 0.459 | +0.015 |

The expert-set tree is the most faithful tree of the ten and beats every layer-profile tree on category agreement, the best of them, `js_divergence` and `profile_jaccard` tied at ARI 0.129 on GPT-2 and `hellinger` at 0.204 on Qwen3, by a factor of four to five. On GPT-2 it is the only space in which the category partition has a POSITIVE silhouette, every profile space being negative there, meaning a word sits no closer to its own category than to the rest. On Qwen3 the divergence family and `profile_jaccard` reach small positive silhouettes too, +0.009 to +0.018, against +0.047 for the expert sets.

What the expert-set tree recovers at the $k = 8$ cut on GPT-2 is close to the dataset's own answer key: a 28-word cluster holding 27 professions, a 20-word one holding 19 sports, 28 words holding 26 clothing items and 19 holding 17 pieces of furniture, and its largest confusion is fruit with vegetables, merged into one 35-word cluster, while 8 sports join the 22 vehicles. That is the mistake a person would make. It never saw a category label. Qwen3's tree makes the same two merges, fruit with vegetables in a 42-word cluster and 8 sports with 21 vehicles, and otherwise puts 28 clothing items, 28 professions and 20 sports in pure clusters and 30 birds in a cluster of 35, splitting furniture into two pure clusters of 12 and 7.

The reading is not that one geometry is good and the other broken. Both carry real structure and they carry DIFFERENT structure, which the two $\eta^2$ columns show as clearly as the ARI does. On GPT-2 the default profile tree explains 0.69 of the variance in depth and little of the categories, and the expert-set tree does the reverse at 0.21 and 0.68. Qwen3 is less extreme, 0.46 against 0.36 on depth and 0.19 against 0.77 on the categories, so its profile trees carry somewhat more category information and somewhat less depth.

**The layer-profile trees divide the words by depth more than by category.** Average linkage wins in both models (cophenetic $r$ 0.677 on GPT-2 against 0.606 for complete and 0.490 for Ward, and 0.679 on Qwen3 against 0.594 and 0.533). At the $k = 8$ cut:

| | GPT-2 | Qwen3 |
|---|---|---|
| adjusted Rand vs the categories | 0.115 | 0.194 |
| adjusted Rand vs the abstraction level | $-0.003$ | 0.010 |
| $\eta^2$, profile entropy | 0.749 | 0.684 |
| $\eta^2$, depth centre of mass | 0.693 | 0.459 |
| $\eta^2$, peak bin | 0.655 | 0.218 |
| $\eta^2$, log expert count | 0.529 | 0.504 |

The clusters explain most of the variance in how widely a word's expert mass spreads along depth, and on GPT-2 where it sits, and much less of the category membership. Pushing $k$ to 20 moves the adjusted Rand only to 0.131 on GPT-2 and 0.204 on Qwen3, so the shortfall is not a matter of cutting in the wrong place. Removing the count gradient through `js_distance_z` drops GPT-2's $\eta^2$ on log expert count from 0.529 to 0.281 while the centre-of-mass $\eta^2$ rises to 0.717, so the depth structure there is not the size confound wearing a different name. On Qwen3 the same control drops the count $\eta^2$ from 0.504 to 0.111 and the centre-of-mass $\eta^2$ from 0.459 to 0.344, so part of Qwen3's depth grouping travels with expert count.

**Where the divisions actually are, and why $k = 8$ is a convention.** Reading `merge_heights.csv` on GPT-2, excluding the root merge, the four largest gaps in the swept range of 2 to 20 clusters are:

| clusters remaining | height | gap to previous | sizes joined |
|---|---|---|---|
| 4 | 0.2136 | 0.0228 | 1 and 50 |
| 2 | 0.2322 | 0.0123 | 141 and 61 |
| 6 | 0.1828 | 0.0112 | 8 and 90 |
| 5 | 0.1907 | 0.0079 | 43 and 98 |

The largest gap of all leaves four clusters, but it joins a single word (`rowing`) to a 50-word group, so it is an outlier merge rather than a structural boundary. The largest gap that divides two real groups leaves **two** clusters, at height 0.2322, joining 141 words to 61. From seven clusters on every gap falls under 0.008, and the merge that leaves eight clusters costs 0.003626, which is essentially free.

That last number should be read carefully, because it qualifies every $k = 8$ figure in this study. Eight is chosen so the cut is directly comparable to the eight categories of the dataset, not because the tree has a division there. Qwen3 behaves the same way, its $k = 8$ merge costing 0.002685 against a largest in-range gap of 0.0329, which joins a 3-word group to 124 words. The composition tables at $k = 8$ are therefore a fair description of what eight groups of these words look like, and not evidence that eight groups exist.

`word_profile_descriptors.csv` says what the profiles are made of. On GPT-2, of 206 words, 109 peak in the first block and 67 in the last, leaving 30 anywhere else, and the middle band holds 1.2 percent of reliable peaks, so nearly every word puts its peak at one end of the model. Qwen3 spreads its peaks along depth, 20 words in the first block and 18 in the last, with 63 percent of reliable peaks in the last fifth, 21 percent in the first and 16 percent in the middle band.

**Which groups the depth axis does divide.** GPT-2, share of words peaking in the last fifth of the model, over the 170 words with a reliable peak:

| Group | late share | Group | late share |
|---|---|---|---|
| level-1 label words | 66.7% | clothing | 30.8% |
| sports | 64.0% | birds | 28.0% |
| professions | 60.0% | furniture | 13.3% |
| vehicles | 44.4% | fruit | 5.6% |
| all words | 34.1% | vegetables | 0.0% |

The permutation test of category separation gives the same ordering (**Earlier definition**, one-sided with 2,000 shuffles, so the floor is 0.0005 where the current two-sided test floors at 0.0001). Five of eight GPT-2 categories are reliably tighter than chance, professions at 0.0643 ($p = 0.0005$), vehicles 0.0426 (0.0005), sports 0.0330 (0.0005), vegetables 0.0231 (0.0075) and fruit 0.0198 (0.0135), while birds (0.12), clothing (0.13) and furniture (0.30) are indistinguishable from an arbitrary set of 20 to 30 words. The same five clear the count-matched space, which is the control that matters, with birds joining them at $p = 0.0495$. Qwen3 separates categories more evenly, all eight clearing $p = 0.0005$ at more than double the mean separation (0.0594 against 0.0257) and a category silhouette of +0.015 against GPT-2's $-0.062$. The two models share their endpoints, professions strongest and furniture weakest in both, and differ in the middle.

The depth ordering of the categories does **not** carry across architectures. Qwen3 is late-heavy overall (63 percent of all words peak in its last fifth) and its per-category late shares, over all words, run birds 90.0 percent, fruit 81.0, vegetables 80.0, sports 75.0, vehicles 59.1, clothing 58.6, furniture 40.0 and professions 28.6. Professions sits near the late end in GPT-2 and at the other end in Qwen3, and fruit and vegetables swap ends the same way. Absolute depth position is an architecture property, the fact that categories differ in depth at all is shared.

**The abstraction level shows up as coherence, not location.** The 8 category-label words sit closer to each other than a random 8 of the 206 do, separation 0.0319 at $p = 0.028$ on GPT-2 and 0.0424 at $p = 0.011$ on Qwen3. In both models the effect weakens under count matching, to $p = 0.061$ on GPT-2 and 0.087 on Qwen3 for the label group, so it is not separable from expert-set size. These p are one-sided (**Earlier definition**). In the expert-set space of the nopad trees at AP 0.6 the two-sided test gives the level partition an `ALL GROUPS` p of 0.216 (Qwen3) and 0.087 (GPT-2), where the one-sided p read 0.904 and 0.981 because the label words there sit slightly farther apart than chance. This is a different statement from module 1's depth test and does not conflict with it. Module 1 asks where the levels sit on average and finds GPT-2 labels deeper than their members. This asks whether the label words share a profile SHAPE. On Qwen3 five of the eight labels peak in the middle band, which holds 19 percent of all words, and both labels with a reliable peak peak there, while the centre of mass does not separate the levels (AUC 0.387, $p = 0.29$). On GPT-2 the direction is the familiar one, 6 of 8 labels peak late against 34 percent of members, and the centre of mass separates the levels at AUC 0.742 with $p = 0.018$, reproducing module 1's finding on the block axis.

**The same statistic on both geometries.** A sceptic could object that the gap above is an artefact of comparing two different statistics, Jaccard on one side and Jensen-Shannon on the other. `profile_jaccard` closes that hole. It is the weighted (Ruzicka) Jaccard,

$$ J(p, q) = \frac{\sum_k \min(p_k, q_k)}{\sum_k \max(p_k, q_k)}, $$

which on set indicator vectors reduces exactly to the ordinary $|A \cap B| / |A \cup B|$, verified to machine precision. Running it on layer profiles asks the Jaccard question of the depth distribution instead of the neuron set, with the functional form held fixed.

It lands at ARI 0.129 on GPT-2 and 0.086 on Qwen3, within the range of the other profile metrics and nowhere near the 0.682 and 0.772 the same index reaches on expert sets. The gap is therefore a property of the representation and not of the measure.

It is also the registry's only L1 member. Profiles are normalized, so $\sum_k \max = 2 - \sum_k \min$ and $\sum_k \min(p,q) = 1 - \mathrm{TV}$, giving $J = (1 - \mathrm{TV})/(1 + \mathrm{TV})$, a strictly decreasing function of total variation distance (verified numerically at Spearman exactly -1). Being a monotone transform of TV, it ranks every pair as TV does, so rank-based statistics read the two identically while average linkage and the linear consumers do not, since a mean of transformed distances is not the transform of a mean.

**The metric sweep, and why the default holds.** Each of the eight registered metrics gets its own tree. On GPT-2 the divergence family and `profile_jaccard` lead on category agreement, `js_divergence` and `profile_jaccard` at ARI 0.129 and `js_distance` and `hellinger` at 0.115, against `cosine` 0.079, `wasserstein` 0.047 and the two signed metrics at essentially nothing, `pearson` 0.003 and `spearman` 0.001. Two things in that table are worth keeping.

First, `pearson` and `spearman` build the most FAITHFUL profile trees on GPT-2, cophenetic 0.856 and 0.888 against 0.677 for the default, and the least USEFUL ones, with the lowest category agreement and the lowest $\eta^2$ on depth (0.096 and 0.111 against 0.693). Cophenetic correlation measures how well a tree reproduces the distances it was built from, not whether those distances carry anything, so it must never be read as a quality score on its own. Second, `wasserstein` has by far the highest $\eta^2$ on the depth centre of mass, 0.884 on GPT-2 and 0.887 on Qwen3, which is exactly what it should do, since it is the only metric in the registry that reads bin ORDER and therefore the purest depth measure available.

The ranking is architecture-dependent and the default is not always the winner. On Qwen3 `hellinger` leads at ARI 0.204 with `js_distance` at 0.194, while `js_divergence` falls to 0.092 and the signed metrics recover to 0.121 (`spearman`) and 0.105 (`pearson`). No metric changes the conclusion of this study, since all eight leave the profile trees far below the expert-set tree on category agreement, but a claim that names one metric should say which model it was measured on.

**The link graphs are the sharpest result of the study.** The same 206 words, at the same threshold, in the same model, drawn twice:

| | GPT-2 Jaccard | GPT-2 profile | Qwen3 Jaccard | Qwen3 profile |
|---|---|---|---|---|
| edges retained | 480 | 453 | 456 | 448 |
| within-category | 365 | 133 | 377 | 198 |
| label to member | 60 | 25 | 49 | 28 |
| cross-category | 55 | 295 | 30 | 222 |
| within-category share, categorized edges | 86.9% | 31.1% | 92.6% | 47.1% |
| the same under label permutation | 12.4% | 12.5% | 12.4% | 12.4% |
| ratio observed over null | **7.02** | **2.49** | **7.50** | **3.79** |
| $p$ | 0.0005 | 0.0005 | 0.0005 | 0.0005 |

The $p$ row is **Earlier definition**, one-sided with 2,000 shuffles. The current two-sided test reaches its 0.0001 floor over all edges for every graph of both nopad trees at AP 0.6.

The profile column above is `js_distance`, and the choice barely matters: across all eight metrics the profile graphs span 25.3 to 31.7 percent within-category on GPT-2 and 34.2 to 47.1 on Qwen3, so every one of them sits far below the expert-set graph. `js_distance` and `js_divergence` produce identical graphs in both models, which is correct rather than a bug: they are monotone transforms of one another, so they rank each word's neighbours identically and a graph built from the top three neighbours cannot tell them apart. `hellinger` matches them exactly on GPT-2 and differs by one edge on Qwen3.

Both similarities beat chance and they are not close to each other. Describing a word by WHICH neurons it recruits puts roughly seven times as many edges inside a category as chance allows, and restricting to the strongest quartile raises that to 93.3 percent on GPT-2 and 98.0 on Qwen3. Describing the same word by WHERE along depth those neurons sit puts only two and a half to four times as many, so the profile geometry carries real category information and carries far less of it. That is the same asymmetry module 2's section 2.4 reports from the supervised side, where Jaccard tracks human typicality at $r$ 0.30 to 0.48 (**Earlier definition**, within-category $\rho_W$ 0.432 / 0.306 on the nopad trees at AP 0.6) while the profile features add barely a point of ranking accuracy on top of it, and Study C shows it is a property of the representations rather than of the model fitted to them.

The paper's other two claims reproduce as well. GPT-2's heaviest edges are handball to squash (the sport) at 0.450, rugby to soccer at 0.433, hockey to soccer at 0.432, handball to hockey at 0.424, judge to lawyer at 0.410 and hockey to squash at 0.380. The surviving cross-domain edges are the interpretable ones, airplane to pilot at 0.248, bicycle to cycling at 0.225, boat to sailing at 0.179, helicopter to pilot at 0.176 and cycling to motorcycle at 0.175, with boat to sailing, bicycle to cycling, cycling to motorcycle and airplane to pilot all reappearing independently on Qwen3. These are the same relations the paper singles out, its examples being "driver" reaching "bus" and "vehicle" and "racing" joining sports to vehicles.

The orthographic alternative was checked and is not the explanation. Only 3 of GPT-2's 55 cross-category Jaccard edges and 0 of Qwen3's 30 join words sharing a leading substring of three characters or more, dress to dresser at 0.260, car to carpenter at 0.105 and beanie to beans at 0.077, so string similarity is the exception rather than the pattern.

### Study C across AP thresholds

From the two final trees, whole-model scope, all five thresholds. `words` counts the words each space could cluster, given as expert-set / profile where they differ, `profile ARI` and $\eta^2$ depth describe the default metric's tree, and the graph column is the share of expert-set link-graph edges joining two words of one category.

| Model | AP | experts | words, set / profile | expert-set ARI | profile ARI | $\eta^2$ depth | graph within-category |
|---|---|---|---|---|---|---|---|
| GPT-2 | 0.5 | 217,149 | 206 | **0.682** | 0.115 | 0.693 | 86.9% |
| GPT-2 | 0.6 | 76,492 | 206 | **0.439** | 0.004 | 0.137 | 81.1% |
| GPT-2 | 0.7 | 29,175 | 206 | **0.286** | $-0.002$ | 0.195 | 68.0% |
| GPT-2 | 0.8 | 9,278 | 189 / 183 | **0.000** | 0.008 | 0.258 | 56.5% |
| GPT-2 | 0.9 | 1,391 | 115 / 83 | **0.000** | 0.050 | 0.802 | 24.3% |
| Qwen3 | 0.5 | 1,573,600 | 206 | **0.772** | 0.194 | 0.459 | 92.6% |
| Qwen3 | 0.6 | 569,820 | 206 | **0.749** | 0.162 | 0.560 | 91.6% |
| Qwen3 | 0.7 | 227,691 | 206 | **0.643** | 0.141 | 0.308 | 87.5% |
| Qwen3 | 0.8 | 82,110 | 206 / 205 | **0.392** | 0.132 | 0.499 | 74.7% |
| Qwen3 | 0.9 | 16,740 | 201 / 196 | **0.000** | 0.000 | 0.084 | 59.3% |

Three readings.

First, **the set-over-profile advantage holds wherever the expert sets carry category structure at all**. The expert-set tree leads at every threshold through AP 0.8 in Qwen3, by a factor of three to five, and through AP 0.7 in GPT-2, where the profile tree is already at chance from AP 0.6.

Second, **GPT-2 loses the category structure of its expert sets much sooner than Qwen3**. Its expert-set ARI falls from 0.682 to 0.286 between AP 0.5 and 0.7 and reaches zero at AP 0.8, while Qwen3 keeps 0.643 at AP 0.7 and 0.392 at AP 0.8, consistent with the faster thinning of GPT-2's expert sets documented in modules 1 and 2. Where both ARIs are at zero, GPT-2 from AP 0.8 and Qwen3 at AP 0.9, the regime is data starvation and not a result, and nothing there should be read as evidence about the representation.

Third, **the profile tree's category agreement is small at every threshold and declines with it, slowly**, from 0.194 at AP 0.5 to 0.132 at AP 0.8 in Qwen3, so the two geometries do not answer the threshold with opposite signs. Its $\eta^2$ on depth is unstable, 0.693 at AP 0.5 against 0.137 at AP 0.6 in GPT-2 and between 0.308 and 0.560 in Qwen3, so the depth reading of the profile trees described above belongs to the lenient threshold.

The sublayer scopes, run at AP 0.5 to 0.7, sharpen the contrast between sublayer types. Expert-set ARI at AP 0.5 on Qwen3 runs `mlp.up_proj` 0.794, `mlp.gate_proj` 0.780, `self_attn.o_proj` 0.760, then falls away through the attention projections, `self_attn.v_proj` 0.694, `self_attn.k_proj` 0.559 and `self_attn.q_proj` 0.477, to `mlp.down_proj` at 0.345. That last figure is the striking one: `mlp.down_proj` is the sublayer whose expert sets lose their category structure fastest, down to 0.014 at AP 0.7, while its layer profiles are strongly depth-determined, $\eta^2$ on depth 0.498 at AP 0.5 and 0.689 at AP 0.7. GPT-2's down projection, `mlp.c_proj`, is likewise its weakest sublayer, 0.331 against 0.638 for `mlp.c_fc`. Whatever down-projection experts encode, it is mostly not category membership.

### What the retired MDS pass established

**Not generated by the current code.** An earlier version of Study C also ran a cross-validated MDS over the same distances, following the way Richie, White, Bhatia and Hout (2020) read their own human similarity data. It was retired when the study was narrowed to its two current deliverables, and MDS survives only as the link graphs' layout engine. Its GPT-2 numbers, measured on the 205-word run that preceded the split of `squash` into two homonyms, are recorded here so they do not have to be rediscovered. Its Qwen3 half ran on the post-trained checkpoint, which these pages no longer report, and was never repeated on Qwen3-1.7B-Base.

The resampling unit was the expert unit, since a layer profile has no subjects to cross-validate over: each of 20 repeats split every word's expert rows in half, built an independent distance matrix from each, fitted the embedding on one and correlated its reconstructed distances against the other. The recoverable dimensionality, the smallest $d$ within one standard error of the best out-of-sample fit, was **2 on GPT-2** (out-of-sample Spearman 0.784 against a split-half reliability ceiling of 0.795, so the plane held essentially everything recoverable and dimensions 3 to 10 bought 0.009 in total).

The axes were shape rather than meaning. On GPT-2 dimension 1 correlated with the depth centre of mass at $\rho = -0.742$ and with the peak bin at -0.570, and dimension 2 with the profile entropy at +0.569. Human typicality never exceeded $|\rho| = 0.21$ on any dimension. Stress-1 of the 2-D solution was 0.157, so on GPT-2 the layout the link graphs use is a fair summary.

## Limitations

**The concept-to-label-word count residual.** The count-matched null is defined over the full pair population, and on that population it works on both axes. Measured on the earlier GPT-2 run at AP 0.5, before the `squash` homonyms were split, over its 20,910 pairs, the raw agreement $S$ correlates with the pair's smaller expert count at Spearman $+0.28$ on the flat axis and $+0.11$ on the block axis, while the standardized $z$ sits at $+0.0035$ and $+0.0029$ respectively. It does NOT remove the residual on the concept-to-label-word sub-population, where about $+0.21$ persists on BOTH axes, $+0.2100$ flat and $+0.2357$ block over 1,576 pairs. The reason is structural. A concept-to-label pair is systematically asymmetric, since the label word holds a smaller expert set than most of the members, between the 1st and the 53rd percentile of the 206 words on GPT-2 at AP 0.5, and the pair's neighbourhood in the count coordinate is drawn from the same asymmetric family, so the null has nothing left to subtract.

Three clarifications matter for reading this correctly. The block axis left that slice-level residual essentially unchanged, moving it by $+0.026$, and on the 1,379 foreign-category rows the two axes are indistinguishable at $+0.2385$ flat and $+0.2412$ block. The apparent jump on the 197 own-category rows that Study A actually fits, from $+0.0856$ flat to $+0.2334$ block, reflects the loss of an ACCIDENTAL CANCELLATION in the flat-axis baseline on those particular rows rather than a new confound. And a bin-count mechanism is DISCONFIRMED rather than merely untested, because a binning effect would move the whole slice and the slice moved by $+0.026$.

This residual does not bias Study A's estimates. Concept expert count is itself uncorrelated with human typicality, Spearman $-0.0414$ at $p = 0.56$ over the 197 concepts of that run and Pearson 0.015 and $-0.001$ over the 198 members of the final trees (module 2, section 2.4, **Earlier definition**, within-category $\rho_W$ 0.057 and $-0.078$ with p 0.43 and 0.28 on the nopad trees at AP 0.6), so a feature confounded with count has no channel through which the confound could manufacture accuracy against that target, and partialling count out shifts `profile_z`'s association with typicality by only 0.008, from $-0.0803$ to $-0.0727$. It is a limitation of describing the label-word profile $z$ as size-free, not a limitation of the study's conclusions.

**`z_cen` calibration is underpowered rather than clean.** The point estimate against concept expert count is $+0.0818$ with a 95 percent confidence interval of $[-0.059, +0.219]$ over the 197 concepts of that run, and 80 percent power to detect the stated $|\rho| < 0.15$ bar only arrives near $\rho = 0.198$ at that sample size, so the bar cannot be established here. The supportable claim is COMPARATIVE rather than absolute. On this exact frame $z_{\text{cen}}$ is measurably cleaner than the label-word $z$, Steiger's $Z = -2.19$ at $p = 0.029$ for two correlations sharing the count variable. The null is also visibly doing its job in the intended direction, since raw $S_{\text{cen}}$ carries $+0.162$ at $p = 0.023$ and standardizing it roughly halves the point estimate to a non-significant $+0.082$.

**`z_cen` is a replacement for `centroid_margin`, not a refinement of it.** The two are moderately concordant at Spearman $+0.41$, about 17 percent shared rank variance. Standardization explains only part of the gap, since the raw un-standardized $S_{\text{cen}}$ already agrees with the margin at only $+0.5145$, so the move from $+0.515$ to $+0.409$ is all the standardization accounts for and the remainder is STRUCTURAL. `centroid_margin` is contrastive, the own-centroid agreement minus the mean of seven foreign ones, while $S_{\text{cen}}$ and therefore $z_{\text{cen}}$ is non-contrastive and carries no foreign term at all. Those answer different questions, one asking whether a concept is closer to its own centroid than to the alternatives and the other asking how close it is to its own centroid full stop. The count-behaviour improvement is supported and is the headline, since the margin carries $+0.215$ at $p = 0.002$ against expert count where $z_{\text{cen}}$ carries $+0.082$, but any reliance on $z_{\text{cen}}$ inheriting the margin's previously validated behaviour is NOT established at 17 percent shared rank variance.

**The centroid null matches on pooled count, not on the number of profiles averaged.** Each entry of the centroid null is placed at $(\log n_c, \log N_j)$, where $N_j$ is the centroid's POOLED expert count. A three-member category's leave-one-out centroid averages two profiles while a twenty-member one averages nineteen, so at equal pooled count those two centroids carry different amounts of averaging noise, and the coordinate does not condition that difference out. The effect is second order, but it is a stated property of the construction rather than something to discover later.

**Study B inherits a presence gate it does not strictly need.** The design frame requires the category LABEL WORD to hold at least one expert before any of that category's concepts enter, which Study A needs and Study B's concept-to-concept features do not. The gate is conservative, so Study B under-reports the number of pairs it could have fitted rather than mis-reporting any number it does fit, and its statistical power at strict thresholds is understated rather than overstated. It was left unchanged deliberately, since removing it would mean introducing a second frame-construction path at the very end of the restructure and changing results a second time, and a stated conservative limitation serves the thesis better than a late structural change.

**`vegetables` is anomalous in both studies at once, and the pooled means hide it.** The accuracies and correlations above are means over eight held-out categories, and `vegetables` is the weakest category on the label-word Jaccard cell of Study A and on the Jaccard cell of Study B, in both models. In Study A it sits at 37.9 percent in GPT-2 and 41.6 in Qwen3 on the label-word Jaccard cell, and below the 50 percent chance line in 11 of the 12 GPT-2 cells and 7 of the 12 Qwen3 cells, so no feature set and neither reference recovers its typicality ordering reliably. In Study B it is GPT-2's only category whose Jaccard correlation with human ratings is negative, $-0.002$, rising to only $+0.138$ on the fullest cell against a category noise ceiling of 0.909, where the next weakest category, `birds`, reaches $+0.092$ and $+0.205$. In Qwen3 it is also last, 0.350 rising to 0.470, against 0.434 and 0.556 for `fruit`, the next weakest. Being simultaneously the worst category against human similarity and the most often inverted in the ranker points at the category itself rather than at either model, and the most likely candidate is polysemy in its member words, several of which are common non-food senses in a language model. It is recorded here as unexplained. All figures above are from `ranker_per_category.csv` and `pair_similarity_per_category.csv` of the two final trees at AP 0.5, and the module draws `ranker_pair_errors.png` precisely so a systematically inverted category can be inspected rather than only averaged over.

**At strict thresholds Study A's numbers can be degenerate.** Measured at AP 0.9 on Qwen3 at the whole-model scope, Study A fits 50 concepts across only 2 categories, and `jaccard_pct` is exactly 0 for 48 of those 50 rows. GPT-2 at AP 0.8 fits 69 concepts across 3 categories, and at AP 0.9, where no category label keeps an expert, it fits nothing. Those cells must not be read as a result. The guards catch the empty case but not this one, where a study runs, writes a table and reports accuracies built almost entirely on a constant feature.
