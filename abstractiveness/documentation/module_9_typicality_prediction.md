# Module 9: Typicality Prediction, Human Pair Similarity and Concept Structure

## Research question

Modules 3 and 5 describe a word pair in two ways. The Jaccard index asks which neurons the two words share, and the layer-profile agreement asks whether the two words spread their experts over depth in the same proportions. Module 4 establishes that the first of those tracks human typicality at Pearson $r$ between 0.21 and 0.45 depending on the model and the AP threshold, 0.320 for GPT-2 and 0.452 for Qwen3 at AP 0.5, falling to 0.209 and 0.328 by AP 0.7.

Does the layer-profile agreement add predictive signal over the Jaccard index?

The module asks that one question with two human targets rather than asking four loosely related questions in four subchapters. Study A predicts which of two same-category concepts humans rate as the more typical, and Study B predicts the measured human similarity of a concept pair. Both are fitted on one generated feature grid whose six cells carry the same names and the same meaning in either study, so a reader can confirm that the same model was asked the same question on both targets. The module is deliberately small, with at most three predictors over a couple of hundred concepts, because the question is whether a second feature carries signal the first lacks rather than how well typicality can be predicted in general.

Study C then drops the target entirely and asks the **unsupervised** question, which every other module of this pipeline leaves unasked:

> Given nothing but the pairwise distances between the words' expert representations, what grouping falls out on its own, and is it the category structure the dataset is built on?

That question is not a variant of the first one. Studies A and B start from a grouping we already believe in and measure how well the representations honour it, so they can only ever report how much of an expected structure is present. Study C starts from nothing, so it can report a structure nobody asked for, and it does. It answers in exactly two views, a **dendrogram** of the layer profiles and a **node-link graph** in the style of Figure 4 of Fedzechkina's ExpertLens paper, one folder each under `9.3_concept_structure/`.

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

The `js_distance` segment names the profile metric, see the metric registry below. Every comparison CSV carries a `metric` column naming the metric a row was built on, and the `jaccard` cell, which uses no profile metric, carries `metric = "none"` and appears once rather than once per metric. With one registered metric this reduces exactly to the tables above, and with the eight registered today the same table repeats once per metric under one vocabulary.

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

### Study C, concept structure

**Target.** None. This study is fitted to nothing and predicts nothing, so it has no train and test split, no held-out category and no accuracy. Everything it reports describes the representation itself, scored against groupings it never saw. It has exactly two deliverables, one folder each.

**Input.** The scope's expert frame, re-keyed onto the block axis for the whole-model scope and left on its own axis otherwise, giving each item $c$ its layer profile $p_c$ over $B$ depth bins ($B = 12$ on GPT-2, $B = 28$ on Qwen3). Every word with a usable profile enters, category labels included. The study is self-sufficient from `scope.expert_df` and independent of the design frame studies A and B share, so it runs even at strict thresholds where those two have no rows left.

#### The dendrogram, and where its divisions appear

**How the tree is built.** Agglomerative, bottom up. Every word starts as its own cluster, and at each step the two closest clusters are merged, so a tree over $n$ words makes exactly $n - 1$ merges. The **height** of a merge is the distance between the two clusters at the moment they joined, under the linkage rule in use, and it is the only quantity the horizontal axis of the figure shows. Under `average` linkage, the winner on both architectures, that distance is

$$ d(A, B) = \frac{1}{|A|\,|B|} \sum_{a \in A} \sum_{b \in B} d(a, b), $$

the mean distance between every member of one cluster and every member of the other. Merges are made in non-decreasing height order, so the tree is drawn narrow near the leaves and wide near the root.

**When a division appears.** A division is a merge read backwards. Cutting the tree at height $h$ severs every link drawn beyond $h$, and what remains are the clusters that existed just before the first merge above $h$. Cutting for $k$ clusters is the same operation stated differently, since undoing the last $k - 1$ merges leaves $k$ groups, so the cut height sits just below the $(k-1)$-th largest merge height. The dendrogram figure draws that height as a dashed line and colors the branches by it.

That says where a division can be **taken**, not where one **exists**. For the second question the quantity is the gap between consecutive merge heights, written to `merge_heights.csv` as `height_gap` and drawn in `merge_heights.png`. A large gap means the next thing the algorithm had to join was much further away than anything it had joined so far, which is the signature of a division the data actually carries. A run of small gaps means the merges there cost nothing, so the groups they produce are an artefact of having to keep merging rather than a boundary. A tree whose gaps are all small has no natural number of clusters at all, and the honest reading of it is that the words form a gradient. The final merge is excluded from every "largest gap" statement, because it joins everything into one cluster and cutting there is not a division.

**One dissimilarity per registered metric.** Study C sweeps `PROFILE_METRICS` exactly as modules 3 and 5 do, so every registered way of comparing two layer profiles gets its own tree and its own graph, and the choice of metric becomes visible rather than assumed. Every metric reports an agreement on a 0 to 100 scale with 100 meaning identical profiles, so one conversion serves all of them,

$$ d(a, b) = \frac{100 - S(a, b)}{100}. $$

For `js_distance` that is exactly $\sqrt{\mathrm{JSD}(p_a, p_b)}$, the Jensen-Shannon distance and a true metric on the simplex, so the default metric's numbers are unchanged by the sweep. The two signed metrics, `pearson` and `spearman`, run to -100 rather than 0, so theirs lands in $[0, 2]$. That is a scale difference rather than a problem, since average linkage and every rank statistic reported here are invariant to it, but it is the reason raw distances must never be compared across metrics. All seven are built in one pass by `layer_profile_metric_matrices`, which shares the profile build and the Jensen-Shannon tensor rather than recomputing them seven times.

One further space, `expert_set_jaccard`, is not a profile metric at all. Its dissimilarity is $1 - J(a,b)$ on the two words' expert SETS, the same Jaccard index module 3 and module 5 use, and it is the one space here that describes a word by **which** neurons it recruits rather than by **where along depth** it puts them. Clustering it with the identical machinery is what turns the set-versus-profile comparison into a controlled one: same words, same linkage, same cut, same scoring, only the description of a word changes. It also places more words than the profile spaces at strict thresholds, since a Jaccard is defined for a word holding a single expert while a layer profile needs `MIN_PROFILE_EXPERTS`.

One extra space, `js_distance_z`, carries the **count-matched null** of the default metric. It exists because of the confound stated in `helpers.layer_profile_matrices`: raw profile agreement is very largely a readout of expert-set **size**, correlating with the pair's smaller expert count at Spearman 0.965 on synthetic data carrying no true signal, so a tree built on the raw distance alone would recover "small-set words against large-set words" and present it as a discovery. One metric carries the control because the null is metric-agnostic by construction, so running it seven times would repeat the same check on seven monotone rescalings of the same profiles. Its scale is ordinal, the shift that makes it non-negative gives its zero no meaning, so it is read by average linkage and rank statistics only.

**Three candidate trees**, ranked by cophenetic correlation, which measures how faithfully a tree reproduces the distances it was built from and is therefore the criterion for which tree to read rather than a result. `average` and `complete` run on the distance matrix directly. Ward is **not** run on the distance matrix, because Ward's criterion is defined through Euclidean variance and a Jensen-Shannon matrix is not Euclidean. It runs instead on the square-rooted profiles $\sqrt{p_c}$, whose Euclidean distance is $\sqrt{2}$ times the Hellinger distance, which is the Euclidean space this problem legitimately has.

**One figure per deliverable.** Each metric produces exactly one dendrogram and one link graph. Every candidate tree is still scored, so the cophenetic ranking that picks one is evidenced, but only the winner is drawn, since three dendrograms per metric across eight metrics would be twenty-four near-identical figures answering a question the ranking CSV answers in one line. The merge-height curve, the cluster quality curve and the cluster mean-profile plot are likewise kept as CSVs and not drawn: they are diagnostics behind the numbers quoted in the Results section rather than things to display, and each can be redrawn from its own table without rerunning anything.

**Scoring the cut.** For every $k$ from 2 to 20, the cut is scored against the structure we expect (adjusted Rand and normalized mutual information against the categories, adjusted Rand against the abstraction level), against internal quality (silhouette, with the largest cluster's share beside it so a cut isolating four outliers is not read as a two-way structure), and against the confounds we do not want, through
$$ \eta^2 = \frac{\sum_j n_j (\bar{x}_j - \bar{x})^2}{\sum_i (x_i - \bar{x})^2} $$
for $x$ in turn the log expert count, the peak bin, the depth centre of mass, the profile entropy and Geary's C. A cut with $\eta^2 = 0.8$ on entropy and an adjusted Rand of 0.1 on the categories has grouped the words by shape, not by meaning.

**How to read the three numbers.** Each answers a different question about the same tree, and each one alone misleads.

- **Cophenetic correlation** asks *is the tree lying about the distances?* A dendrogram is a lossy summary: it claims two words are as far apart as the height where their branches first join, and this correlates that claim against the real distance over all pairs. 1 means nothing was lost in the squashing, 0 means the picture is unrelated to the data. It measures faithfulness to the input distances and **not** whether those distances carry anything, so a perfectly accurate map of an empty field still scores high. That is not hypothetical here: `pearson` and `spearman` build the two most faithful trees in the sweep and the two emptiest.
- **Adjusted Rand index against the categories** asks *did the tree rediscover the categories on its own?* It cuts the tree into as many groups as there are categories, then for every pair of words asks whether the tree and the category agree about putting them together. "Adjusted" means the agreement that random grouping would produce by luck has been subtracted off, so 0 is chance and 1 is an exact match.
- **$\eta^2$ on a descriptor** asks *what did the tree group by instead?* It is the share of that descriptor's variation across words that knowing a word's cluster lets you predict: 0 means the clusters are unrelated to it, 1 means the clusters simply are that descriptor. Read on the depth centre of mass, it says how much of "where in the network this word's experts sit" the clustering has captured.

Read together they are the finding of this study in one line: an adjusted Rand of 0.115 beside an $\eta^2$ of 0.675 on depth says the clusters carry almost nothing about what a word means and a great deal about where in the network it lives.

**The tree's diagnostic.** `category_separation.csv` and `level_separation.csv` ask how well the groups we already believe in survive in the same distance space, so a low agreement between tree and categories can be attributed either to the tree or to the groups. For each group the separation is the mean distance from a member to a non-member minus the mean distance among its own members, with a $p$ value from 2000 label permutations. The permutation null is mandatory rather than optional here, since the groups are very unequal in size (8 label words against 197 members, categories of 20 to 32) and a plain comparison of means would reward the small groups by construction.

**Guards at strict thresholds.** A word enters the study only if `layer_profile_matrices` could build a profile for it, which needs `MIN_PROFILE_EXPERTS = 2` experts. That helper marks a word it could not place by filling its whole row **and column** with NaN, so usability must be read off the DIAGONAL: a usable word sits at distance 0 from itself, an unusable one at NaN. Testing whether a whole row is finite would be wrong, because one unusable word puts a NaN in every other word's row and collapses the usable set to nothing.

How much this matters grows quickly with the threshold. On the Qwen3 `mlp.down_proj` scope, 205 of 205 words are usable at AP 0.5, 203 of 204 at AP 0.6 (the single word `professions` holds one expert there), 184 of 191 at AP 0.7, 120 of 153 at AP 0.8 and 37 of 57 at AP 0.9. Two consequences are handled rather than assumed away. A dissimilarity space with no more usable words than the $k = 8$ cut asks for is skipped with a warning, since the cut height it would draw is not defined, and the rest of the study proceeds. A labelling with fewer than two distinct groups among the survivors, which is what the abstraction level becomes once every category-label word has dropped out, yields NaN for its separation and agreement columns rather than an exception. Every scope therefore contributes the same columns to `sublayer_comparison.csv`, with NaN where a deliverable could not run.

#### The link graphs

Figure 4 of Fedzechkina's ExpertLens paper (`temp_docs/ExpertLens paper-Fedzechkina.pdf`) draws the concept structure of the expert space as a node-link diagram, one node per concept with edge thickness carrying the Jaccard similarity of the two concepts' expert **sets**, and reads three things off it: that within-domain concepts are densely connected, that cross-domain edges are sparser, and that the cross-domain edges which do survive are meaningful, its examples being "driver" reaching "bus" and "vehicle" and "racing" joining sports to vehicles. Note that this figure is a node-link diagram and not an MDS, which is easy to misremember.

The figure is drawn here twice, on the two similarities this pipeline has, so the geometries can be read side by side:

- `expert_set_jaccard`, the quantity the paper uses, which asks **which** neurons two words share
- `layer_profile_agreement`, the quantity every profile-based module here uses, which asks **where** along depth two words put their experts

Three adaptations are needed and each is deliberate. First, the paper draws 40 concepts in 10 domains and can afford every edge, while this dataset has 205 words and 20,910 pairs, which at any global threshold is either a hairball or an empty canvas, so each node keeps its `GRAPH_NEIGHBORS = 3` strongest partners and an edge kept by either endpoint is kept once. That asymmetry is intended, since dropping an edge one concept counts among its strongest because the other does not would hide exactly the hub structure the figure exists to show. Second, node positions come from a 2-D metric MDS rather than from a force layout, which with 205 words would be unreadable and would change with the seed. Each graph is laid out by the same quantity its own edges carry, which matters more than it sounds: an earlier version positioned both graphs by the layer-profile distance while the Jaccard graph's edges carried expert-set overlap, so the layout fought the edges, placing strongly connected words far apart because they happened to differ in depth. Measured on the 2-D category silhouette of the Jaccard graph, matching the layout to the edges moves GPT-2 from -0.120 to +0.073 and Qwen3 from -0.079 to +0.120, which is the difference between categories that read as clumps and categories that read as noise. MDS appears in this module for that one purpose, and nothing about stress or dimensionality is written out because the embedding is furniture here rather than a result. Third, the expert-set Jaccard is computed on the scope frame and not the depth-aggregated one, since a set is unordered and block aggregation would merge distinct units of one block into one bin and inflate every overlap.

The paper's qualitative claims are then made countable. `concept_graph_domain_structure.csv` compares the observed share of edges joining two words of one category against the share obtained when the category labels are shuffled over the words, which preserves the graph and the category sizes and destroys only the assignment, mirroring the resampled-domain baseline the paper uses for the same question. It is reported over all edges and over the strongest quartile separately, since the figure draws attention to the thick edges and a structure carried only by the weak ones would be a different claim. The edge list additionally carries `shared_prefix`, the length of the two words' common leading substring, because a neuron that is an expert for a word FORM rather than for its meaning would produce cross-domain edges like "dress" to "dresser", and the column makes that alternative visible instead of leaving it to be assumed away.

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

The registry holds eight entries, `js_distance`, `wasserstein`, `cosine`, `pearson`, `spearman`, `js_divergence`, `hellinger` and `profile_jaccard`, all defined in module 3's subchapter 3.3, which also groups them into the divergence, correlation, order-reading and L1 families and explains why their raw scales are not comparable. `DEFAULT_PROFILE_METRIC` names `js_distance`, defined as $100(1 - \sqrt{\mathrm{JSD}})$ exactly as module 3 defines it, and `ACTIVE_PROFILE_METRICS` keeps it first so the summary row and the legacy column names always describe it. `layer_profile_matrices` takes a `metric` parameter defaulting to that name and dispatches through the registry. `ACTIVE_METRICS` in module 9 lists the metrics the grid is instantiated on and is now `ACTIVE_PROFILE_METRICS`, shared with modules 3 and 5, so one edit changes what the whole pipeline computes. Registering a metric and listing it there instantiates every cell of both studies on it and touches nothing else. The grid generates five cells per metric plus the single metric-free Jaccard cell, so at eight metrics Study A fits 41 cells per reference and Study B fits 41, against 6 and 6 with one metric, which is about 80 seconds for the five GPT-2 scopes of one AP threshold. The count-matched null is already metric-agnostic, since it standardizes whatever agreement matrix it is handed against count-matched reference entries, so every registered metric gets its $z$ column and its $z_{\text{cen}}$ column for free.

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
├── 9.3_concept_structure/
│   ├── word_profile_descriptors.csv      per word, the vocabulary the clusters are read in
│   ├── depth_band_by_group.csv           early/middle/late shares per category and level
│   ├── dendrograms/
│   │   ├── dendrogram_summary.csv        one row per space, the whole comparison
│   │   ├── expert_set_jaccard/           WHICH neurons, the set geometry
│   │   │   ├── dendrogram_average.png    THE deliverable, leaves coloured by category
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
│       │   ├── concept_graph_edges.csv   edge_type and shared_prefix per edge
│       │   └── concept_graph_domain_structure.csv
│       └── <metric>/                     one per profile metric, same contents
├── sublayer_comparison.csv              cross-scope, written by the executor
└── sublayer_comparison.png
```

`typicality_model_design.csv` stays at the scope root because both studies are built from it. Its columns, in file order, are `concept`, `category`, `human_typicality`, `jaccard_pct`, then per registered metric the label-word pair `profile_<metric>` and `profile_<metric>_z`, then `mean_jaccard_to_members`, then per registered metric the centroid pair `centroid_profile_<metric>` and `centroid_profile_<metric>_z`. At eight registered metrics that is 4 fixed columns plus 4 per metric, 36 in all. With one registered metric it would be the nine columns `concept`, `category`, `human_typicality`, `jaccard_pct`, `profile_js_distance`, `profile_js_distance_z`, `mean_jaccard_to_members`, `centroid_profile_js_distance` and `centroid_profile_js_distance_z`. Each study additionally writes the post-dropna frame it fitted, so the identical-rows guarantee is inspectable rather than only asserted.

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

### Study C, concept structure

Unlike the sections above, this one is **not** provisional. It is reported from full whole-model runs of both architectures on the Richie-HSJ dataset at AP 0.5, GPT-2 in `research_plots_gpt2_richie_hsj_sensefix` and Qwen3-1.7B in `research_plots_qwen_richie_hsj_restructured`, covering all 205 words, every one of which carries a usable profile. GPT-2 holds 216,555 expert rows over 12 blocks, Qwen3 holds 1,188,772 over 28.

**The headline: what a word is described BY decides what the tree finds.** Ten spaces are clustered with identical machinery over the same 205 words. One describes a word by which neurons it recruits, the rest by where along depth it puts them. The trees they produce are not in the same league:

| space | cophenetic | ARI vs category | $\eta^2$ depth | category silhouette |
|---|---|---|---|---|
| **expert_set_jaccard** (GPT-2) | **0.914** | **0.644** | 0.213 | **+0.036** |
| best profile metric (GPT-2) | 0.679 | 0.115 | 0.675 | -0.069 |
| **expert_set_jaccard** (Qwen3) | **0.916** | **0.765** | 0.348 | **+0.044** |
| best profile metric (Qwen3) | 0.674 | 0.118 | 0.589 | -0.051 |

The expert-set tree beats every layer-profile tree by a factor of five to six on category agreement, is by a wide margin the most faithful tree of the ten, and is the only space in which the category partition has a POSITIVE silhouette. Every profile space is negative there, meaning a word sits no closer to its own category than to the rest.

What the expert-set tree recovers at the $k = 8$ cut on GPT-2 is close to the dataset's own answer key: 22 of 23 professions in one cluster, 19 of 20 sports in another, 27 of 29 clothing, 17 of 19 furniture, and the single confusion it does make is fruit with vegetables, merged into one 38-word cluster. That is the mistake a person would make. It never saw a category label.

The reading is not that one geometry is good and the other broken. Both carry real structure and they carry DIFFERENT structure, which the two $\eta^2$ columns show as clearly as the ARI does: the profile trees explain 0.68 of the variance in depth and almost none of the categories, the expert-set tree does the reverse at 0.21 and 0.64.

**The layer-profile trees divide the words by depth, not by category.** Average linkage wins in both models (cophenetic $r$ 0.679 on GPT-2 against 0.625 for complete and 0.458 for Ward, and 0.685 on Qwen3). At the $k = 8$ cut:

| | GPT-2 | Qwen3 |
|---|---|---|
| adjusted Rand vs the categories | 0.115 | 0.060 |
| adjusted Rand vs the abstraction level | -0.001 | 0.035 |
| $\eta^2$, profile entropy | 0.774 | 0.536 |
| $\eta^2$, depth centre of mass | 0.675 | 0.561 |
| $\eta^2$, peak bin | 0.663 | 0.272 |
| $\eta^2$, log expert count | 0.523 | 0.340 |

The clusters explain most of the variance in where and how widely a word's expert mass sits along depth, and almost none of the category membership. Pushing $k$ to 20 moves the adjusted Rand only to 0.133 on GPT-2 and 0.123 on Qwen3, so the shortfall is not a matter of cutting in the wrong place. Removing the count gradient drops GPT-2's $\eta^2$ on log expert count from 0.523 to 0.270 while leaving the centre-of-mass $\eta^2$ at 0.572, so the depth structure is not the size confound wearing a different name.

**Where the divisions actually are, and why $k = 8$ is a convention.** Reading `merge_heights.csv` on GPT-2, excluding the root merge, the four largest gaps in the swept range are:

| clusters remaining | height | gap to previous | sizes joined |
|---|---|---|---|
| 2 | 0.2677 | 0.0389 | 1 and 200 |
| 4 | 0.2217 | 0.0306 | 10 and 50 |
| 5 | 0.1912 | 0.0109 | 35 and 105 |
| 6 | 0.1802 | 0.0108 | 10 and 95 |

The largest gap of all leaves two clusters, but it joins a single word (`turkey`) to the other 200, so it is an outlier split rather than a structural boundary. The largest gap that divides two real groups leaves **four** clusters, at height 0.2217, joining a 10-word group to a 50-word one. Below six clusters every gap falls under 0.011, and the merge that leaves eight clusters costs 0.000926, which is essentially free.

That last number should be read carefully, because it qualifies every $k = 8$ figure in this study. Eight is chosen so the cut is directly comparable to the eight categories of the dataset, not because the tree has a division there. Qwen3 behaves the same way, its $k = 8$ merge costing 0.002976 against a largest in-range gap of 0.0323. The composition tables at $k = 8$ are therefore a fair description of what eight groups of these words look like, and not evidence that eight groups exist.

`cluster_mean_profiles_k8.png` shows what the groups are. On GPT-2 every cluster mean is U-shaped, heavy at block 1 and block 12 and thin through the middle, and the clusters differ almost entirely in the ratio between the two ends. The raw descriptor confirms it: of 205 words, 103 peak at block 1 and 71 at block 12, leaving 31 anywhere else, and the middle band holds 1.8 percent of reliable peaks.

**Which groups the depth axis does divide.** GPT-2, share of words peaking in the last fifth of the model, over the 168 words with a reliable peak:

| Group | late share | Group | late share |
|---|---|---|---|
| level-1 label words | 66.7% | clothing | 33.3% |
| sports | 63.6% | birds | 28.0% |
| professions | 63.2% | furniture | 18.8% |
| vehicles | 45.0% | fruit | 5.6% |
| all words | 34.5% | vegetables | 0.0% |

The permutation test of category separation gives the same ordering. Five of eight GPT-2 categories are reliably tighter than chance, professions at 0.0636 ($p = 0.0005$), vehicles 0.0404 (0.0010), sports 0.0314 (0.0005), fruit 0.0243 (0.0035) and vegetables 0.0225 (0.0105), while birds (0.16), furniture (0.28) and clothing (0.26) are indistinguishable from an arbitrary set of 20 to 30 words. The ranking is **identical** in the count-matched space, which is the control that matters. Qwen3 separates categories more evenly, all eight clearing $p \le 0.0045$ at roughly double the mean separation (0.0481 against 0.0250) and a category silhouette of -0.004 against GPT-2's -0.069. The two models share their endpoints, professions strongest and clothing weakest in both, and differ in the middle.

The depth ordering of the categories does **not** carry across architectures. Qwen3 is late-heavy overall (66 percent of words peak in its last fifth) and its per-category late shares run vehicles 90.9 percent, birds 86.7, fruit 85.7, vegetables 75.0, sports 74.1, clothing 58.6, furniture 45.0 and professions 35.7. Professions is the extreme category in both models and at opposite ends. Absolute depth position is an architecture property, the fact that categories differ in depth at all is shared.

**The abstraction level shows up as coherence, not location.** The 8 category-label words sit closer to each other than a random 8 of the 205 do, separation 0.0336 at $p = 0.0265$ on GPT-2 and 0.0481 at $p = 0.0045$ on Qwen3. On Qwen3 the effect survives count matching ($p = 0.0155$ for the label group, 0.0065 overall) with a positive level silhouette of +0.022, while on GPT-2 it weakens to $p = 0.0815$ and is not separable from expert-set size. This is a different statement from module 2's depth test and does not conflict with it. Module 2 asks where the levels sit on average and finds GPT-2 labels deeper than their members and Qwen3 flat. This asks whether the label words share a profile SHAPE, and on Qwen3 the answer is yes even though their mean depth does not differ. All three Qwen3 label words with a reliable peak peak in the middle band, which holds 14 percent of words overall. On GPT-2 the direction is the familiar one, 6 of 8 labels peak late against 36 percent of members, and the centre of mass separates the levels at AUC 0.737 with $p = 0.0215$, reproducing module 2's finding on the block axis.

**The same statistic on both geometries.** A sceptic could object that the gap above is an artefact of comparing two different statistics, Jaccard on one side and Jensen-Shannon on the other. `profile_jaccard` closes that hole. It is the weighted (Ruzicka) Jaccard,

$$ J(p, q) = \frac{\sum_k \min(p_k, q_k)}{\sum_k \max(p_k, q_k)}, $$

which on set indicator vectors reduces exactly to the ordinary $|A \cap B| / |A \cup B|$, verified to machine precision. Running it on layer profiles asks the Jaccard question of the depth distribution instead of the neuron set, with the functional form held fixed.

It lands at ARI 0.103 on GPT-2 and 0.105 on Qwen3, in the middle of the divergence family and nowhere near the 0.644 and 0.765 the same index reaches on expert sets. The gap is therefore a property of the representation and not of the measure.

It is also the registry's only L1 member. Profiles are normalized, so $\sum_k \max = 2 - \sum_k \min$ and $\sum_k \min(p,q) = 1 - \mathrm{TV}$, giving $J = (1 - \mathrm{TV})/(1 + \mathrm{TV})$, a strictly decreasing function of total variation distance (verified numerically at Spearman exactly -1). Being a monotone transform of TV, it ranks every pair as TV does, so rank-based statistics read the two identically while average linkage and the linear consumers do not, since a mean of transformed distances is not the transform of a mean.

**The metric sweep, and why the default holds.** Each of the eight registered metrics gets its own tree. On GPT-2 the divergence family leads on category agreement, `js_distance` at ARI 0.115, `js_divergence` and `hellinger` at 0.106, against `cosine` 0.065, `wasserstein` 0.054 and the two signed metrics at essentially nothing, `pearson` 0.003 and `spearman` 0.001. Two things in that table are worth keeping.

First, `pearson` and `spearman` build the most FAITHFUL trees, cophenetic 0.839 and 0.872 against 0.679 for the default, and the least USEFUL ones, with the lowest category agreement and the lowest $\eta^2$ on depth (0.082 and 0.113 against 0.675). Cophenetic correlation measures how well a tree reproduces the distances it was built from, not whether those distances carry anything, so it must never be read as a quality score on its own. Second, `wasserstein` has by far the highest $\eta^2$ on the depth centre of mass, 0.892 on GPT-2 and 0.933 on Qwen3, which is exactly what it should do, since it is the only metric in the registry that reads bin ORDER and therefore the purest depth measure available.

The ranking is architecture-dependent and the default is not always the winner. On Qwen3 the order inverts at the top, `pearson` reaching ARI 0.118 and `cosine` 0.114 against `js_distance` 0.060. No metric changes the conclusion of this study, since all seven agree that the trees are far better explained by depth than by category, but a claim that names one metric should say which model it was measured on.

**The link graphs are the sharpest result of the study.** The same 205 words, at the same threshold, in the same model, drawn twice:

| | GPT-2 Jaccard | GPT-2 profile | Qwen3 Jaccard | Qwen3 profile |
|---|---|---|---|---|
| edges retained | 479 | 457 | 462 | 463 |
| within-category | 363 | 136 | 367 | 175 |
| label to member | 60 | 25 | 59 | 31 |
| cross-category | 56 | 296 | 36 | 257 |
| within-category share, categorized edges | 86.6% | 31.5% | 91.1% | 40.5% |
| the same under label permutation | 12.4% | 12.4% | 12.4% | 12.4% |
| ratio observed over null | **6.98** | **2.54** | **7.36** | **3.27** |
| $p$ | 0.0005 | 0.0005 | 0.0005 | 0.0005 |

The profile column above is `js_distance`, and the choice barely matters: across all eight metrics the profile graphs span 22.7 to 31.5 percent within-category on GPT-2 and 30.5 to 40.7 on Qwen3, so every one of them sits far below the expert-set graph. `js_distance`, `js_divergence` and `hellinger` produce identical graphs, at 31.5 percent on GPT-2, which is correct rather than a bug: they are monotone transforms of one another, so they rank each word's neighbours identically and a graph built from the top three neighbours cannot tell them apart.

Both similarities beat chance and they are not close to each other. Describing a word by WHICH neurons it recruits puts roughly seven times as many edges inside a category as chance allows, and restricting to the strongest quartile raises that to 93.3 percent on GPT-2 and 98.0 on Qwen3. Describing the same word by WHERE along depth those neurons sit puts only two and a half to three times as many, so the profile geometry carries real category information and carries far less of it. That is the same asymmetry module 4 reports from the supervised side, where Jaccard tracks human typicality at $r$ 0.32 to 0.45 while the profile features add barely a point of ranking accuracy on top of it, and Study C shows it is a property of the representations rather than of the model fitted to them.

The paper's other two claims reproduce as well. GPT-2's heaviest edges are hockey to soccer at 0.440, rugby to soccer at 0.429, handball to hockey at 0.414, judge to lawyer at 0.412, educator to teacher at 0.354 and basketball to volleyball at 0.353. The surviving cross-domain edges are the interpretable ones, airplane to pilot at 0.249, bicycle to cycling at 0.235, helicopter to pilot at 0.185 and boat to sailing at 0.167, with boat to sailing, bicycle to cycling, cycling to motorcycle and airplane to pilot all reappearing independently on Qwen3. These are the same relations the paper singles out, its examples being "driver" reaching "bus" and "vehicle" and "racing" joining sports to vehicles.

The orthographic alternative was checked and is not the explanation. Only 2 of GPT-2's 56 cross-category Jaccard edges and 0 of Qwen3's 36 join words sharing a leading substring of three characters or more, so the two visible string neighbours in the GPT-2 list, dress to dresser at 0.256 and vase to vulture at 0.214, are the exceptions rather than the pattern.

### Study C across AP thresholds

Reported from the full Qwen3 sweep, all five thresholds and all eight scopes, 40 scopes in total. The whole-model row:

| AP | expert rows | concepts | expert-set ARI | profile ARI | $\eta^2$ depth | graph within-category |
|---|---|---|---|---|---|---|
| 0.5 | 1,188,772 | 197 | **0.765** | 0.060 | 0.561 | 91.1% |
| 0.6 | 418,529 | 197 | **0.718** | 0.147 | 0.503 | 90.3% |
| 0.7 | 165,497 | 197 | **0.675** | 0.132 | 0.336 | 87.4% |
| 0.8 | 59,797 | 168 | **0.433** | 0.175 | 0.497 | 72.5% |
| 0.9 | 12,212 | 47 | 0.000 | 0.006 | 0.109 | 55.6% |

Three readings.

First, **the set-over-profile advantage is not a property of one threshold**. The expert-set tree leads by a factor of five to twelve at AP 0.5 through 0.7 and still doubles the profile tree at AP 0.8. The finding survives the sweep rather than depending on the lenient end of it.

Second, **AP 0.9 is a data-starvation regime and not a result**. Only 47 concepts keep a usable profile there, against 197 at AP 0.5, and both geometries collapse to zero together. Nothing at that threshold should be read as evidence about the representation, only about how little of it survives the filter. The trend across the other four thresholds is the honest one, and it declines smoothly as experts are stripped away.

Third, **the profile tree moves in the opposite direction over the usable range**, 0.060, 0.147, 0.132, 0.175 across AP 0.5 to 0.8. Stripping marginal experts slightly sharpens what little category signal the depth profiles carry, while it steadily erodes the far larger signal the expert sets carry. The two geometries do not merely differ in strength, they respond to the threshold with opposite signs.

The sublayer scopes sharpen this further. Expert-set ARI at AP 0.5 runs `self_attn.o_proj` 0.813, `mlp.up_proj` 0.806, `mlp.gate_proj` 0.748, then falls away through the attention projections to `mlp.down_proj` at 0.030. That last figure is the striking one: `mlp.down_proj` is the one sublayer whose expert sets carry essentially no category structure at any threshold, while its layer profiles are among the most depth-determined in the model, reaching $\eta^2 = 0.850$ on depth at AP 0.9. Whatever `down_proj` experts encode, it is not category membership.

### What the retired MDS pass established

An earlier version of Study C also ran a cross-validated MDS over the same distances, following the way Richie, White, Bhatia and Hout (2020) read their own human similarity data. It was retired when the study was narrowed to its two current deliverables, and MDS survives only as the link graphs' layout engine. Its numbers are recorded here so they do not have to be rediscovered.

The resampling unit was the expert unit, since a layer profile has no subjects to cross-validate over: each of 20 repeats split every word's expert rows in half, built an independent distance matrix from each, fitted the embedding on one and correlated its reconstructed distances against the other. The recoverable dimensionality, the smallest $d$ within one standard error of the best out-of-sample fit, was **2 on GPT-2** (out-of-sample Spearman 0.784 against a split-half reliability ceiling of 0.795, so the plane held essentially everything recoverable and dimensions 3 to 10 bought 0.009 in total) and **8 on Qwen3** (0.951 against 0.959). Qwen3's profiles were far more reliable to begin with, 0.959 against 0.795, so the two architectures differ in kind and not only in degree.

The axes were shape rather than meaning. On GPT-2 dimension 1 correlated with the depth centre of mass at $\rho = -0.742$ and with the peak bin at -0.570, and dimension 2 with the profile entropy at +0.569. On Qwen3 entropy loaded on dimension 1 at -0.652 and depth arrived on dimensions 4 and 5 (+0.810 and -0.725 on the centre of mass), consistent with its higher dimensionality. Human typicality never exceeded $|\rho| = 0.21$ on any GPT-2 dimension, and its largest value anywhere, +0.341 on Qwen3's first dimension, sat on the entropy dimension and was therefore not independent evidence of a typicality axis. Stress-1 of the 2-D solution was 0.157 on GPT-2 and 0.218 on Qwen3, so the layout the link graphs use is a fair summary on GPT-2 and a lossy one on Qwen3.

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
