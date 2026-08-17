"""Module 9: does the depth allocation of a word's experts predict how humans judge it?

Two studies over ONE generated feature grid, both measured against human judgments rather
than against a model-internal proxy.

Study A, folder 9.1, is the within-category pairwise typicality ranking, "of these two
same-category concepts, which is the more typical". It is the formulation the ratings were
actually collected in, the source column is typicality_HSJ_pairwise, it turns a couple of
hundred concepts into thousands of ordered pairs, and it cancels every category-level term
algebraically, since differencing two members of one category removes anything they share.
The REFERENCE a concept is measured against is a factor of the study rather than a separate
subchapter, so each cell is fitted twice, once against the category LABEL WORD and once
against the leave-one-out MEMBER CENTROID, and each arm carries its own Jaccard baseline so
no gain from a profile feature is ever credited to a change of reference. The per-concept
regression that used to be subchapter 9.1 survives only as the regression_accuracy column
beside every ranking accuracy, ranking the same held-out pairs by two predicted ratings, so
the two formulations are compared on one scale rather than accuracy against R^2.

Study B, folder 9.2, regresses the MEASURED human similarity of a pair, Richie and Bhatia's
1 to 7 ratings, on the expert similarity of that same pair. This is external validation
rather than a proxy: the target is a number humans produced for that pair, and the category
is constant within every pair, so category identity alone scores at chance and the shortcut
that haunts a per-concept target is closed by construction.

One feature grid serves both. J is the set feature, S the profile agreement of a registered
metric, z its count-matched null score, and GRID_CELLS names the six subsets of those roles
worth fitting. The grid is GENERATED per metric rather than written out, so registering a
second metric in PROFILE_METRICS and appending it to ACTIVE_METRICS instantiates every cell
of both studies on it and touches nothing else.

Profile features live on the BLOCK axis, sublayers summed within each transformer block, so
a profile is a genuine depth signature rather than a sublayer-alternation pattern. The flat
layer axis is kept for one robustness line only, written once at the whole-model scope.

The module computes every feature itself from the expert frame, so module 3 need not be
enabled and a new metric needs no module 3 schema change. The same helpers over the same
item list make the default metric's label-word columns identical to module 3's CSV by
construction.

Two things this module is careful about, both of which would otherwise make a comparison
meaningless.

First, in-sample fit cannot answer the question. Adding any predictor, including pure
noise, never decreases it, so the richer cell always "wins". Every headline number here is
out of sample, with whole categories held out.

Second, all cells are fitted on exactly the same rows WITHIN A STUDY. A z column is
undefined for words below MIN_PROFILE_EXPERTS, so a per-cell dropna would train the Jaccard
cell on more rows than the others and the scores would not be comparable. Each study drops
rows missing any of ITS OWN candidate features, once, before any model is fit. The scope of
that rule is the study, not the module: Study B never reads the label word and never reads
a centroid, so gating its rows on centroid columns would starve it of data for no reason,
and the two studies are never compared numerically against each other.
"""
import itertools
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.helpers import (save_dataframe, scope_out_dir, scope_summary_row,
                           build_layer_probability_matrix, to_block_axis,
                           expert_set_overlap_matrices, pair_similarity_vector,
                           pair_layer_profile_vectors, layer_profile_matrices,
                           count_matched_z, PROFILE_METRICS, DEFAULT_PROFILE_METRIC,
                           MIN_PROFILE_EXPERTS)
from utils.human_similarity import human_pair_lookup, human_noise_ceiling
from utils.plot_helpers import (COMPARISON_STYLE, COMPARISON_COLORS, comparison_legend,
                                annotate_barh_values, style_comparison_axis)

log = logging.getLogger(__name__)

# Headline metrics for the cross-scope sublayer_comparison table, one per study plus the
# deltas the module exists to measure.
# The summary row reports exactly one profile metric even when several are registered in
# PROFILE_METRICS and active in ACTIVE_METRICS, since a summary row carries one number per
# column. This constant, not "whichever metric happened to be fitted first", decides which
# one, and its name is folded into the affected labels below so a reader of
# sublayer_comparison.csv or its plot can tell which metric the numbers describe without
# opening the per-study CSV.
SUMMARY_METRIC = DEFAULT_PROFILE_METRIC

SUMMARY_LABELS = {
    "rank_acc_jaccard": "Ranking accuracy, Jaccard only",
    "rank_acc_full_grid": f"Ranking accuracy, Jaccard + profile + z ({SUMMARY_METRIC})",
    "rank_acc_delta_profile":
        f"Ranking accuracy gained from the profile features ({SUMMARY_METRIC})",
    "rank_acc_centroid_full_grid":
        f"Ranking accuracy, centroid reference, full cell ({SUMMARY_METRIC})",
    "pair_sim_rho_jaccard": "Pair similarity vs humans, Jaccard only",
    "pair_sim_rho_full_grid":
        f"Pair similarity vs humans, Jaccard + profile + z ({SUMMARY_METRIC})",
}

TARGET = "human_typicality"

# The two studies. A predicts which of two same-category concepts is more typical,
# B predicts the measured human similarity of a pair. Folder names follow the
# "<number>_<name>" convention of the module folders themselves.
STUDY_DIRS = {"A": "9.1_typicality_ranking", "B": "9.2_pair_similarity"}


def study_dir(out_dir, study: str):
    """Folder for one study's outputs inside ``out_dir``, created on demand."""
    path = out_dir / STUDY_DIRS[study]
    path.mkdir(parents=True, exist_ok=True)
    return path


# Below this many usable concepts Study A's leave-one-category-out ranking is not worth
# reporting, since the held-out categories become too thin to carry a paired comparison.
MIN_TRAIN_CONCEPTS = 40

# Study B builds its pairs from at least this many concepts. Below three there is no
# within-category pair to rate, and run_pair_similarity applies the real gate, MIN_PAIR_ROWS
# over the pairs humans actually rated, once its own table is built.
MIN_PAIR_CONCEPTS = 3

# Profile metrics the grid is instantiated on. One entry today. The follow-up project
# registers wasserstein, cosine, pearson, spearman, js_divergence and hellinger in
# PROFILE_METRICS and appends their names here, nothing else changes.
ACTIVE_METRICS = [DEFAULT_PROFILE_METRIC]

# The six grid cells, as the roles each uses. J is the set feature, S the profile
# agreement, z its count-matched null score. The grid is GENERATED per metric rather
# than written out, so both studies and any number of metrics share one vocabulary.
GRID_CELLS = {
    "jaccard": ("J",),
    "profile": ("S",),
    "profile_z": ("z",),
    "jaccard_profile": ("J", "S"),
    "jaccard_profile_z": ("J", "z"),
    "jaccard_profile_both": ("J", "S", "z"),
}

# How each role resolves to a design-frame column, per reference. The jaccard columns
# are fixed names, the profile columns are produced per metric by profile_columns.
REFERENCES = {
    "label_word": {"J": "jaccard_pct", "S": "S", "z": "z"},
    "member_centroid": {"J": "mean_jaccard_to_members", "S": "S_cen", "z": "z_cen"},
}


def profile_columns(metric: str) -> dict:
    """Design-frame column names for one metric, keyed by role."""
    return {"S": f"profile_{metric}", "z": f"profile_{metric}_z",
            "S_cen": f"centroid_profile_{metric}", "z_cen": f"centroid_profile_{metric}_z"}


def build_feature_grid(metrics: list) -> list:
    """One dict per (cell, metric): {"model", "metric", "roles"}. The jaccard cell
    uses no profile metric and appears once with metric "none"."""
    grid = [{"model": "jaccard", "metric": "none", "roles": GRID_CELLS["jaccard"]}]
    for metric in metrics:
        for model, roles in GRID_CELLS.items():
            if model == "jaccard":
                continue
            grid.append({"model": model, "metric": metric, "roles": roles})
    return grid


def nested_pairs(grid: list) -> list:
    """(reduced, full) cell pairs where full extends reduced by exactly one role,
    within one metric, the jaccard cell serving as the reduced model for both
    two-feature cells. These are the well-defined partial F comparisons."""
    pairs = []
    for full in grid:
        for reduced in grid:
            same_metric = reduced["metric"] in ("none", full["metric"])
            if (same_metric and set(reduced["roles"]) < set(full["roles"])
                    and len(full["roles"]) - len(reduced["roles"]) == 1):
                pairs.append((reduced, full))
    return pairs


def resolve_features(cell: dict, reference: str, metric_cols: dict) -> list:
    """Concrete design-frame columns for one grid cell under one reference."""
    mapping = REFERENCES[reference]
    return [mapping[role] if role == "J" else metric_cols[mapping[role]]
            for role in cell["roles"]]


# The two cells every figure marks out, the set feature alone and the fullest cell, which
# is the comparison the module exists to make.
HEADLINE_CELLS = ("jaccard", "jaccard_profile_both")


def study_a_columns(metrics: list) -> list:
    """
    Every candidate feature Study A can fit, over BOTH references, plus its target.

    Dropping the design frame over exactly this list, once, is what makes Study A's twelve
    cells comparable, since each is then fitted on identical rows. The list spans both
    references on purpose: the label-word arm and the member-centroid arm are compared
    against each other inside one figure, so a row usable by one and not the other would
    make that comparison rest on different concepts.
    """
    columns = ["jaccard_pct", "mean_jaccard_to_members", TARGET]
    for metric in metrics:
        columns += list(profile_columns(metric).values())
    return columns


def study_b_columns(metrics: list) -> list:
    """
    What a design-frame row must carry to enter Study B, which is far less than Study A needs.

    Study B reads only two things from this frame, the concept and its category (see
    build_pair_similarity_design, which pulls design["concept"] and design["category"] and
    nothing else, then recomputes every feature it fits as a symmetric PAIR quantity straight
    from expert_allocation_df). Its target, the measured human similarity of a pair, is not a
    column here either, it is joined onto the pair table. So this dropna on jaccard_pct is a
    DEFENSIVE NO-OP rather than a feature gate, not something Study B's features depend on:
    build_design_frame's own presence gate already guarantees a concept row here has a defined
    jaccard_pct, so it is never NaN by the time this list is applied, and the dropna can only
    ever remove zero rows in practice.

    Its identical-rows guarantee is NOT weakened by that. run_pair_similarity applies its own
    single shared dropna over the union of every pair column any of its six cells uses, on the
    pair table, before any cell is fit, which is the row set that actually matters and which
    check_pair_similarity_identical_rows.py verifies by counting the rows each cell sees.

    Why this list holds no profile column, which is a deliberate departure from the obvious
    reading. profile_<metric> on this frame is the CONCEPT-AGAINST-CATEGORY-LABEL-WORD
    agreement, and it is NaN whenever either word falls below MIN_PROFILE_EXPERTS, the label
    word included. Study B never looks at the label word, so gating on that column deletes
    concepts for a property of a word the study ignores, and it deletes them a whole category
    at a time. Measured on Qwen3 at AP 0.9, the label word "vehicles" holds one expert while
    all 22 of its concepts are perfectly usable, so gating on the label-word profile would
    drop the entire vehicles category, take the study from three categories to two, and
    run_pair_similarity, which needs at least three held-out categories, would return empty.
    Without that gate the same run keeps 71 concepts over three categories and 772 rated
    pairs, and the study reports normally. The metrics argument is accepted so the signature
    matches study_a_columns and so a future pair feature read from this frame, rather than
    recomputed, has an obvious place to be declared.
    """
    return ["jaccard_pct"]


# --- study A, typicality ranking -------------------------------------------------------
# Pairs whose two ratings differ by less than this are near-ties, where the human data is
# least reliable. They stay in the training set but accuracy is also reported on the
# clearer subset, since a model cannot be blamed for missing a distinction the ratings
# barely make.
RANK_MARGIN = 0.10

# Below this many usable within-category pairs the grouped accuracy is not worth reporting.
MIN_RANK_PAIRS = 200

# A forced binary choice, so an uninformative model lands here. Every accuracy in study A is
# read against it rather than against 0, which is unreachable.
CHANCE_ACCURACY = 0.5


def _model():
    """Standardised linear regression, Study B's estimator and nothing else.

    Its only two consumers are both inside Study B, run_pair_similarity's cross_val_predict
    over the pair table and the in-sample fits of _partial_f_test, so the units it ever sees
    are the 2,391 within-category PAIRS and never the concepts of Study A. It is not the
    regression baseline of Study A, which is a separate cross-validated ridge built inline in
    run_typicality_ranking.

    Unpenalised least squares because a Study B cell carries at most three predictors on
    those pairs, where there is nothing to regularise, and because an unpenalised fit keeps
    the coefficients readable as the change in rated similarity per standard deviation of a
    feature. The partial F test additionally requires an unpenalised fit, since its residual
    sums of squares and its degrees of freedom are only the quantities the F distribution
    describes when neither model is shrunk.
    """
    return make_pipeline(StandardScaler(), LinearRegression())


# The two axes any profile feature can be computed on. "block" sums the sublayers of a
# transformer block into one depth bin and is the analysis default, "layer" keeps the raw
# flat layer axis and is used only by the sensitivity refit.
PROFILE_AXES = ("block", "layer")


def axis_frame(expert_allocation_df, axis: str):
    """
    The expert frame re-keyed onto ``axis``, rejecting anything that is neither axis name.

    Every profile feature of this module reads depth off this frame, so a typo in an axis
    argument used to fall through to the flat axis silently and produce a full set of
    plausible numbers computed on the wrong axis. The two axes are not interchangeable, a
    single mistaken fallback moved a headline accuracy by 24 points during development, so an
    unrecognised name is an error rather than a default.
    """
    if axis not in PROFILE_AXES:
        raise ValueError(f"unknown profile axis {axis!r}, expected one of {PROFILE_AXES}")
    return to_block_axis(expert_allocation_df) if axis == "block" else expert_allocation_df


def build_design_frame(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame,
                       metrics: list, axis: str = "block") -> pd.DataFrame:
    """
    One row per categorized concept with every candidate feature, computed here from
    the expert rows rather than read from module 3, so the module needs no other
    module enabled and future metrics need no module 3 schema change. The same
    helpers over the same item list (every word, labels included) make the default
    metric's values identical to module 3's CSV by construction.

    Columns. concept and category name the pair, human_typicality is the rating being
    predicted, jaccard_pct is the share of the two words' combined expert sets that both
    hold, and per metric profile_<metric> is how closely their depth allocations agree
    while profile_<metric>_z restates that against real pairs of comparable expert
    counts, so 0 means "no more agreement than two arbitrary words of these sizes show"
    and the raw value's dependence on expert-set size is conditioned out. Read the z
    column when asking whether a pair carries signal.

    Rows are gated exactly as module 3 gates them, on both the concept and its category
    label word holding at least one expert. A word absent from the expert frame has an
    empty set, so its Jaccard is a 0/0 that the helper reports as 0.0 rather than as
    missing, and an ungated row would enter the frame as a silent zero. Keeping module
    3's condition is what makes this frame the same row set as its CSV.

    axis="block" is the analysis default. axis="layer" keeps the flat axis and
    exists only for the sensitivity line of the whole-model scope.

    Label-word features only. Centroid features are added by add_centroid_features,
    and the caller applies the single shared dropna over every candidate feature so
    all cells are fitted on identical rows.
    """
    if TARGET not in concept_metadata.columns:
        log.warning(f"  Module 9 skipped: no {TARGET} column in the metadata.")
        return pd.DataFrame()

    frame = axis_frame(expert_allocation_df, axis)
    # Jaccard reads neuron identity, which no axis change touches, so it takes the raw
    # frame. Only the profile matrices, which read depth, see the aggregated one.
    items = list(concept_metadata["concept"].unique())
    item_row = {item: i for i, item in enumerate(items)}
    _, jaccard, _ = expert_set_overlap_matrices(expert_allocation_df, items)
    with_experts = set(expert_allocation_df["concept"].unique())

    rows = []
    for _, meta in concept_metadata.dropna(subset=["category"]).iterrows():
        concept, category = meta["concept"], meta["category"]
        i, j = item_row.get(concept), item_row.get(category)
        if i is None or j is None:
            continue
        # Module 3's "if u_concept and u_category", see the docstring.
        if concept not in with_experts or category not in with_experts:
            continue
        rows.append({"concept": concept, "category": category,
                     "human_typicality": meta[TARGET],
                     "jaccard_pct": 100.0 * float(jaccard[i, j])})
    design = pd.DataFrame(rows)
    if design.empty:
        return design

    # The full item list, label words included, because the null's reference set is drawn
    # from the pairs of that list, so a shorter one would change the z of every pair.
    for metric in metrics:
        cols = profile_columns(metric)
        _, similarity, z = layer_profile_matrices(frame, items, metric=metric)
        design[cols["S"]] = [similarity[item_row[c], item_row[k]]
                             for c, k in zip(design["concept"], design["category"])]
        design[cols["z"]] = [z[item_row[c], item_row[k]]
                             for c, k in zip(design["concept"], design["category"])]
    return design.dropna(subset=[TARGET, "jaccard_pct"]).reset_index(drop=True)


def _partial_f_test(y: np.ndarray, X_reduced: np.ndarray, X_full: np.ndarray) -> tuple:
    """
    Partial F test for the terms present in X_full but not X_reduced, fitted in sample.

    Complements the cross-validated gain rather than replacing it. The F test asks whether
    the added feature explains significantly more variance in THIS sample, which is a
    question about the fitted coefficient, while the cross-validated gain asks whether it
    helps predict pairs the model has not seen. A feature can pass one and fail the
    other, and both readings are reported.

    Its only caller is Study B, whose unit is the within-category PAIR, so both n and the
    held-out units of the companion cross-validated gain are pairs and not concepts.
    """
    n = len(y)
    p_reduced, p_full = X_reduced.shape[1], X_full.shape[1]
    if p_full <= p_reduced or n <= p_full + 1:
        return np.nan, np.nan

    rss = []
    for X in (X_reduced, X_full):
        model = _model().fit(X, y)
        rss.append(float(np.sum((y - model.predict(X)) ** 2)))
    rss_reduced, rss_full = rss

    df_num = p_full - p_reduced
    df_den = n - p_full - 1
    if rss_full <= 0 or df_den <= 0:
        return np.nan, np.nan
    f_stat = ((rss_reduced - rss_full) / df_num) / (rss_full / df_den)
    return float(f_stat), float(stats.f.sf(f_stat, df_num, df_den))


def _ranker():
    """
    Logistic preference model, fitted WITHOUT an intercept.

    No intercept, and scaling without centring, because both would break the antisymmetry
    the task requires. Features are differences phi(a) - phi(b), so a constant term or a
    subtracted feature mean would let the model express a preference that does not flip
    when the pair is presented in the other order, which is meaningless for "which of these
    two is more typical".
    """
    return make_pipeline(StandardScaler(with_mean=False),
                         LogisticRegression(fit_intercept=False, C=1.0, max_iter=5000))


def add_centroid_features(design: pd.DataFrame, expert_allocation_df: pd.DataFrame,
                          metrics: list, axis: str = "block") -> pd.DataFrame:
    """
    Features measured against the OTHER MEMBERS of a concept's category rather than against
    the category label word: is a typical bird the one most like the word "bird", or the one
    most like the other birds? The second reading is the family-resemblance operationalization
    of typicality, and it is the reason this block exists.

    Every reference is LEAVE-ONE-OUT. Without it a concept in a twenty-member category
    supplies five percent of the object it is scored against, the inflation grows as the
    category shrinks, and part of the feature is self-similarity rather than resemblance.

    Columns. mean_jaccard_to_members is the average expert-set Jaccard to the other members,
    on the 0 to 100 scale of jaccard_pct, and it is this reference's counterpart of the
    label-word Jaccard, so the member-centroid arm carries its own baseline. Per metric,
    centroid_profile_<metric> is the registry agreement between the concept's layer profile
    and the average profile of the other members, and centroid_profile_<metric>_z restates
    that against count-matched entries of the same concept-by-category agreement matrix, so 0
    means "no closer to its own category than an arbitrary concept of this size gets to a
    centroid of this size" and the raw value's dependence on expert-set size is conditioned
    out. Read the z column when asking whether the resemblance carries signal.

    The z is the same construction as the pair z. Every (concept, category) agreement enters
    one flat population, leave-one-out applying to the own-category entry only, each entry is
    placed at (log concept expert count, log centroid pooled expert count), and
    count_matched_z standardizes it against its nearest neighbours in that space. It replaces
    the deleted foreign-centroid margin, a mean subtraction with neither count matching nor
    variance scaling. The earlier claim that a count-matched z cannot transfer to a centroid,
    because a centroid is an average of many profiles and has no expert count of its own, was
    a misreading of what the null controls. The binding sample size in a concept-to-centroid
    comparison is the CONCEPT's count, since the centroid pools every other member's experts
    and is never the noisier side, and the pooled count is what the second coordinate reports.

    The coordinate is NOT symmetrised. The pair null places a pair at (log min, log max)
    because a pair's two words are exchangeable. A concept and a centroid are different kinds
    of object and are never exchanged, so the two counts enter in a fixed order.

    A concept holding fewer than MIN_PROFILE_EXPERTS experts is left out of the member lists
    and gets NaN for both profile columns, matching how layer_profile_matrices gates the
    label-word features: a one-expert word has a spike, not a depth distribution, so it can
    neither be scored on depth nor be averaged into a centroid. One member list serves both
    roles of this reference, so mean_jaccard_to_members and the centroid columns always speak
    about the same set of other members.

    axis="block" is the analysis default, matching module 6's centroid and the label-word
    profile features, because a per-block profile is the interpretable depth signature.
    """
    frame = axis_frame(expert_allocation_df, axis)
    counts_matrix, profiles = build_layer_probability_matrix(frame)
    design = design.copy()
    columns = ["mean_jaccard_to_members"] + [
        column for metric in metrics
        for column in (profile_columns(metric)["S_cen"], profile_columns(metric)["z_cen"])]

    present = [concept for concept in design["concept"] if concept in profiles.index]
    if len(present) < 2:
        for column in columns:
            design[column] = np.nan
        return design

    # Jaccard reads neuron identity, which no axis change touches, so it takes the raw frame,
    # exactly as build_design_frame does for the label-word Jaccard.
    _, jaccard, _ = expert_set_overlap_matrices(expert_allocation_df, present)
    position = {concept: index for index, concept in enumerate(present)}
    concept_counts = counts_matrix.sum(axis=1)

    usable = {concept for concept in position
              if float(concept_counts.loc[concept]) >= MIN_PROFILE_EXPERTS}
    members = (design[design["concept"].isin(usable)]
               .groupby("category")["concept"].apply(list).to_dict())
    categories = sorted(members)

    # Whole-category centroids, computed once. Only the own-category entry needs the concept
    # removed, so every other entry reuses these.
    centroid_of = {category: (profiles.loc[names].to_numpy(dtype=float).mean(axis=0)[None, :],
                              float(concept_counts.loc[names].sum()))
                   for category, names in members.items()}

    # Mean Jaccard to the other members, the model counterpart of how the human typicality
    # proxy was built, each word's row of the similarity matrix averaged over its category.
    mean_jaccard = []
    for concept, category in zip(design["concept"], design["category"]):
        siblings = ([other for other in members.get(category, []) if other != concept]
                    if concept in position else [])
        mean_jaccard.append(100.0 * float(np.mean(
            [jaccard[position[concept], position[other]] for other in siblings]))
            if siblings else np.nan)
    design["mean_jaccard_to_members"] = mean_jaccard

    for metric in metrics:
        cols = profile_columns(metric)
        agreement_of = PROFILE_METRICS[metric]

        # The concept-by-category agreement matrix, flattened into one null population.
        # own holds each design row's own-category agreement and own_coord_rows holds where
        # that agreement sits inside all_values, or None when the row has no usable
        # own-category entry. Both lists are appended to exactly once per design row, at the
        # very end of the row body and never inside a branch, which is what keeps
        # len(own) == len(own_coord_rows) == len(design) true by construction.
        own, own_coord_rows, all_values, all_coords = [], [], [], []
        for concept, category in zip(design["concept"], design["category"]):
            own_value, own_row = np.nan, None
            if concept in usable:
                profile = profiles.loc[concept].to_numpy(dtype=float)[None, :]
                log_concept = np.log(float(concept_counts.loc[concept]))
                for other in categories:
                    if other == category:
                        names = [name for name in members[other] if name != concept]
                        if not names:
                            continue
                        centroid = profiles.loc[names].to_numpy(dtype=float).mean(axis=0)[None, :]
                        pooled = float(concept_counts.loc[names].sum())
                    else:
                        centroid, pooled = centroid_of[other]
                    all_values.append(float(agreement_of(profile, centroid)[0, 0]))
                    all_coords.append((log_concept, np.log(pooled)))
                    if other == category:
                        own_value, own_row = all_values[-1], len(all_values) - 1
            own.append(own_value)
            own_coord_rows.append(own_row)

        z_flat = count_matched_z(np.asarray(all_values, dtype=float),
                                 np.asarray(all_coords, dtype=float).reshape(-1, 2))
        design[cols["S_cen"]] = own
        design[cols["z_cen"]] = [z_flat[row] if row is not None else np.nan
                                 for row in own_coord_rows]
    return design


def within_category_pairs(design: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Every unordered pair of concepts sharing a category, as (pairs, gap).

    pairs is an (m, 2) array of row indices into ``design`` and gap is the typicality
    difference of the first minus the second. Pairs across categories are excluded because
    typicality is only defined relative to a category, and exact ties are excluded because
    they carry no direction to predict.
    """
    categories = design["category"].to_numpy()
    ratings = design[TARGET].to_numpy(dtype=float)
    pairs = [pair for category in np.unique(categories)
             for pair in itertools.combinations(np.flatnonzero(categories == category), 2)]
    if not pairs:
        return np.empty((0, 2), dtype=int), np.empty(0)
    pairs = np.asarray(pairs, dtype=int)
    gap = ratings[pairs[:, 0]] - ratings[pairs[:, 1]]
    keep = gap != 0
    return pairs[keep], gap[keep]


def evaluate_ranker(features: np.ndarray, design: pd.DataFrame,
                    pairs: np.ndarray, gap: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Grouped accuracy of the pairwise ranker on one feature set, its per-category detail, and
    the per-pair outcomes behind both.

    The third return value is what the diagnostic displays consume: one row per held-out pair
    with the two concepts, the human rating gap and whether the model ordered them correctly.
    Aggregate accuracy cannot show whether errors are scattered near-ties or a systematic
    inversion, and those call for different responses.

    Categories are held out whole. Because every pair lies inside one category, holding out
    a category removes every pair containing any of its concepts, so no concept is ever in
    both the training and the test half and the model must generalise to an unseen category.

    Each training pair enters twice, as (a, b) labelled 1 and (b, a) labelled 0, which
    together with the intercept-free model makes the learned preference exactly antisymmetric.

    Three accuracies are reported. accuracy is over all held-out pairs. accuracy_clear is
    over the subset whose ratings differ by at least RANK_MARGIN, where the human data
    actually distinguishes the two concepts. regression_accuracy applies the per-concept
    ridge of the per-concept regression baseline to exactly the same held-out pairs, ranking each pair by its two
    predicted ratings, so the ranking formulation and the regression formulation are
    compared on one scale rather than accuracy against R^2.
    """
    ratings = design[TARGET].to_numpy(dtype=float)
    categories = design["category"].to_numpy()
    n_groups = len(np.unique(categories))

    concept_names = design["concept"].to_numpy()
    hits, hits_clear, regression_hits, per_category, per_pair = [], [], [], [], []
    for train_rows, test_rows in GroupKFold(n_splits=n_groups).split(features, ratings, groups=categories):
        train_mask = np.isin(pairs[:, 0], train_rows) & np.isin(pairs[:, 1], train_rows)
        test_mask = np.isin(pairs[:, 0], test_rows) & np.isin(pairs[:, 1], test_rows)
        if not train_mask.any() or not test_mask.any():
            continue

        difference = features[pairs[train_mask, 0]] - features[pairs[train_mask, 1]]
        design_matrix = np.vstack([difference, -difference])
        labels = np.concatenate([(gap[train_mask] > 0).astype(int), (gap[train_mask] < 0).astype(int)])
        if len(np.unique(labels)) < 2:
            continue

        test_difference = features[pairs[test_mask, 0]] - features[pairs[test_mask, 1]]
        truth = (gap[test_mask] > 0).astype(int)
        correct = _ranker().fit(design_matrix, labels).predict(test_difference) == truth
        hits.append(correct)

        clear = np.abs(gap[test_mask]) >= RANK_MARGIN
        if clear.any():
            hits_clear.append(correct[clear])

        regression = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 30)))
        regression.fit(features[train_rows], ratings[train_rows])
        predicted = regression.predict(features)
        regression_correct = ((predicted[pairs[test_mask, 0]] - predicted[pairs[test_mask, 1]]) > 0) == truth
        regression_hits.append(regression_correct)

        per_category.append({"category": categories[test_rows][0],
                             "n_pairs": int(test_mask.sum()),
                             "accuracy": float(correct.mean()),
                             "regression_accuracy": float(regression_correct.mean())})

        per_pair.append(pd.DataFrame({
            "category": categories[test_rows][0],
            "concept_a": concept_names[pairs[test_mask, 0]],
            "concept_b": concept_names[pairs[test_mask, 1]],
            "gap": gap[test_mask],
            "correct": correct}))

    if not hits:
        return ({"accuracy": np.nan, "accuracy_clear": np.nan, "regression_accuracy": np.nan,
                 "accuracy_sd_across_categories": np.nan, "n_pairs": 0},
                pd.DataFrame(), pd.DataFrame())

    detail = pd.DataFrame(per_category)
    return ({"accuracy": float(np.concatenate(hits).mean()),
             "accuracy_clear": float(np.concatenate(hits_clear).mean()) if hits_clear else np.nan,
             "regression_accuracy": float(np.concatenate(regression_hits).mean()),
             "accuracy_sd_across_categories": float(detail["accuracy"].std(ddof=1)),
             "n_pairs": int(sum(len(h) for h in hits))},
            detail, pd.concat(per_pair, ignore_index=True))


def run_typicality_ranking(design: pd.DataFrame, metrics: list) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Study A: every grid cell run under both references through the grouped pairwise
    evaluation, "of these two same-category concepts, which is the more typical".

    Returns (results, per_category, per_pair). results carries one row per (reference, grid
    cell) with the accuracies, the columns actually fitted, and the comparison against that
    reference's own Jaccard cell. per_category carries the held-out accuracy per category with
    a reference column, and per_pair maps (reference, metric, model) to the per-pair outcomes
    the diagnostic displays consume. The metric belongs in that key for the same reason it
    belongs in the paired t below, a cell name alone is not unique once a second metric is
    registered, and keying on the name would silently overwrite one metric with the other so
    the diagnostics would quietly show only one of them. The jaccard cell carries no profile
    metric and keys on "none", which is what _pair_key resolves.

    Each reference is a self-contained arm. The jaccard cell is fitted twice, on jaccard_pct
    under label_word and on mean_jaccard_to_members under member_centroid, so a profile cell is
    always compared against the set feature of ITS OWN reference and no gain is credited to the
    profile that really belongs to the change of reference.

    The paired t against that baseline is taken over per-category accuracies, which is the
    honest unit here. Thousands of pairs built from a couple of hundred concepts are not
    independent observations, while the eight held-out categories are close to independent
    replicates of the same comparison.
    """
    pairs, gap = within_category_pairs(design)
    if len(pairs) < MIN_RANK_PAIRS:
        log.warning(f"  Skipping Study A: {len(pairs)} within-category pairs, "
                    f"need >= {MIN_RANK_PAIRS}.")
        return pd.DataFrame(), pd.DataFrame(), {}

    grid = build_feature_grid(metrics)
    rows, details, per_pair = [], [], {}
    for reference in REFERENCES:
        for cell in grid:
            metric_cols = profile_columns(cell["metric"]) if cell["metric"] != "none" else {}
            feature_names = resolve_features(cell, reference, metric_cols)
            features = design[feature_names].to_numpy(dtype=float)
            scores, detail, pair_outcomes = evaluate_ranker(features, design, pairs, gap)
            rows.append({"model": cell["model"], "metric": cell["metric"],
                         "reference": reference, "features": " + ".join(feature_names),
                         "n_concepts": len(design), **scores})
            if not detail.empty:
                details.append(detail.assign(model=cell["model"], metric=cell["metric"],
                                             reference=reference))
            if not pair_outcomes.empty:
                per_pair[(reference, cell["metric"], cell["model"])] = pair_outcomes

    results = pd.DataFrame(rows)
    per_category = pd.concat(details, ignore_index=True) if details else pd.DataFrame()

    # Every cell is compared against the Jaccard cell of its own reference on the same
    # held-out categories, keyed on (reference, metric, model) rather than on the model name
    # alone, so a second registered metric cannot silently pool two cells that share a name.
    if not per_category.empty:
        for reference in REFERENCES:
            arm = per_category[per_category.reference == reference]
            if "jaccard" not in set(arm.model):
                continue
            baseline = arm[arm.model == "jaccard"].set_index("category")["accuracy"]
            cells = arm.loc[arm.model != "jaccard", ["model", "metric"]].drop_duplicates()
            for model, metric in zip(cells["model"], cells["metric"]):
                richer = (arm[(arm.model == model) & (arm.metric == metric)]
                          .set_index("category")["accuracy"])
                shared = baseline.index.intersection(richer.index)
                if len(shared) < 3:
                    continue
                where = ((results.model == model) & (results.metric == metric)
                         & (results.reference == reference))
                delta = richer[shared] - baseline[shared]
                results.loc[where, "delta_accuracy_vs_jaccard"] = delta.mean()
                results.loc[where, "categories_improved"] = int((delta > 0).sum())
                results.loc[where, "categories_tested"] = len(shared)
                results.loc[where, "paired_t_p"] = float(
                    stats.ttest_rel(richer[shared], baseline[shared]).pvalue)
    return results, per_category, per_pair


def _grid_order(frame: pd.DataFrame) -> list:
    """
    The (model, metric) cells present in ``frame``, in the fixed GRID_CELLS order.

    Every figure of both studies reads its row order from here, so the bars sit in the same
    sequence whatever metrics are registered and two figures can be laid side by side. The
    order is over PAIRS rather than over model names because a cell name alone stops being a
    unique row as soon as a second metric is active, and indexing a results table on the name
    would then silently keep whichever of the two rows pandas met first.
    """
    present = list(dict.fromkeys(zip(frame["model"], frame["metric"])))
    return [key for model in GRID_CELLS for key in present if key[0] == model]


def _grid_labels(order: list) -> list:
    """
    Tick labels for the cells of _grid_order.

    The metric is named only when more than one is active, so the single-metric figure stays
    readable while a multi-metric one stays unambiguous.
    """
    if len(ACTIVE_METRICS) < 2:
        return [model for model, _ in order]
    return [model if metric == "none" else f"{model} [{metric}]" for model, metric in order]


def _pair_key(reference: str, model: str, metric: str) -> tuple:
    """
    Key into run_typicality_ranking's per_pair map, which is (reference, metric, model).

    The jaccard cell carries no profile metric and is stored under "none", whatever metric
    the figure happens to be displaying, so every selector goes through this helper rather
    than assembling the tuple by hand.
    """
    return (reference, "none" if model == "jaccard" else metric, model)


def _display_metric() -> str:
    """The profile metric the diagnostic figures show when several are registered.

    The figures that compare exactly two cells, the dumbbell, the gap curve and the pair
    error grid, can only draw one metric at a time, and the rest are read from the CSVs.
    They draw SUMMARY_METRIC, the same metric the cross-scope summary row reports, so a
    figure and the summary row can never describe different metrics. Returning
    ACTIVE_METRICS[0] instead would have made the two agree only while the summary metric
    happened to sit first in the list, and prepending a metric would have split them
    silently. The fallback covers the case where SUMMARY_METRIC is not active at all, which
    leaves the figures on the first active metric rather than on nothing.

    The bar figures show every cell of every metric and do not use this.
    """
    if SUMMARY_METRIC in ACTIVE_METRICS:
        return SUMMARY_METRIC
    return ACTIVE_METRICS[0] if ACTIVE_METRICS else "none"


def _metric_suffix() -> str:
    """", <metric>" for a figure title when more than one metric is active, else "".

    A single-metric run has nothing to disambiguate, so the titles stay as they read today.
    As soon as a second metric is registered the drawn metric has to be visible in the
    figure itself, since the figure shows one of several and the file name does not say
    which.
    """
    return f", {_display_metric()}" if len(ACTIVE_METRICS) > 1 else ""


def plot_ranker_comparison(results: pd.DataFrame, per_category: pd.DataFrame, out_dir) -> None:
    """
    Ranking accuracy of every grid cell, one stacked panel per reference.

    Drawn in the shared comparison grammar of utils.plot_helpers, the same one Study B uses,
    so the two figures can be read against each other without re-learning the encoding.
    Colour marks the SERIES here rather than the cell, since each row carries two scores, so
    the headline cells are marked by bold tick labels instead.

    Two series per row. The pairwise ranker is the study's own model, and the regression on
    the same held-out pairs is the per-concept formulation applied to exactly those pairs, so
    the two formulations are compared on one scale rather than accuracy against R^2.

    Accuracy is printed beside each ranker bar together with its distance from chance in
    percentage points, because 0.54 and 0.67 look far more alike on a 0.4 to 0.8 axis than 4
    points above chance and 17 points above chance actually are. The per-category accuracies
    behind each mean are overlaid as light dots on the same row, since a mean over eight
    held-out categories can rest on one of them and a bar alone cannot show that.

    The two panels are the two references. Each is a self-contained arm carrying its own
    Jaccard baseline, so a profile cell is read against the set feature of ITS OWN reference
    and no gain is credited to the profile that really belongs to the change of reference.
    """
    if results.empty:
        return
    style, colors = COMPARISON_STYLE, COMPARISON_COLORS
    references = [reference for reference in REFERENCES if reference in set(results.reference)]
    if not references:
        return
    titles = {"label_word": "Label-word reference", "member_centroid": "Member-centroid reference"}

    fig, axes = plt.subplots(len(references), 1,
                             figsize=(style["figsize"][0], 4.6 * len(references)), squeeze=False)
    width = style["grouped_bar_height"]
    has_reference_column = not per_category.empty and "reference" in per_category.columns

    for axis, reference in zip(axes.ravel(), references):
        arm = results[results.reference == reference]
        order = _grid_order(arm)
        labels = _grid_labels(order)
        frame = arm.set_index(["model", "metric"]).reindex(order)
        y_pos = np.arange(len(order))
        accuracy = frame["accuracy"].astype(float)

        axis.barh(y_pos + width / 2, accuracy, height=width,
                  color=colors["accent"], label="Pairwise ranker")
        axis.barh(y_pos - width / 2, frame["regression_accuracy"].astype(float), height=width,
                  color=colors["base"], label="Regression on the same pairs")
        axis.axvline(CHANCE_ACCURACY, color=colors["reference"], linewidth=1.8, linestyle="--",
                     label=f"Chance ({CHANCE_ACCURACY:.2f})")

        spread = per_category[per_category.reference == reference] if has_reference_column \
            else pd.DataFrame()
        for row, (model, metric) in enumerate(order):
            if spread.empty:
                break
            points = spread.loc[(spread.model == model) & (spread.metric == metric), "accuracy"]
            if points.empty:
                continue
            axis.scatter(points, np.full(len(points), row + width / 2), s=14, zorder=3,
                         color=colors["zero"], alpha=0.55, linewidth=0,
                         label="Per held-out category" if row == 0 else None)

        upper = float(np.nanmax(accuracy)) if accuracy.notna().any() else CHANCE_ACCURACY
        axis.set_xlim(0.3, max(0.8, upper) + 0.18)
        annotate_barh_values(
            axis, accuracy,
            lambda value: f"{value:.3f}  ({100 * (value - CHANCE_ACCURACY):+.1f} pp vs chance)")
        bold = [label for label, (model, _) in zip(labels, order) if model in HEADLINE_CELLS]
        style_comparison_axis(axis, labels, titles.get(reference, reference),
                              "Accuracy on held-out categories", bold=bold)
        comparison_legend(axis, ncol=4)

    fig.suptitle("Which of two same-category concepts is more typical", fontsize=14)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    plt.savefig(out_dir / "typicality_ranker_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pair_similarity(results: pd.DataFrame, per_category: pd.DataFrame, out_dir) -> None:
    """
    Study B: agreement with the human pair ratings, and where it comes from.

    Left, every cell of the generated grid against the mean per-category Spearman, with the
    human noise ceiling drawn as a reference line. The ceiling is what makes the bar heights
    readable, an agreement of 0.58 is not 58 percent of the way to a perfect model, it is 65
    percent of the way to what the raters themselves manage, and no model can pass that line.

    Right, the same comparison resolved per held-out category, the Jaccard cell against the
    fullest cell. This panel exists because gains from the layer profile elsewhere in this
    module turned out to rest on one or two categories, so a mean is not trustworthy here
    until the spread behind it is shown. Each category is drawn against its OWN ceiling,
    since those differ (0.836 for birds against 0.935 for vehicles) and a category cannot be
    blamed for a ceiling its raters set.

    Both panels index rows on (model, metric) rather than on the model name, because the same
    six cell names recur once per registered metric and a name-keyed index or pivot would
    collapse them onto each other.
    """
    if results.empty:
        return
    style, colors = COMPARISON_STYLE, COMPARISON_COLORS
    order = _grid_order(results)
    labels = _grid_labels(order)
    frame = results.set_index(["model", "metric"]).reindex(order)
    ceiling = float(frame["noise_ceiling_mean"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=style["figsize"])

    y_pos = np.arange(len(order))
    highlight = {"jaccard": colors["base"], "jaccard_profile_both": colors["accent"]}
    values = frame["mean_category_spearman"].astype(float)
    axes[0].barh(y_pos, values, height=style["bar_height"],
                 color=[highlight.get(model, colors["muted"]) for model, _ in order])
    axes[0].axvline(ceiling, color=colors["reference"], linewidth=1.8, linestyle="--",
                    label=f"Human noise ceiling ({ceiling:.3f})")
    axes[0].axvline(0.0, color=colors["zero"], linewidth=1.0, linestyle=":")
    axes[0].set_xlim(min(0.0, float(values.min()) - 0.05), ceiling + 0.10)
    annotate_barh_values(axes[0], values,
                         lambda value: f"{value:.3f}  ({value / ceiling:.0%} of ceiling)")
    bold = [label for label, (model, _) in zip(labels, order) if model in HEADLINE_CELLS]
    style_comparison_axis(axes[0], labels, "Agreement with human pair similarity",
                          "Spearman with human ratings, mean over categories", bold=bold)
    comparison_legend(axes[0], ncol=2)

    # Selected on the (model, metric) PAIR rather than on the first row of the grid order
    # carrying the model name. With a second metric registered there is one such row per
    # metric, and taking the first would draw whichever metric the grid happened to emit
    # first while the summary row reported _display_metric's choice.
    base_key = next((key for key in order if key[0] == "jaccard"), None)
    rich_key = next((key for key in order
                     if key == ("jaccard_profile_both", _display_metric())), None)
    detail = pd.DataFrame()
    if base_key is not None and rich_key is not None and not per_category.empty:
        wanted = {base_key, rich_key}
        detail = per_category[[key in wanted for key
                               in zip(per_category.model, per_category.metric)]]
    if not detail.empty:
        # Pivoted on the (model, metric) pair joined into one label rather than on the model
        # name, which stops being unique the moment a second metric is registered.
        cell_of = {base_key: "base", rich_key: "rich"}
        pivot = (detail.assign(cell=[cell_of[key] for key
                                     in zip(detail.model, detail.metric)])
                 .pivot(index="category", columns="cell", values="rho")
                 .reindex(columns=["base", "rich"]).dropna().sort_values("rich"))
        base_label, rich_label = _grid_labels([base_key, rich_key])
        y = np.arange(len(pivot))
        width = style["grouped_bar_height"]
        axes[1].barh(y + width / 2, pivot["rich"], height=width, color=colors["accent"],
                     label=rich_label.replace("_", " "))
        axes[1].barh(y - width / 2, pivot["base"], height=width, color=colors["base"],
                     label=base_label.replace("_", " "))
        # Each category's own ceiling, as a tick, since they differ enough to matter.
        ceilings = detail.drop_duplicates("category").set_index("category")["noise_ceiling"]
        for index, category in enumerate(pivot.index):
            axes[1].plot([ceilings[category]], [index], marker="|", markersize=16,
                         color=colors["reference"], markeredgewidth=2.2,
                         label="Category noise ceiling" if index == 0 else None)
        axes[1].set_xlim(0.0, 1.0)
        style_comparison_axis(axes[1], list(pivot.index), "Per held-out category",
                              "Spearman with human ratings, held-out category")
        comparison_legend(axes[1], ncol=3)

    plt.tight_layout()
    plt.savefig(out_dir / "pair_similarity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _cell(frame: pd.DataFrame, model: str, column: str, **keys) -> float:
    """One cell of a results table filtered on model plus any extra key columns."""
    if frame.empty or column not in frame.columns:
        return np.nan
    mask = frame.model == model
    for key, value in keys.items():
        mask &= frame[key] == value
    row = frame.loc[mask]
    return float(row[column].iloc[0]) if not row.empty and pd.notna(row[column].iloc[0]) else np.nan


def _summary(rank_results: pd.DataFrame, pair_results: pd.DataFrame) -> dict:
    """This module's row of the cross-scope comparison table, one headline per study
    plus the deltas the module exists to measure.

    A summary row carries one number per column, so with several profile metrics active the
    "jaccard_profile_both" model name alone is ambiguous, one row exists per (model, metric)
    pair. Every profile-cell lookup below therefore pins metric=SUMMARY_METRIC explicitly
    rather than taking whichever row happens to come first, which used to be a silent,
    unnamed choice of the first registered metric once a second one was added. The jaccard
    cell needs no such key, its rows carry metric="none" by construction (build_feature_grid)
    and there is exactly one of them. SUMMARY_LABELS folds SUMMARY_METRIC into the text of the
    affected labels so the reported metric is visible on the plot and not just in this
    function. With today's single active metric SUMMARY_METRIC is the only metric that could
    ever be selected, so this is a correctness fix with no numeric effect on the current run.
    The per-metric detail for every registered metric still lives in each study's own
    comparison CSV. The comparison columns of Study A (delta_accuracy_vs_jaccard and its
    companions) are created only when at least one category survived the paired test, so
    _cell checks the column exists before reading it and an all-empty run yields NaN rather
    than a KeyError.
    """
    return {
        "rank_acc_jaccard": _cell(rank_results, "jaccard", "accuracy",
                                  reference="label_word", metric="none"),
        "rank_acc_full_grid": _cell(rank_results, "jaccard_profile_both", "accuracy",
                                    reference="label_word", metric=SUMMARY_METRIC),
        "rank_acc_delta_profile": _cell(rank_results, "jaccard_profile_both",
                                        "delta_accuracy_vs_jaccard", reference="label_word",
                                        metric=SUMMARY_METRIC),
        "rank_acc_centroid_full_grid": _cell(rank_results, "jaccard_profile_both", "accuracy",
                                             reference="member_centroid",
                                             metric=SUMMARY_METRIC),
        "pair_sim_rho_jaccard": _cell(pair_results, "jaccard", "mean_category_spearman",
                                      metric="none"),
        "pair_sim_rho_full_grid": _cell(pair_results, "jaccard_profile_both",
                                        "mean_category_spearman", metric=SUMMARY_METRIC),
    }


# Bin edges for the accuracy-against-gap curve. The last edge exceeds the 0 to 1 rating
# range so the top bin is closed.
GAP_BINS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.01]


def plot_category_dumbbell(per_category: pd.DataFrame, out_dir) -> None:
    """
    Per held-out category, the Jaccard cell's accuracy joined to the fullest cell's, one
    panel per reference.

    The pooled mean hides that the profile features help enormously in one category and hurt
    badly in another. A dumbbell shows direction and size per category at a glance, which is
    the honest way to present a mean that can rest on one or two categories.

    Each panel is one reference and joins that reference's OWN two cells, so the arrow length
    reads as the gain from the profile features within that arm and never mixes in the
    difference between the label word and the member centroid.

    Rows are selected on the (model, metric) pair. With several metrics registered only
    _display_metric's choice is drawn, since a dumbbell compares exactly two cells, and the
    rest are read from ranker_per_category.csv. That metric is named in the legend and the
    title whenever more than one is active.
    """
    if per_category.empty or "reference" not in per_category.columns:
        return
    metric = _display_metric()
    references = [reference for reference in REFERENCES
                  if reference in set(per_category.reference)]
    if not references:
        return
    titles = {"label_word": "Label-word reference", "member_centroid": "Member-centroid reference"}

    panels = []
    for reference in references:
        arm = per_category[per_category.reference == reference]
        base = arm[(arm.model == "jaccard")].set_index("category")["accuracy"]
        rich = arm[(arm.model == "jaccard_profile_both")
                   & (arm.metric == metric)].set_index("category")["accuracy"]
        shared = sorted(set(base.index) & set(rich.index), key=lambda c: rich[c] - base[c])
        if shared:
            panels.append((reference, base, rich, shared))
    if not panels:
        return

    height = max(len(shared) for _, _, _, shared in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 0.5 * height + 2.8),
                             squeeze=False)
    for axis, (reference, base, rich, shared) in zip(axes.ravel(), panels):
        for row, category in enumerate(shared):
            improved = rich[category] >= base[category]
            axis.plot([base[category], rich[category]], [row, row],
                      color="tab:green" if improved else "tab:red", linewidth=2, zorder=1)
            axis.scatter([base[category]], [row], color="grey", s=55, zorder=2,
                         label="jaccard" if row == 0 else "")
            axis.scatter([rich[category]], [row], color="black", s=55, zorder=3,
                         label=f"jaccard + profile + z{_metric_suffix()}" if row == 0 else "")
            axis.annotate(f"{100 * (rich[category] - base[category]):+.1f}",
                          (max(base[category], rich[category]) + 0.005, row),
                          va="center", fontsize=9)
        axis.axvline(CHANCE_ACCURACY, color="black", linestyle=":", linewidth=1)
        axis.set_yticks(range(len(shared)))
        axis.set_yticklabels(shared)
        axis.set_xlabel("Pairwise ranking accuracy (held-out category)")
        axis.set_title(titles.get(reference, reference))
        axis.margins(x=0.12)
        # Below the axes rather than inside them, in the grammar comparison_legend uses, since
        # a legend box in any corner sits on top of one of the category rows.
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False,
                    fontsize=9)
    fig.suptitle(f"Where the profile features help and where they hurt{_metric_suffix()}",
                 fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_dir / "ranker_category_dumbbell.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_gap(per_pair: dict, out_dir) -> None:
    """
    Accuracy against how decisively the humans separated the two concepts of a pair.

    Generalises the single RANK_MARGIN cutoff into a curve. A model that has learned the
    construct is right where the humans were decisive and near chance where they were not, so
    a flat line here would mean the accuracy is not tracking the rating at all.

    Why bin at all. Accuracy is a property of a SET of pairs, since a single pair is either
    right or wrong, so the curve needs groups of pairs wide enough to estimate a proportion
    from. The edges are unequal on purpose, narrow where the pairs crowd together near zero
    and wide out in the sparse tail, which keeps each point resting on a few hundred pairs
    instead of letting the tail points swing on a handful. The count is therefore part of
    reading the curve rather than decoration, so it is drawn against the bin's own boundary,
    with the shaded bands and the boundary rules making the grouping explicit.

    Two curves, the Jaccard cell and the fullest cell, both on the label-word reference,
    selected through _pair_key because per_pair is keyed on (reference, metric, model), with
    the profile curve's legend entry naming its metric when more than one is active. Every
    cell of every reference would be twenty-two overlapping lines, which shows nothing, and
    the centroid arm's own curves are read from its accuracy column instead.
    """
    reference = "label_word"
    metric = _display_metric()
    shown = {}
    for model in HEADLINE_CELLS:
        frame = per_pair.get(_pair_key(reference, model, metric))
        if frame is not None and not frame.empty:
            suffix = "" if model == "jaccard" else _metric_suffix()
            shown[f'{model.replace("_", " ")}{suffix}'] = frame
    if not shown:
        return

    fig, axis = plt.subplots(figsize=(10.5, 5.6))

    # Alternating bands so the reader sees which pairs were pooled into each marker. The
    # legend entry hangs off a shaded band rather than a faint one so its swatch is visible.
    for index, (left, right) in enumerate(zip(GAP_BINS[:-1], GAP_BINS[1:])):
        axis.axvspan(left, right, color="0.5",
                     alpha=0.10 if index % 2 else 0.03, linewidth=0, zorder=0,
                     label="Gap bin, marker at its midpoint, n at its upper edge"
                           if index == 1 else "")
    for edge in GAP_BINS:
        axis.axvline(edge, color="0.65", linewidth=0.7, zorder=1)

    for name, frame in shown.items():
        binned = pd.cut(frame["gap"].abs(), GAP_BINS, right=False)
        grouped = frame.groupby(binned, observed=True)["correct"].agg(["mean", "size"])
        centres = [interval.mid for interval in grouped.index]
        axis.plot(centres, grouped["mean"], marker="o", label=name, zorder=3)

    # Pair counts sit on the bin's UPPER boundary, so "n" reads as "this many pairs fall in
    # the band that closes here", rather than floating at an arbitrary height mid-band.
    counts = next(iter(shown.values()))
    binned = pd.cut(counts["gap"].abs(), GAP_BINS, right=False)
    for interval, size in binned.value_counts().sort_index().items():
        axis.annotate(f"n={int(size)}", (interval.right, 0.985), fontsize=8, rotation=90,
                      ha="right", va="top", color="0.35", zorder=4)

    axis.axhline(0.5, color="black", linestyle=":", linewidth=1, zorder=2,
                 label="Chance (0.5)")
    axis.axvline(RANK_MARGIN, color="tab:red", linestyle="--", linewidth=1.2, zorder=2,
                 label=f"RANK_MARGIN = {RANK_MARGIN}, the accuracy_clear cutoff")

    axis.set_xlabel(
        "Gap in human typicality between the two concepts of a pair,  | τ(a) − τ(b) |\n"
        "τ compacts a concept's pairwise similarity judgments into a single typicality "
        "score:\naveraged over raters, then over its category co-members, then min-max "
        "scaled\nwithin the category, so 0 is its least and 1 its most typical member",
        fontsize=9)
    axis.set_ylabel("Pairwise ranking accuracy")
    axis.set_xlim(GAP_BINS[0], GAP_BINS[-1])
    axis.set_ylim(0.4, 1.0)
    axis.set_xticks(GAP_BINS)
    axis.tick_params(labelsize=9)
    axis.set_title("Accuracy rises with how decisively humans separated the pair")
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_dir / "ranker_accuracy_by_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pair_errors(per_pair: dict, design: pd.DataFrame, out_dir,
                     model: str = "jaccard_profile_both", reference: str = "label_word") -> None:
    """
    Per category, a concept-by-concept grid of which pairs were ranked wrongly.

    Concepts are ordered by human typicality, so a model that gets the ordering roughly right
    shows scattered errors near the diagonal, where neighbouring concepts are hardest to
    separate. Contiguous blocks away from the diagonal mean something is systematically
    inverted, which is the shape to look for in a category scoring below chance.

    Drawn for the fullest cell on the label-word reference by default, which is the model the
    module's headline claim rests on, so the errors shown are the errors of the claim. The
    selection goes through _pair_key since per_pair is keyed on (reference, metric, model),
    and the drawn metric is named in the title when more than one is active.
    """
    frame = per_pair.get(_pair_key(reference, model, _display_metric()))
    if frame is None or frame.empty:
        return
    categories = sorted(frame["category"].unique())
    columns = min(4, len(categories))
    rows = int(np.ceil(len(categories) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 4.6 * rows), squeeze=False)

    for axis, category in zip(axes.ravel(), categories):
        members = (design[design["category"] == category]
                   .sort_values(TARGET, ascending=False)["concept"].tolist())
        position = {concept: index for index, concept in enumerate(members)}
        grid = np.full((len(members), len(members)), np.nan)
        subset = frame[frame["category"] == category]
        for concept_a, concept_b, correct in zip(subset.concept_a, subset.concept_b, subset.correct):
            if concept_a in position and concept_b in position:
                first, second = sorted((position[concept_a], position[concept_b]))
                grid[second, first] = 1.0 if correct else 0.0
        axis.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"{category}  ({100 * subset.correct.mean():.0f}% correct)", fontsize=10)
        axis.set_xticks(range(len(members)))
        axis.set_xticklabels(members, rotation=90, fontsize=5)
        axis.set_yticks(range(len(members)))
        axis.set_yticklabels(members, fontsize=5)
    for axis in axes.ravel()[len(categories):]:
        axis.set_visible(False)
    suffix = "" if model == "jaccard" else _metric_suffix()
    fig.suptitle(f"Misranked pairs, {model.replace('_', ' ')}{suffix} on the "
                 f"{reference.replace('_', ' ')} reference, concepts ordered by human "
                 f"typicality (green correct, red wrong)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "ranker_pair_errors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- study B, human pair similarity as the target -------------------------------------
# Study B is fit on the same generated feature grid as Study A (build_feature_grid), rather
# than on a hand-written feature-set dictionary. These are PAIR quantities (a against b), not
# the concept-to-parent quantities the ranker differences, because the target is a value
# attached to the pair and is symmetric in its two members. Predicting a symmetric target
# from antisymmetric features would be incoherent.

# Below this many rated pairs a grouped correlation is not worth reporting.
MIN_PAIR_ROWS = 200


def build_pair_similarity_design(design: pd.DataFrame, expert_allocation_df: pd.DataFrame,
                                 metrics: list, axis: str = "block") -> pd.DataFrame:
    """
    One row per within-category pair that humans actually rated.

    Columns: category, word_a, word_b (canonical order), human_similarity (1 to 7),
    jaccard_pct, and per metric profile_<metric> and profile_<metric>_z, the pair features
    of every active metric.

    Unlike Study A, which predicts a DIRECTION from an antisymmetric difference of
    per-concept features, this predicts a VALUE from symmetric pair features, so the
    quantities are module 5's pairwise matrices rather than module 3's concept-to-parent
    table. Predicting a symmetric target from antisymmetric features would be incoherent.

    axis="block" is the analysis default and only affects the profile columns, matching
    Study A, since Jaccard reads neuron identity and no axis change touches it.
    """
    lookup = human_pair_lookup()
    available = set(expert_allocation_df["concept"])
    concepts = [c for c in design["concept"] if c in available]
    if len(concepts) < 3:
        return pd.DataFrame()

    jaccard_vector = pair_similarity_vector(expert_allocation_df, concepts)
    frame = axis_frame(expert_allocation_df, axis)
    vectors = {}
    for metric in metrics:
        cols = profile_columns(metric)
        similarity_vector, z_vector = pair_layer_profile_vectors(frame, concepts, metric=metric)
        vectors[cols["S"]] = similarity_vector
        vectors[cols["z"]] = z_vector
    category_of = dict(zip(design["concept"], design["category"]))

    rows, cursor = [], 0
    for first in range(len(concepts)):
        for second in range(first + 1, len(concepts)):
            word_a, word_b = concepts[first], concepts[second]
            key = (word_a, word_b) if word_a < word_b else (word_b, word_a)
            if category_of.get(word_a) == category_of.get(word_b) and key in lookup:
                row = {"category": category_of[word_a], "word_a": key[0], "word_b": key[1],
                       "human_similarity": lookup[key],
                       "jaccard_pct": 100.0 * jaccard_vector[cursor]}
                row.update({name: vec[cursor] for name, vec in vectors.items()})
                rows.append(row)
            cursor += 1
    return pd.DataFrame(rows)


def run_pair_similarity(design: pd.DataFrame, expert_allocation_df: pd.DataFrame,
                        metrics: list, axis: str = "block") -> tuple:
    """
    Predict the human-rated similarity of a pair from the expert similarity of that pair,
    over every cell of the generated feature grid (build_feature_grid).

    This is the question the supervisor asked directly, and it replaces predicting the
    per-item typicality proxy, which was itself derived by averaging each word's row of this
    same human similarity matrix.

    Categories are held out whole, so the model must generalise to a category it never saw.
    The target admits NO category-identity shortcut: the category is constant within every
    pair, so a model given only category identity scores at chance. That is the trap
    a per-concept typicality target had to guard against, and here it is closed by construction.

    Every grid cell is fit on the SAME rows. fitted_columns is built in a first pass over
    the whole grid before any row is dropped, then the pair table is dropped once over the
    union of every column any cell uses plus the target, and MIN_PAIR_ROWS is checked on
    that post-dropna table before any model is fit. profile_<metric>_z is NaN below
    MIN_PROFILE_EXPERTS, so a per-cell dropna would silently give the jaccard cell more
    training rows than the profile cells and the resulting scores would not be comparable,
    the same identical-rows requirement study_a_columns and study_b_columns document.

    Returns (results, per_category), both empty when too few pairs survive. results carries
    one row per grid cell keyed by (model, metric), with group_cv_spearman the pooled
    out-of-fold Spearman and mean_category_spearman the mean of the per-category ones (see
    rho_over_ceiling below for why the two are not interchangeable). per_category carries the
    per-category rho, its noise ceiling, and rho_over_ceiling for every (model, metric,
    category) triple with at least 10 pairs.
    """
    table = build_pair_similarity_design(design, expert_allocation_df, metrics, axis=axis)
    if table.empty:
        return pd.DataFrame(), pd.DataFrame()

    grid = build_feature_grid(metrics)
    fitted_columns = {}
    for cell in grid:
        cols = profile_columns(cell["metric"]) if cell["metric"] != "none" else {}
        columns = ["jaccard_pct" if role == "J" else cols[{"S": "S", "z": "z"}[role]]
                   for role in cell["roles"]]
        fitted_columns[(cell["model"], cell["metric"])] = columns

    # The single shared dropna, over every column any cell uses plus the target, applied once
    # before any cell is fit so all six cells train on exactly the same rows. A per-cell
    # dropna would give the jaccard-only cell more rows than the profile cells, since
    # profile_<metric>_z is undefined below MIN_PROFILE_EXPERTS, and the scores would then be
    # incomparable.
    all_columns = sorted({c for cols in fitted_columns.values() for c in cols}
                         | {"human_similarity"})
    table = table.dropna(subset=all_columns).reset_index(drop=True)
    if len(table) < MIN_PAIR_ROWS or table["category"].nunique() < 3:
        return pd.DataFrame(), pd.DataFrame()

    ceilings = human_noise_ceiling()
    target = table["human_similarity"].to_numpy(dtype=float)
    groups = table["category"].to_numpy()
    n_groups = table["category"].nunique()
    present = list(table["category"].unique())
    mean_ceiling = float(np.mean([ceilings[c] for c in present]))
    results, per_category = [], []

    for cell in grid:
        columns = fitted_columns[(cell["model"], cell["metric"])]
        features = table[columns].to_numpy(dtype=float)
        predicted = cross_val_predict(_model(), features, target,
                                      cv=GroupKFold(n_splits=n_groups), groups=groups)
        rho = float(stats.spearmanr(predicted, target).statistic)
        per_category_rho = []
        for category in sorted(present):
            mask = groups == category
            if mask.sum() < 10:
                continue
            category_rho = float(stats.spearmanr(predicted[mask], target[mask]).statistic)
            per_category_rho.append(category_rho)
            per_category.append({"model": cell["model"], "metric": cell["metric"],
                                 "category": category,
                                 "n_pairs": int(mask.sum()), "rho": round(category_rho, 4),
                                 "noise_ceiling": round(ceilings[category], 4),
                                 "rho_over_ceiling": round(category_rho / ceilings[category], 4)})
        mean_rho = float(np.mean(per_category_rho))
        results.append({"model": cell["model"], "metric": cell["metric"],
                        "features": " + ".join(columns), "n_pairs": len(table),
                        "n_categories": n_groups,
                        "group_cv_spearman": round(rho, 4),
                        "mean_category_spearman": round(mean_rho, 4),
                        "noise_ceiling_mean": round(mean_ceiling, 4),
                        # Divides the MEAN over categories, not the pooled correlation. The
                        # pooled figure carries a between-category component created by the
                        # level miscalibration of out-of-fold predictions for a held-out
                        # category, and can be strongly negative while every within-category
                        # ordering is correct (Qwen3 AP 0.8: pooled -0.167, mean +0.493).
                        # Dividing that by a ceiling would propagate the artefact.
                        "rho_over_ceiling": round(mean_rho / mean_ceiling, 4),
                        "pooled_rho_over_ceiling": round(rho / mean_ceiling, 4),
                        "group_cv_r2": round(float(r2_score(target, predicted)), 4)})

    results_df = pd.DataFrame(results)

    # Delta against jaccard, on the mean-over-categories reading, for every non-jaccard cell.
    jaccard_mean_rho = float(
        results_df.loc[results_df.model == "jaccard", "mean_category_spearman"].iloc[0])
    non_jaccard = results_df.model != "jaccard"
    results_df.loc[non_jaccard, "delta_mean_rho_vs_jaccard"] = (
        results_df.loc[non_jaccard, "mean_category_spearman"] - jaccard_mean_rho)

    # Partial F for every nested_pairs comparison the grid defines, attached to the row of
    # the FULLER cell, one column pair per reduced model since jaccard_profile_both has two
    # well-defined reduced models (jaccard_profile and jaccard_profile_z) and collapsing them
    # into one column would silently overwrite one comparison with the other. Keyed on
    # (model, metric) rather than model alone, the same principle Study A's paired t follows,
    # since Study B has no reference factor and a second registered metric would otherwise
    # collide on a shared cell name.
    y = target
    for reduced, full in nested_pairs(grid):
        f_stat, p_value = _partial_f_test(
            y, table[fitted_columns[(reduced["model"], reduced["metric"])]].to_numpy(dtype=float),
            table[fitted_columns[(full["model"], full["metric"])]].to_numpy(dtype=float))
        where = (results_df.model == full["model"]) & (results_df.metric == full["metric"])
        results_df.loc[where, f"partial_f_vs_{reduced['model']}"] = f_stat
        results_df.loc[where, f"partial_f_p_vs_{reduced['model']}"] = p_value

    return results_df, pd.DataFrame(per_category)


def _run_axis_sensitivity(scope, concept_metadata: pd.DataFrame, dir_a, dir_b) -> None:
    """The one robustness line of the axis decision: the jaccard and jaccard_profile cells
    re-fitted with profiles on the FLAT layer axis, label-word reference. Written once, at
    the whole-model scope and REFERENCE_AP only, not swept, its purpose is to show the
    block-axis choice is not load bearing, not to characterise the flat axis.

    The two studies derive their own frames here exactly as the main run does, so the
    sensitivity line is comparable to the number it is a sensitivity of, rather than to a
    differently gated row set.
    """
    design = build_design_frame(scope.expert_df, concept_metadata, ACTIVE_METRICS, axis="layer")
    if design.empty:
        return
    design = add_centroid_features(design, scope.expert_df, ACTIVE_METRICS, axis="layer")
    study_a = design.dropna(subset=study_a_columns(ACTIVE_METRICS)).reset_index(drop=True)
    study_b = design.dropna(subset=study_b_columns(ACTIVE_METRICS)).reset_index(drop=True)

    if len(study_a) >= MIN_TRAIN_CONCEPTS:
        rank_results, _, _ = run_typicality_ranking(study_a, ACTIVE_METRICS)
        if not rank_results.empty:
            keep_a = rank_results[(rank_results.reference == "label_word")
                                  & rank_results.model.isin(["jaccard", "jaccard_profile"])]
            save_dataframe(keep_a.assign(axis="layer"), dir_a / "axis_sensitivity.csv")

    if len(study_b) >= MIN_PAIR_CONCEPTS:
        pair_results, _ = run_pair_similarity(study_b, scope.expert_df, ACTIVE_METRICS,
                                              axis="layer")
        if not pair_results.empty:
            keep_b = pair_results[pair_results.model.isin(["jaccard", "jaccard_profile"])]
            save_dataframe(keep_b.assign(axis="layer"), dir_b / "axis_sensitivity.csv")


def execute_module_9_typicality_prediction(scope, concept_metadata: pd.DataFrame,
                                           model_dir, axis_sensitivity: bool = False
                                           ) -> tuple[pd.DataFrame, dict]:
    """Execute Module 9: human-typicality ranking (Study A) and human pair-similarity
    regression (Study B) over one generated feature grid.

    Self-sufficient: every feature is computed here from scope.expert_df, so module 3
    need not be enabled and future profile metrics touch no other module. The default
    metric's label-word columns equal module 3's CSV values by construction, asserted
    by tests/check_design_matches_module3.py.

    ONE design frame is built and ONE set of centroid features is added, then each study
    derives its OWN row set from it by dropping the rows missing any of that study's own
    candidate features. The identical-rows rule that makes cells comparable is a rule within
    a study, since it exists so the six or twelve cells of one study can be read against each
    other, and the two studies are never compared numerically, having different targets on
    different units. Sharing one dropna across both would gate Study B on the centroid
    columns and on the category label word, neither of which it reads, and at strict
    thresholds that starves it to nothing while a Study-B-appropriate frame has ample data.
    Both row counts are logged so the difference is visible and its cause readable.

    axis_sensitivity=True additionally re-runs the jaccard and jaccard_profile cells on the
    flat layer axis, whole-model scope at REFERENCE_AP only, writing axis_sensitivity.csv
    into each study folder.

    Returns (rank_results, summary_row).
    """
    out_dir = scope_out_dir(model_dir, scope)
    empty = pd.DataFrame()
    rank_results, pair_results = empty, empty

    design = build_design_frame(scope.expert_df, concept_metadata, ACTIVE_METRICS)
    if design.empty:
        log.warning(f"  [{scope.label}] Skipping module 9: the design frame is empty.")
        return empty, scope_summary_row(scope, n_concepts=0, n_concepts_pair_study=0,
                                        **_summary(empty, empty))

    design = add_centroid_features(design, scope.expert_df, ACTIVE_METRICS)
    # The frame every study derives from, before either study's dropna, so it sits at the
    # scope root rather than inside one study's folder.
    save_dataframe(design, out_dir / "typicality_model_design.csv")

    study_a = design.dropna(subset=study_a_columns(ACTIVE_METRICS)).reset_index(drop=True)
    study_b = design.dropna(subset=study_b_columns(ACTIVE_METRICS)).reset_index(drop=True)
    log.info(f"  [{scope.label}] Design frames from {len(design)} gated concepts: "
             f"Study A keeps {len(study_a)} over {study_a['category'].nunique()} categories, "
             f"Study B keeps {len(study_b)} over {study_b['category'].nunique()} categories. "
             f"Each study drops only the rows missing its OWN candidate features, so Study A "
             f"is gated on the label-word and member-centroid columns of every metric while "
             f"Study B, which recomputes its own symmetric pair features and reads neither, "
             f"is not. The two counts coincide at lenient thresholds and diverge as the "
             f"profile columns thin out.")

    log.info(f"  [{scope.label}] Study A: pairwise typicality ranking...")
    dir_a = study_dir(out_dir, "A")
    if len(study_a) >= MIN_TRAIN_CONCEPTS:
        save_dataframe(study_a, dir_a / "study_a_design.csv")
        rank_results, per_category, per_pair = run_typicality_ranking(study_a, ACTIVE_METRICS)
        if not rank_results.empty:
            save_dataframe(rank_results, dir_a / "ranker_comparison.csv")
            save_dataframe(per_category, dir_a / "ranker_per_category.csv")
            plot_ranker_comparison(rank_results, per_category, dir_a)
            plot_category_dumbbell(per_category, dir_a)
        if per_pair:
            plot_accuracy_by_gap(per_pair, dir_a)
            plot_pair_errors(per_pair, study_a, dir_a)
    else:
        log.warning(f"  [{scope.label}] Skipping Study A: {len(study_a)} usable concepts, "
                    f"need >= {MIN_TRAIN_CONCEPTS}.")

    log.info(f"  [{scope.label}] Study B: human pair similarity...")
    dir_b = study_dir(out_dir, "B")
    if len(study_b) >= MIN_PAIR_CONCEPTS:
        # The concepts Study B's pairs are drawn from. The rows actually fitted are the pairs
        # humans rated, built and gated inside run_pair_similarity.
        save_dataframe(study_b, dir_b / "study_b_design.csv")
        pair_results, pair_per_category = run_pair_similarity(study_b, scope.expert_df,
                                                              ACTIVE_METRICS)
        if not pair_results.empty:
            save_dataframe(pair_results, dir_b / "pair_similarity_comparison.csv")
            save_dataframe(pair_per_category, dir_b / "pair_similarity_per_category.csv")
            plot_pair_similarity(pair_results, pair_per_category, dir_b)
        else:
            log.warning(f"  [{scope.label}] Study B produced no results, too few rated pairs "
                        f"survived on {len(study_b)} concepts.")
    else:
        log.warning(f"  [{scope.label}] Skipping Study B: {len(study_b)} usable concepts, "
                    f"need >= {MIN_PAIR_CONCEPTS}.")

    if axis_sensitivity and scope.is_whole_model:
        _run_axis_sensitivity(scope, concept_metadata, dir_a, dir_b)

    return rank_results, scope_summary_row(scope, n_concepts=len(study_a),
                                           n_concepts_pair_study=len(study_b),
                                           **_summary(rank_results, pair_results))
