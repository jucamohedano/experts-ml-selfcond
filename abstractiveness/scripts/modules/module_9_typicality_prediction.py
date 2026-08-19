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
from matplotlib.lines import Line2D
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.helpers import (save_dataframe, scope_out_dir, scope_summary_row, gearys_c,
                           build_layer_probability_matrix, to_block_axis,
                           expert_set_overlap_matrices, pair_similarity_vector,
                           pair_layer_profile_vectors, layer_profile_matrices,
                           count_matched_z, PROFILE_METRICS, DEFAULT_PROFILE_METRIC,
                           ACTIVE_PROFILE_METRICS, MIN_PROFILE_EXPERTS,
                           layer_profile_metric_matrices, profile_metric_label,
                           SIGNED_PROFILE_METRICS)
from utils.human_similarity import human_pair_lookup, human_noise_ceiling
from modules.module_2_shannon_entropy import PEAK_GAP_RELIABLE_PP
from utils.plot_helpers import (COMPARISON_STYLE, COMPARISON_COLORS, comparison_legend,
                                build_category_color_map,
                                annotate_barh_values, style_comparison_axis,
                                comparison_panel_height)

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
    "cluster_ari_category":
        "Study C, layer-profile dendrogram agreement with the categories (ARI at k=8)",
    "cluster_ari_category_expert_set":
        "Study C, expert-set dendrogram agreement with the categories (ARI at k=8)",
    "cluster_eta2_depth":
        "Study C, dendrogram variance explained on depth centre of mass (k=8 cut)",
    "graph_within_category_pct":
        "Study C, share of expert-set Jaccard edges joining one category",
}

TARGET = "human_typicality"

# The three studies. A predicts which of two same-category concepts is more typical,
# B predicts the measured human similarity of a pair, and C drops the target entirely and
# asks what structure the representations carry on their own, in exactly two views, a
# dendrogram of the layer profiles and the node-link concept graph of Fedzechkina's
# ExpertLens Figure 4. Folder names follow the "<number>_<name>" convention of the module
# folders themselves, and Study C's two views get one subfolder each.
STUDY_DIRS = {"A": "9.1_typicality_ranking", "B": "9.2_pair_similarity",
              "C": "9.3_concept_structure"}


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

# Profile metrics the grid is instantiated on, shared with modules 3 and 5 so one edit
# changes what the whole pipeline computes. All seven registered metrics today: the grid
# generates five cells per metric plus the single metric-free Jaccard cell, so Study A
# fits 36 cells per reference and Study B fits 36, against 6 and 6 with one metric.
# Narrow ACTIVE_PROFILE_METRICS in utils/helpers.py to make a verification run cheap.
ACTIVE_METRICS = list(ACTIVE_PROFILE_METRICS)

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

    # Height scales with the number of grid cells rather than sitting at a fixed 4.6 inches
    # per panel. Each reference draws one row per cell, and the grid is generated per
    # registered metric, so seven metrics turn six rows into thirty-six and a fixed height
    # overlaps every tick label with the bar above it.
    panel_rows = max(len(_grid_order(results[results.reference == reference]))
                     for reference in references)
    fig, axes = plt.subplots(
        len(references), 1, squeeze=False,
        figsize=(style["figsize"][0],
                 comparison_panel_height(panel_rows, grouped=True) * len(references)))
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
        # Left at the row CENTRE, in the gap between the row's two bars, rather than beside
        # the ranker bar it describes. The per-category dots share the ranker bar's half of
        # the row, so aligning the text with that bar prints it straight through them.
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

    # Selected on the (model, metric) PAIR rather than on the first row of the grid order
    # carrying the model name. With a second metric registered there is one such row per
    # metric, and taking the first would draw whichever metric the grid happened to emit
    # first while the summary row reported _display_metric's choice.
    #
    # Resolved BEFORE the figure is built, because the right panel's row count decides how
    # much of its column it is given.
    base_key = next((key for key in order if key[0] == "jaccard"), None)
    rich_key = next((key for key in order
                     if key == ("jaccard_profile_both", _display_metric())), None)
    detail = pd.DataFrame()
    if base_key is not None and rich_key is not None and not per_category.empty:
        wanted = {base_key, rich_key}
        detail = per_category[[key in wanted for key
                               in zip(per_category.model, per_category.metric)]]
    pivot = pd.DataFrame()
    if not detail.empty:
        # Pivoted on the (model, metric) pair joined into one label rather than on the model
        # name, which stops being unique the moment a second metric is registered.
        cell_of = {base_key: "base", rich_key: "rich"}
        pivot = (detail.assign(cell=[cell_of[key] for key
                                     in zip(detail.model, detail.metric)])
                 .pivot(index="category", columns="cell", values="rho")
                 .reindex(columns=["base", "rich"]).dropna().sort_values("rich"))

    # The figure is sized on the LEFT panel, which carries one row per grid cell and so grows
    # with the number of registered metrics. The right panel carries one row per held-out
    # category, eight of them however many metrics are active, so it is given only the top
    # share of its column. Stretching eight bars over a figure sized for thirty-six would draw
    # them as slabs an inch and a half thick.
    # min_height is the historical fixed height, so a run with one registered metric, six
    # rows, reproduces the figure this replaced rather than shrinking it.
    height = comparison_panel_height(len(order), min_height=style["figsize"][1])
    fig = plt.figure(figsize=(style["figsize"][0], height))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    left_axis = fig.add_subplot(outer[0, 0])

    y_pos = np.arange(len(order))
    highlight = {"jaccard": colors["base"], "jaccard_profile_both": colors["accent"]}
    values = frame["mean_category_spearman"].astype(float)
    left_axis.barh(y_pos, values, height=style["bar_height"],
                   color=[highlight.get(model, colors["muted"]) for model, _ in order])
    left_axis.axvline(ceiling, color=colors["reference"], linewidth=1.8, linestyle="--",
                      label=f"Human noise ceiling ({ceiling:.3f})")
    left_axis.axvline(0.0, color=colors["zero"], linewidth=1.0, linestyle=":")
    left_axis.set_xlim(min(0.0, float(values.min()) - 0.05), ceiling + 0.10)
    annotate_barh_values(left_axis, values,
                         lambda value: f"{value:.3f}  ({value / ceiling:.0%} of ceiling)")
    bold = [label for label, (model, _) in zip(labels, order) if model in HEADLINE_CELLS]
    style_comparison_axis(left_axis, labels, "Agreement with human pair similarity",
                          "Spearman with human ratings, mean over categories", bold=bold)
    comparison_legend(left_axis, ncol=2)

    if not pivot.empty:
        share = min(1.0, comparison_panel_height(len(pivot)) / height)
        inner = outer[0, 1].subgridspec(2, 1, height_ratios=[share, max(1e-3, 1.0 - share)])
        right_axis = fig.add_subplot(inner[0, 0])

        base_label, rich_label = _grid_labels([base_key, rich_key])
        y = np.arange(len(pivot))
        width = style["grouped_bar_height"]
        right_axis.barh(y + width / 2, pivot["rich"], height=width, color=colors["accent"],
                        label=rich_label.replace("_", " "))
        right_axis.barh(y - width / 2, pivot["base"], height=width, color=colors["base"],
                        label=base_label.replace("_", " "))
        # Each category's own ceiling, as a tick, since they differ enough to matter.
        ceilings = detail.drop_duplicates("category").set_index("category")["noise_ceiling"]
        for index, category in enumerate(pivot.index):
            right_axis.plot([ceilings[category]], [index], marker="|", markersize=16,
                            color=colors["reference"], markeredgewidth=2.2,
                            label="Category noise ceiling" if index == 0 else None)
        right_axis.set_xlim(0.0, 1.0)
        style_comparison_axis(right_axis, list(pivot.index), "Per held-out category",
                              "Spearman with human ratings, held-out category")
        comparison_legend(right_axis, ncol=3)

    fig.tight_layout()
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


# --- study C, concept structure ---------------------------------------------------------
#
# Studies A and B ask SUPERVISED questions of the expert representations, predicting a human
# target from features. Study C drops the target and asks what structure the representations
# carry on their own. It has exactly two deliverables, each in its own folder under
# 9.3_concept_structure/:
#
#   dendrograms/  agglomerative hierarchical clustering of the LAYER PROFILES, which asks
#                 what nested grouping the depth distributions produce
#   linkgraphs/   the node-link concept graph of Figure 4 of Fedzechkina's ExpertLens paper
#                 (temp_docs/ExpertLens paper-Fedzechkina.pdf), one node per concept with
#                 edge thickness carrying a pairwise similarity, drawn once on the expert-SET
#                 Jaccard the paper uses and once on the layer-PROFILE agreement this
#                 pipeline uses elsewhere, so the two geometries can be read side by side
#
# An earlier version of this study also reported a cross-validated MDS. That pass is retired,
# see "what the retired MDS pass established" in the module 9 documentation for the numbers
# it produced. MDS survives here only as the link graph's LAYOUT ENGINE, which is why
# _layout_coordinates exists and why nothing about stress or dimensionality is written out.

# Cluster counts swept by the dendrogram quality curve. The upper bound is comfortably past
# the 8 categories of the dataset so a category-recovering cut has room to appear at its own
# k rather than being clipped by the sweep.
CLUSTER_K_RANGE = range(2, 21)

# Cluster count the composition table and the dendrogram branch colors use, 8 so the cut is
# directly comparable to the 8 categories of the Richie-HSJ set.
CLUSTER_CUT_K = 8

# Linkage methods compared. "average" (UPGMA) is the one that respects a precomputed metric
# distance, "complete" is its worst-case counterpart, and "ward" is run on the square-rooted
# profiles rather than on the distance matrix, because Ward's criterion is defined through
# Euclidean variance: the Euclidean distance between sqrt(p) and sqrt(q) is sqrt(2) times the
# Hellinger distance, so sqrt-profiles are the Euclidean space this problem legitimately has.
# Feeding a non-Euclidean distance to Ward is a common silent error and is not done here.
DISTANCE_LINKAGES = ("average", "complete")

# Label permutations behind every p value of this study. 2000 resolves p to 0.0005, finer
# than any claim made from these tables needs.
CLUSTER_PERMUTATIONS = 2000

# Edges kept per node in a link graph. Fedzechkina's Figure 4 draws 40 concepts in 10 domains
# and can afford every edge, while this dataset has 205 words and 20,910 pairs, which at any
# global threshold is either a hairball or an empty canvas. Keeping each node's strongest few
# neighbours is the standard thinning for that, and it preserves the property the figure is
# read for, which is who each concept's nearest associates are.
GRAPH_NEIGHBORS = 3

# Restrict Study C to the whole-model scope. False runs it for every scope module 9 runs,
# which is the honest default, since a sublayer scope answers the same question on a
# different slice and its tree is not predictable from the whole model's. It is a switch
# because Study C is the slowest part of module 9, measured at about 70 seconds per scope on
# GPT-2 at AP 0.5. Set True while iterating, restore before a real sweep.
CLUSTER_WHOLE_MODEL_SCOPE_ONLY = False

CLUSTER_SEED = 42

# Study C sweeps the PROFILE_METRICS registry exactly as modules 3 and 5 do, so every
# registered way of comparing two layer profiles gets its own tree and its own graph and the
# choice of metric becomes visible rather than assumed. One dendrogram and one link graph per
# metric, no auxiliary figures, since the two deliverables are what the study is read for.
CLUSTER_METRICS = list(ACTIVE_PROFILE_METRICS)

# The count-matched null space, kept for the DEFAULT metric only. It exists because raw
# profile agreement is very largely a readout of expert-set SIZE (see
# helpers.layer_profile_matrices, Spearman 0.965 with the smaller expert count on synthetic
# data carrying no true signal), so a tree built on the raw distance alone would recover
# "small-set words against large-set words" and look like a discovery. One metric carries the
# control because the null is metric-agnostic by construction, so running it seven times
# would repeat the same check on seven monotone rescalings of the same profiles.
COUNT_MATCHED_SPACE = f"{DEFAULT_PROFILE_METRIC}_z"

# The expert-SET space, clustered beside the per-metric profile spaces and drawn beside the
# per-metric profile graphs. Its dissimilarity is 1 - Jaccard, a true metric on sets, and it
# is the ONE space here that describes a word by WHICH neurons it recruits rather than by
# where along depth it puts them, so its tree is the direct comparison against all the others.
# It also keeps more words than the profile spaces at strict thresholds, since a Jaccard is
# defined for a word holding a single expert while a layer profile needs MIN_PROFILE_EXPERTS.
EXPERT_SET_SPACE = "expert_set_jaccard"

# The expert-SET graph, drawn beside the per-metric profile graphs. This is the quantity
# Fedzechkina's Figure 4 uses, it asks WHICH neurons two words share rather than WHERE along
# depth they sit, and it does not depend on the profile metric, so it is drawn once.
EXPERT_SET_GRAPH = EXPERT_SET_SPACE


def cluster_word_descriptors(prob_matrix: pd.DataFrame,
                             count_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Per-word summaries of the layer distribution, the vocabulary the clusters are read in.

    Columns:
      expert_count   total retained expert units, the size confound to watch
      peak_bin       index of the depth bin holding the most experts
      peak_gap_pp    percentage-point lead of the peak bin over the runner-up, module 2's
                     peak dominance gap. Below PEAK_GAP_RELIABLE_PP the peak is a tie and its
                     LOCATION is not a statistic worth reading, which matters here because
                     the depth split this study finds is stated in peak bins
      peak_reliable  peak_gap_pp at or above PEAK_GAP_RELIABLE_PP
      depth_band     peak bin in the first fifth of the depth axis (early), the last fifth
                     (late), or between them (middle)
      centre_bin     centre of mass along depth, sum_b b * p_b
      entropy_bits   Shannon entropy of the profile, how spread over depth it is
      gearys_c       spatial autocorrelation along depth, low means smooth
      occupied_bins  number of depth bins holding at least one expert
    """
    bins = np.asarray(prob_matrix.columns, dtype=float)
    prob = prob_matrix.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(prob > 0, prob * np.log2(prob), 0.0).sum(axis=1)
    top_two = np.sort(prob, axis=1)[:, -2:]
    peak_gap = 100.0 * (top_two[:, 1] - top_two[:, 0])
    peak_bin = bins[np.argmax(prob, axis=1)]
    depth_fraction = (peak_bin - bins.min()) / max(bins.max() - bins.min(), 1.0)
    return pd.DataFrame({
        "concept": prob_matrix.index,
        "expert_count": count_matrix.sum(axis=1).to_numpy(),
        "peak_bin": peak_bin,
        "peak_depth_fraction": depth_fraction,
        "peak_gap_pp": peak_gap,
        "peak_reliable": peak_gap >= PEAK_GAP_RELIABLE_PP,
        "depth_band": np.where(depth_fraction <= 0.2, "early",
                               np.where(depth_fraction >= 0.8, "late", "middle")),
        "centre_bin": prob @ bins,
        "entropy_bits": entropy,
        "gearys_c": [gearys_c(row) for row in prob],
        "occupied_bins": (prob > 0).sum(axis=1),
    })


def depth_band_report(descriptors: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Share of words peaking early, in the middle, and late, by abstraction level and by
    category, once over all words and once over reliable peaks only.

    The readable form of "which groups did the tree divide", since the tree splits the words
    by where their expert mass sits. The reliable-peak columns say whether the split survives
    module 2's peak ambiguity caveat.
    """
    frame = descriptors.merge(meta[["concept", "category", "abstraction_level"]],
                              on="concept", how="left")
    frame["group"] = np.where(frame["abstraction_level"] == 1, "LEVEL 1 label words",
                              frame["category"].fillna("uncategorized"))
    rows = []
    for name, group in list(frame.groupby("group")) + [("ALL WORDS", frame)]:
        reliable = group[group["peak_reliable"]]
        row = {"group": name, "n": len(group), "n_reliable_peak": len(reliable)}
        for band in ("early", "middle", "late"):
            row[f"{band}_pct"] = round(100.0 * (group["depth_band"] == band).mean(), 1)
            row[f"{band}_pct_reliable"] = (
                round(100.0 * (reliable["depth_band"] == band).mean(), 1)
                if len(reliable) else float("nan"))
        row["median_centre_bin"] = round(float(group["centre_bin"].median()), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("late_pct", ascending=False)


def cluster_spaces(expert_axis_df: pd.DataFrame, items: list) -> tuple:
    """
    Every dissimilarity Study C clusters in, plus the agreement each link graph draws its
    edges from, as ({name: (distance, usable_mask)}, {name: agreement}).

    One entry per registered profile metric, built in a single pass by
    layer_profile_metric_matrices, which shares the profile build and the Jensen-Shannon
    tensor across metrics rather than recomputing them seven times. The default metric
    additionally contributes COUNT_MATCHED_SPACE, its count-matched null.

    Every registered metric reports an AGREEMENT on a 0 to 100 scale with 100 meaning
    identical profiles, so one conversion serves all of them:

        distance = (100 - agreement) / 100

    For js_distance that is exactly sqrt(JSD), the Jensen-Shannon distance and a true metric
    on the simplex, so the default metric's numbers are unchanged by the sweep. The two
    signed metrics, pearson and spearman, run to -100 rather than 0, so theirs lands in
    [0, 2] instead of [0, 1]. That is a scale difference and not a problem, since average
    linkage and every rank statistic reported here are invariant to it, but it is the reason
    raw distances must never be compared ACROSS metrics.
    """
    matrices = layer_profile_metric_matrices(expert_axis_df, items, CLUSTER_METRICS)
    spaces, agreements = {}, {}

    for metric in CLUSTER_METRICS:
        _, agreement, z = matrices[metric]
        distance = (100.0 - agreement) / 100.0
        # A word is usable when layer_profile_matrices could build a profile for it, which it
        # signals by filling that word's whole ROW AND COLUMN with NaN when it could not, for
        # a word holding fewer than MIN_PROFILE_EXPERTS experts. The DIAGONAL is therefore
        # the test, since a usable word agrees perfectly with itself and an unusable one is
        # NaN. Testing whether a whole row is finite would be wrong, and wrong in the worst
        # way: one unusable word puts a NaN in EVERY other word's row, so the mask collapses
        # to nothing and the caller is handed an empty distance matrix. That is exactly what
        # happened on the Qwen3 mlp.down_proj scope at AP 0.6, where the single word
        # "professions" held one expert and took all 203 usable words down with it.
        usable = np.isfinite(np.diag(distance))
        np.fill_diagonal(distance, 0.0)
        spaces[metric] = (distance, usable)
        agreements[metric] = agreement

        if metric == DEFAULT_PROFILE_METRIC:
            spaces[COUNT_MATCHED_SPACE] = _count_matched_space(z, usable, len(items))

    return spaces, agreements


def _count_matched_space(z: np.ndarray, base_usable: np.ndarray, n_items: int) -> tuple:
    """
    The count-matched null turned into a dissimilarity, as (distance, usable_mask).

    z standardizes a pair's agreement against other real pairs of comparable expert-set
    sizes, so a positive z means the two words agree more than two arbitrary words of those
    sizes do. Negating and shifting it to be non-negative makes a dissimilarity in which
    strongly agreeing pairs sit close together. The scale is ORDINAL, the shift gives its
    zero no meaning, so it is read by average linkage and rank statistics only.

    z carries NaNs beyond the unusable words: its diagonal is NaN by construction, since a
    word is not a pair with itself, and an individual pair is NaN wherever the local null had
    too few comparable pairs to standardize against. Linkage cannot take a NaN at all, so the
    mask keeps only words whose off-diagonal entries are ALL finite within the usable block.
    One pass suffices because z is symmetric, so a bad pair marks both of its endpoints and
    dropping every marked word removes every bad pair with them.
    """
    usable = np.zeros(n_items, dtype=bool)
    base = np.flatnonzero(base_usable)
    if len(base):
        sub = z[np.ix_(base, base)]
        off_diagonal = ~np.eye(len(base), dtype=bool)
        usable[base] = (np.isfinite(sub) | ~off_diagonal).all(axis=1)

    finite = z[np.isfinite(z)]
    if not finite.size:
        # No pair got a null score at all, so there is no space to cluster in. An all-NaN
        # matrix with an empty mask makes the caller skip it rather than crash.
        return np.full_like(z, np.nan), np.zeros(n_items, dtype=bool)

    distance = finite.max() - z
    np.fill_diagonal(distance, 0.0)
    return distance, usable


def linkage_set(distance: np.ndarray, prob: np.ndarray) -> dict:
    """
    Every candidate tree over the same words, as {method: (linkage, cophenetic_r)}.

    The cophenetic correlation is how faithfully a tree reproduces the distances it was built
    from, so it is the criterion for which tree to read rather than a result in itself. Near
    1 means the nesting the dendrogram displays is really in the data, low means the tree is
    an artefact of the linkage rule.
    """
    condensed = squareform(distance, checks=False)
    trees = {method: (hierarchy.linkage(condensed, method=method), None)
             for method in DISTANCE_LINKAGES}
    trees = {method: (link, float(hierarchy.cophenet(link, condensed)[0]))
             for method, (link, _) in trees.items()}
    sqrt_condensed = pdist(np.sqrt(prob), metric="euclidean")
    ward = hierarchy.linkage(sqrt_condensed, method="ward")
    trees["ward_hellinger"] = (ward, float(hierarchy.cophenet(ward, sqrt_condensed)[0]))
    return trees


def merge_height_table(link: np.ndarray, items: list) -> pd.DataFrame:
    """
    Every merge the tree makes, in the order it makes them, which is the table that answers
    "where does a division of the tree appear".

    A dendrogram is built bottom up. Each word starts as its own cluster, and at every step
    the two closest clusters are merged, so a tree over n words makes exactly n - 1 merges.
    The HEIGHT of a merge is the distance between the two clusters at the moment they joined,
    under the linkage rule in use, and it is the only quantity the horizontal axis of the
    figure shows.

    A division therefore appears exactly where a merge does, read backwards. Cutting the tree
    at height h severs every link drawn beyond h, and what remains are the clusters that
    existed just before the first merge above h. Cutting for k clusters is the same
    operation, since undoing the last k - 1 merges leaves k groups, so the cut height sits
    just below the (k-1)-th largest merge height.

    Columns:
      step            merge index, 1 to n - 1, in bottom-up order
      height          the distance at which the two clusters joined
      height_gap      this merge's height minus the previous merge's
      clusters_after  how many clusters remain once this merge is done, n - step
      size_a, size_b  the two merged clusters' sizes, 1 for a single word
      example_a, example_b  one member of each side, for reading the table without the figure

    height_gap is the column to sort on. Merges are made in non-decreasing height order, so
    a LARGE gap means the next thing the algorithm had to join was much further away than
    anything it had joined so far, which is the signature of a division that is really in the
    data rather than one imposed by having to keep merging. A tree whose gaps are all small
    is a tree with no natural number of clusters, and the honest reading of it is that the
    words form a gradient rather than groups.
    """
    n = len(items)
    members = {i: [items[i]] for i in range(n)}
    rows = []
    previous = 0.0
    for step, (left, right, height, size) in enumerate(link, start=1):
        left, right = int(left), int(right)
        joined = members[left] + members[right]
        members[n + step - 1] = joined
        rows.append({
            "step": step,
            "height": round(float(height), 6),
            "height_gap": round(float(height) - previous, 6),
            "clusters_after": n - step,
            "size_a": len(members[left]),
            "size_b": len(members[right]),
            "example_a": members[left][0],
            "example_b": members[right][0],
        })
        previous = float(height)
    return pd.DataFrame(rows)


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """
    Share of a descriptor's variance explained by a cluster partition, between-group sum of
    squares over total. Answers "is this partition just a split on expert count", where near
    1 means the clusters ARE that descriptor and near 0 means they ignore it.
    """
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    if len(values) < 2 or values.std() == 0:
        return float("nan")
    grand = values.mean()
    between = sum(len(values[labels == k]) * (values[labels == k].mean() - grand) ** 2
                  for k in np.unique(labels))
    return float(between / ((values - grand) ** 2).sum())


def cluster_quality_curve(link: np.ndarray, distance: np.ndarray, meta: pd.DataFrame,
                          descriptors: pd.DataFrame) -> pd.DataFrame:
    """
    One row per cluster count k, scoring the cut against the structure we expect and against
    the confounds we do not want.

    ari_category, nmi_category   agreement with the categories, member concepts only, since a
                                 category-label word has no parent
    ari_level                    agreement with the abstraction level
    silhouette                   cluster separation in the distance space itself
    largest_cluster_share        guards the silhouette, since a k=2 cut isolating four
                                 outliers scores well and means nothing
    eta2_*                       variance of each descriptor explained by the cut
    """
    rows = []
    # At a strict threshold every surviving word can be a category label, or every one a
    # member of a single category, leaving nothing to score the cut against. The agreement
    # columns are NaN there rather than an exception, since the eta squared columns still
    # describe the cut perfectly well and are the reason to look at a thin scope at all.
    categorized = meta["category"].notna().to_numpy()
    scorable = categorized.sum() >= 2 and meta.loc[categorized, "category"].nunique() >= 2
    levels_scorable = meta["abstraction_level"].nunique() >= 2
    for k in CLUSTER_K_RANGE:
        labels = hierarchy.fcluster(link, t=k, criterion="maxclust")
        row = {
            "k": k,
            "n_clusters_realized": len(np.unique(labels)),
            "largest_cluster_share": float(pd.Series(labels).value_counts().iloc[0] / len(labels)),
            "silhouette": float(silhouette_score(distance, labels, metric="precomputed"))
            if len(np.unique(labels)) > 1 else float("nan"),
            "ari_category": float(adjusted_rand_score(
                meta.loc[categorized, "category"], labels[categorized]))
            if scorable else float("nan"),
            "nmi_category": float(normalized_mutual_info_score(
                meta.loc[categorized, "category"], labels[categorized]))
            if scorable else float("nan"),
            "ari_level": float(adjusted_rand_score(meta["abstraction_level"], labels))
            if levels_scorable else float("nan"),
        }
        row["eta2_expert_count"] = eta_squared(
            np.log10(descriptors["expert_count"].to_numpy(dtype=float)), labels)
        for column in ("peak_bin", "centre_bin", "entropy_bits", "gearys_c"):
            row[f"eta2_{column}"] = eta_squared(descriptors[column].to_numpy(dtype=float), labels)
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_composition(labels: np.ndarray, meta: pd.DataFrame,
                        descriptors: pd.DataFrame) -> pd.DataFrame:
    """
    What each cluster of a chosen cut actually holds: size, descriptor means, how many
    category-label words landed in it, which categories dominate it, and a sample of members.
    The table that answers "unexpected in what way".
    """
    frame = descriptors.assign(cluster=labels).merge(
        meta[["concept", "category", "abstraction_level"]], on="concept", how="left")
    rows = []
    for cluster, group in frame.groupby("cluster"):
        counts = group["category"].value_counts()
        rows.append({
            "cluster": int(cluster),
            "n_words": len(group),
            "n_level_1": int((group["abstraction_level"] == 1).sum()),
            "n_level_2": int((group["abstraction_level"] == 2).sum()),
            "n_categories_present": int(counts.size),
            "top_categories": ", ".join(f"{n} {c}" for n, c in counts.head(3).items()) or "none",
            "purity_pct": round(100.0 * counts.iloc[0] / counts.sum(), 1) if counts.size else float("nan"),
            "median_expert_count": float(group["expert_count"].median()),
            "mean_peak_bin": round(float(group["peak_bin"].mean()), 2),
            "mean_centre_bin": round(float(group["centre_bin"].mean()), 2),
            "mean_entropy_bits": round(float(group["entropy_bits"].mean()), 3),
            "mean_gearys_c": round(float(group["gearys_c"].mean()), 3),
            "members": ", ".join(group.sort_values("expert_count", ascending=False)["concept"].head(12)),
        })
    return pd.DataFrame(rows).sort_values("n_words", ascending=False)


def cluster_category_enrichment(labels: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Category by cluster contingency with an enrichment ratio, observed over the count
    expected if the two labellings were independent. The readable form of the adjusted Rand
    index: a low index with a handful of ratios above 2 means SOME categories are recovered
    and most are not, which one agreement number cannot say.
    """
    frame = meta.assign(cluster=labels).dropna(subset=["category"])
    if frame.empty:
        return pd.DataFrame()
    observed = pd.crosstab(frame["category"], frame["cluster"])
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.to_numpy().sum()
    long = (observed.stack().rename("observed").to_frame()
            .join((observed / expected).stack().rename("enrichment_ratio"))
            .reset_index())
    long["expected"] = (long["observed"] / long["enrichment_ratio"].replace(0, np.nan)).round(2)
    long["enrichment_ratio"] = long["enrichment_ratio"].round(2)
    return long.sort_values("enrichment_ratio", ascending=False)


def _cell_or_nan(frame: pd.DataFrame, column: str, position: int = 0) -> float:
    """
    One cell of a summary frame, or NaN when the frame is empty or lacks the column.

    group_separation returns an EMPTY frame whenever the labelling it was handed has fewer
    than two distinct groups, which is a normal outcome at strict AP thresholds: by AP 0.9
    on a thin sublayer, every category-label word can fall below MIN_PROFILE_EXPERTS, so
    the abstraction level collapses to one value and there is nothing to separate. Reaching
    into that frame positionally is what raised KeyError: 'p_permutation' on the Qwen3
    mlp.down_proj scope at AP 0.9.
    """
    if frame.empty or column not in frame.columns:
        return float("nan")
    return float(frame[column].iloc[position])


def group_separation(distance: np.ndarray, groups: np.ndarray, rng: np.random.Generator,
                     group_label: str = "group") -> pd.DataFrame:
    """
    How well each group of a labelling holds together in the distance space, one row per
    group plus an ALL GROUPS row.

    within_mean is the mean distance among the group's own members, between_mean the mean
    distance from a member to a non-member, separation their difference. p_permutation is the
    share of CLUSTER_PERMUTATIONS label shuffles reaching a separation at least as large. The
    permutation null is mandatory rather than optional here, since the groups are very
    unequal in size (8 label words against 197 members, categories of 20 to 32) and a plain
    comparison of means would reward the small groups by construction.

    This is the tree's diagnostic, not a separate study. It says whether a low agreement
    between the tree and the categories is the tree's doing or the categories' doing.
    """
    labels = pd.Series(groups)
    usable = labels.notna().to_numpy()
    labels = labels[usable].to_numpy()
    distance = distance[np.ix_(usable, usable)]
    unique = np.unique(labels)
    if len(unique) < 2:
        return pd.DataFrame()

    def separations(assignment: np.ndarray) -> dict:
        out = {}
        for name in unique:
            member = assignment == name
            if member.sum() < 2:
                out[name] = np.nan
                continue
            within = distance[np.ix_(member, member)][np.triu_indices(member.sum(), k=1)].mean()
            out[name] = distance[np.ix_(member, ~member)].mean() - within
        return out

    observed = separations(labels)
    observed_all = np.nanmean(list(observed.values()))
    exceed = {name: 0 for name in unique}
    exceed_all = 0
    for _ in range(CLUSTER_PERMUTATIONS):
        shuffled = separations(rng.permutation(labels))
        for name in unique:
            if np.isfinite(shuffled[name]) and shuffled[name] >= observed[name]:
                exceed[name] += 1
        exceed_all += np.nanmean(list(shuffled.values())) >= observed_all

    rows = []
    for name in unique:
        member = labels == name
        within = (distance[np.ix_(member, member)][np.triu_indices(member.sum(), k=1)].mean()
                  if member.sum() > 1 else np.nan)
        rows.append({
            group_label: name,
            "n": int(member.sum()),
            "within_mean": round(float(within), 4),
            "between_mean": round(float(distance[np.ix_(member, ~member)].mean()), 4),
            "separation": round(float(observed[name]), 4),
            "p_permutation": round((exceed[name] + 1) / (CLUSTER_PERMUTATIONS + 1), 4),
        })
    summary = pd.DataFrame(rows).sort_values("separation", ascending=False)
    summary.loc[len(summary)] = {
        group_label: "ALL GROUPS", "n": len(labels), "within_mean": np.nan,
        "between_mean": np.nan, "separation": round(float(observed_all), 4),
        "p_permutation": round((exceed_all + 1) / (CLUSTER_PERMUTATIONS + 1), 4),
    }
    summary["silhouette_all_groups"] = round(
        float(silhouette_score(distance, labels, metric="precomputed")), 4)
    return summary


def plot_dendrogram(link: np.ndarray, items: list, meta: pd.DataFrame, color_map: dict,
                    title: str, out_path, cut_k: int) -> None:
    """
    Horizontal dendrogram with every leaf label painted in its category's color and the
    category-label words in bold, so a category-recovering tree is readable as contiguous
    blocks of one color without opening the CSV.

    Each horizontal link is drawn at the height its two clusters merged at, and branches are
    colored by the cut at ``cut_k`` clusters, which is the dashed line the figure marks: every
    link crossing it is severed, and what survives to its left is one cluster.
    """
    category_of = dict(zip(meta["concept"], meta["category"]))
    level_of = dict(zip(meta["concept"], meta["abstraction_level"]))
    fig, axis = plt.subplots(figsize=(11, max(6.0, 0.16 * len(items))))
    threshold = link[-(cut_k - 1), 2] if cut_k > 1 else 0.0

    hierarchy.dendrogram(link, labels=list(items), orientation="left", ax=axis,
                         color_threshold=threshold, above_threshold_color="#9aa0a6",
                         leaf_font_size=7)
    axis.axvline(threshold, color="#e34948", ls="--", lw=1.2,
                 label=f"cut at k = {cut_k}, height {threshold:.4f}")
    for label in axis.get_yticklabels():
        word = label.get_text()
        label.set_color(color_map.get(category_of.get(word), "#333333"))
        if level_of.get(word) == 1:
            label.set_fontweight("bold")
    axis.set_title(title, fontsize=12)
    axis.set_xlabel("merge height, the distance at which two clusters joined")
    axis.legend(fontsize=9, loc="lower right")
    axis.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def shared_prefix_length(left: str, right: str) -> int:
    """
    Length of the common leading substring of two words, the cheapest available check on
    whether an edge is semantic or orthographic.

    A neuron that is an expert for a word FORM rather than for its meaning would produce
    cross-domain edges like "dress" to "dresser". The column does not settle the question,
    since those two are also semantically related, but an edge list whose cross-domain links
    are mostly long shared prefixes reads very differently from one whose links are not.
    """
    limit = min(len(left), len(right))
    return next((i for i in range(limit) if left[i] != right[i]), limit)


def concept_graph_edges(similarity: np.ndarray, items: list, meta: pd.DataFrame,
                        neighbors: int = GRAPH_NEIGHBORS) -> pd.DataFrame:
    """
    The edge list of a concept graph, each node keeping its ``neighbors`` strongest partners,
    one row per retained undirected edge.

    `same_category` and `edge_type` make Fedzechkina's qualitative claims countable: the paper
    reports that within-domain concepts are strongly connected while cross-domain edges are
    sparser, and singles out the cross-domain edges that survive as the interesting ones, its
    examples being "driver" reaching "bus" and "vehicle" and "racing" joining sports to
    vehicles.

    An edge kept by either endpoint is kept once, so a node can end up with more than
    ``neighbors`` edges. That asymmetry is deliberate: dropping an edge one concept counts
    among its strongest because the other does not would hide exactly the hub structure the
    figure exists to show.
    """
    category_of = dict(zip(meta["concept"], meta["category"]))
    level_of = dict(zip(meta["concept"], meta["abstraction_level"]))
    ranked = np.argsort(-similarity, axis=1)
    keep = set()
    for i in range(len(items)):
        taken = 0
        for j in ranked[i]:
            if int(j) == i:
                continue
            keep.add((min(i, int(j)), max(i, int(j))))
            taken += 1
            if taken >= neighbors:
                break

    rows = []
    for i, j in sorted(keep):
        left, right = items[i], items[j]
        category_left, category_right = category_of.get(left), category_of.get(right)
        both_members = pd.notna(category_left) and pd.notna(category_right)
        same = bool(both_members and category_left == category_right)
        rows.append({
            "concept_a": left,
            "concept_b": right,
            "weight": round(float(similarity[i, j]), 5),
            "category_a": category_left,
            "category_b": category_right,
            "same_category": same,
            "edge_type": ("label to member" if 1 in (level_of.get(left), level_of.get(right))
                          else "within category" if same else "cross category"),
            "shared_prefix": shared_prefix_length(left, right),
        })
    return pd.DataFrame(rows).sort_values("weight", ascending=False)


def graph_domain_structure(edges: pd.DataFrame, meta: pd.DataFrame,
                           rng: np.random.Generator) -> pd.DataFrame:
    """
    Whether a link graph's edges respect the semantic domains, against the permutation
    baseline the ExpertLens paper uses for the same question.

    That paper validates its domain structure against randomly resampled domain groupings,
    its example being "animal" put with "jacket", "liver", "doctor" and "red". The same logic
    here: the observed share of edges joining two words of one category is compared against
    the share obtained when the category labels are shuffled over the words, which preserves
    the graph and the category sizes and destroys only the assignment.

    Reported over all edges and over the strongest quartile separately, since the figure
    draws attention to the thick edges and a structure carried only by the weak ones would be
    a different claim.
    """
    categorized = edges.dropna(subset=["category_a", "category_b"])
    if categorized.empty:
        return pd.DataFrame()
    words = meta.dropna(subset=["category"])
    labels = words["category"].to_numpy()
    category_of = dict(zip(words["concept"], labels))

    def within_share(assignment: dict, frame: pd.DataFrame) -> float:
        same = [assignment[a] == assignment[b] for a, b in
                zip(frame["concept_a"], frame["concept_b"])
                if a in assignment and b in assignment]
        return float(np.mean(same)) if same else float("nan")

    rows = []
    strong = categorized[categorized["weight"] >= categorized["weight"].quantile(0.75)]
    for name, frame in (("all edges", categorized), ("strongest quartile", strong)):
        observed = within_share(category_of, frame)
        null = np.asarray([within_share(dict(zip(words["concept"], rng.permutation(labels))),
                                        frame) for _ in range(CLUSTER_PERMUTATIONS)])
        rows.append({
            "edge_set": name,
            "n_edges": len(frame),
            "within_category_pct": round(100.0 * observed, 1),
            "null_within_category_pct": round(100.0 * float(np.nanmean(null)), 1),
            "null_sd_pct": round(100.0 * float(np.nanstd(null)), 2),
            "ratio_observed_over_null": round(observed / float(np.nanmean(null)), 2),
            "p_permutation": round((int((null >= observed).sum()) + 1) / (CLUSTER_PERMUTATIONS + 1), 4),
        })
    return pd.DataFrame(rows)


def _layout_coordinates(distance: np.ndarray) -> np.ndarray:
    """
    2-D positions for one link graph's nodes, from a metric MDS of that graph's OWN
    dissimilarity.

    MDS appears in this module for this one purpose. The paper lays 40 concepts out by hand
    or by force, which with 205 words would be unreadable and would change with the seed, so
    the layout is an embedding instead.

    Each graph is laid out by the same quantity its edges carry, and that is the whole point
    of passing the distance in rather than fixing one layout for both. An earlier version
    positioned both graphs by the layer-profile distance while the Jaccard graph's edges
    carried expert-set overlap, so the layout actively fought the edges: strongly connected
    words were placed far apart because they happened to differ in depth, and the categories
    that the edges recover cleanly came out visually interleaved. Measured on the 2-D
    category silhouette of the Jaccard graph, matching the layout to the edges moves GPT-2
    from -0.120 to +0.073 and Qwen3 from -0.079 to +0.120, which is the difference between
    categories that read as clumps and categories that read as noise.

    Nothing about stress or dimensionality is reported, because the embedding is furniture
    here rather than a result.
    """
    signature = MDS.__init__.__code__.co_varnames
    kwargs = dict(n_components=2, normalized_stress=True, n_init=4, max_iter=500,
                  random_state=CLUSTER_SEED)
    model = (MDS(metric="precomputed", metric_mds=True, **kwargs)
             if "metric_mds" in signature
             else MDS(dissimilarity="precomputed", metric=True, **kwargs))
    return model.fit_transform(distance)


def plot_concept_graph(coords: np.ndarray, edges: pd.DataFrame, items: list,
                       meta: pd.DataFrame, color_map: dict, title: str, out_path) -> None:
    """
    The node-link concept graph of Figure 4 of the ExpertLens paper, on this dataset.

    Each node is a word colored by its category, with the category-label words as large
    diamonds. Each edge's thickness is the similarity it was drawn on, and cross-category
    edges are darker so the between-domain connections the paper highlights stand out from
    the within-domain mesh.
    """
    index_of = {word: i for i, word in enumerate(items)}
    level = meta.set_index("concept").loc[items, "abstraction_level"].to_numpy()
    category = meta.set_index("concept").loc[items, "category"]
    is_label = level == 1

    fig, axis = plt.subplots(figsize=(16, 14))
    widths = edges["weight"].to_numpy()
    scale = widths.max() if len(widths) and widths.max() > 0 else 1.0
    for row, width in zip(edges.itertuples(index=False), widths):
        i, j = index_of[row.concept_a], index_of[row.concept_b]
        cross = not row.same_category
        axis.plot(coords[[i, j], 0], coords[[i, j], 1], linewidth=0.3 + 3.2 * width / scale,
                  color="#3d3d5c" if cross else "#b9bfd0",
                  alpha=0.75 if cross else 0.55, zorder=1)

    colors = np.asarray([color_map.get(c, "#999999") for c in category])
    axis.scatter(coords[~is_label, 0], coords[~is_label, 1], c=colors[~is_label], s=70,
                 zorder=2, linewidths=0)
    axis.scatter(coords[is_label, 0], coords[is_label, 1], c=colors[is_label], s=320,
                 marker="D", edgecolors="black", linewidths=1.0, zorder=3)
    for word, (x, y) in zip(items, coords):
        label = is_label[index_of[word]]
        axis.annotate(word, (x, y), fontsize=10 if label else 7,
                      fontweight="bold" if label else "normal",
                      xytext=(0, 6), textcoords="offset points", ha="center", zorder=4)

    handles = [Line2D([], [], marker="o", ls="", color=color, label=name, ms=7)
               for name, color in color_map.items()]
    handles += [Line2D([], [], color="#3d3d5c", lw=2, label="cross-category edge"),
                Line2D([], [], color="#b9bfd0", lw=2, label="within-category edge")]
    axis.legend(handles=handles, fontsize=9, ncol=2, loc="best")
    axis.set_title(title, fontsize=13)
    axis.set_xlabel("layout dimension 1 (MDS of this graph's own similarity)")
    axis.set_ylabel("layout dimension 2 (MDS of this graph's own similarity)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_dendrograms(space: str, distance: np.ndarray, usable: np.ndarray, items: list,
                    prob_matrix: pd.DataFrame, meta: pd.DataFrame, descriptors: pd.DataFrame,
                    out_dir, axis_label: str) -> dict:
    """
    The dendrogram deliverable in one dissimilarity space: three trees, the merge table that
    says where their divisions are, the cut at CLUSTER_CUT_K and what it holds, the quality
    curve over every k, and the two group-separation diagnostics.
    """
    keep = np.flatnonzero(usable)
    # A tree needs at least three words to have a shape worth drawing, and scipy raises on
    # an empty condensed matrix rather than returning one. Strict AP thresholds on a thin
    # sublayer thin the usable set out fast, so this is a normal outcome and not an error:
    # the space is skipped, the rest of the study proceeds, and the sweep survives.
    # More words than the cut asks for, so that fcluster and the cut height the dendrogram
    # draws are both well defined. Below that the tree has fewer merges than the cut needs
    # and link[-(cut_k - 1)] would index off the front of the linkage matrix.
    if len(keep) <= CLUSTER_CUT_K:
        log.warning(f"    [{space}] Skipping this space: {len(keep)} of {len(items)} words "
                    f"carry a usable profile, need more than the k={CLUSTER_CUT_K} cut.")
        return {}
    log.info(f"    [{space}] {len(keep)} of {len(items)} words carry a usable profile")

    out_dir.mkdir(parents=True, exist_ok=True)
    kept_items = [items[i] for i in keep]
    distance = distance[np.ix_(keep, keep)]
    kept_meta = meta[meta["concept"].isin(kept_items)].set_index(
        "concept").loc[kept_items].reset_index()
    kept_descriptors = descriptors.set_index("concept").loc[kept_items].reset_index()
    prob = prob_matrix.loc[kept_items].to_numpy(dtype=np.float64)

    trees = linkage_set(distance, prob)
    cophenetic = pd.DataFrame([{"method": method, "cophenetic_r": r}
                               for method, (_, r) in trees.items()]
                              ).sort_values("cophenetic_r", ascending=False)
    save_dataframe(cophenetic, out_dir / "linkage_cophenetic.csv")
    best_method = cophenetic.iloc[0]["method"]

    # Every candidate tree is SCORED, so the cophenetic ranking that picks one is evidenced,
    # but only the winner is DRAWN. Three dendrograms per metric across seven metrics would
    # be twenty-one near-identical figures answering a question the ranking CSV answers in
    # one line.
    curves = []
    for method, (link, _) in trees.items():
        curve = cluster_quality_curve(link, distance, kept_meta, kept_descriptors)
        curve.insert(0, "method", method)
        curves.append(curve)
    quality = pd.concat(curves, ignore_index=True)
    save_dataframe(quality, out_dir / "cluster_quality_by_k.csv")
    best_curve = quality[quality["method"] == best_method]
    plot_dendrogram(trees[best_method][0], kept_items, kept_meta, CLUSTER_COLOR_MAP[0],
                    f"{space}, {best_method} linkage, leaves colored by category",
                    out_dir / f"dendrogram_{best_method}.png", CLUSTER_CUT_K)

    best_link = trees[best_method][0]
    merges = merge_height_table(best_link, kept_items)
    save_dataframe(merges, out_dir / "merge_heights.csv")

    labels = hierarchy.fcluster(best_link, t=CLUSTER_CUT_K, criterion="maxclust")
    save_dataframe(cluster_composition(labels, kept_meta, kept_descriptors),
                   out_dir / f"cluster_composition_k{CLUSTER_CUT_K}.csv")
    save_dataframe(kept_descriptors.assign(cluster=labels).merge(
        kept_meta[["concept", "category", "abstraction_level", "human_typicality"]],
        on="concept", how="left"), out_dir / f"word_cluster_assignment_k{CLUSTER_CUT_K}.csv")
    save_dataframe(cluster_category_enrichment(labels, kept_meta),
                   out_dir / f"cluster_category_enrichment_k{CLUSTER_CUT_K}.csv")

    rng = np.random.default_rng(CLUSTER_SEED)
    category_separation = group_separation(distance, kept_meta["category"].to_numpy(),
                                           rng, "category")
    save_dataframe(category_separation, out_dir / "category_separation.csv")
    level_separation = group_separation(distance, kept_meta["abstraction_level"].to_numpy(),
                                        rng, "abstraction_level")
    save_dataframe(level_separation, out_dir / "level_separation.csv")

    at_cut = best_curve[best_curve["k"] == CLUSTER_CUT_K].iloc[0]
    # The biggest division the tree carries, restricted to the k values the quality curve
    # actually sweeps. The final merge is excluded because it joins everything into one
    # cluster, so its gap is always the largest and cutting there is not a division at all.
    in_range = merges[merges["clusters_after"].between(min(CLUSTER_K_RANGE),
                                                       max(CLUSTER_K_RANGE))]
    biggest = in_range.loc[in_range["height_gap"].idxmax()]
    return {
        "space": space,
        "n_words": len(kept_items),
        "best_linkage": best_method,
        "cophenetic_r": round(float(cophenetic.iloc[0]["cophenetic_r"]), 3),
        "cut_height": round(float(best_link[-(CLUSTER_CUT_K - 1), 2]), 4),
        "largest_height_gap": round(float(biggest["height_gap"]), 4),
        "largest_height_gap_at_k": int(biggest["clusters_after"]),
        "ari_category": round(float(at_cut["ari_category"]), 3),
        "ari_level": round(float(at_cut["ari_level"]), 3),
        "silhouette": round(float(at_cut["silhouette"]), 3),
        "eta2_expert_count": round(float(at_cut["eta2_expert_count"]), 3),
        "eta2_centre_bin": round(float(at_cut["eta2_centre_bin"]), 3),
        "eta2_entropy_bits": round(float(at_cut["eta2_entropy_bits"]), 3),
        "category_silhouette": round(_cell_or_nan(category_separation,
                                                  "silhouette_all_groups"), 3),
        "category_separation_p": _cell_or_nan(category_separation, "p_permutation", -1),
        "level_separation_p": _cell_or_nan(level_separation, "p_permutation", -1),
    }


def run_linkgraph(name: str, label: str, similarity: np.ndarray, distance: np.ndarray,
                  items: list, meta: pd.DataFrame, out_dir) -> dict:
    """
    One link graph: its figure, its edge list, and its domain-structure test.

    ``similarity`` drives the edges and ``distance`` the layout, and the caller passes the
    two forms of the SAME quantity, see _layout_coordinates for why that matters.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = _layout_coordinates(distance)
    edges = concept_graph_edges(similarity, items, meta)
    save_dataframe(edges, out_dir / "concept_graph_edges.csv")
    structure = graph_domain_structure(edges, meta, np.random.default_rng(CLUSTER_SEED))
    save_dataframe(structure, out_dir / "concept_graph_domain_structure.csv")
    plot_concept_graph(coords, edges, items, meta, CLUSTER_COLOR_MAP[0],
                       f"Concept graph, edge thickness = {label} "
                       f"(top {GRAPH_NEIGHBORS} per word)", out_dir / "concept_graph.png")
    counts = edges["edge_type"].value_counts()
    return {
        "graph": name,
        "n_edges": len(edges),
        "within_category": int(counts.get("within category", 0)),
        "label_to_member": int(counts.get("label to member", 0)),
        "cross_category": int(counts.get("cross category", 0)),
        "within_category_pct": _cell_or_nan(structure, "within_category_pct"),
        "null_within_category_pct": _cell_or_nan(structure, "null_within_category_pct"),
        "ratio_observed_over_null": _cell_or_nan(structure, "ratio_observed_over_null"),
        "p_permutation": _cell_or_nan(structure, "p_permutation"),
        "cross_category_with_shared_prefix": int(
            ((edges["edge_type"] == "cross category") & (edges["shared_prefix"] >= 3)).sum()),
    }


# Module-level slot holding the category color map for the current scope, so the plotting
# helpers above do not each need it threaded through. A one-element list rather than a bare
# name so the assignment inside execute_study_c is visible to them without a global statement.
CLUSTER_COLOR_MAP = [{}]


def _study_c_summary(trees: pd.DataFrame, graphs: pd.DataFrame) -> dict:
    """
    The three fields module 9 folds into its cross-scope comparison row, taken from the raw
    js_distance tree and the expert-set Jaccard graph, the two directly comparable to what
    the rest of the pipeline reports. Any deliverable a thin scope skipped comes back NaN
    rather than absent, so every scope contributes the same columns to the table.
    """
    def cell(frame, key_column, key, column):
        if frame.empty:
            return float("nan")
        row = frame[frame[key_column] == key]
        return float(row[column].iloc[0]) if len(row) else float("nan")

    return {
        "cluster_ari_category": cell(trees, "space", DEFAULT_PROFILE_METRIC, "ari_category"),
        "cluster_ari_category_expert_set": cell(trees, "space", EXPERT_SET_SPACE,
                                                "ari_category"),
        "cluster_eta2_depth": cell(trees, "space", DEFAULT_PROFILE_METRIC, "eta2_centre_bin"),
        "graph_within_category_pct": cell(graphs, "graph", EXPERT_SET_GRAPH,
                                          "within_category_pct"),
    }


def _study_c_log_lines(trees: pd.DataFrame, graphs: pd.DataFrame) -> list:
    """One log line per deliverable that actually ran, so a skipped space is silent here
    rather than raising on a missing row."""
    lines = []
    raw = trees[trees["space"] == DEFAULT_PROFILE_METRIC] if not trees.empty else trees
    if len(raw):
        row = raw.iloc[0]
        lines.append(f"Study C dendrogram: {row['best_linkage']} linkage at cophenetic "
                     f"{row['cophenetic_r']:.3f}, cut for k={CLUSTER_CUT_K} at height "
                     f"{row['cut_height']:.4f}, largest merge-height gap at "
                     f"{row['largest_height_gap_at_k']} clusters, ARI vs category "
                     f"{row['ari_category']:.3f} against eta squared "
                     f"{row['eta2_centre_bin']:.3f} on depth centre of mass")
    jaccard = graphs[graphs["graph"] == EXPERT_SET_GRAPH] if not graphs.empty else graphs
    if len(jaccard):
        row = jaccard.iloc[0]
        if np.isfinite(row["within_category_pct"]):
            lines.append(f"Study C linkgraph: {row['within_category_pct']:.1f} percent of "
                         f"expert-set Jaccard edges are within category against a "
                         f"{row['null_within_category_pct']:.1f} percent permutation null, "
                         f"ratio {row['ratio_observed_over_null']:.2f} at "
                         f"p = {row['p_permutation']}")
        else:
            lines.append(f"Study C linkgraph: {int(row['n_edges'])} edges drawn, too few "
                         f"categorized words for the domain-structure test")
    return lines


def execute_study_c_concept_structure(scope, concept_metadata: pd.DataFrame,
                                      module_out_dir) -> dict:
    """
    Study C: the two unsupervised views of the expert representations, written into
    9.3_concept_structure/dendrograms/ and 9.3_concept_structure/linkgraphs/.

    Self-sufficient from scope.expert_df exactly as studies A and B are, and independent of
    their shared design frame, so it runs even at thresholds where the human-target studies
    have no rows left. Returns the fields module 9 folds into its cross-scope comparison row,
    taken from the raw js_distance tree and the expert-set Jaccard graph, which are the two
    directly comparable to what the rest of the pipeline reports.
    """
    empty = _study_c_summary(pd.DataFrame(), pd.DataFrame())
    if CLUSTER_WHOLE_MODEL_SCOPE_ONLY and not scope.is_whole_model:
        log.info(f"  [{scope.label}] Study C skipped, CLUSTER_WHOLE_MODEL_SCOPE_ONLY is set.")
        return empty

    out_dir = module_out_dir / STUDY_DIRS["C"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # The whole-model scope must be read on the block axis, where the flat layer axis
    # interleaves sublayer types and is not a depth axis at all. A sublayer scope already
    # holds exactly one layer per block, so block aggregation there is an identity relabel.
    expert_axis_df = axis_frame(scope.expert_df, "block" if scope.is_whole_model else "layer")
    axis_label = "block (depth)" if scope.is_whole_model else "layer (depth)"

    # Level 1 words first, then level 2, the ordering convention every other module's tables
    # use, so a row of this study's CSVs lines up with theirs.
    present = set(expert_axis_df["concept"])
    items = [w for w in concept_metadata.sort_values(
        ["abstraction_level", "concept"])["concept"] if w in present]
    if len(items) < 20:
        log.warning(f"  [{scope.label}] Skipping Study C: {len(items)} words with experts, "
                    f"too few for a tree or a graph.")
        return empty

    count_matrix, prob_matrix = build_layer_probability_matrix(expert_axis_df)
    count_matrix = count_matrix.reindex(items).fillna(0)
    prob_matrix = prob_matrix.reindex(items).fillna(0.0)
    descriptors = cluster_word_descriptors(prob_matrix, count_matrix)
    save_dataframe(descriptors, out_dir / "word_profile_descriptors.csv")
    save_dataframe(depth_band_report(descriptors, concept_metadata),
                   out_dir / "depth_band_by_group.csv")
    CLUSTER_COLOR_MAP[0] = build_category_color_map(
        concept_metadata.sort_values(["abstraction_level", "concept"])["category"])

    log.info(f"  [{scope.label}] Study C: {len(items)} words, {prob_matrix.shape[1]} "
             f"{axis_label} bins, {int(count_matrix.to_numpy().sum())} expert rows")

    spaces, agreements = cluster_spaces(expert_axis_df, items)

    # The expert-set space, on the SCOPE frame rather than the depth-aggregated one, since a
    # set is unordered and block aggregation would merge distinct units of one block into one
    # bin and inflate every overlap. Every word in items holds at least one expert, so unlike
    # the profile spaces this one places all of them.
    _, jaccard, _ = expert_set_overlap_matrices(scope.expert_df, items)
    jaccard_distance = 1.0 - jaccard
    np.fill_diagonal(jaccard_distance, 0.0)
    spaces[EXPERT_SET_SPACE] = (jaccard_distance, np.ones(len(items), dtype=bool))

    tree_rows = [row for space in spaces
                 if (row := run_dendrograms(space, *spaces[space], items, prob_matrix,
                                            concept_metadata, descriptors,
                                            out_dir / "dendrograms" / space, axis_label))]
    trees = pd.DataFrame(tree_rows)
    if not trees.empty:
        save_dataframe(trees, out_dir / "dendrograms" / "dendrogram_summary.csv")

    # The graphs are drawn over the words the profile space could place, so the layout has a
    # position for every node it has to draw. Below three nodes there is no graph to draw
    # and no layout to fit, and the same thin-scope logic as the trees applies.
    keep = np.flatnonzero(spaces[DEFAULT_PROFILE_METRIC][1])
    if len(keep) < 3:
        log.warning(f"  [{scope.label}] Study C: skipping the link graphs, only {len(keep)} "
                    f"words carry a usable profile.")
        return _study_c_summary(trees, pd.DataFrame())
    graph_items = [items[i] for i in keep]
    graph_meta = concept_metadata[concept_metadata["concept"].isin(graph_items)].set_index(
        "concept").loc[graph_items].reset_index()
    # Expert-set Jaccard is computed on the SCOPE frame and not the depth-aggregated one,
    # since a set is unordered and block aggregation would merge distinct units of one block
    # into one bin and inflate every overlap.
    # Each graph carries its similarity for the edges and the matching dissimilarity for
    # the layout, see _layout_coordinates for why the two must be the same quantity. The
    # expert-set pair is sliced from the matrices already built for its dendrogram above.
    graph_block = np.ix_(keep, keep)
    graphs_to_draw = {EXPERT_SET_GRAPH: ("expert-set Jaccard", jaccard[graph_block],
                                         jaccard_distance[graph_block])}
    for metric in CLUSTER_METRICS:
        graphs_to_draw[metric] = (f"{profile_metric_label(metric)} profile agreement",
                                  agreements[metric][graph_block],
                                  spaces[metric][0][graph_block])

    graph_rows = [run_linkgraph(name, *graphs_to_draw[name], graph_items, graph_meta,
                                out_dir / "linkgraphs" / name) for name in graphs_to_draw]
    graphs = pd.DataFrame(graph_rows)
    save_dataframe(graphs, out_dir / "linkgraphs" / "linkgraph_summary.csv")

    for line in _study_c_log_lines(trees, graphs):
        log.info(f"  [{scope.label}] {line}")
    return _study_c_summary(trees, graphs)


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
    regression (Study B) over one generated feature grid, then unsupervised clustering and
    the two unsupervised views of concept structure (Study C).

    Studies A and B share one design frame and are gated on it. Study C reads
    scope.expert_df directly and shares nothing with them, so it runs even when the design
    frame is empty, which is the case at strict thresholds where the human-target studies
    have no rows left.

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
        # Studies A and B are out, Study C is not: it needs no design frame and no human
        # target, so an empty design is a reason to skip them and still run the clustering.
        log.warning(f"  [{scope.label}] Studies A and B skipped: the design frame is empty. "
                    f"Study C still runs, it reads scope.expert_df directly.")
        clustering = execute_study_c_concept_structure(scope, concept_metadata, out_dir)
        return empty, scope_summary_row(scope, n_concepts=0, n_concepts_pair_study=0,
                                        **_summary(empty, empty), **clustering)

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

    # Study C needs neither design frame. It reads scope.expert_df directly and keeps
    # every word with a usable profile, labels included, so it runs even at thresholds
    # where the two supervised studies have been starved out.
    log.info(f"  [{scope.label}] Study C: concept structure...")
    clustering = execute_study_c_concept_structure(scope, concept_metadata, out_dir)

    if axis_sensitivity and scope.is_whole_model:
        _run_axis_sensitivity(scope, concept_metadata, dir_a, dir_b)

    return rank_results, scope_summary_row(scope, n_concepts=len(study_a),
                                           n_concepts_pair_study=len(study_b),
                                           **_summary(rank_results, pair_results),
                                           **clustering)
