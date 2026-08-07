"""Module 9: exploratory prediction of human typicality from pairwise expert features.

Asks one question. A concept's human typicality rating is known to track how much its
expert set overlaps its category label's (module 4's typicality_vs_jaccard panel, r about
0.30 to 0.38). Does adding the layer-profile agreement of module 3, subchapter 3.2, which
measures depth allocation rather than neuron identity, predict typicality better than
Jaccard alone?

Deliberately small. Linear models on 196 concepts with one or two predictors, because the
point is whether a second feature carries independent signal, not to build the best
possible typicality regressor.

Two subchapters. 9.1 regresses the rating per concept. 9.2 reframes the same data as a
within-category pairwise ranking, "of these two concepts, which is the more typical", which
is closer to how the ratings were collected (the source column is typicality_HSJ_pairwise),
turns 196 concepts into thousands of ordered pairs, and cancels every category-level term
algebraically, since differencing two members of one category removes anything they share.

Two things this module is careful about, both of which would otherwise make the comparison
meaningless.

First, in-sample R^2 cannot answer the question. Adding any predictor, including pure
noise, never decreases it, so the two-feature model always "wins". Every headline number
here is therefore out of sample, from pooled cross-validated predictions.

Second, all feature sets are fit on exactly the same rows. layer_profile_z is undefined
for words below MIN_PROFILE_EXPERTS, so a per-model dropna would train the Jaccard model
on more concepts than the others and the R^2 values would not be comparable. The design
frame drops rows missing ANY candidate feature, once, before any model is fit.
"""
import itertools
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.helpers import (save_dataframe, scope_out_dir, scope_summary_row,
                           build_layer_probability_matrix, to_block_axis,
                           expert_set_overlap_matrices, pair_similarity_vector,
                           pair_layer_profile_vectors)
from utils.human_similarity import human_pair_lookup, human_noise_ceiling
from utils.plot_helpers import build_category_color_map

log = logging.getLogger(__name__)

# Headline metrics for the cross-scope sublayer_comparison table.
SUMMARY_LABELS = {
    "cv_r2_jaccard": "CV R2, Jaccard only",
    "cv_r2_jaccard_plus_profile": "CV R2, Jaccard + layer profile",
    "delta_cv_r2_profile": "CV R2 gained from layer profile",
    "rank_acc_jaccard": "Pairwise accuracy, Jaccard only",
    "rank_acc_jaccard_plus_profile": "Pairwise accuracy, Jaccard + layer profile",
    "rank_acc_delta_profile": "Pairwise accuracy gained from layer profile",
    "pair_similarity_rho": "Pair similarity agreement with humans",
}

TARGET = "human_typicality"

# Below this many usable concepts a cross-validated R^2 on 5 folds is not worth reporting.
MIN_TRAIN_CONCEPTS = 40

N_SPLITS = 5
N_REPEATS = 20
RANDOM_STATE = 0

# The two headline sets are "jaccard" and "jaccard_plus_profile". The single-feature
# layer-profile sets are carried because a gain over Jaccard is only interpretable next to
# what the new feature predicts on its own.
FEATURE_SETS = {
    "jaccard": ["jaccard_pct"],
    "layer_profile": ["layer_profile_similarity_pct"],
    "layer_profile_z": ["layer_profile_z"],
    "jaccard_plus_profile": ["jaccard_pct", "layer_profile_similarity_pct"],
    "jaccard_plus_profile_z": ["jaccard_pct", "layer_profile_z"],
    # Same quantities measured against the OTHER MEMBERS instead of the label word, so the
    # run answers which reference a typicality rating actually tracks.
    "members_jaccard": ["mean_jaccard_to_members"],
    "members_profile": ["layer_profile_similarity_to_centroid"],
    "members_both": ["mean_jaccard_to_members", "layer_profile_similarity_to_centroid"],
    "members_both_plus_margin": ["mean_jaccard_to_members",
                                 "layer_profile_similarity_to_centroid", "centroid_margin"],
}
# Features measured against the OTHER MEMBERS of the category rather than the label word.
# Module 6 established that a category label behaves unlike its member average, so which
# reference better predicts human judgments is a question rather than a detail: is a typical
# bird the one most like the word "bird", or the one most like the other birds?
CENTROID_FEATURES = ["mean_jaccard_to_members", "layer_profile_similarity_to_centroid",
                     "centroid_margin"]

# Features module 3 supplies, measured against the category LABEL WORD. Kept separate from
# the centroid ones because this module computes those itself, so requiring module 3 to
# supply them would abort the module.
PARENT_FEATURES = ["jaccard_pct", "layer_profile_similarity_pct", "layer_profile_z"]

ALL_FEATURES = PARENT_FEATURES + CENTROID_FEATURES

# Nested pairs for the partial F test: (reduced, full, label for the added term).
NESTED_TESTS = [("jaccard", "jaccard_plus_profile", "profile"),
                ("jaccard", "jaccard_plus_profile_z", "profile_z"),
                ("members_jaccard", "members_both", "centroid_profile"),
                ("members_both", "members_both_plus_margin", "centroid_margin")]

# --- subchapter 9.2, pairwise ranking -------------------------------------------------
# Pairs whose two ratings differ by less than this are near-ties, where the human data is
# least reliable. They stay in the training set but accuracy is also reported on the
# clearer subset, since a model cannot be blamed for missing a distinction the ratings
# barely make.
RANK_MARGIN = 0.10

# Below this many usable within-category pairs the grouped accuracy is not worth reporting.
MIN_RANK_PAIRS = 200

# Feature sets for the ranker, as per-concept vectors that get differenced pairwise. The
# 28-bin block profile is carried as a diagnostic rather than a candidate: pair
# differencing cancels the parent profile, so if the profile still fails here, the gain it
# showed in a per-concept regression was the category-identity shortcut rather than depth
# information.
RANK_FEATURE_SETS = {
    "jaccard": ["jaccard_pct"],
    "jaccard_plus_profile": ["jaccard_pct", "layer_profile_similarity_pct", "layer_profile_z"],
    "block_profile": ["__profile__"],
    "jaccard_plus_block_profile": ["jaccard_pct", "__profile__"],
}


def _model():
    """Standardised linear regression.

    Plain least squares rather than a regularised variant: with 196 concepts and at most
    two predictors there is nothing to regularise, and an unpenalised fit keeps the
    coefficients readable as the change in typicality per standard deviation of a feature.
    """
    return make_pipeline(StandardScaler(), LinearRegression())


def build_design_frame(similarity_metrics_df: pd.DataFrame, concept_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Join module 3's concept-to-parent metrics onto the human typicality rating, keeping only
    rows complete on the target and on every candidate feature.

    The single shared dropna is what makes the feature sets comparable, see the module
    docstring. Returns columns concept, category, human_typicality, and ALL_FEATURES.
    """
    if similarity_metrics_df is None or similarity_metrics_df.empty:
        return pd.DataFrame()
    if TARGET not in concept_metadata.columns:
        log.warning(f"  Module 9 skipped: no {TARGET} column in the metadata.")
        return pd.DataFrame()

    missing = [f for f in PARENT_FEATURES if f not in similarity_metrics_df.columns]
    if missing:
        log.warning(f"  Module 9 skipped: module 3 did not supply {missing}.")
        return pd.DataFrame()

    # Only the label-word features are dropped on here. The centroid features are added
    # afterwards by add_centroid_features, and the caller then applies the single shared
    # dropna over ALL_FEATURES so every feature set is still fitted on identical rows.
    design = (similarity_metrics_df[["concept", "category"] + PARENT_FEATURES]
              .merge(concept_metadata[["concept", TARGET]], on="concept", how="inner")
              .dropna(subset=[TARGET] + PARENT_FEATURES)
              .reset_index(drop=True))
    return design


def _cross_validated_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """
    Out-of-sample performance of one feature set.

    cv_r2_mean/sd pool the out-of-fold predictions within each of N_REPEATS shuffles and
    take R^2 per repeat, so the spread reflects how much the estimate moves with the fold
    split rather than the much noisier per-fold value on 39 test concepts.

    group_cv_r2 holds out one whole category at a time, which is the harder question: can
    typicality be predicted for a category the model never trained on. Random folds let a
    model lean on the category-level mean of its training concepts, so the two numbers
    together separate within-category signal from between-category signal.

    cv_spearman is the rank agreement of the pooled out-of-fold predictions with the true
    ratings, reported because R^2 punishes calibration errors that a ranking use would not
    care about.
    """
    per_repeat = []
    for repeat in range(N_REPEATS):
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        oof = cross_val_predict(_model(), X, y, cv=splitter)
        per_repeat.append(r2_score(y, oof))

    oof_first = cross_val_predict(_model(), X, y,
                                  cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE))
    n_groups = len(np.unique(groups))
    group_oof = (cross_val_predict(_model(), X, y, cv=GroupKFold(n_splits=n_groups), groups=groups)
                 if n_groups >= 2 else np.full_like(y, np.nan, dtype=float))

    return {
        "cv_r2_mean": float(np.mean(per_repeat)),
        "cv_r2_sd": float(np.std(per_repeat, ddof=1)),
        "cv_spearman": float(stats.spearmanr(y, oof_first).statistic),
        "group_cv_r2": float(r2_score(y, group_oof)) if n_groups >= 2 else np.nan,
        "oof_predictions": oof_first,
    }


def _partial_f_test(y: np.ndarray, X_reduced: np.ndarray, X_full: np.ndarray) -> tuple:
    """
    Partial F test for the terms present in X_full but not X_reduced, fitted in sample.

    Complements the cross-validated gain rather than replacing it. The F test asks whether
    the added feature explains significantly more variance in THIS sample, which is a
    question about the fitted coefficient, while the cross-validated gain asks whether it
    helps predict concepts the model has not seen. A feature can pass one and fail the
    other, and both readings are reported.
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


def fit_typicality_models(design: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fit every entry of FEATURE_SETS on the shared design frame.

    Returns (results_df, oof_by_model). results_df carries one row per feature set with the
    out-of-sample metrics, the in-sample R^2 for reference only, and the standardised
    coefficients.
    """
    y = design[TARGET].to_numpy(dtype=float)
    groups = design["category"].to_numpy()

    rows, oof_by_model = [], {}
    for name, features in FEATURE_SETS.items():
        X = design[features].to_numpy(dtype=float)
        scores = _cross_validated_r2(X, y, groups)
        oof_by_model[name] = scores.pop("oof_predictions")

        fitted = _model().fit(X, y)
        coefficients = fitted.named_steps["linearregression"].coef_
        rows.append({
            "model": name,
            "features": " + ".join(features),
            "n_features": len(features),
            "n_concepts": len(design),
            **scores,
            "in_sample_r2": float(fitted.score(X, y)),
            **{f"coef_{f}": float(c) for f, c in zip(features, coefficients)},
        })

    results = pd.DataFrame(rows)

    # Nested comparisons, attached to the row of the fuller model.
    for reduced, full, label in NESTED_TESTS:
        if reduced not in FEATURE_SETS or full not in FEATURE_SETS:
            continue
        f_stat, p_value = _partial_f_test(y, design[FEATURE_SETS[reduced]].to_numpy(dtype=float),
                                          design[FEATURE_SETS[full]].to_numpy(dtype=float))
        gain = (results.loc[results.model == full, "cv_r2_mean"].iloc[0]
                - results.loc[results.model == reduced, "cv_r2_mean"].iloc[0])
        results.loc[results.model == full, "delta_cv_r2_vs_jaccard"] = gain
        results.loc[results.model == full, "partial_f"] = f_stat
        results.loc[results.model == full, "partial_f_p"] = p_value
    return results, oof_by_model


# ---------------------------------------------------------------------------
# Subchapter 9.2: within-category pairwise ranking
# ---------------------------------------------------------------------------

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


def _jensen_shannon_similarity(profile: np.ndarray, reference: np.ndarray) -> float:
    """100 * (1 - sqrt(JSD)) in bits, the same transform module 3 applies to pairs, so a
    centroid feature lands on the same scale as the label-word feature it is compared with."""
    mixture = 0.5 * (profile + reference)

    def entropy(vector: np.ndarray) -> float:
        positive = vector[vector > 0]
        return float(-(positive * np.log2(positive)).sum())

    divergence = entropy(mixture) - 0.5 * (entropy(profile) + entropy(reference))
    return 100.0 * (1.0 - np.sqrt(max(divergence, 0.0)))


def add_centroid_features(design: pd.DataFrame, expert_allocation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Three features measured against the other members of a concept's category rather than
    against the category label word.

    Every reference is LEAVE-ONE-OUT. Without that, a concept in a twenty-member category
    supplies five percent of the object it is scored against, the inflation is larger in
    smaller categories, and part of the feature is self-similarity.

    centroid_margin contrasts the own-category centroid against the seven foreign ones. It
    replaces the count-matched z used for pairs, which does not transfer here: a centroid is
    an average of many profiles and has no expert count of its own, so there is nothing to
    count-match against. The margin controls the same nuisance, that some concepts have
    generically central profiles close to everything, and reads directly as "closer to its own
    category than to the others".

    The block axis is used, matching both module 6's centroid and module 9's existing
    __profile__ feature, because a per-block profile is the interpretable depth signature.
    """
    profiles = build_layer_probability_matrix(to_block_axis(expert_allocation_df))[1]
    present = [c for c in design["concept"] if c in profiles.index]
    if len(present) < 2:
        design = design.copy()
        for column in CENTROID_FEATURES:
            design[column] = np.nan
        return design

    _, jaccard, _ = expert_set_overlap_matrices(expert_allocation_df, present)
    position = {concept: index for index, concept in enumerate(present)}
    members = (design[design["concept"].isin(position)]
               .groupby("category")["concept"].apply(list).to_dict())
    centroids = {category: profiles.loc[names].to_numpy(dtype=float)
                 for category, names in members.items() if names}

    mean_jaccard, to_centroid, margin = [], [], []
    for concept, category in zip(design["concept"], design["category"]):
        siblings = ([name for name in members.get(category, []) if name != concept]
                    if concept in position else [])
        if not siblings:
            mean_jaccard.append(np.nan)
            to_centroid.append(np.nan)
            margin.append(np.nan)
            continue

        row = position[concept]
        mean_jaccard.append(100.0 * float(np.mean(
            [jaccard[row, position[other]] for other in siblings])))

        profile = profiles.loc[concept].to_numpy(dtype=float)
        own_similarity = _jensen_shannon_similarity(
            profile, profiles.loc[siblings].to_numpy(dtype=float).mean(axis=0))
        to_centroid.append(own_similarity)

        foreign = [_jensen_shannon_similarity(profile, block.mean(axis=0))
                   for other, block in centroids.items() if other != category]
        margin.append(own_similarity - float(np.mean(foreign)) if foreign else np.nan)

    design = design.copy()
    design["mean_jaccard_to_members"] = mean_jaccard
    design["layer_profile_similarity_to_centroid"] = to_centroid
    design["centroid_margin"] = margin
    return design


def build_concept_feature_blocks(design: pd.DataFrame, expert_allocation_df: pd.DataFrame) -> dict:
    """
    Per-concept feature matrices for the ranker, one per entry of RANK_FEATURE_SETS,
    aligned row for row with ``design``.

    The "__profile__" placeholder expands to the concept's own block-axis layer
    distribution, 28 bins on Qwen3 and 12 on GPT-2, taken on the depth axis rather than the
    flat one because a per-block profile is the interpretable depth signature. A concept
    absent from the expert frame cannot get a profile, so the feature sets that need one are
    dropped rather than filled with zeros, which would assert a flat distribution.
    """
    _, block_profile = build_layer_probability_matrix(to_block_axis(expert_allocation_df))
    have_profile = design["concept"].isin(block_profile.index).all()

    blocks = {}
    for name, columns in RANK_FEATURE_SETS.items():
        if "__profile__" in columns and not have_profile:
            continue
        parts = []
        for column in columns:
            if column == "__profile__":
                parts.append(block_profile.loc[design["concept"]].to_numpy(dtype=float))
            else:
                parts.append(design[[column]].to_numpy(dtype=float))
        blocks[name] = np.hstack(parts)
    return blocks


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
    ridge of subchapter 9.1 to exactly the same held-out pairs, ranking each pair by its two
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


def fit_pairwise_rankers(design: pd.DataFrame, expert_allocation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run every entry of RANK_FEATURE_SETS through the grouped pairwise evaluation.

    Returns (results, per_category). The headline comparison, Jaccard against Jaccard plus
    the layer-profile scalars, additionally carries a paired t test over the per-category
    accuracies, which is the honest test here: the pairs are not independent observations
    (thousands of them are built from a couple of hundred concepts), whereas the eight
    held-out categories are close to independent replicates of the same comparison.
    """
    pairs, gap = within_category_pairs(design)
    if len(pairs) < MIN_RANK_PAIRS:
        log.warning(f"  Skipping pairwise ranker: {len(pairs)} within-category pairs, "
                    f"need >= {MIN_RANK_PAIRS}.")
        return pd.DataFrame(), pd.DataFrame(), {}

    blocks = build_concept_feature_blocks(design, expert_allocation_df)
    rows, details, per_pair = [], [], {}
    for name, features in blocks.items():
        scores, detail, pair_outcomes = evaluate_ranker(features, design, pairs, gap)
        rows.append({"model": name, "features": " + ".join(RANK_FEATURE_SETS[name]),
                     "n_concepts": len(design), **scores})
        if not detail.empty:
            details.append(detail.assign(model=name))
        if not pair_outcomes.empty:
            per_pair[name] = pair_outcomes

    results = pd.DataFrame(rows)
    per_category = pd.concat(details, ignore_index=True) if details else pd.DataFrame()

    if not per_category.empty and {"jaccard", "jaccard_plus_profile"} <= set(per_category.model):
        baseline = per_category[per_category.model == "jaccard"].set_index("category")["accuracy"]
        richer = per_category[per_category.model == "jaccard_plus_profile"].set_index("category")["accuracy"]
        shared = baseline.index.intersection(richer.index)
        if len(shared) >= 3:
            delta = richer[shared] - baseline[shared]
            results.loc[results.model == "jaccard_plus_profile", "delta_accuracy_vs_jaccard"] = delta.mean()
            results.loc[results.model == "jaccard_plus_profile", "categories_improved"] = int((delta > 0).sum())
            results.loc[results.model == "jaccard_plus_profile", "categories_tested"] = len(shared)
            results.loc[results.model == "jaccard_plus_profile", "paired_t_p"] = float(
                stats.ttest_rel(richer[shared], baseline[shared]).pvalue)
    return results, per_category, per_pair


def plot_ranker_comparison(results: pd.DataFrame, per_category: pd.DataFrame, out_dir) -> None:
    """Ranker accuracy per feature set, and the per-category detail behind the headline pair."""
    if results.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    order = [m for m in RANK_FEATURE_SETS if m in set(results.model)]
    frame = results.set_index("model").reindex(order)
    y_pos = np.arange(len(order))
    width = 0.38
    axes[0].barh(y_pos + width / 2, frame["accuracy"].astype(float), height=width,
                 color="#D96A5B", label="pairwise ranker")
    axes[0].barh(y_pos - width / 2, frame["regression_accuracy"].astype(float), height=width,
                 color="#4B5A6A", label="regression, same pairs")
    axes[0].axvline(0.5, color="#2f2f2f", linewidth=1.3, linestyle="--", label="chance")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([m.replace("_", " ") for m in order], fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.4, max(0.8, float(np.nanmax(frame["accuracy"].astype(float))) + 0.06))
    axes[0].set_xlabel("Accuracy on held-out categories", fontsize=10)
    axes[0].set_title("Which of two same-category concepts is more typical", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(axis="y", visible=False)

    headline = per_category[per_category.model.isin(["jaccard", "jaccard_plus_profile"])] \
        if not per_category.empty else pd.DataFrame()
    if not headline.empty:
        pivot = headline.pivot(index="category", columns="model", values="accuracy")
        pivot = pivot.sort_values("jaccard")
        y = np.arange(len(pivot))
        axes[1].barh(y + width / 2, pivot.get("jaccard_plus_profile", pd.Series(dtype=float)),
                     height=width, color="#D96A5B", label="jaccard + layer profile")
        axes[1].barh(y - width / 2, pivot.get("jaccard", pd.Series(dtype=float)),
                     height=width, color="#4B5A6A", label="jaccard")
        axes[1].axvline(0.5, color="#2f2f2f", linewidth=1.3, linestyle="--")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(pivot.index, fontsize=10)
        axes[1].set_xlim(0.2, 1.0)
        axes[1].set_xlabel("Accuracy on the held-out category", fontsize=10)
        axes[1].set_title("Per held-out category", fontsize=12)
        axes[1].legend(loc="lower right", fontsize=9)
        axes[1].grid(axis="y", visible=False)

    plt.tight_layout()
    plt.savefig(out_dir / "typicality_ranker_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reference_comparison(results: pd.DataFrame, out_dir) -> None:
    """
    Subchapter 9.3's headline: does the layer profile add against the category LABEL WORD,
    or against the category CENTROID?

    Four bars, the same two feature sets measured against each of the two references, with
    the partial F p-value for adding the layer profile printed over each "+ profile" bar.
    That single contrast is the evidence that the feature is not inert: against the label
    word it never reaches significance, against the centroid it usually does, so the
    information it carries is about how a concept relates to its fellow members rather than
    to the category's name.

    Both panels are out of sample. The right one holds out whole categories, which is the
    stricter reading and the one that decides whether either reference generalises.
    """
    pairs = [("jaccard", "label word, Jaccard", "#4B5A6A", None),
             ("jaccard_plus_profile", "label word, + layer profile", "#8fa0b3", "jaccard_plus_profile"),
             ("members_jaccard", "centroid, Jaccard", "#D96A5B", None),
             ("members_both", "centroid, + layer profile", "#f0a596", "members_both")]
    frame = results.set_index("model")
    present = [entry for entry in pairs if entry[0] in frame.index]
    if len(present) < 4:
        log.warning("  Skipping 9.3 reference comparison: centroid rows absent.")
        return

    labels = [entry[1] for entry in present]
    colors = [entry[2] for entry in present]
    y_pos = np.arange(len(present))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), sharey=True)
    panels = [(axes[0], "cv_r2_mean", "Random 5-fold CV $R^2$"),
              (axes[1], "group_cv_r2", "Leave-one-category-out $R^2$")]
    for ax, column, title in panels:
        values = pd.Series([float(frame.loc[entry[0], column]) for entry in present])
        ax.barh(y_pos, values, color=colors, height=0.64)
        ax.axvline(0.0, color="#2f2f2f", linewidth=1.2, linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("$R^2$ (out of sample)", fontsize=10)
        ax.grid(axis="y", visible=False)

        span = (values.max() - min(0.0, values.min())) or 1.0
        for i, (entry, value) in enumerate(zip(present, values)):
            pad = 0.03 * span
            ax.text(value + (pad if value >= 0 else -pad), i, f"{value:.3f}", va="center",
                    ha="left" if value >= 0 else "right", fontsize=9)
            # The partial F p-value belongs to the model that ADDED the profile, and is a
            # property of the fit rather than of a fold, so it is annotated once, on the
            # random-fold panel only.
            p_value = frame.loc[entry[0], "partial_f_p"] if entry[3] and "partial_f_p" in frame else np.nan
            if column == "cv_r2_mean" and pd.notna(p_value):
                mark = "significant" if float(p_value) < 0.05 else "not significant"
                ax.text(value + (pad if value >= 0 else -pad), i - 0.32,
                        f"added profile: p = {float(p_value):.3f} ({mark})",
                        va="center", ha="left" if value >= 0 else "right", fontsize=8,
                        color="#1a7f37" if float(p_value) < 0.05 else "#8a8a8a")
        ax.margins(x=0.22)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle("Typicality against the category label word versus the member centroid", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "typicality_reference_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, out_dir) -> None:
    """Out-of-sample R^2 per feature set, under random folds and under held-out categories."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    order = list(FEATURE_SETS)
    frame = results.set_index("model").reindex(order)
    y_pos = np.arange(len(order))
    highlight = {"jaccard": "#4B5A6A", "jaccard_plus_profile": "#D96A5B"}
    colors = [highlight.get(m, "#9aa5b1") for m in order]

    panels = [(axes[0], "cv_r2_mean", "cv_r2_sd", "Random 5-fold CV $R^2$"),
              (axes[1], "group_cv_r2", None, "Leave-one-category-out $R^2$")]
    for ax, col, err_col, title in panels:
        values = frame[col].astype(float)
        errors = frame[err_col].astype(float) if err_col else None
        ax.barh(y_pos, values.fillna(0.0), xerr=errors, color=colors, height=0.62,
                error_kw={"ecolor": "#2f2f2f", "elinewidth": 1.1, "capsize": 3})
        ax.axvline(0.0, color="#2f2f2f", linewidth=1.2, linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("$R^2$ (out of sample)", fontsize=10)
        ax.grid(axis="y", visible=False)
        # Labels sit outside the bar, on the side it grows toward, anchored past the END OF
        # THE ERROR BAR rather than the bar tip, otherwise they are drawn over the whisker.
        span = (values.max() - min(0.0, values.min())) or 1.0
        pad = 0.03 * span
        for i, v in enumerate(values):
            if pd.isna(v):
                continue
            reach = abs(errors.iloc[i]) if errors is not None and pd.notna(errors.iloc[i]) else 0.0
            anchor = v + (reach + pad if v >= 0 else -(reach + pad))
            ax.text(anchor, i, f"{v:.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9)
        ax.margins(x=0.18)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([m.replace("_", " ") for m in order], fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle("Predicting human typicality from pairwise expert features", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "typicality_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_actual(design: pd.DataFrame, oof_by_model: dict, results: pd.DataFrame, out_dir) -> None:
    """Out-of-fold predictions against the true rating, for the two headline feature sets."""
    shown = [m for m in ("jaccard", "jaccard_plus_profile") if m in oof_by_model]
    if not shown:
        return
    color_map = build_category_color_map(design["category"])
    point_colors = [color_map[c] for c in design["category"]]
    y = design[TARGET].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, len(shown), figsize=(6.2 * len(shown), 5.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, shown):
        predicted = oof_by_model[model]
        r2 = results.loc[results.model == model, "cv_r2_mean"].iloc[0]
        ax.scatter(y, predicted, c=point_colors, s=42, edgecolor="black", linewidth=0.4, alpha=0.85)
        limits = [min(y.min(), predicted.min()) - 0.03, max(y.max(), predicted.max()) + 0.03]
        ax.plot(limits, limits, color="#2f2f2f", linestyle="--", linewidth=1.3, label="perfect prediction")
        ax.set_xlim(limits); ax.set_ylim(limits)
        ax.set_title(f"{model.replace('_', ' ')}  ($R^2$ = {r2:.3f})", fontsize=12)
        ax.set_xlabel("Human typicality", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("Predicted typicality (out of fold)", fontsize=11)
    fig.suptitle("Out-of-fold predictions, colored by category", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "typicality_predicted_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _value(frame: pd.DataFrame, model: str, column: str) -> float:
    """One cell of a results table, NaN when the model or column is absent."""
    if frame.empty or "model" not in frame.columns or column not in frame.columns:
        return np.nan
    row = frame.loc[frame.model == model]
    return float(row[column].iloc[0]) if not row.empty and pd.notna(row[column].iloc[0]) else np.nan


def _summary(results: pd.DataFrame, rank_results: pd.DataFrame) -> dict:
    """This module's row of the cross-scope comparison table, both subchapters."""
    return {
        "cv_r2_jaccard": _value(results, "jaccard", "cv_r2_mean"),
        "cv_r2_jaccard_plus_profile": _value(results, "jaccard_plus_profile", "cv_r2_mean"),
        "delta_cv_r2_profile": _value(results, "jaccard_plus_profile", "delta_cv_r2_vs_jaccard"),
        "rank_acc_jaccard": _value(rank_results, "jaccard", "accuracy"),
        "rank_acc_jaccard_plus_profile": _value(rank_results, "jaccard_plus_profile", "accuracy"),
        "rank_acc_delta_profile": _value(rank_results, "jaccard_plus_profile", "delta_accuracy_vs_jaccard"),
    }


# Bin edges for the accuracy-against-gap curve. The last edge exceeds the 0 to 1 rating
# range so the top bin is closed.
GAP_BINS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.01]


def plot_category_dumbbell(per_category: pd.DataFrame, out_dir) -> None:
    """
    Per held-out category, Jaccard accuracy joined to Jaccard-plus-profile accuracy.

    The pooled mean hides that the layer profile helps enormously in one category and hurts
    badly in another. A dumbbell shows direction and size per category at a glance, which is
    the honest way to present a mean that rests on one or two categories.
    """
    base = per_category[per_category.model == "jaccard"].set_index("category")["accuracy"]
    plus = (per_category[per_category.model == "jaccard_plus_profile"]
            .set_index("category")["accuracy"])
    shared = sorted(set(base.index) & set(plus.index), key=lambda c: plus[c] - base[c])
    if not shared:
        return

    fig, axis = plt.subplots(figsize=(9, 0.5 * len(shared) + 2.5))
    for row, category in enumerate(shared):
        improved = plus[category] >= base[category]
        axis.plot([base[category], plus[category]], [row, row],
                  color="tab:green" if improved else "tab:red", linewidth=2, zorder=1)
        axis.scatter([base[category]], [row], color="grey", s=55, zorder=2,
                     label="Jaccard" if row == 0 else "")
        axis.scatter([plus[category]], [row], color="black", s=55, zorder=3,
                     label="+ layer profile" if row == 0 else "")
        axis.annotate(f"{100 * (plus[category] - base[category]):+.1f}",
                      (max(base[category], plus[category]) + 0.005, row), va="center", fontsize=9)
    axis.axvline(0.5, color="black", linestyle=":", linewidth=1)
    axis.set_yticks(range(len(shared)))
    axis.set_yticklabels(shared)
    axis.set_xlabel("pairwise ranking accuracy (held-out category)")
    axis.set_title("Where the layer profile helps and where it hurts")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "ranker_category_dumbbell.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_gap(per_pair: dict, out_dir) -> None:
    """
    Accuracy against how far apart the two human ratings are.

    Generalises the single RANK_MARGIN cutoff into a curve. A model that has learned the
    construct is right where the humans were decisive and near chance where they were not, so
    a flat line here would mean the accuracy is not tracking the rating at all.
    """
    fig, axis = plt.subplots(figsize=(8, 5))
    for name, frame in per_pair.items():
        if frame.empty:
            continue
        binned = pd.cut(frame["gap"].abs(), GAP_BINS, right=False)
        grouped = frame.groupby(binned, observed=True)["correct"].agg(["mean", "size"])
        centres = [interval.mid for interval in grouped.index]
        axis.plot(centres, grouped["mean"], marker="o", label=name)
    counts = next(iter(per_pair.values()))
    binned = pd.cut(counts["gap"].abs(), GAP_BINS, right=False)
    for interval, size in binned.value_counts().sort_index().items():
        axis.annotate(f"n={int(size)}", (interval.mid, 0.42), fontsize=7, ha="center", color="grey")
    axis.axhline(0.5, color="black", linestyle=":", linewidth=1)
    axis.axvline(RANK_MARGIN, color="grey", linestyle="--", linewidth=1)
    axis.set_xlabel("absolute difference between the two human ratings")
    axis.set_ylabel("pairwise ranking accuracy")
    axis.set_ylim(0.4, 1.0)
    axis.set_title("Accuracy rises with how decisively humans separated the pair")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ranker_accuracy_by_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pair_errors(per_pair: dict, design: pd.DataFrame, out_dir,
                     model: str = "jaccard") -> None:
    """
    Per category, a concept-by-concept grid of which pairs were ranked wrongly.

    Concepts are ordered by human typicality, so a model that gets the ordering roughly right
    shows scattered errors near the diagonal, where neighbouring concepts are hardest to
    separate. Contiguous blocks away from the diagonal mean something is systematically
    inverted, which is the shape to look for in a category scoring below chance.
    """
    frame = per_pair.get(model)
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
    fig.suptitle(f"Misranked pairs, {model}, concepts ordered by human typicality "
                 f"(green correct, red wrong)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "ranker_pair_errors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- subchapter 9.4, human pair similarity as the target ------------------------------
# Feature sets for the pair-similarity regression. These are PAIR quantities (a against b),
# not the concept-to-parent quantities the ranker differences, because the target is a value
# attached to the pair and is symmetric in its two members.
PAIR_FEATURE_SETS = {
    "jaccard": ["jaccard_pct"],
    "layer_profile": ["layer_profile_similarity_pct"],
    "jaccard_plus_profile": ["jaccard_pct", "layer_profile_similarity_pct"],
    "jaccard_plus_profile_z": ["jaccard_pct", "layer_profile_similarity_pct", "layer_profile_z"],
}

# Below this many rated pairs a grouped correlation is not worth reporting.
MIN_PAIR_ROWS = 200


def build_pair_similarity_design(design: pd.DataFrame,
                                 expert_allocation_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per within-category pair that humans actually rated.

    Columns: category, word_a, word_b (canonical order), human_similarity (1 to 7), and the
    three pair features.

    Unlike subchapter 9.2, which predicts a DIRECTION from an antisymmetric difference of
    per-concept features, this predicts a VALUE from symmetric pair features, so the
    quantities are module 5's pairwise matrices rather than module 3's concept-to-parent
    table. Predicting a symmetric target from antisymmetric features would be incoherent.
    """
    lookup = human_pair_lookup()
    available = set(expert_allocation_df["concept"])
    concepts = [c for c in design["concept"] if c in available]
    if len(concepts) < 3:
        return pd.DataFrame()

    jaccard_vector = pair_similarity_vector(expert_allocation_df, concepts)
    similarity_vector, z_vector = pair_layer_profile_vectors(expert_allocation_df, concepts)
    category_of = dict(zip(design["concept"], design["category"]))

    rows, cursor = [], 0
    for first in range(len(concepts)):
        for second in range(first + 1, len(concepts)):
            word_a, word_b = concepts[first], concepts[second]
            key = (word_a, word_b) if word_a < word_b else (word_b, word_a)
            if category_of.get(word_a) == category_of.get(word_b) and key in lookup:
                rows.append({"category": category_of[word_a], "word_a": key[0], "word_b": key[1],
                             "human_similarity": lookup[key],
                             "jaccard_pct": 100.0 * jaccard_vector[cursor],
                             "layer_profile_similarity_pct": similarity_vector[cursor],
                             "layer_profile_z": z_vector[cursor]})
            cursor += 1
    return pd.DataFrame(rows)


def fit_pair_similarity_models(design: pd.DataFrame,
                               expert_allocation_df: pd.DataFrame) -> tuple:
    """
    Predict the human-rated similarity of a pair from the expert similarity of that pair.

    This is the question the supervisor asked directly, and it replaces predicting the
    per-item typicality proxy, which was itself derived by averaging each word's row of this
    same human similarity matrix.

    Categories are held out whole, so the model must generalise to a category it never saw.
    The target admits NO category-identity shortcut: the category is constant within every
    pair, so a model given only category identity scores at chance. That is the trap
    subchapter 9.1 had to guard against, and here it is closed by construction.

    Returns (results, per_category), both empty when too few pairs survive.
    """
    table = build_pair_similarity_design(design, expert_allocation_df)
    if len(table) < MIN_PAIR_ROWS or table["category"].nunique() < 3:
        return pd.DataFrame(), pd.DataFrame()

    ceilings = human_noise_ceiling()
    target = table["human_similarity"].to_numpy(dtype=float)
    groups = table["category"].to_numpy()
    n_groups = table["category"].nunique()
    present = list(table["category"].unique())
    mean_ceiling = float(np.mean([ceilings[c] for c in present]))
    results, per_category = [], []

    for name, columns in PAIR_FEATURE_SETS.items():
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
            per_category.append({"model": name, "category": category,
                                 "n_pairs": int(mask.sum()), "rho": round(category_rho, 4),
                                 "noise_ceiling": round(ceilings[category], 4),
                                 "rho_over_ceiling": round(category_rho / ceilings[category], 4)})
        mean_rho = float(np.mean(per_category_rho))
        results.append({"model": name, "features": " + ".join(columns), "n_pairs": len(table),
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
    return pd.DataFrame(results), pd.DataFrame(per_category)


def execute_module_9_typicality_prediction(scope, similarity_metrics_df: pd.DataFrame,
                                           concept_metadata: pd.DataFrame, model_dir) -> tuple[pd.DataFrame, dict]:
    """Execute Module 9: exploratory human-typicality prediction.

    Consumes module 3's concept-to-parent table, so it runs once per scope like the modules
    feeding it. Returns (results_df, summary_row).
    """
    out_dir = scope_out_dir(model_dir, scope)
    design = build_design_frame(similarity_metrics_df, concept_metadata)
    empty = pd.DataFrame()

    if not design.empty:
        # The centroid features need the expert rows, which build_design_frame does not see.
        # The dropna here is the single shared one: a concept missing any feature is dropped
        # for every model, so the label-word and member-average sets are never fitted on
        # different row counts and their R^2 values stay comparable.
        design = (add_centroid_features(design, scope.expert_df)
                  .dropna(subset=ALL_FEATURES + [TARGET])
                  .reset_index(drop=True))

    if len(design) < MIN_TRAIN_CONCEPTS:
        log.warning(f"  [{scope.label}] Skipping typicality models: {len(design)} usable concepts, "
                    f"need >= {MIN_TRAIN_CONCEPTS}.")
        return empty, scope_summary_row(scope, n_concepts=len(design), **_summary(empty, empty))

    # 9.1, per-concept regression.
    log.info(f"  [{scope.label}] Fitting typicality models on {len(design)} concepts...")
    save_dataframe(design, out_dir / "typicality_model_design.csv")
    results, oof_by_model = fit_typicality_models(design)
    save_dataframe(results, out_dir / "typicality_model_comparison.csv")
    plot_model_comparison(results, out_dir)
    plot_predicted_vs_actual(design, oof_by_model, results, out_dir)

    # 9.3, label word against member centroid.
    plot_reference_comparison(results, out_dir)

    # 9.2, within-category pairwise ranking.
    log.info(f"  [{scope.label}] Fitting pairwise typicality rankers...")
    rank_results, per_category, per_pair = fit_pairwise_rankers(design, scope.expert_df)
    if not rank_results.empty:
        save_dataframe(rank_results, out_dir / "typicality_ranker_comparison.csv")
        save_dataframe(per_category, out_dir / "typicality_ranker_per_category.csv")
        plot_ranker_comparison(rank_results, per_category, out_dir)
        plot_category_dumbbell(per_category, out_dir)
    if per_pair:
        plot_accuracy_by_gap(per_pair, out_dir)
        plot_pair_errors(per_pair, design, out_dir)

    # 9.4, human pair similarity as the target.
    log.info(f"  [{scope.label}] Fitting pair-similarity models against human ratings...")
    pair_results, pair_per_category = fit_pair_similarity_models(design, scope.expert_df)
    if not pair_results.empty:
        save_dataframe(pair_results, out_dir / "pair_similarity_comparison.csv")
        save_dataframe(pair_per_category, out_dir / "pair_similarity_per_category.csv")

    summary = _summary(results, rank_results)
    # The mean over categories, not the pooled correlation. Out-of-fold predictions for a
    # held-out category are miscalibrated in level relative to the categories the model
    # trained on, so pooling them manufactures a between-category component that can be
    # strongly negative while every within-category ordering is correct. Measured here at
    # AP 0.8 on Qwen3: pooled -0.167 against +0.493 per category.
    summary["pair_similarity_rho"] = (float(pair_results["mean_category_spearman"].max())
                                      if not pair_results.empty else np.nan)
    return results, scope_summary_row(scope, n_concepts=len(design), **summary)
