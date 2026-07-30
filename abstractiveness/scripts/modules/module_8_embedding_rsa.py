"""
Module 8: Expert set versus embedding semantics (second-order RSA).

Research question: does the semantic structure recovered from expert sets match the
semantic structure the model itself carries at a middle layer? The probing and brain
alignment literature converges on intermediate layers holding the richest semantics,
before the final layers specialise for next-token prediction, so a middle layer is the
reference against which the expert set is read.

Two representational systems, compared through their pairwise concept geometry:
  expert side    -- Jaccard similarity of expert sets, keyed on (layer_idx, unit) and
                    pooled across every block of the chosen analysis sublayer. This is
                    the same expert set modules 2 to 7 describe, so module 8 speaks
                    about the thesis's object of study rather than a variant of it.
                    It is fixed within an AP threshold.
  embedding side -- similarity between concept embeddings at one layer, where a concept
                    embedding is the mean over its positive sentences of the max-pooled
                    activation (precomputed, see precompute_concept_embeddings.py).

Pipeline order (mirrored by execute_module_8_embedding_rsa at the bottom):
  1. compute_embedding_rsa_sweep  -- rho per layer, Mantel p, bootstrap CI, noise ceiling
  2. summarise_rsa_sweep          -- headline middle layer, peak plateau, thirds contrast
  3. plot_rsa_layer_sweep         -- rho against depth, ceiling and middle third marked
  4. plot_rdm_comparison          -- expert and embedding similarity matrices side by side
  5. plot_rsa_scatter             -- the concept pairs behind the headline rho

Because the expert side does not move along the sweep, the depth curve cannot be a
disguised plot of expert density, which is a real hazard when both sides vary by layer:
expert counts differ by orders of magnitude across depth and Jaccard on small sets is
noisy. Pooling the expert side across depth also keeps the comparison honest, since
experts at layer L come from Average Precision on the very activations whose centroid
forms the embedding at layer L, so a same-layer comparison would be partly circular.
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform
from statsmodels.nonparametric.smoothers_lowess import lowess

from utils.helpers import (save_dataframe, pair_similarity_vector, pair_shared_count_vector,
                           embedding_rdm, spearman_rdm_mantel, bootstrap_rdm_rho_ci, spearman_brown,
                           layer_slice, filter_expert_data_to_sublayer, _offdiag)
from utils.plot_helpers import (_plot_heatmap_with_leaders, build_category_color_map,
                                fig_width_for, apply_rotated_leader_labels)

log = logging.getLogger(__name__)

# Primary embedding metric plus the robustness variant. The z-scored correlation is
# primary because a mean over positive sentences is dominated by the sentence-structure
# baseline every concept shares, which compresses a raw cosine toward 1.0.
PRIMARY_METRIC = "correlation_zscored"
ROBUSTNESS_METRIC = "cosine_raw"

# Bootstrap intervals are computed only for the chosen analysis sublayer, which carries
# the headline, the plateau and the thirds contrast. Running them for every layer of a
# 196-layer model would dominate the runtime without informing any reported claim.
N_BOOTSTRAP = 1000

# Per-row defaults for the columns compute_embedding_rsa_sweep only fills in
# conditionally (chosen sublayer, primary metric, split-half cache present).
_NAN_ROW_DEFAULTS = {
    "noise_ceiling_splithalf": np.nan,
    "rho_ci95_lo": np.nan,
    "rho_ci95_hi": np.nan,
    "mantel_p_value": np.nan,
    "rho_expert_vs_embedding_holdout": np.nan,
    "mantel_p_value_holdout": np.nan,
    "rho_holdout_ci95_lo": np.nan,
    "rho_holdout_ci95_hi": np.nan,
}


# ---------------------------------------------------------------------------
# 1. The sweep
# ---------------------------------------------------------------------------

def _sublayers_in_order(layer_mapping: pd.DataFrame) -> list:
    """Distinct sublayer names parsed from layer_name, in ascending layer_idx order."""
    frame = layer_mapping.sort_values("layer_idx")
    subs = frame["layer_name"].astype(str).str.extract(r"^\d+\.L\.\d+\.(.+)$")[0]
    seen, ordered = set(), []
    for s in subs:
        if s not in seen:
            seen.add(s); ordered.append(s)
    return ordered


def _layer_frame(layer_mapping: pd.DataFrame) -> pd.DataFrame:
    """One row per layer with its block and sublayer parsed out of layer_name."""
    frame = layer_mapping.sort_values("layer_idx").reset_index(drop=True).copy()
    parts = frame["layer_name"].astype(str).str.extract(r"^\d+\.L\.(\d+)\.(.+)$")
    frame["block"] = parts[0].astype(int)
    frame["sublayer"] = parts[1]
    n_blocks = frame["block"].max() + 1
    frame["depth_fraction"] = frame["block"] / (n_blocks - 1)
    frame["depth_third"] = pd.cut(frame["depth_fraction"], [-0.01, 1 / 3, 2 / 3, 1.01],
                                  labels=["early", "middle", "late"])
    return frame


def _resolve_concepts(sublayer_expert_df: pd.DataFrame, concept_metadata: pd.DataFrame,
                      embedding_cache: dict) -> list:
    """
    The level-2 concepts carrying a category, present in the embedding cache, sorted.
    Fixed from metadata (not the expert frame) so the concept set stays comparable
    across AP folders and along the sweep. See documentation/module_8_embedding_rsa.md.
    """
    cached = set(embedding_cache["concepts"])
    eligible = concept_metadata[concept_metadata["category"].notna()]["concept"]
    concepts = sorted(c for c in eligible.unique() if c in cached)

    missing = sorted(set(eligible.unique()) - cached)
    if missing:
        log.warning(f"  {len(missing)} concepts have no cached embedding and are excluded "
                    f"(first few: {missing[:5]})")
    return concepts


def _embedding_matrix(embedding_cache: dict, key: str, concepts: list, layer: str) -> np.ndarray:
    """Rows of one embedding matrix for ``concepts`` in order, restricted to one layer."""
    row_of = {c: i for i, c in enumerate(embedding_cache["concepts"])}
    rows = [row_of[c] for c in concepts]
    return embedding_cache[key][rows, :][:, layer_slice(embedding_cache, layer)]


def compute_embedding_rsa_sweep(sublayer_expert_df: pd.DataFrame, concept_metadata: pd.DataFrame,
                                embedding_cache: dict, layer_mapping: pd.DataFrame,
                                sublayer_filter: str = None,
                                n_permutations: int = 9999,
                                seed: int = 42) -> tuple:
    """
    Correlate the expert-set similarity structure against the embedding similarity
    structure of sublayer_filter's own layers, and return (sweep_df, context). Every
    sublayer gets an identical, self-contained sweep restricted to its own layers, see
    documentation/module_8_embedding_rsa.md for the Spearman/Mantel derivation, the
    leave-one-out control, and why the Mantel p is not the interesting axis here.

    Columns of sweep_df, one row per (metric, layer):
      rho_expert_vs_embedding : rank correlation over all concept pairs between the expert
                              Jaccard structure and this layer's embedding structure.
                              Both sides are similarities, so a positive value means
                              concepts sharing experts also sit close in embedding space.
      mantel_p_value         : concept-level permutation p for that rho (primary metric
                              only). Pairs are not independent, so concepts are shuffled
                              rather than pairs.
      mantel_p_fdr_bh         : Benjamini-Hochberg across the layers of one metric.
      rho_ci95_lo/hi         : concept bootstrap interval, chosen analysis sublayer only.
      noise_ceiling_splithalf : split-half reliability of this layer's embedding structure,
                              Spearman-Brown corrected. NaN when the cache holds no
                              genuine halves.
      rho_over_noise_ceiling  : rho_expert_vs_embedding / noise_ceiling_splithalf.
      n_expert_units          : size of the expert pool behind the (fixed) expert side.
    """
    rng = np.random.default_rng(seed)
    concepts = _resolve_concepts(sublayer_expert_df, concept_metadata, embedding_cache)
    if len(concepts) < 30:
        log.warning(f"  Only {len(concepts)} usable concepts, skipping module 8")
        return pd.DataFrame(), {}

    # Expert side: built once, fixed for every layer and every permutation below.
    expert_similarity = squareform(pair_similarity_vector(sublayer_expert_df, concepts))
    n_expert_units = int(sublayer_expert_df.groupby(["layer_idx", "unit"]).ngroups)
    log.info(f"  Expert side: {len(concepts)} concepts over {n_expert_units} distinct expert units")

    # Leave-one-layer-out expert sides: removes each layer's self-contribution to the
    # pooled expert set, see doc section 8.4 (this is the curve depth claims are read from).
    experts_per_layer = sublayer_expert_df.groupby("layer_idx", observed=True).size()
    total_expert_rows = int(experts_per_layer.sum())
    leave_one_out = {}
    for layer_idx in experts_per_layer.index:
        others = sublayer_expert_df[sublayer_expert_df["layer_idx"] != layer_idx]
        if others.empty:
            continue
        leave_one_out[layer_idx] = squareform(pair_similarity_vector(others, concepts))

    layers = _layer_frame(layer_mapping)
    layers = layers[layers["layer"].isin(embedding_cache["layers"])].reset_index(drop=True)
    if sublayer_filter is not None:
        layers = layers[layers["sublayer"] == sublayer_filter].reset_index(drop=True)
    has_halves = not np.array_equal(embedding_cache["half_a"], embedding_cache["half_b"])
    if not has_halves:
        log.warning("  Cache holds no genuine split halves, noise ceilings will be NaN")

    rows = []
    chosen_sublayer_vectors = {}
    for record in layers.itertuples():
        layer = record.layer
        is_chosen = sublayer_filter is None or record.sublayer == sublayer_filter
        features = _embedding_matrix(embedding_cache, "mean", concepts, layer)

        for metric in (PRIMARY_METRIC, ROBUSTNESS_METRIC):
            similarity = embedding_rdm(features, metric)
            row = {
                **_NAN_ROW_DEFAULTS,
                "embedding_metric": metric,
                "layer_idx": record.layer_idx,
                "layer_name": str(record.layer_name),
                "block": record.block,
                "sublayer": record.sublayer,
                "depth_fraction": record.depth_fraction,
                "depth_third": str(record.depth_third),
                "is_analysis_sublayer": is_chosen,
                "n_concepts": len(concepts),
                "n_expert_units": n_expert_units,
                "n_experts_this_layer": int(experts_per_layer.get(record.layer_idx, 0)),
                "expert_share_of_pool_pct": 100.0 * experts_per_layer.get(record.layer_idx, 0) / total_expert_rows,
            }

            if metric == PRIMARY_METRIC:
                if is_chosen:
                    chosen_sublayer_vectors[record.layer_name] = _offdiag(similarity)
                result = spearman_rdm_mantel(expert_similarity, similarity,
                                             n_permutations=n_permutations, rng=rng)
                row["rho_expert_vs_embedding"] = result["spearman_rho"]
                row["mantel_p_value"] = result["mantel_p"]
                if has_halves:
                    half_a = embedding_rdm(_embedding_matrix(embedding_cache, "half_a", concepts, layer), metric)
                    half_b = embedding_rdm(_embedding_matrix(embedding_cache, "half_b", concepts, layer), metric)
                    row["noise_ceiling_splithalf"] = spearman_brown(
                        stats.spearmanr(_offdiag(half_a), _offdiag(half_b)).statistic)
                if is_chosen:
                    row["rho_ci95_lo"], row["rho_ci95_hi"] = bootstrap_rdm_rho_ci(
                        expert_similarity, similarity, n_boot=N_BOOTSTRAP, rng=rng)
                    held_out = leave_one_out.get(record.layer_idx)
                    if held_out is not None:
                        loo = spearman_rdm_mantel(held_out, similarity,
                                                  n_permutations=n_permutations, rng=rng)
                        row["rho_expert_vs_embedding_holdout"] = loo["spearman_rho"]
                        row["mantel_p_value_holdout"] = loo["mantel_p"]
                        row["rho_holdout_ci95_lo"], row["rho_holdout_ci95_hi"] = bootstrap_rdm_rho_ci(
                            held_out, similarity, n_boot=N_BOOTSTRAP, rng=rng)
            else:
                # Robustness variant: point estimate only, see doc section 8.2.
                a, b = _offdiag(expert_similarity), _offdiag(similarity)
                row["rho_expert_vs_embedding"] = (np.nan if a.min() == a.max() or b.min() == b.max()
                                       else stats.spearmanr(a, b).statistic)

            ceiling = row["noise_ceiling_splithalf"]
            row["rho_over_noise_ceiling"] = (row["rho_expert_vs_embedding"] / ceiling
                                     if ceiling and np.isfinite(ceiling) and ceiling > 0 else np.nan)
            rows.append(row)

    sweep = pd.DataFrame(rows)
    for metric, block in sweep.groupby("embedding_metric"):
        finite = block["mantel_p_value"].notna()
        if finite.any():
            sweep.loc[block.index[finite], "mantel_p_fdr_bh"] = stats.false_discovery_control(
                block.loc[finite, "mantel_p_value"].to_numpy())

    ordered = ["embedding_metric", "layer_idx", "layer_name", "block", "sublayer", "depth_fraction",
               "depth_third", "is_analysis_sublayer", "n_concepts", "n_expert_units",
               "n_experts_this_layer", "expert_share_of_pool_pct",
               "rho_expert_vs_embedding", "rho_ci95_lo", "rho_ci95_hi", "mantel_p_value", "mantel_p_fdr_bh",
               "rho_expert_vs_embedding_holdout", "rho_holdout_ci95_lo", "rho_holdout_ci95_hi", "mantel_p_value_holdout",
               "noise_ceiling_splithalf", "rho_over_noise_ceiling"]
    sweep = sweep[[c for c in ordered if c in sweep.columns]]
    sweep = sweep.sort_values(["embedding_metric", "layer_idx"]).reset_index(drop=True)

    # How much the embedding geometry moves with depth, see doc section 8.3.
    stability = np.nan
    names = list(chosen_sublayer_vectors)
    if len(names) > 1:
        pairs = [stats.spearmanr(chosen_sublayer_vectors[a], chosen_sublayer_vectors[b]).statistic
                 for i, a in enumerate(names) for b in names[i + 1:]]
        stability = float(np.median(pairs))
        log.info(f"  Embedding geometry across depth: median cross-layer similarity "
                 f"{stability:.3f} (range {min(pairs):.3f} to {max(pairs):.3f})")

    context = {"concepts": concepts, "expert_similarity": expert_similarity,
               "n_expert_units": n_expert_units, "layers": layers,
               "sublayer_filter": sublayer_filter, "embedding_depth_stability": stability,
               "expert_rows": sublayer_expert_df}
    return sweep, context


# ---------------------------------------------------------------------------
# 2. Summary
# ---------------------------------------------------------------------------

def _count_matched_rho(sub_df: pd.DataFrame, concepts: list,
                       reference_similarity: np.ndarray, k: int) -> float:
    """
    Spearman rho of the sublayer's top-k-by-AP expert Jaccard against a fixed reference
    embedding similarity matrix. k equalizes expert mass across sublayers, see
    documentation/module_8_embedding_rsa.md section 8.4 for the count-matched control.
    """
    top_k = sub_df.nlargest(k, "ap")
    expert_sim = squareform(pair_similarity_vector(top_k, concepts))
    a, b = _offdiag(expert_sim), _offdiag(reference_similarity)
    if a.min() == a.max() or b.min() == b.max():
        return np.nan
    return float(stats.spearmanr(a, b).statistic)


def _middle_layer(chosen: pd.DataFrame) -> pd.Series:
    """
    The chosen sublayer's layer in the block at the midpoint of depth (derived from the
    block count, not hard coded).
    """
    middle_block = int(chosen["block"].max() + 1) // 2
    return chosen.iloc[(chosen["block"] - middle_block).abs().argmin()]


def summarise_rsa_sweep(sweep: pd.DataFrame, context: dict, n_permutations: int = 9999,
                        seed: int = 42) -> pd.DataFrame:
    """
    Headline table: the middle layer's RSA, the peak and its plateau, and whether the
    middle third of the network really does beat the outer thirds. Depth claims (peak,
    plateau, thirds) are read from the LEAVE-ONE-OUT curve, not the pooled headline, see
    documentation/module_8_embedding_rsa.md section 8.4 for why (the density confound).

    peak_plateau_layers: every chosen-sublayer layer whose bootstrap interval overlaps
      the peak layer's (a bare argmax would overclaim a single best layer).
    middle_third_minus_outer_rho / middle_vs_outer_perm_p: mean rho in the middle third
      minus the mean over the outer thirds, with a permutation p from shuffling the
      layer-to-third assignment. Restricted to the chosen sublayer.
    pooled_rho_vs_expert_share_spearman: how much the pooled depth curve is explained by
      expert-share-of-pool alone (the confound the leave-one-out curve corrects for).
    """
    primary = sweep[(sweep["embedding_metric"] == PRIMARY_METRIC) & sweep["is_analysis_sublayer"]]
    primary = primary.dropna(subset=["rho_expert_vs_embedding"])
    if primary.empty:
        return pd.DataFrame()

    # How much of the pooled depth curve is just expert density.
    usable = primary.dropna(subset=["expert_share_of_pool_pct", "rho_expert_vs_embedding"])
    density_confound = (stats.spearmanr(usable["expert_share_of_pool_pct"], usable["rho_expert_vs_embedding"]).statistic
                        if len(usable) > 2 and usable["expert_share_of_pool_pct"].nunique() > 1 else np.nan)

    depth_uses_holdout = primary["rho_expert_vs_embedding_holdout"].notna().any()
    depth_column = "rho_expert_vs_embedding_holdout" if depth_uses_holdout else "rho_expert_vs_embedding"
    ci_lo, ci_hi = (("rho_holdout_ci95_lo", "rho_holdout_ci95_hi") if depth_uses_holdout
                    else ("rho_ci95_lo", "rho_ci95_hi"))
    depth = primary.dropna(subset=[depth_column])

    middle = _middle_layer(primary)
    peak = depth.loc[depth[depth_column].idxmax()]

    if np.isfinite(peak[ci_lo]) and np.isfinite(peak[ci_hi]):
        overlaps = depth[(depth[ci_hi] >= peak[ci_lo]) & (depth[ci_lo] <= peak[ci_hi])]
    else:
        overlaps = depth.loc[[peak.name]]

    thirds = depth.groupby("depth_third", observed=True)[depth_column].mean()
    in_middle = (depth["depth_third"] == "middle").to_numpy()
    rho = depth[depth_column].to_numpy()
    if in_middle.any() and (~in_middle).any():
        observed = rho[in_middle].mean() - rho[~in_middle].mean()
        rng = np.random.default_rng(seed)
        n_at_least = 0
        for _ in range(n_permutations):
            shuffled = rng.permutation(in_middle)
            if rho[shuffled].mean() - rho[~shuffled].mean() >= observed:
                n_at_least += 1
        middle_p = (1 + n_at_least) / (1 + n_permutations)
    else:
        observed, middle_p = np.nan, np.nan

    return pd.DataFrame([{
        "middle_layer_name": middle["layer_name"],
        "middle_layer_idx": middle["layer_idx"],
        "middle_block": middle["block"],
        "middle_spearman_rho": middle["rho_expert_vs_embedding"],
        "middle_rho_ci_lo": middle["rho_ci95_lo"],
        "middle_rho_ci_hi": middle["rho_ci95_hi"],
        "middle_mantel_p": middle["mantel_p_value"],
        "middle_noise_ceiling": middle["noise_ceiling_splithalf"],
        "middle_rho_normalized": middle["rho_over_noise_ceiling"],
        "middle_spearman_rho_loo": middle["rho_expert_vs_embedding_holdout"],
        "depth_curve_column": depth_column,
        "pooled_rho_vs_expert_share_spearman": density_confound,
        "median_cross_layer_emb_similarity": context.get("embedding_depth_stability", np.nan),
        "peak_layer_name": peak["layer_name"],
        "peak_block": peak["block"],
        "peak_depth_frac": peak["depth_fraction"],
        "peak_spearman_rho": peak[depth_column],
        "peak_plateau_layers": ";".join(overlaps["layer_name"].astype(str)),
        "peak_plateau_n_layers": len(overlaps),
        "peak_plateau_depth_lo": overlaps["depth_fraction"].min(),
        "peak_plateau_depth_hi": overlaps["depth_fraction"].max(),
        "mean_rho_early": thirds.get("early", np.nan),
        "mean_rho_middle": thirds.get("middle", np.nan),
        "mean_rho_late": thirds.get("late", np.nan),
        "middle_third_minus_outer_rho": observed,
        "middle_vs_outer_perm_p": middle_p,
        "n_concepts": int(primary["n_concepts"].iloc[0]),
        "n_expert_units": int(primary["n_expert_units"].iloc[0]),
        "n_layers_swept": int(len(sweep[sweep["embedding_metric"] == PRIMARY_METRIC])),
    }])


# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------

def plot_rsa_layer_sweep(sweep: pd.DataFrame, summary: pd.DataFrame, out_path) -> None:
    """
    Spearman rho against depth for one sublayer's own layers, with the bootstrap
    band, the split-half noise ceiling and the middle third of the network marked.
    Layers whose expert side was degenerate leave a gap rather than plotting as zero,
    since no data is not the same as no relationship.
    """
    primary = sweep[(sweep["embedding_metric"] == PRIMARY_METRIC) & sweep["is_analysis_sublayer"]].copy()
    if primary.empty:
        return
    primary = primary.sort_values("layer_idx").reset_index(drop=True)
    x = np.arange(len(primary))

    fig, ax = plt.subplots(figsize=(fig_width_for(len(primary), 0.9, min_w=12.0), 7))

    middle_mask = primary["depth_third"] == "middle"
    if middle_mask.any():
        ax.axvspan(x[middle_mask].min() - 0.5, x[middle_mask].max() + 0.5,
                   color="#cfe3f3", alpha=0.45, zorder=0, label="Middle third of depth")

    if primary["noise_ceiling_splithalf"].notna().any():
        ax.plot(x, primary["noise_ceiling_splithalf"], color="#8a8a8a", linestyle=":", linewidth=1.6,
                zorder=2, label="Noise ceiling (split-half)")

    if primary[["rho_ci95_lo", "rho_ci95_hi"]].notna().all(axis=1).any():
        ax.fill_between(x, primary["rho_ci95_lo"], primary["rho_ci95_hi"],
                        color="#003f5c", alpha=0.18, zorder=1, label="95% concept bootstrap")

    ax.plot(x, primary["rho_expert_vs_embedding"], color="#003f5c", marker="o", markersize=5,
            linewidth=2, zorder=3, label="Expert set vs embedding (pooled)")

    # The leave-one-out curve is the one depth should be read from, so it is drawn
    # alongside the pooled curve rather than replacing it: seeing the two together is
    # what shows how much of the pooled shape was the layer's own contribution.
    if primary["rho_expert_vs_embedding_holdout"].notna().any():
        if primary[["rho_holdout_ci95_lo", "rho_holdout_ci95_hi"]].notna().all(axis=1).any():
            ax.fill_between(x, primary["rho_holdout_ci95_lo"], primary["rho_holdout_ci95_hi"],
                            color="#ff6361", alpha=0.15, zorder=1)
        ax.plot(x, primary["rho_expert_vs_embedding_holdout"], color="#ff6361", marker="s", markersize=5,
                linewidth=2, zorder=3, label="Held-out expert set (layer excluded)")

    if primary["expert_share_of_pool_pct"].notna().any():
        share_ax = ax.twinx()
        share_ax.bar(x, primary["expert_share_of_pool_pct"], color="#8a8a8a", alpha=0.20, zorder=0,
                     width=0.6)
        share_ax.set_ylabel("Share of the pooled expert set % (bars)", color="#6a6a6a")
        share_ax.tick_params(axis="y", colors="#6a6a6a")
        # Bars are held to the lower sixth of the panel so they read as context for the
        # curves rather than competing with them, and so the legend has clear space.
        share_ax.set_ylim(0, max(1.0, primary["expert_share_of_pool_pct"].max() * 6))
        share_ax.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)

    if not summary.empty:
        for name, colour, label in [(summary["middle_layer_name"].iloc[0], "#ffa600", "Middle layer"),
                                    (summary["peak_layer_name"].iloc[0], "#bc5090", "Peak")]:
            hit = primary.index[primary["layer_name"] == name]
            if len(hit):
                i = hit[0]
                y = primary["rho_expert_vs_embedding"].iloc[i]
                ax.scatter([i], [y], s=150, marker="D", color=colour, zorder=5,
                           edgecolor="white", linewidth=1.2, label=f"{label} ({name})")
                ax.plot([i, i], [ax.get_ylim()[0], y], color=colour, linestyle="--",
                        linewidth=1.0, alpha=0.7, zorder=4)

    ax.set_ylabel("Spearman rho, expert set against embedding (curves)")
    ax.set_xlabel("Layer (in depth order)")
    ax.set_title("Expert-set semantics against embedding semantics across depth\n"
                 "(grey bars: each layer's share of the pooled expert set)")
    ax.axhline(0.0, color="#999999", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    apply_rotated_leader_labels(ax, list(primary["layer_name"]), axis="x", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sublayer_comparison(comparison_df: pd.DataFrame, sweeps_by_sublayer: dict, out_path) -> None:
    """
    Top: sublayers ranked by rho, full vs count-matched grouped bars, noise ceiling
    reference line, expert count annotated. Bottom: one small-multiple rho-against-depth
    panel per sublayer (pooled and held-out curves), so the ranking and each sublayer's
    own depth profile are visible together.
    """
    import math
    order = comparison_df.sort_values("rho_full", ascending=True)
    y = np.arange(len(order))
    n_sub = len(order)
    n_rows_sm = math.ceil(n_sub / 3)
    fig = plt.figure(figsize=(12, 4 + 2.2 * n_rows_sm))
    gs = fig.add_gridspec(1 + n_rows_sm, 3, height_ratios=[1.4] + [1] * n_rows_sm)

    top = fig.add_subplot(gs[0, :])
    top.barh(y - 0.2, order["rho_full"], height=0.38, color="#003f5c", label="Full expert set")
    top.barh(y + 0.2, order["rho_count_matched"], height=0.38, color="#ffa600", label="Count-matched")
    if order["noise_ceiling_splithalf"].notna().any():
        top.axvline(order["noise_ceiling_splithalf"].median(), color="#8a8a8a",
                    linestyle=":", label="Noise ceiling")
    top.set_yticks(y)
    top.set_yticklabels(order["sublayer"])
    for yi, n in zip(y, order["n_expert_units"]):
        top.annotate(f"n={n:,}", (0.005, yi), fontsize=8, va="center", color="#555")
    top.set_xlabel("Spearman rho, expert set against embedding (middle block)")
    top.set_title("Which sublayer's expert set best resembles the embedding semantics")
    top.legend(loc="lower right", frameon=True)

    for i, sublayer in enumerate(comparison_df["sublayer"]):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        sw = sweeps_by_sublayer[sublayer]
        sw = sw[sw["embedding_metric"] == PRIMARY_METRIC].sort_values("layer_idx")
        x = np.arange(len(sw))
        ax.plot(x, sw["rho_expert_vs_embedding"], color="#003f5c", linewidth=1.6)
        if sw["rho_expert_vs_embedding_holdout"].notna().any():
            ax.plot(x, sw["rho_expert_vs_embedding_holdout"], color="#ff6361", linewidth=1.4)
        ax.set_title(sublayer, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _category_style(concepts: list, concept_metadata: pd.DataFrame) -> tuple:
    """Category colours, legend and block boundaries, matching module 5's treatment."""
    color_map = build_category_color_map(concept_metadata.dropna(subset=["category"])["category"])
    concept_to_cat = concept_metadata.set_index("concept")["category"].to_dict()
    categories = [concept_to_cat.get(c) for c in concepts]
    colors = [color_map.get(cat, "#000000") for cat in categories]
    present = set(concept_metadata.dropna(subset=["category"])["category"])
    legend = {cat: col for cat, col in color_map.items() if cat in present}
    boundaries = [i for i in range(1, len(concepts)) if categories[i] != categories[i - 1]]
    return colors, legend, boundaries


def _percentile_offdiag(matrix: np.ndarray) -> np.ndarray:
    """
    Copy of ``matrix`` with its off-diagonal replaced by each entry's percentile rank
    within that matrix's own off-diagonal distribution, in [0, 1] (diagonal set to 1.0).
    Puts matrices with different native ranges on the same visual scale, matching the
    rank-based RSA statistic itself. Equal color intensity means equal RELATIVE rank.
    """
    n = matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    ranks = stats.rankdata(matrix[iu]) / len(matrix[iu])
    out = np.zeros_like(matrix, dtype=float)
    out[iu] = ranks
    out = out + out.T
    np.fill_diagonal(out, 1.0)
    return out


def plot_rdm_comparison(expert_similarity: np.ndarray, embedding_similarity: np.ndarray,
                        concepts: list, concept_metadata: pd.DataFrame, layer_name: str,
                        rsa_dir) -> None:
    """
    The two similarity matrices in the same concept order (sorted by category, so
    agreement shows up as matching block structure), one shared colormap. Two variants
    are written: percentile (rank-rescaled, see _percentile_offdiag) and minmax (each
    matrix's own raw autoscale).
    """
    order = (concept_metadata[concept_metadata["concept"].isin(concepts)]
             .sort_values(["category", "concept"])["concept"].tolist())
    index = {c: i for i, c in enumerate(concepts)}
    permutation = [index[c] for c in order]
    colors, legend, boundaries = _category_style(order, concept_metadata)

    SHARED_CMAP = "magma"
    for matrix, name, title in [
            (expert_similarity, "expert_jaccard", "Expert-set Jaccard similarity"),
            (embedding_similarity, "embedding_similarity", f"Embedding similarity at {layer_name}")]:
        ordered = matrix[np.ix_(permutation, permutation)]
        _plot_heatmap_with_leaders(_percentile_offdiag(ordered), order, title + " (percentile)",
                                   rsa_dir / f"rdm_{name}_percentile.png", SHARED_CMAP,
                                   concept_colors=colors, color_legend=legend,
                                   category_boundaries=boundaries)
        _plot_heatmap_with_leaders(ordered, order, title + " (min-max)",
                                   rsa_dir / f"rdm_{name}_minmax.png", SHARED_CMAP,
                                   concept_colors=colors, color_legend=legend,
                                   category_boundaries=boundaries)


def _scatter_rankrank(x: np.ndarray, y: np.ndarray, out_path, layer_name: str) -> None:
    """Rank-rank density, the relationship Spearman rho actually measures, avoiding the
    zero-inflation of raw Jaccard dominating the visual."""
    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(stats.rankdata(x), stats.rankdata(y), gridsize=50, bins="log", cmap="magma")
    fig.colorbar(hb, ax=ax, label="Concept pairs (log)")
    ax.set_xlabel("Rank of expert Jaccard")
    ax.set_ylabel(f"Rank of embedding similarity ({layer_name})")
    ax.set_title("Rank-rank density (the relationship Spearman rho measures)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _scatter_hexbin(x: np.ndarray, y: np.ndarray, same: np.ndarray, out_path, layer_name: str) -> None:
    """Fixed hexbin (perceptually-uniform colormap instead of grey), split into same vs
    different category panels with a binned-median trend, a straight OLS fit, and a
    LOWESS smooth in each, so a reader sees the non-parametric trend against the fit a
    naive linear correlation would draw."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    hb = None
    for ax, mask, title in [(axes[0], same, "Same category"), (axes[1], ~same, "Different category")]:
        if not mask.any():
            continue
        hb = ax.hexbin(x[mask], y[mask], gridsize=45, bins="log", cmap="magma", mincnt=1)
        order = np.argsort(x[mask])
        xb, yb = x[mask][order], y[mask][order]
        edges = np.quantile(xb, np.linspace(0, 1, 11))
        centers = 0.5 * (edges[:-1] + edges[1:])
        meds = [np.median(yb[(xb >= lo) & (xb <= hi)]) if ((xb >= lo) & (xb <= hi)).any() else np.nan
                for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(centers, meds, color="#ffa600", linewidth=2, label="Binned median")
        if xb.std() > 0:
            pearson_r = stats.pearsonr(xb, yb).statistic
            slope, intercept = np.polyfit(xb, yb, 1)
            xs_line = np.array([xb.min(), xb.max()])
            ax.plot(xs_line, intercept + slope * xs_line, color="#2f2f2f", linestyle="--",
                    linewidth=1.6, label=f"OLS (r={pearson_r:.2f})")
            smoothed = lowess(yb, xb, frac=0.4, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="#00b3b3", linestyle=":",
                    linewidth=2.2, label="LOWESS")
        ax.set_title(title)
        ax.set_xlabel("Expert Jaccard")
        ax.legend(loc="upper left")
    axes[0].set_ylabel(f"Embedding similarity ({layer_name})")
    if hb is not None:
        fig.colorbar(hb, ax=axes, label="Concept pairs (log)")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _scatter_binned(shared: np.ndarray, y: np.ndarray, same: np.ndarray, out_path, layer_name: str) -> None:
    """Median embedding similarity (with IQR band) binned by raw shared-expert count,
    shown separately for same- and different-category pairs, each with a straight OLS
    fit and a LOWESS smooth overlaid. The most interpretable variant: turns 19k
    overplotted points into one legible statement.

    The x-axis is categorical (bin label, evenly spaced), not the real shared-expert
    count, so OLS and LOWESS are fit on the RAW (shared, y) pairs in real count units
    and only evaluated at each bin's representative count (BIN_CENTERS_FOR_FIT) before
    being plotted at the same evenly-spaced tick positions as the median line.
    """
    palette = {"same": "#ffa600", "diff": "#003f5c"}
    fig, ax = plt.subplots(figsize=(9, 7))
    bins = [0, 1, 2, 3, 5, 10, np.inf]
    labels = ["0", "1", "2", "3-4", "5-9", "10+"]
    BIN_CENTERS_FOR_FIT = np.array([0, 1, 2, 3.5, 7, 15])
    idx = np.digitize(shared, bins) - 1
    xs = np.arange(len(labels))
    for mask, key, name in [(same, "same", "Same category"), (~same, "diff", "Different category")]:
        med = [np.median(y[mask & (idx == b)]) if (mask & (idx == b)).any() else np.nan
               for b in range(len(labels))]
        lo = [np.percentile(y[mask & (idx == b)], 25) if (mask & (idx == b)).any() else np.nan
              for b in range(len(labels))]
        hi = [np.percentile(y[mask & (idx == b)], 75) if (mask & (idx == b)).any() else np.nan
              for b in range(len(labels))]
        ax.plot(xs, med, color=palette[key], marker="o", linewidth=2, label=f"{name}, median")
        ax.fill_between(xs, lo, hi, color=palette[key], alpha=0.18)

        xb, yb = shared[mask], y[mask]
        if xb.std() > 0:
            slope, intercept = np.polyfit(xb, yb, 1)
            ax.plot(xs, intercept + slope * BIN_CENTERS_FOR_FIT, color=palette[key],
                    linestyle="--", linewidth=1.4, alpha=0.85, label=f"{name}, OLS")
            smoothed = lowess(yb, xb, frac=0.4, return_sorted=True)
            lowess_at_centers = np.interp(BIN_CENTERS_FOR_FIT, smoothed[:, 0], smoothed[:, 1])
            ax.plot(xs, lowess_at_centers, color=palette[key], linestyle=":", linewidth=1.8,
                    alpha=0.85, label=f"{name}, LOWESS")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Shared experts between the pair")
    ax.set_ylabel(f"Embedding similarity ({layer_name})")
    ax.set_title("More shared experts, higher embedding similarity")
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rsa_scatter(expert_similarity: np.ndarray, embedding_similarity: np.ndarray, shared_counts: np.ndarray,
                     concepts: list, concept_metadata: pd.DataFrame, layer_name: str,
                     out_path) -> None:
    """
    The concept pairs behind the headline rho, three variants for selection (the plain
    hexbin on raw Jaccard is unreadable: Jaccard is zero-inflated so almost every point
    sits at x near 0). ``shared_counts`` is the raw shared-expert count per pair (not
    Jaccard), used for the binned variant's more interpretable x-axis.
    """
    iu = np.triu_indices(len(concepts), k=1)
    x, y = expert_similarity[iu], embedding_similarity[iu]
    if x.min() == x.max():
        return

    category_of = concept_metadata.set_index("concept")["category"].to_dict()
    cats = np.array([category_of.get(c) for c in concepts])
    same = (cats[:, None] == cats[None, :])[iu]

    base = out_path.with_suffix("")
    _scatter_binned(shared_counts, y, same, base.with_name(base.name + "_binned.png"), layer_name)
    _scatter_rankrank(x, y, base.with_name(base.name + "_rankrank.png"), layer_name)
    _scatter_hexbin(x, y, same, base.with_name(base.name + "_hexbin.png"), layer_name)


# ---------------------------------------------------------------------------
# 4. Module entry point
# ---------------------------------------------------------------------------

def _write_analysis_sublayer_extras(summary: pd.DataFrame, context: dict, embedding_cache: dict,
                                    concept_metadata: pd.DataFrame, rsa_dir) -> None:
    """
    The RDM heatmaps, pairs scatter and raw similarity-matrix CSVs for the middle layer
    of the analysis sublayer's own sweep. Kept out of every other sublayer's run since
    these are the headline-illustrating extras, not part of the sweep/summary itself.
    """
    headline = summary["middle_layer_name"].iloc[0]
    layer = context["layers"].set_index("layer_name").loc[headline, "layer"]
    features = _embedding_matrix(embedding_cache, "mean", context["concepts"], layer)
    similarity = embedding_rdm(features, PRIMARY_METRIC)
    plot_rdm_comparison(context["expert_similarity"], similarity, context["concepts"],
                        concept_metadata, headline, rsa_dir)
    shared_counts = pair_shared_count_vector(context["expert_rows"], context["concepts"])
    plot_rsa_scatter(context["expert_similarity"], similarity, shared_counts, context["concepts"],
                     concept_metadata, headline, rsa_dir / "rsa_pairs_scatter.png")

    rdms = rsa_dir / "rdms"
    rdms.mkdir(parents=True, exist_ok=True)
    save_dataframe(pd.DataFrame(context["expert_similarity"], index=context["concepts"],
                                columns=context["concepts"]),
                   rdms / "expert_jaccard_matrix.csv", index=True)
    save_dataframe(pd.DataFrame(similarity, index=context["concepts"],
                                columns=context["concepts"]),
                   rdms / f"embedding_similarity_{headline}.csv", index=True)


def execute_module_8_embedding_rsa(sublayer_expert_df: pd.DataFrame, concept_metadata: pd.DataFrame,
                                   rsa_dir, embedding_cache: dict, layer_mapping: pd.DataFrame,
                                   sublayer_filter: str = None, full_expert_df: pd.DataFrame = None,
                                   sublayer_rank: dict = None,
                                   n_permutations: int = 9999, seed: int = 42) -> tuple:
    """
    Execute Module 8: expert set versus embedding semantics. See
    documentation/module_8_embedding_rsa.md for the full pipeline description.

    Loops over every sublayer of the architecture, each building its own expert Jaccard
    matrix (from ``full_expert_df`` when available, otherwise ``sublayer_expert_df``)
    against that sublayer's own embedding layers. The analysis sublayer
    (``sublayer_filter``) writes to ``rsa_dir`` directly plus the headline RDM/scatter
    extras, every other sublayer lands under ``rsa_dir / "sublayers" / <rank>_<name>``,
    where rank comes from ``sublayer_rank`` (the run-wide expert-count ranking frozen at
    the most lenient AP threshold) so the folder names match every other module's.

    Returns (sweep_df, summary_df) for the analysis sublayer, both empty when the cache
    is unavailable.
    """
    if not embedding_cache:
        log.warning("  No concept embedding cache available, skipping module 8. "
                    "Build it with scripts/precompute_concept_embeddings.py")
        return pd.DataFrame(), pd.DataFrame()

    cached_layers = set(embedding_cache["layers"])
    mapped_layers = set(layer_mapping["layer"])
    if cached_layers != mapped_layers:
        raise ValueError(
            f"Embedding cache layers do not match the layer mapping "
            f"({len(cached_layers - mapped_layers)} only in cache, "
            f"{len(mapped_layers - cached_layers)} only in mapping). The raw layer string is "
            f"the sole join key between the cache and the expert frame, so a mismatch would "
            f"silently misalign every result. Rebuild the cache for this config.")

    experts_by_sublayer = full_expert_df if full_expert_df is not None else sublayer_expert_df
    sublayers = _sublayers_in_order(layer_mapping)
    final_sweep, final_summary = pd.DataFrame(), pd.DataFrame()
    per_sublayer = {}
    sweeps_by_sublayer = {}
    for sublayer in sublayers:
        sub_df = filter_expert_data_to_sublayer(experts_by_sublayer, sublayer)
        if sub_df.empty:
            log.warning(f"  No experts for sublayer {sublayer} at this AP, skipping it")
            continue
        is_analysis = (sublayer == sublayer_filter)
        rank = (sublayer_rank or {}).get(sublayer)
        dir_name = f"{rank}_{sublayer}" if rank else sublayer
        target_dir = rsa_dir if is_analysis else (rsa_dir / "sublayers" / dir_name)
        target_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"  [{sublayer}] correlating expert-set similarity against embedding similarity...")
        sweep, context = compute_embedding_rsa_sweep(
            sub_df, concept_metadata, embedding_cache, layer_mapping,
            sublayer_filter=sublayer, n_permutations=n_permutations, seed=seed)
        if sweep.empty:
            continue
        save_dataframe(sweep, target_dir / "embedding_rsa_layer_sweep.csv")
        sweeps_by_sublayer[sublayer] = sweep

        summary = summarise_rsa_sweep(sweep, context, n_permutations=n_permutations, seed=seed)
        if not summary.empty:
            save_dataframe(summary, target_dir / "embedding_rsa_summary.csv")
            row = summary.iloc[0]
            log.info(f"    [{sublayer}] Middle layer {row['middle_layer_name']}: "
                     f"rho={row['middle_spearman_rho']:.3f}, peak at {row['peak_layer_name']} "
                     f"(rho={row['peak_spearman_rho']:.3f}, depth {row['peak_depth_frac']:.2f}), "
                     f"plateau spans {row['peak_plateau_n_layers']} layers")

        plot_rsa_layer_sweep(sweep, summary, target_dir / "rsa_layer_sweep.png")

        if not summary.empty:
            own_rows = sweep[(sweep["embedding_metric"] == PRIMARY_METRIC) & sweep["is_analysis_sublayer"]]
            middle = _middle_layer(own_rows)
            layer = context["layers"].set_index("layer_name").loc[middle["layer_name"], "layer"]
            ref_sim = embedding_rdm(_embedding_matrix(embedding_cache, "mean", context["concepts"], layer),
                                    PRIMARY_METRIC)
            per_sublayer[sublayer] = {
                "sub_df": sub_df, "concepts": context["concepts"], "ref_sim": ref_sim,
                "rho_full": float(middle["rho_expert_vs_embedding"]),
                "noise_ceiling": float(middle["noise_ceiling_splithalf"]),
                "rho_over_ceiling": float(middle["rho_over_noise_ceiling"]),
                "n_expert_units": int(sweep["n_expert_units"].iloc[0]),
                "middle_layer_name": str(middle["layer_name"]),
                "peak_layer_name": str(summary["peak_layer_name"].iloc[0]),
            }

        if is_analysis:
            final_sweep, final_summary = sweep, summary
            if not summary.empty:
                _write_analysis_sublayer_extras(summary, context, embedding_cache,
                                                concept_metadata, rsa_dir)

    if per_sublayer:
        k = min(len(v["sub_df"]) for v in per_sublayer.values())
        comparison_rows = []
        for sub_name, v in per_sublayer.items():
            comparison_rows.append({
                "sublayer": sub_name,
                "is_analysis_sublayer": sub_name == sublayer_filter,
                "n_expert_units": v["n_expert_units"],
                "k_count_matched": k,
                "rho_full": v["rho_full"],
                "rho_count_matched": _count_matched_rho(v["sub_df"], v["concepts"], v["ref_sim"], k),
                "noise_ceiling_splithalf": v["noise_ceiling"],
                "rho_over_noise_ceiling": v["rho_over_ceiling"],
                "middle_layer_name": v["middle_layer_name"],
                "peak_layer_name": v["peak_layer_name"],
            })
        comparison = pd.DataFrame(comparison_rows).sort_values("rho_full", ascending=False)
        save_dataframe(comparison, rsa_dir / "sublayer_comparison.csv")
        log.info(f"  Sublayer comparison written, top sublayer by rho: "
                 f"{comparison.iloc[0]['sublayer']} (rho={comparison.iloc[0]['rho_full']:.3f})")
        plot_sublayer_comparison(comparison, sweeps_by_sublayer, rsa_dir / "sublayer_comparison.png")

    return final_sweep, final_summary
