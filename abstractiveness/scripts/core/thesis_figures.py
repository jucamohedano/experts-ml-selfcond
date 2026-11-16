"""
The Chapter 3 figures redrawn from the CSVs of the two reference trees, with the ranges both models share, into
each tree and into one folder per model named as in the thesis source, and the Chapter 4 figures into the same
folder: the tree comparison, the added-value figure and every cell of Studies A and B under one profile metric per
group, both on ranges the two models share, the depth-resolved overlap, and for v3 the same-token table of the intact
and context-only runs.

Used by hand after the reference trees are regenerated, it reads no expert data. The trees of Chapter 4 are rebuilt
from the stored similarity matrices and must reproduce the stored cluster assignment.

Usage, from scripts/:
    python core/thesis_figures.py [--ap 0.6] [--chapter 3|4] [--dataset nopad|v3]

Explanations: documentation/README.md, Thesis figures.
"""

import argparse
import math
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_1_expert_distribution import (plot_category_entropy_gearys, plot_cumulative_layer_distribution,
                                                  plot_cumulative_layer_distribution_sorted,
                                                  plot_sublayer_auc_vs_share)
from modules.module_2_similarities import (HUMAN_PLOT_SUBJECTS, WEIGHTED_MEAN, plot_category_mean_matrix,
                                           plot_human_similarity_decomposition, plot_human_vs_expert,
                                           plot_text_baseline)
from modules.module_5_typicality_prediction import (CLUSTER_CUT_K, COUNT_MATCHED_SPACE, EXPERT_SET_SPACE,
                                                    CHANCE_ACCURACY, plot_added_value, plot_overlap_by_depth,
                                                    plot_selected_metric_comparison, plot_tree_comparison)
from utils.plotting import PRINT_WIDTH, build_category_color_map

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results"
TREES = {"Qwen3-1.7B-Base": "research_plots_qwen_base_richie_hsj", "GPT-2": "research_plots_gpt2_richie_hsj"}
FIGURE_DIRS = {"nopad": "thesis_figures", "v3": "thesis_figures_v3"}
DISTRIBUTION, SHAPE, SUBLAYERS = (f"1_expert_distribution/{name}" for name in (
    "1.1_layer_expert_distribution", "1.2_distribution_shape", "1.3_sublayer_informativeness"))
PAIRWISE, HUMAN = "2_similarities/2.2_pairwise_similarities", "2_similarities/2.3_human_similarity_validation"
FIGURES = [f"{DISTRIBUTION}/mean_expert_layer_distribution_cumulative_by_block.png",
           f"{DISTRIBUTION}/mean_expert_layer_distribution_cumulative_sorted_by_block.png",
           f"{SHAPE}/category_entropy_gearys_by_block.png", f"{SUBLAYERS}/sublayer_auc_vs_share.png",
           f"{PAIRWISE}/jaccard_category_mean_matrix.png", f"{HUMAN}/similarity_vs_jaccard.png",
           f"{HUMAN}/human_similarity_decomposition.png", f"{HUMAN}/human_similarity_text_baseline.png"]
MODULE5 = "5_typicality_prediction"
DENDROGRAMS = f"{MODULE5}/5.3_concept_structure/dendrograms"
DEPTH = f"{MODULE5}/5.4_depth_resolved_overlap"
PAIR_PARTIALS = f"{MODULE5}/5.2_pair_similarity/pair_similarity_partials.csv"
SPACE_NAMES = {EXPERT_SET_SPACE: "Jaccard index", "js_divergence": "Jensen-Shannon divergence", "js_distance":
               "Jensen-Shannon distance", "hellinger": "Hellinger", "profile_jaccard": "Weighted Jaccard index",
               "cosine": "Cosine", "pearson": "Pearson", "spearman": "Spearman", "wasserstein": "Wasserstein"}


def members_by_category(tree: pathlib.Path) -> dict:
    """{category: (normalized entropy, Geary's C)} of the members, in the order category_shape_summary uses."""
    shape = pd.read_csv(tree / SHAPE / "shannon_entropy_concepts_by_block.csv")
    category = pd.read_csv(tree / DISTRIBUTION / "expert_counts_with_metadata.csv").set_index("concept")["category"]
    shape["category"] = shape["concept"].map(category)
    return {name: (group["shannon_entropy_norm"].to_numpy(), group["gearys_c"].to_numpy())
            for name, group in shape.groupby("category")}


def jaccard_test(tree: pathlib.Path) -> dict:
    """Per-category rho, Mantel p and pair count of the Jaccard index, shaped as spearman_test returns them."""
    table = pd.read_csv(tree / HUMAN / "human_similarity_validation.csv")
    table = table[(table.metric == "jaccard") & (table.category != WEIGHTED_MEAN)]
    return {"groups": table.category.tolist(), "rho_k": table.rho.to_numpy(), "p_k": table.mantel_p.to_numpy(),
            "n_k": table.n_pairs.to_numpy()}


def rebuilt_tree(tree: pathlib.Path, space: str, summary: pd.DataFrame, assignment: pd.DataFrame) -> tuple:
    """(title, detail, linkage) of one stored tree, rebuilt from its matrix, its cut compared with the stored one."""
    folder = PAIRWISE if space == EXPERT_SET_SPACE else "2_similarities/layer_profile_measures/heatmaps"
    name = "jaccard_matrix.csv" if space == EXPERT_SET_SPACE else f"layer_profile_{space}_matrix.csv"
    items = list(assignment.concept)
    agreement = pd.read_csv(tree / folder / name, index_col=0).loc[items, items].to_numpy()
    distance = (100.0 - agreement) / 100.0
    np.fill_diagonal(distance, 0.0)
    method = summary.loc[space, "best_linkage"]
    link = hierarchy.linkage(squareform(distance, checks=False), method=method)
    stored = pd.read_csv(tree / DENDROGRAMS / space / f"word_cluster_assignment_k{CLUSTER_CUT_K}.csv")
    cut = hierarchy.fcluster(link, t=CLUSTER_CUT_K, criterion="maxclust")
    agreement_with_stored = adjusted_rand_score(stored.set_index("concept").loc[items, "cluster"], cut)
    if agreement_with_stored < 1.0 - 1e-9:
        raise ValueError(f"{space}: rebuilt cut differs from the stored one, ARI {agreement_with_stored:.4f}")
    members = assignment.category.notna().to_numpy()
    ari = adjusted_rand_score(assignment.category[members], cut[members])
    print(f"  {space}: {method} linkage, ARI {ari:.3f} (stored {summary.loc[space, 'ari_category']})")
    return SPACE_NAMES[space], f"ARI = {ari:.3f}", link


def tree_comparison_figures(trees: dict, figures: pathlib.Path) -> None:
    """Categories, expert sets and the profile metric whose cut best matches the categories, for each model."""
    for model, tree in trees.items():
        out = figures / model
        out.mkdir(parents=True, exist_ok=True)
        summary = pd.read_csv(tree / DENDROGRAMS / "dendrogram_summary.csv").set_index("space")
        assignment = pd.read_csv(tree / DENDROGRAMS / EXPERT_SET_SPACE / f"word_cluster_assignment_k{CLUSTER_CUT_K}.csv")
        best = summary.drop([EXPERT_SET_SPACE, COUNT_MATCHED_SPACE]).ari_category.idxmax()
        print(f"{model}: best profile metric {best}")
        panels = [rebuilt_tree(tree, space, summary, assignment) for space in (EXPERT_SET_SPACE, best)]
        plot_tree_comparison(panels, list(assignment.concept), assignment, build_category_color_map(assignment.category),
                             [out / "concept_tree_comparison.png", out / "concept_tree_comparison.pdf"])


def chapter_4_trees(trees: dict) -> dict:
    """The trees whose module 5 outputs include the added-value table, with a warning for every other one."""
    ready = {}
    for model, tree in trees.items():
        if (tree / MODULE5 / "added_value.csv").exists():
            ready[model] = tree
        else:
            print(f"WARNING {model}: {tree / MODULE5} predates the added-value outputs, rerun module 5 on it")
    return ready


def shared_range(values: np.ndarray) -> tuple:
    """A range holding every finite value, padded by a tenth of its span and rounded outward to 0.05."""
    values = values[np.isfinite(values)]
    pad = (values.max() - values.min()) / 10
    return math.floor((values.min() - pad) * 20) / 20, math.ceil((values.max() + pad) * 20) / 20


def added_value_figures(trees: dict, figures: pathlib.Path) -> None:
    """Each model's added-value figure at half width, every study on the x range both models share."""
    tables = {model: (pd.read_csv(tree / MODULE5 / "added_value.csv"),
                      pd.read_csv(tree / MODULE5 / "added_value_deltas.csv")) for model, tree in trees.items()}
    xlims = {}
    for study in ("A", "B"):
        values = np.concatenate([np.concatenate([deltas[deltas.study == study].delta.to_numpy(),
                                                 summary.loc[summary.study == study, ["ci_low", "ci_high"]].to_numpy().ravel()])
                                 for summary, deltas in tables.values()] or [np.array([])])
        if np.isfinite(values).any():
            xlims[study] = shared_range(values)
    print(f"Added value, shared ranges: {xlims}")
    for model, (summary, deltas) in tables.items():
        out = figures / model
        out.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            plot_added_value(summary, deltas, out / f"added_value.{suffix}", print_width=PRINT_WIDTH["half"], xlims=xlims)


def selected_metric_figures(trees: dict, figures: pathlib.Path) -> None:
    """Each model's cells under one metric per group at full width, every study on the x range both models share."""
    tables = {model: pd.read_csv(tree / MODULE5 / "selected_metric_comparison.csv") for model, tree in trees.items()
              if (tree / MODULE5 / "selected_metric_comparison.csv").exists()}
    xlims = {}
    for study in ("A", "B"):
        values = np.concatenate([table.loc[table.study == study, "value"].to_numpy() for table in tables.values()]
                                + [np.array([CHANCE_ACCURACY] if study == "A" else [])])
        if np.isfinite(values).any():
            xlims[study] = shared_range(values)
    print(f"Selected metrics, shared ranges: {xlims}")
    for model, table in tables.items():
        out = figures / model
        out.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            plot_selected_metric_comparison(table, out / f"selected_metric_comparison.{suffix}",
                                            print_width=PRINT_WIDTH["full"], xlims=xlims)


def depth_figures(trees: dict, figures: pathlib.Path) -> None:
    """Each model's Jaccard index block by block, with the whole-set lines read from the pair partials."""
    for model, tree in trees.items():
        out = figures / model
        out.mkdir(parents=True, exist_ok=True)
        curve = pd.read_csv(tree / DEPTH / "overlap_by_block.csv")
        shared = tree / DEPTH / "shared_unit_depth.csv"
        by_block = pd.read_csv(shared) if shared.exists() else pd.DataFrame()
        partials = pd.read_csv(tree / PAIR_PARTIALS).query("feature == 'jaccard_pct'").set_index("controls").rho
        jaccard_rho = (partials.get("none", np.nan), partials.get("text", np.nan))
        for suffix in ("png", "pdf"):
            plot_overlap_by_depth(curve, by_block, jaccard_rho, out / f"overlap_by_depth.{suffix}",
                                  print_width=PRINT_WIDTH["half"])


def token_control_table(trees: dict, figures: pathlib.Path) -> None:
    """The same-token summary of each model's intact and context-only v3 runs, in one table."""
    frames = []
    for model, tree in trees.items():
        context_only = tree.parent.with_name(f"{tree.parent.name}_context_only") / tree.name
        for run, root in (("intact", tree), ("context_only", context_only)):
            path = root / DEPTH / "same_token_summary.csv"
            if path.exists():
                frames.append(pd.read_csv(path).assign(model=model, run=run))
            else:
                print(f"WARNING {model}, {run}: no {path}, rerun module 5 on it")
    if frames:
        table = pd.concat(frames, ignore_index=True)
        table = table[["model", "run", *[c for c in table.columns if c not in ("model", "run")]]]
        table.to_csv(figures / "same_token_control.csv", index=False)
        print(f"Same-token control: {len(frames)} runs in {figures / 'same_token_control.csv'}")


def main() -> None:
    """Shared ranges first, then every figure of each model, copied into its tree."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ap", default="0.6")
    parser.add_argument("--chapter", type=int, choices=(3, 4))
    parser.add_argument("--dataset", default="nopad", choices=FIGURE_DIRS)
    args = parser.parse_args()
    trees = {model: RESULTS / f"{folder}_{args.dataset}" / f"AP_{args.ap}" for model, folder in TREES.items()}
    figures = RESULTS / FIGURE_DIRS[args.dataset]
    if args.chapter != 3:
        tree_comparison_figures(trees, figures)
        ready = chapter_4_trees(trees)
        added_value_figures(ready, figures)
        selected_metric_figures(ready, figures)
        depth_figures(ready, figures)
        if args.dataset == "v3":
            token_control_table(trees, figures)
    if args.chapter == 4:
        return

    matrices = {model: pd.read_csv(tree / PAIRWISE / "jaccard_category_mean_matrix.csv", index_col=0)
                for model, tree in trees.items()}
    rho = pd.concat([pd.read_csv(tree / HUMAN / f"human_similarity_{name}.csv")[columns]
                     .stack() for tree in trees.values() for name, columns in (
                         ("decomposition", ["member_part", "pair_part", "rho"]),
                         ("text_baseline", ["rho_text_human", "rho_jaccard_human",
                                            "partial_rho_jaccard_human_given_text"]))])
    vmax = math.ceil(max(matrix.to_numpy().max() for matrix in matrices.values()) * 10) / 10
    ylim = (math.floor(rho.min() * 20) / 20, math.ceil(rho.max() * 10) / 10)
    print(f"Shared ranges: Jaccard 0 to {vmax}, rho {ylim[0]} to {ylim[1]}")

    for model, tree in trees.items():
        out = figures / model
        out.mkdir(parents=True, exist_ok=True)
        profile = pd.read_csv(tree / DISTRIBUTION / "mean_expert_layer_distribution_by_block.csv")
        plot_cumulative_layer_distribution(profile, out, "_by_block", print_width=PRINT_WIDTH["half"])
        plot_cumulative_layer_distribution_sorted(profile, out, "_by_block", print_width=PRINT_WIDTH["half"])
        plot_category_entropy_gearys(pd.read_csv(tree / SHAPE / "category_entropy_gearys_by_block.csv"),
                                     members_by_category(tree), out / "category_entropy_gearys_by_block.png")
        plot_sublayer_auc_vs_share(pd.read_csv(tree / SUBLAYERS / "sublayer_informativeness_summary.csv"), out)
        plot_category_mean_matrix(matrices[model], out, vmax=vmax)
        plot_human_vs_expert(pd.read_csv(tree / HUMAN / "similarity_vs_jaccard.csv"), jaccard_test(tree), "Jaccard",
                             out / "similarity_vs_jaccard.png", subject=HUMAN_PLOT_SUBJECTS["jaccard"])
        (out / "similarity_vs_jaccard.csv").unlink()
        plot_human_similarity_decomposition(pd.read_csv(tree / HUMAN / "human_similarity_decomposition.csv"),
                                            out / "human_similarity_decomposition.png", ylim=ylim)
        plot_text_baseline(pd.read_csv(tree / HUMAN / "human_similarity_text_baseline.csv"),
                           out / "human_similarity_text_baseline.png", ylim=ylim)
        for figure in FIGURES:
            shutil.copyfile(out / pathlib.Path(figure).name, tree / figure)
        print(f"{model}: {len(FIGURES)} figures in {out} and {tree}")


if __name__ == "__main__":
    main()
