"""
The Chapter 3 figures redrawn from the CSVs of the two reference trees, with the ranges both models share, into
each tree and into one folder per model named as in the thesis source.

Used by hand after the reference trees are regenerated, it recomputes nothing and reads no expert data.

Usage, from scripts/:
    python core/thesis_figures.py [--ap 0.6]

Explanations: documentation/README.md, Thesis figures.
"""

import argparse
import math
import pathlib
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_1_expert_distribution import (plot_category_entropy_gearys, plot_cumulative_layer_distribution,
                                                  plot_cumulative_layer_distribution_sorted,
                                                  plot_sublayer_auc_vs_share)
from modules.module_2_similarities import (HUMAN_PLOT_SUBJECTS, WEIGHTED_MEAN, plot_category_mean_matrix,
                                           plot_human_similarity_decomposition, plot_human_vs_expert,
                                           plot_text_baseline)
from utils.plotting import PRINT_WIDTH

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results"
TREES = {"Qwen3-1.7B-Base": "research_plots_qwen_base_richie_hsj_nopad", "GPT-2": "research_plots_gpt2_richie_hsj_nopad"}
DISTRIBUTION, SHAPE, SUBLAYERS = (f"1_expert_distribution/{name}" for name in (
    "1.1_layer_expert_distribution", "1.2_distribution_shape", "1.3_sublayer_informativeness"))
PAIRWISE, HUMAN = "2_similarities/2.2_pairwise_similarities", "2_similarities/2.3_human_similarity_validation"
FIGURES = [f"{DISTRIBUTION}/mean_expert_layer_distribution_cumulative_by_block.png",
           f"{DISTRIBUTION}/mean_expert_layer_distribution_cumulative_sorted_by_block.png",
           f"{SHAPE}/category_entropy_gearys_by_block.png", f"{SUBLAYERS}/sublayer_auc_vs_share.png",
           f"{PAIRWISE}/jaccard_category_mean_matrix.png", f"{HUMAN}/similarity_vs_jaccard.png",
           f"{HUMAN}/human_similarity_decomposition.png", f"{HUMAN}/human_similarity_text_baseline.png"]


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


def main() -> None:
    """Shared ranges first, then every figure of each model, copied into its tree."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ap", default="0.6")
    trees = {model: RESULTS / folder / f"AP_{parser.parse_args().ap}" for model, folder in TREES.items()}

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
        out = RESULTS / "thesis_figures" / model
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
