#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib
import shutil

import numpy as np
import pandas as pd

from selfcond.data import concept_list_to_df
from selfcond.expertise import ExpertiseResult
from selfcond.integrity import (
    check_expertise_complete,
    check_responses_complete,
    expected_sample_count,
    latest_concept_subdir,
)
from selfcond.responses import read_responses_from_cached
from selfcond.models import get_layer_regex
from selfcond.visualization import (
    plot_scatter_pandas,
    plot_metric_per_layer,
    plot_in_dark_mode,
)


def analyze_expertise_for_concept(
    concept_dir: pathlib.Path,
    concept_group: str,
    concept: str,
    expected_samples: int = None,
):
    """
    Analyze the expertise of a specific concept. It expects a `results_dir` with the following tree:

    ```
    results_dir
        concept_group
            concept
                responses (created using compute_responses.py)
    ```

    Args:
        concept_dir: The concept directory, contains a dir `responses` and will contain a dir `expertise`
        concept_group: The concept type
        concept: The concept
        expected_samples: Total samples the cached responses must hold. When given,
            incomplete cached responses are deleted (so compute_responses.py can
            regenerate them) instead of silently producing skewed expertise.
    """

    # Build paths
    cached_responses_dir = concept_dir / "responses"
    concept_exp_dir = concept_dir / "expertise"

    if ExpertiseResult.exists_in_disk(concept_exp_dir):
        complete, reason = check_expertise_complete(concept_exp_dir)
        if complete:
            print("Results found, skipping building")
            return
        print(
            f"Incomplete expertise for {concept_group}/{concept} ({reason}): "
            f"deleting {concept_exp_dir} and recomputing."
        )
        shutil.rmtree(concept_exp_dir)

    if expected_samples is not None and cached_responses_dir.exists():
        complete, reason = check_responses_complete(cached_responses_dir, expected_samples)
        if not complete:
            # Expertise built from partial responses is silently wrong (the tail
            # of the sample order is all positives, so truncation skews AP).
            print(
                f"Incomplete responses for {concept_group}/{concept} ({reason}): "
                f"deleting {concept_dir}. Re-run compute_responses.py to regenerate."
            )
            shutil.rmtree(concept_dir)
            return

    # Read all the responses and labels from storage
    try:
        responses, labels_int, response_names = read_responses_from_cached(
            cached_responses_dir, concept
        )
    except RuntimeError:
        print(f"No responses found for concept {concept}")
        return

    assert (
        labels_int is not None
    ), "Cannot compute expertise, did not find any labels in cached responses."

    if not responses:
        print(f"Found response files but could not load them for concept {concept}")
        return

    concept_exp_dir.mkdir(exist_ok=True, parents=True)

    # from random import shuffle
    # this was added to test independence saliency-labels (Ian Goodfellow tests)
    # shuffle(labels_int)

    expertise_result = ExpertiseResult()
    expertise_result.build(
        responses=responses,
        labels=labels_int,
        concept=concept,
        concept_group=concept_group,
        forcing=True,
    )
    expertise_result.save(concept_exp_dir)


def build_result_figures(
    expertise_result: ExpertiseResult,
    results_dir: str,
    layer_types_regex=None,
    show_figures: bool = False,
):
    """
        Build expertise figures for a specific concept. It expects a `results_dir` with the following tree:

    ```
    results_dir
        concept_group
            concept
                expertise
                    expertise.csv (will be loaded to build results)
                    expertise_info.json (will be loaded to build results)
    ```

    Args:
        expertise_result: ExpertiseResult object with duly loaded results
        results_dir: Where to save the output assets
        show_figures: Show figures or just save?
    """
    print("Building plots")
    df = expertise_result.export_as_pandas()
    info_json = expertise_result.export_extra_info_json()

    concept = df["concept"].iloc[0]
    concept_group = df["group"].iloc[0]

    # Print top AP
    print(df.sort_values(by="ap", ascending=False).iloc[:10])

    # Show correlation of corr and on_value
    plot_scatter_pandas(
        df,
        "ap",
        "on_p50",
        out_dir=results_dir,
        y_lim=[0, 30],
        alpha=0.1,
        title=f"AP vs. ON Value (concept {concept_group}/{concept})",
        also_show=show_figures,
    )

    neurons_at_ap_df = pd.DataFrame(
        index=list(info_json["neurons_at_ap"].keys()),
        data={"neuron count": list(info_json["neurons_at_ap"].values())},
    )
    neurons_at_ap_df.index.name = "ap"
    print(f'\nmaxAP = {np.max(df["ap"])}')

    for k in [10, 100]:
        plot_metric_per_layer(
            df,
            metric="ap",
            out_dir=results_dir,
            top_k=k,
            layer_types_regex=layer_types_regex,
            also_show=show_figures,
        )


def run_expertise_computation(
    root_dir: pathlib.Path,
    model_name: str,
    concepts: str = None,
    k: int = 10,
    show: bool = False,
    skip: bool = False,
    black: bool = False,
    num_per_concept: int = 1000,
):
    """
    Run the expertise computation pipeline.
    """
    plot_in_dark_mode(black)

    # Load concepts from file or list
    if not concepts:
        assert (root_dir / "concept_list.csv").exists()
        concepts_requested = root_dir / "concept_list.csv"
    else:
        if "," in concepts:
            concepts_requested = concepts.split(",")
        else:
            concepts_requested = pathlib.Path(concepts)

    print(concepts_requested)
    concept_df = concept_list_to_df(concepts_requested)

    # The most recently generated expertise is the one a previous interrupted
    # run may have left half written; give it a deep check (csv row count vs.
    # total_neurons) and delete it if broken so it gets recomputed below.
    model_root = root_dir / model_name
    if model_root.exists():
        last_expertise = latest_concept_subdir(model_root, "expertise")
        if last_expertise is not None:
            complete, reason = check_expertise_complete(last_expertise, check_csv_rows=True)
            if not complete:
                print(
                    f"Last generated expertise {last_expertise} is incomplete "
                    f"({reason}): deleting it for recomputation."
                )
                shutil.rmtree(last_expertise)

    # When concepts come from a concept_list.csv the dataset jsons live next to
    # it, which lets us verify cached responses hold the full sample count
    # before building expertise from them.
    data_root = concepts_requested.parent if isinstance(concepts_requested, pathlib.Path) else None

    for row_index, row in concept_df.iterrows():
        concept_dir = root_dir / model_name / row["group"] / row["concept"]
        expected = None
        if data_root is not None:
            data_json = data_root / row["group"] / f"{row['concept']}.json"
            if data_json.exists():
                expected = expected_sample_count(data_json, num_per_concept)
        analyze_expertise_for_concept(
            concept_dir=concept_dir,
            concept=row["concept"],
            concept_group=row["group"],
            expected_samples=expected,
        )

        # Load results and plot
        expertise_dir = concept_dir / "expertise"
        if not ExpertiseResult.exists_in_disk(expertise_dir):
            print(f"[skip] No expertise results in {expertise_dir}")
            continue
        expertise_result = ExpertiseResult()
        expertise_result.load(expertise_dir)
        layer_types_regex = get_layer_regex(model_name=model_name)
        build_result_figures(
            expertise_result=expertise_result,
            results_dir=expertise_dir,
            layer_types_regex=layer_types_regex,
            show_figures=show,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compute_expertise.py",
        description=(
            "This script computes expertise results given a set of model "
            "responses collected using `compute_responses.py`.\nThe expertise "
            "results are saved as a DataFrame with `units` rows and various "
            "informative columns, and a json file with extra information."
        ),
    )
    parser.add_argument(
        "--root-dir",
        type=pathlib.Path,
        help=(
            "Root directory with responses. Should contain responses_"
            "`dir/model/concept_group/concept/responses`"
        ),
        default=pathlib.Path("responses/Qwen3_1.7B_abstractiveness_Richie_HSJ_responses"),
        #required=True,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="The model name",
        default="Qwen/Qwen3-1.7B",
        # required=True,
    )
    parser.add_argument("--concepts", type=str, help="concepts to analyze", default=str(pathlib.Path("assets/Qwen3-30B-A3B-Instruct-2507_abstractiveness_150_cot/concept_list.csv")))
    parser.add_argument("--k", type=int, help="Top K neurons to plot", default=10)
    parser.add_argument("--show", action="store_true", help="Show images or just save", default=False)
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Force skip for concepts with existing expertise results.",
        default=False,
    )
    parser.add_argument("--black", action="store_true", help="Figures in black mode", default=False)
    parser.add_argument(
        "--num-per-concept",
        type=int,
        help=(
            "Max sentences per concept per label used when the responses were "
            "computed (must match compute_responses.py); needed to verify "
            "cached responses are complete."
        ),
        default=1000,
    )
    args = parser.parse_args()

    run_expertise_computation(
        root_dir=args.root_dir,
        model_name=args.model_name,
        concepts=args.concepts,
        k=args.k,
        show=args.show,
        skip=args.skip,
        black=args.black,
        num_per_concept=args.num_per_concept,
    )
