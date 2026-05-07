#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import logging
import pathlib

from selfcond.data import (
    concept_list_to_df,
    PytorchTransformersTokenizer,
    ConceptDataset,
)
from selfcond.responses import cache_responses
from selfcond.models import collect_responses_info, PytorchTransformersModel

logging.basicConfig(level=logging.WARNING)


def compute_and_save_responses(
    model_name: str,
    model_cache_dir: pathlib.Path,
    data_path: pathlib.Path,
    concept_group: str,
    concept: str,
    tokenizer: PytorchTransformersTokenizer,
    batch_size: int,
    response_save_path: pathlib.Path,
    num_per_concept: int,
    seq_len: int,
    device: str,
    verbose: bool = False,
) -> None:
    """
    Loads data in ``data_path`` compatible with ``ConceptDataset`` and performs inference on a specific model.
    The model is uniquely identified with the ``model_name``, compatible with Huggingface Transformers names.

    Args:
        model_name: The Huggingface Transformers model name, eg., gpt2-large.
        model_cache_dir: Cache directory (can be None).
        data_path: Where the data is found locally.
        concept_group: The concept type or group, eg., "sense"
        concept: The concept name, eg., "football".
        tokenizer: The Huggingface Transformers tokenizer.
        batch_size: Inference batch size
        response_save_path: Where to store the responses.
        num_per_concept: Number of positive (or negative) exemplars per concept.
        seq_len: Max sequence length.
        device: Device where to run inference on.
        verbose: Verbosity flag.
    """
    local_data_file = data_path / concept_group / f"{concept}.json"
    if not local_data_file.exists():
        print(f"Skipping {local_data_file}, file not found.")
        return

    if (response_save_path / model_name / concept_group / concept / "responses").exists():
        print(f"Skipping, already computed responses {local_data_file}")
        return

    random_seed = 1234

    dataset = ConceptDataset(
        json_file=local_data_file,
        seq_len=seq_len,
        num_per_concept=num_per_concept,
        random_seed=random_seed,
        tokenizer=tokenizer,
    )

    save_path = response_save_path / model_name / dataset.concept_group / dataset.concept
    if (save_path / "responses").exists():
        print(f"Skipping {dataset.concept_group}/{dataset.concept}")
        return

    if verbose:
        print(dataset, flush=True)

    save_path.mkdir(parents=True, exist_ok=True)

    # Load the model
    tm_model = PytorchTransformersModel(
        model_name, seq_len=dataset.seq_len, cache_dir=model_cache_dir, device=device
    )

    # Select responses
    responses_info_interm = collect_responses_info(model_name=model_name, model=tm_model)

    # Construct a response generator
    cache_responses(
        model=tm_model,
        dataset=dataset,
        batch_size=batch_size,
        response_infos=responses_info_interm,
        save_path=save_path / "responses",
    )

    return


def run_response_computation(
    model_name_or_path: str,
    data_path: pathlib.Path,
    responses_path: pathlib.Path,
    concepts: str = "",
    model_cache: pathlib.Path = None,
    tok_cache: pathlib.Path = None,
    seq_len: int = 128,
    num_per_concept: int = 1000,
    inf_batch_size: int = 30,
    device: str = None,
):
    """
    Run the response computation pipeline.
    """
    if model_cache is not None and tok_cache is None:
        tok_cache = model_cache

    responses_path.mkdir(exist_ok=True, parents=True)

    if not concepts:
        assert (data_path / "concept_list.csv").exists()
        concepts_requested = data_path / "concept_list.csv"
    else:
        concepts_requested = concepts.split(",")

    # Normalize concept list into a dataframe
    concept_df = concept_list_to_df(concepts_requested)

    # Load a tokenizer for sentence pre-processing
    tokenizer = PytorchTransformersTokenizer(model_name_or_path, tok_cache)

    # Read responses for all concepts in concept_df
    for _, row in concept_df.iterrows():
        concept, concept_group = row["concept"], row["group"]

        if concept in ["positive", "negative"] and concept_group == "keyword":
            continue

        print(f"Running inference to read responses on concept {concept_group}/{concept}")
        compute_and_save_responses(
            model_name=model_name_or_path,
            model_cache_dir=model_cache,
            data_path=data_path,
            concept_group=concept_group,
            concept=concept,
            seq_len=seq_len,
            num_per_concept=num_per_concept,
            batch_size=inf_batch_size,
            response_save_path=responses_path,
            tokenizer=tokenizer,
            verbose=True,
            device=device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compute_responses.py",
        description=(
            "This script reads responses from a pytorch model, given a dataset. "
            "The models are fetched the HuggingFace Transformers repository. "
            "The data is tokenized with the appropriate tokenization technique, "
            "and the responses are maxpooled in the temporal dimension, being "
            "agnostic to the sentence length. The obtained responses are saved, "
            "at batch level, as a dict `{layer_name: response_tensor}`."
        ),
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        help="Model name (from pytorch-transformers module)",
        default="gpt2",
        #required=True,
    )
    parser.add_argument(
        "--model-cache",
        type=pathlib.Path,
        default=None,
        help=(
            "Model cache dir. If not set, the model will be fetched "
            "from the transformers repository."
        ),
        required=False,
    )
    parser.add_argument(
        "--tok-cache",
        type=pathlib.Path,
        default=None,
        help=(
            "Tokenizer cache dir. If not set, the model will be fetched from "
            "the transformers repository."
        ),
        required=False,
    )
    parser.add_argument(
        "--data-path",
        type=pathlib.Path,
        help=(
            "Path to a duly formatted dataset. If --concepts not set, "
            "assumes a concept_list.csv file inside the data path."
        ),
        default=pathlib.Path("assets/Qwen3-30B-A3B-Instruct-2507_abstractiveness_150_cot"),
        #required=True,
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="",
        help=(
            "Run only selected concept (comma separated), formatted as concept_group/concept_name."
        ),
        required=False,
    )
    parser.add_argument(
        "--responses-path",
        type=pathlib.Path,
        help="Path where to save the responses.",
        default=pathlib.Path("responses/GPT2_abstractiveness_150_responses"),
        #required=True,
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        help="Max sequence length allowed in tokens.",
        default=1024,
    )
    parser.add_argument(
        "--num-per-concept",
        type=int,
        help="Max number of sentences per concept, per label",
        default=1000,
    )
    parser.add_argument("--inf-batch-size", type=int, help="Inference batch size", default=8)
    parser.add_argument("--device", type=str, help="Device to use", default="cuda")

    args = parser.parse_args()

    run_response_computation(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        responses_path=args.responses_path,
        concepts=args.concepts,
        model_cache=args.model_cache,
        tok_cache=args.tok_cache,
        seq_len=args.seq_len,
        num_per_concept=args.num_per_concept,
        inf_batch_size=args.inf_batch_size,
        device=args.device,
    )
