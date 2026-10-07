"""
Loading expert rows and mapping raw layer names onto one indexed layer axis.

Used by the executor, module 4 and the embedding cache builder.

Explanations: documentation/module_1_expert_distribution.md, section 1.1.
"""

import logging
import pathlib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_experts_data(root, model, threshold) -> pd.DataFrame:
    """Load and concatenate expertise data for a given model and activation percentage (AP) threshold."""
    all_rows = []
    path = pathlib.Path(root) / model
    for csv_file in path.glob("**/expertise/expertise.csv"):
        experts_data = pd.read_csv(csv_file)
        experts = experts_data[experts_data["ap"] >= threshold].copy()
        all_rows.append(experts)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# Per-architecture rules turning raw layer strings into sequential layer indices.
LAYER_ARCHITECTURES = {
    "gpt2": {
        "block_regex": r"h\.(\d+)",
        "sublayer_regex": r"h\.\d+\.(.*?):0",
        "sublayers": ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    },
    "qwen3": {
        "block_regex": r"layers\.(\d+)",
        "sublayer_regex": r"layers\.\d+\.(.*?):0",
        "sublayers": [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ],
    },
}


def build_layer_mapping_from_layers(unique_layers: pd.DataFrame, architecture: str) -> pd.DataFrame:
    """Turn a DataFrame with a single 'layer' column of raw layer strings into the standard (layer, layer_idx, layer_name) mapping for the given architecture."""
    spec = LAYER_ARCHITECTURES[architecture]
    n_sub = len(spec["sublayers"])
    block_nums = unique_layers['layer'].str.extract(spec["block_regex"]).astype(int)[0]
    sub_layer_strs = unique_layers['layer'].str.extract(spec["sublayer_regex"])[0]
    sub_idx = sub_layer_strs.map({sub: i for i, sub in enumerate(spec["sublayers"])})

    unmapped = sorted(sub_layer_strs[sub_idx.isna()].dropna().unique())
    if unmapped:
        raise ValueError(
            f"Architecture '{architecture}' has layer strings with sublayers not in its spec: "
            f"{unmapped}. Add them (in forward-pass order) to "
            f"LAYER_ARCHITECTURES['{architecture}']['sublayers']."
        )

    unique_layers = unique_layers.copy()
    unique_layers['layer_idx'] = (block_nums * n_sub) + sub_idx + 1
    unique_layers['layer_name'] = (
        unique_layers['layer_idx'].astype(str) + ".L." +
        block_nums.astype(str) + "." +
        sub_layer_strs
    )

    mapping_df = unique_layers.sort_values('layer_idx').reset_index(drop=True)
    ordered_names = mapping_df['layer_name']
    mapping_df['layer_name'] = pd.Categorical(mapping_df['layer_name'], categories=ordered_names, ordered=True)
    return mapping_df


def init_global_layer_mapping(responses_dir, model, mapping_path: pathlib.Path, architecture: str = "gpt2") -> pd.DataFrame:
    """Initialize a mapping that translates original model layer strings to sequential layer indices and formatted layer names for the given architecture (a key of LAYER_ARCHITECTURES, e.g."""
    if mapping_path.exists():
        mapping_df = pd.read_csv(mapping_path)
        ordered_names = mapping_df.sort_values('layer_idx')['layer_name']
        mapping_df['layer_name'] = pd.Categorical(mapping_df['layer_name'], categories=ordered_names, ordered=True)
        return mapping_df

    log.info(f"Generating global layer mapping ({architecture}) for the first time...")
    search_path = pathlib.Path(responses_dir) / model
    first_csv = next(search_path.glob("**/expertise/expertise.csv"))
    unique_layers = pd.read_csv(first_csv, usecols=['layer']).drop_duplicates()

    mapping_df = build_layer_mapping_from_layers(unique_layers, architecture)

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(mapping_path, index=False)

    return mapping_df


def filter_expert_data_to_sublayer(expert_allocation_df: pd.DataFrame, sublayer: str) -> pd.DataFrame:
    """Restrict expert data to one sublayer type (e.g."""
    if not sublayer:
        return expert_allocation_df
    sublayer_of = expert_allocation_df['layer_name'].astype(str).str.extract(r'^\d+\.L\.\d+\.(.+)$')[0]
    filtered = expert_allocation_df[sublayer_of == sublayer].copy()
    filtered['layer_name'] = filtered['layer_name'].cat.remove_unused_categories()
    return filtered


def expert_counts_with_metadata(expert_allocation_df: pd.DataFrame, concept_metadata: pd.DataFrame) -> pd.DataFrame:
    """Expert count per word merged with its metadata, plus log_frequency, level 1 rows first."""
    counts = expert_allocation_df.groupby("concept").size().reset_index(name="expert_count")
    merged = counts.merge(concept_metadata, on="concept")
    if 'frequency' in merged.columns:
        merged = merged[merged['frequency'] > 0].copy()
        merged['log_frequency'] = np.log10(merged['frequency'])
    desired_order = ["concept", "category", "abstraction_level", "frequency", "log_frequency",
                     "frequency_zipf_subtlex_us_lemma", "frequency_zipf_wikipedia_lemma",
                     "frequency_zipf_openwebtext_lemma", "frequency_zipf_fineweb_lemma",
                     "human_typicality", "expert_count"]
    merged = merged[[col for col in desired_order if col in merged.columns]]
    if 'abstraction_level' in merged.columns:
        merged = merged.sort_values(['abstraction_level', 'concept']).reset_index(drop=True)
    return merged
