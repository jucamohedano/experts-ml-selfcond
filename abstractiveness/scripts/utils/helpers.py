import logging
import pathlib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

def set_folder_log(folder_path: pathlib.Path) -> None:
    """
    Configure logging to write output to a main.log file in the specified folder.
    Removes any existing FileHandlers from previous iterations before attaching a new one.
    """
    folder_path.mkdir(parents=True, exist_ok=True)
    
    log_file_path = folder_path / "main.log"
    root_logger = logging.getLogger() 
    
    # Find and remove any existing FileHandlers (from previous loop iterations)
    for handler in root_logger.handlers[:]: 
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close() 
            
    # Create and attach a new FileHandler for the current folder
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

def save_dataframe(output_dataframe: pd.DataFrame, out_path: pathlib.Path, index: bool = False) -> pathlib.Path:
    """
    Save a pandas DataFrame to a CSV file at the specified output path.
    Creates parent directories if they don't exist.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_csv(out_path, index=index)
    return out_path

def load_experts_data(root, model, threshold) -> pd.DataFrame:
    """
    Load and concatenate expertise data for a given model and activation percentage (AP) threshold.
    Filters to include only expert records where AP >= threshold.
    Returns a combined DataFrame with all qualifying expert records, or an empty DataFrame if none found.
    """
    all_rows = []
    path = pathlib.Path(root) / model
    for csv_file in path.glob("**/expertise/expertise.csv"):
        experts_data = pd.read_csv(csv_file)
        experts = experts_data[experts_data["ap"] >= threshold].copy()
        all_rows.append(experts)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

# Per-architecture rules for turning raw layer strings (e.g. "model.layers.0.mlp.down_proj:0")
# into sequential 1..N layer indices. `sublayers` is listed in forward-pass order within one
# transformer block; that ordering defines how layer_idx increments inside a block. To support
# a new model, add an entry here rather than editing the mapping logic below.
LAYER_ARCHITECTURES = {
    # GPT-2: 12 blocks x 4 sublayers = 48 layers. Layer strings look like
    # "transformer.h.0.attn.c_attn:0".
    "gpt2": {
        "block_regex": r"h\.(\d+)",
        "sublayer_regex": r"h\.\d+\.(.*?):0",
        "sublayers": ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    },
    # Qwen3: 28 blocks x 7 sublayers = 196 layers. Layer strings look like
    # "model.layers.0.mlp.down_proj:0".
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
    """
    Turn a DataFrame with a single 'layer' column of raw layer strings into the standard
    (layer, layer_idx, layer_name) mapping for the given architecture. layer_idx is a
    contiguous 1..N index ordered by (block number, sublayer position within the block),
    where the within-block order is LAYER_ARCHITECTURES[architecture]['sublayers'].
    Raises ValueError if any layer string carries a sublayer missing from that spec.
    """
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
    """
    Initialize a mapping that translates original model layer strings to sequential layer
    indices and formatted layer names for the given architecture (a key of LAYER_ARCHITECTURES,
    e.g. "gpt2" or "qwen3"). If the mapping file already exists, load from it; otherwise compute
    it from the first expertise.csv found under responses_dir/model and cache it for reuse.
    Returns a categorical DataFrame with layer, layer_idx, and layer_name columns.
    """
    if mapping_path.exists():
        mapping_df = pd.read_csv(mapping_path)
        ordered_names = mapping_df.sort_values('layer_idx')['layer_name']
        mapping_df['layer_name'] = pd.Categorical(mapping_df['layer_name'], categories=ordered_names, ordered=True)
        return mapping_df

    log.info(f"Generating global layer mapping ({architecture}) for the first time...")
    # Peek at the first expertise.csv we can find, loading ONLY the layer column for speed
    search_path = pathlib.Path(responses_dir) / model
    first_csv = next(search_path.glob("**/expertise/expertise.csv"))
    unique_layers = pd.read_csv(first_csv, usecols=['layer']).drop_duplicates()

    mapping_df = build_layer_mapping_from_layers(unique_layers, architecture)

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(mapping_path, index=False)

    return mapping_df

def filter_expert_data_to_sublayer(expert_allocation_df: pd.DataFrame, sublayer: str) -> pd.DataFrame:
    """
    Restrict expert data to one sublayer type (e.g. 'mlp.gate_proj'): keeps only rows
    whose layer_name ends in that projection, prunes the layer_name categories to the
    retained layers, and leaves layer_idx values untouched (so they stay non-contiguous
    across blocks by design). Returns the input unchanged when sublayer is falsy, so
    callers can pass a config value directly.
    """
    if not sublayer:
        return expert_allocation_df
    sublayer_of = expert_allocation_df['layer_name'].astype(str).str.extract(r'^\d+\.L\.\d+\.(.+)$')[0]
    filtered = expert_allocation_df[sublayer_of == sublayer].copy()
    filtered['layer_name'] = filtered['layer_name'].cat.remove_unused_categories()
    return filtered


def gearys_c(values) -> float:
    """
    Geary's C spatial autocorrelation of a 1-D sequence (e.g. a layer distribution in
    depth order), with binary adjacent-neighbor weights. The general definition

        C = (N-1) * sum_ij w_ij (x_i - x_j)^2 / (2 W sum_i (x_i - mean)^2)

    with w_ij = 1 iff |i-j| == 1 (so W = 2(N-1)) algebraically reduces to the chain form
    implemented here:

        C = sum_i (x[i+1] - x[i])^2 / (2 * sum_i (x[i] - mean)^2)

    The numerator is the squared discrete first derivative, so C measures local
    step-to-step change: C ~ 1 no spatial structure, C < 1 smooth/clumped (neighbors
    alike), C > 1 jagged/alternating. Scale-invariant, so raw counts and normalized
    probabilities give the same value. Returns NaN for degenerate inputs (< 3 values
    or zero variance).
    """
    if values is None or len(values) < 3:
        return np.nan
    values = np.asarray(values, dtype=float)
    sum_of_squares = np.sum((values - values.mean()) ** 2)
    if sum_of_squares == 0:
        return np.nan
    return float(np.sum(np.diff(values) ** 2) / (2 * sum_of_squares))


def build_layer_probability_matrix(expert_allocation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a concept-by-layer expert count matrix and its row-normalized probability matrix.
    Groups expert allocation rows by concept and layer_idx, counts rows per cell, then
    normalizes each concept's row to sum to 1.0 across layers.
    layer_idx is a plain int column (not categorical), so groupby only yields columns for
    layers that actually have at least one retained expert; at stricter AP thresholds a whole
    layer can have zero experts across every concept. The matrix is reindexed to the full set
    of layers so it always has one column per layer of the current support, keeping it aligned
    with layer_name.cat.categories used elsewhere (e.g. module 6's per-layer x-axis) even when
    some layers are entirely empty at the given threshold. The support is parsed from the
    layer_name categories (whose leading number IS the layer_idx) rather than assumed to be
    1..N, so it stays correct when the data is filtered to one sublayer type and the retained
    layer_idx values are non-contiguous (e.g. 3, 7, 11, ... for mlp.c_fc).
    Returns: (count_matrix, prob_matrix), both indexed by concept with layer_idx as columns.
    """
    count_matrix = expert_allocation_df.groupby(['concept', 'layer_idx'], observed=False).size().unstack(fill_value=0)
    layer_support = [int(str(name).split('.', 1)[0]) for name in expert_allocation_df['layer_name'].cat.categories]
    count_matrix = count_matrix.reindex(columns=layer_support, fill_value=0)
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return count_matrix, prob_matrix