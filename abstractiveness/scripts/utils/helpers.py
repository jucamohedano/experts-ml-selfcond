import logging
import pathlib
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

def init_global_layer_mapping(responses_dir, model, mapping_path: pathlib.Path) -> pd.DataFrame:
    """
    Initialize a mapping that translates original model layer strings to sequential 1-48 layer indices and formatted layer names.
    If mapping already exists, load from file; otherwise compute from the first expertise.csv and cache for reuse.
    Returns a categorical DataFrame with layer, layer_idx, and layer_name columns.
    """
    if mapping_path.exists():
        mapping_df = pd.read_csv(mapping_path)
        ordered_names = mapping_df.sort_values('layer_idx')['layer_name']
        mapping_df['layer_name'] = pd.Categorical(mapping_df['layer_name'], categories=ordered_names, ordered=True)
        return mapping_df
        
    log.info("Generating global layer mapping for the first time...")
    # Peek at the first expertise.csv we can find, loading ONLY the layer column for speed
    search_path = pathlib.Path(responses_dir) / model
    first_csv = next(search_path.glob("**/expertise/expertise.csv"))
    unique_layers = pd.read_csv(first_csv, usecols=['layer']).drop_duplicates()
    
    SUB_LAYERS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    block_nums = unique_layers['layer'].str.extract(r'h\.(\d+)').astype(int)[0]
    sub_layer_strs = unique_layers['layer'].str.extract(r'h\.\d+\.(.*?):0')[0]
    sub_idx = sub_layer_strs.map({sub: i for i, sub in enumerate(SUB_LAYERS)})

    unique_layers['layer_idx'] = (block_nums * 4) + sub_idx + 1
    unique_layers['layer_name'] = (
        unique_layers['layer_idx'].astype(str) + ".L." + 
        block_nums.astype(str) + "." + 
        sub_layer_strs
    )
    
    mapping_df = unique_layers.sort_values('layer_idx').reset_index(drop=True)
    ordered_names = mapping_df['layer_name']
    mapping_df['layer_name'] = pd.Categorical(mapping_df['layer_name'], categories=ordered_names, ordered=True)
    
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(mapping_path, index=False)

    return mapping_df

def build_layer_probability_matrix(expert_allocation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a concept-by-layer expert count matrix and its row-normalized probability matrix.
    Groups expert allocation rows by concept and layer_idx, counts rows per cell, then
    normalizes each concept's row to sum to 1.0 across layers.
    layer_idx is a plain int column (not categorical), so groupby only yields columns for
    layers that actually have at least one retained expert; at stricter AP thresholds a whole
    layer can have zero experts across every concept. The matrix is reindexed to the full set
    of layers (from layer_name's categorical dtype) so it always has one column per model
    layer, keeping it aligned with layer_name.cat.categories used elsewhere (e.g. module 6's
    per-layer x-axis) even when some layers are entirely empty at the given threshold.
    Returns: (count_matrix, prob_matrix), both indexed by concept with layer_idx as columns.
    """
    count_matrix = expert_allocation_df.groupby(['concept', 'layer_idx'], observed=False).size().unstack(fill_value=0)
    n_layers = expert_allocation_df['layer_name'].cat.categories.size
    count_matrix = count_matrix.reindex(columns=range(1, n_layers + 1), fill_value=0)
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return count_matrix, prob_matrix