import json
import logging
import pathlib
import numpy as np
import pandas as pd
from scipy import stats

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


def pair_similarity_vector(expert_sets_df: pd.DataFrame, concepts: list) -> np.ndarray:
    """
    Jaccard similarity of raw expert sets for every unordered pair of ``concepts``,
    as a flat vector aligned with np.triu_indices(len(concepts), k=1). Expert sets are
    binary membership keyed on (layer_idx, unit); no probability normalization.
    """
    presence = expert_sets_df.assign(present=1).pivot_table(
        index="concept", columns=["layer_idx", "unit"], values="present", fill_value=0)
    A = presence.reindex(concepts, fill_value=0).values.astype(np.float32)

    intersection = A @ A.T
    sizes = A.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)
    return jaccard[np.triu_indices(len(concepts), k=1)]


def pair_shared_count_vector(expert_sets_df: pd.DataFrame, concepts: list) -> np.ndarray:
    """
    Raw count of shared experts for every unordered concept pair, aligned with
    np.triu_indices(len(concepts), k=1). Unlike Jaccard, this is not normalized by
    set size, so it is the more interpretable x-axis for a per-pair scatter (Jaccard
    compresses most pairs toward zero regardless of how many experts they actually share).
    """
    presence = expert_sets_df.assign(present=1).pivot_table(
        index="concept", columns=["layer_idx", "unit"], values="present", fill_value=0)
    A = presence.reindex(concepts, fill_value=0).values.astype(np.float32)
    intersection = A @ A.T
    return intersection[np.triu_indices(len(concepts), k=1)]


# ---------------------------------------------------------------------------
# Representational Similarity Analysis (module 8)
#
# An RDM here is a square (n_concepts x n_concepts) similarity or dissimilarity
# matrix whose off-diagonal, taken in np.triu_indices(n, k=1) order, is the vector
# every statistic below operates on. Square form is required because the Mantel
# null permutes CONCEPTS, which re-indexes rows and columns together and cannot be
# expressed on a flat pair vector.
# ---------------------------------------------------------------------------

def embedding_rdm(feature_matrix: np.ndarray, metric: str = "correlation_zscored") -> np.ndarray:
    """
    Concept-by-concept similarity of embedding vectors, as a square matrix.

    feature_matrix is (n_concepts, n_units), one row per concept. Two metrics:

    correlation_zscored (default): each unit is z-scored across concepts, then
      similarity is the Pearson correlation between concept vectors over units.
      Standardizing per unit removes the baseline every concept shares, since a
      mean over positive sentences is dominated by generic sentence structure and
      the concept signal is a small perturbation on top of it. Correlation rather
      than cosine additionally removes each concept's overall activation
      magnitude, which tracks word frequency and would otherwise enter the RDM as
      a nuisance dimension.
    cosine_raw: cosine similarity of the unstandardized vectors, kept as a
      robustness variant. Expect it to be compressed toward 1.0 for exactly the
      reason above.

    Units that do not vary across concepts carry no discriminative information and are
    dropped under the z-scored metric rather than producing NaN. The threshold is
    relative to the layer's activation scale rather than an exact zero test: a unit
    whose spread is float noise would otherwise be amplified into a full-magnitude
    z-score and inject that noise into every pair. The deviations are computed once and
    reused rather than recomputed after the columns are dropped, because numpy's
    pairwise reduction is sensitive to memory layout, so the same near-constant column
    can measure as exactly zero before the copy and non-zero after it (or the reverse),
    which silently reintroduces division by zero.
    """
    X = np.asarray(feature_matrix, dtype=np.float64)

    if metric == "correlation_zscored":
        deviations = X.std(axis=0)
        tolerance = 1e-10 * max(1.0, float(np.abs(X).max(initial=0.0)))
        informative = deviations > tolerance
        if not informative.any():
            return np.full((len(X),) * 2, np.nan)
        X = X[:, informative]
        X = (X - X.mean(axis=0)) / deviations[informative]
        X = X - X.mean(axis=1, keepdims=True)
    elif metric != "cosine_raw":
        raise ValueError(f"Unknown embedding RDM metric: {metric}")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    similarity = X @ X.T
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _offdiag(square_matrix: np.ndarray) -> np.ndarray:
    """Upper-triangle (k=1) of a square matrix as a flat vector."""
    n = square_matrix.shape[0]
    return square_matrix[np.triu_indices(n, k=1)]


def _rank_square(square_matrix: np.ndarray, centre: bool = False) -> np.ndarray:
    """
    Square matrix holding the ranks of the off-diagonal values, mirrored into both
    triangles with a zero diagonal, so that permuting rows and columns permutes the
    ranks with them. Optionally mean-centred over the off-diagonal.
    """
    n = square_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    ranks = stats.rankdata(square_matrix[iu])
    if centre:
        ranks = ranks - ranks.mean()
    out = np.zeros((n, n), dtype=np.float64)
    out[iu] = ranks
    return out + out.T


def spearman_rdm_mantel(rdm_a: np.ndarray, rdm_b: np.ndarray, n_permutations: int = 9999,
                        rng: np.random.Generator = None) -> dict:
    """
    Spearman correlation between the off-diagonals of two square RDMs, with a
    concept-level Mantel permutation p-value.

    spearman_rho: rank correlation over all n(n-1)/2 concept pairs.
    mantel_p: pairs are not independent observations (each concept appears in n-1
      pairs), so the null shuffles CONCEPTS rather than pairs. One RDM is re-indexed
      as rdm[p][:, p], which preserves its internal geometry and breaks only the
      concept-to-concept correspondence under test. One-sided, smallest reportable
      value 1/(n_permutations + 1), matching the convention in module 1.

    Permuting rows and columns is a bijection on the off-diagonal multiset, so the
    ranks are computed once and permuted along with the matrix rather than being
    recomputed per shuffle, and the permuted rank vector's mean and standard
    deviation are invariant. Spearman under permutation therefore reduces to a
    single dot product, and the ravel form below measured about 6x faster than
    gathering the upper triangle each time.

    Returns rho=NaN, p=1.0 when either off-diagonal is constant, since with no
    variation there is nothing to correlate and pearsonr would be undefined.
    """
    rng = rng if rng is not None else np.random.default_rng(42)
    n = rdm_a.shape[0]
    a_off, b_off = _offdiag(rdm_a), _offdiag(rdm_b)

    if a_off.min() == a_off.max() or b_off.min() == b_off.max():
        return {"spearman_rho": np.nan, "mantel_p": 1.0}

    rho = float(stats.spearmanr(a_off, b_off).statistic)

    # Uncentred ranks on the permuted side are safe because the fixed side is
    # centred and has a zero diagonal, so the omitted mean term contributes zero.
    rank_a = _rank_square(rdm_a)
    rank_b_centred = _rank_square(rdm_b, centre=True)

    observed = float(rank_a.ravel() @ rank_b_centred.ravel())
    n_at_least = 0
    for _ in range(n_permutations):
        p = rng.permutation(n)
        if rank_a[p][:, p].ravel() @ rank_b_centred.ravel() >= observed:
            n_at_least += 1

    return {"spearman_rho": rho, "mantel_p": (1 + n_at_least) / (1 + n_permutations)}


def bootstrap_rdm_rho_ci(rdm_a: np.ndarray, rdm_b: np.ndarray, n_boot: int = 1000,
                         rng: np.random.Generator = None, ci: float = 95.0) -> tuple:
    """
    Percentile confidence interval for the Spearman correlation between two RDMs,
    resampling CONCEPTS with replacement (the unit of observation, as in the Mantel
    null above). Pairs where a resampled concept meets itself are excluded, since
    they are similarity-by-identity rather than evidence and would inflate rho.

    Used to report a peak plateau rather than a bare argmax: the layers whose
    intervals overlap the peak's are statistically indistinguishable from it, which
    matters because the layer-sweep literature reports a broad middle region rather
    than one best layer. Returns (lo, hi), or (NaN, NaN) if too few resamples were
    usable.
    """
    rng = rng if rng is not None else np.random.default_rng(42)
    n = rdm_a.shape[0]
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        i, j = np.triu_indices(n, k=1)
        keep = idx[i] != idx[j]
        if keep.sum() < 30:
            continue
        a = rdm_a[idx[i][keep], idx[j][keep]]
        b = rdm_b[idx[i][keep], idx[j][keep]]
        if a.min() == a.max() or b.min() == b.max():
            continue
        rhos.append(stats.spearmanr(a, b).statistic)

    if len(rhos) < n_boot // 10:
        return (np.nan, np.nan)
    tail = (100.0 - ci) / 2.0
    return (float(np.percentile(rhos, tail)), float(np.percentile(rhos, 100.0 - tail)))


def spearman_brown(reliability: float) -> float:
    """
    Spearman-Brown correction for a split-half reliability, 2r / (1 + r). The two
    halves each hold half the sentences, so the raw split-half correlation
    underestimates the reliability of the full-data estimate. Apply exactly once.
    """
    if not np.isfinite(reliability) or reliability <= -1.0:
        return np.nan
    return float(2.0 * reliability / (1.0 + reliability))


def load_concept_embeddings(cache_path: pathlib.Path) -> dict:
    """
    Load the concept embedding cache written by precompute_concept_embeddings.py.

    Returns a dict with concepts (row order), layers (raw layer strings, the join key
    onto layer_mapping_*.csv), offsets (start index of each layer on the unit axis),
    mean (n_concepts x total_units, recombined from the split-half sums), half_a and
    half_b (the same for each half, used for the per-layer noise ceiling), and
    provenance from the sidecar JSON.

    Returns None with a warning naming the command that produces the cache, so a
    missing cache skips module 8 rather than failing the whole AP sweep.
    """
    cache_path = pathlib.Path(cache_path)
    if not cache_path.exists():
        log.warning(f"Concept embedding cache not found at {cache_path}. "
                    f"Module 8 will be skipped. Build it with: "
                    f"python scripts/precompute_concept_embeddings.py --config <config_key>")
        return None

    data = np.load(cache_path, allow_pickle=False)
    n_even = data["n_even"].astype(np.float64)[:, None]
    n_odd = data["n_odd"].astype(np.float64)[:, None]

    sidecar = cache_path.with_suffix(".json")
    provenance = json.loads(sidecar.read_text()) if sidecar.exists() else {}

    return {
        "concepts": [str(c) for c in data["concepts"]],
        "layers": [str(l) for l in data["layers"]],
        "offsets": data["offsets"],
        "mean": (data["sum_even"] + data["sum_odd"]) / (n_even + n_odd),
        "half_a": data["sum_even"] / n_even,
        "half_b": data["sum_odd"] / n_odd,
        "provenance": provenance,
    }


def layer_slice(embedding_cache: dict, layer: str) -> slice:
    """Column slice of one raw layer string within the cache's flat unit axis."""
    i = embedding_cache["layers"].index(layer)
    return slice(int(embedding_cache["offsets"][i]), int(embedding_cache["offsets"][i + 1]))


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