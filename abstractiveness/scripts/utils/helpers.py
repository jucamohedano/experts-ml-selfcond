import json
import logging
import pathlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import xlogy

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


# ---------------------------------------------------------------------------
# Analysis scopes
#
# Every module runs its analysis once per scope: the whole model first, then one
# scope per sublayer type. The whole-model scope writes to the module's own folder
# so the big picture is what you see first; each sublayer scope writes to
# <module>/sublayers/<rank>_<sublayer>/, where rank is that sublayer's position by
# expert count. The rank is frozen at the most lenient AP threshold of the sweep so
# a given prefix names the same sublayer in every AP_x folder of a run.
# ---------------------------------------------------------------------------

WHOLE_MODEL_SCOPE_KEY = "whole_model"


@dataclass(frozen=True, eq=False)
class AnalysisScope:
    """
    One unit of analysis: a slice of the expert data plus where its outputs belong.

    key       : WHOLE_MODEL_SCOPE_KEY, or the sublayer type (e.g. "mlp.gate_proj")
    label     : text for plot titles ("whole model" / the sublayer name)
    dir_name  : sublayers/ subfolder name ("1_mlp.gate_proj"), None for the whole model
    expert_df : the expert rows this scope analyzes
    """
    key: str
    label: str
    dir_name: str | None
    is_whole_model: bool
    expert_df: pd.DataFrame


def sublayer_expert_counts(expert_allocation_df: pd.DataFrame) -> pd.Series:
    """
    Expert-row count per sublayer type, descending. Index is the sublayer name parsed
    out of layer_name, so it reflects whatever sublayers the frame actually contains.
    """
    sublayer_of = expert_allocation_df['layer_name'].astype(str).str.extract(r'^\d+\.L\.\d+\.(.+)$')[0]
    return sublayer_of.value_counts().sort_values(ascending=False)


def load_or_build_sublayer_rank(rank_path: pathlib.Path, build_expert_df, reference_ap: float) -> pd.DataFrame:
    """
    Rank sublayer types by expert count, cached at rank_path across the whole AP sweep.

    The ranking is computed once from the expert data at reference_ap (the most lenient
    threshold, where every sublayer still has experts) and reused for every threshold,
    so "1_mlp.gate_proj" names the same sublayer in AP_0.5 and in AP_0.9 and output
    paths stay comparable across the sweep. Ranking per threshold instead would let a
    sublayer drift between prefixes as stricter thresholds thin the sets unevenly.

    build_expert_df is a zero-argument callable returning the reference expert frame; it
    is only invoked when the cache is missing, since loading that frame costs about 40
    seconds on Qwen3. Returns a DataFrame with rank, sublayer, n_experts, reference_ap.
    """
    if rank_path.exists():
        return pd.read_csv(rank_path)

    log.info(f"  Building sublayer rank from AP {reference_ap} expert counts (first run)...")
    counts = sublayer_expert_counts(build_expert_df())
    rank_df = pd.DataFrame({
        "rank": range(1, len(counts) + 1),
        "sublayer": counts.index,
        "n_experts": counts.values,
        "reference_ap": reference_ap,
    })
    save_dataframe(rank_df, rank_path)
    return rank_df


def build_analysis_scopes(full_expert_df: pd.DataFrame, sublayer_rank: pd.DataFrame) -> list:
    """
    Build the scope list every module iterates over: the whole model first, then one
    scope per sublayer in rank order. Sublayers left with no experts at the current AP
    threshold are skipped with a warning rather than producing empty output folders.
    """
    scopes = [AnalysisScope(key=WHOLE_MODEL_SCOPE_KEY, label="whole model", dir_name=None,
                            is_whole_model=True, expert_df=full_expert_df)]
    for _, row in sublayer_rank.sort_values("rank").iterrows():
        sublayer = row["sublayer"]
        scope_df = filter_expert_data_to_sublayer(full_expert_df, sublayer)
        if scope_df.empty:
            log.warning(f"  No experts in sublayer '{sublayer}' at this AP threshold, skipping its scope.")
            continue
        scopes.append(AnalysisScope(key=sublayer, label=sublayer,
                                    dir_name=f"{int(row['rank'])}_{sublayer}",
                                    is_whole_model=False, expert_df=scope_df))
    return scopes


def scope_out_dir(module_dir: pathlib.Path, scope: AnalysisScope) -> pathlib.Path:
    """Output folder for one scope of one module, created on demand."""
    out_dir = module_dir if scope.is_whole_model else module_dir / "sublayers" / scope.dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def scope_summary_row(scope: AnalysisScope, **metrics) -> dict:
    """
    One row of a module's sublayer_comparison table: the scope's identity followed by
    that module's headline metrics. n_experts is carried on every row because every
    percentage or correlation below it has to be read against the mass it rests on.
    """
    return {"scope": scope.label, "is_whole_model": scope.is_whole_model,
            "n_experts": len(scope.expert_df), **metrics}


def axis_variants(scope: AnalysisScope) -> list:
    """
    Layer-axis variants an order-based module should run for a given scope, as
    (filename_suffix, expert_df, axis_label) tuples with the canonical variant first.

    Modules split into two kinds. Set-based ones (3, 5, and module 7's global prototype)
    read expert rows as an unordered set, so the layer axis never enters and they ignore
    this helper. Order-based ones (1, 2, 6, 7's per-layer part) read layer_idx as depth,
    and for those the whole-model scope is genuinely ambiguous: the flat axis carries
    every layer at full resolution but interleaves sublayer types, while the block axis
    carries the same expert mass on a real depth axis. Both are produced and both are
    kept, suffixed "_by_layer" and "_by_block", with the block variant first because it
    is the one downstream modules consume. A sublayer scope already holds exactly one
    layer per block, so its two variants would be identical and only one is run.
    """
    if scope.is_whole_model:
        return [("_by_block", to_block_axis(scope.expert_df), "block (depth) axis"),
                ("_by_layer", scope.expert_df, "flat layer axis")]
    return [("", scope.expert_df, "layer axis")]


def to_block_axis(expert_allocation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-key expert rows from the flat layer axis onto a block (depth) axis, summing the
    sublayers within each transformer block into one bin.

    build_layer_mapping_from_layers numbers layers as block * n_sublayers + sublayer
    position, so on Qwen3 indices 1..7 are all block 0. Any descriptor that reads
    layer_idx as depth therefore misreads that flat axis: gearys_c differences adjacent
    indices, which on the flat axis compares q_proj against k_proj of the SAME block and
    measures sublayer-type alternation rather than depth smoothness, while peak_layer
    returns whichever sublayer is densest for nearly every word. Aggregating to blocks
    gives a genuine depth axis carrying the full expert mass of the model, with one bin
    per block (28 on Qwen3, 12 on GPT-2).

    layer_idx becomes block + 1 and layer_name becomes "<block+1>.B.<block>", preserving
    the leading-number-is-layer_idx convention that build_layer_probability_matrix parses.
    Categories cover every block of the input's layer support, including blocks with no
    experts at a strict threshold. Applying this to a single-sublayer frame is an identity
    relabel, since such a frame already holds exactly one layer per block.
    """
    block_of_row = expert_allocation_df['layer_name'].astype(str).str.extract(r'^\d+\.L\.(\d+)\.')[0].astype(int)
    support = sorted(pd.Series(expert_allocation_df['layer_name'].cat.categories)
                     .astype(str).str.extract(r'^\d+\.L\.(\d+)\.')[0].astype(int).unique())

    block_axis_df = expert_allocation_df.copy()
    block_axis_df['layer_idx'] = block_of_row.to_numpy() + 1
    block_axis_df['layer_name'] = pd.Categorical(
        [f"{b + 1}.B.{b}" for b in block_of_row],
        categories=[f"{b + 1}.B.{b}" for b in support], ordered=True)
    return block_axis_df


def expert_presence_matrix(expert_sets_df: pd.DataFrame, concepts: list) -> sparse.csr_matrix:
    """
    Binary concept-by-expert presence matrix, one row per entry of ``concepts`` in the
    given order and one column per distinct (layer_idx, unit) pair present in
    expert_sets_df. Concepts absent from the frame become all-zero rows, so the caller
    always gets a matrix of exactly len(concepts) rows.

    Sparse rather than a dense pivot_table because the expert space is the whole neuron
    space: filtered to one sublayer it is already about 172k (layer_idx, unit) pairs on
    Qwen3, and unfiltered it is closer to 400k, so a dense 204-by-400k float matrix costs
    hundreds of megabytes to express data that is a fraction of a percent non-zero.

    Experts are keyed on the (layer_idx, unit) PAIR because the raw `unit` column is only
    the neuron index within a layer, so identical indices from different layers would
    otherwise collapse into one column and inflate every intersection. Duplicate
    (concept, layer_idx, unit) rows are collapsed to a single 1, matching the mean
    aggregation of the pivot_table this replaces, so a repeated expertise row can never
    count twice.
    """
    row_of = {concept: i for i, concept in enumerate(concepts)}
    rows = expert_sets_df["concept"].map(row_of)
    keep = rows.notna().to_numpy()

    layer_idx = expert_sets_df.loc[keep, "layer_idx"].to_numpy(dtype=np.int64)
    unit = expert_sets_df.loc[keep, "unit"].to_numpy(dtype=np.int64)
    if len(layer_idx) == 0:
        return sparse.csr_matrix((len(concepts), 0), dtype=np.int32)

    # Fold the pair into one integer key before factorizing: both parts are non-negative
    # ints, so multiplying the layer by (max unit + 1) keeps the mapping injective.
    pair_codes = pd.factorize(layer_idx * (unit.max() + 1) + unit)[0]

    presence = sparse.csr_matrix(
        (np.ones(len(pair_codes), dtype=np.int32),
         (rows.to_numpy()[keep].astype(np.int32), pair_codes)),
        shape=(len(concepts), pair_codes.max() + 1), dtype=np.int32)
    # COO-to-CSR sums duplicates, so flatten any resulting 2s back to a binary indicator.
    presence.data[:] = 1
    return presence


def expert_set_overlap_matrices(expert_sets_df: pd.DataFrame, concepts: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pairwise expert-set overlap for every concept pair, as three square (n, n) arrays
    aligned with ``concepts``: (intersection, jaccard, overlap).

    intersection holds raw shared-expert counts, with each concept's own set size on the
    diagonal. jaccard is |A and B| / |A or B| and overlap is |A and B| / min(|A|, |B|),
    both as fractions in [0, 1] (callers scale to percentages), and both defined as 0
    when their denominator is 0 so a concept with no retained experts at a strict AP
    threshold does not produce NaNs.
    """
    presence = expert_presence_matrix(expert_sets_df, concepts)
    intersection = np.asarray((presence @ presence.T).todense(), dtype=np.float64)

    sizes = np.asarray(presence.sum(axis=1)).ravel()
    union = sizes[:, None] + sizes[None, :] - intersection
    min_size = np.minimum(sizes[:, None], sizes[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)
        overlap = np.where(min_size > 0, intersection / min_size, 0.0)
    return intersection, jaccard, overlap


def category_alignment_metrics(pair_similarity: np.ndarray, concept_categories: np.ndarray,
                               n_permutations: int = 0, rng: np.random.Generator = None) -> dict:
    """
    How well expert-set similarity tracks category membership, over all concept pairs.

    roc_auc (primary): P(random same-category pair is more similar than a random
    different-category pair) -- rank-based (equals Mann-Whitney U / (n1*n2)), so immune
    to the same/different pair imbalance and to the skewed shape of Jaccard values.
    Computed via the rank-sum identity with average ranks for ties (identical to
    sklearn's roc_auc_score); ranking once and re-summing per shuffle is what keeps
    9999 permutations cheap.
    category_alignment_r (companion): Pearson correlation between the same-category
    indicator (1 if a pair shares a category, 0 otherwise) and pair_similarity itself,
    r = corr(same, pair_similarity) in [-1, 1], called point-biserial because one of
    the two inputs is binary. Positive r means same-category pairs run more similar on
    average, and r^2 is the fraction of pair_similarity's variance explained by category
    membership alone (the RSA-style linear reading of categorical-model alignment).
    Unlike the rank-based AUC, r assumes a roughly linear relationship, so a skewed
    similarity distribution or a few extreme pairs can pull r away from what the AUC's
    ranking shows, which is why the two are read together rather than interchangeably.
    mantel_p: permutation p-value for the AUC. Pairs are not independent observations
    (each concept appears in n-1 pairs), so category labels are shuffled across
    CONCEPTS (never across pairs), the indicator rebuilt, and the AUC recomputed --
    preserving the similarity geometry and category sizes while breaking only the
    concept-to-category correspondence under test. Smallest reportable value is
    1/(n_permutations + 1). n_permutations=0 (the default) skips the test entirely and
    returns NaN, for callers that want the AUC as a cheap descriptor rather than a
    hypothesis test; module 1's informativeness table is the one place that pays for it.

    Degenerate sublayers whose pair similarities are all identical (e.g. at strict AP
    thresholds no concept pair shares any expert) short-circuit to AUC=0.5, r=NaN,
    p=1.0: with zero similarity variation there is nothing to align, and pearsonr on a
    constant vector is undefined (would only emit a ConstantInputWarning).
    """
    n = len(concept_categories)
    iu = np.triu_indices(n, k=1)
    same = (concept_categories[:, None] == concept_categories[None, :])[iu]

    if pair_similarity.min() == pair_similarity.max():
        return {"roc_auc": 0.5, "category_alignment_r": np.nan, "mantel_p": 1.0}

    # Category sizes (hence the number of same-category pairs n1) are invariant under
    # label permutation, so ranks and the U-statistic constants are precomputed once.
    ranks = stats.rankdata(pair_similarity)
    n1 = int(same.sum())
    n0 = len(same) - n1
    rank_offset = n1 * (n1 + 1) / 2

    def rank_auc(same_mask):
        return (ranks[same_mask].sum() - rank_offset) / (n1 * n0)

    auc = rank_auc(same)
    r, _ = stats.pearsonr(same.astype(float), pair_similarity)

    if not n_permutations:
        return {"roc_auc": auc, "category_alignment_r": r, "mantel_p": np.nan}

    n_at_least = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(concept_categories)
        same_perm = (permuted[:, None] == permuted[None, :])[iu]
        if rank_auc(same_perm) >= auc:
            n_at_least += 1
    mantel_p = (1 + n_at_least) / (1 + n_permutations)

    return {"roc_auc": auc, "category_alignment_r": r, "mantel_p": mantel_p}


def pair_similarity_vector(expert_sets_df: pd.DataFrame, concepts: list) -> np.ndarray:
    """
    Jaccard similarity of raw expert sets for every unordered pair of ``concepts``,
    as a flat vector aligned with np.triu_indices(len(concepts), k=1). Expert sets are
    binary membership keyed on (layer_idx, unit); no probability normalization.
    """
    _, jaccard, _ = expert_set_overlap_matrices(expert_sets_df, concepts)
    return jaccard[np.triu_indices(len(concepts), k=1)]


def pair_shared_count_vector(expert_sets_df: pd.DataFrame, concepts: list) -> np.ndarray:
    """
    Raw count of shared experts for every unordered concept pair, aligned with
    np.triu_indices(len(concepts), k=1). Unlike Jaccard, this is not normalized by
    set size, so it is the more interpretable x-axis for a per-pair scatter (Jaccard
    compresses most pairs toward zero regardless of how many experts they actually share).
    """
    intersection, _, _ = expert_set_overlap_matrices(expert_sets_df, concepts)
    return intersection[np.triu_indices(len(concepts), k=1)]


# ---------------------------------------------------------------------------
# Layer-profile similarity (modules 3, 4, 5)
#
# The expert-set primitives above compare two words by WHICH neurons they share.
# This block compares them by HOW their experts are spread over the layers: two
# words can put the same proportion of their experts at the same depths and still
# share no neuron at all, which every set-based metric scores as zero.
#
# The measure is Jensen-Shannon on the row-normalized layer profiles, reported as
# a similarity on Jaccard's 0-100 scale, plus a z-score against other real pairs of
# comparable expert counts. The z is what makes the metric comparable across pairs:
# Jensen-Shannon on normalized profiles is already scale-invariant algebraically, but
# plug-in entropy is biased low by about (K-1)/(2 n ln2) bits, so a small expert set
# yields a spuriously spiky profile and an inflated divergence, and raw similarity
# ends up tracking expert-set size far more than it tracks any real agreement.
# ---------------------------------------------------------------------------

# Below this many experts a word has no usable layer profile: the multinomial
# estimate is nearly all noise and its null is degenerate. Such words are NaN
# throughout, deliberately unlike expert_set_overlap_matrices, which returns 0 for
# an empty set. Zero is TRUE for Jaccard (the word shares no experts) but FALSE
# here, since it would assert "maximally different layer distribution" about a word
# that has no layer distribution at all.
MIN_PROFILE_EXPERTS = 2

# Row block size for the pairwise mixture term. The full (n, n, L) tensor is 65 MB
# at float64 on Qwen3 (204 words x 196 layers); blocking holds it near 10 MB.
_JSD_BLOCK_ROWS = 32

# Reference-set size for the count-matched null, as a fraction of the available pairs,
# with an absolute floor and ceiling. It scales rather than sitting at a constant because
# the reference set must be a small enough SHARE of the data to stay local in the
# (log n_a, log n_b) count space, or the strong count gradient inside the neighbourhood
# leaks back into z. Measured on synthetic null data, a fixed k=200 leaves a residual
# count correlation of +0.04 at 204 words (20,706 pairs, k is 1% of them) but +0.41 at 60
# words (1,770 pairs, where the same k is 11%).
#
# On the Richie-HSJ item set this is insurance rather than a live fix: the pool holds 204
# words and never drops below 187 even at AP 0.9, since a word needs only
# MIN_PROFILE_EXPERTS experts to enter it, so the fraction resolves to 174-207 across the
# whole sweep and a fixed 200 would behave the same. It matters for smaller item sets,
# such as the 60-concept Mitchell stimulus set used elsewhere in this repository, where a
# constant k would land squarely in the leaky regime. The floor keeps the standard
# deviation estimable there.
NULL_NEIGHBOR_FRACTION = 0.01
NULL_NEIGHBORS_MIN = 40
NULL_NEIGHBORS_MAX = 300


def _entropy_bits(prob: np.ndarray) -> np.ndarray:
    """
    Shannon entropy in bits along the last axis. xlogy returns 0 where p == 0, so the
    0*log(0) = 0 convention needs no masking.
    """
    return -xlogy(prob, prob).sum(axis=-1) / np.log(2.0)


def _profile_jsd_matrix(prob: np.ndarray) -> np.ndarray:
    """
    Pairwise Jensen-Shannon divergence in bits between the rows of ``prob``, as a square
    (n, n) array. JSD(p, q) = H(m) - (H(p) + H(q))/2 with m = (p + q)/2, which is bounded
    in [0, 1] when H is in bits.

    The mixture entropy is the only term needing a pairwise tensor, so it is accumulated
    in row blocks rather than materializing (n, n, L) at once.
    """
    n = len(prob)
    row_entropy = _entropy_bits(prob)
    mixture_entropy = np.empty((n, n), dtype=np.float64)

    for start in range(0, n, _JSD_BLOCK_ROWS):
        stop = min(start + _JSD_BLOCK_ROWS, n)
        mixture = 0.5 * (prob[start:stop, None, :] + prob[None, :, :])
        mixture_entropy[start:stop] = _entropy_bits(mixture)

    jsd = mixture_entropy - 0.5 * (row_entropy[:, None] + row_entropy[None, :])
    # Floating-point error can push an identical pair a hair below 0 or a disjoint pair a
    # hair above 1, and the square root downstream would turn the former into a NaN.
    return np.clip(jsd, 0.0, 1.0)


def _jsd_to_similarity(jsd: np.ndarray) -> np.ndarray:
    """
    Jensen-Shannon divergence in bits to a 0-100 similarity, 100 * (1 - sqrt(JSD)).

    The square root is the Jensen-Shannon DISTANCE, a true metric, so the result behaves
    like a proper distance for any downstream model consuming it as a feature. It also
    expands the near-zero region where the observed values bunch, which keeps usable
    variance instead of saturating at 100.
    """
    return 100.0 * (1.0 - np.sqrt(jsd))


# scipy.spatial.distance.cdist does not forward a `base` argument to its jensenshannon
# kernel, so what it returns is the Jensen-Shannon distance in NATS. The divergence scales
# with the log base, JSD_bits = JSD_nats / ln 2, and the distance is its square root, so
# dividing by sqrt(ln 2) converts exactly.
_SQRT_LN2 = np.sqrt(np.log(2.0))


def _zero_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pairs where either profile carries no mass at all, as an (n, m) boolean mask.

    An all-zero row is not a profile, it is the absence of one, and every kernel below
    marks such pairs NaN rather than returning a number for them. That is the same
    convention MIN_PROFILE_EXPERTS enforces upstream, restated at the kernel so a caller
    reaching these functions directly cannot get a silently meaningless value. It matters
    because the library functions disagree about the degenerate case: cdist's
    jensenshannon divides by the row sum and returns inf, its braycurtis returns NaN, and
    scipy.stats.wasserstein_distance raises outright.
    """
    return (a.sum(axis=1)[:, None] <= 0) | (b.sum(axis=1)[None, :] <= 0)


def _js_distance_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Jensen-Shannon distance in bits, sqrt(JSD), from scipy's kernel."""
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = cdist(a, b, metric="jensenshannon") / _SQRT_LN2
    return np.where(_zero_mass(a, b), np.nan, distance)


def _js_distance_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """The historical profile agreement, 100 * (1 - sqrt(JSD bits)). Square (n, n)
    when q is None, cross (n, m) otherwise. Higher means more similar.

    Delegates to scipy.spatial.distance.cdist's `jensenshannon` kernel rather than
    computing the mixture entropy here, so the definition a reader has to trust is the
    library's documented one."""
    return 100.0 * (1.0 - _js_distance_bits(*_pair_inputs(p, q)))


def _js_divergence_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """
    Jensen-Shannon DIVERGENCE as an agreement, 100 * (1 - JSD bits).

    Same quantity as js_distance without the square root, so the two rank every pair
    identically and differ only in spacing. The root is a monotone transform, which is
    why registering both is not redundant only for the linear consumers: the ranker and
    the correlations see one metric, the ordinary least squares cells of module 9 see
    two, since a linear model in sqrt(JSD) is not a linear model in JSD. The unrooted
    form compresses the near-zero region the rooted one expands, so it carries less
    resolution where the observed values bunch, which is the reason js_distance and not
    this one is the default.
    """
    return 100.0 * (1.0 - _js_distance_bits(*_pair_inputs(p, q)) ** 2)


def _pair_inputs(p: np.ndarray, q: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """The (P, Q) a cross-form kernel operates on. Every registered metric is symmetric
    and needs no separate square path, so f(P) is exactly f(P, P)."""
    return (p, p) if q is None else (p, q)


def _correlation_agreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    100 * Pearson r between every row of a and every row of b, via cdist's `correlation`
    kernel, which returns 1 - r. A row of zero variance has no deviation to correlate and
    comes back NaN from scipy, which is the reading this pipeline wants and states.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (1.0 - cdist(a, b, metric="correlation"))


def _cosine_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """
    Cosine agreement between layer profiles, 100 * cos(p, q), via cdist's `cosine` kernel,
    which returns 1 - cos.

    Profiles are non-negative, so this lives in [0, 100]: 100 for identical shapes, 0 for
    profiles on disjoint layers. Unlike the divergence family it is scale-free by
    construction rather than by normalization, and it weights the bins a profile actually
    occupies, so two words agreeing on a few heavily loaded blocks score high even when
    they disagree about the long tail of near-empty ones.

    A row of length 0 has no direction, so its cosine is NaN rather than 0.
    """
    a, b = _pair_inputs(p, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        agreement = 100.0 * (1.0 - cdist(a, b, metric="cosine"))
    return np.where(_zero_mass(a, b), np.nan, agreement)


def _pearson_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """
    Pearson correlation between layer profiles across bins, on a 100 * r scale.

    Centring each profile by its own mean is what separates this from cosine: the shared
    baseline every word carries, the model's global expert density over depth, is removed
    from both sides, so the metric reads the DEVIATION from that baseline. A pair can then
    score negative, meaning one word is heavy where the other is light, which no member of
    the divergence family can express. Identical profiles score 100, and a pair of
    one-block spikes on different blocks scores below 0 rather than at it.

    A profile that is exactly uniform over the bins has zero variance and no deviation to
    correlate, so its row is NaN.
    """
    return _correlation_agreement(*_pair_inputs(p, q))


def _spearman_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """
    Spearman rank correlation between layer profiles across bins, on a 100 * rho scale.

    The rank transform is applied WITHIN each profile, ranking that word's bins against
    each other, and the correlation is then Pearson on those ranks, which is what
    scipy.stats.rankdata followed by cdist's `correlation` kernel computes. It answers "do
    these two words order the blocks the same way", which is the weakest and most robust
    reading in the registry: the magnitudes drop out entirely, so a pair agreeing on which
    blocks are busiest scores high even when one word's distribution is far peakier.

    Ties are averaged, which matters here more than in most rank statistics. At a strict AP
    threshold most bins of a small expert set are exactly 0, they all share one average
    rank, and the correlation is then carried by the few occupied blocks. A profile whose
    bins are all equal, including an all-zero row, has constant ranks and comes back NaN.
    """
    a, b = _pair_inputs(p, q)
    return _correlation_agreement(stats.rankdata(a, axis=1), stats.rankdata(b, axis=1))


def _hellinger_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """
    Hellinger agreement, 100 * (1 - H), with H = sqrt(1 - BC) and BC the Bhattacharyya
    coefficient sum_k sqrt(p_k q_k).

    A true metric on the simplex, bounded in [0, 1], finite on disjoint support, so it
    shares the properties that made Jensen-Shannon the original choice. It differs in
    where it puts its resolution: the square root inside the sum lifts the small bins, so
    Hellinger is more sensitive than the divergence family to agreement in a profile's
    THIN tail and less dominated by its mode. The whole matrix is one matrix product of
    the square-rooted profiles, so it needs none of the blocking the mixture entropy does.
    """
    a, b = _pair_inputs(p, q)
    # Computed as the Euclidean distance between the square-rooted profiles over sqrt(2),
    # which is the Hellinger distance identically, rather than as sqrt(1 - BC). The two are
    # equal on paper and NOT equal in floating point: for similar profiles BC approaches 1
    # and the subtraction cancels catastrophically, which on a pair whose true distance is
    # 2.16e-09 returned essentially 0, a 100 percent relative error, where this form errs by
    # 7e-18. Similar pairs are exactly the regime the clustering of module 9's Study C reads.
    distance = cdist(np.sqrt(a), np.sqrt(b), metric="euclidean") / np.sqrt(2.0)
    return np.where(_zero_mass(a, b), np.nan, 100.0 * (1.0 - distance))


def _profile_jaccard_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """
    Weighted Jaccard (Ruzicka) agreement between layer profiles, on a 0 to 100 scale.

        J(p, q) = sum_k min(p_k, q_k) / sum_k max(p_k, q_k)

    The continuous generalization of the Jaccard index used on expert SETS elsewhere in
    this pipeline: replace each set by its indicator vector and this returns the ordinary
    |A and B| / |A or B|. Registering it makes the pipeline's two geometries comparable
    under ONE functional form, so "which neurons does a word recruit" and "where along
    depth does it put them" can be contrasted without the measure itself changing between
    them, which is the only reason the difference between them can be attributed to the
    representation rather than to the statistic.

    It also fills the one family the registry was missing. Profiles are normalized, so
    sum_k max = 2 - sum_k min, and sum_k min(p, q) = 1 - TV with TV the total variation
    distance. Hence

        J = (1 - TV) / (1 + TV),

    a strictly decreasing function of TV, which makes this the registry's only L1 or
    total-variation member beside the divergence, correlation and transport families.
    Being a monotone transform of TV, it ranks every pair exactly as TV does, so any
    rank-based statistic reads the two identically while average linkage and the linear
    consumers do not, since a mean of transformed distances is not the transform of a mean.

    One minus this agreement is a true metric on the simplex, which is what makes it safe
    to cluster on.
    """
    a, b = _pair_inputs(p, q)
    # cdist's `braycurtis` kernel is sum|u - v| / sum|u + v|, which on two normalized
    # non-negative profiles has denominator 2 and is therefore the total variation distance
    # exactly. J = (1 - TV) / (1 + TV) follows from sum max = 2 - sum min and
    # sum min = 1 - TV, so the weighted Jaccard comes straight out of a library kernel with
    # no pairwise tensor to build.
    with np.errstate(divide="ignore", invalid="ignore"):
        total_variation = cdist(a, b, metric="braycurtis")
        jaccard = (1.0 - total_variation) / (1.0 + total_variation)
    return np.where(_zero_mass(a, b), np.nan, 100.0 * jaccard)


def _wasserstein_similarity(p: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    """
    First Wasserstein (earth mover's) agreement over the bin axis, on a 0 to 100 scale.

    W1 between two distributions on an evenly spaced axis is the L1 distance between their
    cumulative distribution functions, sum_k |CDF_p(k) - CDF_q(k)| over the first K-1 bins.
    Its maximum is K-1, all mass at the first bin against all mass at the last, so the
    reported agreement is 100 * (1 - W1 / (K-1)).

    This is the ONLY metric in the registry that reads bin ORDER, and that is the point of
    including it. Every other measure is permutation invariant, so a word peaking at block
    3 and a word peaking at block 4 are exactly as different to them as a word peaking at
    block 27, which is the wrong reading of a depth axis. Wasserstein charges the distance
    the mass has to travel, so near misses in depth score as near misses.

    The consequence is that it is meaningful only on an axis whose adjacency is real, that
    is on the BLOCK axis, where bin k+1 is one transformer block deeper than bin k. On the
    flat layer axis, adjacent bins are different projection types of the SAME block, so the
    transport cost it measures is largely sublayer alternation rather than depth. Callers
    on the flat axis get a number, and it is not a depth statement.

    The CDF difference is the only term needing a pairwise tensor, so it is accumulated in
    row blocks, mirroring _profile_jsd_matrix.
    """
    a, b = _pair_inputs(p, q)
    n_bins = a.shape[1]
    if n_bins < 2:
        return np.full((len(a), len(b)), 100.0)

    # scipy.stats.wasserstein_distance takes support values and weights, so the bins are the
    # support and each profile is the weighting over it. It is a per-pair call rather than a
    # vectorised one, which costs about 0.6 seconds on a 205 by 205 matrix against 2
    # milliseconds for the cumulative-difference form it replaces. That is affordable at one
    # matrix per scope and buys a definition the reader can check against scipy's own docs.
    # The two agree to 1.8e-14, so the previous hand-rolled version was correct.
    bins = np.arange(n_bins, dtype=np.float64)
    empty = _zero_mass(a, b)
    distance = np.full((len(a), len(b)), np.nan)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            # wasserstein_distance rejects a zero-sum weight vector outright, so the
            # degenerate pairs are skipped and left NaN rather than guarded inside scipy.
            if not empty[i, j]:
                distance[i, j] = stats.wasserstein_distance(bins, bins, u, v)
    return 100.0 * (1.0 - np.clip(distance / (n_bins - 1), 0.0, 1.0))


# Registry of profile agreement measures. Contract: f(P, Q=None) -> (n, m) agreement
# matrix over row-normalized profiles, square when Q is None, ORIENTED so that higher
# always means more similar. Orientation is the entire contract, scale deliberately is
# not, because every consumer standardizes or ranks the feature and forcing a shared
# scale would manufacture false comparability. Adding a metric means adding one entry
# here, a display label below, and its name in ACTIVE_PROFILE_METRICS.
#
# The eight entries are four families. The divergence family (js_distance, js_divergence,
# hellinger) compares the profiles as distributions and is bounded, symmetric and finite on
# disjoint support. The correlation family (cosine, pearson, spearman) compares them as
# vectors over bins, and the latter two can go negative because they read deviation from a
# baseline rather than overlap. Wasserstein stands alone as the only metric reading bin
# ORDER, so it is meaningful only on the block axis, where adjacent bins are adjacent
# depths. profile_jaccard is the L1 family, a monotone transform of total variation, and is
# the same functional form as the expert-SET Jaccard used elsewhere, which is what lets the
# set geometry and the depth geometry be compared without changing the statistic.
PROFILE_METRICS = {
    "js_distance": _js_distance_similarity,
    "wasserstein": _wasserstein_similarity,
    "cosine": _cosine_similarity,
    "pearson": _pearson_similarity,
    "spearman": _spearman_similarity,
    "js_divergence": _js_divergence_similarity,
    "hellinger": _hellinger_similarity,
    "profile_jaccard": _profile_jaccard_similarity,
}

DEFAULT_PROFILE_METRIC = "js_distance"

# Metrics computed by every consumer of the registry: modules 3 and 5 draw one figure per
# entry, module 9 instantiates every cell of both studies on each. Narrow this list to make
# a verification run cheap, restore it before a real sweep. DEFAULT_PROFILE_METRIC stays
# FIRST, because it is the metric the cross-scope summary rows and the legacy column names
# report, and a reader comparing against an older results tree should find it in place.
ACTIVE_PROFILE_METRICS = list(PROFILE_METRICS)

# Short display names for figure titles and axis labels. Kept out of PROFILE_METRICS so the
# registry stays a plain name-to-callable mapping its contract check can iterate.
PROFILE_METRIC_LABELS = {
    "js_distance": "Jensen-Shannon distance",
    "wasserstein": "Wasserstein",
    "cosine": "Cosine",
    "pearson": "Pearson",
    "spearman": "Spearman",
    "js_divergence": "Jensen-Shannon divergence",
    "hellinger": "Hellinger",
    "profile_jaccard": "Weighted Jaccard",
}

# Metrics whose agreement can be negative, because they measure deviation from a per-profile
# baseline rather than overlap. A figure drawn on one of these needs a signed axis and a
# reference line at 0, and reading its value as a percentage would be wrong.
SIGNED_PROFILE_METRICS = {"pearson", "spearman"}


def profile_metric_label(metric: str) -> str:
    """Display name for a registered metric, falling back to the raw key."""
    return PROFILE_METRIC_LABELS.get(metric, metric)


def profile_metric_value_label(metric: str) -> str:
    """
    Axis label for a metric's raw agreement value.

    Signed metrics are reported as a correlation times 100 rather than as a percentage,
    since "-42%" of an agreement is not a quantity anyone can read.
    """
    if metric in SIGNED_PROFILE_METRICS:
        return f"{profile_metric_label(metric)} correlation x 100"
    return "Agreement % (100 = identical profiles)"


def count_matched_z(values: np.ndarray, coords: np.ndarray,
                    neighbors: int = None) -> np.ndarray:
    """
    Standardize each entry of ``values`` against its nearest entries in ``coords``
    space, self excluded. The coordinate-agnostic core of the count-matched null:
    the pair null places pairs at (log min count, log max count) because a pair's
    two words are exchangeable, the concept-to-centroid null of module 9 places
    entries at (log concept count, log centroid pooled count) because those two
    objects are different kinds and their order carries meaning. Returns a flat
    array aligned with ``values``, NaN where fewer than 2 * NULL_NEIGHBORS_MIN
    entries exist or the local sd is 0.
    """
    z = np.full(len(values), np.nan)
    finite = np.isfinite(values) & np.isfinite(coords).all(axis=1)
    # A standard deviation over a handful of neighbours is not worth reporting.
    if finite.sum() < 2 * NULL_NEIGHBORS_MIN:
        return z

    usable_values = values[finite]
    usable_coords = coords[finite]
    if neighbors is None:
        neighbors = int(np.clip(round(NULL_NEIGHBOR_FRACTION * len(usable_values)),
                                NULL_NEIGHBORS_MIN, NULL_NEIGHBORS_MAX))
    k = min(neighbors, len(usable_values) - 1)
    _, idx = cKDTree(usable_coords).query(usable_coords, k=k + 1)

    # Drop each entry from its own reference set, so an entry never helps define the mean
    # it is measured against. Identity is matched rather than position, because many
    # entries share exact coordinates and the self-match need not land in column 0. Where
    # duplicates are numerous enough to push the self-match out of the neighbourhood
    # entirely, the last (furthest) column is dropped instead, which keeps every row at
    # exactly k references.
    self_match = idx == np.arange(len(usable_values))[:, None]
    drop = self_match & (self_match.cumsum(axis=1) == 1)
    drop[~self_match.any(axis=1), -1] = True
    keep = idx[~drop].reshape(len(usable_values), k)

    reference = usable_values[keep]
    mu = reference.mean(axis=1)
    sigma = reference.std(axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        z[finite] = np.where(sigma > 0, (usable_values - mu) / sigma, np.nan)
    return z


def _empirical_pair_null(similarity: np.ndarray, counts: np.ndarray,
                         neighbors: int = None) -> np.ndarray:
    """
    Standardize each pair's similarity against OTHER REAL PAIRS of comparable expert
    counts, returning the square (n, n) array of z-scores.

    The reference set matters more than the arithmetic. An earlier version drew both
    profiles from the pooled global layer profile, which is a null of no word-specific
    layer structure whatsoever. Real words do have idiosyncratic profiles, so every real
    pair scored far below that baseline (mean z about -12 on Qwen3 at AP 0.6) and the
    sign of z carried no information about the pair. The question the metric is asked is
    "do these two words agree on depth MORE THAN TWO ARBITRARY WORDS of these sizes do",
    so the comparison set has to be arbitrary words, not synthetic draws.

    Each pair is placed at (log min(n_a, n_b), log max(n_a, n_b)), which is symmetric in
    the pair by construction, and compared against its ``neighbors`` nearest pairs in that
    space, excluding itself. A k-nearest-neighbour reference rather than a fixed grid
    because the count distribution is heavily skewed: a grid would leave the sparse
    high-count corner with too few pairs to estimate a standard deviation from, while
    kNN adapts its bandwidth to the local density automatically and needs no interpolation
    or empty-cell handling.

    Building this symmetrised coordinate and gating both words on MIN_PROFILE_EXPERTS stays
    here as pair-specific, the neighbourhood search, self-exclusion and standardisation now
    delegate to count_matched_z, the coordinate-agnostic core shared with other null
    placements such as module 9's concept-to-centroid null.
    """
    n = len(counts)
    z = np.full((n, n), np.nan)
    iu = np.triu_indices(n, k=1)
    pair_values = similarity[iu]
    n_a, n_b = counts[iu[0]], counts[iu[1]]

    usable = np.isfinite(pair_values) & (n_a >= MIN_PROFILE_EXPERTS) & (n_b >= MIN_PROFILE_EXPERTS)
    pair_values_masked = np.where(usable, pair_values, np.nan)
    coords = np.full((len(pair_values), 2), np.nan)
    coords[usable] = np.stack([np.log(np.minimum(n_a, n_b)[usable]),
                               np.log(np.maximum(n_a, n_b)[usable])], axis=1)

    flat = count_matched_z(pair_values_masked, coords, neighbors)
    z[iu] = flat
    z.T[iu] = flat
    return z


def _layer_profiles(expert_allocation_df: pd.DataFrame, items: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Row-normalized layer profiles and expert counts for ``items``, in the given order.

    build_layer_probability_matrix groups by concept, so a word with no expert rows never
    enters its index; reindexing to ``items`` turns that absence into the NaN row this
    module treats as "no usable profile".
    """
    count_matrix, prob_matrix = build_layer_probability_matrix(expert_allocation_df)
    prob = prob_matrix.reindex(items).to_numpy(dtype=np.float64)
    counts = count_matrix.reindex(items).sum(axis=1).to_numpy(dtype=np.float64)
    return prob, counts


def layer_profile_matrices(expert_allocation_df: pd.DataFrame, items: list, *,
                           null_neighbors: int = None,
                           metric: str = DEFAULT_PROFILE_METRIC
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pairwise layer-profile agreement for every pair of ``items``, as three square (n, n)
    arrays: (jsd_bits, similarity_pct, z).

    jsd_bits is the Jensen-Shannon divergence between the two words' layer profiles, in
    bits, 0 for identical profiles and 1 for disjoint ones. similarity_pct is
    100 * (1 - sqrt(jsd_bits)), on the same scale and direction as Jaccard, with a
    diagonal of 100. z standardizes similarity_pct against OTHER REAL PAIRS of comparable
    expert counts (see _empirical_pair_null): positive means the two words agree in layer
    allocation more than two arbitrary words of those sizes do, negative means less.
    Read z, not similarity_pct, when asking whether a pair carries signal.

    The reason is that raw similarity is very largely a readout of expert-set size. On
    synthetic data where every word is drawn from one shared global profile, so that no
    pair has any true agreement at all, raw similarity still correlates with the pair's
    smaller expert count at Spearman 0.965. Removing that confound is the entire purpose
    of the z column.

    Because z is defined relative to the pairs actually present, it is a WITHIN-RUN
    ranking: its mean over all pairs is near 0 by construction, so it answers "which pairs
    agree more than comparable pairs" and not "do words agree on depth in absolute terms".
    Comparisons of z across AP thresholds or across scopes are therefore comparisons of
    relative structure, not of level.

    Rows and columns of words holding fewer than MIN_PROFILE_EXPERTS experts are NaN in
    all three arrays, as is the z diagonal (a word against itself has no meaningful null).

    Every caller must pass the SAME item list, since the null's reference set is drawn
    from the pairs of ``items``, so a shorter list would change the z of a given pair.

    metric names a PROFILE_METRICS entry, js_distance by default, and only the
    similarity and z outputs depend on it, jsd_bits always reports the Jensen-Shannon
    divergence.

    For several metrics at once call layer_profile_metric_matrices, which this delegates
    to: it shares the profile build and the Jensen-Shannon tensor across the metrics
    instead of repeating them per call.
    """
    return layer_profile_metric_matrices(expert_allocation_df, items, [metric],
                                         null_neighbors=null_neighbors)[metric]


def layer_profile_metric_matrices(expert_allocation_df: pd.DataFrame, items: list,
                                  metrics: list = None, *,
                                  null_neighbors: int = None) -> dict:
    """
    layer_profile_matrices for several metrics at once, as {metric: (jsd_bits,
    similarity, z)}. Defaults to ACTIVE_PROFILE_METRICS.

    Every metric reads the SAME row-normalized profiles over the same item list, so the
    profile build and the Jensen-Shannon tensor are computed once here rather than once
    per metric. Only the agreement matrix and its count-matched null are per metric, and
    the null is metric-agnostic by construction, so a newly registered metric gets its z
    with no null-side change at all.

    jsd_bits is the same array for every metric, since it always reports the
    Jensen-Shannon divergence regardless of which agreement was asked for. Each metric
    gets its own copy so a caller mutating one cannot reach into another's.

    See layer_profile_matrices for what the three arrays mean, why z rather than the raw
    similarity is the reading to trust, and why every caller must pass the same item list.
    """
    metrics = list(ACTIVE_PROFILE_METRICS if metrics is None else metrics)
    prob, counts = _layer_profiles(expert_allocation_df, items)
    n = len(items)
    usable = np.isfinite(counts) & (counts >= MIN_PROFILE_EXPERTS) & np.isfinite(prob).all(axis=1)

    if usable.sum() < 2:
        return {metric: (np.full((n, n), np.nan), np.full((n, n), np.nan),
                         np.full((n, n), np.nan)) for metric in metrics}

    idx = np.flatnonzero(usable)
    block = np.ix_(idx, idx)
    jsd = np.full((n, n), np.nan)
    jsd[block] = _profile_jsd_matrix(prob[idx])

    out = {}
    for metric in metrics:
        similarity = np.full((n, n), np.nan)
        similarity[block] = PROFILE_METRICS[metric](prob[idx])
        z = _empirical_pair_null(similarity, counts, null_neighbors)
        np.fill_diagonal(z, np.nan)
        out[metric] = (jsd.copy(), similarity, z)
    return out


def pair_layer_profile_vectors(expert_allocation_df: pd.DataFrame, items: list,
                               **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Layer-profile similarity and its null z-score for every unordered pair of ``items``,
    as two flat vectors aligned with np.triu_indices(len(items), k=1), mirroring
    pair_similarity_vector. Keyword arguments pass through to layer_profile_matrices.
    """
    _, similarity, z = layer_profile_matrices(expert_allocation_df, items, **kwargs)
    iu = np.triu_indices(len(items), k=1)
    return similarity[iu], z[iu]


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