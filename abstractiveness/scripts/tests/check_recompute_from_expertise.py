"""
Every kept measure of modules 1 and 2, recomputed from the raw expertise.csv files without the pipeline code,
agrees with a results tree. Run once after each full sweep, at a threshold where every word has experts.

Checks, on the whole model and the block axis wherever depth order matters:
  1. 1.1 mean distribution and expert counts per level, each level's profile summing to 100%, and the
     summary's 50% and 90% blocks, top blocks and thirds of depth.
  2. 1.2 entropy, normalised entropy and Geary's C per word, and label and member-average entropy per category.
  3. 1.3 expert share and mean Geary's C per sublayer and level, empty layers counted as zeros.
  4. 2.1 Jaccard and overlap of every member with its label, and the core shared by every member.
  5. 2.2 the pairwise Jaccard matrix and the category-mean matrix.
  6. 2.3 Spearman rho of Jaccard and of the probing-text TF-IDF against the human ratings, and the partial rho.
  7. 2.4 and 2.5 Spearman's rho of every panel, over all words or, when a variable is defined relative to the
     category, as the size-weighted mean of the per-category rho, and the six rows of the two partial figures.

Usage, from scripts/:
    python tests/check_recompute_from_expertise.py [--config gpt2_richie_hsj] [--ap 0.6]
"""
import argparse
import json
import pathlib
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import entropy, pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from abstractiveness_executor import MODEL_CONFIGS
from core.human_similarity_ratings import load_human_similarity
from _report import run

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXACT, ROUNDED = 1e-9, 1e-4
CATEGORY_RELATIVE = {"human_typicality", "jaccard_pct", "overlap_pct"}


def read_experts(job: tuple) -> tuple:
    """Concept, layer strings and units of the experts in one expertise.csv."""
    path, threshold = job
    frame = pd.read_csv(path, usecols=["ap", "layer", "unit", "concept"])
    experts = frame[frame["ap"] >= threshold]
    return str(frame["concept"].iloc[0]), experts["layer"].to_numpy(), experts["unit"].to_numpy()


def geary(values) -> float:
    """Geary's C of a sequence, NaN when constant or shorter than three."""
    values = np.asarray(values, dtype=float)
    spread = ((values - values.mean()) ** 2).sum()
    return np.nan if len(values) < 3 or spread == 0 else float((np.diff(values) ** 2).sum() / (2 * spread))


def rank_partial(frame: pd.DataFrame, x: str, y: str, controls: tuple) -> float:
    """Spearman rho of x and y partial on the controls, the Pearson r of their ranks' residuals on the controls'."""
    ranks = frame[[x, y, *controls]].rank()
    design = np.column_stack([np.ones(len(ranks)), ranks[list(controls)].to_numpy()])
    residual = lambda column: ranks[column] - design @ np.linalg.lstsq(design, ranks[column], rcond=None)[0]
    return pearsonr(residual(x), residual(y)).statistic


def spearman(frame: pd.DataFrame, x: str, y: str, controls: tuple = (), within: bool = False) -> float:
    """rank_partial over every word, or within each category averaged with weights proportional to its size, leaving
    out categories smaller than three words plus one per control."""
    frame = frame.dropna(subset=[x, y, *controls, *(["category"] if within else [])])
    if not within:
        return rank_partial(frame, x, y, controls)
    rhos = [(rank_partial(group, x, y, controls), len(group)) for _, group in frame.groupby("category")
            if len(group) >= 3 + len(controls)]
    rhos = [(rho, n) for rho, n in rhos if np.isfinite(rho)]
    return float(np.average([rho for rho, _ in rhos], weights=[n for _, n in rhos]))


class Recomputed:
    """Expert sets, block profiles and per-word tables of one model and threshold, built from expertise.csv."""

    def __init__(self, config: str, threshold: float):
        cfg = MODEL_CONFIGS[config]
        self.cfg, self.threshold = cfg, threshold
        self.tree = ROOT / "results" / cfg["output_subdir"] / f"AP_{threshold}"
        files = sorted((ROOT / "responses" / cfg["responses_subdir"] / cfg["model_subdir"]).glob(
            "**/expertise/expertise.csv"))
        with Pool(8) as pool:
            loaded = pool.map(read_experts, [(path, threshold) for path in files])
        mapping = pd.read_csv(ROOT / "assets" / cfg["layer_mapping_file"]).set_index("layer")
        parsed = mapping["layer_name"].str.extract(r"^\d+\.L\.(\d+)\.(.+)$")
        mapping["block"], mapping["sublayer"] = parsed[0].astype(int), parsed[1]
        self.mapping, self.n_blocks = mapping, int(mapping["block"].max()) + 1
        meta = pd.DataFrame(json.load((ROOT / "assets" / cfg["metadata_file"]).open())).set_index("concept")
        self.meta = meta
        self.words = [concept for concept, _, _ in loaded]
        self.rows = {}
        for concept, layers, units in loaded:
            idx = mapping.loc[layers, "layer_idx"].to_numpy()
            self.rows[concept] = pd.DataFrame({"layer_idx": idx, "unit": units, "block": mapping.loc[layers, "block"].to_numpy(),
                                               "sublayer": mapping.loc[layers, "sublayer"].to_numpy()})
        self.sets = {c: set(zip(r["layer_idx"], r["unit"])) for c, r in self.rows.items()}
        self.n = {c: len(s) for c, s in self.sets.items()}
        self.profile = {c: np.bincount(r["block"], minlength=self.n_blocks).astype(float) for c, r in self.rows.items()}

    def jaccard(self, concepts: list) -> np.ndarray:
        """Pairwise Jaccard in percent, 0 where both sets are empty."""
        keys = {}
        rows, cols = [], []
        for i, concept in enumerate(concepts):
            for key in self.sets.get(concept, ()):
                rows.append(i)
                cols.append(keys.setdefault(key, len(keys)))
        presence = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(concepts), max(len(keys), 1)))
        inter = (presence @ presence.T).toarray()
        sizes = np.asarray(presence.sum(axis=1)).ravel()
        union = sizes[:, None] + sizes[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, 100 * inter / union, 0.0)

    def members(self, category: str) -> list:
        """Members of a category holding experts."""
        return [c for c in self.meta.index[self.meta["category"] == category] if self.n.get(c, 0) > 0]


def check_distribution(m: Recomputed, failures: list) -> None:
    """Sections 1.1 and 1.2 against their block-axis tables."""
    section = m.tree / "1_expert_distribution"
    dist = pd.read_csv(section / "1.1_layer_expert_distribution/mean_expert_layer_distribution_by_block.csv")
    summary = pd.read_csv(section / "1.1_layer_expert_distribution/mean_expert_layer_distribution_summary_by_block.csv"
                          ).set_index("abstraction_level")
    for level in (1, 2):
        words = [c for c in m.words if m.meta.loc[c, "abstraction_level"] == level and m.n[c] > 0]
        if not words:
            continue
        mean = 100 * np.mean([m.profile[c] / m.n[c] for c in words], axis=0)
        theirs = dist[dist["abstraction_level"] == level]
        if np.abs(mean - theirs["mean_expert_allocation_pct"].to_numpy()).max() > EXACT:
            failures.append(f"1.1 level {level}: mean distribution differs")
        if (np.sum([m.profile[c] for c in words], axis=0) != theirs["total_expert_count"].to_numpy()).any():
            failures.append(f"1.1 level {level}: expert counts differ")
        if abs(mean.sum() - 100) > EXACT:
            failures.append(f"1.1 level {level}: mean distribution sums to {mean.sum()}")
        row = summary.loc[level]
        cumulative, by_mass = np.cumsum(mean), np.cumsum(np.sort(mean)[::-1])
        thirds = [mean[p].sum() for p in np.array_split(np.arange(len(mean)), 3)]
        expected = [int(np.argmax(cumulative >= 50)), int(np.argmax(cumulative >= 90)),
                    int(np.argmax(by_mass >= 50)) + 1, int(np.argmax(by_mass >= 90)) + 1]
        got = [int(row[column]) for column in ("cumulative_50pct_position", "cumulative_90pct_position",
                                               "top_bins_50pct", "top_bins_90pct")]
        if expected != got or np.abs(np.subtract(thirds, row[
                ["first_third_pct", "middle_third_pct", "last_third_pct"]].to_numpy(dtype=float))).max() > EXACT:
            failures.append(f"1.1 level {level}: summary {got} against {expected}")

    shape = pd.read_csv(section / "1.2_distribution_shape/shannon_entropy_concepts_by_block.csv").set_index("concept")
    worst = 0.0
    for concept, row in shape.iterrows():
        p = m.profile[concept] / m.n[concept]
        bits = entropy(p, base=2)
        mine = np.array([bits, bits / np.log2(m.n_blocks), geary(p)])
        theirs = row[["shannon_entropy", "shannon_entropy_norm", "gearys_c"]].to_numpy(dtype=float)
        worst = max(worst, np.nanmax(np.abs(mine - theirs)))
    if worst > EXACT:
        failures.append(f"1.2 entropy and Geary per word differ by {worst:.1e}")
    categories = pd.read_csv(section / "1.2_distribution_shape/shannon_entropy_categories_by_block.csv").set_index("category")
    for category, row in categories.iterrows():
        average = np.mean([m.profile[c] / m.n[c] for c in m.members(category)], axis=0)
        mine = [entropy(m.profile[category] / m.n[category], base=2), entropy(average, base=2)]
        if np.abs(np.subtract(mine, row[["shannon_entropy", "shannon_entropy_average"]].to_numpy(dtype=float))).max() > EXACT:
            failures.append(f"1.2 {category}: label or member-average entropy differs")


def check_sublayers(m: Recomputed, failures: list) -> None:
    """Section 1.3 shares and mean Geary's C per sublayer and level."""
    table = pd.read_csv(m.tree / "1_expert_distribution/1.3_sublayer_informativeness/sublayer_informativeness.csv")
    total = sum(m.n.values())
    for _, row in table.iterrows():
        sublayer, level = row["sublayer"], row["abstraction_level"]
        layers = sorted(m.mapping.loc[m.mapping["sublayer"] == sublayer, "layer_idx"])
        words = [c for c in m.words if m.meta.loc[c, "abstraction_level"] == level]
        counts = [m.rows[c].loc[m.rows[c]["sublayer"] == sublayer, "layer_idx"].value_counts()
                  .reindex(layers, fill_value=0).to_numpy() for c in words]
        share = 100 * sum(c.sum() for c in counts) / total
        gearys = [geary(c) for c in counts if c.sum() > 0]
        if abs(share - row["share_of_all_experts_pct"]) > EXACT:
            failures.append(f"1.3 {sublayer} level {level}: share {share} against {row['share_of_all_experts_pct']}")
        if gearys and abs(np.nanmean(gearys) - row["gearys_c"]) > EXACT:
            failures.append(f"1.3 {sublayer} level {level}: Geary {np.nanmean(gearys)} against {row['gearys_c']}")


def check_similarities(m: Recomputed, failures: list) -> np.ndarray:
    """Sections 2.1 and 2.2, returns the Jaccard matrix over the metadata order."""
    section = m.tree / "2_similarities"
    pairs = pd.read_csv(section / "2.1_category_concept_similarities/category_concept_similarity_metrics.csv")
    for _, row in pairs.iterrows():
        a, b = m.sets[row["concept"]], m.sets[row["category"]]
        mine = [100 * len(a & b) / len(a | b), 100 * len(a & b) / min(len(a), len(b))]
        if np.abs(np.subtract(mine, [row["jaccard_pct"], row["overlap_pct"]])).max() > EXACT:
            failures.append(f"2.1 {row['concept']}: Jaccard or overlap differs")
    core = pd.read_csv(section / "2.1_category_concept_similarities/category_core_overlap.csv").set_index("category")
    for category in core.index.drop("MEAN"):
        size = len(set.intersection(*(m.sets[c] for c in m.members(category))))
        if size != core.loc[category, "core_size_all_members"]:
            failures.append(f"2.1 {category}: core of every member {size} against {core.loc[category, 'core_size_all_members']}")

    concepts = list(m.meta.index)
    jaccard = m.jaccard(concepts)
    stored = pd.read_csv(section / "2.2_pairwise_similarities/jaccard_matrix.csv", index_col=0)
    order = [concepts.index(c) for c in stored.index]
    gap = np.abs(jaccard[np.ix_(order, order)] - stored.to_numpy())
    np.fill_diagonal(gap, 0)
    if gap.max() > 1e-6:
        failures.append(f"2.2 Jaccard matrix differs by {gap.max():.1e}")
    means = pd.read_csv(section / "2.2_pairwise_similarities/jaccard_category_mean_matrix.csv", index_col=0)
    for a in means.index:
        for b in means.columns:
            rows, cols = [concepts.index(c) for c in m.members(a)], [concepts.index(c) for c in m.members(b)]
            block = jaccard[np.ix_(rows, cols)]
            value = block[~np.eye(len(rows), dtype=bool)].mean() if a == b else block.mean()
            if abs(value - means.loc[a, b]) > 1e-6:
                failures.append(f"2.2 category mean {a} and {b}: {value} against {means.loc[a, b]}")
    return jaccard


def check_human(m: Recomputed, jaccard: np.ndarray, failures: list) -> None:
    """Section 2.3, Jaccard and text against the human ratings and the partial rho."""
    section = m.tree / "2_similarities/2.3_human_similarity_validation"
    concepts = list(m.meta.index)
    position = {c: i for i, c in enumerate(concepts)}
    sentences = ROOT / "assets" / m.cfg["probing_sentences_dir"] / "custom"
    documents = [" ".join(json.load((sentences / f"{c}.json").open())["sentences"]["positive"]) for c in concepts]
    vectors = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(documents)
    text = (vectors @ vectors.T).toarray()
    validation = pd.read_csv(section / "human_similarity_validation.csv")
    baseline = pd.read_csv(section / "human_similarity_text_baseline.csv").set_index("category")
    for category, group in load_human_similarity().groupby("category"):
        keep = group[group["concept_a"].map(m.n).gt(0) & group["concept_b"].map(m.n).gt(0)]
        a, b = keep["concept_a"].map(position).to_numpy(), keep["concept_b"].map(position).to_numpy()
        expert, words, human = jaccard[a, b], text[a, b], keep["mean_rating"].to_numpy()
        rhos = {"jaccard": spearmanr(expert, human).statistic, "text_tfidf": spearmanr(words, human).statistic}
        for metric, rho in rhos.items():
            stored = validation[(validation["metric"] == metric) & (validation["category"] == category)]["rho"].iloc[0]
            if abs(rho - stored) > ROUNDED:
                failures.append(f"2.3 {category} {metric}: rho {rho:.4f} against {stored}")
        expert_text = spearmanr(expert, words).statistic
        partial = (rhos["jaccard"] - expert_text * rhos["text_tfidf"]) / np.sqrt(
            (1 - expert_text ** 2) * (1 - rhos["text_tfidf"] ** 2))
        if abs(partial - baseline.loc[category, "partial_rho_jaccard_human_given_text"]) > EXACT:
            failures.append(f"2.3 {category}: partial rho {partial:.6f} against the table")


def check_correlations(m: Recomputed, failures: list) -> int:
    """Sections 2.4 and 2.5, the Spearman rho of every named panel of the correlation summaries."""
    section = m.tree / "2_similarities"
    corpus = m.cfg["frequency_corpus"]
    pairs = pd.read_csv(section / "2.1_category_concept_similarities/category_concept_similarity_metrics.csv")
    frame = pd.DataFrame({"concept": m.words, "expert_count": [m.n[c] or np.nan for c in m.words]})
    frame = frame.merge(pairs[["concept", "jaccard_pct", "overlap_pct"]], on="concept", how="left")
    frame["human_typicality"] = frame["concept"].map(m.meta[m.cfg["typicality_column"]]).astype(float)
    frame["frequency"] = frame["concept"].map(m.meta[f"frequency_zipf_{corpus}_lemma"]).astype(float)
    frame["category"] = frame["concept"].map(m.meta["category"])
    names = {"human_typicality": "human_typicality", "expert_count": "expert_count", "jaccard_pct": "jaccard",
             "overlap_pct": "overlap", "frequency": f"frequency_{corpus}"}

    def grouped(x: str, ys: list) -> dict:
        return {f"{names[x]}_vs_{names[y]}": spearman(frame, x, y, within=bool({x, y} & CATEGORY_RELATIVE))
                for y in ys}

    def partials(x: str, other: str) -> dict:
        data, base = frame.dropna(subset=[x, "jaccard_pct", other, "category"]), f"partial_correlation_jaccard_{x}"
        return {f"{base}_no_control": spearman(data, x, "jaccard_pct"),
                base: spearman(data, x, "jaccard_pct", (other,)),
                f"{base}_within_given_{other}": spearman(data, x, "jaccard_pct", (other,), within=True)}

    expected = {
        "2.4_typicality": grouped("human_typicality", ["expert_count", "jaccard_pct", "overlap_pct", "frequency"])
        | partials("human_typicality", "frequency"),
        "2.5_frequency_correlations": grouped("frequency", ["expert_count", "jaccard_pct", "overlap_pct",
                                                            "human_typicality"])
        | partials("frequency", "human_typicality"),
    }
    compared = 0
    for folder, values in expected.items():
        summary = pd.read_csv(section / folder / "correlation_summary.csv").set_index("plot_name")["spearman_rho"]
        for name, value in values.items():
            if name not in summary.index:
                failures.append(f"{folder} {name}: no row in correlation_summary.csv")
                continue
            if pd.isna(summary[name]):
                continue
            compared += 1
            if abs(value - summary[name]) > EXACT:
                failures.append(f"{folder} {name}: {value:.6f} against {summary[name]:.6f}")
    return compared


def checks(failures: list, config: str = "gpt2_richie_hsj", threshold: float = 0.6) -> str:
    m = Recomputed(config, threshold)
    check_distribution(m, failures)
    check_sublayers(m, failures)
    jaccard = check_similarities(m, failures)
    check_human(m, jaccard, failures)
    compared = check_correlations(m, failures)
    return (f"{config} at AP {threshold}: {len(m.words)} words, sections 1.1 to 2.5 recomputed, "
            f"{compared} correlation coefficients")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gpt2_richie_hsj", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--ap", type=float, default=0.6)
    arguments = parser.parse_args()
    sys.exit(run(__file__, lambda failures: checks(failures, arguments.config, arguments.ap)))
