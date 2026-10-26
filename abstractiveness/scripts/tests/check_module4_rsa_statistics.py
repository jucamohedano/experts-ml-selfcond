"""
Module 4's tests are two-sided and each draws from its own generator: the Mantel test of the expert-set against the
embedding similarity, the bootstrap interval, and the contrast of the middle third of depth against the outer thirds.

Runs rdm_mantel, compute_embedding_rsa_sweep and summarise_rsa_sweep on synthetic concepts, expert sets and
embeddings.

Checks:
  1. rdm_mantel gives scipy's Spearman rho over the concept pairs, the permutation floor for related matrices, the same
     p when one matrix changes sign, and NaN for a constant matrix.
  2. A layer's rho, Mantel p and bootstrap interval, pooled and held out, do not depend on the layers swept before it.
  3. The middle-third contrast has the exact two-sided p over all C(12, 4) assignments of twelve blocks, and with
     twenty-eight blocks a p in [floor, 1] that does not change when the curve is mirrored.

Usage, from scripts/:
    python tests/check_module4_rsa_statistics.py
"""
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules import module_4_embedding_rsa as m4
from utils.statistics import PERMUTATIONS
from _report import run

N_CONCEPTS, N_UNITS, WIDTH = 40, 60, 8


def mapping(blocks) -> pd.DataFrame:
    return pd.DataFrame({"layer_idx": [b + 1 for b in blocks], "layer_name": [f"{b}.L.{b}.mlp" for b in blocks],
                         "layer": [f"h.{b}.mlp" for b in blocks]})


def fixture(blocks: int = 6, seed: int = 0) -> tuple:
    """Concepts whose expert sets and embeddings both follow one latent geometry, the last layer's only weakly."""
    rng = np.random.default_rng(seed)
    concepts = [f"c{i:02d}" for i in range(N_CONCEPTS)]
    meta = pd.DataFrame({"concept": concepts, "category": np.repeat(list("abcd"), N_CONCEPTS // 4)})
    latent = rng.normal(size=(N_CONCEPTS, 4))
    rows = []
    for block in range(blocks):
        drive = latent @ rng.normal(size=(4, N_UNITS)) + rng.normal(0, 1, (N_CONCEPTS, N_UNITS))
        for i, unit in zip(*np.nonzero(drive > 1.5)):
            rows.append({"concept": concepts[i], "layer_idx": block + 1, "unit": unit, "ap": 0.7})
    noise = [1.0] * (blocks - 1) + [12.0]
    mean = np.hstack([latent @ rng.normal(size=(4, WIDTH)) + rng.normal(0, s, (N_CONCEPTS, WIDTH)) for s in noise])
    cache = {"concepts": concepts, "layers": list(mapping(range(blocks))["layer"]),
             "offsets": np.arange(blocks + 1) * WIDTH, "mean": mean,
             "half_a": mean + rng.normal(0, 1, mean.shape), "half_b": mean + rng.normal(0, 1, mean.shape)}
    return pd.DataFrame(rows), meta, cache


def sweep_table(rho: np.ndarray) -> pd.DataFrame:
    """A primary-metric sweep over len(rho) blocks whose held-out curve is rho."""
    return m4._layer_frame(mapping(range(len(rho)))).assign(
        embedding_metric=m4.PRIMARY_METRIC, is_analysis_sublayer=True, rho_expert_vs_embedding=rho,
        rho_expert_vs_embedding_holdout=rho, rho_ci95_lo=rho - 0.05, rho_ci95_hi=rho + 0.05,
        rho_holdout_ci95_lo=rho - 0.05, rho_holdout_ci95_hi=rho + 0.05, mantel_p_value=1e-4,
        noise_ceiling_splithalf=0.9, rho_over_noise_ceiling=rho / 0.9, expert_share_of_pool_pct=100 / len(rho),
        n_concepts=N_CONCEPTS, n_expert_units=500)


def check_mantel() -> str:
    experts, meta, cache = fixture()
    concepts = cache["concepts"]
    a = m4.squareform(m4.pair_similarity_vector(experts, concepts))
    b = m4.embedding_rdm(m4._embedding_matrix(cache, "mean", concepts, "h.0.mlp"))
    rho, p = m4.rdm_mantel(a, b)
    scipy_rho = stats.spearmanr(m4._offdiag(a), m4._offdiag(b)).statistic
    assert abs(rho - scipy_rho) < 1e-12 and p == 1 / (PERMUTATIONS + 1), f"related: rho {rho} p {p}"
    weak_rho, weak_p = m4.rdm_mantel(a, m4.embedding_rdm(m4._embedding_matrix(cache, "mean", concepts, "h.5.mlp")))
    flipped_rho, flipped_p = m4.rdm_mantel(a, -m4.embedding_rdm(m4._embedding_matrix(cache, "mean", concepts,
                                                                                       "h.5.mlp")))
    assert abs(flipped_rho + weak_rho) < 1e-12 and flipped_p == weak_p, \
        f"sign change: p {weak_p} against {flipped_p}, the test must be two-sided"
    constant = m4.rdm_mantel(a, np.ones_like(b))
    assert np.isnan(constant[0]) and np.isnan(constant[1]), f"constant matrix gives {constant}"
    return f"Mantel: related p {p}, weak rho {weak_rho:.3f} at p {weak_p} either sign"


def check_independence() -> str:
    experts, meta, cache = fixture()
    full, _ = m4.compute_embedding_rsa_sweep(experts, meta, cache, mapping(range(6)), sublayer_filter="mlp")
    alone, _ = m4.compute_embedding_rsa_sweep(experts, meta, cache, mapping([5]), sublayer_filter="mlp")
    columns = ["rho_expert_vs_embedding", "mantel_p_value", "rho_ci95_lo", "rho_ci95_hi",
               "rho_expert_vs_embedding_holdout", "mantel_p_value_holdout", "rho_holdout_ci95_lo",
               "rho_holdout_ci95_hi"]
    last = full[(full.embedding_metric == m4.PRIMARY_METRIC) & (full.block == 5)][columns].reset_index(drop=True)
    single = alone[alone.embedding_metric == m4.PRIMARY_METRIC][columns].reset_index(drop=True)
    assert last.equals(single), f"last layer depends on the layers before it:\n{last}\n{single}"
    return f"last layer alone: p {single.mantel_p_value[0]:.4f}, interval [{single.rho_ci95_lo[0]:.3f}, " \
           f"{single.rho_ci95_hi[0]:.3f}] unchanged"


def check_middle_third() -> str:
    rng = np.random.default_rng(2)
    rho = 0.6 + rng.normal(0, 0.02, 12)
    rho[4:8] -= 0.08
    summary = m4.summarise_rsa_sweep(sweep_table(rho), {}).iloc[0]
    middle = np.isin(np.arange(12), range(4, 8))
    observed = rho[middle].mean() - rho[~middle].mean()
    null = []
    for chosen in itertools.combinations(range(12), 4):
        flags = np.isin(np.arange(12), chosen)
        null.append(rho[flags].mean() - rho[~flags].mean())
    exact = np.mean(np.abs(null) >= abs(observed) - 1e-12)
    assert abs(summary.middle_third_minus_outer_rho - observed) < 1e-12, f"contrast {summary}"
    assert abs(summary.middle_vs_outer_perm_p - exact) < 1e-12, \
        f"12 blocks: p {summary.middle_vs_outer_perm_p} against exact two-sided {exact}"

    rho = 0.6 + rng.normal(0, 0.02, 28)
    rho[10:19] -= 0.03
    low = m4.summarise_rsa_sweep(sweep_table(rho), {}).iloc[0].middle_vs_outer_perm_p
    high = m4.summarise_rsa_sweep(sweep_table(1.2 - rho), {}).iloc[0].middle_vs_outer_perm_p
    assert 1 / (PERMUTATIONS + 1) <= low <= 1 and low == high, f"28 blocks: p {low}, mirrored {high}"
    return f"middle third: exact p {exact:.4f} over 495 assignments, 28 blocks p {low} either way"


def checks(failures: list) -> str:
    return ", ".join([check_mantel(), check_independence(), check_middle_third()])


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
