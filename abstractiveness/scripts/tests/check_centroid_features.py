"""Verifies the member-centroid features are leave-one-out, gated, and correctly aligned.

The hard part of add_centroid_features is bookkeeping. Each design row's own-category
agreement is one entry of a flat concept-by-category population, and its z is read back out
of that population by index, so a misalignment of one row would hand every concept another
concept's z while the table still looks entirely healthy. This check rebuilds the whole
population independently, keyed on (concept, category) instead of on row position, and
requires the module's columns to match it entry for entry.

Checks. Leave-one-out genuinely holds, proven two ways, once by matching an independently
computed leave-one-out value and once by requiring the module's value to DIFFER from the
include-self value. mean_jaccard_to_members is NaN for a concept with no siblings. Both
profile columns are NaN for a concept below MIN_PROFILE_EXPERTS. Every per-metric column
exists for every entry of ACTIVE_METRICS. The deleted foreign-centroid margin column is gone.

Run from scripts/: python tests/check_centroid_features.py.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modules.module_9_typicality_prediction import (
    add_centroid_features, profile_columns, ACTIVE_METRICS)
from utils.helpers import (build_layer_probability_matrix, to_block_axis, count_matched_z,
                           MIN_PROFILE_EXPERTS)

BLOCKS = 12
# The deep assertions below recompute this one metric from its definition, so they are
# written against its name rather than looped over the registry. If the module's default
# moves off it the fixture stops describing anything the module runs, and every per-metric
# lookup would raise a bare KeyError several hundred lines later, so the dependency is
# stated here instead.
METRIC = "js_distance"
assert METRIC in ACTIVE_METRICS, (
    f"check_centroid_features is written against METRIC={METRIC!r}, which is no longer in "
    f"ACTIVE_METRICS={ACTIVE_METRICS}. Point METRIC at an active metric, or generalise the "
    "recomputed reference implementation below to the metric registry.")


def _js_similarity(profile: np.ndarray, reference: np.ndarray) -> float:
    """100 * (1 - sqrt(JSD in bits)), written out here rather than imported.

    The module reaches this quantity through the PROFILE_METRICS registry, whose
    implementation accumulates a blocked cross mixture with scipy's xlogy. Recomputing it
    from the definition keeps the check from confirming the module against the same code
    path it is testing.
    """
    mixture = 0.5 * (profile + reference)

    def entropy(vector: np.ndarray) -> float:
        positive = vector[vector > 0]
        return float(-(positive * np.log2(positive)).sum())

    divergence = entropy(mixture) - 0.5 * (entropy(profile) + entropy(reference))
    return 100.0 * (1.0 - np.sqrt(max(divergence, 0.0)))


def _expert_rows(rng, concept: str, draws: int, center: float, unit_low: int) -> list:
    """Expert rows for one concept, drawn from a Gaussian bump over blocks so members of a
    category share depth, and from a category-local unit pool so they share experts. Repeated
    (block, unit) pairs are dropped, since the presence matrix counts a pair once."""
    weights = np.exp(-0.5 * ((np.arange(BLOCKS) - center) / 2.0) ** 2) + 0.02
    blocks = rng.choice(BLOCKS, size=draws, p=weights / weights.sum())
    units = rng.integers(unit_low, unit_low + 1500, size=draws)

    seen, rows = set(), []
    for block, unit in zip(blocks, units):
        if (int(block), int(unit)) in seen:
            continue
        seen.add((int(block), int(unit)))
        rows.append({"concept": concept, "layer_idx": int(block) + 1, "unit": int(unit),
                     "layer_name": f"{int(block) + 1}.L.{int(block)}.mlp.c_fc"})
    return rows


def _fixture() -> tuple:
    """Six categories of twelve concepts, plus three edge rows the bookkeeping must survive.

    The population has to be large enough that count_matched_z returns real numbers rather
    than the all-NaN it gives on a handful of entries, or every NaN assertion below would
    pass for the wrong reason. Six categories of twelve give about 500 concept-by-category
    agreements against a floor of eighty.

    The three edge rows come FIRST, since an off-by-one in the row bookkeeping is most likely
    to appear at the boundary between rows that do and do not produce an own-category entry.
    solo_0 is the only member of its category, so leaving it out empties its own reference.
    ghost holds no experts at all. tiny holds exactly one, below MIN_PROFILE_EXPERTS.

    cat0_00 is given a very large expert set and cat5 a set of very small ones, so some
    entries place the concept ABOVE the centroid in expert count. A symmetrised coordinate
    would reorder exactly those, which is what makes the fixed-order coordinate testable.
    """
    rng = np.random.default_rng(7)
    rows, design = [], []

    rows += _expert_rows(rng, "solo_0", 30, 6.0, 9000)
    design.append(("solo_0", "solo"))
    design.append(("ghost", "cat0"))
    rows += _expert_rows(rng, "tiny", 1, 3.0, 0)
    design.append(("tiny", "cat0"))

    for category in range(6):
        for member in range(12):
            concept = f"cat{category}_{member:02d}"
            if concept == "cat0_00":
                draws = 400
            elif category == 5:
                draws = 3
            else:
                draws = int(rng.integers(8, 60))
            rows += _expert_rows(rng, concept, draws, category + 1.5 + rng.normal(0, 0.4),
                                 category * 300)
            design.append((concept, f"cat{category}"))

    experts = pd.DataFrame(rows)
    # The production frame carries layer_name as an ordered Categorical (helpers builds it
    # that way so layers sort in model order), and build_layer_probability_matrix relies on
    # the .cat accessor, so the fixture has to match that dtype rather than plain object.
    ordered = sorted(experts["layer_name"].unique(), key=lambda name: int(name.split(".")[0]))
    experts["layer_name"] = pd.Categorical(experts["layer_name"], categories=ordered, ordered=True)

    frame = pd.DataFrame(design, columns=["concept", "category"])
    frame["human_typicality"] = np.linspace(0.9, 0.1, len(frame))
    return frame, experts


def _population(design: pd.DataFrame, experts: pd.DataFrame, symmetrised: bool = False) -> tuple:
    """The concept-by-category agreement population, rebuilt from the specification.

    Returns (values, coords, own_of), where own_of maps (concept, category) to the flat index
    of that row's OWN-category entry and simply has no key for a row that produces none. That
    dictionary is a different bookkeeping device from the module's positional list of indices
    and Nones, which is the point: the two have to agree row for row.

    symmetrised=True places each entry at (log min, log max) instead of the fixed
    (log concept, log centroid) order, purely so the check can prove the module does NOT do
    that.
    """
    counts, profiles = build_layer_probability_matrix(to_block_axis(experts))
    count_of = counts.sum(axis=1)
    usable = {concept for concept in design["concept"]
              if concept in profiles.index and float(count_of.loc[concept]) >= MIN_PROFILE_EXPERTS}

    members = {}
    for concept, category in zip(design["concept"], design["category"]):
        if concept in usable:
            members.setdefault(category, []).append(concept)

    values, coords, own_of = [], [], {}
    for concept, category in zip(design["concept"], design["category"]):
        if concept not in usable:
            continue
        profile = profiles.loc[concept].to_numpy(dtype=float)
        concept_count = float(count_of.loc[concept])
        for other in sorted(members):
            names = ([name for name in members[other] if name != concept]
                     if other == category else members[other])
            if not names:
                continue
            pooled = float(count_of.loc[names].sum())
            if other == category:
                own_of[(concept, category)] = len(values)
            values.append(_js_similarity(profile,
                                         profiles.loc[names].to_numpy(dtype=float).mean(axis=0)))
            first, second = np.log(concept_count), np.log(pooled)
            coords.append((min(first, second), max(first, second)) if symmetrised
                          else (first, second))
    return np.asarray(values), np.asarray(coords), own_of


def _with_self(design: pd.DataFrame, experts: pd.DataFrame) -> dict:
    """The own-category agreement a NON leave-one-out implementation would report, keyed by
    concept. Averaging the concept into the reference it is scored against is the exact
    mistake this feature exists to avoid, so the module's value has to differ from it."""
    counts, profiles = build_layer_probability_matrix(to_block_axis(experts))
    count_of = counts.sum(axis=1)
    usable = {concept for concept in design["concept"]
              if concept in profiles.index and float(count_of.loc[concept]) >= MIN_PROFILE_EXPERTS}

    members = {}
    for concept, category in zip(design["concept"], design["category"]):
        if concept in usable:
            members.setdefault(category, []).append(concept)

    return {concept: _js_similarity(profiles.loc[concept].to_numpy(dtype=float),
                                    profiles.loc[members[category]].to_numpy(dtype=float).mean(axis=0))
            for concept, category in zip(design["concept"], design["category"])
            if concept in usable}


def main() -> int:
    design, experts = _fixture()
    out = add_centroid_features(design, experts, ACTIVE_METRICS)
    cols = profile_columns(METRIC)

    assert len(out) == len(design), f"{len(out)} rows out of {len(design)} in"
    assert list(out["concept"]) == list(design["concept"]), "rows were reordered"

    # Every per-metric column, for every registered metric, and nothing left of the margin.
    for metric in ACTIVE_METRICS:
        for role in ("S_cen", "z_cen"):
            assert profile_columns(metric)[role] in out.columns, \
                f"{profile_columns(metric)[role]} missing for metric {metric}"
    assert "mean_jaccard_to_members" in out.columns, "mean_jaccard_to_members missing"
    assert "centroid_margin" not in out.columns, "centroid_margin must not survive"

    values, coords, own_of = _population(design, experts)
    expected_z = count_matched_z(values, coords)
    assert np.isfinite(expected_z).sum() >= 60, \
        "the rebuilt population yields almost no finite z, fix the fixture"

    similarity = out[cols["S_cen"]].to_numpy(dtype=float)
    z_score = out[cols["z_cen"]].to_numpy(dtype=float)

    # The alignment check. Each row's columns must be the entry the (concept, category) key
    # points at, and rows with no own-category entry must be NaN in both columns.
    for row, (concept, category) in enumerate(zip(out["concept"], out["category"])):
        if (concept, category) in own_of:
            index = own_of[(concept, category)]
            assert np.isclose(similarity[row], values[index], atol=1e-10), \
                f"{concept}: similarity {similarity[row]} != rebuilt {values[index]}"
            assert np.allclose(z_score[row], expected_z[index], atol=1e-10, equal_nan=True), \
                f"{concept}: z {z_score[row]} != rebuilt {expected_z[index]}"
        else:
            assert np.isnan(similarity[row]), f"{concept} has no own centroid but a similarity"
            assert np.isnan(z_score[row]), f"{concept} has no own centroid but a z"

    # A finite z can never sit on a NaN similarity, which is the failure a shifted index
    # produces first.
    assert not (np.isfinite(z_score) & ~np.isfinite(similarity)).any(), \
        "a row has a z without a similarity, the own-entry indices are misaligned"
    assert np.isfinite(z_score).sum() >= 60, \
        "almost no row carries a finite z, the NaN assertions below would be vacuous"

    # Leave-one-out, the second way: the module's value must NOT be the include-self value.
    include_self = _with_self(design, experts)
    gaps = [abs(similarity[row] - include_self[concept])
            for row, concept in enumerate(out["concept"])
            if concept in include_self and np.isfinite(similarity[row])]
    assert gaps, "no row carries both a similarity and an include-self counterpart"
    assert min(gaps) > 1e-6, \
        (f"the smallest gap to the include-self value is {min(gaps):.3g}, the reference is "
         f"not leave-one-out")

    indexed = out.set_index("concept")

    # A concept with no siblings has no member reference at all.
    assert np.isnan(indexed.loc["solo_0", "mean_jaccard_to_members"]), \
        "solo_0 is the only member of its category and must have no mean Jaccard"
    assert np.isnan(indexed.loc["solo_0", cols["S_cen"]]), "solo_0 must have no centroid"
    assert np.isnan(indexed.loc["ghost", "mean_jaccard_to_members"]), \
        "ghost holds no experts and must have no mean Jaccard"

    # Below MIN_PROFILE_EXPERTS there is no depth distribution to score.
    assert np.isnan(indexed.loc["tiny", cols["z_cen"]]), \
        f"tiny holds fewer than {MIN_PROFILE_EXPERTS} experts and must have no z"
    assert np.isnan(indexed.loc["tiny", cols["S_cen"]]), \
        f"tiny holds fewer than {MIN_PROFILE_EXPERTS} experts and must have no similarity"
    assert np.isfinite(indexed.loc["tiny", "mean_jaccard_to_members"]), \
        "the profile gate must not remove tiny's Jaccard, which needs no distribution"

    # Members of a real category are scored on both features and land on the 0 to 100 scale.
    ordinary = out[out["concept"].str.startswith("cat")]
    scored = ordinary[ordinary["concept"] != "tiny"]
    assert scored[cols["S_cen"]].between(0, 100).all(), "similarity off scale"
    assert scored["mean_jaccard_to_members"].between(0, 100).all(), "mean Jaccard off scale"

    # The coordinate is a FIXED order, not a symmetrised one. The fixture puts some concepts
    # above their centroid in expert count, so min/max would genuinely reorder those entries,
    # and the resulting z would not be the module's.
    reordered = int((coords[:, 0] > coords[:, 1]).sum())
    assert reordered > 0, \
        "no entry places the concept above the centroid in count, fix the fixture"
    _, swapped_coords, _ = _population(design, experts, symmetrised=True)
    swapped_z = count_matched_z(values, swapped_coords)
    assert not np.allclose(expected_z, swapped_z, atol=1e-8, equal_nan=True), \
        "a symmetrised coordinate gives the same z here, fix the fixture"
    matches_swapped = [np.allclose(z_score[row], swapped_z[own_of[(concept, category)]],
                                   atol=1e-10, equal_nan=True)
                       for row, (concept, category) in enumerate(zip(out["concept"], out["category"]))
                       if (concept, category) in own_of]
    assert not all(matches_swapped), "the z matches a symmetrised coordinate throughout"

    print(f"{len(out)} rows, {int(np.isfinite(z_score).sum())} with a finite z, "
          f"{reordered} entries where the concept outweighs its centroid")
    print("OK check_centroid_features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
