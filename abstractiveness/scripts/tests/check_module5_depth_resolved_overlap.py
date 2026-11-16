"""
Module 5's Study D resolves the Jaccard index block by block and locates the units shared by words that open with the
same token.

Synthetic frame of 12 blocks, three categories of four members, their three label words and a homonym word in two
categories. Within every category the units two members share in block 4 order the ratings exactly. tuxedo
(clothing) and trolley (vehicles) open with the same token and share six units, all in block 11, and one other pair
across categories shares one unit in block 7.

Checks:
  1. depth_resolved_curve has one row per block, quarters counted from 1, a within-category rho of 1 in block 4 and
     no text partial without text_similarity.
  2. shared_unit_depth finds the one same-token pair at the top percentile, with every shared unit in the last
     quarter and in block 11, leaves the homonyms out, and gives integer counts whose shares sum to 1 per group.
  3. Without block 11 the same-token pair shares no unit and its depth is NaN, and without tokens all three frames
     are empty.
  4. plot_overlap_by_depth at a half-page print width saves a figure no wider than that width, legends included.
  5. Two members of different categories without experts leave their pair's Jaccard index undefined, so the
     summary counts it among n_pairs and not among n_pairs_with_experts, the pairs its mean reads.
  6. plot_overlap_by_depth names what each line measures: one block's units against all units, the text partialled
     out, rho_W on the top axis, the share of each group's shared units per block on the bottom axis, the members'
     own units as the reference, and the quarter rules in the x label, in print and on screen, the two y labels
     clear of each other.
  7. random_unit_set draws units until the set holds the target count of expert rows and stops at the first unit
     that reaches it. size_matched_null gives one row per block, 0 pairs without a shared unit in block 4 and all of
     them elsewhere, a 95% range around the mean of the random sets, block 4's observed rho of 1 above that range,
     the same values for the same seed, and a text-partialled null only when the pair table has text_similarity.
  8. With the null merged into the curve, plot_overlap_by_depth adds the band of random sets of the block's size to
     the top legend, in print and on screen.

Usage, from scripts/:
    python tests/check_module5_depth_resolved_overlap.py
"""
import itertools
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.module_5_typicality_prediction import (TARGET, depth_resolved_curve, plot_overlap_by_depth,
                                                    random_unit_set, shared_unit_depth, size_matched_null)
from utils.plotting import print_style
from _figures import closed_figures, legend_texts
from _report import run

N_BLOCKS = 12
MEMBERS = {"clothing": ["tuxedo", "scarf", "jacket", "boot"],
           "vehicles": ["trolley", "bus", "van", "ship"],
           "fruit": ["apple", "mango", "kiwi", "lemon"]}
HOMONYMS = {"squash__clothing": "clothing", "squash__vehicles": "vehicles"}


def expert_frame(units: dict) -> pd.DataFrame:
    """Expert rows of GPT-2's mlp.c_fc alone, units given per concept as (block, unit) pairs."""
    names = [f"{b + 1}.L.{b}.mlp.c_fc" for b in range(N_BLOCKS)]
    rows = [(concept, block + 1, unit, names[block]) for concept, pairs in units.items() for block, unit in pairs]
    frame = pd.DataFrame(rows, columns=["concept", "layer_idx", "unit", "layer_name"])
    frame["layer_name"] = pd.Categorical(frame["layer_name"], categories=names, ordered=True)
    return frame


def fixture():
    """The expert frame, members, pair table, Study A design and token frame of the synthetic model."""
    rng = np.random.default_rng(0)
    concepts = [c for words in MEMBERS.values() for c in words] + list(MEMBERS) + list(HOMONYMS)
    units = {c: [(int(b), 1000 * i + k) for k, b in enumerate(rng.integers(0, N_BLOCKS, 12))]
             for i, c in enumerate(concepts)}
    next_unit, ratings = 100_000, []
    for category, words in MEMBERS.items():
        for rank, (a, b) in enumerate(itertools.combinations(words, 2)):
            shared = [(4, next_unit + k) for k in range(rank + 1)]
            next_unit += rank + 1
            units[a] += shared
            units[b] += shared
            ratings.append((category, a, b))
    units["tuxedo"] += [(11, 200_000 + k) for k in range(6)]
    units["trolley"] += [(11, 200_000 + k) for k in range(6)]
    units["scarf"] += [(7, 300_000)]
    units["mango"] += [(7, 300_000)]
    units["squash__clothing"] += [(9, 400_000 + k) for k in range(20)]
    units["squash__vehicles"] += [(9, 400_000 + k) for k in range(20)]
    df = expert_frame(units)

    def block_four_jaccard(a, b):
        first, second = {u for blk, u in units[a] if blk == 4}, {u for blk, u in units[b] if blk == 4}
        return len(first & second) / len(first | second)

    pair_table = pd.DataFrame(ratings, columns=["category", "word_a", "word_b"])
    pair_table["human_similarity"] = [block_four_jaccard(a, b) for _, a, b in ratings]
    assert pair_table.groupby("category").human_similarity.nunique().eq(6).all(), "fixture ratings tie"
    members = pd.DataFrame([(c, c, category) for category, words in MEMBERS.items() for c in words]
                           + [(c, "squash", category) for c, category in HOMONYMS.items()],
                           columns=["concept", "word", "category"])
    design = pd.DataFrame({"concept": members.concept[:12], "category": members.category[:12],
                           TARGET: rng.normal(size=12)})
    first_ids = {c: i for i, c in enumerate(members.concept)}
    first_ids["trolley"] = first_ids["tuxedo"]
    first_ids["squash__vehicles"] = first_ids["squash__clothing"]
    tokens = pd.DataFrame({"n_tokens": 2, "first_token_id": [first_ids[c] for c in members.concept],
                           "first_piece": [" t" if c in ("tuxedo", "trolley") else f" {c[:2]}"
                                           for c in members.concept]},
                          index=pd.Index(members.concept, name="concept"))
    return df, members, pair_table, design, tokens


def checks(failures: list) -> str:
    df, members, pair_table, design, tokens = fixture()
    curve = depth_resolved_curve(df, pair_table, design)
    assert abs(curve.loc[curve.block == 4, "pair_rho"].item() - 1.0) < 1e-12 and len(curve) == 12, curve
    assert curve.loc[curve.block == 4, "quarter"].item() == 2 and np.isnan(curve.pair_rho_given_text).all()
    assert curve.quarter.tolist() == [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3, curve.quarter.tolist()
    assert abs(curve.shared_share.sum() - 1.0) < 1e-12 and abs(curve.union_share.sum() - 1.0) < 1e-12

    by_block, same, summary = shared_unit_depth(df, members, tokens)
    assert len(same) == 1 and same.shared_last_quarter_share.item() == 1.0 and same.percentile.item() == 1.0, same
    assert same.shared_depth.item() == 1.0 and same.first_piece.item() == " t", same
    assert by_block.loc[by_block.block == 11, "same_token_share"].item() == 1.0
    assert not same.concept_a.str.startswith("squash").any()
    for group in ("within_category", "same_token", "other_cross", "all_units"):
        assert abs(by_block[f"{group}_share"].sum() - 1.0) < 1e-12 and by_block[f"{group}_units"].dtype.kind == "i"
    groups = summary.set_index("group")
    assert set(groups.index) == {"same_token", "first_letter_only", "other_cross", "within_category"}, groups
    assert groups.loc["same_token", "ratio_to_other_cross"] > 10, groups

    _, _, apart = shared_unit_depth(df[df.layer_idx != 12], members, tokens)
    assert np.isnan(apart.set_index("group").loc["same_token", "shared_depth"])
    assert all(frame.empty for frame in shared_unit_depth(df, members, tokens.iloc[0:0]))
    empty = shared_unit_depth(df[~df.concept.isin(["boot", "ship"])], members, tokens)[2].set_index("group")
    assert (empty.n_pairs - empty.n_pairs_with_experts).to_dict() == {
        "same_token": 0, "first_letter_only": 0, "other_cross": 1, "within_category": 0}, empty

    with tempfile.TemporaryDirectory() as tmp:
        fig = closed_figures(plot_overlap_by_depth, curve, by_block, (0.34, 0.14), pathlib.Path(tmp) / "d.png",
                             print_width=3.25)[1][0]
        with print_style():
            width = fig.get_tightbbox(fig.canvas.get_renderer()).width
        assert width <= 3.25 + 0.05, f"print figure {width:.2f} in wide"

        partial = curve.assign(pair_rho_given_text=curve.pair_rho / 2)
        top_entries = ["This block's units", "All units (whole model)",
                       "This block's units, text partialled out", "All units, text partialled out"]
        bottom_entries = ["Pairs within a category", "Pairs across categories, same first token",
                          "Other pairs across categories", "All expert units of the members (reference)"]
        for print_width in (None, 3.25):
            fig = closed_figures(plot_overlap_by_depth, partial, by_block, (0.34, 0.14), pathlib.Path(tmp) / "e.png",
                                 print_width=print_width)[1][0]
            top, bottom = fig.axes
            assert legend_texts(top) == top_entries, legend_texts(top)
            assert legend_texts(bottom) == bottom_entries, legend_texts(bottom)
            assert r"$\rho_W$" in top.get_ylabel() and "ratings" in top.get_ylabel(), top.get_ylabel()
            assert "shared units" in bottom.get_ylabel() and "%" in bottom.get_ylabel(), bottom.get_ylabel()
            assert "quarters" in bottom.get_xlabel(), bottom.get_xlabel()
            with print_style(print_width is not None):
                renderer = fig.canvas.get_renderer()
                labels = [axis.yaxis.label.get_window_extent(renderer) for axis in (top, bottom)]
            assert not labels[0].overlaps(labels[1]), f"y labels collide at print width {print_width}"
            if print_width:
                with print_style():
                    width = fig.get_tightbbox(fig.canvas.get_renderer()).width
                assert width <= 3.25 + 0.05, f"print figure with every label {width:.2f} in wide"
    column_rows = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    for seed in range(20):
        chosen = random_unit_set(column_rows, 10, np.random.default_rng(seed))
        assert column_rows[chosen].sum() >= 10 > column_rows[chosen[:-1]].sum(), (seed, chosen)
        assert len(set(chosen)) == len(chosen)
    null = size_matched_null(df, pair_table, n_draws=50, seed=0)
    assert null.block.tolist() == list(range(N_BLOCKS)), null
    assert set(null.columns) == {"block", "zero_share", "random_pair_rho_mean", "random_pair_rho_low",
                                 "random_pair_rho_high", "random_pair_rho_given_text_mean",
                                 "random_pair_rho_given_text_low", "random_pair_rho_given_text_high",
                                 "random_zero_share"}, null.columns
    zero = null.set_index("block").zero_share
    assert zero[4] == 0.0 and (zero.drop(4) == 1.0).all(), zero
    four = null.set_index("block").loc[4]
    assert four.random_pair_rho_low <= four.random_pair_rho_mean <= four.random_pair_rho_high < 1.0, four
    finite = null.dropna(subset=["random_pair_rho_mean"])
    assert (finite.random_pair_rho_low <= finite.random_pair_rho_high).all() and len(finite) >= 1
    assert null.random_pair_rho_given_text_mean.isna().all()
    pd.testing.assert_frame_equal(null, size_matched_null(df, pair_table, n_draws=50, seed=0))
    texted = pair_table.assign(text_similarity=np.random.default_rng(1).random(len(pair_table)))
    with_text = size_matched_null(df, texted, n_draws=20, seed=0)
    assert np.isfinite(with_text.set_index("block").loc[4, "random_pair_rho_given_text_mean"]), with_text

    band = "Random units of the block's size, 95% range"
    merged = curve.merge(null, on="block")
    with tempfile.TemporaryDirectory() as tmp:
        for print_width in (None, 3.25):
            fig = closed_figures(plot_overlap_by_depth, merged, by_block, (0.34, 0.14), pathlib.Path(tmp) / "f.png",
                                 print_width=print_width)[1][0]
            assert band in legend_texts(fig.axes[0]), legend_texts(fig.axes[0])
            if print_width:
                with print_style():
                    width = fig.get_tightbbox(fig.canvas.get_renderer()).width
                assert width <= 3.25 + 0.05, f"print figure with the band {width:.2f} in wide"

    return (f"12 blocks, block-4 rho 1, same-token pair at percentile 1 with every shared unit in block 11, "
            f"block-4 random sets {four.random_pair_rho_mean:.2f} [{four.random_pair_rho_low:.2f}, "
            f"{four.random_pair_rho_high:.2f}], "
            f"ratio {groups.loc['same_token', 'ratio_to_other_cross']:.0f} to the other pairs across categories")


if __name__ == "__main__":
    sys.exit(run(__file__, checks))
