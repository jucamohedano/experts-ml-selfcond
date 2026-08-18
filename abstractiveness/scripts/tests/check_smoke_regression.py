"""Regression invariants of the module 9 restructure, measured on a real smoke run.

The restructure moved module 9 onto the BLOCK layer axis and rebuilt its studies. This
script proves the refactor changed structure and not numbers where it must not have, by
comparing a freshly generated results tree against the pre-refactor baseline tree.

Four invariants.

1. Sublayer scopes of modules 3, 4 and 5 reproduce byte for byte. A sublayer scope holds
   exactly one layer per transformer block, so to_block_axis is an identity relabel there
   and every number must survive it untouched. This is the ONLY evidence that the block
   axis preserved the count-matched z on sublayer scopes, because the unit check of the
   same property carries a fixture too small for the null to produce a finite z, so its z
   assertion compares NaN against NaN and proves nothing. Do not narrow this to a subset
   of files or of columns.
2. Whole-model jaccard_pct is unchanged. Jaccard reads neuron identity as an unordered
   set, so it is axis free and must match the baseline exactly. It is also the ONLY
   whole-model column that may be compared against the baseline, see the note below.
3. Row sets are unchanged, module 3's row count and module 9's design-frame concept set.
4. The restructured study outputs exist and are shaped correctly, ranker_comparison.csv
   carrying six grid cells times two references per metric, and
   pair_similarity_comparison.csv carrying six grid cells per metric. Also, cross-tree,
   the smoke jaccard cell of each study matches the baseline's own jaccard row by name,
   ranker_comparison.csv's jaccard/label_word cell against the baseline's
   typicality_ranker_comparison.csv jaccard row on n_pairs and accuracy, and
   pair_similarity_comparison.csv's jaccard cell against the baseline's
   pair_similarity_comparison.csv jaccard row on mean_category_spearman. Both baseline
   files live directly under 9_typicality_prediction/, this baseline predates the
   9.2_pairwise_ranking/9.4_human_pair_similarity layout some briefs assume.

What must NOT be compared. Every PROFILE column of a WHOLE-MODEL scope legitimately
differs from the baseline, because the baseline was written on the flat layer axis before
the block-axis change and the smoke run is on the block axis. On GPT-2 at AP 0.5 the
difference reaches about 23 points on the similarity scale. That is the intended effect of
the refactor, not a defect, so the whole-model comparison here is restricted to jaccard_pct
and to row sets.

Usage, from scripts/:
    python tests/check_smoke_regression.py [smoke_root] [baseline_root] [ap_folder]
The defaults point at the GPT-2 Richie-HSJ smoke tree and its _sensefix baseline. Passing
the two Qwen3 roots runs the same checks on that architecture without editing this file.
"""
import filecmp
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils.helpers import ACTIVE_PROFILE_METRICS, DEFAULT_PROFILE_METRIC

DEFAULT_SMOKE = "../results/research_plots_gpt2_richie_hsj_smoke_m9"
DEFAULT_BASELINE = "../results/research_plots_gpt2_richie_hsj_sensefix"
DEFAULT_AP = "AP_0.5"

# Relative tolerance for a value that is allowed to differ only by floating point noise.
# One ULP of a double is about 2.2e-16, so 1e-13 has headroom well past a handful of ULPs.
# It also covers the parse noise of pandas' default float reader, which is not round-trip
# exact, so this check is blind to differences smaller than about 1 ULP of parse noise on
# top of the library-version ULP itself. Harmless at this tolerance, worth naming.
FLOAT_TOLERANCE = 1e-13

# The two declared reasons a baseline sublayer CSV of modules 3, 4 or 5 may fail the
# byte comparison. Both are properties of how the smoke run is CONFIGURED or of the
# baseline's age, neither is a property of the module 9 restructure, and each still has to
# clear the numeric equality below. Anything not on these lists is a hard failure.
#
# UPSTREAM_DEPENDENT_FILES are module 4 outputs whose inputs come from modules 2 and 7,
# which the smoke run leaves disabled, so the file is simply not produced.
UPSTREAM_DEPENDENT_FILES = {"shannon_entropy_vs_expert_count.csv",
                            "jaccard_vs_cosine_typicality.csv"}
# UPSTREAM_DEPENDENT_ROWS are the rows those same two modules contribute to module 4's
# correlation_summary.csv, which always carries one row per panel.
UPSTREAM_DEPENDENT_ROWS = {"shannon_entropy_vs_expert_count", "jaccard_vs_cosine_typicality"}
# SCHEMA_ADDED_COLUMNS are columns a restructured module now writes that the baseline
# predates. Their presence is intended, every column the baseline DID write must still
# match value for value.
#
# The module 3 entry is the multi-metric extension: the module now writes one agreement and
# one z column per entry of ACTIVE_PROFILE_METRICS, generated from the registry rather than
# written out, so the allowance is generated the same way. The DEFAULT metric's original
# column names are deliberately NOT in this set, because they are still written and must
# still match the baseline exactly, which is the whole evidence that the extension left the
# existing metric alone.
SCHEMA_ADDED_COLUMNS = {
    "human_similarity_validation.csv": {"coefficient", "test"},
    "category_concept_similarity_metrics.csv": {
        column for metric in ACTIVE_PROFILE_METRICS
        for column in (f"layer_profile_{metric}", f"layer_profile_{metric}_z")},
}

# RENAMED_FILES and RENAMED_ROWS cover the same extension, which named every layer-profile
# artifact after the metric behind it. The baseline predates that, so its unqualified name
# IS the default metric's file or row and is compared against it. A rename must not change a
# value, so these pairs still have to clear the byte or numeric comparison below.
RENAMED_FILES = {
    "layer_profile_matrix.csv": f"layer_profile_{DEFAULT_PROFILE_METRIC}_matrix.csv",
    "layer_profile_z_matrix.csv": f"layer_profile_{DEFAULT_PROFILE_METRIC}_z_matrix.csv",
}
RENAMED_ROWS = {
    "human_similarity_validation.csv": {
        "metric": {"layer_profile": f"layer_profile_{DEFAULT_PROFILE_METRIC}",
                   "layer_profile_z": f"layer_profile_{DEFAULT_PROFILE_METRIC}_z"}},
}

# SCHEMA_ADDED_ROWS are rows the baseline could not carry because the metric did not exist
# when it was written. Subchapter 5.3 now validates every registered metric, so a scope
# gains two rows per category per non-default metric. Generated from the registry, so
# registering a metric extends the allowance with it.
SCHEMA_ADDED_ROWS = {
    "human_similarity_validation.csv": {
        f"layer_profile_{metric}{suffix}"
        for metric in ACTIVE_PROFILE_METRICS for suffix in ("", "_z")},
}

# Text columns allowed to differ, because they carry a DISPLAY label rather than a value and
# the label now names the metric ("layer profile" became "layer profile, Jensen-Shannon
# distance"). The numeric columns of the same row are still compared exactly.
RENAMED_TEXT_COLUMNS = {"human_similarity_validation.csv": {"metric_label"}}

# The rows of module 4's correlation_summary.csv that read the layer profile and its
# count-matched z. On a sublayer scope these are exactly the numbers the block-axis change
# passes through, so they are asserted individually rather than only inside the file sweep.
PROFILE_SUMMARY_ROWS = ("jaccard_vs_layer_profile_allpairs",
                        "jaccard_vs_layer_profile_z_allpairs")

# Key columns used to align two versions of one CSV, tried in order.
KEY_CANDIDATES = (["plot_name"], ["metric", "category"], ["concept"])


def read_first(root: pathlib.Path, pattern: str):
    """First CSV matching ``pattern`` under ``root``, or None if no file matches.

    A glob-then-next idiom is used all over this script to reach into a results tree
    whose exact folder name (which sublayer rank, which module numbering) is not known
    ahead of time. ``next(root.glob(pattern))`` on an empty match raises an uncaught
    StopIteration, a raw traceback instead of a reported failure line, so every call site
    below goes through this helper and checks for None instead.
    """
    match = next(root.glob(pattern), None)
    return pd.read_csv(match) if match is not None else None


def numeric_frames_match(base: pd.DataFrame, smoke: pd.DataFrame, name: str) -> list:
    """Value equality of two versions of one CSV, aligned on a key, within FLOAT_TOLERANCE.

    Returns a list of problem strings, empty when the two agree. Columns the baseline never
    wrote are allowed when declared in SCHEMA_ADDED_COLUMNS, baseline rows missing from the
    smoke run are allowed when declared in UPSTREAM_DEPENDENT_ROWS.
    """
    problems = []
    allowed_new = SCHEMA_ADDED_COLUMNS.get(name, set())
    added = set(smoke.columns) - set(base.columns)
    if added - allowed_new:
        problems.append(f"{name}: undeclared new columns {sorted(added - allowed_new)}")
    dropped = set(base.columns) - set(smoke.columns)
    if dropped:
        problems.append(f"{name}: columns dropped {sorted(dropped)}")

    key = next((k for k in KEY_CANDIDATES if set(k) <= set(base.columns)), None)
    if key is None:
        problems.append(f"{name}: no key column to align on")
        return problems

    # Relabel the baseline's key values onto the names the smoke run uses, so a row that was
    # only RENAMED is still aligned and its values still compared. Applied to the baseline
    # rather than the smoke run because the rename is the direction the code moved.
    base = base.copy()
    for column, mapping in RENAMED_ROWS.get(name, {}).items():
        if column in base.columns:
            base[column] = base[column].replace(mapping)

    base_indexed = base.set_index(key)
    smoke_indexed = smoke.set_index(key)
    lost = [k for k in base_indexed.index if k not in set(smoke_indexed.index)]
    unexplained = [k for k in lost
                   if (k if isinstance(k, str) else k[0]) not in UPSTREAM_DEPENDENT_ROWS]
    if unexplained:
        problems.append(f"{name}: rows lost {unexplained}")
    allowed_new_rows = SCHEMA_ADDED_ROWS.get(name, set())
    gained = [k for k in smoke_indexed.index if k not in set(base_indexed.index)]
    unexplained_gained = [k for k in gained
                          if (k if isinstance(k, str) else k[0]) not in allowed_new_rows]
    if unexplained_gained:
        problems.append(f"{name}: rows added {unexplained_gained}")

    relabelled = RENAMED_TEXT_COLUMNS.get(name, set())
    shared_rows = [k for k in base_indexed.index if k in set(smoke_indexed.index)]
    for column in [c for c in base_indexed.columns if c in smoke_indexed.columns]:
        left = base_indexed.loc[shared_rows, column]
        right = smoke_indexed.loc[shared_rows, column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            close = np.isclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                               rtol=FLOAT_TOLERANCE, atol=0.0, equal_nan=True)
            if not close.all():
                worst = np.nanmax(np.abs(left.to_numpy(dtype=float)
                                         - right.to_numpy(dtype=float)))
                problems.append(f"{name}: column {column} differs, max abs diff {worst}")
        elif column in relabelled:
            continue
        elif not left.fillna("").astype(str).equals(right.fillna("").astype(str)):
            problems.append(f"{name}: text column {column} differs")
    return problems


def main() -> int:
    smoke_root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SMOKE)
    baseline_root = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BASELINE)
    ap_folder = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_AP
    B, S = baseline_root / ap_folder, smoke_root / ap_folder
    failures, notes = [], []

    for root in (B, S):
        if not root.is_dir():
            print(f"FAIL check_smoke_regression\n  missing results root {root}")
            return 1

    # --- invariant 1, byte for byte sublayer CSVs of modules 3, 4 and 5 ------------------
    identical = 0
    checked = 0
    for module_glob in ("3_*", "4_*", "5_*"):
        for baseline_csv in sorted(B.glob(f"{module_glob}/sublayers/*/*.csv")):
            checked += 1
            relative = baseline_csv.relative_to(B)
            smoke_csv = S / relative
            if not smoke_csv.exists():
                # A file the multi-metric extension renamed is not missing, it moved to the
                # name carrying its metric. Comparison then proceeds normally against it,
                # so the rename still has to leave every value untouched.
                renamed = RENAMED_FILES.get(baseline_csv.name)
                if renamed is not None and (smoke_csv.parent / renamed).exists():
                    notes.append(f"renamed by the multi-metric extension, compared against "
                                 f"{renamed}: {relative}")
                    smoke_csv = smoke_csv.parent / renamed
                elif baseline_csv.name in UPSTREAM_DEPENDENT_FILES:
                    notes.append(f"absent by configuration, modules 2 and 7 disabled: {relative}")
                    continue
                else:
                    failures.append(f"missing {smoke_csv}")
                    continue
            if filecmp.cmp(baseline_csv, smoke_csv, shallow=False):
                identical += 1
                continue
            # Not byte equal. It survives only if every value the baseline wrote is still
            # there and still equal, with any difference falling under a declared reason.
            problems = numeric_frames_match(pd.read_csv(baseline_csv), pd.read_csv(smoke_csv),
                                            baseline_csv.name)
            if problems:
                failures.extend(f"not byte equal and {problem}" for problem in problems)
            else:
                notes.append(f"not byte equal, values equal within {FLOAT_TOLERANCE}: {relative}")

    if checked == 0:
        failures.append("invariant 1 found no sublayer CSVs to compare")

    # The profile and profile-z rows of module 4, asserted by name so that narrowing the
    # sweep above could never silently drop the columns the block-axis change touches.
    for baseline_csv in sorted(B.glob("4_*/sublayers/*/correlation_summary.csv")):
        smoke_csv = S / baseline_csv.relative_to(B)
        if not smoke_csv.exists():
            failures.append(f"missing {smoke_csv}")
            continue
        base_rows = pd.read_csv(baseline_csv).set_index("plot_name")
        smoke_rows = pd.read_csv(smoke_csv).set_index("plot_name")
        for row in PROFILE_SUMMARY_ROWS:
            if row not in base_rows.index or row not in smoke_rows.index:
                failures.append(f"{baseline_csv.parent.name}: {row} missing on one side")
                continue
            left = float(base_rows.loc[row, "pearson_r"])
            right = float(smoke_rows.loc[row, "pearson_r"])
            if not np.isclose(left, right, rtol=FLOAT_TOLERANCE, atol=0.0):
                failures.append(f"{baseline_csv.parent.name}: {row} pearson_r "
                                f"{left} -> {right}")

    # --- invariants 2 and 3, module 3 whole-model jaccard and row set --------------------
    base3 = read_first(B, "3_*/category_concept_similarity_metrics.csv")
    smoke3 = read_first(S, "3_*/category_concept_similarity_metrics.csv")
    if base3 is None or smoke3 is None:
        failures.append("module 3 whole-model category_concept_similarity_metrics.csv "
                        "missing on one side, invariants 2 and 3 not checked")
    else:
        if len(base3) != len(smoke3):
            failures.append(f"module 3 rows {len(base3)} -> {len(smoke3)}")
        if set(base3["concept"]) != set(smoke3["concept"]):
            failures.append("module 3 whole-model concept set changed")
        merged = base3.merge(smoke3, on="concept", suffixes=("_b", "_s"))
        jaccard_diff = (merged["jaccard_pct_b"] - merged["jaccard_pct_s"]).abs().max()
        if not jaccard_diff == 0.0:
            failures.append(f"module 3 whole-model jaccard_pct changed, max diff {jaccard_diff}")

    # --- invariant 3, module 9's design-frame concept set --------------------------------
    base_design = read_first(B, "9_*/typicality_model_design.csv")
    smoke_design = read_first(S, "9_*/typicality_model_design.csv")
    if base_design is None or smoke_design is None:
        failures.append("module 9 typicality_model_design.csv missing on one side, "
                        "design-frame concept set not checked")
    elif set(base_design["concept"]) != set(smoke_design["concept"]):
        failures.append("module 9 design-frame concept set changed")

    # --- invariant 4, the restructured study outputs exist and are shaped correctly -------
    rank = read_first(S, "9_*/9.1_typicality_ranking/ranker_comparison.csv")
    metrics = []
    if rank is None:
        failures.append("smoke ranker_comparison.csv missing, invariant 4 ranking shape "
                        "and the ranking half of the cross-tree comparison not checked")
    else:
        metrics = sorted(set(rank["metric"]) - {"none"})
        # build_feature_grid emits the jaccard cell ONCE, with metric "none", plus the five
        # profile cells per metric, so a grid is 1 + 5 * len(metrics) cells and never
        # 6 * len(metrics). Study A fits every cell under both references.
        expected_rank_rows = (1 + 5 * max(len(metrics), 1)) * 2
        if len(rank) != expected_rank_rows:
            failures.append(f"ranker_comparison.csv has {len(rank)} rows, expected "
                            f"{expected_rank_rows} for {len(metrics)} metric(s)")
        if set(rank["reference"]) != {"label_word", "member_centroid"}:
            failures.append(f"ranker_comparison.csv references {sorted(set(rank['reference']))}")

    pair = read_first(S, "9_*/9.2_pair_similarity/pair_similarity_comparison.csv")
    if pair is None:
        failures.append("smoke pair_similarity_comparison.csv missing, invariant 4 pair "
                        "shape and the pair half of the cross-tree comparison not checked")
    else:
        expected_pair_rows = 1 + 5 * max(len(metrics), 1)
        if len(pair) != expected_pair_rows:
            failures.append(f"pair_similarity_comparison.csv has {len(pair)} rows, expected "
                            f"{expected_pair_rows}")

    # --- invariant 4, cross-tree: the smoke jaccard cells against the baseline's own jaccard
    # ranking. The baseline predates the multi-reference and multi-scope layout, its jaccard
    # tables live directly under 9_typicality_prediction/ with one row named "jaccard" each,
    # not under the 9.2_pairwise_ranking/9.4_human_pair_similarity paths module 9 used to use.
    base_rank = read_first(B, "9_*/typicality_ranker_comparison.csv")
    if rank is None or base_rank is None:
        failures.append("ranker jaccard row missing on one side of the cross-tree "
                        "comparison (source CSV absent)")
    else:
        smoke_j = rank[(rank["model"] == "jaccard") & (rank["reference"] == "label_word")]
        base_j = base_rank[base_rank["model"] == "jaccard"]
        if smoke_j.empty or base_j.empty:
            failures.append("ranker jaccard row missing on one side of the cross-tree comparison")
        else:
            if int(smoke_j["n_pairs"].iloc[0]) != int(base_j["n_pairs"].iloc[0]):
                failures.append(f"ranker jaccard n_pairs {base_j['n_pairs'].iloc[0]} -> "
                                f"{smoke_j['n_pairs'].iloc[0]}")
            left, right = float(base_j["accuracy"].iloc[0]), float(smoke_j["accuracy"].iloc[0])
            if not np.isclose(left, right, rtol=FLOAT_TOLERANCE, atol=0.0):
                failures.append(f"ranker jaccard accuracy {left} -> {right}")

    base_pair = read_first(B, "9_*/pair_similarity_comparison.csv")
    if pair is None or base_pair is None:
        failures.append("pair jaccard row missing on one side of the cross-tree "
                        "comparison (source CSV absent)")
    else:
        smoke_pj = pair[pair["model"] == "jaccard"]
        base_pj = base_pair[base_pair["model"] == "jaccard"]
        if smoke_pj.empty or base_pj.empty:
            failures.append("pair jaccard row missing on one side of the cross-tree comparison")
        else:
            left = float(base_pj["mean_category_spearman"].iloc[0])
            right = float(smoke_pj["mean_category_spearman"].iloc[0])
            if not np.isclose(left, right, rtol=FLOAT_TOLERANCE, atol=0.0):
                failures.append(f"pair jaccard mean_category_spearman {left} -> {right}")

    for note in notes:
        print("  note: " + note)
    print(f"  invariant 1: {identical} of {checked} baseline sublayer CSVs byte identical, "
          f"{len(notes)} explained")
    if failures:
        print("FAIL check_smoke_regression")
        for line in failures:
            print("  " + line)
        return 1
    print("OK check_smoke_regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
