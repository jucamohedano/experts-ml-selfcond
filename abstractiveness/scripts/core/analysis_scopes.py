"""
Analysis scopes (whole model, then one per sublayer), their output folders and layer axes.

Used by the executor and every module to decide what to analyze and where to write it.

Explanations: abstractiveness/CLAUDE.md, sections Analysis scopes and Layer-axis variants.
"""

import logging
import pathlib
from dataclasses import dataclass
import pandas as pd
from core.expert_data_loading import filter_expert_data_to_sublayer
from utils.logging_and_io import save_dataframe

log = logging.getLogger(__name__)


WHOLE_MODEL_SCOPE_KEY = "whole_model"


@dataclass(frozen=True, eq=False)
class AnalysisScope:
    """One unit of analysis: a slice of the expert data plus where its outputs belong."""
    key: str
    label: str
    dir_name: str | None
    is_whole_model: bool
    expert_df: pd.DataFrame


def sublayer_expert_counts(expert_allocation_df: pd.DataFrame) -> pd.Series:
    """Expert-row count per sublayer type, descending."""
    sublayer_of = expert_allocation_df['layer_name'].astype(str).str.extract(r'^\d+\.L\.\d+\.(.+)$')[0]
    return sublayer_of.value_counts().sort_values(ascending=False)


def load_or_build_sublayer_rank(rank_path: pathlib.Path, build_expert_df, reference_ap: float) -> pd.DataFrame:
    """Rank sublayer types by expert count, cached at rank_path across the whole AP sweep."""
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
    """Build the scope list every module iterates over: the whole model first, then one scope per sublayer in rank order."""
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


def scope_section_dir(module_dir: pathlib.Path, scope: AnalysisScope, section_dir: str) -> pathlib.Path:
    """Output folder for one section of a merged module, <module>[/sublayers/<scope>]/<section>."""
    out_dir = scope_out_dir(module_dir, scope) / section_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def scope_summary_row(scope: AnalysisScope, **metrics) -> dict:
    """One row of a module's sublayer_comparison table: the scope's identity followed by that module's headline metrics."""
    return {"scope": scope.label, "is_whole_model": scope.is_whole_model,
            "n_experts": len(scope.expert_df), **metrics}


def axis_variants(scope: AnalysisScope) -> list:
    """Layer-axis variants an order-based module should run for a given scope, as (filename_suffix, expert_df, axis_label) tuples with the canonical variant first."""
    if scope.is_whole_model:
        return [("_by_block", to_block_axis(scope.expert_df), "block (depth) axis"),
                ("_by_layer", scope.expert_df, "flat layer axis")]
    return [("", scope.expert_df, "layer axis")]


def to_block_axis(expert_allocation_df: pd.DataFrame) -> pd.DataFrame:
    """Re-key expert rows from the flat layer axis onto a block (depth) axis, summing the sublayers within each transformer block into one bin."""
    block_of_row = expert_allocation_df['layer_name'].astype(str).str.extract(r'^\d+\.L\.(\d+)\.')[0].astype(int)
    support = sorted(pd.Series(expert_allocation_df['layer_name'].cat.categories)
                     .astype(str).str.extract(r'^\d+\.L\.(\d+)\.')[0].astype(int).unique())

    block_axis_df = expert_allocation_df.copy()
    block_axis_df['layer_idx'] = block_of_row.to_numpy() + 1
    block_axis_df['layer_name'] = pd.Categorical(
        [f"{b + 1}.B.{b}" for b in block_of_row],
        categories=[f"{b + 1}.B.{b}" for b in support], ordered=True)
    return block_axis_df
