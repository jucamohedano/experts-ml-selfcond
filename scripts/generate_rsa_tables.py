#!/usr/bin/env python3
"""
Generate RSA statistics tables

Extracts key metrics from RSA results for Attention and MLP layers and
appends formatted tables to my_docs/rsa_experiment.md.
"""

import json
import logging
import pathlib
import sys
from typing import Dict, List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def load_latest_result(run_dir: pathlib.Path) -> Optional[pd.DataFrame]:
    """Load the latest results DataFrame from a run directory."""
    subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    
    latest = sorted(subdirs)[-1]
    result_file = latest / "rsa_results.csv"
    
    if not result_file.exists():
        return None
        
    return pd.read_csv(result_file)



def extract_stats(df: pd.DataFrame, condition_name: str) -> Dict:
    """Extract summary statistics from results DataFrame."""
    n_regions = len(df)
    
    # Significance counts
    n_sig_raw = df["sig_raw"].sum() if "sig_raw" in df.columns else 0
    n_sig_fdr = df["sig_fdr"].sum() if "sig_fdr" in df.columns else 0
    
    # Effect size stats
    mean_d = df["cohens_d"].mean()
    max_d = df["cohens_d"].max()
    
    # t-statistic (Mean)
    mean_t = df["t_stat"].mean() if "t_stat" in df.columns else 0.0
    
    # RSA stats (Mean and Std)
    mean_expert = df["mean_rho_a"].mean()
    std_expert = df["mean_rho_a"].std()
    
    mean_full = df["mean_rho_b"].mean()
    std_full = df["mean_rho_b"].std()
    
    return {
        "Condition": condition_name,
        "Sig (Raw)": n_sig_raw,
        "Sig (FDR)": n_sig_fdr,
        "Mean d": mean_d,
        "Mean t": mean_t,
        "Max d": max_d,
        "Mean Expert": mean_expert,
        "Std Expert": std_expert,
        "Mean Full": mean_full,
        "Std Full": std_full,
    }

def generate_markdown_table(stats_list: List[Dict], caption: str) -> str:
    """Generate a Markdown table from a list of stats dictionaries."""
    if not stats_list:
        return ""
        
    # Define column order
    cols = ["Condition", "Sig (Raw)", "Sig (FDR)", "Mean d", "Max d", "Mean Expert (SD)", "Mean Full (SD)"]
    
    # Header
    md = f"\n### {caption}\n\n"
    md += "| " + " | ".join(cols) + " |\n"
    md += "|" + "|".join(["---" for _ in cols]) + "|\n"
    
    # Rows
    for row in stats_list:
        expert_str = f"{row['Mean Expert']:.3f} ({row['Std Expert']:.3f})"
        full_str = f"{row['Mean Full']:.3f} ({row['Std Full']:.3f})"
        
        line = f"| {row['Condition']} | {row['Sig (Raw)']} | {row['Sig (FDR)']} | {row['Mean d']:+.3f} | {row['Max d']:+.3f} | {expert_str} | {full_str} |"
        md += line + "\n"
        
    return md

def main():
    base_dir = pathlib.Path("results/rsa/Qwen3-30B-A3B-Instruct-2507_custom_60/Qwen3-30B-A3B-Instruct-2507_gpt2")
    output_doc = pathlib.Path("my_docs/rsa_experiment.md")
    
    if not base_dir.exists():
        log.error(f"Base directory not found: {base_dir}")
        sys.exit(1)
        
    # --- Process Attention Layers ---
    attn_stats = []
    attn_dirs = sorted(list(base_dir.glob("*_attn.c_proj")))
    
    log.info(f"Found {len(attn_dirs)} Attention layer runs")
    
    for d in attn_dirs:
        # Clean condition name: remove suffix and _vs_full
        clean_name = d.name.replace("_vs_full_attn.c_proj", "")
        
        df = load_latest_result(d)
        if df is not None:
            stats = extract_stats(df, clean_name)
            attn_stats.append(stats)
            
    # Sort by Mean d descending
    attn_stats.sort(key=lambda x: x["Mean d"], reverse=True)
    
    # --- Process MLP Layers ---
    mlp_stats = []
    mlp_dirs = sorted(list(base_dir.glob("*_mlp.c_fc")))
    
    log.info(f"Found {len(mlp_dirs)} MLP layer runs")
    
    for d in mlp_dirs:
        clean_name = d.name.replace("_vs_full_mlp.c_fc", "")
        
        df = load_latest_result(d)
        if df is not None:
            stats = extract_stats(df, clean_name)
            mlp_stats.append(stats)
            
    # Sort by Mean d descending
    mlp_stats.sort(key=lambda x: x["Mean d"], reverse=True)
    
    # --- Process attn.c_attn Layers ---
    attn_cattn_stats = []
    attn_cattn_dirs = sorted(list(base_dir.glob("*_attn.c_attn")))
    
    log.info(f"Found {len(attn_cattn_dirs)} attn.c_attn layer runs")
    
    for d in attn_cattn_dirs:
        clean_name = d.name.replace("_vs_full_attn.c_attn", "")
        
        df = load_latest_result(d)
        if df is not None:
            stats = extract_stats(df, clean_name)
            attn_cattn_stats.append(stats)
            
    # Sort by Mean d descending
    attn_cattn_stats.sort(key=lambda x: x["Mean d"], reverse=True)

    # --- Process mlp.c_proj Layers ---
    mlp_cproj_stats = []
    mlp_cproj_dirs = sorted(list(base_dir.glob("*_mlp.c_proj")))
    
    log.info(f"Found {len(mlp_cproj_dirs)} mlp.c_proj layer runs")
    
    for d in mlp_cproj_dirs:
        clean_name = d.name.replace("_vs_full_mlp.c_proj", "")
        
        df = load_latest_result(d)
        if df is not None:
            stats = extract_stats(df, clean_name)
            mlp_cproj_stats.append(stats)
            
    # Sort by Mean d descending
    mlp_cproj_stats.sort(key=lambda x: x["Mean d"], reverse=True)
    
    # Generate tables
    table_attn = generate_markdown_table(attn_stats, "Table 1: RSA Statistics - Attention Layers (`attn.c_proj`)")
    table_mlp = generate_markdown_table(mlp_stats, "Table 2: RSA Statistics - MLP Layers (`mlp.c_fc`)")
    table_attn_cattn = generate_markdown_table(attn_cattn_stats, "Table 3: RSA Statistics - Attention Layers (`attn.c_attn`)")
    table_mlp_cproj = generate_markdown_table(mlp_cproj_stats, "Table 4: RSA Statistics - MLP Layers (`mlp.c_proj`)")
    
    # Print all tables to console (always)
    print(table_attn)
    print(table_mlp)
    print(table_attn_cattn)
    print(table_mlp_cproj)
    
    # Append to documentation
    if output_doc.exists():
        with open(output_doc, "a") as f:
            f.write("\n\n## Aggregated Results (Generated)\n")
            f.write(table_attn)
            f.write("\n")
            f.write(table_mlp)
            f.write("\n")
            f.write(table_attn_cattn)
            f.write("\n")
            f.write(table_mlp_cproj)
        log.info(f"Appended tables to {output_doc}")
    else:
        log.warning(f"Output document not found: {output_doc}. Tables printed to console only.")

if __name__ == "__main__":
    main()
