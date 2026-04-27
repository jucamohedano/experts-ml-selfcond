import logging
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- LOGGING AND STYLE CONFIGURATION ---
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# High-contrast palette for clear visibility
ABSTRACTION_COLORS = {1: "#003f5c", 2: "#bc5090", 3: "#ffa600"} 
sns.set_theme(style="whitegrid")

# --- CONCEPT METADATA ---
metadata_list = [
    # --- LEVEL 1: Root Categories (Frequenza altissima in Wikipedia) ---
    {"concept": "animal", "abstraction_level": 1, "frequency": 1250000, "typicality": np.nan, "parent": None},
    {"concept": "plant", "abstraction_level": 1, "frequency": 850000, "typicality": np.nan, "parent": None},
    {"concept": "object", "abstraction_level": 1, "frequency": 920000, "typicality": np.nan, "parent": None},

    # --- LEVEL 2: Subcategories (Parent: animal) ---
    {"concept": "mammal", "abstraction_level": 2, "frequency": 145000, "typicality": 0.95, "parent": "animal"},
    {"concept": "insect", "abstraction_level": 2, "frequency": 110000, "typicality": 0.70, "parent": "animal"},
    {"concept": "bird", "abstraction_level": 2, "frequency": 180000, "typicality": 0.85, "parent": "animal"},
    {"concept": "reptile", "abstraction_level": 2, "frequency": 75000, "typicality": 0.60, "parent": "animal"},
    {"concept": "fish", "abstraction_level": 2, "frequency": 160000, "typicality": 0.80, "parent": "animal"},

    # --- LEVEL 2: Subcategories (Parent: plant) ---
    {"concept": "fruit", "abstraction_level": 2, "frequency": 135000, "typicality": 0.90, "parent": "plant"},
    {"concept": "vegetable", "abstraction_level": 2, "frequency": 85000, "typicality": 0.82, "parent": "plant"},
    {"concept": "tree", "abstraction_level": 2, "frequency": 240000, "typicality": 0.92, "parent": "plant"},
    {"concept": "flower", "abstraction_level": 2, "frequency": 195000, "typicality": 0.88, "parent": "plant"},

    # --- LEVEL 2: Subcategories (Parent: object) ---
    {"concept": "vehicle", "abstraction_level": 2, "frequency": 210000, "typicality": 0.94, "parent": "object"},
    {"concept": "furniture", "abstraction_level": 2, "frequency": 70000, "typicality": 0.88, "parent": "object"},
    {"concept": "clothing", "abstraction_level": 2, "frequency": 115000, "typicality": 0.90, "parent": "object"},
    {"concept": "musical instrument", "abstraction_level": 2, "frequency": 95000, "typicality": 0.75, "parent": "object"},

    # --- LEVEL 3: Specific Concepts (Animals) ---
    {"concept": "dog", "abstraction_level": 3, "frequency": 250000, "typicality": 0.98, "parent": "mammal"},
    {"concept": "elephant", "abstraction_level": 3, "frequency": 35000, "typicality": 0.65, "parent": "mammal"},
    {"concept": "tiger", "abstraction_level": 3, "frequency": 42000, "typicality": 0.70, "parent": "mammal"},
    {"concept": "bee", "abstraction_level": 3, "frequency": 28000, "typicality": 0.88, "parent": "insect"},
    {"concept": "ant", "abstraction_level": 3, "frequency": 22000, "typicality": 0.80, "parent": "insect"},
    {"concept": "eagle", "abstraction_level": 3, "frequency": 55000, "typicality": 0.92, "parent": "bird"},
    {"concept": "penguin", "abstraction_level": 3, "frequency": 18000, "typicality": 0.25, "parent": "bird"},
    {"concept": "snake", "abstraction_level": 3, "frequency": 48000, "typicality": 0.90, "parent": "reptile"},
    {"concept": "salmon", "abstraction_level": 3, "frequency": 15000, "typicality": 0.75, "parent": "fish"},

    # --- LEVEL 3: Specific Concepts (Plants) ---
    {"concept": "apple", "abstraction_level": 3, "frequency": 65000, "typicality": 0.96, "parent": "fruit"},
    {"concept": "banana", "abstraction_level": 3, "frequency": 32000, "typicality": 0.85, "parent": "fruit"},
    {"concept": "carrot", "abstraction_level": 3, "frequency": 12000, "typicality": 0.92, "parent": "vegetable"},
    {"concept": "oak", "abstraction_level": 3, "frequency": 45000, "typicality": 0.90, "parent": "tree"},
    {"concept": "rose", "abstraction_level": 3, "frequency": 58000, "typicality": 0.97, "parent": "flower"},

    # --- LEVEL 3: Specific Concepts (Objects) ---
    {"concept": "car", "abstraction_level": 3, "frequency": 650000, "typicality": 0.99, "parent": "vehicle"},
    {"concept": "airplane", "abstraction_level": 3, "frequency": 120000, "typicality": 0.78, "parent": "vehicle"},
    {"concept": "chair", "abstraction_level": 3, "frequency": 95000, "typicality": 0.98, "parent": "furniture"},
    {"concept": "bed", "abstraction_level": 3, "frequency": 110000, "typicality": 0.92, "parent": "furniture"},
    {"concept": "shirt", "abstraction_level": 3, "frequency": 35000, "typicality": 0.88, "parent": "clothing"},
    {"concept": "guitar", "abstraction_level": 3, "frequency": 85000, "typicality": 0.94, "parent": "musical instrument"},
]
METADATA = pd.DataFrame(metadata_list)

# --- DATA PROCESSING UTILS ---

def prepare_data(df, meta):
    """Parse layer strings into sortable labels and compute relative expert weights."""
    merged = df.merge(meta, on="concept")
    
    # Standardize naming: transformer.h.12.attn -> L.12.attn
    merged['layer_label'] = merged['layer'].str.replace('transformer.h.', 'L.', regex=False)
    
    # Extract numerical index and component type for strict sorting
    merged['layer_idx'] = merged['layer'].str.extract(r'h\.(\d+)').astype(int)
    merged['layer_type_code'] = merged['layer'].apply(lambda x: 0 if 'attn' in x.lower() else 1)
    
    # Share of a single expert relative to its concept's total population
    counts = merged.groupby('concept').size()
    merged['expert_share_pct'] = merged['concept'].apply(lambda x: (1 / counts[x]) * 100)
    
    # Create categorical ordering for the X-axis
    sorted_unique_labels = (
        merged.sort_values(['layer_idx', 'layer_type_code', 'layer_label'])['layer_label']
        .unique()
    )
    merged['layer_label'] = pd.Categorical(merged['layer_label'], categories=sorted_unique_labels, ordered=True)
    
    return merged.sort_values(['layer_idx', 'layer_type_code'])

# --- CORE PLOTTING FUNCTIONS ---

def plot_layer_distribution(df, meta, out_dir):
    """Plot global expert distribution across all model blocks."""
    merged = prepare_data(df, meta)
    
    # Group by level and layer to get summed shares
    stats_df = merged.groupby(['abstraction_level', 'layer_label'], observed=False)['expert_share_pct'].sum().reset_index()
    
    # Normalize by the number of concepts within each abstraction level
    concepts_per_level = meta.groupby('abstraction_level')['concept'].nunique()
    stats_df['avg_level_share'] = stats_df.apply(lambda x: x['expert_share_pct'] / concepts_per_level[x['abstraction_level']], axis=1)

    stats_df.to_csv(out_dir / "A_layer_distribution_stats.csv", index=False)

    plt.figure(figsize=(31.2, 10.4))
    sns.barplot(data=stats_df, x="layer_label", y="avg_level_share", hue="abstraction_level", 
                palette=ABSTRACTION_COLORS, edgecolor="black", linewidth=0.5)
    
    plt.title("Mean Expert Distribution across Model Components (L.X.attn vs L.X.mlp)", fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylabel("Average % Contribution per Concept", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "A_global_layer_distribution.png", dpi=300)
    plt.close()

def plot_per_concept_distribution(df, meta, out_dir):
    """Plot expert counts for each individual concept in the dataset."""
    merged = prepare_data(df, meta)
    concept_dir = out_dir / "per_concept_plots"
    concept_dir.mkdir(exist_ok=True)

    for concept in merged['concept'].unique():
        data = merged[merged['concept'] == concept]
        abs_lvl = data['abstraction_level'].iloc[0]
        
        dist = data.groupby('layer_label', observed=False).size().reset_index(name='expert_count')
        dist['pct'] = (dist['expert_count'] / len(data)) * 100
        dist.to_csv(concept_dir / f"{concept}_data.csv", index=False)

        plt.figure(figsize=(28.6, 7.8))
        sns.barplot(data=dist, x="layer_label", y="pct", color=ABSTRACTION_COLORS[abs_lvl], edgecolor="black")
        
        plt.title(f"Concept: {concept.upper()} | Abstraction Level {abs_lvl} | Total Experts: {len(data)}", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.ylabel("% Share of Total Experts")
        plt.tight_layout()
        plt.savefig(concept_dir / f"{concept}_distribution.png", dpi=200)
        plt.close()

def plot_hierarchy_overlap(df, meta, out_dir):
    """Test expert retention and Jaccard similarity in parent-child concept pairs."""
    results = []
    for _, row in meta.dropna(subset=["parent"]).iterrows():
        child, parent = row["concept"], row["parent"]
        u_child = set(df[df["concept"] == child]["unit"])
        u_parent = set(df[df["concept"] == parent]["unit"])
        
        if u_child and u_parent:
            intersection = len(u_child.intersection(u_parent))
            results.append({
                "hierarchy": f"{parent} -> {child}",
                "retention_pct": (intersection / len(u_parent)) * 100,
                "jaccard_pct": (intersection / len(u_child.union(u_parent))) * 100
            })
    
    res_df = pd.DataFrame(results)
    if res_df.empty: return
    res_df.to_csv(out_dir / "B_hierarchy_metrics.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20.8, 15.6))

    sns.barplot(data=res_df, x="hierarchy", y="retention_pct", ax=ax1, color="#34495e", edgecolor="black")
    ax1.set_title("Expert Retention: % of Parent units preserved by Child", fontsize=18)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    sns.barplot(data=res_df, x="hierarchy", y="jaccard_pct", ax=ax2, color="#27ae60", edgecolor="black")
    ax2.set_title("Jaccard Similarity Index % across Hierarchy", fontsize=18)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(out_dir / "B_hierarchy_overlap.png", dpi=300)
    plt.close()

def plot_similarity_matrices(df, meta, out_dir):
    """Generate pairwise similarity heatmaps for all concepts."""
    concepts = meta['concept'].unique()
    sets = {c: set(df[df["concept"] == c]["unit"]) for c in concepts}
    
    n = len(concepts)
    overlap_mtx = np.zeros((n, n))
    jaccard_mtx = np.zeros((n, n))

    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            s1, s2 = sets[c1], sets[c2]
            if not s1 or not s2: continue
            inter = len(s1.intersection(s2))
            overlap_mtx[i, j] = (inter / min(len(s1), len(s2))) * 100
            jaccard_mtx[i, j] = (inter / len(s1.union(s2))) * 100

    pd.DataFrame(overlap_mtx, index=concepts, columns=concepts).to_csv(out_dir / "D_overlap_matrix.csv")
    pd.DataFrame(jaccard_mtx, index=concepts, columns=concepts).to_csv(out_dir / "D_jaccard_matrix.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(31.2, 13.0))
    
    sns.heatmap(overlap_mtx, xticklabels=concepts, yticklabels=concepts, cmap="magma", ax=ax1)
    ax1.set_title("Pairwise Overlap % (relative to smaller set size)", fontsize=20)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    sns.heatmap(jaccard_mtx, xticklabels=concepts, yticklabels=concepts, cmap="viridis", ax=ax2)
    ax2.set_title("Pairwise Jaccard Similarity Index %", fontsize=20)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(out_dir / "D_similarity_heatmaps.png", dpi=300)
    plt.close()

def plot_frequency_typicality(df, meta, out_dir):
    """Analyze correlation between concept statistics and expert population."""
    counts = df.groupby("concept").size().reset_index(name="expert_count")
    merged = counts.merge(meta, on="concept")
    # Generates the CSV with frequency, abstraction_level, typicality, and expert counts
    merged.to_csv(out_dir / "C_frequency_typicality_data.csv", index=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20.8, 9.1))

    r_f, p_f = stats.pearsonr(merged['frequency'], merged['expert_count'])
    sns.regplot(data=merged, x="frequency", y="expert_count", ax=ax1, scatter_kws={'color': '#444e86'}, line_kws={'color':'red'})
    ax1.set_title(f"Frequency Correlation (r={r_f:.2f}, p={p_f:.3f})", fontsize=16)

    typ_data = merged.dropna(subset=['typicality'])
    if len(typ_data) > 2:
        r_t, p_t = stats.pearsonr(typ_data['typicality'], typ_data['expert_count'])
        sns.regplot(data=typ_data, x="typicality", y="expert_count", ax=ax2, scatter_kws={'color': '#955196'}, line_kws={'color':'red'})
        ax2.set_title(f"Typicality Correlation (r={r_t:.2f}, p={p_t:.3f})", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(out_dir / "C_correlations.png", dpi=300)
    plt.close()

# --- MAIN EXECUTION ---

def load_data(root, model, threshold):
    """Load expertise CSVs recursively and filter by AP threshold."""
    all_rows = []
    path = pathlib.Path(root) / model
    for csv_file in path.glob("**/expertise/expertise.csv"):
        df = pd.read_csv(csv_file)
        # Filter for quality neurons (AP)
        experts = df[df["ap"] >= threshold].copy()
        experts["concept"] = csv_file.parent.parent.name
        all_rows.append(experts)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

if __name__ == "__main__":
    SOURCE_DIR = "./risultati_locali/GPT2_abstractiveness_20_responses/" 
    TARGET_MODEL = "gpt2"
    OUTPUT_ROOT = pathlib.Path("./research_plots")
    
    for ap in [0.5, 0.6, 0.7, 0.8, 0.9]:
        out_path = OUTPUT_ROOT / f"AP_{ap}"
        out_path.mkdir(parents=True, exist_ok=True)
        
        log.info(f"--- Starting Analysis Suite for AP Threshold: {ap} ---")
        df_experts = load_data(SOURCE_DIR, TARGET_MODEL, ap)
        
        if not df_experts.empty:
            plot_layer_distribution(df_experts, METADATA, out_path)
            plot_per_concept_distribution(df_experts, METADATA, out_path)
            plot_hierarchy_overlap(df_experts, METADATA, out_path)
            plot_similarity_matrices(df_experts, METADATA, out_path)
            plot_frequency_typicality(df_experts, METADATA, out_path)
            log.info(f"Plots and CSVs generated successfully in {out_path}")
        else:
            log.warning(f"No experts found for threshold {ap}. Skipping folder.")