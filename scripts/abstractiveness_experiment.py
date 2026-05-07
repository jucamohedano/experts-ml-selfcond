import logging
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define a high-contrast color palette for the different abstraction levels
ABSTRACTION_COLORS = {1: "#003f5c", 2: "#ffa600"} 
sns.set_theme(style="whitegrid")

# Recursively searches for expertise CSV files in the model directory, 
# filters the neurons based on the given Average Precision (AP) threshold, 
# and compiles them into a single dataframe.
def load_data(root, model, threshold) -> pd.DataFrame:
    """Load expertise CSVs recursively and filter by AP threshold."""
    all_rows = []
    path = pathlib.Path(root) / model
    for csv_file in path.glob("**/expertise/expertise.csv"):
        df = pd.read_csv(csv_file)
        experts = df[df["ap"] >= threshold].copy()
        experts["concept"] = csv_file.parent.parent.name
        all_rows.append(experts)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# Merges the raw expert data with the JSON metadata (frequency, typicality)
# It cleans up the layer names and calculates the relative percentage of experts 
# per concept to allow for fair comparisons across different concepts.
def prepare_data(df, meta) -> pd.DataFrame:
    """Parse layer strings into sortable labels and compute relative expert weights."""
    merged = df.merge(meta, on="concept")
    
    merged['layer_label'] = (
        merged['layer']
        .str.replace('transformer.h.', 'L.', regex=False)
        .str.replace(r':0$', '', regex=True)
    )
    
    merged['layer_idx'] = merged['layer'].str.extract(r'h\.(\d+)').astype(int)
    merged['layer_type_code'] = merged['layer'].apply(lambda x: 0 if 'attn' in x.lower() else 1)
    
    counts = merged.groupby('concept').size()
    merged['expert_share_pct'] = merged['concept'].apply(lambda x: (1 / counts[x]) * 100)
    
    sorted_unique_labels = (
        merged.sort_values(['layer_idx', 'layer_type_code', 'layer_label'])['layer_label']
        .unique()
    )
    merged['layer_label'] = pd.Categorical(merged['layer_label'], categories=sorted_unique_labels, ordered=True)
    
    return merged.sort_values(['layer_idx', 'layer_type_code'])


# Generates Plot A: A global bar chart showing how experts for different 
# abstraction levels are distributed across the model's attention and MLP layers.
def plot_layer_distribution(df, meta, out_dir):
    """Plot global expert distribution across all model blocks."""
    merged = prepare_data(df, meta)
    
    stats_df = merged.groupby(['abstraction_level', 'layer_label'], observed=False)['expert_share_pct'].sum().reset_index()
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


# Iterates through every single concept and generates an individual bar chart 
# showing exactly where its specific experts are located in the model architecture.
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


# Generates Plot B: Analyzes the hierarchical relationships (e.g., animal -> dog) 
# and creates bar charts showing how many expert neurons are shared between categories and specific concepts.
# Generates Plot B: Analyzes the hierarchical relationships (e.g., animal -> dog) 
# and creates bar charts showing how many expert neurons are shared between parents and children.
def plot_hierarchy_overlap(df, meta, out_dir):
    """Test expert retention and Jaccard similarity in category-concept concept pairs."""
    results = []
    for _, row in meta.dropna(subset=["category"]).iterrows():
        concept, category = row["concept"], row["category"]
        u_concept = set(df[df["concept"] == concept]["unit"])
        u_category = set(df[df["concept"] == category]["unit"])
        
        if u_concept and u_category:
            intersection = len(u_concept.intersection(u_category))
            results.append({
                "hierarchy": f"{category} -> {concept}",
                "retention_pct": (intersection / len(u_category)) * 100,
                "jaccard_pct": (intersection / len(u_concept.union(u_category))) * 100
            })
    
    res_df = pd.DataFrame(results)
    if res_df.empty: return
    res_df.to_csv(out_dir / "B_hierarchy_metrics.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40.56, 20.28))

    sns.barplot(data=res_df, x="hierarchy", y="retention_pct", ax=ax1, color="#34495e", edgecolor="black")
    ax1.set_title("Expert Retention: % of Category units preserved by Concept", fontsize=18)
    ax1.set_xticklabels([])
    ax1.set_xlabel("")

    # Draw leader lines pointing to the bottom (y=0) of each bar
    for i, bar in enumerate(ax1.patches):
        bar_x = bar.get_x() + (bar.get_width() / 2)
        label = res_df['hierarchy'].iloc[i]
        
        ax1.annotate(
            "",
            xy=(bar_x, 0), 
            xytext=(0, -10),
            textcoords="offset points",
            arrowprops=dict(
                arrowstyle="-", 
                color="black", 
                linewidth=1.0,
                shrinkA=0,
                shrinkB=0
            )
        )
        
        ax1.annotate(
            label,
            xy=(bar_x, 0), 
            xytext=(2, -8),
            textcoords="offset points",
            ha='right', 
            va='top', 
            rotation=45, 
            fontsize=10
        )

    sns.barplot(data=res_df, x="hierarchy", y="jaccard_pct", ax=ax2, color="#27ae60", edgecolor="black")
    ax2.set_title("Jaccard Similarity Index % across Hierarchy", fontsize=18)
    ax2.set_xticklabels([])
    ax2.set_xlabel("") 

    # Draw leader lines for the right barplot
    for i, bar in enumerate(ax2.patches):
        bar_x = bar.get_x() + (bar.get_width() / 2)
        label = res_df['hierarchy'].iloc[i]
        
        ax2.annotate(
            "",
            xy=(bar_x, 0), 
            xytext=(0, -10),
            textcoords="offset points",
            arrowprops=dict(
                arrowstyle="-", 
                color="black", 
                linewidth=1.0,
                shrinkA=0,
                shrinkB=0
            )
        )

        ax2.annotate(
            label,
            xy=(bar_x, 0), 
            xytext=(2, -8),
            textcoords="offset points",
            ha='right', 
            va='top', 
            rotation=45, 
            fontsize=10
        )

    plt.subplots_adjust(hspace=0.4, bottom=0.15)
    plt.savefig(out_dir / "B_hierarchy_overlap.png", dpi=300, bbox_inches="tight")
    plt.close()


# Generates Plot C: Creates two side-by-side scatter plots with regression lines 
# to show if the total number of experts correlates with a word's Frequency or Typicality.
def plot_frequency_typicality(df, meta, out_dir):
    """Analyze correlation between concept statistics and expert population."""
    counts = df.groupby("concept").size().reset_index(name="expert_count")
    merged = counts.merge(meta, on="concept")
    merged.to_csv(out_dir / "C_frequency_typicality_data.csv", index=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20.8, 9.1))

    freq_data = merged.dropna(subset=['frequency', 'expert_count'])
    if len(freq_data) > 2:
        r_f, p_f = stats.pearsonr(freq_data['frequency'], freq_data['expert_count'])
        sns.regplot(data=freq_data, x="frequency", y="expert_count", ax=ax1, scatter_kws={'color': '#444e86'}, line_kws={'color':'red'})
        ax1.set_title(f"Frequency vs Expert Count (r={r_f:.2f}, p={p_f:.3f})", fontsize=16)
    else:
        ax1.set_title("Frequency Correlation (insufficient data)", fontsize=16)

    typ_data = merged.dropna(subset=['typicality', 'expert_count'])
    if len(typ_data) > 2:
        r_t, p_t = stats.pearsonr(typ_data['typicality'], typ_data['expert_count'])
        sns.regplot(data=typ_data, x="typicality", y="expert_count", ax=ax2, scatter_kws={'color': '#955196'}, line_kws={'color':'red'})
        ax2.set_title(f"Typicality vs Expert Count (r={r_t:.2f}, p={p_t:.3f})", fontsize=16)
    else:
        ax2.set_title("Typicality Correlation (insufficient data)", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(out_dir / "C_correlations.png", dpi=300)
    plt.close()


# Generates Plot D: Calculates the pairwise similarity between every possible concept pair 
# and visualizes it as two large heatmaps (one for absolute overlap, one for Jaccard index).
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

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(79.09, 32.96))
    
    sns.heatmap(overlap_mtx, xticklabels=False, yticklabels=concepts, cmap="magma", ax=ax1)
    ax1.set_title("Pairwise Overlap % (relative to smaller set size)", fontsize=20)
    ax1.set_xlabel("")

    # Draw leader lines pointing to the bottom edge (y=n) of the heatmap
    for i, concept in enumerate(concepts):
        x_pos = i + 0.5
        
        ax1.annotate(
            "",  
            xy=(x_pos, n), 
            xytext=(0, -10),
            textcoords="offset points",
            arrowprops=dict(
                arrowstyle="-", 
                color="black", 
                linewidth=1.0,
                shrinkA=0,
                shrinkB=0
            )
        )
        
        ax1.annotate(
            concept,
            xy=(x_pos, n), 
            xytext=(3, -6),
            textcoords="offset points",
            ha='right', 
            va='top', 
            rotation=45, 
            fontsize=10
        )

    sns.heatmap(jaccard_mtx, xticklabels=False, yticklabels=concepts, cmap="viridis", ax=ax2)
    ax2.set_title("Pairwise Jaccard Similarity Index %", fontsize=20)
    ax2.set_xlabel("")

    # Draw leader lines for the right heatmap
    for i, concept in enumerate(concepts):
        x_pos = i + 0.5 
        
        ax2.annotate(
            "",  
            xy=(x_pos, n), 
            xytext=(0, -10),
            textcoords="offset points",
            arrowprops=dict(
                arrowstyle="-", 
                color="black", 
                linewidth=1.0,
                shrinkA=0,
                shrinkB=0
            )
        )
        
        ax2.annotate(
            concept,
            xy=(x_pos, n), 
            xytext=(3, -6),
            textcoords="offset points",
            ha='right', 
            va='top', 
            rotation=45, 
            fontsize=10
        )

    plt.subplots_adjust(bottom=0.15)
    plt.savefig(out_dir / "D_similarity_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()


# Generates Plot E: Tests for bias in the underlying dataset by plotting 
# Wikipedia Frequency directly against Typicality ratings to see if they correlate.
def plot_freq_vs_typ_correlation(meta, out_dir):
    """Analyze correlation specifically between Frequency and Typicality."""
    valid_data = meta.dropna(subset=['frequency', 'typicality'])
    
    plt.figure(figsize=(10.4, 9.1))
    
    if len(valid_data) > 2:
        r, p = stats.pearsonr(valid_data['frequency'], valid_data['typicality'])
        sns.regplot(data=valid_data, x="frequency", y="typicality", 
                    scatter_kws={'color': '#2ca02c'}, line_kws={'color':'red'})
        plt.title(f"Frequency vs. Typicality Correlation (r={r:.2f}, p={p:.3f})", fontsize=16)
    else:
        plt.title("Frequency vs. Typicality Correlation (insufficient data)", fontsize=16)     
    plt.tight_layout()
    plt.savefig(out_dir / "E_freq_vs_typicality.png", dpi=300)
    plt.close()


# Main execution block: Loads metadata, iterates through different AP thresholds, 
# and triggers all plotting functions, saving outputs into threshold-specific folders.
if __name__ == "__main__":
    SOURCE_DIR = "../results/GPT2_abstractiveness_150_responses/" 
    TARGET_MODEL = "gpt2"
    OUTPUT_ROOT = pathlib.Path("../results/research_plots_!50")
    
    METADATA = pd.read_json("../assets/metadata.json")
    
    for ap in [0.5, 0.6, 0.7, 0.8, 0.9]:
        out_path = OUTPUT_ROOT / f"AP_{ap}"
        out_path.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Starting Analysis Suite for AP Threshold: {ap}")
        df_experts = load_data(SOURCE_DIR, TARGET_MODEL, ap)
        
        if not df_experts.empty:
            plot_layer_distribution(df_experts, METADATA, out_path)
            plot_per_concept_distribution(df_experts, METADATA, out_path)
            plot_hierarchy_overlap(df_experts, METADATA, out_path)
            plot_frequency_typicality(df_experts, METADATA, out_path)
            plot_similarity_matrices(df_experts, METADATA, out_path)
            plot_freq_vs_typ_correlation(METADATA, out_path)
            log.info(f"Plots and CSVs generated successfully in {out_path}")
        else:
            log.warning(f"No experts found for threshold {ap}. Skipping folder.")