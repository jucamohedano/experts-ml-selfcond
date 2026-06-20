import logging
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import entropy
import re

np.random.seed(42)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Add a StreamHandler so logs still print to the console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def set_folder_log(folder_path: pathlib.Path):
    """Updates the logger to write to a main.log inside the specified folder."""
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

# Define a high-contrast color palette for the different abstraction levels
ABSTRACTION_COLORS = {1: "#003f5c", 2: "#ffa600"} 
sns.set_theme(style="whitegrid")


# ==========================================
# DATA PREPARATION 
# ==========================================

def load_experts_data(root, model, threshold) -> pd.DataFrame:
    """Loads and concatenates expertise data for a given model and AP threshold, filtering for experts only."""
    all_rows = []
    path = pathlib.Path(root) / model
    for csv_file in path.glob("**/expertise/expertise.csv"):
        df = pd.read_csv(csv_file)
        experts = df[df["ap"] >= threshold].copy()
        all_rows.append(experts)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

def format_layer_labels(df, meta) -> pd.DataFrame:
    """Parse layer strings into an explicit 1-48 sequence and format labels elegantly."""
    merged = df.merge(meta, on="concept")
    
    SUB_LAYERS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    block_nums = merged['layer'].str.extract(r'h\.(\d+)').astype(int)[0]
    sub_layer_strs = merged['layer'].str.extract(r'h\.\d+\.(.*?):0')[0]
    sub_idx = sub_layer_strs.map({sub: i for i, sub in enumerate(SUB_LAYERS)})

    merged['layer_idx'] = (block_nums * 4) + sub_idx + 1
    merged['layer_name'] = (
        merged['layer_idx'].astype(str) + ".L." + 
        block_nums.astype(str) + "." + 
        sub_layer_strs
    )
    
    ordered_names = merged[['layer_idx', 'layer_name']].drop_duplicates().sort_values('layer_idx')['layer_name']
    merged['layer_name'] = pd.Categorical(merged['layer_name'], categories=ordered_names, ordered=True)

    return merged.sort_values('layer_idx')

def compute_expert_share(df) -> pd.DataFrame:
    """Compute relative expert weights for a pre-formatted dataframe."""
    counts = df.groupby('concept').size()
    df['unit_weight_pct'] = df['concept'].map(counts).apply(lambda x: (1 / x) * 100)
    return df

def save_layer_mapping_csv(formatted_df, out_path):
    """Saves a reference dictionary linking original model layers to the new 1-48 sequence labels."""
    mapping_df = formatted_df[['layer', 'layer_idx', 'layer_name']].drop_duplicates()
    mapping_df = mapping_df.sort_values('layer_idx').reset_index(drop=True)
    mapping_df.to_csv(out_path, index=False)


# ==========================================
# PLOTTING HELPERS
# ==========================================

def _plot_bar_with_leaders(df, x_col, y_col, title, ylabel, out_path, color):
    """Reusable helper to plot a bar chart with leader line annotations."""
    plt.figure(figsize=(40.56, 10.14))
    ax = sns.barplot(data=df, x=x_col, y=y_col, color=color, edgecolor="black")
    
    ax.set_title(title, fontsize=18)
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=14)

    for i, bar in enumerate(ax.patches):
        bar_x = bar.get_x() + (bar.get_width() / 2)
        label = df[x_col].iloc[i]
        
        ax.annotate("", xy=(bar_x, 0), xytext=(0, -10), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="black", linewidth=1.0, shrinkA=0, shrinkB=0))
        ax.annotate(label, xy=(bar_x, 0), xytext=(2, -8), textcoords="offset points",
                    ha='right', va='top', rotation=45, fontsize=10)

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_heatmap_with_leaders(matrix, concepts, title, out_path, cmap):
    """Reusable helper to plot a heatmap with leader line annotations."""
    n = len(concepts)
    plt.figure(figsize=(39.5, 32.96))
    ax = sns.heatmap(matrix, xticklabels=False, yticklabels=concepts, cmap=cmap)
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("")

    for i, concept in enumerate(concepts):
        x_pos = i + 0.5 
        ax.annotate("", xy=(x_pos, n), xytext=(0, -10), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="black", linewidth=1.0, shrinkA=0, shrinkB=0))
        ax.annotate(concept, xy=(x_pos, n), xytext=(3, -6), textcoords="offset points",
                    ha='right', va='top', rotation=45, fontsize=10)

    plt.subplots_adjust(bottom=0.15)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# MODULE 1: LAYER EXPERT DISTRIBUTION
# ==========================================

def save_expert_counts_metadata(df, meta, dist_dir):
    """Save the metadata combined with expert counts into the distribution folder."""
    counts = df.groupby("concept").size().reset_index(name="expert_count")
    merged = counts.merge(meta, on="concept")
    if 'frequency' in merged.columns:
        merged = merged[merged['frequency'] > 0].copy()
        merged['log_frequency'] = np.log10(merged['frequency'])
    desired_order = [
        "concept", 
        "category", 
        "abstraction_level", 
        "frequency", 
        "log_frequency", 
        "typicality", 
        "expert_count"
    ]
    final_order = [col for col in desired_order if col in merged.columns]
    merged = merged[final_order]
    merged.to_csv(dist_dir / "expert_counts_with_metadata.csv", index=False)
    return merged


def plot_layer_distribution(df, meta, dist_dir):
    """Plot global expert distribution and save per-concept distributions."""
    merged = format_layer_labels(df, meta)
    merged = compute_expert_share(merged)
    
    # Global Distribution
    stats_df = merged.groupby(['abstraction_level', 'layer_name'], observed=False)['unit_weight_pct'].sum().reset_index()
    concepts_per_level = meta.groupby('abstraction_level')['concept'].nunique()
    stats_df['avg_level_share'] = stats_df.apply(lambda x: x['unit_weight_pct'] / concepts_per_level[x['abstraction_level']], axis=1)

    stats_df.to_csv(dist_dir / "mean_expert_layer_distribution.csv", index=False)

    plt.figure(figsize=(31.2, 10.4))
    sns.barplot(data=stats_df, x="layer_name", y="avg_level_share", hue="abstraction_level", 
                palette=ABSTRACTION_COLORS, edgecolor="black", linewidth=0.5)
    
    plt.title("Mean Expert Distribution across model layers", fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.xlabel("Model Layer", fontsize=14)
    plt.ylabel("Average % of Experts", fontsize=14)
    plt.tight_layout()
    plt.savefig(dist_dir / "mean_expert_layer_distribution.png", dpi=300)
    plt.close()

    # Per-Concept Distributions
    concept_dir = dist_dir / "per_concept"
    for concept in merged['concept'].unique():
        data = merged[merged['concept'] == concept]
        abs_lvl = data['abstraction_level'].iloc[0]
        
        dist = data.groupby('layer_name', observed=False).size().reset_index(name='expert_count')
        dist['pct'] = (dist['expert_count'] / len(data)) * 100
        dist.to_csv(concept_dir / f"{concept}_data.csv", index=False)

        plt.figure(figsize=(28.6, 7.8))
        sns.barplot(data=dist, x="layer_name", y="pct", color=ABSTRACTION_COLORS[abs_lvl], edgecolor="black")
        
        plt.title(f"Concept: {concept.upper()} | Abstraction Level {abs_lvl} | Total Experts: {len(data)}", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.xlabel("Model Layer", fontsize=14)
        plt.ylabel("% of Experts", fontsize=14)
        plt.tight_layout()
        plt.savefig(concept_dir / f"{concept}_distribution.png", dpi=200)
        plt.close()
        
# ==========================================
# MODULE 2: SHANNON ENTROPY ANALYSIS AND PEAK/AVERAGE LAYER DISTRIBUTIONS
# ==========================================

def compute_shannon_entropy(df, meta, dist_dir):
    """Modular function to calculate Shannon entropy, peaks, and averages for Concepts and Categories."""
    merged = format_layer_labels(df, meta)
    dist_matrix = merged.groupby(['concept', 'layer_idx'], observed=False).size().unstack(fill_value=0)
    prob_matrix = dist_matrix.div(dist_matrix.sum(axis=1), axis=0)
    layer_indices = prob_matrix.columns.values
    # Entropy for individual concepts
    concept_results = []
    for concept in prob_matrix.index:
        probs = prob_matrix.loc[concept].values
        if np.sum(probs) == 0: continue 
        
        ent = entropy(probs, base=2) 
        avg_layer = np.sum(probs * layer_indices)
        peak_layer = layer_indices[np.argmax(probs)]
        experts_count = dist_matrix.loc[concept].sum()
        
        concept_results.append({
            "concept": concept,
            "shannon_entropy": ent,
            "peak_layer": peak_layer,
            "avg_layer": avg_layer,
            "experts_count": experts_count
        })
    concept_df = pd.DataFrame(concept_results)
    concept_df.to_csv(dist_dir / "shannon_entropy_concepts.csv", index=False)
    
    # Entropy for categories based on two definitions: the category label itself and the average of its members
    category_results = []
    for category in meta['category'].dropna().unique():
        members = meta[meta['category'] == category]['concept'].tolist()
        valid_members = [m for m in members if m in prob_matrix.index]
        
        # Definition A: The Word Itself (Category Label)
        if category in prob_matrix.index:
            lbl_probs = prob_matrix.loc[category].values
            lbl_ent = entropy(lbl_probs, base=2)
            lbl_avg = np.sum(lbl_probs * layer_indices)
            lbl_peak = layer_indices[np.argmax(lbl_probs)]
        else:
            lbl_probs, lbl_ent, lbl_avg, lbl_peak = None, np.nan, np.nan, np.nan
            
        # Definition B: Average of the Category Members
        if valid_members:
            avg_probs = prob_matrix.loc[valid_members].mean(axis=0).values
            avg_ent = entropy(avg_probs, base=2)
            avg_layer_avg = np.sum(avg_probs * layer_indices)
            avg_peak_avg = layer_indices[np.argmax(avg_probs)]
        else:
            avg_probs, avg_ent, avg_layer_avg, avg_peak_avg = None, np.nan, np.nan, np.nan
            
        # Correlation between the two definitions
        if (lbl_probs is not None) and (avg_probs is not None):
            if np.std(lbl_probs) > 0 and np.std(avg_probs) > 0:
                dist_corr, _ = stats.pearsonr(lbl_probs, avg_probs)
            else:
                dist_corr = 0.0
        else:
            dist_corr = np.nan
            
        if valid_members or (lbl_probs is not None):
            category_results.append({
                "category": category,
                "shannon_entropy": lbl_ent,
                "peak_layer": lbl_peak,
                "avg_layer": lbl_avg,
                "shannon_entropy_average": avg_ent,
                "peak_layer_average": avg_peak_avg,
                "avg_layer_average": avg_layer_avg,
                "pearson_correlation_distributions": dist_corr,
                "member_count": len(valid_members)
            })
            
    category_df = pd.DataFrame(category_results)
    category_df.to_csv(dist_dir / "shannon_entropy_categories.csv", index=False)

    return concept_df, category_df

def plot_peak_average_distributions(df, meta, concept_df, category_df, dist_dir):
    """
    Combines the peak layer bar plot with a faded KDE representing 
    the average layer distributions for Concepts, Categories, and Category Averages.
    """
    merged = format_layer_labels(df, meta)
    layer_labels = merged['layer_name'].cat.categories.tolist()
    
    concept_subset = concept_df[['concept', 'peak_layer', 'avg_layer']].rename(columns={'concept': 'name'}).copy()
    concept_subset['group_type'] = 'Specific Concepts'
    category_subset = category_df[['category', 'peak_layer', 'avg_layer']].rename(columns={'category': 'name'}).copy()
    category_subset['group_type'] = 'Broad Categories'
    category_avg_subset = category_df[['category', 'peak_layer_average', 'avg_layer_average']].rename(
        columns={'category': 'name', 'peak_layer_average': 'peak_layer', 'avg_layer_average': 'avg_layer'}).copy()
    category_avg_subset['group_type'] = 'Broad Categories (Avg)'
    
    # Concatenate all three
    plot_df = pd.concat([concept_subset, category_subset, category_avg_subset], ignore_index=True).dropna(subset=['peak_layer', 'avg_layer'])
    # Convert to 0-based index for correct plotting alignment
    plot_df['peak_layer_idx'] = plot_df['peak_layer'] - 1
    plot_df['avg_layer_idx'] = plot_df['avg_layer'] - 1
    
    fig, ax1 = plt.subplots(figsize=(18, 6))
    sns.set_theme(style="whitegrid")
    palette = {
        "Specific Concepts": "#D96A5B", 
        "Broad Categories": "#4B5A6A",
        "Broad Categories (Avg)": "#ECA926" 
    }

    group_counts = plot_df.groupby(['peak_layer_idx', 'group_type'], observed=False).size().reset_index(name='count')
    totals = plot_df['group_type'].value_counts()
    group_counts['percentage'] = group_counts.apply(
        lambda row: (row['count'] / totals[row['group_type']]) * 100, axis=1
    )
    sns.barplot(
        data=group_counts,
        x='peak_layer_idx', 
        y='percentage',
        hue='group_type',
        palette=palette,
        edgecolor='black',
        linewidth=0.8,
        ax=ax1,
        alpha=0.9,
        order=range(len(layer_labels)) 
    )
    # Format X and primary Y axes
    ax1.set_xticks(range(len(layer_labels)))
    ax1.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=9)
    ax1.set_xlabel("Model Layer", fontsize=11)
    ax1.set_ylabel("% of Group Peaking in this Layer (Bars)", fontsize=11)
    ax1.set_xlim(-0.5, len(layer_labels) - 0.5)
    ax2 = ax1.twinx() 
    sns.kdeplot(
        data=plot_df, 
        x='avg_layer_idx', 
        hue='group_type', 
        palette=palette, 
        fill=True, 
        alpha=0.2, 
        linewidth=2,
        ax=ax2,
        legend=False,
        common_norm=False, 
        clip=(-0.5, len(layer_labels) - 0.5)
    )
    ax2.set_xlim(-0.5, len(layer_labels) - 0.5)
    ax2.set_ylabel("Density of Average Layer (Curves)", fontsize=11)
    ax2.grid(False) 
    plt.title("Distribution of Peak Expert Layers vs. Average Layer Allocation", fontsize=14, pad=15)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, title="Concept Grouping", loc='upper right')
    
    plt.tight_layout()
    plt.savefig(dist_dir / "peak_average_layers.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def plot_shannon_entropies(concept_df, category_df, dist_dir):
    """
    Plots a violin plot comparing the Shannon entropy distributions
    of specific concepts versus broad categories to test the hypothesis
    of expert concentration.
    """
    concept_subset = concept_df[['concept', 'shannon_entropy']].rename(columns={'concept': 'name'}).copy()
    concept_subset['group_type'] = 'Specific Concepts'
    category_subset = category_df[['category', 'shannon_entropy']].rename(columns={'category': 'name'}).copy()
    category_subset['group_type'] = 'Broad Categories'
    plot_df = pd.concat([concept_subset, category_subset], ignore_index=True).dropna(subset=['shannon_entropy'])
    
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    palette = {"Specific Concepts": "#D96A5B", "Broad Categories": "#4B5A6A"}
    # Create the Violin Plot
    ax = sns.violinplot(
        data=plot_df, 
        x='group_type', 
        y='shannon_entropy', 
        hue='group_type',
        legend=False,
        palette=palette,
        inner='quartile', # Draws dashed lines at the 25th, 50th (median), and 75th percentiles
        linewidth=1.5,
        alpha=0.8
    )
    # Overlay a Strip Plot for individual data points
    sns.stripplot(
        data=plot_df, 
        x='group_type', 
        y='shannon_entropy', 
        color='black', 
        alpha=0.4, 
        size=4,
        jitter=True, # Spreads the dots horizontally so they don't overlap completely
        ax=ax
    )
    # Add text labels for the quartile dotted lines
    groups = ['Specific Concepts', 'Broad Categories']
    for i, group in enumerate(groups):
        group_data = plot_df[plot_df['group_type'] == group]['shannon_entropy']
        if not group_data.empty:
            q1 = group_data.quantile(0.25)
            median = group_data.quantile(0.50)
            q3 = group_data.quantile(0.75)
            bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
            ax.text(i + 0.15, q1, 'Q1 (25%)', va='center', ha='left', fontsize=9, color='#333333', bbox=bbox_props)
            ax.text(i + 0.15, median, 'Median', va='center', ha='left', fontsize=9, fontweight='bold', color='#333333', bbox=bbox_props)
            ax.text(i + 0.15, q3, 'Q3 (75%)', va='center', ha='left', fontsize=9, color='#333333', bbox=bbox_props)
    ax.set_xlabel("Concept Grouping", fontsize=12, labelpad=10)
    ax.set_ylabel("Shannon Entropy (Bits)", fontsize=12, labelpad=10)
    
    plt.title("Shannon Entropy of Expert Allocations: Categories vs. Concepts", fontsize=14, pad=15)
    
    plt.tight_layout()
    save_path = dist_dir / "category_concept_shannon_entropies.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# MODULE 3: CATEGORY-CONCEPT SIMILARITIES
# ==========================================

def plot_hierarchy_similarities(df, meta, sim_dir):
    """Calculates and plots Jaccard and Overlap similarity metrics."""
    unit_sets = df.groupby("concept")["unit"].apply(set).to_dict()
    
    results = []
    for _, row in meta.dropna(subset=["category"]).iterrows():
        concept, category = row["concept"], row["category"]
        u_concept = unit_sets.get(concept, set())  # Set A
        u_category = unit_sets.get(category, set()) # Set B
        
        if u_concept and u_category:
            intersection = len(u_concept.intersection(u_category))
            union = len(u_concept.union(u_category))
            len_a = len(u_concept)
            len_b = len(u_category)
            
            results.append({
                "hierarchy": f"{category} -> {concept}",
                "jaccard_pct": (intersection / union) * 100,
                "overlap_pct": (intersection / min(len_a, len_b)) * 100
            })
    
    res_df = pd.DataFrame(results)
    if res_df.empty: return
    res_df.to_csv(sim_dir / "category_concept_similarity_metrics.csv", index=False)

    # Generate the 2 distinct plots
    metrics = [
        ("jaccard_pct", "Jaccard Similarity Index % (Global Equivalence)", "#27ae60"),
        ("overlap_pct", "Overlap Coefficient % (Strict Subsetting)", "#8e44ad")
    ]
    
    for col, title, color in metrics:
        _plot_bar_with_leaders(res_df, "hierarchy", col, title, "Percentage %", 
                               sim_dir / f"{col.replace('_pct', '')}_hierarchy.png", color)


# ==========================================
# MODULE 4: CORRELATIONS
# ==========================================

def plot_correlations(merged_meta, corr_dir):
    """Plot correlations for freq vs count, typ vs count, and freq vs typ."""
    
    # Frequency vs Expert Count
    freq_data = merged_meta.dropna(subset=['log_frequency', 'expert_count'])
    plt.figure(figsize=(10.4, 9.1))
    if len(freq_data) > 2:
        r_f, p_f = stats.pearsonr(freq_data['log_frequency'], freq_data['expert_count'])
        sns.regplot(data=freq_data, x="log_frequency", y="expert_count", scatter_kws={'color': '#444e86'}, line_kws={'color':'red'})
        plt.title(f"Frequency(Log10) vs Expert Count (r={r_f:.2f}, p={p_f:.2e})", fontsize=16)
    plt.xlabel("Wikipedia Frequency(Log10)"); plt.ylabel("Expert Count")
    plt.tight_layout(); plt.savefig(corr_dir / "frequency_vs_expert_count.png", dpi=300); plt.close()

    # Typicality vs Expert Count
    typ_data = merged_meta.dropna(subset=['typicality', 'expert_count'])
    plt.figure(figsize=(10.4, 9.1))
    if len(typ_data) > 2:
        r_t, p_t = stats.pearsonr(typ_data['typicality'], typ_data['expert_count'])
        sns.regplot(data=typ_data, x="typicality", y="expert_count", scatter_kws={'color': '#955196'}, line_kws={'color':'red'})
        plt.title(f"Typicality vs Expert Count (r={r_t:.2f}, p={p_t:.2e})", fontsize=16)
    plt.xlabel("Typicality"); plt.ylabel("Expert Count")
    plt.tight_layout(); plt.savefig(corr_dir / "typicality_vs_expert_count.png", dpi=300); plt.close()

    # Frequency vs Typicality
    ft_data = merged_meta.dropna(subset=['log_frequency', 'typicality'])
    plt.figure(figsize=(10.4, 9.1))
    if len(ft_data) > 2:
        r_ft, p_ft = stats.pearsonr(ft_data['log_frequency'], ft_data['typicality'])
        sns.regplot(data=ft_data, x="log_frequency", y="typicality", scatter_kws={'color': '#2ca02c'}, line_kws={'color':'red'})
        plt.title(f"Frequency(Log10) vs Typicality (r={r_ft:.2f}, p={p_ft:.2e})", fontsize=16)
    plt.xlabel("Wikipedia Frequency(Log10)"); plt.ylabel("Typicality")
    plt.tight_layout(); plt.savefig(corr_dir / "frequency_vs_typicality.png", dpi=300); plt.close()


# ==========================================
# MODULE 5: HEATMAPS
# ==========================================

def plot_all_heatmaps(df, meta, heat_dir):
    """Calculates Jaccard and Overlap metrics globally and generates CSVs and heatmaps."""
    concepts = meta['concept'].unique()
    
    presence_df = df.assign(present=1).pivot_table(index='concept', columns='unit', values='present', fill_value=0)
    presence_df = presence_df.reindex(concepts, fill_value=0)
    A = presence_df.values  
    
    # Set sizes and operations
    intersection = np.dot(A, A.T) 
    sizes = A.sum(axis=1)         
    union = sizes[:, None] + sizes[None, :] - intersection
    min_size = np.minimum(sizes[:, None], sizes[None, :])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_mtx = np.where(union > 0, (intersection / union) * 100, 0.0)
        overlap_mtx = np.where(min_size > 0, (intersection / min_size) * 100, 0.0)

    # Dictionary of metrics to streamline saving and plotting
    matrices = {
        "jaccard": (jaccard_mtx, "Pairwise Jaccard Similarity Index %", "magma"),
        "overlap": (overlap_mtx, "Pairwise Overlap Coefficient %", "magma")
    }

    for name, (mtx, title, cmap) in matrices.items():
        pd.DataFrame(mtx, index=concepts, columns=concepts).to_csv(heat_dir / f"{name}_matrix.csv")
        _plot_heatmap_with_leaders(mtx, concepts, title, heat_dir / f"{name}_heatmap.png", cmap)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    SOURCE_DIR = "../results/GPT2_abstractiveness_150_responses/" 
    TARGET_MODEL = "gpt2"
    OUTPUT_ROOT = pathlib.Path("../results/research_plots_150_5_test_peak_with_3_colours_violin_plot_nicely_formatted")
    
    METADATA = pd.read_json("../assets/metadata.json")
    
    for ap in [0.6]:
        out_path = OUTPUT_ROOT / f"AP_{ap}"
        
        set_folder_log(out_path)
        
        # Build strict directory structure
        dist_dir = out_path / "layer_expert_distribution"
        concept_dir = dist_dir / "per_concept"
        sim_dir = out_path / "category_concept_similarities"
        corr_dir = out_path / "correlations"
        heat_dir = out_path / "heatmaps"
        
        for d in [concept_dir, sim_dir, corr_dir, heat_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        log.info(f"Starting Refactored Analysis Suite for AP Threshold: {ap}")
        df_experts = load_experts_data(SOURCE_DIR, TARGET_MODEL, ap)
        
        if not df_experts.empty:
            formatted_base_df = format_layer_labels(df_experts, METADATA)
            save_layer_mapping_csv(formatted_base_df, "../assets/layer_mapping_reference.csv")
            # Module 1: Layer Expert Distribution
            log.info("  Generating combined metadata counts...")
            merged_meta = save_expert_counts_metadata(df_experts, METADATA, dist_dir)
            log.info("  Generating expert layer distributions...")
            plot_layer_distribution(df_experts, METADATA, dist_dir)
            # Module 2: Shannon Entropy Analysis and Peak/Average Layer Distributions
            log.info("  Computing and plotting Shannon entropy analysis...")
            concept_df, category_df = compute_shannon_entropy(df_experts, METADATA, dist_dir)      
            log.info("  Generating peak vs. average layer distributions plot...")
            plot_peak_average_distributions(df_experts, METADATA, concept_df, category_df, dist_dir)
            log.info("  Generating Shannon entropy violin plots...")
            plot_shannon_entropies(concept_df, category_df, dist_dir)
            # Module 3: Category-Concept Similarities
            log.info("  Generating hierarchy similarities (Jaccard & Overlap)...")
            plot_hierarchy_similarities(df_experts, METADATA, sim_dir)
            # Module 4: Correlations
            log.info("  Generating correlation plots...")
            plot_correlations(merged_meta, corr_dir)
            # Module 5: Heatmaps
            log.info("  Generating all pairwise heatmaps and CSV matrices (Jaccard & Overlap)...")
            plot_all_heatmaps(df_experts, METADATA, heat_dir)
        
            log.info(f"  All analysis outputs cleanly structured in {out_path}")
        else:
            log.warning(f"No experts found for threshold {ap}. Skipping folder.")