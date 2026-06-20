import logging
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 
from scipy.stats import entropy, spearmanr, zscore, pearsonr
from scipy.spatial.distance import pdist, squareform, jensenshannon
from sklearn.metrics.pairwise import cosine_similarity

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
    df['unit_weight_pct'] = 100 / df['concept'].map(counts)
    return df

def save_layer_mapping_csv(formatted_df, out_path):
    """Saves a reference dictionary linking original model layers to the new 1-48 sequence labels."""
    mapping_df = formatted_df[['layer', 'layer_idx', 'layer_name']].drop_duplicates()
    mapping_df = mapping_df.sort_values('layer_idx').reset_index(drop=True)
    mapping_df.to_csv(out_path, index=False)


# ==========================================
# PLOTTING HELPERS
# ==========================================

def _plot_bar_with_leaders(df, x_col, y_col, title, y_label, out_path, color, x_label=None, figsize=(40.56, 10.14), orient='v'):
    """Reusable helper to plot a bar chart with dynamic sizing and orientation-aware leader lines."""
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df, x=x_col, y=y_col, color=color, edgecolor="black", orient=orient)
    ax.set_title(title, fontsize=18)

    if orient == 'v':
        ax.set_xticklabels([])
        if x_label: ax.set_xlabel(x_label, fontsize=14)
        else: ax.set_xlabel("")
        if y_label: ax.set_ylabel(y_label, fontsize=14)
        else: ax.set_ylabel("")
            
        for i, bar in enumerate(ax.patches):
            bar_x = bar.get_x() + (bar.get_width() / 2)
            label = df[x_col].iloc[i] # The categorical label is on the X axis
            
            ax.annotate("", xy=(bar_x, 0), xytext=(0, -10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="black", linewidth=1.0, shrinkA=0, shrinkB=0))
            ax.annotate(label, xy=(bar_x, 0), xytext=(2, -8), textcoords="offset points",
                        ha='right', va='top', rotation=45, fontsize=10)
        plt.subplots_adjust(bottom=0.25)

    elif orient == 'h':
        ax.set_yticklabels([])
        if x_label: ax.set_xlabel(x_label, fontsize=14)
        else: ax.set_xlabel("")
        if y_label: ax.set_ylabel(y_label, fontsize=14)
        else: ax.set_ylabel("")
        
        for i, bar in enumerate(ax.patches):
            bar_y = bar.get_y() + (bar.get_height() / 2)
            label = df[y_col].iloc[i] # The categorical label is on the Y axis
            
            ax.annotate("", xy=(0, bar_y), xytext=(-10, 0), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="black", linewidth=1.0, shrinkA=0, shrinkB=0))
            ax.annotate(label, xy=(0, bar_y), xytext=(-12, 0), textcoords="offset points",
                        ha='right', va='center', fontsize=12)
        plt.subplots_adjust(left=0.25)

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
    stats_df['avg_concept_weight'] = stats_df['unit_weight_pct'] / stats_df['abstraction_level'].map(concepts_per_level)

    stats_df.to_csv(dist_dir / "mean_expert_layer_distribution.csv", index=False)

    plt.figure(figsize=(31.2, 10.4))
    sns.barplot(data=stats_df, x="layer_name", y="avg_concept_weight", hue="abstraction_level", 
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

def compute_shannon_entropy(df, meta, shan_dir):
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
    concept_df.to_csv(shan_dir / "shannon_entropy_concepts.csv", index=False)
    
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
    category_df.to_csv(shan_dir / "shannon_entropy_categories.csv", index=False)
    
    # Statistical Test: Mann-Whitney U Test to compare the distributions of Shannon entropy between Concepts and Categories
    clean_concept_ent = concept_df['shannon_entropy'].dropna()
    clean_category_ent = category_df['shannon_entropy'].dropna()
    
    if len(clean_category_ent) > 0 and len(clean_concept_ent) > 0:
        u_stat, p_val = stats.mannwhitneyu(clean_category_ent, clean_concept_ent, alternative='less')
        log.info(f"  [Stats] Mann-Whitney U Test (Categories < Concepts): U={u_stat:.1f}, p-value={p_val:.4e}")
        if p_val < 0.05:
            log.info("  [Stats] -> Hypothesis CONFIRMED: Categories are significantly more concentrated.")
        else:
            log.info("  [Stats] -> Hypothesis REJECTED: Difference is not statistically significant.")
    else:
        log.warning("  [Stats] Insufficient valid entropy data to perform Mann-Whitney U Test.")

    return concept_df, category_df

def plot_peak_average_distributions(df, meta, concept_df, category_df, shan_dir):
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
    plt.savefig(shan_dir / "peak_average_layers.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def plot_shannon_entropies(concept_df, category_df, shan_dir):
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
    save_path = shan_dir / "category_concept_shannon_entropies.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_category_entropies_bar(category_df, shan_dir):
    """
    Plots a bar chart of Shannon Entropy for each category label.
    Sorted to clearly show which categories are most localized (lowest entropy)
    vs most distributed (highest entropy).
    """
    df = category_df.dropna(subset=['shannon_entropy']).copy()
    df = df.sort_values(by='shannon_entropy', ascending=False).reset_index(drop=True)
    df['category_display'] = df['category'].str.title()
    out_path = shan_dir / "category_shannon_entropies_bar.png"
    _plot_bar_with_leaders(
        df=df,
        x_col="shannon_entropy",
        y_col="category_display",
        title="Shannon Entropy of Expert Allocations by Category",
        y_label=None,
        out_path=out_path,
        color="#4B5A6A",
        x_label="Shannon Entropy (Bits)",
        figsize=(12, 14),
        orient='h'  
    )


# ==========================================
# MODULE 3: CATEGORY-CONCEPT SIMILARITIES
# ==========================================

def plot_hierarchy_similarities(df, meta, sim_dir):
    """Calculates, plots, and returns Jaccard and Overlap similarity metrics."""
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
                "concept": concept,
                "category": category,
                "hierarchy": f"{category} -> {concept}",
                "jaccard_pct": (intersection / union) * 100,
                "overlap_pct": (intersection / min(len_a, len_b)) * 100
            })
    
    res_df = pd.DataFrame(results)
    if res_df.empty: return res_df
    
    res_df.to_csv(sim_dir / "category_concept_similarity_metrics.csv", index=False)

    # Generate the 2 distinct plots
    metrics = [
        ("jaccard_pct", "Jaccard Similarity Index % (Global Equivalence)", "#27ae60"),
        ("overlap_pct", "Overlap Coefficient % (Strict Subsetting)", "#8e44ad")
    ]
    
    for col, title, color in metrics:
        _plot_bar_with_leaders(res_df, "hierarchy", col, title, "Percentage %", 
                               sim_dir / f"{col.replace('_pct', '')}_hierarchy.png", color)
                               
    return res_df

# ==========================================
# MODULE 4: CORRELATIONS
# ==========================================

def plot_correlations(merged_meta, sim_df, corr_dir):
    """Plot correlations for freq vs count, typ vs count, freq vs typ, and similarities."""
    # 1. Frequency vs Expert Count
    freq_data = merged_meta.dropna(subset=['log_frequency', 'expert_count'])
    plt.figure(figsize=(10.4, 9.1))
    if len(freq_data) > 2:
        r_f, p_f = stats.pearsonr(freq_data['log_frequency'], freq_data['expert_count'])
        sns.regplot(data=freq_data, x="log_frequency", y="expert_count", scatter_kws={'color': '#444e86'}, line_kws={'color':'red'})
        plt.title(f"Frequency(Log10) vs Expert Count (r={r_f:.2f}, p={p_f:.2e})", fontsize=16)
    plt.xlabel("Wikipedia Frequency(Log10)"); plt.ylabel("Expert Count")
    plt.tight_layout(); plt.savefig(corr_dir / "frequency_vs_expert_count.png", dpi=300); plt.close()

    # 2. Typicality vs Expert Count
    typ_data = merged_meta.dropna(subset=['typicality', 'expert_count'])
    plt.figure(figsize=(10.4, 9.1))
    if len(typ_data) > 2:
        r_t, p_t = stats.pearsonr(typ_data['typicality'], typ_data['expert_count'])
        sns.regplot(data=typ_data, x="typicality", y="expert_count", scatter_kws={'color': '#955196'}, line_kws={'color':'red'})
        plt.title(f"Typicality vs Expert Count (r={r_t:.2f}, p={p_t:.2e})", fontsize=16)
    plt.xlabel("Typicality"); plt.ylabel("Expert Count")
    plt.tight_layout(); plt.savefig(corr_dir / "typicality_vs_expert_count.png", dpi=300); plt.close()

    # 3. Frequency vs Typicality
    ft_data = merged_meta.dropna(subset=['log_frequency', 'typicality'])
    plt.figure(figsize=(10.4, 9.1))
    if len(ft_data) > 2:
        r_ft, p_ft = stats.pearsonr(ft_data['log_frequency'], ft_data['typicality'])
        sns.regplot(data=ft_data, x="log_frequency", y="typicality", scatter_kws={'color': '#2ca02c'}, line_kws={'color':'red'})
        plt.title(f"Frequency(Log10) vs Typicality (r={r_ft:.2f}, p={p_ft:.2e})", fontsize=16)
    plt.xlabel("Wikipedia Frequency(Log10)"); plt.ylabel("Typicality")
    plt.tight_layout(); plt.savefig(corr_dir / "frequency_vs_typicality.png", dpi=300); plt.close()

    if sim_df is not None and not sim_df.empty:
        # Merge the metadata with the similarity dataframe
        corr_data = merged_meta.merge(sim_df, on=["concept", "category"], how="inner")
        
        # 4. Typicality vs Jaccard Similarity
        typ_jac_data = corr_data.dropna(subset=['typicality', 'jaccard_pct'])
        plt.figure(figsize=(10.4, 9.1))
        if len(typ_jac_data) > 2:
            r_tj, p_tj = stats.pearsonr(typ_jac_data['typicality'], typ_jac_data['jaccard_pct'])
            sns.regplot(data=typ_jac_data, x="typicality", y="jaccard_pct", scatter_kws={'color': '#ff7c43'}, line_kws={'color':'red'})
            plt.title(f"Typicality vs Jaccard Similarity % (r={r_tj:.2f}, p={p_tj:.2e})", fontsize=16)
        plt.xlabel("Typicality"); plt.ylabel("Jaccard Similarity Index %")
        plt.tight_layout(); plt.savefig(corr_dir / "typicality_vs_jaccard.png", dpi=300); plt.close()

        # 5. Frequency vs Jaccard Similarity
        freq_jac_data = corr_data.dropna(subset=['log_frequency', 'jaccard_pct'])
        plt.figure(figsize=(10.4, 9.1))
        if len(freq_jac_data) > 2:
            r_fj, p_fj = stats.pearsonr(freq_jac_data['log_frequency'], freq_jac_data['jaccard_pct'])
            sns.regplot(data=freq_jac_data, x="log_frequency", y="jaccard_pct", scatter_kws={'color': '#ffa600'}, line_kws={'color':'red'})
            plt.title(f"Frequency(Log10) vs Jaccard Similarity % (r={r_fj:.2f}, p={p_fj:.2e})", fontsize=16)
        plt.xlabel("Wikipedia Frequency(Log10)"); plt.ylabel("Jaccard Similarity Index %")
        plt.tight_layout(); plt.savefig(corr_dir / "frequency_vs_jaccard.png", dpi=300); plt.close()


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
# MODULE 6: DUAL CATEGORY DEFINITIONS (JSD)
# ==========================================

def compute_dual_category_jsd(df, meta, jsd_dir):
    """
    Computes Jensen-Shannon Divergence and gathers all necessary metadata
    (Entropy, Typicality, Probability Distributions) for Macro/Micro plotting.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import entropy
    from scipy.spatial.distance import jensenshannon
    
    merged = format_layer_labels(df, meta)
    layer_labels = merged['layer_name'].cat.categories.tolist()
    
    dist_matrix = merged.groupby(['concept', 'layer_idx'], observed=False).size().unstack(fill_value=0)
    prob_matrix = dist_matrix.div(dist_matrix.sum(axis=1), axis=0)
    
    results = []
    
    for category in meta['category'].dropna().unique():
        cat_meta = meta[meta['category'] == category]
        members = cat_meta['concept'].tolist()
        valid_members = [m for m in members if m in prob_matrix.index]
        
        if category in prob_matrix.index and valid_members:
            # P: Category Label distribution (Prototype)
            P = prob_matrix.loc[category].values
            
            # Q: Averaged Member distribution (Exemplars)
            Q = prob_matrix.loc[valid_members].mean(axis=0).values
            
            # JSD Math
            js_distance = jensenshannon(P, Q, base=2)
            jsd = js_distance ** 2
            
            # Additional Metrics for Visualizations
            p_ent = entropy(P, base=2)
            valid_meta = cat_meta[cat_meta['concept'].isin(valid_members)]
            avg_typ = valid_meta['typicality'].mean() if 'typicality' in valid_meta.columns else np.nan
            
            results.append({
                "category": category,
                "member_count": len(valid_members),
                "avg_typicality": avg_typ,
                "shannon_entropy_label": p_ent,
                "jensen_shannon_divergence": jsd,
                "P_dist": P,
                "Q_dist": Q 
            })
            
    jsd_df = pd.DataFrame(results)
    
    if not jsd_df.empty:
        jsd_df = jsd_df.sort_values(by="jensen_shannon_divergence").reset_index(drop=True)
        
        # Save a clean CSV without the massive array columns
        clean_csv = jsd_df.drop(columns=['P_dist', 'Q_dist'])
        clean_csv.to_csv(jsd_dir / "dual_category_jsd.csv", index=False)
    
    # Return both the dataframe and the layer labels for the X-axis of the micro plot
    return jsd_df, layer_labels

def plot_jsd_macro_lollipop(jsd_df, jsd_dir):
    """Visualization 1: Macro View of Divergence across all categories."""
    df = jsd_df.dropna(subset=['jensen_shannon_divergence']).copy()
    df = df.sort_values('jensen_shannon_divergence', ascending=True).reset_index(drop=True)
    df['category_display'] = df['category'].str.title()
    
    plt.figure(figsize=(10, 12))
    
    # Draw the lines
    plt.hlines(y=df['category_display'], xmin=0, xmax=df['jensen_shannon_divergence'], 
               color='gray', alpha=0.5, linewidth=1.5)
    
    # Draw the points, colored by typicality
    scatter = plt.scatter(df['jensen_shannon_divergence'], df['category_display'], 
                          c=df['avg_typicality'], cmap='viridis', 
                          s=100, edgecolor='black', alpha=0.9, zorder=3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Average Member Typicality', fontsize=12)
    
    plt.title("Jensen-Shannon Divergence: Prototype vs. Exemplar Models", fontsize=16, pad=15)
    plt.xlabel("Jensen-Shannon Divergence (Bits)\n← High Overlap (Prototype = Exemplars) | Low Overlap (Prototype ≠ Exemplars) →", fontsize=12)
    plt.ylabel("")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(jsd_dir / "macro_jsd_lollipop.png", dpi=300)
    plt.close()


def plot_jsd_micro_distributions(jsd_df, layer_labels, jsd_dir):
    """Visualization 2: Overlaid Area Charts for the Lowest and Highest JSD categories."""
    if len(jsd_df) < 2: return
    
    # Find the extremes
    min_cat = jsd_df.loc[jsd_df['jensen_shannon_divergence'].idxmin()]
    max_cat = jsd_df.loc[jsd_df['jensen_shannon_divergence'].idxmax()]
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    x = np.arange(len(layer_labels))
    
    targets = [
        (axes[0], min_cat, "Lowest Divergence (Highly Aligned)"),
        (axes[1], max_cat, "Highest Divergence (Highly Separated)")
    ]
    
    for ax, data, subtitle in targets:
        # Trace P (Prototype / Label)
        ax.fill_between(x, data['P_dist'], color="#D96A5B", alpha=0.4, label="Prototype (Category Label)")
        ax.plot(x, data['P_dist'], color="#D96A5B", linewidth=2)
        
        # Trace Q (Exemplar / Average of Members)
        ax.fill_between(x, data['Q_dist'], color="#4B5A6A", alpha=0.4, label="Exemplars (Avg. of Members)")
        ax.plot(x, data['Q_dist'], color="#4B5A6A", linewidth=2)
        
        cat_name = data['category'].title()
        jsd_val = data['jensen_shannon_divergence']
        
        ax.set_title(f"Category: '{cat_name}' | {subtitle} | JSD: {jsd_val:.4f}", fontsize=14)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.legend(loc="upper right")
        ax.set_xlim(0, len(layer_labels) - 1)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Format the shared X-axis
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=9)
    axes[1].set_xlabel("Model Layer", fontsize=12)
    
    plt.suptitle("Micro View: Layer Allocations Behind Jensen-Shannon Divergence", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(jsd_dir / "micro_jsd_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_jsd_vs_entropy_scatter(jsd_df, jsd_dir):
    """Visualization 3: Scatter plot checking if concentrated concepts diverge more."""
    df = jsd_df.dropna(subset=['jensen_shannon_divergence', 'shannon_entropy_label']).copy()
    
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation for the title
    if len(df) > 2:
        r, p = stats.pearsonr(df['shannon_entropy_label'], df['jensen_shannon_divergence'])
        title_suffix = f"(r={r:.2f}, p={p:.2e})"
    else:
        title_suffix = ""

    # regplot automatically adds a line of best fit and confidence intervals
    sns.regplot(
        data=df, 
        x="shannon_entropy_label", 
        y="jensen_shannon_divergence", 
        scatter_kws={'color': '#2ca02c', 's': 70, 'edgecolor': 'black', 'alpha': 0.8}, 
        line_kws={'color':'#d62728', 'linewidth': 2}
    )
    
    # Add category text labels to the points so you know who is who
    for i in range(df.shape[0]):
        plt.text(
            df['shannon_entropy_label'].iloc[i] + 0.02, 
            df['jensen_shannon_divergence'].iloc[i], 
            df['category'].iloc[i], 
            horizontalalignment='left', size='small', color='black', alpha=0.7
        )
        
    plt.title(f"Label Entropy vs. Prototype/Exemplar Divergence {title_suffix}", fontsize=15, pad=15)
    plt.xlabel("Shannon Entropy of Category Label (Bits)\n← Concentrated | Uniform →", fontsize=12)
    plt.ylabel("Jensen-Shannon Divergence (Bits)\n← Aligned with Members | Diverged from Members →", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(jsd_dir / "scatter_jsd_vs_entropy.png", dpi=300)
    plt.close()

# ==========================================
# MODULE 7: EMPIRICAL COSINE TYPICALITY
# ==========================================

def compute_empirical_typicality(df, meta, typ_dir)-> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes Centroid-based cosine typicality of expert allocations.
    Calculates both a Global Prototype and Per-Layer Prototypes.
    """ 
    merged = format_layer_labels(df, meta)
    merged['present'] = 1 
    ordered_layers = merged[['layer_idx', 'layer_name']].drop_duplicates().sort_values('layer_idx')
    global_results = []
    layer_results = []
    
    for category in meta['category'].dropna().unique():
        cat_meta = meta[meta['category'] == category]
        members = cat_meta['concept'].tolist()
        valid_members = [m for m in members if m in merged['concept'].values]
        if len(valid_members) < 2:
            continue  
        cat_df = merged[merged['concept'].isin(valid_members)].copy()
        
        # Global Prototype (Averaged space across ALL layers)
        cat_df['layer_unit'] = cat_df['layer_name'].astype(str) + "_" + cat_df['unit'].astype(str)
        global_pivot = cat_df.pivot_table(index='concept', columns='layer_unit', values='present', fill_value=0)
        global_pivot = global_pivot.reindex(valid_members, fill_value=0)
        
        if not global_pivot.empty:
            # Calculate Global Prototype (Centroid)
            centroid_global = global_pivot.mean(axis=0).values.reshape(1, -1)
            # Compare each member to the Prototype
            sims_global = cosine_similarity(global_pivot.values, centroid_global).flatten()
            
            for i, concept in enumerate(global_pivot.index):
                h_typ_arr = cat_meta[cat_meta['concept'] == concept]['typicality'].values
                h_typ = h_typ_arr[0] if len(h_typ_arr) > 0 else np.nan
                global_results.append({
                    'category': category,
                    'concept': concept,
                    'human_typicality': h_typ,
                    'global_cosine_typicality': sims_global[i]
                })

        # Per-Layer Prototype 
        for _, row in ordered_layers.iterrows():
            l_idx = row['layer_idx']
            l_name = row['layer_name']
            layer_df_sub = cat_df[cat_df['layer_idx'] == l_idx]
            layer_pivot = layer_df_sub.pivot_table(index='concept', columns='unit', values='present', fill_value=0)
            layer_pivot = layer_pivot.reindex(valid_members, fill_value=0)
            
            if layer_pivot.shape[1] > 0:
                centroid_layer = layer_pivot.mean(axis=0).values.reshape(1, -1)
                sims_layer = cosine_similarity(layer_pivot.values, centroid_layer).flatten()
            else:
                sims_layer = np.zeros(len(valid_members))
                
            for i, concept in enumerate(layer_pivot.index):
                h_typ_arr = cat_meta[cat_meta['concept'] == concept]['typicality'].values
                h_typ = h_typ_arr[0] if len(h_typ_arr) > 0 else np.nan
                layer_results.append({
                    'category': category,
                    'concept': concept,
                    'layer_idx': l_idx,
                    'layer_name': l_name,
                    'human_typicality': h_typ,
                    'layer_cosine_typicality': sims_layer[i]
                })
    global_df = pd.DataFrame(global_results)
    layer_df = pd.DataFrame(layer_results)
    global_df.to_csv(typ_dir / "global_prototype_typicality.csv", index=False)
    layer_df.to_csv(typ_dir / "layer_prototype_typicality.csv", index=False)
    return global_df, layer_df

def generate_category_typicality_reports(global_df, layer_df, typ_dir):
    """
    Creates isolated reports for each category, outputting double bar plots,
    correlation scatter plots, and a CSV tracking the most typical word per layer.
    """
    for category in global_df['category'].unique():
        cat_dir = typ_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        cat_global = global_df[global_df['category'] == category].dropna(subset=['human_typicality', 'global_cosine_typicality']).copy()
        
        if len(cat_global) > 1:
            # Sort by human typicality to make the bars slope downwards cleanly
            cat_global = cat_global.sort_values('human_typicality', ascending=False)
            melted_df = cat_global.melt(
                id_vars=['concept'],
                value_vars=['human_typicality', 'global_cosine_typicality'],
                var_name='Metric', value_name='Score'
            )
            melted_df['Metric'] = melted_df['Metric'].replace({
                'human_typicality': 'Human Rating (Hard-coded)',
                'global_cosine_typicality': 'Model Prototype (Cosine Sim)'
            })

            plt.figure(figsize=(14, 6))
            sns.barplot(
                data=melted_df, x='concept', y='Score', hue='Metric', 
                palette=['#4B5A6A', '#D96A5B'], edgecolor="black"
            )
            plt.title(f"'{category.title()}' Typicality: Human vs. Model Prototype", fontsize=16)
            plt.xlabel("Concepts (Ordered by Human Typicality)", fontsize=12)
            plt.ylabel("Typicality Score (0 to 1)", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Metric")
            plt.tight_layout()
            plt.savefig(cat_dir / f"{category}_typicality_comparison_bar.png", dpi=300)
            plt.close()
            plt.figure(figsize=(8, 6))
            
            # Catch constant array warnings safely
            if cat_global['human_typicality'].std() > 0 and cat_global['global_cosine_typicality'].std() > 0:
                r, p = pearsonr(cat_global['human_typicality'], cat_global['global_cosine_typicality'])
                title_str = f"'{category.title()}' Typicality Correlation\nPearson r = {r:.2f} (p={p:.3f})"
            else:
                title_str = f"'{category.title()}' Typicality Correlation\n(Insufficient variance for correlation)" 
            sns.regplot(
                data=cat_global, x='human_typicality', y='global_cosine_typicality', 
                color='#d62728', scatter_kws={'s': 80, 'edgecolor': 'black'}
            )

            # Annotate each dot with the specific word
            for i in range(len(cat_global)):
                plt.text(
                    cat_global['human_typicality'].iloc[i],
                    cat_global['global_cosine_typicality'].iloc[i],
                    f"  {cat_global['concept'].iloc[i]}",
                    fontsize=9, alpha=0.8, va='center'
                )
            plt.title(title_str, fontsize=14)
            plt.xlabel("Human Typicality (Hard-coded)", fontsize=12)
            plt.ylabel("Model Prototype Typicality (Cosine Sim)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(cat_dir / f"{category}_typicality_correlation_scatter.png", dpi=300)
            plt.close()

        # Most Typical Word Per Layer
        cat_layer = layer_df[layer_df['category'] == category].copy()
        if not cat_layer.empty:
            # Find the row index of the max typicality score for each layer
            idx = cat_layer.groupby('layer_idx')['layer_cosine_typicality'].idxmax()
            top_per_layer = cat_layer.loc[idx, ['layer_idx', 'layer_name', 'concept', 'layer_cosine_typicality', 'human_typicality']]
            top_per_layer.rename(columns={
                'concept': 'most_typical_model_concept',
                'layer_cosine_typicality': 'cosine_similarity_score',
                'human_typicality': 'hardcoded_human_score'
            }, inplace=True)
            top_per_layer = top_per_layer.sort_values('layer_idx')
            top_per_layer.to_csv(cat_dir / f"{category}_most_typical_per_layer.csv", index=False)
            plot_most_typical_concept_per_layer(top_per_layer, category, cat_dir)
            
def plot_most_typical_concept_per_layer(df, category, cat_dir):
    """
    Plots the evolution of the most typical concept across layers.
    Uses a stem-like visual so the word itself acts as the 'bar'.
    """
    plt.figure(figsize=(24, 8))
    plt.vlines(x=range(len(df)), ymin=0, ymax=df['cosine_similarity_score'], 
               color='gray', alpha=0.3, linewidth=2)
    for i, row in df.reset_index(drop=True).iterrows():
        plt.scatter(i, row['cosine_similarity_score'], color='#4B5A6A', s=20, zorder=3)
        plt.text(
            i, row['cosine_similarity_score'] + 0.02, # slight vertical offset
            row['most_typical_model_concept'].title(), 
            color='#D96A5B', ha='left', va='bottom', 
            rotation=60, fontsize=12, fontweight='bold'
        )
        
    plt.title(f"'{category.title()}' Evolution: The Most Typical Concept per Layer", fontsize=18, pad=20)
    plt.xlabel("Model Layer", fontsize=14)
    plt.ylabel("Cosine Similarity to Layer Prototype", fontsize=14)
    plt.xticks(ticks=range(len(df)), labels=df['layer_name'], rotation=45, ha='right', fontsize=9)
    
    # Dynamically adjust Y limit to ensure long words don't get cut off at the top
    max_y = df['cosine_similarity_score'].max()
    plt.ylim(0, max_y + 0.3)
    plt.xlim(-1, len(df))
    
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(cat_dir / f"{category}_most_typical_evolution.png", dpi=300)
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    SOURCE_DIR = "../results/GPT2_abstractiveness_150_responses/" 
    TARGET_MODEL = "gpt2"
    OUTPUT_ROOT = pathlib.Path("../results/research_plots_150_revised")
    
    METADATA = pd.read_json("../assets/metadata.json")
    
    for ap in [0.5, 0.6, 0.7, 0.8, 0.9]:
        out_path = OUTPUT_ROOT / f"AP_{ap}"
        
        set_folder_log(out_path)
        
        # Build strict directory structure
        dist_dir = out_path / "layer_expert_distribution"
        concept_dir = dist_dir / "per_concept"
        shan_dir = out_path / "shannon_entropy_analysis"
        sim_dir = out_path / "category_concept_similarities"
        corr_dir = out_path / "correlations"
        heat_dir = out_path / "heatmaps"
        jsd_dir = out_path / "dual_category_jsd"
        typ_dir = out_path / "typicality_analysis"
        
        for d in [concept_dir, shan_dir, sim_dir, corr_dir, heat_dir, jsd_dir, typ_dir]:
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
            concept_df, category_df = compute_shannon_entropy(df_experts, METADATA, shan_dir)      
            log.info("  Generating peak vs. average layer distributions plot...")
            plot_peak_average_distributions(df_experts, METADATA, concept_df, category_df, shan_dir)
            log.info("  Generating Shannon entropy violin plots...")
            plot_shannon_entropies(concept_df, category_df, shan_dir)
            log.info("  Generating individual category entropy bar chart...")
            plot_category_entropies_bar(category_df, shan_dir)
            # Module 3: Category-Concept Similarities
            log.info("  Generating hierarchy similarities (Jaccard & Overlap)...")
            sim_df =plot_hierarchy_similarities(df_experts, METADATA, sim_dir)
            # Module 4: Correlations
            log.info("  Generating correlation plots...")
            plot_correlations(merged_meta, sim_df, corr_dir)
            # Module 5: Heatmaps
            log.info("  Generating all pairwise heatmaps and CSV matrices (Jaccard & Overlap)...")
            plot_all_heatmaps(df_experts, METADATA, heat_dir)
            # Module 6: Dual Category Definitions (JSD)
            log.info("  Computing Dual Category Definitions (JSD)...")
            jsd_df, jsd_layer_labels = compute_dual_category_jsd(df_experts, METADATA, jsd_dir)
            log.info("  Generating JSD visualisations...")
            plot_jsd_macro_lollipop(jsd_df, jsd_dir)
            plot_jsd_micro_distributions(jsd_df, jsd_layer_labels, jsd_dir)
            plot_jsd_vs_entropy_scatter(jsd_df, jsd_dir)
            # Module 7: Empirical Cosine Typicality
            log.info("  Computing Empirical Cosine Typicality...")
            typ_global_df, typ_layer_df = compute_empirical_typicality(df_experts, METADATA, typ_dir)
            generate_category_typicality_reports(typ_global_df, typ_layer_df, typ_dir)
                
            log.info(f"  All analysis outputs cleanly structured in {out_path}")
        else:
            log.warning(f"No experts found for threshold {ap}. Skipping folder.")
            