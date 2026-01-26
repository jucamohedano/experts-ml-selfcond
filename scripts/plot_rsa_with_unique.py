
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from datetime import datetime
import re
import scipy.stats as stats

# Load all results
base_dir = pathlib.Path('results/rsa/Qwen3-30B-A3B-Instruct-2507_custom_60/Qwen3-30B-A3B-Instruct-2507_gpt2')
layer_components = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
results = {comp: {} for comp in layer_components}

# Regex to parse directory names
# Matches: cot-ap-0.5_vs_full_attn.c_attn
# Matches: cot-ap-0.5-unique-corr_0.8_vs_full_attn.c_attn
regex_pattern = re.compile(r'cot-ap-([\d\.]+)(?:-unique-corr_([\d\.]+))?_vs_full_(.+)')

# Collect all directories first
all_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

for d in all_dirs:
    match = regex_pattern.match(d.name)
    if match:
        ap_val = float(match.group(1))
        unique_corr = float(match.group(2)) if match.group(2) else None
        comp = match.group(3)
        
        if comp in layer_components:
            # Construct a label
            if unique_corr:
                label = f"{ap_val} (U{unique_corr})"
                sort_key = (ap_val, 1, unique_corr) # standard < unique
            else:
                label = f"{ap_val}"
                sort_key = (ap_val, 0, 0)
            
            # Check for rsa_results.csv inside the latest timestamp folder
            # The directory structure is base_dir / run_name / timestamp / rsa_results.csv
            # But wait, looking at the user's snippet, it seems d IS the run_name directory
            # and inside it has timestamp directories.
            
            timestamp_dirs = list(d.glob('*'))
            timestamp_dirs = [td for td in timestamp_dirs if td.is_dir()]
            
            if timestamp_dirs:
                # Get latest
                def get_timestamp(path):
                    try:
                        return datetime.strptime(path.name, '%Y-%m-%d_%H-%M-%S')
                    except:
                        try: 
                             # Try parsing with microseconds just in case, though format above suggests not
                             return datetime.strptime(path.name, '%Y-%m-%d_%H-%M-%S.%f')
                        except:
                             return datetime.min

                latest_dir = max(timestamp_dirs, key=get_timestamp)
                csv_file = latest_dir / 'rsa_results.csv'
                
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    results[comp][(sort_key, label)] = df

# Create four subplots
fig, axes = plt.subplots(1, 4, figsize=(26, 6), sharey=True)

for ax, comp, title in zip(axes, layer_components, ['Attention C_ATTN', 'Attention C_PROJ', 'MLP C_FC', 'MLP C_PROJ']):
    expert_means = []
    full_means = []
    labels = []
    
    # Sort collected data
    comp_data = results[comp]
    sorted_keys = sorted(comp_data.keys()) # Sorts by sort_key tuple
    
    expert_errs = []
    full_errs = []

    for key, label in sorted_keys:
        df = comp_data[(key, label)]
        
        # Calculate mean and 95% CI for Experts
        expert_vals = df['mean_rho_a']
        expert_mean = expert_vals.mean()
        expert_sem = stats.sem(expert_vals)
        expert_ci = expert_sem * stats.t.ppf((1 + 0.95) / 2., len(expert_vals)-1)
        
        # Calculate mean and 95% CI for Full
        full_vals = df['mean_rho_b']
        full_mean = full_vals.mean()
        full_sem = stats.sem(full_vals)
        full_ci = full_sem * stats.t.ppf((1 + 0.95) / 2., len(full_vals)-1)
        
        expert_means.append(expert_mean)
        full_means.append(full_mean)
        expert_errs.append(expert_ci)
        full_errs.append(full_ci)
        labels.append(label)
    
    if expert_means:
        x = np.arange(len(labels))
        width = 0.35
        
        # Plot with error bars (capsize adds the little horizontal lines at the top/bottom)
        ax.bar(x - width/2, expert_means, width, yerr=expert_errs, label='Expert', alpha=0.8, capsize=5)
        ax.bar(x + width/2, full_means, width, yerr=full_errs, label='Full', alpha=0.8, capsize=5)
        
        ax.set_xlabel('Condition (AP / Unique)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(frameon=True)
        ax.grid(True, linestyle=':', alpha=0.3, axis='y')

axes[0].set_ylabel('Mean RSA Correlation (with 95% CI)', fontsize=12)
fig.suptitle('Brain-Model RSA Alignment by Component and Method', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('rsa_by_component_with_unique_ci.png', dpi=200, bbox_inches='tight')
print('Plot saved as: rsa_by_component_with_unique_ci.png')
print(f'Processed {len(sorted_keys)} conditions for each component.')
