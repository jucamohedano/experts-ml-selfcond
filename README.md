# Brain-Aligned Expert Neurons in GPT-2

> **Note:** This repository is a fork of the [selfcond](https://github.com/xavisuau/selfcond) codebase by [Suau et al. (ICML 2022)](https://arxiv.org/abs/2201.12239). I've extended their original work to include fMRI alignment and RSA experiments.

---

## Project Overview

This project investigates whether **expert neurons** in Large Language Models (LLMs) align with human brain activity during semantic processing. 

Recent mechanistic interpretability work suggests that models encode concepts using sparse, specialized "expert" neurons. We test whether these sparse representations are more **brain-like** than standard dense layer embeddings by comparing them to fMRI data using **Representational Similarity Analysis (RSA)**.

### Scientific Context & Attribution

This work is scientifically grounded in the **ExpertLens** framework introduced by **Fedzechkina et al.**, which demonstrated that expert neurons capture human-like semantic structure.

> **Key Difference:** While Fedzechkina et al. validated experts against *behavioral* similarity judgments (e.g., MEN dataset), we extend their methodology to validate against **biological neural activity** (fMRI data from Mitchell et al., 2008).

*See:* [Fedzechkina et al. (2025)](https://arxiv.org/abs/2502.12328) "ExpertLens: Activation Steering Features are Highly Interpretable"

### Key Research Question
> *Does filtering language model representations to include only concept-specific expert neurons improve alignment with human neural responses, compared to using dense layer embeddings?*

## Analysis Pipeline

Our pipeline connects mechanistic interpretability (Expert Neurons) with cognitive neuroscience (RSA).

1.  **Expert Extraction**: We identify concept-specific neurons in **GPT-2 Small** using Average Precision (AP) on a dataset of 60 concrete nouns (Mitchell et al., 2008).
2.  **Unique Experts**: We filter redundant neurons to isolate distinct semantic signals.
3.  **Model RDMs**: We construct Representational Dissimilarity Matrices (RDMs) from the activity of these expert subsets.
4.  **Brain RDMs**: We process fMRI data from 9 subjects, tessellating the brain into 3D grid regions to compute corresponding neural RDMs.
5.  **RSA & Statistics**: We compare Model and Brain RDMs using Spearman correlation, assessing significance with permutation testing and FDR correction.

## Dataset
The generated dataset of Chain-of-Thought sentences used for expert neuron extraction in this project is publicly available on Hugging Face:
[jucamohedano/Qwen3-30B-A3B-Instruct-2507_custom_60_cot](https://huggingface.co/datasets/jucamohedano/Qwen3-30B-A3B-Instruct-2507_custom_60_cot)

To use it locally with this codebase, clone the dataset directly into the `assets/` directory (ensure you have `git-lfs` installed to fetch the actual large files):

```bash
mkdir -p assets
git clone https://huggingface.co/datasets/jucamohedano/Qwen3-30B-A3B-Instruct-2507_custom_60_cot assets/Qwen3-30B-A3B-Instruct-2507_custom_60_cot
```

## Repository Structure

```
experts-ml-selfcond/
├── assets/                 # Dedicated storage for HuggingFace datasets and large assets
├── conf/                   # Hydra configuration files
│   ├── task/               # Tasks (rsa, brain_rdm, steering, etc.)
│   └── model/              # Model configs (gpt2, pythia)
├── scripts/                # Core analysis scripts
│   ├── compute_expertise.py# Calculate neuron AP scores
│   ├── compute_rsa.py      # Run RSA comparison
│   ├── compute_brain_rdm.py# Process fMRI data to RDMs
│   └── steering_validation.py # Causal validation of experts
│   └── ...                 # Additional utility/preprocessing scripts
├── selfcond/               # Shared library code
│   ├── brain_data.py       # fMRI loading & processing
│   └── stats.py            # Permutation testing & FDR stats
├── my_docs/                # Documentation & Paper drafts
│   ├── expert_neurons_paper/ # LaTeX source for the paper
│   ├── pipeline_overview.md  # Detailed pipeline docs
│   └── rsa_experiment.md     # Experiment logs & results
└── run_pipeline.py         # Main entry point CLI
```

> **Note:** The `scripts/` and `selfcond/` directories contain many additional files (e.g., for dataset generation, preprocessing, or legacy experiments) not listed here. The core analysis is driven by the files above via `run_pipeline.py`.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended for reproducibility)

### Installation

First, install `uv` (the recommended, blazing-fast Python package manager):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and set it up:

```bash
git clone https://github.com/jucamohedano/experts-ml-selfcond.git
cd experts-ml-selfcond

# Create a virtual environment
uv venv

# Activate it (Linux/macOS)
source .venv/bin/activate 

# Sync dependencies from the lockfile
uv sync
```

Alternatively, use pip (may not guarantee exact versions):
```bash
pip install -r requirements.txt
```

### Running the Analysis

The entire workflow is managed via `run_pipeline.py` using Hydra for configuration.

**1. Compute Brain RDMs**
Process the raw fMRI `.mat` files into RDMs for each brain region.
```bash
python run_pipeline.py task=brain_rdm
```

**2. Extract Expert Features**
Identify expert neurons and generate model RDMs for different AP thresholds.
```bash
python run_pipeline.py task=word_feature_extraction task.ap_threshold=0.6
```

**3. Run RSA Comparison**
Compare the Model RDMs against Brain RDMs.
```bash
python run_pipeline.py task=rsa
```

**4. Steering Validation (Optional)**
Test if "experts" causally control generation.
```bash
python run_pipeline.py task=steering_validation
```

## Key Findings

*   **Projection Layers Matter**: The MLP projection layer (`mlp.c_proj`) contains the most brain-aligned signal ($d=0.627$ vs dense baseline).
*   **Sparsity improves Alignment**: Filtering for experts (AP $\geq$ 0.6) significantly improves alignment compared to using all neurons.
*   **Component Specificity**: MLP layers generally align better than attention layers, which require stricter filtering.

For full details, see the final project report: [JuanCamachoMohedano-257536-Project-Experts.pdf](my_docs/JuanCamachoMohedano-257536-Project-Experts.pdf).

> **Robustness Check (see branch `fix/voxel-normalization`):** A per-voxel normalization control was applied to the brain RDM computation ([`agent/plan.md`](agent/plan.md), [`my_docs/voxel_normalization_analysis.md`](my_docs/voxel_normalization_analysis.md)). This **reverses** the main expert neuron advantage (e.g., `mlp.c_proj` at AP ≥ 0.6 drops from $d=+0.627$ to $d=-0.523$), indicating that high-variance voxels were driving the original brain alignment signal. The expert neuron hypothesis is sensitive to this methodological choice.

## Results Archive

Complete results from all analysis runs — including original and voxel-normalized brain RDMs, RSA statistics, expert neuron tables, model RDMs, and visualizations — are archived on Hugging Face:

[`jucamohedano/experts-ml-selfcond-result`](https://huggingface.co/datasets/jucamohedano/experts-ml-selfcond-result)

The archive is a single compressed file (`results_complete.tar.xz`, ~1.6 GB for 55,282 files). To download and extract:

```bash
# Download the results archive from Hugging Face Hub
hf download jucamohedano/experts-ml-selfcond-result results_complete.tar.xz --repo-type dataset

# Extract
tar -xJf results_complete.tar.xz
```

**What's inside:**

| Subdirectory | Contents |
|---|---|
| `brain_rdm/` | Per-subject 60×60 RDMs (tessellated into 3D grid regions), with and without per-voxel normalization |
| `compute_responses/` | Cached GPT-2 model activations for 60 concepts × 1,000 sentences each |
| `compute_expertise/` | Average Precision (AP) scores per neuron per layer per concept |
| `unique_experts/` | Filtered unique expert neuron sets (redundancy-controlled) |
| `expert_overlap/` | Jaccard similarity matrices between concept expert sets |
| `word_features/` | Per-layer model RDMs (expert and dense conditions) |
| `subspace_gaze/` | PCA/LAT concept subspaces, similarity matrices, UMAP embeddings |
| `steering/` | Steering vector outputs and generation samples |
| `rsa/` | RSA comparison results (Spearman rho, paired t-tests, Cohen's d, FDR-corrected p-values) |

See [`my_docs/pipeline_overview.md`](my_docs/pipeline_overview.md) for the full pipeline diagram and [`my_docs/experiment_tracking.md`](my_docs/experiment_tracking.md) for a log of all runs.

## Attribution & License

This project is a research fork based on:
**Self-Conditioning Pre-Trained Language Models**  
*Xavier Suau, Luca Zappella, Nicholas Apostoloff*  
International Conference on Machine Learning (ICML), 2022.

If you use the original self-conditioning methods, please cite their work. If you use the brain-alignment/RSA extensions, please refer to this repository.
