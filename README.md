# Brain-Aligned Expert Neurons in GPT-2

> **Note:** This repository builds upon the codebase from **Suau et al. (2022)**, "Self-Conditioning Pre-Trained Language Models". We extend their framework for expert neuron extraction to investigate the biological plausibility of sparse model representations.
> *Original Work:* [Suau et al. (ICML 2022)](https://arxiv.org/abs/2201.12239) | [GitHub](https://github.com/xavisuau/selfcond)

---

## 🧠 Project Overview

This project investigates whether **expert neurons** in Large Language Models (LLMs) align with human brain activity during semantic processing. 

Recent mechanistic interpretability work suggests that models encode concepts using sparse, specialized "expert" neurons. We test whether these sparse representations are more **brain-like** than standard dense layer embeddings by comparing them to fMRI data using **Representational Similarity Analysis (RSA)**.

### Scientific Context & Attribution

This work is scientifically grounded in the **ExpertLens** framework introduced by **Fedzechkina et al.**, which demonstrated that expert neurons capture human-like semantic structure.

> **Key Difference:** While Fedzechkina et al. validated experts against *behavioral* similarity judgments (e.g., MEN dataset), we extend their methodology to validate against **biological neural activity** (fMRI data from Mitchell et al., 2008).

*See:* [Fedzechkina et al. (2025)](https://arxiv.org/abs/2502.12328) "ExpertLens: Activation Steering Features are Highly Interpretable"

### Key Research Question
> *Does filtering language model representations to include only concept-specific expert neurons improve alignment with human neural responses, compared to using dense layer embeddings?*

## 🚀 Analysis Pipeline

Our pipeline connects mechanistic interpretability (Expert Neurons) with cognitive neuroscience (RSA).

1.  **Expert Extraction**: We identify concept-specific neurons in **GPT-2 Small** using Average Precision (AP) on a dataset of 60 concrete nouns (Mitchell et al., 2008).
2.  **Unique Experts**: We filter redundant neurons to isolate distinct semantic signals.
3.  **Model RDMs**: We construct Representational Dissimilarity Matrices (RDMs) from the activity of these expert subsets.
4.  **Brain RDMs**: We process fMRI data from 9 subjects, tessellating the brain into 3D grid regions to compute corresponding neural RDMs.
5.  **RSA & Statistics**: We compare Model and Brain RDMs using Spearman correlation, assessing significance with permutation testing and FDR correction.

## 📂 Repository Structure

```
ml-selfcond/
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

## 🛠️ Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended for reproducibility)

### Installation

Clone the repository and sync dependencies using `uv`:

```bash
git clone https://github.com/jucamohedano/ml-selfcond.git
cd ml-selfcond

# Create venv and install dependencies from lockfile
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

## 📊 Key Findings

*   **Projection Layers Matter**: The MLP projection layer (`mlp.c_proj`) contains the most brain-aligned signal ($d=0.627$ vs dense baseline).
*   **Sparsity improves Alignment**: Filtering for experts (AP $\geq$ 0.6) significantly improves alignment compared to using all neurons.
*   **Component Specificity**: MLP layers generally align better than attention layers, which require stricter filtering.

For full details, see the paper draft in `my_docs/expert_neurons_paper/`.

## 📜 Attribution & License

This project is a research fork based on:
**Self-Conditioning Pre-Trained Language Models**  
*Xavier Suau, Luca Zappella, Nicholas Apostoloff*  
International Conference on Machine Learning (ICML), 2022.

If you use the original self-conditioning methods, please cite their work. If you use the brain-alignment/RSA extensions, please refer to this repository.
