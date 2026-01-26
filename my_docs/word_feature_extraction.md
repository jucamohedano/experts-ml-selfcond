# Word Feature Extraction and RDM Computation

## Overview

This document describes the implementation of **Expert-Based Word Representations** and **Representational Dissimilarity Matrices (RDMs)**. The goal is to create feature vectors for each word using neurons (optionally filtered by expertise) and compute pairwise dissimilarity matrices.

## Task Modes

The task supports three modes of operation:

| Mode | Config | Description |
|------|--------|-------------|
| **Expert** | `use_all_neurons=false` | Uses neurons with AP ≥ threshold (default) |
| **Full** | `use_all_neurons=true` | Uses ALL neurons (no AP filtering) |
| **Unique Expert** | `unique_experts_path=<path>` | Uses only unique experts (not shared across concepts) |

## Configuration

### Hydra Config: `conf/task/word_feature_extraction.yaml`

```yaml
task:
  name: word_feature_extraction
  ap_threshold: 0.5           # Minimum AP score for expert selection
  generation_type: cot        # Data variant tag
  run_tag: ${task.generation_type}-ap-${task.ap_threshold}
  activations_path: null      # Path to concept responses (required)
  expertise_path: null        # Path to expertise CSVs (required unless unique_experts_path or use_all_neurons)
  layer: all                  # "all" or specific layer name
  use_all_neurons: false      # Set to true for Full RDM computation
  unique_experts_path: null   # Path to unique experts results (for unique RDM computation)
```

## Running the Pipeline

### Expert RDMs (default)
```bash
python run_pipeline.py \
  task=word_feature_extraction \
  model=Qwen3-30B-A3B-Instruct-2507_gpt2 \
  task.activations_path=responses/.../gpt2/custom \
  task.expertise_path=responses/.../gpt2/custom \
  task.run_tag=cot-ap-0.5
```

### Full RDMs (all neurons)
```bash
python run_pipeline.py \
  task=word_feature_extraction \
  model=Qwen3-30B-A3B-Instruct-2507_gpt2 \
  task.activations_path=responses/.../gpt2/custom \
  task.use_all_neurons=true \
  task.run_tag=full
```

### Unique Expert RDMs
```bash
python run_pipeline.py \
  task=word_feature_extraction \
  model=Qwen3-30B-A3B-Instruct-2507_gpt2 \
  task.activations_path=responses/.../gpt2/custom \
  task.unique_experts_path=results/unique_experts/.../corr-0.9/2025-XX-XX \
  task.run_tag=ap-0.5-unique-corr_0.9
```

## Output Structure

```
results/word_features/{concept_group}/{model}/{run_tag}/{timestamp}/
├── word_features.pkl              # {layer: {concept: vector}}
├── word_features_metadata.json    # Configuration and expert counts
├── feature_matrices/              # Per-layer feature matrices
│   ├── {layer}_features.npy       # [num_words, num_features]
│   └── {layer}_concepts.txt       # Concept order (alphabetical)
└── rdms/                          # RDM outputs
    ├── {layer}_rdm.npy            # RDM matrix [60, 60]
    ├── {layer}_rdm.png            # Heatmap visualization
    └── {layer}_rdm.csv            # Labeled CSV
```

## Mathematical Formulation

### Expert-Based Word Representations

For each word $w_i$ and layer $l$:

$$\mathbf{f}_l(w_i) = \frac{1}{|S_i^+|}\sum_{s \in S_i^+} \mathbf{a}_l(s)[E_l]$$

Where $E_l$ is the **unified expert pool** (union of experts across all words) ensuring all representations exist in the same feature space.

### RDM Computation

$$D_l^{\text{expert}}(i,j) = 1 - \text{Pearson}(\mathbf{f}_l(w_i), \mathbf{f}_l(w_j))$$

## Script Implementation

### `scripts/word_features.py`

**Main Function:** `extract_word_features()`

**Parameters:**
- `activations_path`: Path to concept responses
- `expertise_path`: Path to expertise CSVs (optional if `use_all_neurons=True` or `unique_experts_path` set)
- `ap_threshold`: Minimum AP score for expert selection
- `layer`: Layer filter ("all" or specific layer name)
- `use_all_neurons`: If True, uses all neurons (no expert filtering)
- `unique_experts_path`: Path to unique expert CSVs

## Available RDM Conditions

| Condition | Path | Description |
|-----------|------|-------------|
| Expert | `cot-ap-0.5/...` | Standard experts (AP≥0.5) |
| Full | `full/...` | All neurons |
| Unique 0.9 | `ap-0.5-unique-corr_0.9/...` | Unique experts (corr≤0.9) |
| Unique 0.8 | `ap-0.5-unique-corr_0.8/...` | Unique experts (corr≤0.8) |

## References

- Implementation: `scripts/word_features.py`
- Configuration: `conf/task/word_feature_extraction.yaml`
- Pipeline: `run_pipeline.py`
