# Experiment Tracking with Hydra

This repository uses [Hydra](https://hydra.cc/) for experiment configuration and tracking. This setup replaces manual bash scripts with structured, reproducible, and flexible configuration management.

## Overview

The tracking system organizes experiments by:
1. **Task**: The analysis being run (e.g., `unique_experts`, `compute_responses`).
2. **Dataset**: The concept group being analyzed (e.g., `openai_custom_60`).
3. **Model**: The generation and processing model pair (e.g., `gpt2_Qwen3-8B-FP8`).
4. **Run Tag**: A parameterized identifier (e.g., `corr-0.9`, `responses-1024`).
5. **Timestamp**: Unique directory for each execution.

### Directory Structure

Inputs and outputs are organized as follows:

```
ml-selfcond/
├── conf/                       # Hydra configuration
│   ├── config.yaml             # Main config & defaults
│   ├── model/                  # Model pair configs
│   ├── concept_group/          # Dataset/Concept lists
│   └── task/                   # Task-specific configs
│       ├── unique_experts.yaml
│       ├── expert_overlap.yaml
│       ├── compute_responses.yaml
│       ├── compute_expertise.yaml
│       └── visualize_overlap.yaml
├── run_pipeline.py             # Main entry point
└── results/                    # Experiment outputs
    └── [task_name]/
        └── [dataset_name]/
            └── [model_name]/
                └── [run_tag]/
                    └── [timestamp]/
                        ├── [outputs]
                        ├── .hydra/           # Saved run config
                        └── run_pipeline.log  # Execution log
```

## Usage

### Basic Execution

Run the pipeline by specifying the desired task. The default task is `unique_experts` if not specified.

```bash
python run_pipeline.py task=[task_name]
```

### Available Tasks

#### 1. Compute Responses (`compute_responses`)
Generates model responses for the specified concepts.

```bash
python run_pipeline.py task=compute_responses
```
**Key Parameters:**
- `task.seq_len`: Sequence length (default: 1024)
- `task.inf_batch_size`: Inference batch size (default: 8)
- `task.num_per_concept`: Number of generations per concept (default: 1000)

#### 2. Compute Expertise (`compute_expertise`)
Analyzes neuron activations to identify experts.

```bash
python run_pipeline.py task=compute_expertise
```
**Key Parameters:**
- `task.k`: Top K neurons (default: 10)

#### 3. Unique Experts Analysis (`unique_experts`)
Identifies unique experts by filtering highly correlated neurons.

```bash
python run_pipeline.py task=unique_experts
```
**Key Parameters:**
- `task.correlation_threshold`: Pearson correlation threshold for redundancy (default: 0.9)
- `task.ap_threshold`: Average Precision threshold (default: 0.5)

#### 4. Expert Overlap Analysis (`expert_overlap`)
Computes Jaccard similarity between experts of different concepts and generates visualizations.

```bash
# Run on original experts (default)
python run_pipeline.py task=expert_overlap

# Run on UNIQUE experts (from a previous run)
python run_pipeline.py task=expert_overlap \
    task.input_dir=results/unique_experts/.../2025-11-20_10-41-14 \
    task.pattern="**/*_expertise_unique.csv"
```
**Visualization Parameters (`task.visualization`):**
- `low_threshold`: Minimum similarity to draw an edge (default: 0.2)
- `layout_mode`: Graph layout (`mds`, `spring`, `hybrid`)
- `topk`: Number of top pairs to save

#### 5. Visualize Overlap (`visualize_overlap`)
Regenerate visualizations from an existing similarity matrix without recomputing overlap.

```bash
python run_pipeline.py task=visualize_overlap \
    task.csv_path=/path/to/similarity_matrix.csv \
    task.visualization.low_threshold=0.2 \
    task.visualization.layout_mode=hybrid
```

### Overriding Parameters

You can override any configuration value from the command line using dot notation.

```bash
# Change layout mode for overlap visualization
python run_pipeline.py task=expert_overlap task.visualization.layout_mode=spring

# Run for a specific subset of concepts
python run_pipeline.py concept_group.concepts="[airplane,car,dog]"

# Change model pair
python run_pipeline.py model=some_other_model
```

### Multirun (Sweeps)

Run multiple experiments sequentially with different parameters using `-m` (multirun):

```bash
# Compare graph layouts
python run_pipeline.py -m task=visualize_overlap \
    task.csv_path=... \
    task.visualization.layout_mode=mds,spring,hybrid
```

## Configuration Files

### `conf/config.yaml`
Global defaults and composition.

### `conf/task/`
Contains specific configuration for each task type.

**Example `conf/task/expert_overlap.yaml`:**
```yaml
name: expert_overlap
ap_threshold: 0.5
run_tag: overlap-tau-${task.ap_threshold}
input_dir: null
pattern: "**/expertise.csv"
visualization:
  low_threshold: 0.2
  layout_mode: mds
  topk: 10
  min_edge_width: 0.2
  max_edge_width: 15.0
  dpi: 200
  fontsize: 9
  spring_k: 0.5
  hybrid_iterations: 100
  seed: 0
```
