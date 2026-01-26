# Self-Conditioning Pipeline Overview

This document provides a visual overview of the self-conditioning analysis pipeline, including the brain-model RSA comparison.

## Pipeline Diagram

### ASCII Version

```
                              SELF-CONDITIONING PIPELINE

    MODEL SIDE                                           BRAIN SIDE
    ----------                                           ----------

    +------------------+                              +------------------+
    |  LABELED DATA    |    "The airplane flew..."   |  BRAIN DATA      |    Mitchell 2008
    |  (sentences)     |    "The cat slept..."       |  (fMRI)          |    9 subjects, 60 words
    +--------+---------+                              +--------+---------+
             |                                                  |
             v                                                  v
    +------------------+                              +------------------+
    | COMPUTE RESPONSES|    Run sentences through    | TESSELLATE BRAIN |    3D grid (e.g. 2x10x5)
    |                  |    model, record activations|                  |    ~100 regions/subject
    +--------+---------+                              +--------+---------+
             |                                                  |
             |    Output: [neurons x sentences]                 |    Output: regions + voxels
             v                                                  v
    +------------------+                              +------------------+
    | COMPUTE EXPERTISE|    For each neuron:         | COMPUTE BRAIN    |    For each region:
    |                  |    AP = distinguishes?      | RDMs             |    60x60 dissimilarity
    +--------+---------+                              +--------+---------+
             |                                                  |
             |    Output: expertise.csv                         |    Output: brain_rdms.pkl
             |                                                  |
             +-------------------+                              |
             |                   |                              |
             v                   |                              |
    +------------------+         |                              |
    |  WORD FEATURES   |    Expert activations                  |
    |                  |    for 60 words                        |
    +--------+---------+         |                              |
             |                   |                              |
             |    Output: Model RDMs (60x60 per layer)          |
             |                   |                              |
             +-------------------+------------------------------+
             |                   |                              |
             |                   |                              |
    +--------v--------+ +--------v--------+ +--------v--------+ |
    | EXPERT OVERLAP  | | SUBSPACE GAZE   | |   STEERING      | |
    |                 | |                 | |                 | |
    | Jaccard sim.    | | Extract concept | | Add concept     | |
    | between         | | vectors via     | | vector to       | |
    | concept pairs   | | PCA/LAT         | | activations     | |
    +-----------------+ +-----------------+ +-----------------+ |
             |                   |                 |            |
             v                   v                 v            |
    +-----------------+ +-----------------+ +-----------------+ |
    |   Heatmaps      | | Cosine sim.     | | Concept-biased  | |
    |   Graphs        | | UMAP plots      | | text generation | |
    +-----------------+ +-----------------+ +-----------------+ |
                                                                |
             +--------------------------------------------------+
             |
             v
    +------------------------------------------------------------------+
    |                              RSA                                  |
    |    Compare Model RDMs (per layer) with Brain RDMs (per region)   |
    |    Spearman correlation on upper triangle (n*(n-1)/2 pairs)      |
    +--------+---------------------------------------------------------+
             |
             |    Output: rsa_results.csv (region x layer correlations)
             v
    +------------------------------------------------------------------+
    |                     PERMUTATION TESTING                          |
    |    Shuffle word labels, recompute RSA, build null distribution   |
    |    GPU-accelerated, 10k+ permutations                            |
    +--------+---------------------------------------------------------+
             |
             |    Output: p-values per (region, layer) pair
             v
    +------------------------------------------------------------------+
    |                     FDR CORRECTION                               |
    |    Benjamini-Hochberg correction for n_regions x n_layers tests  |
    |    (Currently not yielding significant results)                  |
    +--------+---------------------------------------------------------+
             |
             v
    +-----------------+ +-----------------+ +-----------------+
    | Manhattan Plot  | | Layer-Type      | | Ranked Pairs    |
    | (significant    | | Comparison      | | Table           |
    |  pairs)         | | (attn vs MLP)   | |                 |
    +-----------------+ +-----------------+ +-----------------+
```

## Key Idea

Identify which neurons "care about" each concept (**experts**), then use them for:
- **Analysis**: Understanding concept relationships and representations
- **Generation control**: Steering text generation toward specific concepts
- **Brain comparison**: RSA between model and brain representations

## Stage Descriptions

### Model Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| Compute Responses | `compute_responses.py` | Run labeled sentences through model, cache activations |
| Compute Expertise | `compute_expertise.py` | Calculate Average Precision (AP) per neuron |
| Word Features | `word_features.py` | Build 60x60 Model RDMs from expert activations |
| Expert Overlap | `analyze_expert_overlap.py` | Compute Jaccard similarity between concepts |
| Subspace Gaze | `subspace_gaze.py` | Extract concept vectors using PCA/LAT |
| Steering | `steering_validation.py` | Validate concept vectors via generation |

### Brain Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| Load Brain Data | `selfcond/brain_data.py` | Load Mitchell 2008 fMRI data, average trials |
| Tessellate | `selfcond/brain_data.py` | Divide brain into 3D grid regions |
| Brain RDMs | `compute_brain_rdm.py` | Compute 60x60 RDM per region per subject |

### RSA Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| RSA | `compute_rsa.py` | Spearman correlation between model and brain RDMs |
| Permutation Test | `selfcond/stats.py` | GPU-accelerated permutation testing |
| FDR Correction | `selfcond/stats.py` | Benjamini-Hochberg multiple comparison correction |

## Data Flow

### Model Side
1. **Input**: Sentences labeled by concept (e.g., "airplane" sentences vs. others)
2. **Activations**: `[neurons x sentences]` matrix per layer
3. **Expertise**: `expertise.csv` with AP scores per neuron
4. **Experts**: Subset of neurons with AP > threshold (e.g., 0.5)
5. **Model RDMs**: 60x60 dissimilarity matrix per layer

### Brain Side
1. **Input**: fMRI `.mat` files (9 subjects, 360 trials each)
2. **Word Activations**: `[n_words x V]` averaged responses per word (e.g. 60 x ~20k voxels)
3. **Regions**: Grid cells per subject (configurable, e.g. 2x10x5 = ~100 regions)
4. **Brain RDMs**: n_words x n_words dissimilarity matrix per region per subject

### RSA Comparison
1. **Input**: Model RDMs (per layer) + Brain RDMs (per subject, per region)
2. **RSA Correlation**: Spearman r between upper triangles (n_words*(n_words-1)/2 pairs, e.g. 1770 for 60 words)
3. **Permutation Test**: Shuffle words, recompute RSA, build null distribution
4. **FDR Correction**: Correct for n_regions x n_layers comparisons (e.g. ~4,800 for 100 regions x 48 layers)
5. **Output**: Significant (region, layer) pairs, visualizations

## Current Status

- **RSA correlations**: Computed successfully
- **Permutation testing**: Implemented with GPU acceleration
- **FDR correction**: Applied but not yielding significant results (see `brain_rdm_and_rsa.md` for discussion)
