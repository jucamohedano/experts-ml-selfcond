# Self-Conditioning Pipeline Overview

This document provides a visual overview of the self-conditioning analysis pipeline.

## Pipeline Diagram

### ASCII Version

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SELF-CONDITIONING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │  LABELED DATA    │    "The airplane flew..." (label=1)
    │  (sentences)     │    "The cat slept..."    (label=0)
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ COMPUTE RESPONSES│    Run sentences through model
    │                  │    Record activations per layer
    └────────┬─────────┘
             │
             │    Output: [neurons × sentences] per layer
             ▼
    ┌──────────────────┐
    │ COMPUTE EXPERTISE│    For each neuron:
    │                  │    AP = how well it distinguishes concept?
    └────────┬─────────┘
             │
             │    Output: expertise.csv (neuron, layer, AP score)
             │
             ├─────────────────────────────────────┐
             │                                     │ (optional: use raw experts)
             ▼                                     │
    ┌──────────────────┐                           │
    │  FILTER EXPERTS  │    Keep neurons with AP > threshold
    │   (optional)     │    Remove correlated/redundant neurons
    └────────┬─────────┘                           │
             │                                     │
             │    Output: Unique expert sets       │
             │                                     │
             ├─────────────────┬───────────────────┤
             ▼                 ▼                   ▼
    ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
    │ EXPERT OVERLAP │ │ SUBSPACE GAZE  │ │   STEERING     │
    │                │ │                │ │                │
    │ Jaccard sim.   │ │ Extract concept│ │ Add concept    │
    │ between        │ │ vectors via    │ │ vector to      │
    │ concept pairs  │ │ PCA/LAT        │ │ activations    │
    └────────────────┘ └────────────────┘ └────────────────┘
             │                 │                 │
             ▼                 ▼                 ▼
    ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
    │   Heatmaps     │ │ Cosine sim.    │ │ Concept-biased │
    │   Graphs       │ │ UMAP plots     │ │ text generation│
    └────────────────┘ └────────────────┘ └────────────────┘
```

### Mermaid Version

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#ffffff'}}}%%
flowchart TB
    subgraph PIPELINE["🔬 SELF-CONDITIONING PIPELINE"]
        direction TB
        
        DATA["📄 <b>LABELED DATA</b><br/><i>(sentences)</i><br/>─────────<br/>'The airplane flew...' label=1<br/>'The cat slept...' label=0"]
        
        COMPUTE["⚙️ <b>COMPUTE RESPONSES</b><br/>─────────<br/>Run sentences through model<br/>Record activations per layer"]
        
        EXPERTISE["📊 <b>COMPUTE EXPERTISE</b><br/>─────────<br/>For each neuron:<br/>AP = how well it distinguishes concept?"]
        
        FILTER["🔍 <b>FILTER EXPERTS</b><br/><i>(optional)</i><br/>─────────<br/>Keep neurons with AP > threshold<br/>Remove correlated/redundant"]
        
        OVERLAP["🔗 <b>EXPERT OVERLAP</b><br/>─────────<br/>Jaccard similarity<br/>between concept pairs"]
        
        GAZE["👁️ <b>SUBSPACE GAZE</b><br/>─────────<br/>Extract concept vectors<br/>via PCA/LAT"]
        
        STEER["🎯 <b>STEERING</b><br/>─────────<br/>Add concept vector<br/>to activations"]
        
        OUT1["📈 Heatmaps & Graphs"]
        OUT2["📐 Cosine sim. & UMAP plots"]
        OUT3["✍️ Concept-biased generation"]
    end
    
    DATA --> |"[neurons × sentences] per layer"| COMPUTE
    COMPUTE --> EXPERTISE
    EXPERTISE --> |"expertise.csv<br/>(neuron, layer, AP score)"| FILTER
    EXPERTISE -.-> |"Optional: use raw experts"| OVERLAP
    FILTER --> |"Unique expert sets"| OVERLAP
    FILTER --> GAZE
    FILTER --> STEER
    OVERLAP --> OUT1
    GAZE --> OUT2
    STEER --> OUT3

    style DATA fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style COMPUTE fill:#fff8e1,stroke:#ff8f00,stroke-width:2px
    style EXPERTISE fill:#fff8e1,stroke:#ff8f00,stroke-width:2px
    style FILTER fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
    style OVERLAP fill:#e8f5e9,stroke:#43a047,stroke-width:2px
    style GAZE fill:#e8f5e9,stroke:#43a047,stroke-width:2px
    style STEER fill:#e8f5e9,stroke:#43a047,stroke-width:2px
    style OUT1 fill:#fce4ec,stroke:#d81b60,stroke-width:2px
    style OUT2 fill:#fce4ec,stroke:#d81b60,stroke-width:2px
    style OUT3 fill:#fce4ec,stroke:#d81b60,stroke-width:2px
```

## Key Idea

Identify which neurons "care about" each concept (**experts**), then use them for:
- **Analysis**: Understanding concept relationships and representations
- **Generation control**: Steering text generation toward specific concepts

## Stage Descriptions

| Stage | Script | Description |
|-------|--------|-------------|
| Compute Responses | `compute_responses.py` | Run labeled sentences through model, cache activations |
| Compute Expertise | `compute_expertise.py` | Calculate Average Precision (AP) per neuron |
| Filter Experts | `filter_unique_experts.py` | Remove redundant neurons via correlation |
| Expert Overlap | `analyze_expert_overlap.py` | Compute Jaccard similarity between concepts |
| Subspace Gaze | `subspace_gaze.py` | Extract concept vectors using PCA/LAT |
| Steering | `steering_validation.py` | Validate concept vectors via generation |

## Data Flow

1. **Input**: Sentences labeled by concept (e.g., "airplane" sentences vs. others)
2. **Activations**: `[neurons × sentences]` matrix per layer
3. **Expertise**: `expertise.csv` with AP scores per neuron
4. **Experts**: Subset of neurons with AP > threshold (e.g., 0.5)
5. **Output**: Visualizations, concept vectors, or steered generations

