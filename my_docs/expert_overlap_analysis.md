Summary of each script:

## Script Summaries

### 1. **`analyze_expert_overlap.py`**
- Extracts expert neurons from `expertise.csv` files (using AP threshold)
- Computes pairwise Jaccard similarity between expert sets for all concept pairs
- Saves a similarity matrix CSV and analysis metadata JSON
- Output: similarity matrix and statistics

### 2. **`visualize_expert_overlap.py`**
- Loads a Jaccard similarity matrix CSV (from `analyze_expert_overlap.py`)
- Generates two visualizations:
  - Heatmap: similarity matrix
  - Network graph: concepts as nodes, edges weighted by similarity (MDS/spring/hybrid layout)
- Saves: PNG heatmap, PNG graph, JSON of top-10 similarities

### 3. **`compute_shared_experts_stats.py`**
- Loads expert sets for multiple concepts
- Computes intersection (neurons that are experts for all concepts)
- Computes union (all unique expert neurons across concepts)
- Reports statistics: shared count, unique count, percentages
- Output: JSON summary with intersection/union statistics

### 4. **`expert_analysis.py`** (module, not a script)
- Defines the `ExpertSet` dataclass representing expert neurons
- Utilities: extract experts from CSV, compute Jaccard similarity, create similarity matrices
- Core library functions used by the analysis scripts

**Workflow**: `compute_shared_experts.py` → `analyze_expert_overlap.py` → `visualize_expert_overlap.py`