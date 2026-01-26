---
name: Paper TODO Revisions
overview: "Organize and execute 17 TODOs for the expert neurons paper, grouped by section: Introduction (2), Methodology (3), Results (6), Discussion (5), and Appendix (1)."
todos:
  - id: intro-fork
    content: Add fork attribution and GPT-2 justification paragraph to Introduction
    status: completed
  - id: method-dataset
    content: Add dataset size justification (400+/1000-) to Methodology III-A
    status: completed
  - id: method-threshold
    content: Fix correlation thresholds from {0.85, 0.9, 0.95} to {0.8, 0.9} in Methodology III-C
    status: completed
  - id: results-plots
    content: Generate top-100-ap.png plots for glass and screwdriver concepts
    status: completed
  - id: results-mds
    content: Generate MDS plot with lower similarity threshold for appendix
    status: completed
  - id: results-layer-justify
    content: Add blue placeholder for layer selection rationale (pending literature review)
    status: completed
  - id: results-fdr
    content: Add explanation for FDR-corrected significance = 0
    status: completed
  - id: results-sensitivity
    content: Clarify 'sensitivity to thresholding' meaning in attention layers section
    status: completed
  - id: results-redundancy
    content: "Expand redundancy discussion re: dataset/model dependence"
    status: completed
  - id: results-unique-ap
    content: Explain why unique expert filtering only at AP 0.5/0.6
    status: completed
  - id: discuss-pruning
    content: Add paragraph contrasting expert selection with pruning
    status: completed
  - id: discuss-tarigopula
    content: Cite Tarigopula et al. for hierarchical structure alignment
    status: completed
  - id: discuss-model-size
    content: Expand model size limitation to mention Pythia-12b
    status: completed
  - id: discuss-intra
    content: Add future work note on intra-category definitions
    status: completed
  - id: appendix-pseudo
    content: Add unique expert filtering pseudocode to appendix
    status: completed
isProject: false
---

# Paper TODO Revisions Plan

## Introduction Section

### TODO 2: Fork Attribution and GPT-2 Justification

**Action**: Add a paragraph explaining:

- The code is forked from Suau et al.'s "Self-conditioning Pre-Trained Language Models"
- GPT-2 Small was chosen because it's already set up in the framework, has only 12 transformer blocks, and runs on consumer hardware

**Location**: Add after the first paragraph of Section I or at the end of Section I.D (Contributions), referencing the fork.

---

## Methodology Section

### TODO 1: Dataset Size Justification

**Action**: In Section III-A (Data Construction), add a sentence after the bullet points explaining why 400 positives and 1000 negatives were chosen:

> "These sizes were selected based on Fedzechkina et al.'s finding that they provide sufficient discriminative power while remaining computationally tractable."

**Location**: After line 16 in `methodology.tex`

### TODO 3: Fix Graph-Based Grouping Thresholds

**Action**: Change line 73 from `\tau_{\text{corr}} \in \{0.85, 0.9, 0.95\}` to `\tau_{\text{corr}} \in \{0.8, 0.9\}` to match the actual thresholds used in Table I (Results section).

**Location**: Line 73 in `methodology.tex`

### TODO 2 (Methodology portion): Credit Prior Work for Expert Extraction

**Status**: Already done - Section III-B already states "We adopt the expert extraction methodology from Fedzechkina et al."

---

## Results Section

### TODO 4: Add Top-100 AP Plots for Glass and Screwdriver

The plots are currently stored at:

- responses/Qwen3-30B-A3B-Instruct-2507_custom_60_responses_cot/gpt2/custom/screwdriver/expertise/top-100-ap.pn
- responses/Qwen3-30B-A3B-Instruct-2507_custom_60_responses_cot/gpt2/custom/glass/expertise/top-100-ap.png

Copy to `my_docs/expert_neurons_paper/figures/` and add figure references in Section IV-A.

### TODO 5: Add MDS Plot with Lower Similarity Threshold to Appendix

**Action**: location of the plot is in `results/expert_overlap/Qwen3-30B-A3B-Instruct-2507_custom_60_cot/Qwen3-30B-A3B-Instruct-2507_gpt2/overlap-tau-0.5/2026-01-24_23-38-28/graph_outputs/graph_threshold_0.05_hybrid.png`. Add to appendix and cite in Section IV-B where cross-category similarities are mentioned.

### TODO 6: Justify Layer Choice (attn.c_proj and mlp.c_fc)

**Action**: Leave a blue placeholder text in Section IV-C:

> `\textcolor{blue}{[Layer selection rationale pending literature review]}`

Note: The Suau et al. paper does NOT explicitly justify why these 4 layers were chosen. The likely reason is that these are the primary linear transformation layers where "neurons" with interpretable activations exist.

**>> Edit**: Since the paper from Suau et al. does not justify it, then we don't have to do it. But we can deconstruct a transformer block and briefly say the layers that it's composed of. We are using the 12 trnasformer blocks gpt2 model, i.e. 48 layers in total that we compute the expertise from.

### TODO 7: Explain FDR Correction Yielding 0 Significance

**Action**: Add explanation in Section IV-C after Table II or in the Summary subsection:

> "The limited FDR-corrected significance reflects the stringent correction for 87 brain regions × multiple conditions. With 87 comparisons per condition at $\alpha = 0.05$, the effective per-test threshold after FDR correction is substantially lower, requiring larger effect sizes to survive correction."

### TODO 8: Clarify "Sensitivity to Thresholding"

**Context**: "Sensitivity to thresholding" means attention layers' brain alignment changes substantially depending on which AP threshold is used. At AP ≥ 0.5, there's minimal benefit (d = 0.027), but at AP ≥ 0.7, there's strong benefit (d = 0.536). This is "sensitive" because small threshold changes dramatically affect results.

**Action**: Consider rewording the subsection title or adding a clarifying sentence at the beginning of Section IV-C.2. Question whether it is really sensitivity or not.

### TODO 13: Discuss Redundancy Dependence on Dataset/Model Scale

**Action**: Expand the redundancy discussion in Section IV-A to note:

- Redundancy rates likely depend on dataset size and composition
- Smaller models may have different redundancy patterns than larger models

### TODO 17: Explain Why Unique Expert Filtering Only for AP ∈ {0.5, 0.6}

**Action**: Add clarification in Section IV-C after unique expert results:

> "Unique expert filtering was analyzed only at AP thresholds 0.5 and 0.6. At higher thresholds, the already-small expert sets showed minimal redundancy (as reported in Section IV-A), making unique filtering unnecessary."

**Verification**: The results tables show unique expert conditions only at 0.5 and 0.6 AP thresholds. This is consistent with the low redundancy rates in Table I.

---

## Discussion Section

### TODO 9: Contrast with Pruning

**Action**: Add paragraph in Section V:

> "Interestingly, our approach contrasts with traditional pruning methods. While pruning removes neurons to improve efficiency or generalization, expert identification selects neurons that *already* encode meaningful concepts. This suggests that the brain-aligned representations emerge naturally in LLMs rather than requiring architectural modification." cite the papers performing pruninig that we have in our bibliography.

### TODO 10: Cite Tarigopula et al. for Hierarchical Structure Alignment

**Action**: Add reference to Tarigopula et al. (already in `related_work.tex` as `\cite{TARIGOPULA202389}`) in the discussion when discussing how pruning/selection improves brain alignment:

> "This aligns with Tarigopula et al.'s finding that pruning DNNs based on behavioral similarity improves neural prediction in visual cortex~\cite{TARIGOPULA202389}."

### TODO 11: Limitation - Model Size

**Action**: Already partially addressed in Section V.C (Limitations, "Model Scope"). Expand to explicitly mention:

> "Future work should test larger models such as Pythia-12b, where ExpertLens reported stronger alignment effects."
For reference from Fedzechkina et. al: "ExpertLens at τ=0.5 outperforms standard embedding-based similarity.
* **Which models were tested for different τ?**
  Main experiments used the **Pythia family** (Pythia-70m, Pythia-1b, Pythia-12b) across multiple checkpoints (1, 512, 1k, 4k, 36k, 72k, 143k). The authors also show that the results generalize to **Gemma-2b** (and Gemma-2b-IFT) in the appendix, and they ran pilot dataset-generation with **GPT-4, Mistral-7b-Instruct,** and an internal 80b model."

### TODO 12: Future Work - Intra-Category Definitions

**Action**: Add to limitations or future work:

> "Additionally, our concept definitions use inter-category discrimination (positive vs. negative from other categories). Generating datasets with intra-category distinctions (e.g., discriminating 'dog' from 'cat' within animals) could reveal finer-grained expert organization."

---

## Appendix

### TODO 16: Add Unique Expert Filtering Pseudocode

**Action**: Add new appendix section with pseudocode for the redundancy filtering algorithm based on [`scripts/filter_unique_experts.py`](scripts/filter_unique_experts.py):

```
Algorithm: Unique Expert Filtering
Input: Expert set E_l(c), correlation threshold τ_corr
Output: Unique expert set E_l^unique(c)

1. Compute pairwise Pearson correlation ρ_mn for all (m,n) ∈ E_l(c)
2. Build graph G = (V, E) where V = E_l(c), E = {(m,n) : ρ_mn > τ_corr}
3. Find connected components {G_1, ..., G_k} in G
4. For each component G_i:
   - Select m* = argmax_{m ∈ G_i} AP(m,c)
   - Add m* to E_l^unique(c)
5. Return E_l^unique(c)
```

---

## Summary by Section

| Section | TODOs | Complexity |

|---------|-------|------------|

| Introduction | 1 | Text edit |

| Methodology | 2 | Text edits |

| Results | 6 | 2 figure generations + text edits |

| Discussion | 4 | Text edits |

| Appendix | 1 | Add pseudocode |

## Questions Answered

- **TODO 8 ("sensitivity to thresholding")**: This means attention layers show dramatically different brain alignment depending on AP threshold choice, unlike MLP layers which are more stable.
- **TODO 17**: The reason for only testing unique experts at AP 0.5/0.6 is justified by the low redundancy rates at higher thresholds - there's nothing to filter.