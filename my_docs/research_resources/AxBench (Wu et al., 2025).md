Here’s a **tight AxBench summary**, plus the **direct answer on LAT vs Diff Mean vs PCA**, grounded in what the authors actually evaluate.

---

## What is **AxBench** (Wu et al., 2025)?

* **Goal:** Provide a **standardized benchmark** for evaluating *concept detectors* used for **steering and interpretability** in LLMs.
* **Core claim:** *Simple linear methods outperform complex sparse autoencoders (SAEs)* for both **concept detection** and **causal steering**.
* **Models tested:** LLaMA-2, Mistral, Gemma (multiple sizes)
* **Evaluation axes:**

  * **Detection quality** (how well a direction separates concept vs non-concept)
  * **Causal steering effectiveness** (does intervening along the direction actually change model behavior?)
  * **Stability / robustness**

---

## Concept detector methods compared

Among others, AxBench evaluates:

* **Diff Mean**
* **LAT (Linear Activation Tuning)**
* **PCA**
* Sparse Autoencoders (SAEs)
* Random / control directions

---

## Ranking of **Diff Mean vs LAT vs PCA**

### 🥇 **Best overall: Diff Mean**

* Consistently **top-performing baseline**
* Strongest **concept separability**
* Most **reliable causal steering**
* Extremely simple: mean(activations | concept) − mean(activations | ¬concept)

**Authors’ conclusion:**

> Diff Mean is *hard to beat* and often outperforms SAEs.

---

### 🥈 **Second: LAT**

* Performs **similarly but slightly worse** than Diff Mean
* More sensitive to:

  * hyperparameters
  * layer choice
  * concept imbalance
* Still clearly **better than PCA**

---

### 🥉 **Worst of the three: PCA**

* Captures **variance**, not semantic discrimination
* Often misaligned with the concept of interest
* Weak causal steering
* Authors treat PCA mainly as a **negative baseline**

---

## High-level ordering (as implied by results)

> **Diff Mean > LAT >> PCA**

This ordering holds across:

* models
* layers
* concept types
* steering metrics

---

## Key interpretation (important)

AxBench argues:

* **Concept detectors should optimize discrimination, not reconstruction**
* Methods like PCA (and even SAEs) optimize the *wrong objective*
* Linear discriminative directions (Diff Mean) align best with causal control

---

## One-sentence takeaway (paper-style)

> AxBench demonstrates that simple linear discriminative directions—especially Diff Mean—consistently outperform LAT, PCA, and sparse autoencoders in both concept detection and causal steering effectiveness.

If you want, I can:

* map **Diff Mean ↔ ExpertLens AP experts**
* explain **why PCA fails geometrically**
* suggest which method best matches *your RSA / neuroscience setup*
