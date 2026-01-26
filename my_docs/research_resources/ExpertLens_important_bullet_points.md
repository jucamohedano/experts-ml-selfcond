## Fedzechkina et al. - ExpertLens Activation steering features are highly interpretable

**Best τ:** **τ = 0.5** — the paper reports the strongest human–model alignment at this threshold. 

**Summary (concise):**

* At **τ = 0.5** ExpertLens (Jaccard overlap of expert sets) gives the highest Spearman correlation with MEN human similarity (final checkpoint: **≈0.70 / 0.77 / 0.79** for 70m / 1b / 12b). Human–human agreement ≈ 0.84 for reference. 
* **As τ increases (0.6→0.9):** expert overlap and alignment to humans **decrease** — expert sets shrink (expert count falls approx. logarithmically with τ) so intersections become sparser and correlations drop.
* **Model-size effects:** larger models retain better alignment at higher τ (smaller models suffer more empty intersections at high τ). 
* **Learning dynamics / stability:** higher τ (more specialised experts) takes longer to stabilize during training — specialized experts emerge later. 
* **Layer distribution:** raising τ shifts expert prevalence (e.g., toward earlier MLP layers; attention-layer patterns become bimodal). 
* **Robustness:** threshold-free measures (cosine / KL on raw APs) show similar patterns, and ExpertLens at τ=0.5 outperforms standard embedding-based similarity.
* **Which models were tested for different τ?**
  Main experiments used the **Pythia family** (Pythia-70m, Pythia-1b, Pythia-12b) across multiple checkpoints (1, 512, 1k, 4k, 36k, 72k, 143k). The authors also show that the results generalize to **Gemma-2b** (and Gemma-2b-IFT) in the appendix, and they ran pilot dataset-generation with **GPT-4, Mistral-7b-Instruct,** and an internal 80b model.

* **Alignment = semantic similarity *and* structural / organizational match.**

  * **Semanticity (pairwise):** ExpertLens expert-set overlap (Jaccard at **τ = 0.5**) yields the highest Spearman correlation with human pairwise similarity (MEN): **≈0.70 / 0.77 / 0.79** for 70m / 1b / 12b (human–human ≈ 0.84). Expert-based measures outperform both single-word and sentence embeddings.
  * **Structural organization (concept/domain level):** Using expert overlap at τ = 0.5, ExpertLens **reconstructs human-interpretable concept domains** (clusters, domain-level Jaccard overlap, graphs shown for domains), i.e., it captures broader conceptual organization beyond pairwise similarity.
* **Background / definition:** prior work shows transformer neurons tend to be *polysemantic* (activate for multiple concepts). The authors pick this up as a core issue to examine. 

* **Empirical finding — experts *are* polysemantic:** individual expert neurons often appear in expert sets for related concepts (e.g., many neurons in the “cat” expert set also appear in the “dog” expert set).

* **-but along *human-interpretable* lines:** the polysemanticity of experts is structured — experts for semantically related items co-occur, while experts for unrelated domains (e.g., “cat” vs “car”) do **not** tend to overlap. In short: expert polysemanticity is meaningful, not random.

* **Non-expert polysemantic neurons exist but are uninformative:** some neurons activate for many unrelated concepts, but these are **not** predictive of those concepts and therefore do not show up in expert sets. The paper argues the predictive (expert) neurons are the ones that matter. 

* **Causal relevance of experts (implication for polysemanticity):** intervening on expert neurons increases the probability the model expresses the concept, whereas intervening on non-expert neurons does not — supporting that expert (even if polysemantic) neurons are key drivers. (See main text + App. B.)

* **Dataset / negative-set effects on apparent polysemanticity:** increasing negative-set size raises activation of polysemous neurons and *reduces* expert overlap at high τ (authors interpret this as larger negatives including concept-related sentences or activating more polysemous units). 

* **AP / activation magnitude nuance:** the authors report *no* difference in raw AP distributions between experts that are shared vs. not-shared across a concept pair — meaning shared-expert membership is not simply explained by higher raw AP magnitudes. 

* **Overall interpretation (authors’ claim):** ExpertLens tends to select polysemantic neurons that are *semantically coherent* (aligned with human categories), filtering out broadly polysemantic but uninformative units — hence ExpertLens yields interpretable, human-relevant polysemanticity.
