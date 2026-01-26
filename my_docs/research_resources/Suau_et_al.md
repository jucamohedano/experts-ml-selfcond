## Suau et al. - Self-conditioning Pre-Trained Language Models

* **Goal / idea:** TLMs contain internal *expert units* that detect concepts; you can *steer generation* by identifying those units and setting their activations to the values they take when the concept is present (a “self-conditioning” Product-of-Experts view). 

* **How experts are found:** For a concept c, compute each neuron’s **Average Precision (AP)** treating the neuron’s outputs as a predictor for presence of c; rank units by AP and pick top-k as experts. Intervention do(c,k) sets those units to their expected value given c. 

* **Models used / scope:** Experiments primarily on **GPT-2 family** (GPT2-M and GPT2-L) — GPT2-L shown for qualitative examples; comparisons (parity experiments) use GPT2-M. Method is model-specific (works per pretrained model). 

* **Key quantitative findings:**

  * Very **few units** suffice to steer generation (median **k ≈ 15** to reach gender parity; in some cases as few as 3 units in smaller models). 
  * Achieves **gender parity** at **lower perplexity** than FUDGE and PPLM-BoW (i.e., higher quality conditioned text). 
  * Conditioning strength (k) needed correlates strongly with model’s **unconditional bias** (easier to nudge concepts the model is already biased toward). 

* **Qualitative strengths:** Can condition on **fine-grained senses / homographs** and preserves linguistic plausibility better than external-expert methods (less repetition, lower collapse). 

* **Mechanistic claim / interpretation:** TLMs internally implement a conditional PoE—expert units form the internal conditional model p(y|x); intervening on top experts raises p(y|x) while preserving p(x). This explains why small, targeted interventions work. 

* **Practical advantages:** No fine-tuning or extra parameters; cheap to compute top experts for a concept (order of tens of seconds on GPU for moderate concept datasets). 

* **Limitations (authors note):** concept representation depends on available positive/negative example sets; experts treated greedily/independently (joint dependencies unexplored); turning experts *off* is not an effective removal strategy; potential for misuse (easy to steer models). 

If you want, I can extract the exact numbers/tables (e.g., median k, perplexity distributions, AP thresholds) or produce a 1-sentence comparison table between Suau et al. and Fedzechkina et al.


### Important questions I have answered about the paper:
Let me check Appendix B and section 5.4 more closely for information about layer-specific AP values:
Now let me check Appendix B (page 12) more carefully, and see if there are any figures showing layer-specific AP comparisons:
Based on my analysis of the paper, here are the answers to your two questions:

## 1. **Why did they choose these specific layers?**

The authors **do not explicitly explain why they chose these four linear layers** (A, Aproj, B, Bproj) from the transformer architecture. <doc=Suau et al. - 2023 - Self-conditioning pre-trained language models_1.pdf;page=4>

The paper simply states: "Details on the layers analyzed in TLM architectures are shown in Appendix B" <doc=Suau et al. - 2023 - Self-conditioning pre-trained language models_1.pdf;page=4>, and Appendix B only shows the visual diagram (Figure 5 and the related image) without providing justification for the choice. <doc=Suau et al. - 2023 - Self-conditioning pre-trained language models_1.pdf;page=12>

The most likely implicit reason is that these are the **primary computational layers** where the actual transformations happen:
- They capture both attention mechanisms (A, Aproj) and feed-forward processing (B, Bproj)
- These are where the "neurons" or "units" that could serve as experts naturally exist
- Other components (LayerNorm, residual connections, etc.) don't have the same kind of parametric units

---

## 2. **Do they claim attn.c_attn shows higher AP than attn.c_proj?**

**No, the paper does NOT make this claim** or provide layer-specific AP value comparisons. 

After reviewing the methodology (Section 4), experimental analysis (Section 5), and appendices, I found:

- They measure AP across **all units in all four layer types** and rank them together <doc=Suau et al. - 2023 - Self-conditioning pre-trained language models_1.pdf;page=3>
- Their method selects the **top-k experts irrespective of which layer they come from** <doc=Suau et al. - 2023 - Self-conditioning pre-trained language models_1.pdf;page=3>
- Section 5.4 validates that their AP-based ranking is effective, but **doesn't break down results by layer type** <doc=Suau et al. - 2023 - Self-conditioning pre-trained language models_1.pdf;page=7;page=8>

The paper treats all four linear layer types uniformly and doesn't discuss whether certain layer types tend to have higher AP values for concepts.