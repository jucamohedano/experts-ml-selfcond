# Module 3: Concept-to-Category Expert Similarity

## Research question

Do the expert units associated with a concept overlap strongly with the expert units associated with the category that contains it, and would that support a hierarchical organization where category labels capture a meaningful parent structure?

## Analysis

Notation. For each item $c$ (concept or category label), let $E_c$ denote its set of expert units after AP filtering, where each element is a **(layer, unit) pair**:

$$E_c = \{(\ell, u) : \text{unit } u \text{ in layer } \ell \text{ is an expert for } c\} \subseteq \{1,\dots,L\} \times \mathcal{U}.$$

A neuron is identified by its (layer, unit) pair because the raw `unit` column holds only the neuron index within a layer, and these indices repeat across layers. The sets are therefore built by zipping `layer_idx` with `unit` (`set(zip(layer_idx, unit))` per concept), which gives $|E_c| = n_c$, the concept's expert count. The module iterates over every `(concept, category)` row of the metadata with a non-null category, that is, every hierarchical pair $(c, k)$ where $k$ is the parent category of concept $c$. Pairs where either $E_c$ or $E_k$ is empty are skipped.

### 3.1 Expert-set similarity between a concept and its parent category

**Mathematical formulation.** Two set-similarity metrics are computed for each pair $(c, k)$, both expressed as percentages. The Jaccard index measures *global equivalence* of the two sets, the intersection over the union:

$$J(c, k) = 100 \cdot \frac{|E_c \cap E_k|}{|E_c \cup E_k|} \in [0, 100].$$

$J$ is symmetric and penalizes any mismatch in either direction: it can only be high when the two sets are close to identical, so a small concept set inside a much larger category set scores low even if the concept's experts are all shared. The overlap coefficient instead measures *subset containment*, the intersection over the smaller set:

$$O(c, k) = 100 \cdot \frac{|E_c \cap E_k|}{\min(|E_c|,\, |E_k|)} \in [0, 100],$$

which reaches 100 whenever one set is entirely contained in the other, regardless of the size difference. Read together, the two metrics separate two hypotheses. A high $J$ would mean concept and category recruit *the same* experts, whereas a high $O$ with a low $J$ would mean the smaller set (typically the concept's or the category's, whichever is smaller) is largely *inside* the other, a signature compatible with hierarchical containment rather than identity. Note that $J(c,k) \le O(c,k)$ always, since $|E_c \cup E_k| \ge \min(|E_c|, |E_k|)$.

**Generated data structures.** One CSV, `category_concept_similarity_metrics.csv`, with one row per concept-category pair. Alongside the two percentages, the raw set sizes behind them are stored, because a percentage alone hides its scale: an overlap of 20% resting on a 15-expert concept set is far less stable than the same 20% resting on a 1,500-expert set (one shared expert more or less moves the former by nearly 7 points), which is the same small-denominator caveat module 1 (subchapter 1.3) raises for layer percentages.

| Column | Type | Symbol | Description |
|--------|------|--------|-------------|
| concept | string | $c$ | The concept being compared. |
| category | string | $k$ | The parent category label. |
| hierarchy | string | (none) | String describing the concept-category relationship ("$k$ -> $c$"). |
| jaccard_pct | float | $J(c,k)$ | Percentage similarity computed with the Jaccard index. |
| overlap_pct | float | $O(c,k)$ | Percentage similarity computed with the overlap coefficient. |
| shared_expert_units | int | $\|E_c \cap E_k\|$ | Raw number of (layer, unit) experts shared by concept and category. |
| concept_expert_units | int | $\|E_c\|$ | Size of the concept's expert set. |
| category_expert_units | int | $\|E_k\|$ | Size of the category's expert set. |

Example (head of `AP_0.6/3_category_concept_similarities/category_concept_similarity_metrics.csv` in `research_plots_qwen_richie_hsj_with_sublayer_analysis`):

| concept | category | hierarchy | jaccard_pct | overlap_pct | shared_expert_units | concept_expert_units | category_expert_units |
|---|---|---|---|---|---|---|---|
| bed | furniture | furniture -> bed | 16.7411 | 28.9575 | 75 | 264 | 259 |
| bench | furniture | furniture -> bench | 5.2632 | 12.1547 | 22 | 181 | 259 |

Two **bar charts** plot the same table, one per metric, with each pair's `hierarchy` tick label colored in its category's color (matching the palette used by module 5's heatmaps), so the category blocks are readable directly from the axis:

- `jaccard_hierarchy.png`, `jaccard_pct` ($J(c,k)$, x-axis) plotted against `hierarchy` (y-axis, one bar per concept-category pair), showing Jaccard similarity as a percentage for each concept-category hierarchy.
- `overlap_hierarchy.png`, `overlap_pct` ($O(c,k)$, x-axis) plotted against `hierarchy` (y-axis), showing the overlap coefficient as a percentage for each concept-category hierarchy.

## Results

At AP=0.6, across the 147 concept-category pairs, Jaccard similarity is consistently low: mean 4.0%, median 2.5%, and 90% of pairs fall below 10%. By the more forgiving overlap-coefficient measure (which ignores how much larger the category's expert set is), the picture improves but is still modest: mean 15.2%, median 10.0%.

So a concept's expert set is, on average, far from identical to its category's expert set (low Jaccard), and even by the containment reading only a limited minority of the smaller set's experts are shared with the other (overlap near 15% on average). The bar charts make the pair-to-pair variation visible, but on their own they do not establish whether this is more than chance. Module 5's pairwise heatmaps test whether within-category pairs are systematically higher than across-category pairs, which is the sharper version of this question and the place where the categorical signal is clearest.

### Across AP thresholds

Both metrics decline steadily as the AP threshold tightens:

| AP | n pairs | Jaccard mean / median | Overlap mean / median | % pairs with jaccard < 10% |
|---|---|---|---|---|
| 0.5 | 147 | 5.6% / 4.1% | 17.4% / 15.3% | 85% |
| 0.6 | 147 | 4.0% / 2.5% | 15.2% / 10.0% | 90% |
| 0.7 | 147 | 2.6% / 0.8% | 13.2% / 4.8% | 95% |
| 0.8 | 128 | 1.6% / 0.0% | 7.6% / 0.0% | 95% |
| 0.9 | 59 | 0.6% / 0.0% | 4.5% / 0.0% | 97% |

Concept-to-own-category identity overlap is low at *every* threshold: even at the most lenient AP=0.5, the median pair shares only 4.1% Jaccard, and 85% of pairs sit below 10%. By AP=0.8 and 0.9, **the median concept-category pair shares zero experts at all** (median = 0.0% on both metrics). Part of the decline is mechanical, since stricter AP keeps fewer experts per concept overall, shrinking every set and therefore every intersection, and n also shrinks at AP=0.8/0.9 because some concepts have too few retained experts to compute a set-based similarity at all. The overall conclusion is that a concept and its category-label word recruit largely *different* specific neurons at every threshold tested. The hierarchical relationship, to the extent modules 4 and 5 detect one, is carried by a modest shared minority of experts rather than by set identity.
