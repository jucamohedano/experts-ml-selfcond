# Logical fixes

This file records logical corrections made to the analysis pipeline. It is kept short and is meant to be extended as further fixes are applied.

## Expert identity keyed on (layer, unit) pairs (modules 3 and 5)

The expertise data identifies each expert neuron by a `unit` index that is unique only within a single layer, because the same index is reused across the 48 layers. Modules 3 and 5 previously grouped experts by the bare `unit` value, which merged identically indexed neurons from different layers into one element. This raised the intersection between concepts that shared nothing more than an index coincidence, adding a roughly uniform baseline of similarity to every pair and compressing the contrast between within-category and across-category pairs. For example, the module 5 Jaccard within-versus-across ratio at AP=0.6 was about 1.4x under the old keying and 6.6x after the correction. Both modules now key expert sets on the `(layer_idx, unit)` pair, so that two experts are counted as shared only when they are the same neuron in the same layer.
