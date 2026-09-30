"""
Layer-distribution measures: each word's layer probability profile and its Geary's C.

Shared by modules 1, 3 and 5.

Explanations: documentation/module_1_expert_distribution.md.
"""

import numpy as np
import pandas as pd


PEAK_GAP_RELIABLE_PP = 1.0


def gearys_c(values) -> float:
    """Geary's C spatial autocorrelation of a 1-D sequence (e.g."""
    if values is None or len(values) < 3:
        return np.nan
    values = np.asarray(values, dtype=float)
    sum_of_squares = np.sum((values - values.mean()) ** 2)
    if sum_of_squares == 0:
        return np.nan
    return float(np.sum(np.diff(values) ** 2) / (2 * sum_of_squares))


def build_layer_probability_matrix(expert_allocation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a concept-by-layer expert count matrix and its row-normalized probability matrix."""
    count_matrix = expert_allocation_df.groupby(['concept', 'layer_idx'], observed=False).size().unstack(fill_value=0)
    layer_support = [int(str(name).split('.', 1)[0]) for name in expert_allocation_df['layer_name'].cat.categories]
    count_matrix = count_matrix.reindex(columns=layer_support, fill_value=0)
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return count_matrix, prob_matrix
