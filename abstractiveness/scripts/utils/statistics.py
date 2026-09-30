"""
Textbook statistics shared by several files.

Used by the human-rating noise ceiling and by module 4's split-half ceiling.

Explanations: documentation/module_4_embedding_rsa.md.
"""

import numpy as np


def spearman_brown(reliability: float) -> float:
    """Spearman-Brown correction for a split-half reliability, 2r / (1 + r)."""
    if not np.isfinite(reliability) or reliability <= -1.0:
        return np.nan
    return float(2.0 * reliability / (1.0 + reliability))
