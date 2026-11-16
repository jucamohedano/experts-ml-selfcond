"""
The text similarity of two concepts: TF-IDF cosine of their positive probing sentences.

Used by module 2's text baseline (section 2.3) and module 5's text cells and controls.

Explanations: documentation/module_2_similarities.md, section 2.3.
"""

import functools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from core.probing_sentences import load_positive_documents


@functools.lru_cache(maxsize=2)
def text_similarity_matrix(dataset_dir: str, concepts: tuple) -> np.ndarray:
    """Cosine between the TF-IDF vectors of every two concepts' positive sentences, NaN for a concept without sentences."""
    documents = load_positive_documents(dataset_dir, concepts)
    vectors = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(documents)
    similarity = (vectors @ vectors.T).toarray()
    missing = np.array([not document for document in documents])
    similarity[missing, :] = np.nan
    similarity[:, missing] = np.nan
    return similarity


def text_similarity_frame(dataset_dir, concepts: list) -> pd.DataFrame | None:
    """text_similarity_matrix as a square frame labelled by concept, None without a dataset folder."""
    if dataset_dir is None:
        return None
    concepts = list(concepts)
    return pd.DataFrame(text_similarity_matrix(str(dataset_dir), tuple(concepts)), index=concepts, columns=concepts)
