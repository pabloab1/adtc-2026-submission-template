"""
Pluggable text-embedding layer.

Why this exists: retrieval.py should never care HOW text gets turned into
vectors, only that it can call .fit(texts) once and .encode(texts) after.
That means you can swap TfidfEmbedder for a real embedding API (OpenAI,
Anthropic, Cohere, a local sentence-transformers model, etc.) later — when
you add the PDF-based knowledge base — by writing one new class here and
changing a single line in retrieval.py.

TfidfEmbedder is the default because it needs no API key and no model
download, so the test set works immediately. It's a bag-of-words method:
it matches on shared words, not meaning, so it works best when the query
uses similar vocabulary to the stored question. That's a real limitation
to know about — see README.md.
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Embedder(ABC):
    """Base interface every embedder must implement."""

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Learn vocabulary/parameters from the full corpus of texts."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Turn a list of strings into a 2D array of vectors."""
        raise NotImplementedError

    def similarity(self, query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
        """Cosine similarity between one query vector and many corpus vectors."""
        return cosine_similarity(query_vec, corpus_vecs)[0]


class TfidfEmbedder(Embedder):
    """
    Default embedder. Fits one shared TF-IDF vectorizer across all
    languages' text at once (word-level, char n-grams as backup for
    languages with heavy diacritics like Yoruba, where word-level tokens
    can be sparse). No external downloads required.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",   # character n-grams: robust to diacritics/tone marks and small datasets
            ngram_range=(2, 4),
            lowercase=True,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit(texts) before .encode(texts).")
        return self.vectorizer.transform(texts).toarray()


# ---------------------------------------------------------------------------
# STUB for later: uncomment and fill in when you're ready to plug in a real
# embedding API for the PDF-based knowledge base. retrieval.py doesn't need
# to change — just swap which class gets instantiated in retrieval.py's
# `build_index()`.
# ---------------------------------------------------------------------------
#
# class APIEmbedder(Embedder):
#     def __init__(self, client, model_name: str):
#         self.client = client
#         self.model_name = model_name
#
#     def fit(self, texts: list[str]) -> None:
#         pass  # API embedders don't need a separate fit step
#
#     def encode(self, texts: list[str]) -> np.ndarray:
#         response = self.client.embeddings.create(model=self.model_name, input=texts)
#         return np.array([d.embedding for d in response.data])
