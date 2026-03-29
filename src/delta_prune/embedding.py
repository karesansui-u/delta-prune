"""Embedding adapters for semantic pre-filtering.

Optional dependency: install with `pip install delta-prune[fast]`
"""

from __future__ import annotations

from typing import Protocol


class Embedding(Protocol):
    """Minimal embedding interface: texts in, vectors out."""

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedding:
    """Local embedding via sentence-transformers (free, ~100ms per batch)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embedding pre-filter. "
                "Install with: pip install delta-prune[fast]"
            )
        self._model = SentenceTransformer(model)

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts)
        return [e.tolist() for e in embeddings]
