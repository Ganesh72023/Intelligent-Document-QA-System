"""
embeddings.py - Generates and manages embeddings using Sentence Transformers.

Uses a lightweight model suitable for semantic similarity search.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast, good quality
EMBEDDING_DIM = 384


HAS_ST = True  # Loaded lazily inside _load


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for generating text embeddings.
    
    Caches the model and supports batch encoding for efficiency.
    """

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @classmethod
    def get_instance(cls, model_name: str = EMBEDDING_MODEL) -> "EmbeddingModel":
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def _load(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("sentence-transformers not installed.") from e
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded. Dim: %d", self._model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.
        
        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.
        
        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()


def save_embeddings(
    embeddings: np.ndarray,
    metadata: list,
    path: Path,
) -> None:
    """Save embeddings and metadata to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "metadata": metadata}, f)
    logger.info("Saved %d embeddings to %s", len(embeddings), path)


def load_embeddings(path: Path) -> tuple:
    """Load embeddings and metadata from disk."""
    with open(Path(path), "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["metadata"]
