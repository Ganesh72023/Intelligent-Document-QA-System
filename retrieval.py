"""
retrieval.py - Semantic retrieval using FAISS for fast nearest-neighbor search.

Supports configurable top-k, multiple retrieval configurations, and optional reranking.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from .chunking import Chunk, ChunkingConfig, create_chunks, CHUNKING_CONFIGS
from .embeddings import EmbeddingModel, save_embeddings, load_embeddings

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS not available. Using brute-force cosine similarity.")


@dataclass
class RetrievalConfig:
    name: str
    chunking: ChunkingConfig
    top_k: int = 5
    rerank: bool = False
    description: str = ""

    @property
    def setting_str(self) -> str:
        rerank_str = "rerank=yes" if self.rerank else "rerank=no"
        return f"chunk={self.chunking.chunk_size_words}w, top_k={self.top_k}, {rerank_str}"


RETRIEVAL_CONFIGS: Dict[str, RetrievalConfig] = {
    "A": RetrievalConfig(
        name="A",
        chunking=CHUNKING_CONFIGS["small"],
        top_k=3,
        rerank=False,
        description="Small chunks (300w), top-3, no reranking. Fast, low token cost.",
    ),
    "B": RetrievalConfig(
        name="B",
        chunking=CHUNKING_CONFIGS["medium"],
        top_k=5,
        rerank=False,
        description="Medium chunks (500w), top-5, no reranking. Balanced.",
    ),
    "C": RetrievalConfig(
        name="C",
        chunking=CHUNKING_CONFIGS["large"],
        top_k=5,
        rerank=True,
        description="Large chunks (800w), top-5, with reranking. Best quality, highest cost.",
    ),
}


@dataclass
class RetrievalResult:
    query: str
    retrieved: List[Tuple[float, Chunk]]  # (score, chunk)
    retrieval_time_ms: float
    token_count: int
    config: RetrievalConfig

    @property
    def top_chunk(self) -> Optional[Chunk]:
        return self.retrieved[0][1] if self.retrieved else None


class FaissIndex:
    """
    FAISS-based vector index for fast nearest-neighbor search.
    Falls back to brute-force cosine similarity if FAISS is unavailable.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._index = None
        self._chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk], model: EmbeddingModel) -> None:
        """Build the FAISS index from chunks."""
        logger.info("Building index for %d chunks...", len(chunks))
        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, show_progress=True)

        if HAS_FAISS:
            index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)
            index.add(embeddings.astype("float32"))
            self._index = index
            logger.info("FAISS IndexFlatIP built with %d vectors", index.ntotal)
        else:
            self._index = embeddings  # Store raw embeddings for brute-force
            logger.info("Brute-force index built with %d vectors", len(embeddings))

        self._chunks = chunks
        self._embeddings = embeddings

    def search(
        self,
        query: str,
        model: EmbeddingModel,
        top_k: int = 5,
    ) -> List[Tuple[float, Chunk]]:
        """
        Search for the most similar chunks to the query.
        
        Returns:
            List of (score, chunk) tuples, sorted by descending score.
        """
        if not self._chunks:
            logger.warning("Index is empty. Build the index first.")
            return []

        q_emb = model.encode_single(query).astype("float32")

        if HAS_FAISS and isinstance(self._index, faiss.Index):
            scores, indices = self._index.search(
                q_emb.reshape(1, -1), min(top_k, len(self._chunks))
            )
            results = [
                (float(scores[0][i]), self._chunks[indices[0][i]])
                for i in range(len(indices[0]))
                if indices[0][i] >= 0
            ]
        else:
            # Brute-force cosine similarity
            scores = np.dot(self._embeddings, q_emb)
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = [(float(scores[i]), self._chunks[i]) for i in top_indices]

        return results

    def save(self, path: Path) -> None:
        """Save the index to disk."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"chunks": self._chunks, "embeddings": self._embeddings}
        if HAS_FAISS and isinstance(self._index, faiss.Index):
            faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".pkl"), "wb") as f:
            import pickle
            pickle.dump(data, f)
        logger.info("Saved index to %s", path)

    def load(self, path: Path) -> bool:
        """Load the index from disk."""
        import pickle
        path = Path(path)
        pkl_path = path.with_suffix(".pkl")
        if not pkl_path.exists():
            return False
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._embeddings = data["embeddings"]
        faiss_path = path.with_suffix(".faiss")
        if HAS_FAISS and faiss_path.exists():
            self._index = faiss.read_index(str(faiss_path))
        else:
            self._index = self._embeddings
        logger.info("Loaded index with %d chunks", len(self._chunks))
        return True


def simple_rerank(
    results: List[Tuple[float, Chunk]],
    query: str,
) -> List[Tuple[float, Chunk]]:
    """
    Simple keyword-based reranking boost.
    
    Boosts chunks whose text contains query terms.
    This is a lightweight alternative to a cross-encoder reranker.
    """
    query_terms = set(query.lower().split())
    reranked = []
    for score, chunk in results:
        text_lower = chunk.text.lower()
        term_hits = sum(1 for t in query_terms if t in text_lower and len(t) > 3)
        # Boost score slightly for keyword hits
        boosted = score + term_hits * 0.02
        reranked.append((boosted, chunk))

    return sorted(reranked, key=lambda x: x[0], reverse=True)


WORDS_PER_TOKEN = 0.75  # Rough approximation for English text


class Retriever:
    """
    Main retriever that uses a FaissIndex with configurable chunking.
    Supports multiple retrieval configurations for experimentation.
    """

    def __init__(
        self,
        pages: list,
        model: EmbeddingModel = None,
        cache_dir: Path = None,
    ):
        self.pages = pages
        self.model = model or EmbeddingModel.get_instance()
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self._indices: Dict[str, FaissIndex] = {}
        self._chunks_by_config: Dict[str, List[Chunk]] = {}

    def _get_index(self, config: RetrievalConfig) -> FaissIndex:
        """Get or build the FAISS index for a configuration."""
        key = config.chunking.name
        if key not in self._indices:
            index = FaissIndex(dimension=self.model.dimension)
            cache_path = self.cache_dir / f"index_{key}"
            if not index.load(cache_path):
                # Build from scratch
                chunks = create_chunks(self.pages, config.chunking)
                self._chunks_by_config[key] = chunks
                index.build(chunks, self.model)
                index.save(cache_path)
            self._indices[key] = index
        return self._indices[key]

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig = None,
    ) -> RetrievalResult:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: Question or search query.
            config: Retrieval configuration to use.
        
        Returns:
            RetrievalResult with retrieved chunks and metrics.
        """
        config = config or RETRIEVAL_CONFIGS["B"]
        t_start = time.time()

        index = self._get_index(config)
        results = index.search(query, self.model, top_k=config.top_k)

        if config.rerank and results:
            results = simple_rerank(results, query)

        retrieval_time_ms = (time.time() - t_start) * 1000

        # Estimate token count for context
        total_words = sum(c.word_count for _, c in results)
        token_count = int(total_words / WORDS_PER_TOKEN)

        logger.debug(
            "Retrieved %d chunks in %.1f ms (config=%s, tokens~%d)",
            len(results), retrieval_time_ms, config.name, token_count,
        )

        return RetrievalResult(
            query=query,
            retrieved=results,
            retrieval_time_ms=retrieval_time_ms,
            token_count=token_count,
            config=config,
        )

    def build_context(
        self,
        result: RetrievalResult,
        max_tokens: int = 3000,
    ) -> str:
        """
        Build a context string for the LLM from retrieved chunks.
        
        Each chunk is labeled as [SOURCE N] for citation tracking.
        """
        parts = []
        estimated_tokens = 0

        for i, (score, chunk) in enumerate(result.retrieved, 1):
            chunk_tokens = int(chunk.word_count / WORDS_PER_TOKEN)
            if estimated_tokens + chunk_tokens > max_tokens:
                break

            parts.append(
                f"[SOURCE {i}]\n"
                f"Paper: {chunk.paper_title} (arXiv:{chunk.arxiv_id}v{chunk.version})\n"
                f"Page: {chunk.page_number}"
                + (f", Section: {chunk.section}" if chunk.section else "")
                + f"\nSimilarity: {score:.3f}\n\n"
                f"{chunk.text}\n"
            )
            estimated_tokens += chunk_tokens

        return "\n---\n".join(parts)
