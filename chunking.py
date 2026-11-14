"""
chunking.py - Configurable text chunking with overlap and metadata preservation.

Implements word-based chunking with configurable sizes and overlap.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from .document_loader import PageContent

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    paper_key: str
    paper_title: str
    arxiv_id: str
    version: int
    page_number: int
    section: str
    text: str
    chunk_index: int
    total_chunks_in_page: int
    word_count: int
    char_count: int

    @property
    def source_ref(self) -> str:
        """Human-readable source reference."""
        return (
            f"{self.paper_title} (arXiv:{self.arxiv_id}v{self.version})"
            f" - Page {self.page_number}"
            + (f", Section: {self.section}" if self.section else "")
        )


@dataclass
class ChunkingConfig:
    chunk_size_words: int = 500       # Target chunk size in words
    overlap_words: int = 50           # Word overlap between consecutive chunks
    min_chunk_words: int = 20         # Minimum words per chunk (smaller chunks discarded)
    respect_paragraphs: bool = True   # Try to break at paragraph boundaries

    @property
    def name(self) -> str:
        return f"w{self.chunk_size_words}-o{self.overlap_words}"


CHUNKING_CONFIGS = {
    "small": ChunkingConfig(chunk_size_words=300, overlap_words=30),
    "medium": ChunkingConfig(chunk_size_words=500, overlap_words=50),
    "large": ChunkingConfig(chunk_size_words=800, overlap_words=80),
}


def _split_words(text: str) -> List[str]:
    """Split text into words, preserving whitespace signals."""
    return text.split()


def _words_to_text(words: List[str]) -> str:
    """Rejoin words into text."""
    return " ".join(words)


def chunk_page(
    page: PageContent,
    config: ChunkingConfig,
    global_chunk_counter: List[int],
) -> List[Chunk]:
    """
    Chunk a single page of text.
    
    Strategy:
    1. Split by paragraphs if respect_paragraphs=True.
    2. Build chunks of approximately chunk_size_words words.
    3. Add overlap_words from the previous chunk.
    4. Keep metadata from the source page.
    """
    text = page.text.strip()
    if not text:
        return []

    # Split into paragraphs first
    if config.respect_paragraphs:
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
    else:
        paragraphs = [text]

    # Flatten all words with paragraph break markers
    all_words = []
    para_boundaries = set()
    for para in paragraphs:
        if all_words:
            para_boundaries.add(len(all_words))
        all_words.extend(_split_words(para))

    if not all_words:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(all_words):
        end = min(start + config.chunk_size_words, len(all_words))

        # Try to extend to a paragraph boundary if close
        if config.respect_paragraphs and end < len(all_words):
            # Find nearest paragraph boundary after end
            candidates = [b for b in para_boundaries if start < b <= end + 30]
            if candidates:
                end = max(candidates)

        chunk_words = all_words[start:end]
        if len(chunk_words) < config.min_chunk_words:
            # Merge tiny trailing chunk into previous
            if chunks:
                prev = chunks[-1]
                merged_text = prev.text + " " + _words_to_text(chunk_words)
                chunks[-1] = Chunk(
                    chunk_id=prev.chunk_id,
                    paper_key=prev.paper_key,
                    paper_title=prev.paper_title,
                    arxiv_id=prev.arxiv_id,
                    version=prev.version,
                    page_number=prev.page_number,
                    section=prev.section,
                    text=merged_text,
                    chunk_index=prev.chunk_index,
                    total_chunks_in_page=prev.total_chunks_in_page,
                    word_count=len(merged_text.split()),
                    char_count=len(merged_text),
                )
            break

        chunk_text = _words_to_text(chunk_words)
        global_chunk_counter[0] += 1
        chunk = Chunk(
            chunk_id=f"{page.paper_key}-p{page.page_number}-c{chunk_index}",
            paper_key=page.paper_key,
            paper_title=page.paper_title,
            arxiv_id=page.arxiv_id,
            version=page.version,
            page_number=page.page_number,
            section=page.section,
            text=chunk_text,
            chunk_index=chunk_index,
            total_chunks_in_page=0,  # filled in below
            word_count=len(chunk_words),
            char_count=len(chunk_text),
        )
        chunks.append(chunk)
        chunk_index += 1

        # Move forward with overlap, ensuring strictly positive progress
        next_start = end - config.overlap_words
        if next_start <= start or end >= len(all_words):
            next_start = end
        start = next_start

    # Update total_chunks_in_page
    for c in chunks:
        c.total_chunks_in_page = len(chunks)

    return chunks


def create_chunks(
    pages: List[PageContent],
    config: ChunkingConfig = None,
) -> List[Chunk]:
    """
    Create chunks from all pages with the given configuration.
    
    Args:
        pages: Extracted pages from DocumentLoader.
        config: Chunking configuration.
    
    Returns:
        List of Chunk objects with metadata.
    """
    config = config or CHUNKING_CONFIGS["medium"]
    counter = [0]
    all_chunks = []

    for page in pages:
        chunks = chunk_page(page, config, counter)
        all_chunks.extend(chunks)

    logger.info(
        "Created %d chunks from %d pages (config: %s)",
        len(all_chunks), len(pages), config.name,
    )
    return all_chunks


def get_chunk_stats(chunks: List[Chunk]) -> dict:
    """Return statistics about chunks."""
    if not chunks:
        return {}
    word_counts = [c.word_count for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_words": sum(word_counts) / len(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "by_paper": {
            key: sum(1 for c in chunks if c.paper_key == key)
            for key in set(c.paper_key for c in chunks)
        },
    }
