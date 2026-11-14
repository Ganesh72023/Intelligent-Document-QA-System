"""
test_retrieval.py - Unit tests for document loading, chunking, and retrieval logic.
"""

from pathlib import Path
import pytest
from src.document_loader import PageContent, detect_section
from src.chunking import ChunkingConfig, chunk_page, create_chunks
from src.retrieval import RetrievalConfig, simple_rerank


def test_detect_section():
    text = "1. Introduction\nDeep learning models have evolved..."
    sec = detect_section(text)
    assert sec == "1. Introduction"


def test_chunking_with_overlap():
    words = "word " * 100
    page = PageContent(
        paper_key="test_paper",
        paper_title="Test Title",
        arxiv_id="1234.5678",
        version=1,
        page_number=1,
        text=words,
        section="Intro",
        word_count=100,
    )
    cfg = ChunkingConfig(chunk_size_words=40, overlap_words=10, min_chunk_words=5, respect_paragraphs=False)
    counter = [0]
    chunks = chunk_page(page, cfg, counter)

    assert len(chunks) >= 3
    assert chunks[0].paper_key == "test_paper"
    assert chunks[0].page_number == 1
    assert "Test Title" in chunks[0].source_ref


def test_simple_rerank():
    class DummyChunk:
        def __init__(self, text):
            self.text = text

    c1 = DummyChunk("This text talks about general computer vision.")
    c2 = DummyChunk("This passage explains scaled dot-product attention and Transformer weights.")

    raw_results = [(0.75, c1), (0.70, c2)]
    reranked = simple_rerank(raw_results, "scaled dot-product attention")

    # c2 contains multiple query terms and should receive an empirical boost over c1
    assert reranked[0][1] == c2
