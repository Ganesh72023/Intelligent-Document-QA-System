"""
document_loader.py - Loads and extracts text from the three research papers using PyMuPDF.

Preserves page numbers, section headings, and source metadata.
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    logger.error("PyMuPDF (fitz) not installed. Run: pip install pymupdf")


PAPER_METADATA = {
    "attention_is_all_you_need": {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "version": 7,
        "url": "https://arxiv.org/pdf/1706.03762v7",
        "filename": "attention_is_all_you_need.pdf",
    },
    "moshi": {
        "title": "Moshi: A Speech-Text Foundation Model for Real-Time Dialogue",
        "arxiv_id": "2410.00037",
        "version": 1,
        "url": "https://arxiv.org/pdf/2410.00037v1",
        "filename": "moshi.pdf",
    },
    "dynamic_chunking": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955",
        "version": 2,
        "url": "https://arxiv.org/pdf/2507.07955v2",
        "filename": "dynamic_chunking.pdf",
    },
}

SECTION_PATTERNS = [
    r"^\d+\.?\s+[A-Z][A-Za-z\s]+$",           # "1. Introduction"
    r"^[A-Z][A-Z\s]{4,}$",                       # "ABSTRACT", "INTRODUCTION"
    r"^\d+\.\d+\.?\s+[A-Za-z\s]+$",              # "2.1 Background"
    r"^Abstract$",
    r"^References$",
    r"^Appendix",
]

SECTION_RE = [re.compile(p, re.MULTILINE) for p in SECTION_PATTERNS]


@dataclass
class PageContent:
    paper_key: str
    paper_title: str
    arxiv_id: str
    version: int
    page_number: int
    text: str
    section: str = ""
    word_count: int = 0


def detect_section(text: str, prev_section: str = "") -> str:
    """
    Detect section heading from page text.
    Returns the section name or the previous section if none detected.
    """
    lines = text.strip().split("\n")[:10]  # Check first 10 lines
    for line in lines:
        line = line.strip()
        if 3 < len(line) < 80:
            for pattern in SECTION_RE:
                if pattern.match(line):
                    return line
    return prev_section


def extract_pdf_text(pdf_path: Path, paper_key: str) -> List[PageContent]:
    """
    Extract text from a PDF file, page by page.
    
    Args:
        pdf_path: Path to the PDF file.
        paper_key: Key from PAPER_METADATA.
    
    Returns:
        List of PageContent objects with text and metadata.
    """
    if not HAS_FITZ:
        raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    meta = PAPER_METADATA[paper_key]
    pages = []
    current_section = ""

    doc = fitz.open(str(pdf_path))
    logger.info(
        "Extracting %s (%d pages) from %s",
        meta["title"], len(doc), pdf_path.name,
    )

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        if not text or len(text.strip()) < 20:
            logger.debug("Skipping nearly-empty page %d", page_num + 1)
            continue

        # Detect section
        current_section = detect_section(text, current_section)

        word_count = len(text.split())
        pages.append(
            PageContent(
                paper_key=paper_key,
                paper_title=meta["title"],
                arxiv_id=meta["arxiv_id"],
                version=meta["version"],
                page_number=page_num + 1,
                text=text,
                section=current_section,
                word_count=word_count,
            )
        )

    doc.close()
    logger.info(
        "Extracted %d pages from %s (total words: %d)",
        len(pages), meta["title"],
        sum(p.word_count for p in pages),
    )
    return pages


def save_extracted_text(pages: List[PageContent], output_dir: Path) -> Path:
    """Save extracted text to JSON for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pages:
        return output_dir

    paper_key = pages[0].paper_key
    output_path = output_dir / f"{paper_key}.json"
    data = [asdict(p) for p in pages]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Saved extracted text to %s", output_path)
    return output_path


def load_extracted_text(json_path: Path) -> List[PageContent]:
    """Load previously extracted text from JSON."""
    data = json.loads(Path(json_path).read_text())
    return [PageContent(**d) for d in data]


class DocumentLoader:
    """
    Loads all three research papers and manages their extracted content.
    """

    def __init__(self, papers_dir: Path, extracted_dir: Path):
        self.papers_dir = Path(papers_dir)
        self.extracted_dir = Path(extracted_dir)
        self.pages: Dict[str, List[PageContent]] = {}

    def load_all(self, force_extract: bool = False) -> Dict[str, List[PageContent]]:
        """
        Load all three papers, extracting text if needed.
        
        Returns:
            Dict mapping paper_key -> list of PageContent objects.
        """
        for paper_key, meta in PAPER_METADATA.items():
            pdf_path = self.papers_dir / meta["filename"]
            json_path = self.extracted_dir / f"{paper_key}.json"

            # Use cached extraction if available
            if json_path.exists() and not force_extract:
                logger.info("Loading cached extraction for %s", meta["title"])
                self.pages[paper_key] = load_extracted_text(json_path)
            elif pdf_path.exists():
                pages = extract_pdf_text(pdf_path, paper_key)
                save_extracted_text(pages, self.extracted_dir)
                self.pages[paper_key] = pages
            else:
                logger.error(
                    "PDF not found: %s. Please download it from %s",
                    pdf_path, meta["url"],
                )
                self.pages[paper_key] = []

        total = sum(len(v) for v in self.pages.values())
        logger.info(
            "Loaded %d pages total from %d papers",
            total, len(self.pages),
        )
        return self.pages

    @property
    def all_pages(self) -> List[PageContent]:
        """Return all pages from all papers, sorted by paper and page number."""
        result = []
        for pages in self.pages.values():
            result.extend(pages)
        return result

    def get_stats(self) -> dict:
        """Return extraction statistics."""
        return {
            paper_key: {
                "pages": len(pages),
                "words": sum(p.word_count for p in pages),
                "sections": list(set(p.section for p in pages if p.section)),
            }
            for paper_key, pages in self.pages.items()
        }
