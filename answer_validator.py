"""
answer_validator.py - Validates that generated answers cite retrieved sources correctly.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

REFUSAL_PHRASES = [
    "do not contain sufficient information",
    "provided research papers do not",
    "cannot answer",
    "not covered in",
    "no information about",
    "outside the scope",
    "not mentioned in",
    "unable to find",
]


@dataclass
class ValidationResult:
    is_valid: bool
    has_citations: bool
    citations_verified: bool
    unsupported_claims: List[str]
    is_refusal: bool
    confidence: float
    issues: List[str]


def extract_citations(text: str) -> List[str]:
    """Extract citation references from answer text."""
    # Match patterns like [SOURCE 1], [1706.03762], (page 5), arXiv:xxxx
    patterns = [
        r'\[SOURCE \d+\]',
        r'\[arXiv:\d+\.\d+\]',
        r'arXiv:\d+\.\d+',
        r'\(page \d+\)',
        r'\[p\.\s*\d+\]',
        r'page\s+\d+',
    ]
    citations = []
    for pat in patterns:
        citations.extend(re.findall(pat, text, re.IGNORECASE))
    return list(set(citations))


def validate_answer(
    answer: str,
    retrieved_chunks: List[Tuple[float, Any]],
    question: str,
) -> ValidationResult:
    """
    Validate that the answer is grounded in the retrieved sources.
    
    Checks:
    1. Answer is not empty.
    2. Answer contains citations or references.
    3. Answer is a valid refusal when appropriate.
    4. No hallucinated paper names.
    """
    issues = []

    if not answer or len(answer.strip()) < 10:
        return ValidationResult(
            is_valid=False,
            has_citations=False,
            citations_verified=False,
            unsupported_claims=[],
            is_refusal=False,
            confidence=0.0,
            issues=["Answer is empty or too short."],
        )

    # Check for refusal
    is_refusal = any(phrase in answer.lower() for phrase in REFUSAL_PHRASES)
    if is_refusal:
        return ValidationResult(
            is_valid=True,
            has_citations=False,
            citations_verified=True,
            unsupported_claims=[],
            is_refusal=True,
            confidence=0.95,
            issues=[],
        )

    # Check for citations
    citations = extract_citations(answer)
    has_citations = len(citations) > 0

    if not has_citations:
        issues.append("Answer lacks explicit citations to source passages.")

    # Check that cited source numbers correspond to retrieved chunks
    source_nums = re.findall(r'\[SOURCE (\d+)\]', answer)
    max_source = len(retrieved_chunks)
    invalid_sources = [n for n in source_nums if int(n) > max_source]
    citations_verified = len(invalid_sources) == 0

    if invalid_sources:
        issues.append(f"Answer references non-existent sources: {invalid_sources}")

    # Check for hallucinated paper names (papers not in our library)
    hallucinated = []
    allowed_papers = ["attention is all you need", "moshi", "dynamic chunking", "1706.03762", "2410.00037", "2507.07955"]
    # Look for paper-like references in the answer
    paper_refs = re.findall(r'"([^"]{10,60})"', answer)
    for ref in paper_refs:
        if not any(allowed in ref.lower() for allowed in allowed_papers):
            # Could be a hallucinated paper
            hallucinated.append(ref)

    unsupported = hallucinated

    is_valid = has_citations and citations_verified and len(issues) == 0
    confidence = 0.9 if is_valid else max(0.1, 0.9 - len(issues) * 0.2)

    return ValidationResult(
        is_valid=is_valid,
        has_citations=has_citations,
        citations_verified=citations_verified,
        unsupported_claims=unsupported,
        is_refusal=False,
        confidence=confidence,
        issues=issues,
    )
