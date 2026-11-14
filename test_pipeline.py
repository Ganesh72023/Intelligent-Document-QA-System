"""
test_pipeline.py - Tests for the multi-step RAG pipeline, question analyzer, and validator.
"""

import pytest
from src.question_analyzer import analyze_question
from src.answer_validator import validate_answer, extract_citations


def test_question_analyzer_identifies_attention():
    q = "What is the scaled dot-product attention in Transformer?"
    res = analyze_question(q)
    assert res.is_answerable is True
    assert "attention_transformer" in res.detected_topics
    assert res.out_of_scope is False


def test_question_analyzer_flags_out_of_scope():
    q = "What is the capital city of France?"
    res = analyze_question(q)
    assert res.out_of_scope is True
    assert res.is_answerable is False


def test_extract_citations():
    text = "The model uses multi-head attention [SOURCE 1] and [SOURCE 2] on page 5."
    citations = extract_citations(text)
    assert any("SOURCE 1" in c for c in citations)
    assert any("SOURCE 2" in c for c in citations)


def test_validate_answer_refusal():
    ans = "The provided research papers do not contain sufficient information to answer this question."
    val = validate_answer(ans, [], "What is the stock price of Apple?")
    assert val.is_valid is True
    assert val.is_refusal is True


def test_validate_answer_missing_citation():
    ans = "Transformers use 8 attention heads in the base model."
    val = validate_answer(ans, [("chunk1", None)], "How many heads?")
    assert val.has_citations is False
    assert val.is_valid is False
    assert any("lacks explicit citations" in issue for issue in val.issues)
