"""
question_analyzer.py - Analyzes questions to determine topic and answerability.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

PAPER_TOPICS = {
    "attention_transformer": [
        "attention", "transformer", "self-attention", "multi-head", "positional encoding",
        "encoder", "decoder", "scaled dot-product", "query", "key", "value",
        "feed-forward", "layer normalization", "residual", "bleu", "translation",
        "attention is all you need", "1706.03762",
    ],
    "moshi": [
        "moshi", "speech", "audio", "real-time", "dialogue", "spoken",
        "text-to-speech", "speech-text", "streaming audio", "acoustic",
        "helium", "inner monologue", "2410.00037",
    ],
    "dynamic_chunking": [
        "dynamic chunk", "hierarchical", "sequence modeling", "chunk boundary",
        "end-to-end", "variable length", "segmentation", "2507.07955",
        "dynamic chunking",
    ],
}

OUT_OF_SCOPE_INDICATORS = [
    "stock price", "weather", "recipe", "movie", "sport", "news",
    "president", "capital city", "population", "covid", "election",
    "cryptocurrency", "bitcoin", "medical diagnosis",
]


@dataclass
class QuestionAnalysis:
    question: str
    detected_topics: List[str]
    is_answerable: bool
    confidence: float
    needs_multi_paper: bool
    out_of_scope: bool
    reasoning: str


def analyze_question(question: str) -> QuestionAnalysis:
    """
    Analyze a question to determine whether it can be answered from the papers.
    
    Returns:
        QuestionAnalysis with topic detection and answerability assessment.
    """
    q_lower = question.lower()

    # Check for obviously out-of-scope questions
    out_of_scope = any(indicator in q_lower for indicator in OUT_OF_SCOPE_INDICATORS)
    if out_of_scope:
        return QuestionAnalysis(
            question=question,
            detected_topics=[],
            is_answerable=False,
            confidence=0.9,
            needs_multi_paper=False,
            out_of_scope=True,
            reasoning="Question is about topics outside the three research papers.",
        )

    # Detect which papers are relevant
    detected_topics = []
    for topic, keywords in PAPER_TOPICS.items():
        if any(kw in q_lower for kw in keywords):
            detected_topics.append(topic)

    needs_multi_paper = len(detected_topics) > 1
    is_answerable = len(detected_topics) > 0

    if is_answerable:
        # Confidence based on keyword match strength
        all_matches = sum(
            sum(1 for kw in kws if kw in q_lower)
            for topic, kws in PAPER_TOPICS.items()
            if topic in detected_topics
        )
        confidence = min(0.95, 0.5 + all_matches * 0.1)
        reasoning = f"Question matches topics: {', '.join(detected_topics)}"
    else:
        confidence = 0.3
        reasoning = "No clear match to paper topics found; retrieval may still surface relevant content."
        is_answerable = True  # Let retrieval decide

    return QuestionAnalysis(
        question=question,
        detected_topics=detected_topics,
        is_answerable=is_answerable,
        confidence=confidence,
        needs_multi_paper=needs_multi_paper,
        out_of_scope=False,
        reasoning=reasoning,
    )
