"""
evaluate_quality.py - Rigorous answer quality evaluation against the 17 ground truth questions.

Evaluates the RAG pipeline across three retrieval configurations:
- Config A: Chunk size 300 words, Top-k=3, Reranking=Disabled
- Config B: Chunk size 500 words, Top-k=5, Reranking=Disabled
- Config C: Chunk size 800 words, Top-k=5, Reranking=Enabled

Assigns standard grades:
- correct
- wrong
- refused_correctly
- refused_wrongly
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.document_loader import DocumentLoader, PAPER_METADATA
from src.chunking import CHUNKING_CONFIGS
from src.embeddings import EmbeddingModel
from src.retrieval import Retriever, RETRIEVAL_CONFIGS
from src.groq_client import GroqClient
from src.pipeline import RAGPipeline
from src.metrics import MetricsCollector, RequestRecord
from src.answer_validator import REFUSAL_PHRASES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_quality")

PAPERS_DIR = ROOT_DIR / "papers"
EXTRACTED_DIR = ROOT_DIR / "extracted_text"
RESULTS_DIR = ROOT_DIR / "results"
EVAL_DIR = ROOT_DIR / "evaluation"


def grade_response(question_id: str, answer_text: str, expected_grade: str, ground_truth: dict) -> str:
    """Grade response against ground truth facts."""
    ans_lower = answer_text.lower()
    is_refusal = any(p in ans_lower for p in REFUSAL_PHRASES)

    if expected_grade == "refused_correctly":
        return "refused_correctly" if is_refusal else "wrong"

    if expected_grade == "correct":
        if is_refusal:
            return "refused_wrongly"

        gt_entry = ground_truth.get(question_id, {})
        key_facts = gt_entry.get("key_facts", [])
        if not key_facts:
            return "correct" if len(answer_text) > 40 else "wrong"

        matches = 0
        for fact in key_facts:
            keywords = [w.lower().strip("(),.:^") for w in fact.split() if len(w) > 3 and w.isalpha()]
            hit_count = sum(1 for kw in keywords if kw in ans_lower)
            if keywords and (hit_count / len(keywords)) >= 0.3:
                matches += 1

        if matches >= max(1, len(key_facts) // 2):
            return "correct"
        return "wrong"

    return "wrong"


async def run_quality_evaluation(
    source: str = "stand-in",
    groq_api_key: str = "",
    metrics: Optional[MetricsCollector] = None,
) -> MetricsCollector:
    metrics = metrics or MetricsCollector()

    # Load questions and ground truth
    questions = json.loads((EVAL_DIR / "test_questions.json").read_text(encoding="utf-8"))
    ground_truth = json.loads((EVAL_DIR / "ground_truth.json").read_text(encoding="utf-8"))

    # Load documents
    loader = DocumentLoader(PAPERS_DIR, EXTRACTED_DIR)
    loader.load_all()
    all_pages = loader.all_pages
    logger.info("Loaded %d pages from 3 papers", len(all_pages))

    # Initialize Embeddings & Retriever
    emb_model = EmbeddingModel.get_instance()
    retriever = Retriever(all_pages, emb_model, cache_dir=ROOT_DIR / "cache")

    # Setup client
    if source == "real" and groq_api_key:
        client = GroqClient(api_key=groq_api_key)
        logger.info("Using REAL Groq API client with model: %s", client.model)
    else:
        # Stand-in client pointed at local simulator
        client = GroqClient(api_key="stand-in", model="stand-in-llama", api_base="http://127.0.0.1:8765/v1")
        logger.info("Using STAND-IN model service at %s", client.api_base)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as reset_client:
                await reset_client.post("http://127.0.0.1:8765/config", json={
                    "rpm_limit": 60,
                    "daily_request_budget": 10000,
                    "daily_token_budget": 500000,
                    "failure_probability": 0.0,
                    "rate_limit_probability": 0.0,
                    "base_latency_ms": 300.0,
                    "latency_variation_ms": 100.0,
                })
                await reset_client.post("http://127.0.0.1:8765/reset")
            logger.info("Reset stand-in server for evaluation suite.")
        except Exception as e:
            logger.warning("Could not reset stand-in server: %s", e)

    # Evaluate across all three retrieval configurations
    for config_key in ["A", "B", "C"]:
        retrieval_cfg = RETRIEVAL_CONFIGS[config_key]
        scenario_name = f"retrieval_config_{config_key}"
        logger.info("--- Starting Evaluation for Config %s (%s) ---", config_key, retrieval_cfg.setting_str)

        pipeline = RAGPipeline(
            retriever=retriever,
            groq_client=client,
            metrics=metrics,
            retrieval_config=retrieval_cfg,
            scenario=scenario_name,
            source=source,
        )

        for q_item in questions:
            qid = q_item["id"]
            question = q_item["question"]
            expected = q_item["expected_grade"]

            res = await pipeline.process(question, request_id=f"{config_key}-{qid}", stream=True)
            grade = grade_response(qid, res.answer, expected, ground_truth)

            if res.record:
                res.record.answer_grade = grade
                res.record.retrieval_setting = f"Config_{config_key}_{retrieval_cfg.setting_str}"

            logger.info("Config %s | %s: %s (grade: %s)", config_key, qid, question[:45], grade)
            await asyncio.sleep(0.1)

    return metrics


if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY", "")
    src = "real" if api_key else "stand-in"
    m = asyncio.run(run_quality_evaluation(source=src, groq_api_key=api_key))
    csv_file = RESULTS_DIR / "results.csv"
    m.save_csv(csv_file)
    logger.info("Evaluation finished. CSV saved to %s", csv_file)
