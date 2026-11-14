"""
pipeline.py - Multi-step RAG pipeline integrating retrieval, generation, and validation.

Steps:
  1. Receive and analyze the question
  2. Retrieve relevant paper passages
  3. Check retrieval relevance
  4. Call Groq (or stand-in) to generate response
  5. Validate answer and citations
  6. Retry retrieval if evidence insufficient (bounded)
  7. Return final response
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .question_analyzer import analyze_question, QuestionAnalysis
from .retrieval import Retriever, RetrievalConfig, RetrievalResult, RETRIEVAL_CONFIGS
from .groq_client import GroqClient, GroqResponse
from .answer_validator import validate_answer, ValidationResult
from .metrics import RequestRecord, MetricsCollector
from .rate_limiter import RateLimiter
from .retry_manager import RetryManager, RetryConfig

logger = logging.getLogger(__name__)

BUDGET_EXHAUSTED_MSG = (
    "Your request cannot be processed right now because the service's daily model usage "
    "budget has been reached. Your request has not been sent to the model. "
    "Please try again after the budget resets."
)


@dataclass
class PipelineStep:
    name: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class PipelineResult:
    request_id: str
    question: str
    answer: str
    success: bool
    steps: List[PipelineStep]
    retrieval_result: Optional[RetrievalResult] = None
    groq_response: Optional[GroqResponse] = None
    validation: Optional[ValidationResult] = None
    analysis: Optional[QuestionAnalysis] = None
    record: Optional[RequestRecord] = None
    error: Optional[str] = None

    @property
    def total_ms(self) -> float:
        if self.steps:
            return self.steps[-1].end_ms - self.steps[0].start_ms
        return 0.0


class RAGPipeline:
    """
    Multi-step RAG pipeline with:
    - Question analysis
    - Document retrieval
    - Relevance checking
    - LLM generation (Groq or stand-in)
    - Answer validation
    - Retry loop for insufficient evidence
    """

    MAX_RETRIEVAL_RETRIES = 2
    MIN_RELEVANCE_SCORE = 0.3

    def __init__(
        self,
        retriever: Retriever,
        groq_client: GroqClient,
        rate_limiter: Optional[RateLimiter] = None,
        retry_manager: Optional[RetryManager] = None,
        metrics: Optional[MetricsCollector] = None,
        retrieval_config: RetrievalConfig = None,
        scenario: str = "default",
        source: str = "real",
        standin_url: Optional[str] = None,
    ):
        self.retriever = retriever
        self.groq_client = groq_client
        self.rate_limiter = rate_limiter
        self.retry_manager = retry_manager or RetryManager(RetryConfig())
        self.metrics = metrics or MetricsCollector()
        self.retrieval_config = retrieval_config or RETRIEVAL_CONFIGS["B"]
        self.scenario = scenario
        self.source = source
        self.standin_url = standin_url

    def _now_ms(self) -> float:
        return time.time() * 1000

    async def process(
        self,
        question: str,
        request_id: Optional[str] = None,
        stream: bool = True,
    ) -> PipelineResult:
        """
        Execute the full RAG pipeline for a question.
        
        Args:
            question: User question.
            request_id: Optional request ID for tracking.
            stream: Whether to use streaming for LLM generation.
        
        Returns:
            PipelineResult with answer, citations, and metrics.
        """
        request_id = request_id or str(uuid.uuid4())[:8]
        t_global_start = time.time()
        steps = []
        record = RequestRecord(
            request_id=request_id,
            scenario=self.scenario,
            source=self.source,
            start_time=t_global_start,
            retrieval_setting=self.retrieval_config.name,
            model=self.groq_client.model,
        )

        def step(name: str) -> PipelineStep:
            s = PipelineStep(name=name, start_ms=self._now_ms())
            steps.append(s)
            return s

        # ----------------------------------------------------------------
        # STEP 1: Analyze the question
        # ----------------------------------------------------------------
        s1 = step("question_analysis")
        analysis = analyze_question(question)
        s1.end_ms = self._now_ms()
        s1.details = {"topics": analysis.detected_topics, "answerable": analysis.is_answerable}
        logger.info("[%s] Q: %s | Topics: %s", request_id, question[:60], analysis.detected_topics)

        if analysis.out_of_scope:
            record.status = "ok"
            record.answer_grade = "refused_correctly"
            record.total_ms = (time.time() - t_global_start) * 1000
            self.metrics.add(record)
            return PipelineResult(
                request_id=request_id,
                question=question,
                answer=BUDGET_EXHAUSTED_MSG.replace("budget", "scope"),
                success=True,
                steps=steps,
                analysis=analysis,
                record=record,
            )

        # ----------------------------------------------------------------
        # STEP 2: Retrieve relevant passages (with bounded retry loop)
        # ----------------------------------------------------------------
        retrieval_result = None
        for retrieval_attempt in range(self.MAX_RETRIEVAL_RETRIES + 1):
            s2 = step(f"retrieval_attempt_{retrieval_attempt}")

            # Use wider top-k on retry
            config = self.retrieval_config
            if retrieval_attempt > 0:
                from dataclasses import replace
                config = replace(config, top_k=min(config.top_k + 2, 10))

            retrieval_result = self.retriever.retrieve(question, config)
            s2.end_ms = self._now_ms()
            s2.details = {
                "chunks": len(retrieval_result.retrieved),
                "tokens": retrieval_result.token_count,
                "time_ms": retrieval_result.retrieval_time_ms,
            }

            # ----------------------------------------------------------------
            # STEP 3: Check retrieval relevance
            # ----------------------------------------------------------------
            s3 = step("relevance_check")
            top_score = retrieval_result.retrieved[0][0] if retrieval_result.retrieved else 0.0
            relevant = top_score >= self.MIN_RELEVANCE_SCORE
            s3.end_ms = self._now_ms()
            s3.details = {"top_score": top_score, "relevant": relevant}

            if relevant or retrieval_attempt >= self.MAX_RETRIEVAL_RETRIES:
                break

            logger.info("[%s] Low relevance (%.3f), retrying retrieval", request_id, top_score)

        context = self.retriever.build_context(retrieval_result)

        # ----------------------------------------------------------------
        # STEP 4: Check rate limits and generate response
        # ----------------------------------------------------------------
        s4 = step("llm_generation")

        # Check daily budget first
        if self.rate_limiter and self.rate_limiter.is_daily_exhausted():
            s4.end_ms = self._now_ms()
            s4.success = False
            record.status = "rate_limited"
            record.total_ms = (time.time() - t_global_start) * 1000
            self.metrics.add(record)
            return PipelineResult(
                request_id=request_id,
                question=question,
                answer=BUDGET_EXHAUSTED_MSG,
                success=False,
                steps=steps,
                retrieval_result=retrieval_result,
                analysis=analysis,
                record=record,
            )

        # Acquire rate limit slot
        if self.rate_limiter:
            estimated_tokens = retrieval_result.token_count + 512  # context + expected output
            try:
                acquired = await self.rate_limiter.acquire(estimated_tokens=estimated_tokens)
                if not acquired:
                    s4.end_ms = self._now_ms()
                    s4.success = False
                    record.status = "rate_limited"
                    record.total_ms = (time.time() - t_global_start) * 1000
                    self.metrics.add(record)
                    return PipelineResult(
                        request_id=request_id,
                        question=question,
                        answer=BUDGET_EXHAUSTED_MSG,
                        success=False,
                        steps=steps,
                        retrieval_result=retrieval_result,
                        analysis=analysis,
                        record=record,
                    )
            except asyncio.TimeoutError:
                s4.end_ms = self._now_ms()
                s4.success = False
                record.status = "rate_limited"
                record.total_ms = (time.time() - t_global_start) * 1000
                self.metrics.add(record)
                return PipelineResult(
                    request_id=request_id,
                    question=question,
                    answer="Rate limit wait time exceeded. Please try again later.",
                    success=False,
                    steps=steps,
                    retrieval_result=retrieval_result,
                    analysis=analysis,
                    record=record,
                )

        # Call Groq with retry logic
        outcome = await self.retry_manager.execute_with_retry(
            self.groq_client.generate,
            question,
            context,
            stream=stream,
            request_id=request_id,
        )

        groq_response = outcome.result
        s4.end_ms = self._now_ms()
        s4.success = outcome.success
        s4.details = {
            "attempts": outcome.attempts,
            "retry_wait_ms": outcome.total_wait_ms,
            "gave_up": outcome.gave_up,
        }

        # Update record with LLM metrics
        if groq_response:
            record.first_token_ms = groq_response.first_token_ms
            record.tokens_in = groq_response.tokens_in
            record.tokens_out = groq_response.tokens_out
            record.http_status = groq_response.http_status
            record.model = groq_response.model

        record.retries = outcome.attempts - 1
        record.retry_wait_ms = outcome.total_wait_ms
        record.given_up = outcome.gave_up

        if not outcome.success or not groq_response or not groq_response.success:
            s4.success = False
            error_msg = outcome.final_error or (groq_response.error if groq_response else "Unknown error")
            record.status = "error"
            record.error_type = error_msg or ""
            record.total_ms = (time.time() - t_global_start) * 1000
            self.metrics.add(record)
            return PipelineResult(
                request_id=request_id,
                question=question,
                answer=f"An error occurred while generating the response: {error_msg}",
                success=False,
                steps=steps,
                retrieval_result=retrieval_result,
                groq_response=groq_response,
                analysis=analysis,
                record=record,
                error=error_msg,
            )

        answer = groq_response.content

        # ----------------------------------------------------------------
        # STEP 5: Validate answer and citations
        # ----------------------------------------------------------------
        s5 = step("answer_validation")
        validation = validate_answer(
            answer=answer,
            retrieved_chunks=retrieval_result.retrieved,
            question=question,
        )
        s5.end_ms = self._now_ms()
        s5.details = {
            "is_valid": validation.is_valid,
            "has_citations": validation.has_citations,
            "is_refusal": validation.is_refusal,
            "issues": validation.issues,
        }

        # ----------------------------------------------------------------
        # STEP 6: Finalize record and return
        # ----------------------------------------------------------------
        record.status = "ok"
        record.total_ms = (time.time() - t_global_start) * 1000
        record.working_ms = s4.duration_ms

        # Grade the answer
        if validation.is_refusal:
            record.answer_grade = "refused_correctly"
        elif validation.is_valid:
            record.answer_grade = "correct"
        else:
            record.answer_grade = "wrong"

        self.metrics.add(record)

        logger.info(
            "[%s] Done in %.0fms | Tokens: %d in / %d out | Grade: %s",
            request_id,
            record.total_ms,
            record.tokens_in,
            record.tokens_out,
            record.answer_grade,
        )

        return PipelineResult(
            request_id=request_id,
            question=question,
            answer=answer,
            success=True,
            steps=steps,
            retrieval_result=retrieval_result,
            groq_response=groq_response,
            validation=validation,
            analysis=analysis,
            record=record,
        )
