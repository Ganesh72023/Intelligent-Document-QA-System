"""
metrics.py - Performance metrics collection, calculation, and CSV export.
"""

import csv
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

RESULTS_CSV_COLUMNS = [
    "request_id", "scenario", "source", "start_time",
    "first_token_ms", "total_ms", "status", "retries",
    "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
    # Additional columns
    "queue_ms", "working_ms", "retry_wait_ms", "http_status",
    "error_type", "request_timeout_ms", "given_up", "model",
]


@dataclass
class RequestRecord:
    request_id: str
    scenario: str
    source: str                # "real" or "stand-in"
    start_time: float
    first_token_ms: float = 0.0
    total_ms: float = 0.0
    status: str = "ok"         # "ok", "rate_limited", "timeout", "error"
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    retrieval_setting: str = ""
    answer_grade: str = ""     # "correct", "wrong", "refused_correctly", "refused_wrongly"
    queue_ms: float = 0.0
    working_ms: float = 0.0
    retry_wait_ms: float = 0.0
    http_status: int = 200
    error_type: str = ""
    request_timeout_ms: float = 0.0
    given_up: bool = False
    model: str = ""


class MetricsCollector:
    """Thread-safe metrics collector."""

    def __init__(self):
        self.records: List[RequestRecord] = []
        self._start_time = time.time()

    def add(self, record: RequestRecord) -> None:
        self.records.append(record)

    def save_csv(self, path: Path) -> None:
        """Save all records to results.csv."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
            writer.writeheader()
            for rec in self.records:
                row = asdict(rec)
                # Ensure all required columns are present
                writer.writerow({k: row.get(k, "") for k in RESULTS_CSV_COLUMNS})
        logger.info("Saved %d records to %s", len(self.records), path)

    def compute_percentiles(self, values: List[float], percentiles=(50, 95, 99)) -> Dict[int, float]:
        """Compute percentiles from a list of values."""
        if not values:
            return {p: 0.0 for p in percentiles}
        arr = np.array(values)
        return {p: float(np.percentile(arr, p)) for p in percentiles}

    def summarize(self, scenario: Optional[str] = None) -> Dict[str, Any]:
        """Compute summary statistics for all or a specific scenario."""
        recs = self.records
        if scenario:
            recs = [r for r in recs if r.scenario == scenario]

        if not recs:
            return {}

        total = len(recs)
        ok = [r for r in recs if r.status == "ok"]
        rate_limited = [r for r in recs if r.status == "rate_limited"]
        timed_out = [r for r in recs if r.status == "timeout"]
        errors = [r for r in recs if r.status == "error"]
        given_up = [r for r in recs if r.given_up]

        # Latency (only successful requests)
        first_token_ms = [r.first_token_ms for r in ok if r.first_token_ms > 0]
        total_ms = [r.total_ms for r in ok if r.total_ms > 0]

        ft_pcts = self.compute_percentiles(first_token_ms)
        tt_pcts = self.compute_percentiles(total_ms)

        # Token stats
        tokens_out = [r.tokens_out for r in ok if r.tokens_out > 0]
        avg_tokens = float(np.mean(tokens_out)) if tokens_out else 0.0

        # Answer quality
        correct = sum(1 for r in recs if r.answer_grade == "correct")
        wrong = sum(1 for r in recs if r.answer_grade == "wrong")
        refused_correctly = sum(1 for r in recs if r.answer_grade == "refused_correctly")
        refused_wrongly = sum(1 for r in recs if r.answer_grade == "refused_wrongly")

        # Retries
        avg_retries = float(np.mean([r.retries for r in recs])) if recs else 0.0

        return {
            "scenario": scenario or "all",
            "total": total,
            "succeeded": len(ok),
            "succeeded_pct": 100.0 * len(ok) / total,
            "rate_limited": len(rate_limited),
            "rate_limited_pct": 100.0 * len(rate_limited) / total,
            "timed_out": len(timed_out),
            "timed_out_pct": 100.0 * len(timed_out) / total,
            "errors": len(errors),
            "given_up": len(given_up),
            "given_up_pct": 100.0 * len(given_up) / total,
            "avg_retries": avg_retries,
            "first_token_p50_ms": ft_pcts.get(50, 0.0),
            "first_token_p95_ms": ft_pcts.get(95, 0.0),
            "first_token_p99_ms": ft_pcts.get(99, 0.0),
            "total_time_p50_ms": tt_pcts.get(50, 0.0),
            "total_time_p95_ms": tt_pcts.get(95, 0.0),
            "total_time_p99_ms": tt_pcts.get(99, 0.0),
            "avg_tokens_out": avg_tokens,
            "correct": correct,
            "wrong": wrong,
            "refused_correctly": refused_correctly,
            "refused_wrongly": refused_wrongly,
        }

    def save_summary(self, path: Path) -> None:
        """Save JSON summary."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Overall + per scenario
        scenarios = list(set(r.scenario for r in self.records))
        summary = {
            "overall": self.summarize(),
            "by_scenario": {s: self.summarize(s) for s in scenarios},
        }
        path.write_text(json.dumps(summary, indent=2))
        logger.info("Saved summary to %s", path)
