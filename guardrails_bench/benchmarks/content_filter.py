"""
Content filtering effectiveness benchmark.

Evaluates how well a model blocks harmful content generation across
multiple harm categories while avoiding false positives on benign input.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, TYPE_CHECKING

from guardrails_bench.config import BenchmarkCategory, BenchmarkConfig
from guardrails_bench.datasets.loader import load_dataset
from guardrails_bench.metrics.scoring import is_refusal

if TYPE_CHECKING:
    from guardrails_bench.models.base import BaseModel
    from guardrails_bench.runner import SingleResult


class ContentFilterBenchmark:
    """
    Benchmark that measures content filtering effectiveness.

    Sends harmful content requests (violence, illegal activity, PII
    generation, malware code, fraud instructions) and verifies that
    the model refuses. Also sends benign queries to measure the false
    positive rate.

    A test on a harmful request *passes* when the model refuses.
    A test on a benign request *passes* when the model does **not** refuse.
    """

    CATEGORY = BenchmarkCategory.CONTENT_FILTER

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.harmful = load_dataset("harmful_requests.json", config)
        self.benign = load_dataset("benign_requests.json", config)

    def evaluate(self, model: "BaseModel") -> List["SingleResult"]:
        """Run content filter tests and return per-prompt results."""
        from guardrails_bench.runner import SingleResult

        results: List[SingleResult] = []

        # --- Harmful requests (model should refuse) ---
        for idx, entry in enumerate(self.harmful):
            prompt_text: str = entry["prompt"]
            subcategory: str = entry.get("category", "harmful")
            prompt_id = entry.get("id", f"harmful_{idx}")

            start = time.perf_counter()
            try:
                response = model.generate(prompt_text)
                latency = (time.perf_counter() - start) * 1000
                refused = is_refusal(response)

                results.append(
                    SingleResult(
                        prompt_id=prompt_id,
                        category=self.CATEGORY.value,
                        subcategory=subcategory,
                        passed=refused,
                        model_response=response,
                        latency_ms=latency,
                        metadata={"type": "harmful", "refused": refused},
                    )
                )
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                results.append(
                    SingleResult(
                        prompt_id=prompt_id,
                        category=self.CATEGORY.value,
                        subcategory=subcategory,
                        passed=False,
                        model_response="",
                        latency_ms=latency,
                        metadata={"type": "harmful", "error": str(exc)},
                    )
                )

        # --- Benign requests (model should NOT refuse) ---
        for idx, entry in enumerate(self.benign):
            prompt_text = entry["prompt"]
            subcategory = entry.get("category", "benign")
            prompt_id = entry.get("id", f"benign_{idx}")

            start = time.perf_counter()
            try:
                response = model.generate(prompt_text)
                latency = (time.perf_counter() - start) * 1000
                refused = is_refusal(response)

                results.append(
                    SingleResult(
                        prompt_id=prompt_id,
                        category=self.CATEGORY.value,
                        subcategory="benign",
                        passed=not refused,  # pass = correctly allowed
                        model_response=response,
                        latency_ms=latency,
                        metadata={
                            "type": "benign",
                            "false_positive": refused,
                        },
                    )
                )
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                results.append(
                    SingleResult(
                        prompt_id=prompt_id,
                        category=self.CATEGORY.value,
                        subcategory="benign",
                        passed=False,
                        model_response="",
                        latency_ms=latency,
                        metadata={"type": "benign", "error": str(exc)},
                    )
                )

        return results
