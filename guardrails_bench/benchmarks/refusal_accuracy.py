"""
Refusal accuracy benchmark.

Measures false positive and false negative refusal rates to determine
how precisely a model's safety guardrails distinguish between genuinely
harmful requests and benign ones.
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


class RefusalAccuracyBenchmark:
    """
    Benchmark for measuring refusal precision and recall.

    Uses two balanced datasets:
        - **Harmful requests** that *should* be refused.
        - **Benign requests** that *should not* be refused.

    Metrics derived:
        - True positive rate (correctly refused harmful requests).
        - False positive rate (incorrectly refused benign requests).
        - False negative rate (incorrectly allowed harmful requests).
        - Balanced accuracy across both classes.
    """

    CATEGORY = BenchmarkCategory.REFUSAL_ACCURACY

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.harmful = load_dataset("harmful_requests.json", config)
        self.benign = load_dataset("benign_requests.json", config)

    def evaluate(self, model: "BaseModel") -> List["SingleResult"]:
        """Run refusal accuracy tests and return per-prompt results."""
        from guardrails_bench.runner import SingleResult

        results: List[SingleResult] = []

        # --- Harmful: model should refuse (True Positive) ---
        for idx, entry in enumerate(self.harmful):
            prompt_text: str = entry["prompt"]
            prompt_id = entry.get("id", f"refusal_harmful_{idx}")

            start = time.perf_counter()
            try:
                response = model.generate(prompt_text)
                latency = (time.perf_counter() - start) * 1000
                refused = is_refusal(response)

                results.append(
                    SingleResult(
                        prompt_id=prompt_id,
                        category=self.CATEGORY.value,
                        subcategory="harmful",
                        passed=refused,  # pass = correctly refused
                        model_response=response,
                        latency_ms=latency,
                        metadata={
                            "expected_refusal": True,
                            "actual_refusal": refused,
                            "classification": "TP" if refused else "FN",
                        },
                    )
                )
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                results.append(
                    SingleResult(
                        prompt_id=prompt_id,
                        category=self.CATEGORY.value,
                        subcategory="harmful",
                        passed=False,
                        model_response="",
                        latency_ms=latency,
                        metadata={"error": str(exc)},
                    )
                )

        # --- Benign: model should NOT refuse (True Negative) ---
        for idx, entry in enumerate(self.benign):
            prompt_text = entry["prompt"]
            prompt_id = entry.get("id", f"refusal_benign_{idx}")

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
                            "expected_refusal": False,
                            "actual_refusal": refused,
                            "classification": "FP" if refused else "TN",
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
                        metadata={"error": str(exc)},
                    )
                )

        return results
