"""
Statistical analysis utilities for benchmark results.

Provides confidence intervals, model comparison helpers, and
summary statistics for interpreting benchmark outcomes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from guardrails_bench.runner import BenchmarkResult, BenchmarkResults


def compute_confidence_interval(
    pass_count: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute a Wilson score confidence interval for a binomial proportion.

    The Wilson interval is preferred over the normal approximation for
    small sample sizes and proportions near 0 or 1.

    Args:
        pass_count: Number of successes (passed tests).
        total: Total number of trials.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound) as floats in [0, 1].
    """
    if total == 0:
        return (0.0, 0.0)

    z = _z_score(confidence)
    p_hat = pass_count / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * total)) / total
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


def _z_score(confidence: float) -> float:
    """Return the z-score for a given confidence level."""
    # Common confidence levels
    z_map = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }
    return z_map.get(confidence, 1.960)


def compare_models(
    results: "BenchmarkResults",
) -> Dict[str, Dict[str, Any]]:
    """
    Compare all models across benchmark categories.

    Returns a dictionary per model containing per-category pass rates,
    overall score, and rankings.

    Args:
        results: Complete benchmark results.

    Returns:
        Nested dictionary with model comparison data::

            {
                "GPT-4o": {
                    "categories": {"jailbreak": 0.92, ...},
                    "overall": 87.5,
                    "rank": 1,
                },
                ...
            }
    """
    scores = results.overall_scores()

    # Sort by overall score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    comparison: Dict[str, Dict[str, Any]] = {}
    for rank, (model_name, overall) in enumerate(ranked, start=1):
        model_results = results.get_model_results(model_name)
        categories = {
            r.category.value: round(r.pass_rate, 4) for r in model_results
        }
        comparison[model_name] = {
            "categories": categories,
            "overall": overall,
            "rank": rank,
        }

    return comparison


def statistical_summary(
    results: List["BenchmarkResult"],
) -> Dict[str, Any]:
    """
    Compute summary statistics across multiple benchmark results.

    Args:
        results: List of benchmark results (typically for one model).

    Returns:
        Dictionary containing mean, std, min, max pass rates and
        total test counts.
    """
    if not results:
        return {
            "mean_pass_rate": 0.0,
            "std_pass_rate": 0.0,
            "min_pass_rate": 0.0,
            "max_pass_rate": 0.0,
            "total_tests": 0,
            "total_passed": 0,
        }

    pass_rates = np.array([r.pass_rate for r in results])
    total_tests = sum(r.total_tests for r in results)
    total_passed = sum(r.passed for r in results)

    return {
        "mean_pass_rate": round(float(np.mean(pass_rates)), 4),
        "std_pass_rate": round(float(np.std(pass_rates)), 4),
        "min_pass_rate": round(float(np.min(pass_rates)), 4),
        "max_pass_rate": round(float(np.max(pass_rates)), 4),
        "total_tests": total_tests,
        "total_passed": total_passed,
    }
