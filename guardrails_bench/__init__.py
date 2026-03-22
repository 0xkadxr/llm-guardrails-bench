"""
LLM Guardrails Bench - A standardized benchmark suite for evaluating
LLM safety guardrails and defense mechanisms.
"""

__version__ = "0.1.0"
__author__ = "Abdelkader Benmeriem"

from guardrails_bench.runner import BenchmarkRunner
from guardrails_bench.config import BenchmarkConfig, ModelRegistry

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "ModelRegistry",
    "__version__",
]
