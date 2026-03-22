"""
Main benchmark runner that orchestrates test execution across models
and benchmark categories.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from guardrails_bench.config import (
    BenchmarkCategory,
    BenchmarkConfig,
    ModelConfig,
    ModelRegistry,
)

console = Console()


@dataclass
class SingleResult:
    """Result of a single test prompt against a model."""

    prompt_id: str
    category: str
    subcategory: str
    passed: bool
    model_response: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated result for one benchmark category on one model."""

    model_name: str
    category: BenchmarkCategory
    total_tests: int
    passed: int
    failed: int
    error_count: int
    results: List[SingleResult]
    duration_seconds: float

    @property
    def pass_rate(self) -> float:
        """Fraction of tests that passed (0.0 to 1.0)."""
        return self.passed / self.total_tests if self.total_tests > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "model_name": self.model_name,
            "category": self.category.value,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "error_count": self.error_count,
            "pass_rate": round(self.pass_rate, 4),
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class BenchmarkResults:
    """Complete benchmark results across all models and categories."""

    results: Dict[str, List[BenchmarkResult]] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    config: Optional[Dict[str, Any]] = None

    def get_model_results(self, model_name: str) -> List[BenchmarkResult]:
        """Get all benchmark results for a specific model."""
        return self.results.get(model_name, [])

    def get_category_results(
        self, category: BenchmarkCategory
    ) -> Dict[str, BenchmarkResult]:
        """Get results across all models for a specific category."""
        out: Dict[str, BenchmarkResult] = {}
        for model_name, model_results in self.results.items():
            for r in model_results:
                if r.category == category:
                    out[model_name] = r
        return out

    def overall_scores(self) -> Dict[str, float]:
        """Compute a weighted overall safety score per model (0-100)."""
        scores: Dict[str, float] = {}
        for model_name, model_results in self.results.items():
            if not model_results:
                scores[model_name] = 0.0
                continue
            total_pass = sum(r.pass_rate for r in model_results)
            scores[model_name] = round(
                (total_pass / len(model_results)) * 100, 1
            )
        return scores

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full result set."""
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "models": {
                name: [r.to_dict() for r in results]
                for name, results in self.results.items()
            },
            "overall_scores": self.overall_scores(),
        }

    def save(self, path: Path) -> None:
        """Persist results to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        console.print(f"[green]Results saved to {path}[/green]")


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

_BENCHMARK_MAP: Dict[BenchmarkCategory, Type] = {}


def _ensure_benchmarks_loaded() -> None:
    """Lazily import and register benchmark classes."""
    if _BENCHMARK_MAP:
        return

    from guardrails_bench.benchmarks.content_filter import ContentFilterBenchmark
    from guardrails_bench.benchmarks.data_leakage import DataLeakageBenchmark
    from guardrails_bench.benchmarks.instruction_hierarchy import (
        InstructionHierarchyBenchmark,
    )
    from guardrails_bench.benchmarks.jailbreak import JailbreakBenchmark
    from guardrails_bench.benchmarks.refusal_accuracy import RefusalAccuracyBenchmark

    _BENCHMARK_MAP[BenchmarkCategory.JAILBREAK] = JailbreakBenchmark
    _BENCHMARK_MAP[BenchmarkCategory.CONTENT_FILTER] = ContentFilterBenchmark
    _BENCHMARK_MAP[BenchmarkCategory.INSTRUCTION_HIERARCHY] = (
        InstructionHierarchyBenchmark
    )
    _BENCHMARK_MAP[BenchmarkCategory.DATA_LEAKAGE] = DataLeakageBenchmark
    _BENCHMARK_MAP[BenchmarkCategory.REFUSAL_ACCURACY] = RefusalAccuracyBenchmark


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """
    Orchestrates benchmark execution across multiple models and categories.

    Args:
        models: List of model configurations to evaluate.
        config: Benchmark configuration (uses defaults if ``None``).

    Example::

        from guardrails_bench import BenchmarkRunner
        from guardrails_bench.config import ModelConfig, ModelProvider

        models = [
            ModelConfig(name="GPT-4o", provider=ModelProvider.OPENAI, model_id="gpt-4o"),
        ]
        runner = BenchmarkRunner(models=models)
        results = runner.run()
        results.save(Path("results/run.json"))
    """

    def __init__(
        self,
        models: List[ModelConfig],
        config: Optional[BenchmarkConfig] = None,
    ) -> None:
        self.models = models
        self.config = config or BenchmarkConfig()
        _ensure_benchmarks_loaded()

    # ------------------------------------------------------------------

    def _instantiate_model(self, model_config: ModelConfig):
        """Create a model instance from its configuration."""
        model_cls = ModelRegistry.get(model_config.provider.value)
        return model_cls(model_config)

    def _instantiate_benchmark(self, category: BenchmarkCategory):
        """Create a benchmark instance for the given category."""
        bench_cls = _BENCHMARK_MAP.get(category)
        if bench_cls is None:
            raise ValueError(f"No benchmark registered for category {category}")
        return bench_cls(self.config)

    # ------------------------------------------------------------------

    def run_single(
        self,
        model_config: ModelConfig,
        category: BenchmarkCategory,
    ) -> BenchmarkResult:
        """
        Run a single benchmark category against a single model.

        Returns:
            A ``BenchmarkResult`` containing per-prompt outcomes.
        """
        model = self._instantiate_model(model_config)
        benchmark = self._instantiate_benchmark(category)

        start = time.perf_counter()
        results: List[SingleResult] = benchmark.evaluate(model)
        duration = time.perf_counter() - start

        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        errors = sum(
            1 for r in results if r.metadata.get("error") is not None
        )

        return BenchmarkResult(
            model_name=model_config.name,
            category=category,
            total_tests=len(results),
            passed=passed,
            failed=failed,
            error_count=errors,
            results=results,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------

    def run(self, parallel: bool = True) -> BenchmarkResults:
        """
        Run all configured benchmarks across all models.

        Args:
            parallel: Whether to run model/benchmark pairs concurrently.

        Returns:
            Aggregated ``BenchmarkResults``.
        """
        categories = self.config.categories
        total_tasks = len(self.models) * len(categories)

        all_results = BenchmarkResults(
            config=self.config.model_dump(mode="json"),
        )

        console.print(
            f"\n[bold cyan]LLM Guardrails Bench[/bold cyan] v0.1.0\n"
            f"  Models     : {len(self.models)}\n"
            f"  Categories : {len(categories)}\n"
            f"  Total runs : {total_tasks}\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running benchmarks...", total=total_tasks)

            if parallel and total_tasks > 1:
                self._run_parallel(
                    categories, all_results, progress, task
                )
            else:
                self._run_sequential(
                    categories, all_results, progress, task
                )

        self._print_summary(all_results)
        return all_results

    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        categories: List[BenchmarkCategory],
        all_results: BenchmarkResults,
        progress: Progress,
        task,
    ) -> None:
        for model_config in self.models:
            for category in categories:
                progress.update(
                    task,
                    description=f"{model_config.name} / {category.value}",
                )
                result = self.run_single(model_config, category)
                all_results.results.setdefault(model_config.name, []).append(
                    result
                )
                progress.advance(task)

    def _run_parallel(
        self,
        categories: List[BenchmarkCategory],
        all_results: BenchmarkResults,
        progress: Progress,
        task,
    ) -> None:
        with ThreadPoolExecutor(
            max_workers=self.config.max_concurrent
        ) as executor:
            future_map = {}
            for model_config in self.models:
                for category in categories:
                    future = executor.submit(
                        self.run_single, model_config, category
                    )
                    future_map[future] = (model_config.name, category)

            for future in as_completed(future_map):
                model_name, category = future_map[future]
                progress.update(
                    task, description=f"{model_name} / {category.value}"
                )
                try:
                    result = future.result()
                    all_results.results.setdefault(model_name, []).append(
                        result
                    )
                except Exception as exc:
                    console.print(
                        f"[red]Error: {model_name}/{category.value}: {exc}[/red]"
                    )
                progress.advance(task)

    # ------------------------------------------------------------------

    def _print_summary(self, results: BenchmarkResults) -> None:
        """Print a rich summary table to the console."""
        scores = results.overall_scores()

        table = Table(title="Benchmark Summary", show_lines=True)
        table.add_column("Model", style="bold")
        for cat in self.config.categories:
            table.add_column(cat.value, justify="center")
        table.add_column("Overall", justify="center", style="bold green")

        for model_config in self.models:
            name = model_config.name
            row = [name]
            for cat in self.config.categories:
                cat_results = results.get_category_results(cat)
                if name in cat_results:
                    rate = cat_results[name].pass_rate
                    row.append(f"{rate:.0%}")
                else:
                    row.append("-")
            row.append(f"{scores.get(name, 0):.1f}")
            table.add_row(*row)

        console.print()
        console.print(table)
        console.print()
