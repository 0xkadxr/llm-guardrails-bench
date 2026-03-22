#!/usr/bin/env python3
"""
Example: Compare two models on a specific benchmark category.

Usage:
    python examples/compare_models.py

This script demonstrates how to run a single benchmark category
against multiple models and display a side-by-side comparison.
"""

from pathlib import Path

from rich.console import Console
from rich.table import Table

from guardrails_bench import BenchmarkRunner
from guardrails_bench.config import (
    BenchmarkCategory,
    BenchmarkConfig,
    ModelConfig,
    ModelProvider,
)
from guardrails_bench.metrics.scoring import (
    attack_success_rate,
    category_breakdown,
    robustness_score,
)

console = Console()


def main() -> None:
    # ── Models to compare ──────────────────────────────────────────────
    models = [
        ModelConfig(
            name="GPT-4o",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o",
        ),
        ModelConfig(
            name="Claude 3.5 Sonnet",
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-3-5-sonnet-20241022",
        ),
    ]

    config = BenchmarkConfig(
        categories=[BenchmarkCategory.JAILBREAK],
        max_concurrent=2,
    )

    runner = BenchmarkRunner(models=models, config=config)

    # ── Run only the jailbreak benchmark ───────────────────────────────
    console.print("\n[bold]Running jailbreak benchmark comparison...[/bold]\n")

    all_results = {}
    for model_config in models:
        result = runner.run_single(model_config, BenchmarkCategory.JAILBREAK)
        all_results[model_config.name] = result

    # ── Display comparison ─────────────────────────────────────────────
    table = Table(title="Jailbreak Resistance Comparison", show_lines=True)
    table.add_column("Metric", style="bold")
    for model_config in models:
        table.add_column(model_config.name, justify="center")

    # Overall pass rate
    table.add_row(
        "Overall Pass Rate",
        *[f"{r.pass_rate:.1%}" for r in all_results.values()],
    )

    # Attack success rate
    table.add_row(
        "Attack Success Rate",
        *[
            f"{attack_success_rate(r.results):.1%}"
            for r in all_results.values()
        ],
    )

    # Robustness score
    table.add_row(
        "Robustness Score",
        *[
            f"{robustness_score(r.results):.1f}/100"
            for r in all_results.values()
        ],
    )

    console.print(table)
    console.print()

    # ── Per-category breakdown ─────────────────────────────────────────
    for model_name, result in all_results.items():
        console.print(f"[bold cyan]{model_name}[/bold cyan] breakdown:")
        breakdown = category_breakdown(result.results)
        for subcat, stats in breakdown.items():
            bar_len = int(stats["pass_rate"] * 20)
            bar = "=" * bar_len + "-" * (20 - bar_len)
            console.print(
                f"  {subcat:<15} [{bar}] {stats['pass_rate']:.0%} "
                f"({stats['passed']}/{stats['total']})"
            )
        console.print()


if __name__ == "__main__":
    main()
