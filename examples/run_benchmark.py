#!/usr/bin/env python3
"""
Example: Run the full benchmark suite against one or more models.

Usage:
    python examples/run_benchmark.py

Ensure you have set the relevant API keys in your environment:
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

from pathlib import Path

from guardrails_bench import BenchmarkRunner
from guardrails_bench.config import (
    BenchmarkCategory,
    BenchmarkConfig,
    ModelConfig,
    ModelProvider,
)
from guardrails_bench.reporting import generate_leaderboard, generate_markdown_report


def main() -> None:
    # ── Define models to evaluate ──────────────────────────────────────
    models = [
        ModelConfig(
            name="GPT-4o",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o",
            temperature=0.0,
            max_tokens=1024,
        ),
        ModelConfig(
            name="Claude 3.5 Sonnet",
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-3-5-sonnet-20241022",
            temperature=0.0,
            max_tokens=1024,
        ),
    ]

    # ── Configure the benchmark ────────────────────────────────────────
    config = BenchmarkConfig(
        categories=[
            BenchmarkCategory.JAILBREAK,
            BenchmarkCategory.CONTENT_FILTER,
            BenchmarkCategory.INSTRUCTION_HIERARCHY,
            BenchmarkCategory.DATA_LEAKAGE,
            BenchmarkCategory.REFUSAL_ACCURACY,
        ],
        max_concurrent=4,
        timeout=60,
        output_dir=Path("results"),
        verbose=True,
    )

    # ── Run ────────────────────────────────────────────────────────────
    runner = BenchmarkRunner(models=models, config=config)
    results = runner.run(parallel=True)

    # ── Save outputs ───────────────────────────────────────────────────
    results.save(Path("results/benchmark_results.json"))

    report = generate_markdown_report(
        results, output_path=Path("results/report.md")
    )
    print("Markdown report written to results/report.md")

    leaderboard = generate_leaderboard(
        results, output_path=Path("results/leaderboard.md")
    )
    print("Leaderboard written to results/leaderboard.md")


if __name__ == "__main__":
    main()
