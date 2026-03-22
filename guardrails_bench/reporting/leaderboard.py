"""
Leaderboard generation.

Produces a Markdown leaderboard table ranking models across all
benchmark categories, suitable for embedding in README files or
publishing as standalone reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from guardrails_bench.metrics.analysis import compare_models

if TYPE_CHECKING:
    from guardrails_bench.runner import BenchmarkResults


def generate_leaderboard(
    results: "BenchmarkResults",
    output_path: Optional[Path] = None,
    title: str = "LLM Safety Guardrails Leaderboard",
) -> str:
    """
    Generate a Markdown leaderboard table.

    Args:
        results: Complete benchmark results.
        output_path: Optional path to write the leaderboard file.
        title: Heading for the leaderboard.

    Returns:
        Rendered Markdown string.
    """
    comparison = compare_models(results)

    # Collect all unique categories across models
    all_categories: list[str] = []
    for data in comparison.values():
        for cat in data["categories"]:
            if cat not in all_categories:
                all_categories.append(cat)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        "> Automated evaluation of LLM safety guardrails. "
        "Higher scores indicate stronger safety properties."
    )
    lines.append("")

    # Build header
    cat_headers = " | ".join(
        _format_category_name(c) for c in all_categories
    )
    header = f"| Rank | Model | {cat_headers} | **Overall** |"
    separator_parts = ["| :--: | :---- |"]
    for _ in all_categories:
        separator_parts.append(" :---: |")
    separator_parts.append(" :---: |")
    separator = "".join(separator_parts)

    lines.append(header)
    lines.append(separator)

    # Sort by overall score descending
    ranked = sorted(
        comparison.items(), key=lambda x: x[1]["overall"], reverse=True
    )

    for model_name, data in ranked:
        rank = data["rank"]
        rank_display = _rank_badge(rank)
        cat_values = " | ".join(
            f"{data['categories'].get(c, 0):.0%}" for c in all_categories
        )
        overall = data["overall"]
        lines.append(
            f"| {rank_display} | {model_name} | {cat_values} | **{overall:.1f}** |"
        )

    lines.append("")
    lines.append(
        "*Scores represent pass rates (higher = safer). "
        "Overall is a weighted composite score (0-100).*"
    )
    lines.append("")

    leaderboard = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(leaderboard, encoding="utf-8")

    return leaderboard


def _format_category_name(category: str) -> str:
    """Format a category slug into a human-readable column header."""
    return category.replace("_", " ").title()


def _rank_badge(rank: int) -> str:
    """Return a display string for the rank."""
    badges = {1: ":1st_place_medal:", 2: ":2nd_place_medal:", 3: ":3rd_place_medal:"}
    return badges.get(rank, str(rank))
