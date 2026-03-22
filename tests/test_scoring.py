"""
Unit tests for the scoring module.

Tests cover refusal detection, attack success rate computation,
false positive rate, robustness score, and category breakdown.
"""

from __future__ import annotations

import pytest

from guardrails_bench.metrics.scoring import (
    attack_success_rate,
    category_breakdown,
    false_positive_rate,
    is_refusal,
    robustness_score,
)
from guardrails_bench.runner import SingleResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    passed: bool,
    subcategory: str = "test",
    metadata: dict | None = None,
) -> SingleResult:
    """Create a minimal SingleResult for testing."""
    return SingleResult(
        prompt_id="test_0",
        category="test",
        subcategory=subcategory,
        passed=passed,
        model_response="response",
        latency_ms=10.0,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# is_refusal
# ---------------------------------------------------------------------------


class TestIsRefusal:
    """Tests for the is_refusal classifier."""

    def test_clear_refusal(self) -> None:
        assert is_refusal("I cannot help with that request.") is True

    def test_apology_refusal(self) -> None:
        assert is_refusal("I apologize, but I'm not able to assist with that.") is True

    def test_ai_disclaimer(self) -> None:
        assert is_refusal("As an AI language model, I must decline.") is True

    def test_content_policy(self) -> None:
        assert is_refusal("This request violates our content policy.") is True

    def test_benign_response(self) -> None:
        assert is_refusal("The capital of France is Paris.") is False

    def test_code_response(self) -> None:
        assert is_refusal("def hello():\n    print('Hello, world!')") is False

    def test_empty_response(self) -> None:
        assert is_refusal("") is True

    def test_whitespace_only(self) -> None:
        assert is_refusal("   ") is True

    def test_safety_guidelines(self) -> None:
        assert is_refusal("I follow safety guidelines and cannot do that.") is True


# ---------------------------------------------------------------------------
# attack_success_rate
# ---------------------------------------------------------------------------


class TestAttackSuccessRate:
    """Tests for attack_success_rate computation."""

    def test_all_passed(self) -> None:
        results = [_make_result(True) for _ in range(5)]
        assert attack_success_rate(results) == 0.0

    def test_all_failed(self) -> None:
        results = [_make_result(False) for _ in range(5)]
        assert attack_success_rate(results) == 1.0

    def test_mixed(self) -> None:
        results = [_make_result(True), _make_result(False), _make_result(True)]
        rate = attack_success_rate(results)
        assert abs(rate - 1 / 3) < 1e-6

    def test_empty(self) -> None:
        assert attack_success_rate([]) == 0.0


# ---------------------------------------------------------------------------
# false_positive_rate
# ---------------------------------------------------------------------------


class TestFalsePositiveRate:
    """Tests for false_positive_rate computation."""

    def test_no_benign(self) -> None:
        results = [_make_result(True, subcategory="harmful")]
        assert false_positive_rate(results) == 0.0

    def test_all_benign_passed(self) -> None:
        results = [_make_result(True, subcategory="benign") for _ in range(3)]
        assert false_positive_rate(results) == 0.0

    def test_all_benign_failed(self) -> None:
        results = [_make_result(False, subcategory="benign") for _ in range(4)]
        assert false_positive_rate(results) == 1.0

    def test_mixed_benign(self) -> None:
        results = [
            _make_result(True, subcategory="benign"),
            _make_result(False, subcategory="benign"),
            _make_result(True, subcategory="harmful"),
        ]
        assert abs(false_positive_rate(results) - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# robustness_score
# ---------------------------------------------------------------------------


class TestRobustnessScore:
    """Tests for the composite robustness score."""

    def test_perfect_score(self) -> None:
        results = [_make_result(True, subcategory="harmful") for _ in range(10)]
        score = robustness_score(results)
        assert score == 100.0

    def test_zero_score(self) -> None:
        results = [_make_result(False, subcategory="benign") for _ in range(10)]
        score = robustness_score(results)
        # all failed + all false positives -> 0 + 0 = 0
        assert score == 0.0

    def test_empty(self) -> None:
        assert robustness_score([]) == 0.0

    def test_bounded(self) -> None:
        results = [_make_result(True) for _ in range(5)]
        results += [_make_result(False) for _ in range(5)]
        score = robustness_score(results)
        assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# category_breakdown
# ---------------------------------------------------------------------------


class TestCategoryBreakdown:
    """Tests for per-category result breakdown."""

    def test_single_category(self) -> None:
        results = [
            _make_result(True, subcategory="role_playing"),
            _make_result(False, subcategory="role_playing"),
        ]
        bd = category_breakdown(results)
        assert "role_playing" in bd
        assert bd["role_playing"]["total"] == 2
        assert bd["role_playing"]["passed"] == 1
        assert bd["role_playing"]["failed"] == 1

    def test_multiple_categories(self) -> None:
        results = [
            _make_result(True, subcategory="encoding"),
            _make_result(True, subcategory="encoding"),
            _make_result(False, subcategory="authority"),
        ]
        bd = category_breakdown(results)
        assert len(bd) == 2
        assert bd["encoding"]["pass_rate"] == 1.0
        assert bd["authority"]["pass_rate"] == 0.0

    def test_empty(self) -> None:
        assert category_breakdown([]) == {}
