"""Tests for GSM8K answer extraction and scoring.

Answer parsing is where a reasoning benchmark quietly goes wrong. A parser that
is too strict scores a correct model as broken the moment quantization degrades
its output formatting, which confounds arithmetic with instruction-following --
exactly the confound this ladder exists to avoid.
"""

from __future__ import annotations

import pytest

from scripts.run_sector_ladder import (
    extract_gold,
    extract_predicted,
    is_correct,
    parse_number,
)


# ---------------------------------------------------------------------------
# gold answers
# ---------------------------------------------------------------------------


def test_extracts_gold_from_gsm8k_format() -> None:
    answer = "She has 3 apples and buys 5 more.\n#### 8"
    assert extract_gold(answer) == pytest.approx(8.0)


def test_extracts_gold_with_thousands_separators() -> None:
    assert extract_gold("Total revenue.\n#### 1,250") == pytest.approx(1250.0)


def test_extracts_negative_and_decimal_gold() -> None:
    assert extract_gold("#### -42") == pytest.approx(-42.0)
    assert extract_gold("#### 3.5") == pytest.approx(3.5)


def test_missing_gold_marker_returns_none() -> None:
    assert extract_gold("The answer is 8.") is None


# ---------------------------------------------------------------------------
# predicted answers
# ---------------------------------------------------------------------------


def test_takes_the_last_number_after_chain_of_thought() -> None:
    """The standard GSM8K convention: intermediate arithmetic, then the result."""
    completion = "First 3 x 4 = 12, then 12 - 5 = 7. The final answer is 7."
    assert extract_predicted(completion) == pytest.approx(7.0)


def test_tolerates_a_model_that_stops_following_the_format() -> None:
    """A degraded model may drop 'the final answer is' but still compute.

    Requiring a fixed answer format would score that as an arithmetic failure,
    confounding two different capabilities.
    """
    assert extract_predicted("3 x 4 = 12\n12 - 5 = 7") == pytest.approx(7.0)


def test_handles_thousands_separators_and_currency_text() -> None:
    assert extract_predicted("He earns $1,250 in total.") == pytest.approx(1250.0)


def test_returns_none_when_there_is_no_number() -> None:
    assert extract_predicted("I cannot solve this problem.") is None


def test_returns_none_for_empty_completion() -> None:
    assert extract_predicted("") is None


def test_gibberish_without_digits_yields_no_answer() -> None:
    assert extract_predicted("brand diidi _Nothing WL ToolsICS SL") is None


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def test_correct_answer_scores_correct() -> None:
    assert is_correct("The answer is 42.", 42.0)


def test_wrong_answer_scores_incorrect() -> None:
    assert not is_correct("The answer is 41.", 42.0)


def test_missing_answer_scores_incorrect_rather_than_raising() -> None:
    """A model that produces no number is wrong, not an error to be skipped."""
    assert not is_correct("I cannot solve this.", 42.0)


def test_float_tolerance_is_tight_enough_to_reject_near_misses() -> None:
    assert is_correct("42.00001", 42.0)
    assert not is_correct("42.1", 42.0)


def test_integer_and_decimal_forms_of_the_same_value_agree() -> None:
    assert is_correct("The answer is 8.0", 8.0)
    assert is_correct("The answer is 8", 8.0)


# ---------------------------------------------------------------------------
# the number parser itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("42", 42.0), ("-7", -7.0), ("+3", 3.0), ("1,000", 1000.0), ("0.5", 0.5)],
)
def test_parse_number_accepts_common_forms(raw: str, expected: float) -> None:
    assert parse_number(raw) == pytest.approx(expected)


def test_parse_number_rejects_non_numeric() -> None:
    assert parse_number("abc") is None
    assert parse_number("") is None
