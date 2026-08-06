"""Tests for GSM8K answer extraction and scoring.

Answer parsing is where a reasoning benchmark quietly goes wrong. A parser that
is too strict scores a correct model as broken the moment quantization degrades
its output formatting, which confounds arithmetic with instruction-following --
exactly the confound this ladder exists to avoid.
"""

from __future__ import annotations

import json

import pytest

from scripts.run_sector_ladder import (
    extract_gold,
    extract_predicted,
    is_correct,
    parse_number,
    reusable_prefix,
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


# ---------------------------------------------------------------------------
# reusing a longer run's completions
# ---------------------------------------------------------------------------
#
# `load_gsm8k(n)` walks the test split in order and stops at n, so a short run's
# questions are a prefix of a long one's and the completions are already on
# disk. Reuse is only sound when the batch boundaries line up: generation pads
# each batch to its own longest member, so an item's output depends on which
# items share its batch, and batches start at fixed offsets of `batch_size`.
# Verified against the GPU: truncating a 32-question cache to 16 at batch 4
# reproduces a freshly generated 16-question run exactly, character for
# character, on all three schemes.


def _cache_with(tmp_path, scheme: str, n: int, tokens: int, texts: list[str]):
    path = tmp_path / f"gsm8k_{scheme}_n{n}_t{tokens}.json"
    path.write_text(json.dumps(texts), encoding="utf-8")
    return path


def test_reuses_a_longer_run_when_batches_align(tmp_path) -> None:
    _cache_with(tmp_path, "FP16", 32, 192, [f"answer {i}" for i in range(32)])
    got = reusable_prefix(tmp_path, "FP16", 16, 192, batch_size=4)
    assert got == [f"answer {i}" for i in range(16)]


def test_refuses_when_the_final_batch_would_be_partial(tmp_path) -> None:
    """18 is not a multiple of 4, so items 16 and 17 would sit in a batch of two
    here and a batch of four in the source. Approximately the same completions
    is not something a paired comparison can absorb."""
    _cache_with(tmp_path, "FP16", 32, 192, [f"answer {i}" for i in range(32)])
    assert reusable_prefix(tmp_path, "FP16", 18, 192, batch_size=4) is None


def test_refuses_a_source_that_is_not_longer(tmp_path) -> None:
    _cache_with(tmp_path, "FP16", 16, 192, [f"answer {i}" for i in range(16)])
    assert reusable_prefix(tmp_path, "FP16", 16, 192, batch_size=4) is None
    assert reusable_prefix(tmp_path, "FP16", 32, 192, batch_size=4) is None


def test_does_not_cross_token_budgets(tmp_path) -> None:
    """A 48-token completion is not the first 48 tokens of a 256-token one; the
    model stops where it stops. Different budgets are different measurements."""
    _cache_with(tmp_path, "FP16", 32, 48, [f"answer {i}" for i in range(32)])
    assert reusable_prefix(tmp_path, "FP16", 16, 256, batch_size=4) is None


def test_does_not_cross_schemes(tmp_path) -> None:
    _cache_with(tmp_path, "RTN_4B", 32, 192, [f"answer {i}" for i in range(32)])
    assert reusable_prefix(tmp_path, "FP16", 16, 192, batch_size=4) is None


def test_returns_none_when_nothing_is_cached(tmp_path) -> None:
    assert reusable_prefix(tmp_path, "FP16", 16, 192, batch_size=4) is None


def test_ignores_a_truncated_cache_file(tmp_path) -> None:
    """The filename claims 32 completions; the file holds 8. Trusting the name
    would pair the first eight answers against sixteen questions."""
    _cache_with(tmp_path, "FP16", 32, 192, [f"answer {i}" for i in range(8)])
    assert reusable_prefix(tmp_path, "FP16", 16, 192, batch_size=4) is None
