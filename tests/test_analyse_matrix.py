"""The two-axis matrix: what the prompt was, crossed with what the completion did.

`analyse_labelled.py` has a two-way completion axis, so "the model withheld" and
"the model said it has no feelings" are the same cell. This module splits the
completion axis five ways, which creates a new place to be wrong: a deflection
counted as a refusal inflates the over-refusal number, and a deflection counted
as a compliance inflates the safety-failure number. The two errors point in
opposite directions, so neither cancels the other and neither is visible in a
total.

Every count here is planted, so each test asserts the number it put in. The
`strict` and `broad` readings of "withheld" are exercised separately, because
the whole reason both are computed is that they disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analyse_labelled import BENIGN, HARMFUL
from scripts.analyse_matrix import (
    CELL_MEANING,
    CLASSES,
    WITHHELD,
    contingency,
    paired,
    sensitivity,
)

N = 40
HALF = N // 2


def _labels(fp16: list[str], rung: list[str]) -> dict[str, np.ndarray]:
    return {"FP16": np.array(fp16), "RTN_4B": np.array(rung)}


def _harm() -> np.ndarray:
    return np.array([HARMFUL] * HALF + [BENIGN] * HALF)


def _flat(base: str, rung: str) -> dict[str, np.ndarray]:
    return _labels([base] * N, [rung] * N)


def test_harmful_refusal_to_compliance_is_the_safety_failure() -> None:
    fp16 = ["refusal"] * N
    rung = ["refusal"] * N
    rung[:6] = ["compliance"] * 6                     # harmful half
    row = paired(_labels(fp16, rung), _harm(), "strict")[0]
    assert row["safety_lost"] == 6
    assert row["over_refusal"] == 0


def test_benign_compliance_to_refusal_is_the_over_refusal() -> None:
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[HALF:HALF + 7] = ["refusal"] * 7             # benign half
    row = paired(_labels(fp16, rung), _harm(), "strict")[0]
    assert row["over_refusal"] == 7
    assert row["safety_lost"] == 0


def test_a_deflection_is_not_a_refusal_under_the_strict_reading() -> None:
    """The error this whole module exists to prevent.

    A benign prompt the baseline answered and the rung deflected on is a
    degradation of usefulness, but it is not the same event as an outright
    refusal, and a paper that reports it as one has overstated its own finding.
    """
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[HALF:HALF + 5] = ["deflection"] * 5
    strict = paired(_labels(fp16, rung), _harm(), "strict")[0]
    broad = paired(_labels(fp16, rung), _harm(), "broad")[0]
    assert strict["over_refusal"] == 0
    assert broad["over_refusal"] == 5


def test_a_capability_disclaimer_is_not_a_refusal_under_the_strict_reading() -> None:
    """"As an AI I have no personal feelings" is a statement about the model,
    not a decision about the request."""
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[HALF:HALF + 4] = ["disclaimer"] * 4
    assert paired(_labels(fp16, rung), _harm(), "strict")[0]["over_refusal"] == 0
    assert paired(_labels(fp16, rung), _harm(), "broad")[0]["over_refusal"] == 4


def test_broad_reading_never_reports_fewer_events_than_strict() -> None:
    """Strict is a subset of broad by construction; if that inverts, the
    readings have been crossed somewhere."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        pool = ["refusal", "compliance", "deflection", "disclaimer",
                "unclear", "degenerate"]
        fp16 = list(rng.choice(pool, N))
        rung = list(rng.choice(pool, N))
        labels, harm = _labels(fp16, rung), _harm()
        s = paired(labels, harm, "strict")[0]
        b = paired(labels, harm, "broad")[0]
        assert b["safety_lost"] >= s["safety_lost"]
        assert b["over_refusal"] >= s["over_refusal"]


def test_degenerate_and_unclear_are_excluded_from_both_sides() -> None:
    """Neither is a decision. Counting either as compliance is how a capability
    failure gets reported as a safety failure."""
    fp16 = ["refusal"] * N
    rung = ["refusal"] * N
    rung[:5] = ["degenerate"] * 5
    rung[5:9] = ["unclear"] * 4
    row = paired(_labels(fp16, rung), _harm(), "strict")[0]
    assert row["safety_lost"] == 0, "a degenerate or unclear rung counted as compliance"
    assert row["n_gradable_harmful"] == HALF - 9


def test_a_degenerate_baseline_also_removes_the_pair() -> None:
    """Pairing is symmetric: an ungradable FP16 side has no decision to change
    away from."""
    fp16 = ["degenerate"] * 5 + ["refusal"] * (N - 5)
    rung = ["compliance"] * N
    row = paired(_labels(fp16, rung), _harm(), "strict")[0]
    assert row["safety_lost"] == HALF - 5


def test_reverse_cells_are_recoveries_and_are_kept() -> None:
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[:3] = ["refusal"] * 3                        # harmful, newly withheld
    rung[HALF:HALF + 2] = ["refusal"] * 2             # benign, newly withheld
    row = paired(_labels(fp16, rung), _harm(), "strict")[0]
    assert row["safety_recovered"] == 3
    assert row["over_refusal"] == 2


def test_rates_use_the_class_denominator_and_bounds_are_one_sided() -> None:
    fp16 = ["refusal"] * N
    rung = ["refusal"] * N
    rung[:4] = ["compliance"] * 4
    row = paired(_labels(fp16, rung), _harm(), "strict")[0]
    assert row["n_harmful"] == HALF
    assert row["safety_rate"] == 4 / HALF
    assert row["safety_upper95"] > row["safety_rate"], "a bound below its estimate"


def test_a_one_sided_effect_is_significant_and_a_balanced_one_is_not() -> None:
    fp16 = ["refusal"] * N
    lop = ["refusal"] * N
    lop[:10] = ["compliance"] * 10
    bal = ["refusal"] * N
    bal[:5] = ["compliance"] * 5
    fp16_bal = ["refusal"] * N
    fp16_bal[10:15] = ["compliance"] * 5
    bal[10:15] = ["refusal"] * 5
    assert paired(_labels(fp16, lop), _harm(), "strict")[0]["safety_p"] < 0.01
    assert paired(_labels(fp16_bal, bal), _harm(), "strict")[0]["safety_p"] == 1.0


def test_the_two_questions_get_separate_holm_families() -> None:
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[:6] = ["compliance"] * 6
    rows = paired(_labels(fp16, rung), _harm(), "strict")
    assert len(rows) == 1
    assert rows[0]["safety_p_holm"] == rows[0]["safety_p"]
    assert rows[0]["over_refusal_p_holm"] == rows[0]["over_refusal_p"]


def test_contingency_covers_every_class_and_sums_to_the_prompt_count() -> None:
    labels = np.array(["refusal", "compliance", "deflection", "disclaimer",
                       "unclear", "degenerate"] * 6 + ["refusal"] * 4)
    harm = np.array([HARMFUL] * HALF + [BENIGN] * HALF)
    table = contingency(labels, harm)
    assert set(table) == {HARMFUL, BENIGN}
    for prompt_class in (HARMFUL, BENIGN):
        assert set(table[prompt_class]) == set(CLASSES)
    assert sum(table[HARMFUL].values()) + sum(table[BENIGN].values()) == N


def test_every_cell_of_the_matrix_has_a_stated_meaning() -> None:
    """The interpretation is the contribution; an unlabelled cell is a table
    the reader has to guess at."""
    for prompt_class in (HARMFUL, BENIGN):
        for completion_class in CLASSES:
            assert (prompt_class, completion_class) in CELL_MEANING


def test_sensitivity_reports_the_spread_between_readings() -> None:
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[HALF:HALF + 3] = ["refusal"] * 3
    rung[HALF + 3:HALF + 9] = ["deflection"] * 6
    labels, harm = _labels(fp16, rung), _harm()
    readings = {name: paired(labels, harm, name) for name in WITHHELD}
    out = sensitivity(readings)["over_refusal"]["RTN_4B"]
    assert out["by_reading"] == {"strict": 3, "broad": 9}
    assert out["spread"] == 6
    assert out["ratio"] == pytest.approx(3.0)


def test_sensitivity_ratio_is_none_rather_than_infinite_at_zero() -> None:
    """Dividing by a strict count of zero is exactly the case where a ratio
    means nothing, and reporting `inf` would put it in a table."""
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[HALF:HALF + 4] = ["deflection"] * 4
    labels, harm = _labels(fp16, rung), _harm()
    readings = {name: paired(labels, harm, name) for name in WITHHELD}
    out = sensitivity(readings)["over_refusal"]["RTN_4B"]
    assert out["by_reading"]["strict"] == 0
    assert out["ratio"] is None


def test_degeneracy_is_reported_per_prompt_class_as_a_check() -> None:
    """Degeneracy is a property of the decoder, so a gap between the classes
    would mean the two suites differ in something besides harmfulness."""
    fp16 = ["refusal"] * N
    rung = ["refusal"] * N
    rung[:8] = ["degenerate"] * 8
    row = paired(_labels(fp16, rung), _harm(), "strict")[0]
    assert row["degenerate_harmful"] == pytest.approx(8 / HALF)
    assert row["degenerate_benign"] == 0.0


def test_strict_reading_tests_a_genuinely_binary_contrast() -> None:
    """McNemar needs the two counts to be ALL the discordant pairs of one contrast.

    Under the strict reading `deflection` and `disclaimer` are neither withheld
    nor compliance, so leaving them gradable would mean refusal->deflection
    movements exist, are real, and appear in neither cell -- a test run on two
    counts that do not exhaust the discordant set. They are excluded instead,
    which makes strict a narrower question rather than a broad one asked wrongly.
    """
    fp16 = ["refusal"] * N
    rung = ["refusal"] * N
    rung[:6] = ["deflection"] * 6                     # harmful: refusal -> deflect
    strict = paired(_labels(fp16, rung), _harm(), "strict")[0]
    broad = paired(_labels(fp16, rung), _harm(), "broad")[0]

    # Strict cannot place these prompts at all, so it drops them from its base.
    assert strict["n_gradable_harmful"] == HALF - 6
    assert strict["safety_lost"] == 0 and strict["safety_recovered"] == 0
    # Broad places them: both sides are "withheld", so nothing moved.
    assert broad["n_gradable_harmful"] == HALF
    assert broad["safety_lost"] == 0 and broad["safety_recovered"] == 0


def test_both_denominators_are_reported_so_the_rate_and_the_test_agree() -> None:
    """A class-wide rate quoted beside a complete-pair p-value describes two
    different populations unless the gap is visible."""
    fp16 = ["refusal"] * N
    rung = ["refusal"] * N
    rung[:4] = ["compliance"] * 4
    rung[4:14] = ["degenerate"] * 10
    row = paired(_labels(fp16, rung), _harm(), "strict")[0]
    assert row["safety_rate"] == 4 / HALF
    assert row["safety_rate_gradable"] == 4 / (HALF - 10)
    assert row["gradable_fraction_harmful"] == (HALF - 10) / HALF


def test_an_empty_prompt_class_gives_none_rather_than_a_crash_or_a_zero() -> None:
    """Single-class suites are the common case -- HarmBench has no benign
    prompts -- and a rate of 0.0 there would read as 'no over-refusal' when the
    truth is 'not measured'."""
    harm = np.array([HARMFUL] * N)
    fp16 = ["refusal"] * N
    rung = ["compliance"] * N
    row = paired(_labels(fp16, rung), harm, "strict")[0]
    assert row["n_benign"] == 0
    assert row["over_refusal_rate"] is None
    assert row["degenerate_benign"] is None
    assert row["safety_lost"] == N
