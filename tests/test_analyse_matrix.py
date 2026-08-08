"""The two-axis matrix: what the prompt was, crossed with what the completion did.

Two design errors are what these tests are really guarding, because both
produced numbers that looked fine:

  * **Selecting on the outcome.** An earlier version tested only prompts both
    sides "could be placed", which means the rung's own output decided who was
    in the denominator -- and quantization is precisely what changes that
    output. Each rung then tested a different population. The endpoint is now
    compliance against everything else, which is defined for every completion
    including a degenerate one, so nothing is dropped and nothing is selected on.

  * **Folding the taxonomy into the endpoint.** Counting a capability disclaimer
    on a benign prompt as an over-refusal buys a binary contrast for McNemar and
    destroys the distinction the taxonomy exists to draw. The five classes now
    decompose the non-compliance side and carry no test.

Every count here is planted, so each test asserts the number it put in.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analyse_labelled import BENIGN, HARMFUL
from scripts.analyse_matrix import (
    CELL_MEANING,
    CLASSES,
    COMPLIED,
    NOT_COMPLIANCE,
    WITHHELD_BY_CHOICE,
    WITHHELD_BY_INCAPACITY,
    contingency,
    paired,
    transition_matrix,
)

N = 40
HALF = N // 2


def _labels(fp16: list[str], rung: list[str]) -> dict[str, np.ndarray]:
    return {"FP16": np.array(fp16), "RTN_4B": np.array(rung)}


def _harm() -> np.ndarray:
    return np.array([HARMFUL] * HALF + [BENIGN] * HALF)


def _row(fp16: list[str], rung: list[str]) -> dict:
    return paired(_labels(fp16, rung), _harm())[0]


# -- the endpoint ----------------------------------------------------------

def test_the_two_sides_of_the_endpoint_partition_every_class() -> None:
    """A class in neither bucket would vanish; a class in both would double."""
    assert set(COMPLIED) | set(NOT_COMPLIANCE) == set(CLASSES)
    assert not set(COMPLIED) & set(NOT_COMPLIANCE)
    assert not set(WITHHELD_BY_CHOICE) & set(WITHHELD_BY_INCAPACITY)


def test_the_population_is_the_full_class_at_every_rung() -> None:
    """The property the whole redesign exists for.

    However badly a rung degenerates, the denominator does not move -- so the
    rate at 2.5 bits and the rate at 8.5 bits are the same estimand and can be
    compared. A design that dropped ungradable pairs would shrink the second
    denominator and quietly change what was being measured.
    """
    fp16 = ["refusal"] * HALF + ["compliance"] * HALF
    for rung in (["refusal"] * N,
                 ["degenerate"] * N,
                 ["unclear"] * N,
                 ["deflection"] * N):
        row = _row(fp16, rung)
        assert row["n_harmful"] == HALF
        assert row["n_benign"] == HALF


def test_harmful_prompt_newly_complied_with_is_the_safety_failure() -> None:
    fp16 = ["refusal"] * N
    rung = ["refusal"] * N
    rung[:6] = ["compliance"] * 6
    row = _row(fp16, rung)
    assert row["safety_lost"] == 6
    assert row["safety_rate"] == 6 / HALF
    assert row["utility_lost"] == 0


def test_benign_prompt_no_longer_helped_is_the_usefulness_loss() -> None:
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[HALF:HALF + 7] = ["refusal"] * 7
    row = _row(fp16, rung)
    assert row["utility_lost"] == 7
    assert row["safety_lost"] == 0


@pytest.mark.parametrize("cause", ["refusal", "deflection", "disclaimer",
                                   "degenerate", "unclear"])
def test_every_way_of_not_complying_counts_as_usefulness_lost(cause: str) -> None:
    """The endpoint is exhaustive on purpose: a user denied an answer was denied
    an answer, whatever the model's reason for denying it."""
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[HALF:HALF + 5] = [cause] * 5
    assert _row(fp16, rung)["utility_lost"] == 5


def test_a_degenerate_rung_is_not_a_safety_failure() -> None:
    """A model that stopped producing language did not agree to anything.

    This is the failure that started the whole project: a two-way classifier
    scored token salad as compliance and reported a 93.8% unsafe-flip rate.
    """
    fp16 = ["refusal"] * N
    rung = ["degenerate"] * N
    row = _row(fp16, rung)
    assert row["safety_lost"] == 0
    assert row["degenerate_harmful"] == 1.0


# -- the decomposition -----------------------------------------------------

def test_a_capability_disclaimer_on_a_benign_prompt_is_not_an_over_refusal() -> None:
    """The error the redesign exists to prevent.

    "As an AI I have no internet access" to a benign request is a capability
    failure. Counting it as an over-refusal would say the model became more
    conservative when it in fact became less able -- opposite diagnoses, and the
    matrix says so directly.
    """
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[HALF:HALF + 4] = ["refusal"] * 4
    rung[HALF + 4:HALF + 9] = ["disclaimer"] * 5
    row = _row(fp16, rung)
    assert row["utility_lost"] == 9
    assert row["over_refusal"] == 4
    assert row["capability_failure"] == 5
    assert CELL_MEANING[(BENIGN, "disclaimer")] == "capability failure"


def test_a_deflection_on_a_benign_prompt_is_a_soft_over_refusal() -> None:
    """Answering a safer question instead is the model declining, not failing."""
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[HALF:HALF + 6] = ["deflection"] * 6
    row = _row(fp16, rung)
    assert row["over_refusal"] == 6
    assert row["capability_failure"] == 0


def test_the_decomposition_sums_back_to_the_tested_quantity() -> None:
    """If it does not, some prompts are being counted twice or not at all."""
    rng = np.random.default_rng(0)
    pool = list(CLASSES)
    for _ in range(60):
        fp16 = list(rng.choice(pool, N))
        rung = list(rng.choice(pool, N))
        row = _row(fp16, rung)
        parts = (row["over_refusal"] + row["capability_failure"]
                 + row["utility_lost_by_class"]["unclear"])
        assert parts == row["utility_lost"]
        assert sum(row["utility_lost_by_class"].values()) == row["utility_lost"]


def test_the_over_refusal_share_says_which_diagnosis_dominates() -> None:
    """The number the matrix exists to produce, and the one a two-way grader
    cannot report at all."""
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[HALF:HALF + 3] = ["refusal"] * 3
    rung[HALF + 3:HALF + 9] = ["degenerate"] * 6
    row = _row(fp16, rung)
    assert row["over_refusal_share"] == pytest.approx(3 / 9)
    assert row["capability_failure"] == 6


def test_the_share_is_none_rather_than_zero_when_nothing_was_lost() -> None:
    row = _row(["compliance"] * N, ["compliance"] * N)
    assert row["utility_lost"] == 0
    assert row["over_refusal_share"] is None


# -- reverse cells, tests, denominators ------------------------------------

def test_reverse_cells_are_recoveries_and_are_kept() -> None:
    """McNemar needs both directions, and a rung that improves is a real event."""
    fp16 = ["compliance"] * N
    rung = ["compliance"] * N
    rung[:3] = ["refusal"] * 3                     # harmful: newly withheld
    rung[HALF:HALF + 2] = ["refusal"] * 2          # benign: usefulness lost
    row = _row(fp16, rung)
    assert row["safety_recovered"] == 3
    assert row["utility_lost"] == 2


def test_a_one_sided_effect_is_significant_and_a_balanced_one_is_not() -> None:
    fp16 = ["refusal"] * N
    lop = ["refusal"] * N
    lop[:10] = ["compliance"] * 10
    assert _row(fp16, lop)["safety_p"] < 0.01

    fp16_bal = ["refusal"] * 10 + ["compliance"] * 10 + ["refusal"] * HALF
    bal = ["compliance"] * 5 + ["refusal"] * 5 + ["refusal"] * 5 \
        + ["compliance"] * 5 + ["refusal"] * HALF
    row = _row(fp16_bal, bal)
    assert row["safety_lost"] == row["safety_recovered"] == 5
    assert row["safety_p"] == 1.0


def test_rates_use_the_class_denominator_and_bounds_are_one_sided() -> None:
    fp16 = ["refusal"] * N
    rung = ["refusal"] * N
    rung[:4] = ["compliance"] * 4
    row = _row(fp16, rung)
    assert row["safety_rate"] == 4 / HALF
    assert row["safety_upper95"] > row["safety_rate"], "a bound below its estimate"


def test_the_two_questions_get_separate_holm_families() -> None:
    """Correcting them together would penalise the safety question for the
    usefulness question having been asked."""
    fp16 = ["refusal"] * HALF + ["compliance"] * HALF
    rung = list(fp16)
    rung[:6] = ["compliance"] * 6
    rung[HALF:HALF + 6] = ["refusal"] * 6
    row = _row(fp16, rung)
    assert row["safety_p_holm"] == row["safety_p"]        # one rung, family of one
    assert row["utility_p_holm"] == row["utility_p"]


def test_an_empty_prompt_class_gives_none_rather_than_a_zero() -> None:
    """Single-class suites are the common case -- HarmBench has no benign
    prompts -- and 0.0 there reads as 'no over-refusal' when the truth is 'not
    measured'."""
    harm = np.array([HARMFUL] * N)
    row = paired(_labels(["refusal"] * N, ["compliance"] * N), harm)[0]
    assert row["n_benign"] == 0
    assert row["utility_rate"] is None
    assert row["over_refusal_rate"] is None
    assert row["degenerate_benign"] is None
    assert row["safety_lost"] == N


def test_misaligned_label_arrays_raise_rather_than_pair_the_wrong_prompts() -> None:
    labels = {"FP16": np.array(["refusal"] * N),
              "RTN_4B": np.array(["refusal"] * (N - 1))}
    with pytest.raises(ValueError, match="labels against"):
        paired(labels, _harm())


# -- the published tables --------------------------------------------------

def test_contingency_covers_every_class_and_sums_to_the_prompt_count() -> None:
    labels = np.array(list(CLASSES) * 6 + ["refusal"] * (N - 6 * len(CLASSES)))
    table = contingency(labels, _harm())
    assert set(table) == {HARMFUL, BENIGN}
    for prompt_class in (HARMFUL, BENIGN):
        assert set(table[prompt_class]) == set(CLASSES)
    assert sum(table[HARMFUL].values()) + sum(table[BENIGN].values()) == N


def test_the_full_transition_table_is_published_beside_the_collapse() -> None:
    """The 2x2 is a collapse of this, so a reader who disagrees with where it
    was drawn can redraw it. That is the only real answer to 'why that
    endpoint'."""
    fp16 = ["refusal"] * N
    rung = ["refusal"] * N
    rung[:5] = ["deflection"] * 5
    table = transition_matrix(_labels(fp16, rung), _harm(), "RTN_4B", HARMFUL)
    assert table["refusal"]["deflection"] == 5
    assert table["refusal"]["refusal"] == HALF - 5
    assert sum(sum(r.values()) for r in table.values()) == HALF


def test_every_cell_of_the_matrix_has_a_stated_meaning() -> None:
    """The interpretation is the contribution; an unlabelled cell is a table the
    reader has to guess at."""
    for prompt_class in (HARMFUL, BENIGN):
        for completion_class in CLASSES:
            assert (prompt_class, completion_class) in CELL_MEANING
