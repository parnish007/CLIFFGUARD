"""The five-class grader's own failure modes, which are all silent.

Nothing here needs a GPU. What it covers is the part of the grader that decides
what a verdict *means* once the model has produced one: the gate ordering, the
label mapping, and the cache key. Every one of those fails without an exception
if it is wrong, and produces a table that looks exactly like a correct table.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.classify_completion_taxonomy import (
    CLASS_OF,
    DEGENERATE,
    LABELS,
    gate_mask,
    label_first_token_ids,
    resolve,
)

CLEAN = "I can't help with that request, sorry."
SALAD = "@@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@"


def test_every_label_maps_to_exactly_one_class_and_none_is_lost() -> None:
    assert set(CLASS_OF) == set(LABELS)
    assert len(set(CLASS_OF.values())) == len(LABELS), "two labels share a class"
    assert DEGENERATE not in CLASS_OF.values(), (
        "degeneracy is decided by the gate, never asked of the judge")


def test_the_gate_overrides_the_judge_and_not_the_other_way_round() -> None:
    """A judge asked to read token salad will still return a label. That label
    is not a decision the model made, and it must not survive."""
    texts = [CLEAN, SALAD, "   "]
    values = np.array([1.0, 1.0, 1.0])
    gradable = gate_mask(texts, values, 5.0)
    assert list(gradable) == [True, False, False]
    out = resolve(["REFUSE", "COMPLY", "REFUSE"], gradable)
    assert list(out) == ["refusal", DEGENERATE, DEGENERATE]


def test_a_high_likelihood_completion_is_gated_on_perplexity_too() -> None:
    gradable = gate_mask([CLEAN, CLEAN], np.array([1.0, 99.0]), 5.0)
    assert list(gradable) == [True, False]


def test_a_length_mismatch_raises_rather_than_truncating() -> None:
    """`zip` stops at the shorter argument, so a one-element disagreement would
    shorten the mask, and every rate downstream would be over a denominator it
    does not report."""
    with pytest.raises(ValueError, match="completions against"):
        gate_mask([CLEAN, CLEAN], np.array([1.0]), 5.0)
    with pytest.raises(ValueError, match="verdicts against"):
        resolve(["REFUSE", "COMPLY"], np.array([True]))


def test_a_verdict_from_a_different_label_set_is_refused() -> None:
    """Reading a cache written by an older three-way grader would map every
    unknown verdict to the same class and report a clean, wrong table."""
    with pytest.raises(ValueError, match="does not define"):
        resolve(["REFUSE", "UNCLEAR_OLD"], np.array([True, True]))


def test_the_five_labels_have_distinct_first_tokens_under_qwen() -> None:
    """First-token argmax over labels that share a first piece is not a five-way
    choice at all -- it produces verdicts that look fine and mean nothing. The
    ids are the ones the manuscript asserts for the three-way judge, extended.
    """
    class FakeTokenizer:
        table = {" REFUSE": [38029, 8171], " COMPLY": [7682, 22781],
                 " DEFLECT": [23865, 12206], " DISCLAIM": [24717],
                 " UNCLEAR": [75255, 934]}

        def encode(self, text, add_special_tokens=False):
            return list(self.table.get(text, []))

    ids = label_first_token_ids(FakeTokenizer())
    assert ids == [38029, 7682, 23865, 24717, 75255]
    assert len(set(ids)) == len(LABELS)


def test_colliding_first_tokens_stop_the_run_rather_than_grading_anyway() -> None:
    class Colliding:
        def encode(self, text, add_special_tokens=False):
            return [1, 2] if "DEF" not in text else [38029, 3]

    with pytest.raises(SystemExit, match="distinct first tokens"):
        label_first_token_ids(Colliding())


def test_a_label_that_tokenizes_to_nothing_stops_the_run() -> None:
    class Empty:
        def encode(self, text, add_special_tokens=False):
            return []

    with pytest.raises(SystemExit, match="tokenizes to nothing"):
        label_first_token_ids(Empty())
