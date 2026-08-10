"""Permuting which class is offered under which letter, without moving a word.

The corrected scorer replaced label words with single-token letters A-E. That
removes the tokenization asymmetry the paper is about, and it introduces a
question the paper cannot answer by argument: a judge may prefer a LETTER,
independently of what the letter stands for. Testing that needs the assignment
to be a parameter, and needs the permuted prompt to differ from the canonical
one in nothing but the assignment.

Two properties carry the whole design, and both fail silently if broken.

The canonical order must reproduce the published prompt BYTE FOR BYTE. It is
hashed into the cache fingerprint, so a stray newline would invalidate every
corrected grading on disk and quietly re-run 2,900 judge pairs to produce
numbers that no longer match the manuscript.

And a permuted grading must not be confusable with the canonical one. The
fingerprint has to move, or a permuted verdict would be served from -- or
written into -- the canonical cache, and the option-order experiment would
silently overwrite the result it exists to check.
"""

from __future__ import annotations

import itertools

import pytest

from scripts import classify_completion_taxonomy as five
from scripts import classify_completions_judge as three

GRADERS = pytest.mark.parametrize("mod", [three, five],
                                  ids=["three-way", "five-way"])


@GRADERS
def test_the_default_order_reproduces_the_published_prompt(mod) -> None:
    assert mod.letter_template() == mod.LETTER_TEMPLATE
    assert mod.letter_template(mod.LABELS) == mod.LETTER_TEMPLATE


@GRADERS
def test_every_permutation_offers_every_class_exactly_once(mod) -> None:
    for order in itertools.permutations(mod.LABELS):
        text = mod.letter_template(order)
        for k, name in enumerate(order):
            assert f"{mod.LETTERS[k]}. {mod.LETTER_OPTIONS[name]}" in text
        # No class is dropped and none is offered twice: the option bodies open
        # with a distinctive past-tense word, so counting those counts options.
        for name in mod.LABELS:
            first = mod.LETTER_OPTIONS[name].split(":")[0]
            assert text.count(f". {first}:") == 1


@GRADERS
def test_a_permutation_changes_only_the_assignment(mod) -> None:
    """Same sentences, different letters. Nothing else may differ."""
    order = tuple(reversed(mod.LABELS))
    canonical, permuted = mod.letter_template(), mod.letter_template(order)
    assert canonical != permuted

    def strip_letters(text: str) -> list[str]:
        lines = []
        for line in text.splitlines():
            for letter in mod.LETTERS:
                if line.startswith(f"{letter}. "):
                    line = line[3:]
                    break
            lines.append(line)
        return lines

    assert sorted(strip_letters(canonical)) == sorted(strip_letters(permuted))


@GRADERS
def test_a_non_permutation_is_refused_rather_than_scored(mod) -> None:
    doubled = (mod.LABELS[0],) * len(mod.LABELS)
    with pytest.raises(SystemExit):
        mod.letter_template(doubled)
    with pytest.raises(SystemExit):
        mod.letter_template(mod.LABELS[:-1])
    with pytest.raises(SystemExit):
        mod.letter_template(mod.LABELS + ("NONSENSE",))


@GRADERS
def test_distinct_permutations_give_distinct_prompts(mod) -> None:
    """Which is what keeps their cache fingerprints apart.

    The fingerprint hashes the template, so this is the property that stops a
    permuted grading from being served out of the canonical cache. Checked over
    every permutation rather than a sample, because there are only 6 and 120.
    """
    seen = {mod.letter_template(o) for o in itertools.permutations(mod.LABELS)}
    import math
    assert len(seen) == math.factorial(len(mod.LABELS))


@GRADERS
def test_the_option_bodies_name_no_letter(mod) -> None:
    """A body that said "as in option B" would break under permutation.

    Nothing does today. This test exists so that nothing starts to: the failure
    would be a prompt that contradicts itself, and the grader would answer it
    anyway.
    """
    for name, body in mod.LETTER_OPTIONS.items():
        for letter in mod.LETTERS:
            assert f"option {letter}" not in body, name
            assert f"({letter})" not in body, name
