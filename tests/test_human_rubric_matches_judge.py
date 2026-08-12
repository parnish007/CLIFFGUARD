"""The human rubric and the judge's option text must stay the same words.

The validation compares a person's labels against a model judge's on identical
completions, and that only isolates the GRADER if both were asked the same
question. Give the human a better-worded rubric and a disagreement between them
measures the rubric instead -- which is a real effect, just not the one the
paper claims to be measuring.

Nothing enforces that by construction: the rubric lives in a markdown file a
person edits and the judge's lives in a Python constant. So this asserts it.
It has already caught one drift, where wrapping the definitions in a markdown
blockquote pushed `>` characters into the middle of every wrapped line.

Editorial matter AROUND the definitions is fine and expected -- how to work
through the sheet, which file to open, what the repeats are for. This checks the
five definitions only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTRUCTIONS = REPO / "docs" / "paper" / "human_sample" / "instructions.md"


def taxonomy_module():
    path = REPO / "scripts" / "classify_completion_taxonomy.py"
    spec = importlib.util.spec_from_file_location("_taxonomy_rubric", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalise(text: str) -> str:
    """Collapse whitespace and case; markdown wraps where Python does not."""
    return re.sub(r"\s+", " ", text).strip().lower()


@pytest.mark.skipif(not INSTRUCTIONS.is_file(),
                    reason="human sample not drawn in this checkout")
@pytest.mark.parametrize("label", ["REFUSE", "COMPLY", "DEFLECT", "DISCLAIM",
                                   "UNCLEAR"])
def test_each_definition_is_the_judges_own_words(label: str) -> None:
    options = taxonomy_module().LETTER_OPTIONS
    # The judge's text is "LABELLED: <definition>"; the markdown drops the
    # verb form and uses a dash, so only the definition body is comparable.
    body = normalise(options[label].split(":", 1)[1])
    document = normalise(INSTRUCTIONS.read_text(encoding="utf-8"))
    assert body in document, (
        f"the {label} definition in instructions.md is no longer the judge's "
        "own wording. Either revert it, or accept that the comparison now "
        "measures the rubric as well as the grader and say so in the paper.")


@pytest.mark.skipif(not INSTRUCTIONS.is_file(),
                    reason="human sample not drawn in this checkout")
def test_the_instructions_name_the_sheet_that_is_actually_labelled() -> None:
    """A guide pointing at the wrong file is worse than no guide.

    Two sheets sit in that directory -- the drawn sample and the kit built from
    it -- and only the kit carries the repeated rows that produce the
    reliability figure. Labelling the other one silently loses it.
    """
    document = INSTRUCTIONS.read_text(encoding="utf-8")
    assert "sheet_to_label.csv" in document
