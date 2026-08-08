"""The methodology document must describe the pipeline that exists.

A reference someone consults to avoid re-deriving a decision is only worth
having if it is true. The failure mode is specific and slow: a constant is tuned
in the code, the document keeps the old value, and six weeks later somebody
reads the document, believes it, and reasons from a number that is no longer
real. Nothing errors; the reasoning is simply wrong.

So the load-bearing facts are checked against their source rather than trusted.
This is deliberately narrow -- it pins the values a reader would ACT on
(thresholds, label sets, token ids, file paths), not the prose around them.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "methodology.md"


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


def test_the_document_exists_and_is_not_a_stub(doc: str) -> None:
    assert len(doc) > 4000, "the methodology document has been gutted"


def test_degeneracy_thresholds_match_the_gate(doc: str) -> None:
    """These four decide what counts as language. A stale value here would send
    someone looking for a bug in the gate that is not there."""
    from scripts.run_behavioural_ladder import (
        CLASSIFY_WINDOW,
        DEGENERACY_NLL_MULTIPLE,
        MAX_TOKEN_SHARE,
        MIN_ALPHA_FRACTION,
        MIN_DISTINCT_TRIGRAM,
    )

    for value, label in ((DEGENERACY_NLL_MULTIPLE, "NLL multiple"),
                         (MIN_DISTINCT_TRIGRAM, "distinct-trigram"),
                         (MAX_TOKEN_SHARE, "max token share"),
                         (MIN_ALPHA_FRACTION, "alphabetic fraction"),
                         (CLASSIFY_WINDOW, "classify window")):
        assert str(value) in doc, f"{label} ({value}) is not stated in the document"


def test_the_marker_list_size_matches(doc: str) -> None:
    from scripts.run_behavioural_ladder import REFUSAL_MARKERS

    assert f"{len(REFUSAL_MARKERS)} decline markers" in doc


def test_the_marker_variant_sizes_match(doc: str) -> None:
    """The non-monotonicity result is stated in terms of these sizes."""
    from scripts.classify_completions_judge import MARKER_VARIANTS

    sizes = [len(v) for v in MARKER_VARIANTS.values()]
    rendered = " → ".join(str(s) for s in sizes)
    assert rendered in doc, f"variant sizes {sizes} not stated as '{rendered}'"


def test_every_label_of_both_graders_appears(doc: str) -> None:
    from scripts.classify_completion_taxonomy import LABELS as FIVE
    from scripts.classify_completions_judge import LABELS as THREE

    for label in set(FIVE) | set(THREE):
        assert label in doc, f"{label} is not documented"


def test_the_endpoint_partition_matches_the_code(doc: str) -> None:
    """If the document and the code disagree about which classes count as
    over-refusal versus capability failure, the matrix is being read wrongly."""
    from scripts.analyse_matrix import (
        COMPLIED,
        WITHHELD_BY_CHOICE,
        WITHHELD_BY_INCAPACITY,
    )

    for name in COMPLIED + WITHHELD_BY_CHOICE + WITHHELD_BY_INCAPACITY:
        assert name in doc, f"class {name} is not documented"

    # And the split itself, in the direction the code implements.
    choice = " + ".join(WITHHELD_BY_CHOICE)
    incapacity = " + ".join(WITHHELD_BY_INCAPACITY)
    assert choice in doc, f"'{choice}' (the declined side) is not stated"
    assert incapacity in doc, f"'{incapacity}' (the could-not side) is not stated"


def test_documented_judge_token_ids_are_the_real_ones(doc: str) -> None:
    """Asserted at load time in the grader, so a wrong value here would send a
    reader hunting a tokenizer problem that does not exist."""
    for label, token_id in (("REFUSE", 38029), ("COMPLY", 7682),
                            ("DEFLECT", 23865), ("DISCLAIM", 24717),
                            ("UNCLEAR", 75255)):
        assert f"{label} {token_id}" in doc, f"{label}={token_id} not documented"


def test_every_script_the_document_names_exists(doc: str) -> None:
    named = set(re.findall(r"(scripts/[a-z_0-9]+\.py)", doc))
    named |= {f"scripts/{n}" for n in re.findall(r"`([a-z_0-9]+\.py)`", doc)}
    assert named, "the document names no script at all"
    missing = sorted(n for n in named if not (REPO / n).exists())
    assert not missing, f"documented scripts that do not exist: {missing}"


def test_every_sibling_doc_it_links_to_exists(doc: str) -> None:
    linked = set(re.findall(r"\]\((\w[\w_]*\.md)\)", doc))
    missing = sorted(n for n in linked if not (REPO / "docs" / n).exists())
    assert not missing, f"broken doc links: {missing}"


def test_the_reversed_decisions_are_recorded(doc: str) -> None:
    """The section that stops an argument happening twice. Each of these was
    shipped one way, found wrong, and changed; losing the record means
    re-deriving it, or worse, quietly reverting to the wrong version."""
    for marker in ("confound", "post-treatment", "not an identity",
                   "margin", "package, not a concept"):
        assert marker in doc, f"the reversal keyed on '{marker}' is missing"


def test_the_standing_working_rules_survive(doc: str) -> None:
    for rule in ("git add -A", "untracked", "Push only when asked"):
        assert rule in doc, f"working rule '{rule}' is missing"
