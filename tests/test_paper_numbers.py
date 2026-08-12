"""The manuscript's prose numbers must match the measurements they cite.

Tables and figures are generated and cannot drift. Prose is typed and can, and a
wrongly-rounded bound turns a true statement false: the simultaneous upper bound
is 4.6194%, so "below 4.6%" is both imprecise and wrong. This test runs the same
checker in CI so that drift fails the build rather than reaching a reader.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

STATS = REPO_ROOT / "docs" / "paper" / "review_stats.json"
TEX = REPO_ROOT / "docs" / "paper" / "cliff_artifact.tex"


@pytest.mark.skipif(not STATS.exists() or not TEX.exists(),
                    reason="paper sources are not present in this checkout")
def test_paper_prose_matches_measurements() -> None:
    import json
    import re

    from scripts.check_paper_numbers import (
        CHECKS, DATA_CHECKS, live_tex, load_stats)

    # Through the checker's own loader, not json.loads: the probe checks read
    # the corrected refit spliced over review_stats.json, and a test that
    # assembled its own object would verify the manuscript against whichever
    # grading it happened to reconstruct.
    stats = load_stats()
    data = json.loads((REPO_ROOT / "docs" / "paper" / "data.json").read_text(encoding="utf-8"))
    # live_tex, not the raw source: it strips commented-out sentences and the
    # original-scorer dagger, exactly as the CLI checker does. A test reading
    # raw source both accepts numbers the reader never sees and rejects ones
    # they do.
    text = live_tex(TEX.read_text(encoding="utf-8"))
    missing = [
        (check.label, check.value(data))
        for check in DATA_CHECKS
        if re.search(check.pattern(check.value(data)), text) is None
    ] + [
        (check.label, check.value(stats))
        for check in CHECKS
        if re.search(check.pattern(check.value(stats)), text) is None
    ]
    assert not missing, (
        "manuscript prose disagrees with review_stats.json: "
        + ", ".join(f"{label} should be {value}" for label, value in missing))


@pytest.mark.skipif(not STATS.exists() or not TEX.exists(),
                    reason="paper sources are not present in this checkout")
def test_checks_are_anchored_to_context() -> None:
    """A check that matches a bare number is not a check.

    The first version of this guard searched for substrings, so its GSM8K
    assertion for "62" was satisfied by the author's email address and by an
    unrelated "4.62" -- it could not fail. Every pattern must carry surrounding
    words, so require more than digits and separators.
    """

    from scripts.check_paper_numbers import CHECKS, load_stats

    # Through the checker's own loader, not json.loads: the probe checks read
    # the corrected refit spliced over review_stats.json, and a test that
    # assembled its own object would verify the manuscript against whichever
    # grading it happened to reconstruct.
    stats = load_stats()
    bare = [c.label for c in CHECKS
            if not re.search(r"[A-Za-z]{3}", c.pattern(c.value(stats)))]
    assert not bare, f"checks with no textual context: {bare}"
