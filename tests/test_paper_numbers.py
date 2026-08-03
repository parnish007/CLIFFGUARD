"""The manuscript's prose numbers must match the measurements they cite.

Tables and figures are generated and cannot drift. Prose is typed and can, and a
wrongly-rounded bound turns a true statement false: the simultaneous upper bound
is 4.6194%, so "below 4.6%" is both imprecise and wrong. This test runs the same
checker in CI so that drift fails the build rather than reaching a reader.
"""

from __future__ import annotations

from pathlib import Path
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

    from scripts.check_paper_numbers import CHECKS

    stats = json.loads(STATS.read_text(encoding="utf-8"))
    text = TEX.read_text(encoding="utf-8")
    missing = [(c.label, c.expected(stats)) for c in CHECKS
               if c.expected(stats) not in text]
    assert not missing, (
        "manuscript prose disagrees with review_stats.json: "
        + ", ".join(f"{label} should be {value}" for label, value in missing))
