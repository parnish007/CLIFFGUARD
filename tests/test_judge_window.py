"""How much of a completion the judge reads is part of the instrument.

The judge truncates each completion before showing it to the model. That cap
was a flat 600 characters, chosen when every run used a 48-token generation
budget -- at which the longest completion on record is 465 characters, so the
cap never bound and every published verdict was formed on the whole text.

A 256-token budget breaks that. These models produce about 5.1 characters per
token, so a 256-token completion averages ~1310 characters and a 600-character
cap would hide more than half of it. Worse, it hides the *tail*: the part where
a model that hedges first and answers later does the answering. Grading a
256-token run under a 600-character cap measures a ~117-token budget wearing a
256-token label, which would answer the truncation objection with a quieter
instance of the same defect.

These tests pin the two properties that make raising the cap safe rather than
merely convenient.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "artifacts" / "runs"


def judge_module():
    """Import the script without executing its main().

    It imports torch lazily inside functions, so loading the module is cheap
    and does not require a GPU.
    """
    path = REPO / "scripts" / "classify_completions_judge.py"
    spec = importlib.util.spec_from_file_location("_judge_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_window_fits_a_256_token_completion() -> None:
    """256 tokens at ~5.1 chars/token needs well over 1300 characters."""
    mod = judge_module()
    assert mod.COMPLETION_CHARS >= 1400, (
        f"the judge shows only {mod.COMPLETION_CHARS} characters, which cannot "
        "hold a 256-token completion; a long-window run graded through this "
        "window is not a long-window measurement")


def test_an_unbound_window_does_not_change_the_cache_identity() -> None:
    """Widening the window must not invalidate caches it could not have altered.

    Keying on the raw window number would re-judge every 48-token run the
    moment the default moved, to reproduce labels that are provably identical
    -- nothing there comes within 130 characters of even the old cap. That cost
    would land in the one session that cannot spare it.

    So the window enters the fingerprint only when some completion is longer
    than it, and is absent otherwise. Absent rather than a sentinel: a sentinel
    is still a change to the hashed object, and would invalidate exactly the
    caches this preserves. The value below is the fingerprint the existing
    Qwen2.5-3B run was actually stored under.
    """
    import hashlib

    mod = judge_module()

    def fingerprint(chars: int | None) -> str:
        identity = {
            "judge": "Qwen/Qwen2.5-7B-Instruct",
            "four_bit": True,
            "labels": list(mod.LABELS),
            "template": mod.JUDGE_TEMPLATE,
            "n_prompts": 500,
        }
        if chars is not None:
            identity["completion_chars"] = chars
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    assert fingerprint(None) == "98850b5cca2ec1a8", (
        "the cache identity of an unbound window has changed, so every "
        "existing 48-token verdict would be re-judged to reproduce itself")
    assert fingerprint(2000) != fingerprint(None), (
        "a window that actually truncates must produce a different cache "
        "identity, or long-window verdicts would be pooled with short-window "
        "ones")


def test_the_window_is_part_of_the_cache_fingerprint() -> None:
    """Two caps must not share a verdict cache.

    The fingerprint already separates judges and templates for this reason. The
    window belongs with them: the same text graded at 600 and at 2000
    characters can receive different verdicts, so a long-window run reusing a
    short-window cache would compare two generation budgets while silently
    varying the instrument as well.
    """
    source = (REPO / "scripts" / "classify_completions_judge.py").read_text(
        encoding="utf-8")
    # Scoped to the block that builds the hashed object, so the assertion is
    # about the cache identity rather than about the word appearing anywhere in
    # the file -- it appears in the argparse help too.
    identity = source.split("identity: dict[str, Any] = {")[1].split(
        "fingerprint = hashlib.sha256(")[0]
    assert "completion_chars" in identity, (
        "the judge's completion window never enters the cache identity, so "
        "re-judging through a narrower window would silently reuse verdicts "
        "formed through a wider one")
    assert "if longest >" in identity, (
        "the window is added unconditionally, which invalidates every cache it "
        "could not have affected; it must be added only when it truncates")


@pytest.mark.skipif(not RUNS.exists(), reason="no run directories in this checkout")
def test_raising_the_window_cannot_change_an_existing_verdict() -> None:
    """Nothing the judge has ever graded reaches even the old 600-char cap.

    This is what makes the change a widening rather than a revision. It holds
    only for runs the judge actually read: the GSM8K ladders store completions
    past 600 characters, but they are scored by exact numeric match and carry
    no judge output, so they are excluded by looking for the judge's own file.
    """
    longest = 0
    judged_runs = 0
    for results in sorted(RUNS.glob("*/results")):
        if not (results / "judge_classification.json").exists():
            continue
        judged_runs += 1
        for path in sorted(results.glob("completions_*.json")):
            blob = json.loads(path.read_text(encoding="utf-8"))
            texts = blob["completions"] if isinstance(blob, dict) else blob
            longest = max(longest, max((len(t) for t in texts), default=0))

    if not judged_runs:
        pytest.skip("no judged runs in this checkout")
    assert longest <= 600, (
        f"a judged completion is {longest} characters, past the 600-character "
        "cap that was in force when it was graded. Raising the cap therefore "
        "changes what the judge reads for text already reported, and those "
        "runs must be re-judged rather than reused.")
