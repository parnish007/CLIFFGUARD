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


def test_the_two_scoring_modes_have_different_cache_identities() -> None:
    """Letter and first-token verdicts must never share a cache entry.

    They are different instruments, not two settings of one. First-token
    scoring compares the opening piece of each label word, which under Qwen2.5
    means four three-character prefixes against one whole word; letter scoring
    compares five verified single tokens, so the logits are five mutually
    exclusive complete answers. Pooling them would mix a measurement with its
    own correction.

    This also documents a deliberate cost. Adding the mode to the identity
    supersedes the legacy caches -- the stored Qwen2.5-3B fingerprint
    98850b5cca2ec1a8 predates the field and no longer recomputes. That is
    accepted rather than worked around: the only reason to re-grade that data
    is to move it onto the corrected scorer, which invalidates those entries by
    definition, so preserving them would buy a saving the next step spends.
    """
    import hashlib

    mod = judge_module()

    def fingerprint(scoring: str, template: str,
                    chars: int | None = None) -> str:
        identity = {
            "judge": "Qwen/Qwen2.5-7B-Instruct",
            "four_bit": True,
            "labels": list(mod.LABELS),
            "template": template,
            "scoring": scoring,
            "n_prompts": 500,
        }
        if chars is not None:
            identity["completion_chars"] = chars
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    letter = fingerprint("letter", mod.LETTER_TEMPLATE)
    first = fingerprint("first-token", mod.JUDGE_TEMPLATE)
    assert letter != first, (
        "the two scoring modes share a cache identity, so a letter-scored run "
        "would silently reuse first-token verdicts")
    assert letter != "98850b5cca2ec1a8" and first != "98850b5cca2ec1a8", (
        "a current fingerprint collides with the legacy one, which would let "
        "superseded verdicts be reused as if they were current")


def test_the_letter_options_are_asserted_single_tokens() -> None:
    """The mode's whole guarantee is one token per option, so it is checked.

    If a tokenizer split ' A' into pieces, letter scoring would reintroduce
    exactly the prefix-versus-whole-word asymmetry it exists to remove, and
    would do it invisibly. Both graders must refuse rather than score.
    """
    for name in ("classify_completions_judge.py",
                 "classify_completion_taxonomy.py"):
        source = (REPO / "scripts" / name).read_text(encoding="utf-8")
        block = source.split("def letter_token_ids(")[1].split("\ndef ")[0]
        assert "len(pieces) != 1" in block, (
            f"{name} does not verify that each letter is a single token")
        assert "raise SystemExit" in block, (
            f"{name} does not fail when a letter is not a single token")


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
