"""The Colab restore step unpacks an archive over the checkout. Keep it fenced.

Nothing else in the test suite reads notebook code, so this block was the one
piece of executable logic in the project with no coverage at all -- and it is the
piece that writes to the repository. It was `extractall('.')`.

The failure it allows is not path traversal, and saying so precisely matters
because the imprecise version invites the wrong fix. CPython's `extractall`
strips `..` and leading separators, so a member named `../../x` lands inside the
working directory rather than outside it; the test below asserts that, so the
claim stays checked rather than remembered. What it does allow, with no
traversal at all, is a member named `scripts/classify_completions_judge.py`
replacing the grader, or `.git/hooks/post-checkout` running on the next git
command. Both were reproducible.

The archive comes from the user's own Drive, so this is integrity rather than
attack surface -- but the integrity in question is a project whose central
argument is that its instruments and its frozen protocol were not edited between
being written and being run.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("colab_round4.ipynb", "colab_round5.ipynb")

# One legitimate member, and every way an archive could reach somewhere it
# should not.
HOSTILE = {
    "artifacts/runs/good/manifest.json": '{"ok": true}',
    "../../escaped.txt": "outside the checkout",
    "/abs/rooted.txt": "absolute path",
    "scripts/classify_completions_judge.py": "# REPLACED GRADER",
    ".git/hooks/post-checkout": "#!/bin/sh\necho pwned",
    "docs/preregistration_round5.md": "# a different protocol entirely",
    "artifacts/runs/../../escaped_via_middle.txt": "traversal mid-path",
}


def restore_block(notebook: str) -> str:
    """The prior-runs stanza, lifted out of the notebook it lives in."""
    nb = json.loads((REPO / "notebooks" / notebook).read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if "prior_runs.zip" not in text:
            continue
        start = text.index("# A Drive zip still overrides")
        end = text.index("# Rebuild only what is still missing")
        return text[start:end]
    raise AssertionError(f"{notebook}: no prior_runs.zip block")


@pytest.fixture()
def checkout(tmp_path: Path) -> Path:
    work = tmp_path / "repo"
    (work / "scripts").mkdir(parents=True)
    (work / "docs").mkdir()
    (work / ".git" / "hooks").mkdir(parents=True)
    (work / "scripts" / "classify_completions_judge.py").write_text(
        "# THE REAL GRADER", encoding="utf-8")
    (work / "docs" / "preregistration_round5.md").write_text(
        "# THE REAL PROTOCOL", encoding="utf-8")
    drive = tmp_path / "drive"
    drive.mkdir()
    with zipfile.ZipFile(drive / "prior_runs.zip", "w") as zf:
        for name, body in HOSTILE.items():
            zf.writestr(name, body)
    return work


def run_restore(notebook: str, checkout: Path) -> None:
    here = Path.cwd()
    os.chdir(checkout)
    try:
        exec(restore_block(notebook),
             {"pathlib": __import__("pathlib"), "chr": chr, "print": print,
              "DRIVE_ROOT": checkout.parent / "drive"})
    finally:
        os.chdir(here)


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_the_archive_cannot_replace_the_grader(notebook, checkout) -> None:
    run_restore(notebook, checkout)
    assert (checkout / "scripts" / "classify_completions_judge.py").read_text(
        encoding="utf-8") == "# THE REAL GRADER"


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_the_archive_cannot_replace_the_frozen_protocol(notebook, checkout) -> None:
    """The one file a preregistered round must not be able to rewrite.

    Round 5 restores in the same cell that hashes this file, restore first. An
    archive carrying its own copy would have been unpacked and then recorded as
    the protocol in force.
    """
    run_restore(notebook, checkout)
    assert (checkout / "docs" / "preregistration_round5.md").read_text(
        encoding="utf-8") == "# THE REAL PROTOCOL"


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_the_archive_cannot_install_a_git_hook(notebook, checkout) -> None:
    run_restore(notebook, checkout)
    assert not (checkout / ".git" / "hooks" / "post-checkout").exists()


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_the_legitimate_member_still_arrives(notebook, checkout) -> None:
    """A fence that refuses everything is not a fix, it is a broken restore."""
    run_restore(notebook, checkout)
    assert (checkout / "artifacts" / "runs" / "good" / "manifest.json").exists()


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_dot_dot_lands_inside_artifacts_rather_than_outside(
        notebook, checkout) -> None:
    """Pins the claim the security finding got wrong.

    `artifacts/runs/../../escaped_via_middle.txt` is permitted -- the filter and
    CPython both drop the '..' -- and it lands under artifacts/runs/, which is
    where it was allowed to be. The property is that a member cannot reach
    anywhere it was not permitted, not that its name looked suspicious.
    """
    run_restore(notebook, checkout)
    assert not (checkout / "escaped_via_middle.txt").exists()
    assert not (checkout.parent / "escaped_via_middle.txt").exists()
    assert (checkout / "artifacts" / "runs" / "escaped_via_middle.txt").exists()


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_nothing_is_written_outside_the_checkout(notebook, checkout) -> None:
    run_restore(notebook, checkout)
    assert not (checkout / "escaped.txt").exists()
    assert not (checkout.parent / "escaped.txt").exists()
    assert not (checkout / "abs").exists()


def test_extractall_alone_would_have_allowed_all_of_it(tmp_path: Path) -> None:
    """The unfenced call, run once, so the fix is measured against a fact.

    If a future CPython starts refusing these members on its own, this test
    fails and the fence can be reconsidered on evidence rather than dropped on
    a guess.
    """
    work = tmp_path / "plain"
    work.mkdir()
    archive = tmp_path / "hostile.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, body in HOSTILE.items():
            zf.writestr(name, body)
    here = Path.cwd()
    os.chdir(work)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(".")
    finally:
        os.chdir(here)
    assert (work / "scripts" / "classify_completions_judge.py").exists()
    assert (work / ".git" / "hooks" / "post-checkout").exists()
    assert (work / "docs" / "preregistration_round5.md").exists()
    # ...and, equally, that it does NOT escape, which is why "path traversal"
    # is the wrong name for this.
    assert not (tmp_path / "escaped.txt").exists()
    assert (work / "escaped.txt").exists()
