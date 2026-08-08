"""Published documentation must not describe things that do not exist.

Documentation rots quietly. A script gets renamed, a flag gets dropped, a design
gets superseded -- and the page describing it keeps sitting there being believed.
This repository had exactly that: `docs/setup.md` documented a prompt-injection
gate-stack with Raspberry Pi tiers, and told the reader to run
`run_evaluation_3050.py --folds A`, a flag that script has never had. Anyone
following it got an argparse error on their first command.

These checks cover only TRACKED markdown -- the pages someone else can read.
Local working notes are exempt by design; they are scratch, not claims.

Three properties, each of which failed at least once:

  1. every repository path a doc names exists
  2. every documented command's flags exist in that script's parser
  3. every relative doc link resolves
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]

# Written as a glob in prose ("scripts/build_paper_*.py"), not a real path.
_PROSE_GLOBS = {"scripts/build_paper_"}


def _tracked_markdown() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO,
                         capture_output=True, text=True).stdout.split()
    return [REPO / p for p in out]


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


@pytest.fixture(scope="module")
def docs() -> list[Path]:
    found = _tracked_markdown()
    assert found, "no tracked markdown found -- is this a git checkout?"
    return found


def test_every_repository_path_a_doc_names_exists(docs: list[Path]) -> None:
    """A doc pointing at a deleted script sends someone hunting for it."""
    # The lookbehind keeps this to REPOSITORY paths. `cliffguard/` also names the
    # folder the notebook creates on Google Drive, so `<Drive>/cliffguard/x.json`
    # would otherwise be read as a missing package file. A repository path is
    # never preceded by a slash.
    pattern = re.compile(r"(?<![/\w])(?:scripts|notebooks|cliffguard|configs)/"
                         r"[A-Za-z0-9_./-]+\.(?:py|ipynb|yaml|yml|json|jsonl)")
    problems: list[str] = []
    for doc in docs:
        text = doc.read_text(encoding="utf-8", errors="replace")
        for match in sorted(set(pattern.findall(text))):
            if match in _PROSE_GLOBS or not (REPO / match).exists():
                if match in _PROSE_GLOBS:
                    continue
                problems.append(f"{_rel(doc)} names {match}, which does not exist")
    assert not problems, "\n".join(problems)


def test_every_documented_command_uses_flags_that_exist(docs: list[Path]) -> None:
    """A documented command with a wrong flag fails the moment it is pasted.

    The parser is the authority: each script is asked for its own --help and the
    documented flags are checked against it. Scripts that cannot even produce
    help are reported too -- that is worse than a bad flag, and it is how the
    superseded entry points were found.
    """
    command = re.compile(r"python\s+(scripts/[A-Za-z0-9_]+\.py)((?:\s+[^\n`|>]*)?)")
    problems: list[str] = []
    help_cache: dict[str, set[str] | None] = {}

    for doc in docs:
        text = doc.read_text(encoding="utf-8", errors="replace")
        for script, args in command.findall(text):
            flags = set(re.findall(r"(--[a-z][a-z0-9-]*)", args))
            if not flags:
                continue
            if not (REPO / script).exists():
                problems.append(f"{_rel(doc)} runs {script}, which does not exist")
                continue
            if script not in help_cache:
                proc = subprocess.run([sys.executable, script, "--help"],
                                      cwd=REPO, capture_output=True, text=True)
                help_cache[script] = (
                    set(re.findall(r"(--[a-z][a-z0-9-]*)", proc.stdout))
                    if proc.returncode == 0 else None)
            known = help_cache[script]
            if known is None:
                problems.append(
                    f"{_rel(doc)} documents {script}, which cannot even print "
                    "--help (it does not import)")
                continue
            unknown = sorted(flags - known)
            if unknown:
                problems.append(
                    f"{_rel(doc)} runs {script} with {unknown}, which its parser "
                    "does not accept")
    assert not problems, "\n".join(problems)


def test_every_relative_doc_link_resolves(docs: list[Path]) -> None:
    link = re.compile(r"\[[^\]]+\]\(([^)#:]+\.md)(?:#[^)]*)?\)")
    problems: list[str] = []
    for doc in docs:
        text = doc.read_text(encoding="utf-8", errors="replace")
        for target in sorted(set(link.findall(text))):
            if not (doc.parent / target).resolve().exists():
                problems.append(f"{_rel(doc)} links to {target}, which is missing")
    assert not problems, "\n".join(problems)


def test_the_superseded_design_is_not_presented_as_current(docs: list[Path]) -> None:
    """The gate-stack, the hardware tiers and the five-fold orchestrator are
    gone. They may be *mentioned* -- saying what was removed is useful -- but a
    published page must not instruct anyone to run them."""
    banned = re.compile(r"(?:scripts/dry_run\.py|scripts/run_full_evaluation\.py)"
                        r"[^\n`]*--(?:tier|config)")
    problems = [f"{_rel(d)} still instructs a superseded entry point"
                for d in docs
                if banned.search(d.read_text(encoding="utf-8", errors="replace"))]
    assert not problems, "\n".join(problems)
