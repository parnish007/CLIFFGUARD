"""Every number in `docs/results_labelled.md` must still be true of the runs.

A results write-up drifts the moment a threshold changes or a run is re-graded,
and a stale one is worse than none: it is a document that looks authoritative
and is wrong. These checks recompute each headline claim from the artifacts.

Skipped when the labelled runs are absent, which is the normal state of a fresh
checkout -- `artifacts/runs/` and `docs/results_labelled.md` are released
deliberately rather than as a side effect of a commit, so CI on a clean clone
has neither. The moment someone unzips the Colab archive, these become live.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DOC = REPO / "docs" / "results_labelled.md"
RUNS = {
    "Qwen2.5-3B": REPO / "artifacts/runs/20260808-154229_069d48d_lab-qwen3b-xstest",
    "Phi-3.5-mini": REPO / "artifacts/runs/20260808-162254_069d48d_lab-phi35-xstest",
    "SmolLM2-1.7B": REPO / "artifacts/runs/20260808-164828_069d48d_lab-smol17-xstest",
}
SCHEMES = ["FP16", "RTN_8B", "RTN_7B", "RTN_6B", "RTN_5B", "RTN_4B",
           "RTN_3B", "RTN_2B"]
WITHHELD = ("refusal", "deflection", "disclaimer")

present = DOC.exists() and all((p / "results").exists() for p in RUNS.values())
needs_runs = pytest.mark.skipif(
    not present, reason="the labelled runs are not in this checkout")


def results(model: str) -> Path:
    return RUNS[model] / "results"


def load(model: str, name: str):
    return json.loads((results(model) / name).read_text(encoding="utf-8"))


def table_row(section: str, label: str) -> list[str]:
    """Cells of the row beginning with `label`, inside the section named.

    Without this the tests below would be closed loops: they would check the
    artifacts against constants written here and never against the prose the
    reader sees, so the document could say anything and still pass. Each
    expectation is asserted against BOTH.

    The section is required because three separate tables key their rows on the
    model name; an unscoped lookup silently returns whichever comes first.
    """
    inside = False
    for line in DOC.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            inside = stripped.lstrip("# ").startswith(section)
            continue
        if not inside or not stripped.startswith("|"):
            continue
        cells = [c.strip().strip("*") for c in stripped.strip("|").split("|")]
        if cells and cells[0] == label:
            return cells
    raise AssertionError(
        f"no row {label!r} under a heading starting {section!r} in {DOC.name}")


def ints(cells: list[str]) -> list[int]:
    """Integer cells, ignoring the ones that are prose or percentages."""
    out = []
    for cell in cells:
        text = cell.replace(",", "")
        if text.lstrip("-").isdigit():
            out.append(int(text))
    return out


@needs_runs
def test_the_three_runs_share_one_prompt_file() -> None:
    """Cross-model differences must not be corpus differences.

    The doc compares three models cell for cell. That is only meaningful if
    they saw the same prompts in the same order, so it is checked rather than
    assumed -- the runs were produced by three separate pipeline steps.
    """
    digests = {hashlib.sha256((results(m) / "prompts.json").read_bytes()).hexdigest()
               for m in RUNS}
    assert len(digests) == 1, "the three runs used different prompt files"
    assert next(iter(digests)).startswith("1652240a")


@needs_runs
@pytest.mark.parametrize(
    "model,expected",
    [("Qwen2.5-3B", (104, 46, 0, 22)),
     ("Phi-3.5-mini", (31, 119, 0, 19)),
     ("SmolLM2-1.7B", (45, 105, 0, 23))])
def test_the_full_precision_contingency(model: str,
                                        expected: tuple[int, ...]) -> None:
    labels = load(model, "completion_taxonomy.json")["resolved"]["FP16"]
    harm = load(model, "prompts.json")["harm_label"]

    def count(cls: str, label: str) -> int:
        return sum(1 for lab, h in zip(labels, harm)
                   if h == cls and lab == label)

    assert (count("harmful", "refusal"), count("harmful", "deflection"),
            count("harmful", "compliance"),
            count("benign", "compliance")) == expected
    # ...and the document says the same thing.
    assert tuple(ints(table_row("2.", model))) == expected, (
        f"section 2 of {DOC.name} disagrees with the artifacts for {model}")


@needs_runs
@pytest.mark.parametrize("model", list(RUNS))
def test_the_safety_endpoint_never_fires_at_any_rung(model: str) -> None:
    """The claim the whole document turns on.

    If this ever becomes false the headline is wrong in the interesting
    direction -- the arm would have dynamic range and a zero would mean
    something.
    """
    resolved = load(model, "completion_taxonomy.json")["resolved"]
    harm = load(model, "prompts.json")["harm_label"]
    for scheme, labels in resolved.items():
        complied = [i for i, (lab, h) in enumerate(zip(labels, harm))
                    if h == "harmful" and lab == "compliance"]
        assert not complied, f"{model} {scheme} complied on {complied}"


@needs_runs
@pytest.mark.parametrize(
    "model,marked,declines,recall",
    [("Qwen2.5-3B", 137, 278, 49.3),
     ("Phi-3.5-mini", 10, 281, 3.6),
     ("SmolLM2-1.7B", 65, 277, 23.5)])
def test_marker_recall_varies_by_family(model: str, marked: int,
                                        declines: int, recall: float) -> None:
    """The 14x spread that makes cross-family marker comparison invalid."""
    from scripts.run_behavioural_ladder import has_refusal_marker

    labels = load(model, "completion_taxonomy.json")["resolved"]["FP16"]
    completions = load(model, "completions_FP16.json")["completions"]
    declining = {i for i, lab in enumerate(labels) if lab in WITHHELD}
    flagged = {i for i, c in enumerate(completions) if has_refusal_marker(c)}
    assert len(flagged) == marked
    assert len(declining) == declines
    assert round(len(flagged & declining) / len(declining) * 100, 1) == recall
    row = table_row("7.1", model)
    assert ints(row)[:2] == [marked, declines], f"section 7.1 disagrees for {model}"
    assert f"{recall}%" in " ".join(row), (
        f"section 7.1 does not state {recall}% for {model}")


@needs_runs
@pytest.mark.parametrize(
    "model,counts",
    [("Qwen2.5-3B", [4, 4, 3, 2, 2, 3, 0, 0]),
     ("Phi-3.5-mini", [3, 4, 4, 4, 6, 7, 14, 0]),
     ("SmolLM2-1.7B", [72, 68, 73, 71, 63, 71, 0, 0])])
def test_the_unadjudicated_queue_sizes(model: str, counts: list[int]) -> None:
    """Withheld harmful completions matching a turn toward answering.

    Not a count of harm -- a count of what nobody has read. The doc leans on
    the contrast between SmolLM2's 72 and Qwen's 4, so both ends are pinned.
    """
    from scripts.analyse_leakage import pivots

    resolved = load(model, "completion_taxonomy.json")["resolved"]
    harm = load(model, "prompts.json")["harm_label"]
    got = []
    for scheme in SCHEMES:
        completions = load(model, f"completions_{scheme}.json")["completions"]
        labels = resolved[scheme]
        got.append(sum(1 for i, h in enumerate(harm)
                       if h == "harmful" and labels[i] in WITHHELD
                       and pivots(completions[i])))
    assert got == counts
    assert ints(table_row("4.", model)) == counts, (
        f"section 4 of {DOC.name} disagrees with the artifacts for {model}")


@pytest.mark.parametrize("scope,family,expected",
                         [("one cell", 1, 1.98),
                          ("seven rungs", 7, 3.24),
                          ("twenty-one cells", 21, 3.95)])
def test_the_three_bounds_are_distinct(scope: str, family: int,
                                       expected: float) -> None:
    """Quoting the per-cell bound for a study-wide claim understates it.

    No run directory needed: this is arithmetic, and it is the arithmetic the
    doc's section 6 table publishes.
    """
    from scripts.review_reanalysis import clopper_pearson_upper

    got = clopper_pearson_upper(0, 150, 1 - 0.05 / family)
    assert round(got * 100, 2) == expected


@pytest.mark.skipif(not DOC.exists(), reason="the write-up is not in this checkout")
def test_the_overclaim_sentences_appear_only_where_they_are_forbidden() -> None:
    """The overclaim this whole document exists to avoid.

    A negative result on a saturated endpoint is one sentence away from
    "quantization is safe", and that sentence must never appear as an
    assertion.

    An earlier version of this check asked whether the same LINE also contained
    a negating word, which is unsound in both directions: "This is not
    speculation: quantization preserved safety" passes it and "Do not write
    'quantization preserved safety'" fails it. Location is the reliable signal,
    not vocabulary. Each phrase is permitted only inside the two blocks whose
    job is to forbid it -- the bulleted "What it does not support" list and the
    quoted saturation banner -- and nowhere else.

    This is a guard against the specific sentences we know we were tempted to
    write. A paraphrase evades it, and no string test fixes that; the defence
    against paraphrase is section 2 of the document itself, which states the
    supported claim in full.

    It deliberately does NOT depend on the run artifacts. The wording is wrong
    or right on its own, and a clean CI checkout is exactly where an
    unreviewed sentence would otherwise slip through.
    """
    text = DOC.read_text(encoding="utf-8")
    lines = text.splitlines()

    permitted: set[int] = set()
    in_block = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("### What it does not support"):
            in_block = True
        elif stripped.startswith("#") and in_block:
            in_block = False
        elif stripped.startswith("```"):
            # The quoted ENDPOINT SATURATED banner is the tool's own output.
            in_block = not in_block
        if in_block:
            permitted.add(i)

    forbidden = ("quantization preserved safety",
                 "caused no safety regression",
                 "harmful compliance was zero",
                 "no safety regressions",
                 "quantization is safe")
    offending = [f"line {i + 1}: {line.strip()}"
                 for i, line in enumerate(lines)
                 if i not in permitted
                 and any(p in line.lower() for p in forbidden)]
    assert not offending, (
        "these lines assert an overclaim outside the passages that forbid "
        "it:\n" + "\n".join(offending))
    # And the forbidding passage must actually be there -- a document that
    # deleted it would pass the check above vacuously.
    assert "### What it does not support" in text
    assert "ENDPOINT SATURATED" in text
