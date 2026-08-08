"""Every generated figure has a home, and every referenced one exists.

Two failures are easy to create while editing and neither breaks a build that
happens to have a stale PDF on disk:

  * a figure is generated and then used nowhere, so a reader of the repository
    cannot tell whether it was dropped deliberately or forgotten;
  * a figure is renamed and the page that embedded it silently 404s. This
    happened: renaming `labelled_*` to `fig_labelled_*` to match the
    manuscript's convention broke all three image links in the local write-up
    at once, and nothing complained.

"Used" means used by EITHER surface -- the manuscript or the local results
write-up. Some figures belong only to the write-up on purpose; that is a
decision, and this test records it rather than second-guessing it.

Skipped wholesale on a clean checkout, where `docs/paper/` is absent by design.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "docs" / "paper"
TEX = PAPER / "cliff_artifact.tex"
WRITEUP = REPO / "docs" / "results_labelled.md"

needs_paper = pytest.mark.skipif(
    not TEX.exists(), reason="the manuscript is not in this checkout")


def live_tex() -> str:
    """Source with comments stripped, so a commented-out include is not 'used'."""
    return re.sub(r"(?<!\\)%[^\n]*", "", TEX.read_text(encoding="utf-8"))


def included_figures() -> set[str]:
    return set(re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([A-Za-z0-9_]+)\.pdf\}",
                          live_tex()))


def embedded_figures() -> set[str]:
    """Figure stems the local write-up embeds as PNG."""
    if not WRITEUP.exists():
        return set()
    return set(re.findall(r"\]\(paper/figures/([A-Za-z0-9_]+)\.png\)",
                          WRITEUP.read_text(encoding="utf-8")))


@needs_paper
def test_every_figure_the_paper_includes_exists() -> None:
    missing = sorted(f for f in included_figures()
                     if not (PAPER / "figures" / f"{f}.pdf").exists())
    assert not missing, f"\\includegraphics of missing figures: {missing}"


@needs_paper
def test_every_table_the_paper_inputs_exists() -> None:
    used = set(re.findall(r"\\input\{tables/([A-Za-z0-9_]+)\}", live_tex()))
    missing = sorted(t for t in used
                     if not (PAPER / "tables" / f"{t}.tex").exists())
    assert not missing, f"\\input of missing tables: {missing}"


@needs_paper
def test_no_generated_figure_is_orphaned() -> None:
    """A figure used by neither surface is either dead or forgotten."""
    on_disk = {p.stem for p in (PAPER / "figures").glob("*.pdf")}
    used = included_figures() | embedded_figures()
    orphaned = sorted(on_disk - used)
    assert not orphaned, (
        "generated but used by neither the manuscript nor the write-up: "
        f"{orphaned}. Either cite it or stop generating it.")


@needs_paper
def test_every_write_up_image_resolves() -> None:
    """The check that renaming figures broke."""
    if not WRITEUP.exists():
        pytest.skip("the write-up is not in this checkout")
    broken = [m for m in re.findall(r"\]\(([^)]+\.png)\)",
                                    WRITEUP.read_text(encoding="utf-8"))
              if not (WRITEUP.parent / m).exists()]
    assert not broken, f"the write-up embeds missing images: {broken}"


@needs_paper
def test_no_latex_control_word_lost_its_backslash() -> None:
    r"""Catch `\ref` mangled into `ef` by a shell heredoc.

    Writing LaTeX through a bash heredoc interprets `\r`, `\n`, `\t` as escape
    characters, so `\ref{x}` arrives as CR + `ef{x}` and `\newline` as LF +
    `ewline`. It happened twice in one editing session. LaTeX then compiles
    without a single warning, because `Table~ef{tab:labelled}` is perfectly
    valid input -- it simply typesets the word "ef" and no reference at all,
    and the defect is visible only by reading the rendered PDF.

    Word-matching the mangled forms was the wrong approach: the first version
    of this test listed `\ref` and `\newline`, then missed `$\times$` becoming
    `$<TAB>imes$` because "imes" was not on the list. The residue of an eaten
    escape is always the CONTROL CHARACTER it produced, so that is what to look
    for -- a stray TAB, or a CR outside a CRLF pair, is essentially never
    legitimate in this manuscript and always means an escape was interpreted.

    Checked against the source bytes rather than the log, because the log says
    nothing.
    """
    raw = TEX.read_bytes()
    problems: list[str] = []
    for i, byte in enumerate(raw):
        if byte == 0x09:
            what = r"TAB, probably an eaten \t (\times, \textbf, \tabular)"
        elif byte == 0x0D and raw[i + 1:i + 2] != b"\n":
            what = r"lone CR, probably an eaten \r (\ref, \right, \rule)"
        else:
            continue
        line_no = raw[:i].count(b"\n") + 1
        context = raw[max(0, i - 40):i + 30].decode("utf-8", "replace")
        problems.append(f"line {line_no}: {what}\n    ...{context}...")
    assert not problems, (
        "control characters in the LaTeX source, which is what a shell heredoc "
        "leaves behind when it eats a backslash:\n" + "\n".join(problems))


@needs_paper
def test_no_known_mangled_control_word_survives() -> None:
    """Belt and braces: the specific mangled spellings, in case one is typed.

    The byte-level check above catches escapes eaten at write time. This catches
    the same words arriving some other way -- a copy-paste from a mangled file,
    say -- where the control character has already been normalised to a space.
    """
    text = live_tex()
    mangled = {
        "ef{": r"\ref",
        "ewline": r"\newline",
        "extbf{": r"\textbf",
        "extit{": r"\textit",
        "imes$": r"\times",
        "abular": r"\tabular",
    }
    problems: list[str] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for bad, intended in mangled.items():
            for hit in re.finditer(re.escape(bad), line):
                start = hit.start()
                # A legitimate occurrence is preceded by the letter the escape
                # ate; a mangled one follows whitespace, a tie, or nothing.
                before = line[start - 1] if start else ""
                if before.isalpha():
                    continue
                problems.append(
                    f"line {line_no}: {line.strip()[:70]!r} looks like "
                    f"{intended} with its backslash eaten")
    assert not problems, "\n".join(problems)


@needs_paper
def test_the_paper_never_claims_the_graders_were_unanimous() -> None:
    """One of four grader comparisons runs the other way.

    Three separate passages once said the shift was "consistent in sign under
    every grader we tried" while the grader section said three of four agreed.
    Llama-3.3-70B on Qwen2.5-3B gives 1 transition toward refusal against 2
    toward compliance. A claim of unanimity states no number, so the numeric
    consistency check in `check_paper_numbers.py` cannot see it; this catches
    the phrasing instead.
    """
    text = re.sub(r"\s+", " ", live_tex())
    banned = [
        r"under every grader",
        r"every grader we tried",
        r"all (?:four|4) graders agree",
        r"unanimous",
        r"replicates in direction across (?:four|4) graders",
    ]
    hits = [p for p in banned if re.search(p, text, re.IGNORECASE)]
    assert not hits, (
        "the paper claims grader unanimity, which its own agreement section "
        f"contradicts: {hits}")


@needs_paper
def test_the_paper_does_not_plot_the_number_it_promises_not_to_plot() -> None:
    """The in-sample pivot count stays out of the manuscript's figures.

    The paper says of that count: "we therefore do not carry it into any table,
    figure or bound in this manuscript". It is a rate estimated in-sample from a
    regular expression fitted in-sample on the same completions, which is the
    kind of quantity the paper spends its length objecting to -- so plotting it
    would not merely break a promise, it would commit the error being argued
    against.

    The figure exists, and is embedded in the local write-up where it is
    labelled a queue for adjudication. That is why the orphan check above
    passes and why this test is needed: nothing else distinguishes "used
    deliberately elsewhere" from "about to be pasted into the paper".
    """
    assert "fig_labelled_unadjudicated" not in live_tex(), (
        "the manuscript includes the pattern-flagged withheld figure, but it "
        "states that this count is not carried into any figure. Either remove "
        "the figure or remove the claim -- and the claim is the load-bearing "
        "one.")


@needs_paper
def test_every_cross_reference_resolves() -> None:
    text = live_tex()
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", text))
    assert not sorted(refs - labels), f"dangling \\ref: {sorted(refs - labels)}"


@needs_paper
def test_the_labelled_section_floats_are_all_called_out() -> None:
    """A float nobody points at is a float the reader meets without context.

    Scoped to the round-3 additions rather than the whole paper: the earlier
    sections place some figures without an explicit callout, which is a style
    choice made before this test existed and not one to churn now.
    """
    text = live_tex()
    ours = {label for label in re.findall(r"\\label\{(fig:labelled[^}]*)\}", text)}
    refs = set(re.findall(r"\\ref\{([^}]+)\}", text))
    assert ours, "no round-3 figures found -- has the section been removed?"
    assert not sorted(ours - refs), (
        f"round-3 figures never referenced in the text: {sorted(ours - refs)}")
