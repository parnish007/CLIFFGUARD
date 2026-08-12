"""Assemble the reframed manuscript from the verified one, without retyping it.

`docs/paper/cliff_artifact.tex` is the research record: complete, internally
consistent, and every load-bearing number in it cross-checked against the JSON
that produced it. Its problem is not its evidence, it is its hierarchy --- five
reviewers agreed the paper needs a new ordering of claims and not a new
evidentiary object.

So this builds the reframed paper by MOVING blocks of the record rather than
rewriting them. Section bodies are extracted by line range and re-levelled
(a \\section that becomes a subsection of the new Part II, a subsection promoted
to an appendix); only genuinely new connective text --- title, abstract,
introduction, the tier statement, the part openers, the conclusion --- is
written here. Nothing is deleted: everything the main line sheds moves to an
appendix.

That choice is the whole point. Retyping ~120 load-bearing quantities into a
fresh document is exactly the operation that detaches a claim from the file that
justifies it, and this paper's credibility rests on that attachment. Extracting
byte ranges cannot introduce a transcription error, and if the record changes
the reframed paper regenerates.

`cliff_artifact.tex` is opened read-only and never written.

Usage:
  python scripts/build_main_paper.py
  python scripts/build_main_paper.py --check   # verify block map, write nothing
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

SOURCE = Path("docs/paper/cliff_artifact.tex")
TARGET = Path("docs/paper/cliffguardpaper.tex")

# Blocks of the record, by the line the sectioning command sits on. `end` is
# exclusive. Verified against the source at build time: if a heading has moved,
# the build fails rather than silently shipping the wrong span.
#
#   name          (start, end, anchor_text_that_must_appear_at_start)
BLOCKS: dict[str, tuple[int, int, str]] = {
    "protocol":    (378, 560, r"\subsection{The CliffGuard protocol}"),
    "related":     (560, 661, r"\section{Related work}"),
    "method":      (661, 751, r"\section{Method}"),
    "degeneracy":  (751, 878, r"\subsection{Degeneracy must be detected"),
    "scorers":     (878, 941, r"\subsection{Two label scorers"),
    "refusal":     (941, 1362, r"\section{Quantization shifts the refusal"),
    "law":         (1362, 1461, r"\subsection{Three regimes"),
    "artifact":    (1461, 1548, r"\section{Where the reported cliff comes from}"),
    "monotone":    (1548, 1762, r"\subsection{The phrase-list estimate is not monotone"),
    "labelled":    (1762, 2187, r"\subsection{A third family"),
    "scorer":      (2201, 2314, r"\subsection{The label scorer was not comparing"),
    "round4":      (2314, 2499, r"\subsection{The whole ladder, graded twice"),
    "budget":      (2499, 2586, r"\subsection{The generation budget"),
    "regen":       (2586, 2632, r"\subsection{Greedy decoding did not reproduce"),
    "standing":    (2632, 2675, r"\subsection{What the three tests leave standing}"),
    "human":       (2675, 2888, r"\section{An author-blinded annotation audit"),
    "capability":  (2888, 2963, r"\section{Arithmetic accuracy degrades"),
    "probe":       (2963, 3147, r"\section{One frozen refusal direction"),
    "discussion":  (3147, 3183, r"\section{Discussion}"),
    "prediction":  (3183, 3293, r"\subsection{What this implies for choosing"),
    "review":      (3293, 3398, r"\section{What external review raised"),
    "limits":      (3398, 3626, r"\section{Limitations}"),
    "conclusion":  (3626, 3753, r"\section{Conclusion}"),
    # Stops at 3800, which is the record's own \appendix. Including it would
    # reset the appendix counter halfway through this paper's appendices and
    # produce two of them lettered A.
    "repro":       (3753, 3800, r"\section{Reproducibility}"),
    # Stops before the bibliography, which the assembler appends itself.
    "design":      (3801, 3969, r"\section{Appendix: the follow-up design"),
}

# The two lines that close the document, appended once after the last block.
TAIL = "\n\\bibliography{refs_verified}\n\\end{document}\n"

PREAMBLE_END = 125     # first line of \title
DOC_START = 145        # \begin{document}
FRONT_MATTER = (149, 170)   # repo disclosure block, reused verbatim


def load() -> list[str]:
    if not SOURCE.is_file():
        raise SystemExit(
            f"{SOURCE} not found. The manuscript is gitignored, so this runs "
            "only in a checkout that has it.")
    return SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)


def block(lines: list[str], name: str, level: str | None = None,
          retitle: str | None = None) -> str:
    """Extract a block, optionally re-levelling and retitling its heading.

    Re-levelling is what makes the move safe: the same body serves as a section
    in the record and a subsection here, and `\\label` travels with it so every
    `\\ref` in the moved text still resolves.
    """
    start, end, anchor = BLOCKS[name]
    body = lines[start - 1:end - 1]
    head = body[0]
    if not head.lstrip().startswith(anchor.split("{")[0]) or anchor.split("{", 1)[1].rstrip("}") not in "".join(body[:3]):
        raise SystemExit(
            f"block {name!r} no longer starts with {anchor!r} at line {start}; "
            f"found {head.strip()[:70]!r}. The record moved -- fix BLOCKS.")
    if level or retitle:
        # Several headings in the record wrap across two source lines, so the
        # heading is however many lines it takes for its braces to balance --
        # not the first line. Getting this wrong leaves the tail of a title
        # stranded in the body as a stray closing brace, which TeX reports far
        # from where it happened.
        consumed, depth, started = 0, 0, False
        for i, line in enumerate(body):
            depth += line.count("{") - line.count("}")
            started = started or "{" in line
            consumed = i + 1
            if started and depth == 0:
                break
        else:
            raise SystemExit(f"unbalanced heading braces in block {name!r}")
        heading = "".join(body[:consumed])
        m = re.match(r"^\\((?:sub)*section)\*?\{(.*)\}\s*$", heading, re.S)
        if not m:
            raise SystemExit(
                f"cannot parse heading of {name!r}: {heading.strip()[:90]!r}")
        # Default to the level the block already has. Defaulting to `section`
        # instead silently promoted every retitled subsection to a top-level
        # section, which is how Decisions 3 and 4 escaped Part II's opening
        # section and became siblings of it.
        title = retitle if retitle else " ".join(m.group(2).split())
        body = [f"\\{level or m.group(1)}{{{title}}}\n"] + body[consumed:]
    return "".join(body)


NEW_ABSTRACT = r"""
\begin{abstract}
\noindent
Post-training quantization is widely reported to erode refusal behaviour. We ask
a prior question: \textbf{when an evaluation reports that quantization changed
refusal, how much of the reported effect belongs to the model and how much to
the pipeline that measured it?}

We introduce \textbf{CliffGuard}, a paired, degeneration-aware protocol that
holds the model, the prompts and the generated text fixed and varies one
measurement decision at a time. Because every completion is stored, the same
text can be re-scored under alternative gates, estimators and graders, so the
pipeline's contribution is measured rather than assumed. We apply it to a ladder
from 8.5 down to 2.5 stored bits per parameter on four small instruction-tuned
models.

Four measurement decisions move the answer, each on identical stored text. The
\emph{degeneration gate}: at 2.5 bits on Qwen2.5-3B a refusal-phrase list reads
38.4\% refusal-to-compliance behind a perplexity-only gate and 0.2\% behind a
composite one. The \emph{estimator}: at 4.5 bits, where nothing is degenerate,
that same list reads 7.7 times the semantic judge. The judge's \emph{label
extraction}: reading a label from first-token logits over words that do not
tokenize alike moves 11--18\% of verdicts, halves the pooled drift slope from
1.15 to 0.51 points per bit, and reverses its sign on Phi-3.5-mini. And the
\emph{option assignment}: permuting which letter carries which class, with the
wording untouched, moves absolute refusal rates by up to 10.4 points while
moving the paired quantization difference by 1.6. That asymmetry is the paper's
methodological result --- \textbf{an absolute rate from such a judge is not
invariant to a choice no protocol records; a paired difference largely is} ---
and it is why every claim here is a paired one.

Against 300 hand-labelled completions, weighted to the ladder's strata, the
phrase list agrees with a person's reading of the declining class on 57.5\% of
completions against 80.2 and 79.6\% for two judge extractions; the two
extractions are not separated from each other. What survives of the behavioural
question is narrow: one model at one rung, which three independent graders do
not reproduce at significance.

The sole annotator is an author, so \S\ref{sec:human} is an audit rather than an
independent validation; the compliance class is not a harm measurement; and one
quantizer family on small models is the scope.
\end{abstract}
"""

NEW_INTRO = r"""
\section{Introduction}
\label{sec:intro}

A growing literature reports that compressing a language model damages its
refusal behaviour. The claim is consequential --- quantized checkpoints are what
most people actually run --- and it is usually established the same way: generate
completions before and after compression, decide which ones are refusals, and
compare the rates.

Every step after ``generate'' is an instrument, and this paper is about what
those instruments contribute to the answer. The question we ask is not whether
quantization is safe. It is:

\begin{keybox}
\textbf{When an evaluation reports that quantization changed refusal behaviour,
how much of the reported effect is a property of the model, and how much is a
property of the pipeline that measured it?}
\end{keybox}

\noindent That question is answerable, and answerable cheaply, because the
measurement can be separated from the generation. If completions are stored,
the same text can be scored again under a different gate, a different estimator,
a different grader, or a different label representation, with the model, the
prompts, the decoding and the text all held fixed. Whatever the estimate moves
by is the pipeline's contribution, and no grader has to be correct for that
subtraction to be valid.

\paragraph{What we find.} The pipeline's contribution is large --- large enough
that, on our own data, it decides the sign of the headline effect. We report it
as four decisions, in descending order of how much they move the answer, and we
found the last of them by looking for it after the first three had already
forced us to withdraw a result of our own.

\paragraph{How to read this paper.} The findings below are stated in three
tiers, and the tiering is a claim about evidence rather than about interest.
\textbf{Tier 1} is what this design establishes well: controlled perturbations
of a measurement pipeline applied to identical stored text, where the
counterfactual is exact and no grader needs to be trusted. \textbf{Tier 2} is
validation and stress-testing --- human labels, external graders, generation
budget, regeneration --- which support the Tier 1 claims and are individually
weaker. \textbf{Tier 3} is what this paper measures about quantized models
themselves, which is the part we would defend least: two models carry the
refusal arm, one grader family produced the labels, and independent graders do
not reproduce the significance. Readers who want only the defensible
contribution can stop after Part~II.
"""

NEW_FINDINGS = r"""
\begin{keybox}
\textbf{Findings, by tier.} A dagger marks a quantity still carried by the
original label scorer (\S\ref{sec:scorers}).

\medskip
\textbf{Tier 1 --- the measurement result.} Controlled perturbations on
identical stored completions. Neither grader needs to be right for these.

\begin{enumerate}[leftmargin=1.4em]
  \item \textbf{The degeneration gate moves the estimate by two orders of
        magnitude.} At 2.5 bits on Qwen2.5-3B one refusal-phrase list reads
        38.4\% refusal-to-compliance behind a perplexity-only gate and 0.2\%
        behind our composite gate; the judge behind the composite gate reads
        0.0\%. The first step is the gate, the second the grader
        (\S\ref{sec:artifact}).
  \item \textbf{The estimator moves it again, where nothing is degenerate.} At
        4.5 bits both gates admit every completion, so only the grader term is
        left --- and the phrase list reads 7.7 times the judge. One fixed list
        covers 51.3\%, 25.7\% and 3.6\% of a five-way judge's declining class
        across three model families at full precision, at precision $1.000$ in
        every case, so the label sets nest and the disagreement runs one way
        only. The estimate is \emph{non-monotone} in the phrase list, so
        enlarging that list is not guaranteed to improve it, or even to move it
        in a consistent direction (\S\ref{sec:artifact}).
  \item \textbf{How a label is read off the judge moves 11--18\% of verdicts.}
        Both our graders read a label from first-token logits over label words
        that do not tokenize alike under Qwen2.5. Re-scoring identical stored
        text with verified single-token options halves the pooled drift slope,
        from $1.15$ points per bit removed, 95\% CI $[0.75, 1.55]$, to $0.51$,
        $[0.07, 0.96]$ --- and the halving is a disagreement rather than an
        attenuation: Qwen2.5-3B holds at $+1.16$ while Phi-3.5-mini
        \emph{reverses sign}, $+1.29$ to $-0.13$ with an interval covering zero
        (\S\ref{sec:scorer}).
  \item \textbf{Which letter carries which class moves absolute rates by ten
        points and the paired contrast by one.} With the option wording
        untouched, full-precision refusal travels across 10.4 points on
        Qwen2.5-3B and 8.8 on Phi-3.5-mini, while the paired
        FP16-to-4.5-bit difference travels across 1.6 and 1.2 points, reverses
        direction under none of the assignments tested, and leaves the
        significance decision unchanged at all three. \textbf{An absolute rate
        from this instrument is not invariant to a choice no protocol records;
        a paired difference largely is} (\S\ref{sec:round4}).
\end{enumerate}

\medskip
\textbf{Tier 2 --- what validates and stresses those claims.}

\begin{enumerate}[leftmargin=1.4em, resume]
  \item \textbf{Against 300 blinded human labels, the phrase list is measuring a
        narrower construct.} Weighted to the ladder's strata it agrees with a
        person's reading of the declining class on 57.5\% of completions against
        80.2 and 79.6\% for the two judge extractions --- $+22.7$
        $[+14.9, +30.1]$ and $+22.1$ $[+13.7, +30.2]$ points under a stratified
        paired bootstrap. The gap is structured: the list recovers 81.1\% of
        genuine refusals and 14.6\% of deflections. The two judges are not
        separated on accuracy, $+0.6$ $[-2.6, +3.6]$, with equivalence at
        $\pm3$ points not established either (\S\ref{sec:human}). One
        annotator, who is an author.
  \item \textbf{The declining class is mostly not refusal.} Weighted, it is
        $35.9$\% refusal, $38.0$\% deflection and $26.1$\% capability
        disclaimer, so a ``refusal rate'' read off it is about two-thirds
        something else (\S\ref{sec:human}).
  \item \textbf{The generation budget was load-bearing and is now measured.}
        Between 92.7 and 100\% of full-precision completions reach the 48-token
        cap. At 256 tokens, paired on one act of decoding, refusals do become
        something else --- but they become deflections almost always, 47 of the
        48 that move, and the harmful-compliance cell stays empty on two of
        three models (\S\ref{sec:budget}).
  \item \textbf{Greedy decoding does not reproduce, and it barely matters.}
        Regenerating the same completions changes 9--12\% of them; verdicts move
        on 0.2--1.0\% (\S\ref{sec:reproducibility}).
\end{enumerate}

\medskip
\textbf{Tier 3 --- what this says about quantized models, which is least.}

\begin{enumerate}[leftmargin=1.4em, resume]
  \item \textbf{Inside the coherent band, lower precision produced \emph{more}
        declining --- under the original scorer on both models, and under the
        second scorer on one of two.} What is left is one model at one rung:
        Qwen2.5-3B at 4.5 bits, 7 transitions toward compliance against 32
        toward refusal, $p<0.0001$. \textbf{Three independent graders reproduce
        the direction on three of four comparisons and the significance on
        none} (\S\ref{sec:ladder}).
  \item \textbf{Transitions stay low, but how low is a property of the scorer.}
        At or below 2.2\% at every rung under the original scorer, simultaneous
        95\% bound 4.62\% over 14 cells; re-graded, 4.6\% and 7.72\%. A
        deployment bound read off this instrument inherits that factor of two
        (\S\ref{sec:ladder}).
  \item \textbf{Arithmetic capability collapses at a model-specific bit-width}
        one full bit apart across families, and a frozen refusal probe does not
        notice (Appendices~\ref{sec:capability} and~\ref{sec:probe}).
\end{enumerate}
\end{keybox}
"""

PART_II = r"""
\part*{Part II --- How much of the effect is the pipeline?}
\addcontentsline{toc}{part}{Part II --- How much of the effect is the pipeline?}

\section{Four measurement decisions, on identical stored text}
\label{sec:pipeline}
\label{sec:remeasure}

This is the part of the paper we would defend. Each subsection below changes one
decision in the measurement pipeline and nothing else --- same prompts, same
model, same quantization, same generated text --- and reports what the estimate
does. The counterfactual is exact, and the argument needs neither instrument to
be correct: it establishes only that two defensible pipelines disagree, and by
how much.

The four are ordered by how much they move the answer.
"""

PART_III = r"""
\part*{Part III --- What the instruments are worth}
\addcontentsline{toc}{part}{Part III --- What the instruments are worth}

\section{Human labels and independent graders}
\label{sec:validation}

Part~II establishes that the instruments disagree. It cannot say which is
closer to a person's reading, because every comparison in it is between
automatic instruments. This part puts them in front of a person, and in front
of graders from other model families.
"""

PART_IV = r"""
\part*{Part IV --- The ladder itself}
\addcontentsline{toc}{part}{Part IV --- The ladder itself}

\section{The quantization ladder, as a case study}
\label{sec:ladder}

What follows is the behavioural measurement the pipeline of Part~II was built
to make, and it is the part of this paper with the weakest support. It is
reported here rather than first because Part~II has already established that
the numbers in it are instrument-dependent, and because one of the two headline
results it originally carried did not survive that establishment.

Two of the four models carry this arm. The tables below are the original
scorer's\os{} unless marked; \S\ref{sec:scorer} gives the second scorer's
reading of the same completions, and \S\ref{sec:limits} states which of the two
we would quote, which is neither.
"""

NEW_CONCLUSION = r"""
\section{Conclusion}
\label{sec:conclusion}

\paragraph{The position we would defend.} Not that quantization is safe, and not
that it makes models more conservative. That \textbf{the size of a reported
quantization--safety effect can be dominated by choices in the pipeline that
measures it} --- which gate, which estimator, which grader, which label strings,
which letter carries which class, which generation budget --- and that a paper
reporting such an effect without pinning those choices has not yet distinguished
the model from the instrument. We include our own paper in that, which is why
Part~IV is where it is.

\paragraph{The one methodological result we would carry forward.} An absolute
rate from a first-token multiple-choice judge is not invariant to a choice no
protocol records: permuting which letter carries which class moves it by ten
points. A paired difference computed from the same judge on the same
completions moves by one. If that asymmetry holds beyond the two models tested
here, it says something practical about how compression, fine-tuning and
alignment evaluations should be reported --- paired, on stored text, with the
label representation pinned --- and it says it without requiring anyone to have
solved the problem of building a correct refusal grader.

\paragraph{What no result here establishes.} That any of our graders is right in
absolute terms; the human audit ranks two of them against a third and is an
author's own labelling. That any harm occurred anywhere: every safety-shaped
quantity here is a harmful prompt crossed with a compliance-\emph{shaped}
response, and an empty cell means no substantive compliance was labelled, not
that no harm was done. And that any of it generalises past round-to-nearest on
small models --- one quantizer family, four models of which two carry the
refusal arm, English, one arithmetic benchmark, greedy decoding, one grader
family sharing a lineage with two of the models judged.

\paragraph{What would change the picture, in order of cost.} Independent
annotation by people who are not authors, which is the cheapest and would
convert \S\ref{sec:human} from an audit into a validation. A blinded sample
drawn from below 4 bits, where the degeneration gate actually fires and where
this paper has never tested it against a person. A deployed quantizer and a
larger target, since AWQ and GPTQ are what the reports we are arguing with
actually used. And one confirmatory run under a protocol frozen in advance,
because the design here was adjusted while looking at results and no amount of
disclosure about that changes what it is.

Appendix~\ref{sec:review} records what external review raised against earlier
versions of this manuscript and what happened to each objection, including the
ones that killed results we had reported.
"""


def build() -> str:
    lines = load()
    out: list[str] = []
    add = out.append

    add("".join(lines[:PREAMBLE_END - 1]))
    add(r"""\title{\bfseries CliffGuard: Auditing Measurement Dependence in\\
       Quantization--Refusal Evaluation\\[2mm]
       \bfseries\large How Much of a Reported Safety Cliff Is the Gate,
       the Estimator,\\
       \bfseries\large the Grader, and the Option Order?}
""")
    add("".join(lines[130:DOC_START - 1]))          # author, date, fuzz
    add("".join(lines[DOC_START - 1:FRONT_MATTER[1] - 1]))
    add(NEW_ABSTRACT)
    # The record had no contents page and did not need one: it was a flat list
    # of sections read front to back. This version is four parts plus seven
    # appendices, and the whole point of the reordering is that a reader can
    # stop after Part II -- which they can only do if they can see that Part II
    # ends somewhere.
    add("\n\\clearpage\n{\\small\\tableofcontents}\n\\clearpage\n")
    add(NEW_INTRO)
    add(NEW_FINDINGS)

    add("\n\\part*{Part I --- Background and protocol}\n"
        "\\addcontentsline{toc}{part}{Part I --- Background and protocol}\n")
    add(block(lines, "related"))
    add(block(lines, "method", retitle="The CliffGuard measurement framework"))
    add(block(lines, "protocol"))
    add(block(lines, "degeneracy"))
    add(block(lines, "scorers"))

    # Labels are NOT renamed. The moved bodies carry hundreds of \ref calls
    # keyed on the record's own names, and a label never reaches the page, so
    # renaming would cost every cross-reference and buy nothing.
    add(PART_II)
    add(block(lines, "artifact", level="subsection",
              retitle="Decisions 1 and 2: the gate, then the estimator"))
    add(block(lines, "monotone", level="subsubsection"))
    add(block(lines, "scorer",
              retitle="Decision 3: how the label is read off the judge"))
    add(block(lines, "round4",
              retitle="Decision 4: which letter carries which class"))

    add(PART_III)
    add(block(lines, "human", level="subsection",
              retitle="An author-blinded annotation audit"))
    add(block(lines, "labelled"))

    add("\n\\section{Robustness of the measurement}\n\\label{sec:robust}\n")
    add(block(lines, "budget"))
    add(block(lines, "regen"))
    add(block(lines, "standing"))

    add(PART_IV)
    add(block(lines, "refusal", level="subsection",
              retitle="The paired ladder, under both scorers"))
    add(block(lines, "discussion"))
    add(block(lines, "limits"))
    add(NEW_CONCLUSION)

    add("\n\\appendix\n"
        "\\part*{Appendices}\n"
        "\\addcontentsline{toc}{part}{Appendices}\n")
    add(block(lines, "law", level="section",
              retitle="Three regimes, and a drift coefficient inside one"))
    add(block(lines, "capability"))
    add(block(lines, "probe"))
    add(block(lines, "prediction", level="section",
              retitle="What this would imply for choosing a quantizer"))
    add(block(lines, "review", level="section",
              retitle="Revision and audit trail: what external review raised"))
    add(block(lines, "repro"))
    add(block(lines, "design", level="section",
              retitle="The follow-up design, and what running part of it showed"))
    add(TAIL)
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="validate the block map against the source, write nothing")
    ap.add_argument("--out", type=Path, default=TARGET)
    args = ap.parse_args()

    text = build()
    if args.check:
        print(f"block map OK: {len(BLOCKS)} blocks resolve against {SOURCE}")
        return 0
    if args.out.resolve() == SOURCE.resolve():
        raise SystemExit("refusing to overwrite the research record")
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}  ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
