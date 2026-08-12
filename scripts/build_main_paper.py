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

# Blocks of the record, located by their own heading text rather than by line
# number. Line numbers were the first design and they were wrong: editing the
# record by four lines silently shifted every span after it, and the only reason
# that did not ship was a separate anchor check. Searching for the heading makes
# the map survive edits to the record, which is the normal case rather than the
# exceptional one.
#
# Order is document order. A block ends where the next one begins, so the list
# must name every region including the ones the reframed paper does not emit.
BACKSLASH = chr(92)

ANCHORS: list[tuple[str, str]] = [
    ("_abstract",     BACKSLASH + "begin{abstract}"),
    ("_intro",      r"\section{Introduction}"),
    ("protocol",    r"\subsection{The CliffGuard protocol}"),
    ("related",     r"\section{Related work}"),
    ("method",      r"\section{Method}"),
    ("degeneracy",  r"\subsection{Degeneracy must be detected"),
    ("scorers",     r"\subsection{Two label scorers"),
    ("refusal",     r"\section{Quantization shifts the refusal"),
    ("law",         r"\subsection{Three regimes"),
    ("artifact",    r"\section{Where the reported cliff comes from}"),
    ("monotone",    r"\subsection{The phrase-list estimate is not monotone"),
    ("labelled",    r"\subsection{A third family"),
    ("_remeasure",  r"\section{Re-measuring the same completions"),
    ("scorer",      r"\subsection{The label scorer was not comparing"),
    ("round4",      r"\subsection{The whole ladder, graded twice"),
    ("budget",      r"\subsection{The generation budget"),
    ("regen",       r"\subsection{Greedy decoding did not reproduce"),
    ("standing",    r"\subsection{What the three tests leave standing}"),
    ("human",       r"\section{An author-blinded annotation audit"),
    ("capability",  r"\section{Arithmetic accuracy degrades"),
    ("probe",       r"\section{One frozen refusal direction"),
    ("discussion",  r"\section{Discussion}"),
    ("prediction",  r"\subsection{What this implies for choosing"),
    ("review",      r"\section{What external review raised"),
    ("limits",      r"\section{Limitations}"),
    ("conclusion",  r"\section{Conclusion}"),
    ("repro",       r"\section{Reproducibility}"),
    ("_appendix",     BACKSLASH + "appendix"),
    ("design",      r"\section{Appendix: the follow-up design"),
    ("_bib",          BACKSLASH + "bibliography{"),
]


def locate(lines: list[str]) -> dict[str, tuple[int, int]]:
    """Resolve every anchor to a 1-indexed span, and check document order."""
    at: list[tuple[str, int]] = []
    for name, anchor in ANCHORS:
        hits = [i + 1 for i, ln in enumerate(lines) if ln.startswith(anchor)]
        if not hits:
            raise SystemExit(
                f"anchor for {name!r} not found in the record: {anchor!r}. "
                "The heading was edited -- update ANCHORS.")
        if len(hits) > 1:
            raise SystemExit(
                f"anchor for {name!r} matches {len(hits)} lines {hits}; it must "
                "identify exactly one heading.")
        at.append((name, hits[0]))
    order = [n for n, _ in at]
    if [n for n, _ in sorted(at, key=lambda x: x[1])] != order:
        raise SystemExit("ANCHORS are not in document order; fix the list.")
    spans = {}
    for i, (name, line) in enumerate(at):
        spans[name] = (line, at[i + 1][1] if i + 1 < len(at) else len(lines) + 1)
    return spans


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
    start, end = SPANS[name]
    body = lines[start - 1:end - 1]
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
estimate's sensitivity to each decision is measured rather than assumed. We
apply it to a ladder
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
methodological result --- \textbf{in the configurations tested, paired contrasts
were far less sensitive to the labelling than absolute rates were} --- and it is
why every claim here is a paired one. Three assignments on two models is not a
theorem about paired evaluation, and we report an observed asymmetry rather than
a property.

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
by is its \emph{sensitivity} to that decision, and no grader has to be correct
for the comparison to be valid.

We say sensitivity rather than contribution deliberately. A contribution implies
a decomposition, $\Delta = \Delta_{	ext{gate}} + \Delta_{	ext{grader}} +
\dots$, and these factors interact: \S
ef{sec:artifact} shows the grader term
depends on which completions the gate admitted, so the terms are not additive
and the order in which they are varied matters. What each subsection below
reports is the movement produced by changing one decision from the configuration
this paper actually used --- a one-at-a-time sensitivity around a baseline, not
a unique attribution. A full factorial would be needed for that, and we did not
run one.

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
themselves, and it is the weakest: two models carry the refusal arm, one grader
family produced the labels, and independent graders do not reproduce the
significance. It is reported as exploratory throughout.
"""

NEW_FINDINGS = r"""
\begin{keybox}
	extbf{Contributions.} Four controlled perturbations of a measurement
pipeline, each applied to identical stored completions. Neither grader needs to
be correct for any of them: the counterfactual is exact because only one
decision changes.

\begin{enumerate}[leftmargin=1.4em]
  \item 	extbf{The degeneration gate moves the estimate by two orders of
        magnitude.} At 2.5 bits on Qwen2.5-3B one refusal-phrase list reads
        38.4\% refusal-to-compliance behind a perplexity-only gate and 0.2\%
        behind our composite gate; the judge behind the composite gate reads
        0.0\% (\S
ef{sec:artifact}).
  \item 	extbf{The estimator moves it again, where nothing is degenerate.} At
        4.5 bits both gates admit every completion, so only the grader term
        remains --- and the phrase list reads 7.7 times the judge. One fixed
        list covers 51.3\%, 25.7\% and 3.6\% of a five-way judge's declining
        class across three model families, at precision $1.000$ in every case,
        so the label sets nest and the disagreement runs one way. The estimate
        is \emph{non-monotone} in the phrase list (\S
ef{sec:artifact}).
  \item 	extbf{How a label is read off the judge moves 11--18\% of verdicts.}
        Both graders read a label from first-token logits over label words that
        do not tokenize alike under Qwen2.5. Re-scoring identical text with
        verified single-token options halves the pooled drift slope, $1.15$
        points per bit $[0.75, 1.55]$ to $0.51$ $[0.07, 0.96]$, and the halving
        is a disagreement rather than an attenuation: Qwen2.5-3B holds at
        $+1.16$ while Phi-3.5-mini \emph{reverses sign}, $+1.29$ to $-0.13$
        with an interval covering zero (\S
ef{sec:scorer}).
  \item 	extbf{Which letter carries which class moves absolute rates by ten
        points and the paired contrast by one.} With the option wording
        untouched, full-precision refusal travels across 10.4 points on
        Qwen2.5-3B and 8.8 on Phi-3.5-mini, while the paired
        FP16-to-4.5-bit difference travels across 1.6 and 1.2, reverses
        direction under none of the three assignments tested, and leaves the
        significance decision unchanged. 	extbf{In the configurations tested,
        paired contrasts were far less sensitive to the labelling than absolute
        rates were} (\S
ef{sec:round4}).
\end{enumerate}

\medskip

oindent	extbf{What supports and bounds them.} Part~III puts the instruments
in front of a person: on 300 blinded completions, weighted to the ladder's
strata, the phrase list agrees with a human reading of the declining class 22
points less often than either judge extraction, while the two extractions are
not separated from each other (\S
ef{sec:human}). The same labels show that
the declining class is $35.9$\% refusal, $38.0$\% deflection and $26.1$\%
capability disclaimer. Part~III also reports what a five-times-longer generation
budget and an exact regeneration do to the measurement. 	extbf{What this paper
establishes about quantized models themselves is thinner}, and Part~IV reports
it as exploratory: one model at one rung survives the re-grade --- Qwen2.5-3B at
4.5 bits, 7 transitions toward compliance against 32 toward refusal --- and
three independent graders reproduce its direction on three of four comparisons
and its significance on none.
\end{keybox}
"""

NEW_LIMITS = r"""
\section{Limitations}
\label{sec:limits}

Stated once here, at the strength each deserves. Appendix~\ref{sec:limitsfull}
gives the long form, including the reasoning behind each concession and the
several cases where a robustness test we ran removed a result we had reported.

\begin{itemize}[leftmargin=1.4em]
  \item \textbf{This is not a safety measurement.} The prompt set carries no
        per-prompt harmfulness annotation, so a baseline refusal that becomes
        compliance may be a correction of an over-refusal rather than a
        failure. Every safety-shaped quantity here is a harmful prompt crossed
        with a compliance-\emph{shaped} response; an empty cell means no
        substantive compliance was labelled, not that no harm was done.
  \item \textbf{The endpoint is broader than refusal.} The graders' declining
        class merges refusal, deflection and capability disclaimer, and
        \S\ref{sec:human} measures the mixture at $35.9$/$38.0$/$26.1$\%. Read
        ``refusal'' as ``declining'' throughout, including in the title.
  \item \textbf{The human audit is not independent.} One annotator, who is an
        author, blinded to the row metadata but not to the hypotheses. It ranks
        the phrase list against the two judges; it cannot establish that either
        judge is correct, and no inter-annotator agreement exists. Two
        non-author annotators with adjudication is the single highest-return
        experiment left.
  \item \textbf{The degeneration gate is unvalidated where it is
        load-bearing.} It decides 38.4\% against 0.2\% at 2.5 bits, and the
        human sample covers only FP16 and 4.5 bits, where it rejects nothing.
        We establish that the gate choice matters; we do not establish that
        ours is the right gate.
  \item \textbf{No reference instrument was established.} The two label
        extractions differ by $0.6$ points $[-2.6, +3.6]$ against human labels
        --- no difference detected, equivalence not established --- yet they
        disagree about the sign of the drift effect on Phi-3.5-mini. The paper
        shows instrument instability without identifying a correct instrument,
        and every ladder-wide quantity inherits that.
  \item \textbf{Sensitivity, not decomposition.} Each perturbation is
        one-at-a-time around the configuration we used. The factors interact
        --- the grader term depends on what the gate admitted --- so these are
        not additive contributions, and no factorial design was run.
  \item \textbf{The main ladder is censored at 48 tokens}, which between 92.7
        and 100\% of full-precision completions reach. \S\ref{sec:budget} runs
        256 tokens paired on one act of decoding and finds refusals become
        deflections rather than compliance, but absolute response-type rates on
        the main ladder remain properties of a censored protocol.
  \item \textbf{Greedy decoding only.} One completion per prompt estimates a
        deterministic decision, not a behavioural distribution; regenerating
        changes 9--12\% of completions and 0.2--1.0\% of verdicts. Sampled
        decoding across seeds is not run.
  \item \textbf{The corpus was selected with an instrument this paper
        indicts.} The HH-RLHF strata are response-derived, split by a
        refusal-phrase heuristic, which affects prevalence and class balance;
        and because the strata are contiguous, interrupted external-grader
        sweeps covered them unevenly.
  \item \textbf{Exploratory, not confirmatory.} The design was adjusted while
        looking at results --- the scorer correction, the option-order audit,
        the long-generation and regeneration runs and the human audit were all
        added after inspecting earlier ones. Intervals and $p$-values here
        price sampling error under that adaptivity and do not make any of it
        confirmatory.
  \item \textbf{Scope.} One quantizer family (round-to-nearest), four models of
        which two carry the refusal arm, English, one arithmetic benchmark, one
        judge family sharing a lineage with two of the models judged. Nothing
        here establishes behaviour under AWQ, GPTQ, GGUF k-quants or mixed
        precision.
\end{itemize}
"""


PART_II = r"""
\part*{Part II --- How much of the effect is the pipeline?}
\addcontentsline{toc}{part}{Part II --- How much of the effect is the pipeline?}

\section{Four measurement decisions, on identical stored text}
\label{sec:pipeline}
\label{sec:remeasure}

Each subsection below changes one decision in the measurement pipeline and
nothing else --- same prompts, same
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


SPANS: dict[str, tuple[int, int]] = {}


def build() -> str:
    global SPANS
    lines = load()
    SPANS = locate(lines)
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
    add(NEW_LIMITS)
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
    add(block(lines, "limits", level="section",
              retitle="Limitations, in full")
        .replace(r"\label{sec:limits}", r"\label{sec:limitsfull}", 1))
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
        print(f"block map OK: {len(ANCHORS)} anchors resolve, in document "
                  f"order, against {SOURCE}")
        return 0
    if args.out.resolve() == SOURCE.resolve():
        raise SystemExit("refusing to overwrite the research record")
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}  ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
