"""Verify that numbers written in the manuscript prose match the measurements.

Tables and figures are generated, so they cannot drift. Inline prose is typed by
hand and can, which makes it the weakest link in the paper's provenance chain.
Rounding a bound the wrong way is enough to turn a true statement false: the
simultaneous upper bound is 4.6194%, so "below 4.6%" is an overstatement of
precision that also happens to be wrong.

Each check pairs a value recomputed from docs/paper/review_stats.json with a
regex that must match it *in context*. Context is the point. A bare substring
search for "62" is satisfied by the author's email address and by the "4.62" in
an unrelated bound, so it would pass even after the claim it guards had been
deleted -- a check that cannot fail is worse than no check, because it reads
like assurance. Every pattern here therefore anchors the number to words from
the sentence that carries it.

Exit status is non-zero if any check fails. tests/test_paper_numbers.py runs the
same list, so drift fails the build.

Usage:
  python scripts/check_paper_numbers.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, NamedTuple


class Check(NamedTuple):
    """One quoted quantity, recomputed, plus the context it must appear in.

    `pattern` receives the formatted value and returns a regex. Whitespace in
    the returned pattern is normalised to `\\s+` so a claim may wrap across
    lines, which LaTeX source does constantly.
    """

    label: str
    value: Callable[[dict[str, Any]], str]
    pattern: Callable[[str], str]


def _rx(text: str) -> str:
    """Whitespace-insensitive pattern from a literal-ish template."""
    return r"\s+".join(part for part in text.split() if part)


# ---------------------------------------------------------------------------
# recomputed quantities
# ---------------------------------------------------------------------------


def _transitions(stats: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for block in stats["transitions"].values() for r in block["rows"]]


def _row(stats: dict[str, Any], model: str, bits: float) -> dict[str, Any]:
    return next(r for r in stats["transitions"][model]["rows"] if r["bits"] == bits)


def _gsm_row(stats: dict[str, Any], model: str, bits: float) -> dict[str, Any]:
    return next(r for r in stats["gsm8k"][model]["rows"] if r["bits"] == bits)


def _probe_row(stats: dict[str, Any], model: str, bits: float) -> dict[str, Any]:
    return next(r for r in stats["probe"][model]["rows"] if r["bits"] == bits)


def _gate_row(stats: dict[str, Any], model: str, bits: float) -> dict[str, Any]:
    return next(r for r in stats["gate_by_grader"][model] if r["bits"] == bits)


def _fmt(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _relative_loss(stats: dict[str, Any]) -> str:
    """Percentage of its own baseline Qwen2.5-3B loses at 4.5 bits.

    Quoted as a loss rather than as retained accuracy: "62% retained" and "38%
    lost" describe the same fall, and only one of them is what the sentence in
    the manuscript says.
    """
    block = stats["gsm8k"]["Qwen2.5-3B"]
    row = _gsm_row(stats, "Qwen2.5-3B", 4.5)
    return f"{100 - 100 * row['accuracy'] / block['fp16_accuracy']:.0f}"


def _gsm_paired_interval(stats: dict[str, Any]) -> str:
    r"""Exact 95% interval for the GSM8K accuracy difference at Qwen 4.5 bits.

    Under McNemar the whole signal is in the discordant pairs. With
    $d = \text{lost} + \text{gained}$ and $p = \text{lost}/d$, the accuracy
    difference is $(1-2p)\,d/n$, so an exact Clopper-Pearson interval for $p$
    maps to an exact interval for the difference. The map is DECREASING in $p$
    -- the upper end of the difference comes from the lower end of $p$ -- and
    getting that backwards produces an interval that does not contain its own
    point estimate, which is how the first version of this was caught.

    Exact throughout, to agree with the exact McNemar the paper reports. A
    normal-approximation interval could exclude zero where the exact test does
    not, and the paper leans on that test.
    """
    from scipy import stats as sps

    row = _gsm_row(stats, "Qwen2.5-3B", 4.5)
    lost, gained = row["lost"], row["gained"]
    d, n = lost + gained, 200
    lo_p = 0.0 if lost == 0 else sps.beta.ppf(0.025, lost, d - lost + 1)
    hi_p = 1.0 if lost == d else sps.beta.ppf(0.975, lost + 1, d - lost)
    return f"{100 * (1 - 2 * hi_p) * d / n:.1f} {100 * (1 - 2 * lo_p) * d / n:.1f}"


def _gsm_min_resolvable(stats: dict[str, Any]) -> str:
    """Smallest discordant imbalance the GSM8K design could reject at 0.05."""
    from scipy import stats as sps

    row = _gsm_row(stats, "Qwen2.5-3B", 4.5)
    d = row["lost"] + row["gained"]
    for lost in range(d // 2 + 1, d + 1):
        if sps.binomtest(lost, d, 0.5).pvalue < 0.05:
            return f"{lost} {d - lost}"
    return "none"


def _max_conditional(stats: dict[str, Any]) -> str:
    """Largest P(rung complies | FP16 refused) anywhere on either ladder.

    The reviewer-requested companion to the joint rate. Computed over the FP16
    row of the stored transition matrix. Degenerate and unclear endpoints are
    left in the denominator; at the maximising cell there are none of either, so
    the reported figure does not depend on that choice, and the paper says so.
    """
    best = 0.0
    for block in stats["transitions"].values():
        for row in block["rows"]:
            refused = row["matrix"]["refusal"]
            denom = sum(refused.values())
            if denom:
                best = max(best, refused["compliance"] / denom)
    return f"{100 * best:.2f}"


def _refused_denominators(stats: dict[str, Any]) -> str:
    """How many prompts each model's FP16 baseline refused."""
    out = []
    for model in ("Qwen2.5-3B", "Phi-3.5-mini"):
        block = stats["transitions"][model]
        out.append(str(sum(block["rows"][0]["matrix"]["refusal"].values())))
    return " ".join(out)


CHECKS: tuple[Check, ...] = (
    # ---- what n=200 on GSM8K could and could not resolve --------------------
    Check("GSM8K paired interval",
          _gsm_paired_interval,
          lambda v: _rx(r"accuracy difference of \$\[{}, {}\]\$".format(
              *v.split()))),
    Check("GSM8K smallest resolvable",
          _gsm_min_resolvable,
          lambda v: _rx(r"rejects at \$0\.05\$ is {} against {}".format(
              *v.split()))),
    # ---- the conditional rate a reviewer asks for beside the joint one -----
    Check("max conditional transition rate",
          _max_conditional,
          lambda v: _rx(rf"\\textbf{{{v}\\%}}, at Qwen2\.5-3B's 4\.5-bit rung")),
    Check("FP16-refused denominators",
          _refused_denominators,
          lambda v: _rx(r"the {} \(Qwen2\.5-3B\) and {} \(Phi-3\.5-mini\) "
                        r"prompts the FP16".format(*v.split()))),
    # ---- the simultaneous bound, the number most easily mis-rounded --------
    Check(
        "simultaneous upper bound",
        lambda s: f"{100 * max(r['upper95_simultaneous'] for r in _transitions(s)):.2f}",
        lambda v: _rx(rf"at most {v}") + r"\\?%",
    ),
    # ---- drift ------------------------------------------------------------
    Check("pooled kappa",
          lambda s: f"{s['drift']['_pooled']['estimate']:.2f}",
          lambda v: _rx(rf"{v} refusal-rate points per bit|{v}\$ points per bit")),
    Check("pooled kappa interval",
          lambda s: (f"[{s['drift']['_pooled']['ci_low']:.2f}, "
                     f"{s['drift']['_pooled']['ci_high']:.2f}]"),
          # Built by hand rather than with re.escape: escaping the space inside
          # the interval leaves a trailing backslash on the first whitespace
          # token, which _rx then joins into a literal-backslash match.
          lambda v: _rx(r"CI \$"
                        + v.replace("[", r"\[").replace("]", r"\]")
                           .replace(".", r"\.")
                        + r"\$, prompt-level")),
    Check("Qwen kappa",
          lambda s: f"{s['drift']['Qwen2.5-3B']['kappa']:.2f}",
          lambda v: _rx(rf"kappa}}_{{\\text{{Qwen2.5-3B}}}} = {v}")),
    Check("Phi kappa",
          lambda s: f"{s['drift']['Phi-3.5-mini']['kappa']:.2f}",
          lambda v: _rx(rf"kappa}}_{{\\text{{Phi-3.5-mini}}}} = {v}")),
    Check("Qwen anchor error",
          lambda s: f"{s['drift']['Qwen2.5-3B']['anchor_error_pp']:.1f}",
          lambda v: _rx(rf"{v} points below the observed one")),
    Check("Phi anchor error",
          lambda s: f"{s['drift']['Phi-3.5-mini']['anchor_error_pp']:.1f}",
          lambda v: _rx(rf"and {v} points below on Phi-3.5-mini")),
    Check("Qwen top-rung gap",
          lambda s: f"{s['drift']['Qwen2.5-3B']['top_rung_minus_fp16_pp']:.1f}",
          lambda v: _rx(rf"refusal moves by \$\+{v}\$ points on Qwen2\.5-3B")),
    Check("Phi top-rung gap",
          lambda s: f"{s['drift']['Phi-3.5-mini']['top_rung_minus_fp16_pp']:.1f}",
          lambda v: _rx(rf"and \$\+{v}\$ on Phi-3\.5-mini")),
    # ---- transitions ------------------------------------------------------
    Check("max transition rate",
          lambda s: f"{100 * max(r['rate_itt'] for r in _transitions(s)):.1f}",
          lambda v: _rx(rf"never exceed {v}") + r"\\?%"),
    Check("Qwen 4.5 to-refuse count",
          lambda s: str(_row(s, "Qwen2.5-3B", 4.5)["to_refusal"]),
          lambda v: _rx(rf"Qwen2.5-3B newly refuses {v} prompts")),
    Check("Qwen 4.5 to-comply count",
          lambda s: str(_row(s, "Qwen2.5-3B", 4.5)["to_compliance"]),
          lambda v: _rx(rf"against {v} in the reverse direction|"
                        rf"against {v} in the opposite direction")),
    # Bound to the family subscript, not to a bare "p=". A pattern that matches
    # any Holm-adjusted p-value would pass even if the manuscript quoted the
    # per-model number while calling the 14-cell family primary, which is
    # exactly the substitution worth catching.
    Check("Qwen 4.5 p_7",
          lambda s: f"{_row(s, 'Qwen2.5-3B', 4.5)['mcnemar_p_holm']:.3f}",
          lambda v: _rx(rf"opposite direction \(\$p_\{{14\}}=0\.021\$, "
                        rf"\$p_\{{7\}}={v}\$\)")),
    Check("Phi 4.5 to-refuse count",
          lambda s: str(_row(s, "Phi-3.5-mini", 4.5)["to_refusal"]),
          lambda v: _rx(rf"Phi-3.5-mini (?:shows )?{v} against")),
    Check("Phi 4.5 p_7",
          lambda s: f"{_row(s, 'Phi-3.5-mini', 4.5)['mcnemar_p_holm']:.3f}",
          lambda v: _rx(rf"p_\{{7\}}={v}\$\)\. Both survive")),
    Check("Phi 4.5 p_14 in keybox",
          lambda s: f"{_row(s, 'Phi-3.5-mini', 4.5)['mcnemar_p_holm_all_cells']:.3f}",
          lambda v: _rx(rf"21 against 4\s*\(\$p_\{{14\}}={v}\$")),
    Check("Qwen 4.5 strict-family p",
          lambda s: f"{_row(s, 'Qwen2.5-3B', 4.5)['mcnemar_p_holm_all_cells']:.3f}",
          lambda v: _rx(rf"survive on both models \(\$p={v}\$ for Qwen2\.5-3B")),
    Check("Phi 4.5 strict-family p",
          lambda s: f"{_row(s, 'Phi-3.5-mini', 4.5)['mcnemar_p_holm_all_cells']:.3f}",
          lambda v: _rx(rf"\$p={v}\$ for Phi-3\.5-mini")),
    Check("Qwen 5.5 strict-family p",
          lambda s: f"{_row(s, 'Qwen2.5-3B', 5.5)['mcnemar_p_holm_all_cells']:.3f}",
          lambda v: _rx(rf"5\.5-bit result does not \(\$p={v}\$\)")),
    Check("Qwen 3.5 gradable pairs",
          lambda s: str(_row(s, "Qwen2.5-3B", 3.5)["n_gradable"]),
          lambda v: _rx(rf"Qwen2.5-3B retains {v} of 500 pairs")),
    # ---- artifact decomposition ------------------------------------------
    Check("phrase-list peak",
          lambda s: f"{100 * _gate_row(s, 'Qwen2.5-3B', 2.5)['marker_nll']:.1f}",
          lambda v: _rx(rf"phrase list reports {v}") + r"\\?%\s+under a perplexity-only"),
    Check("phrase-list peak under composite gate",
          lambda s: f"{100 * _gate_row(s, 'Qwen2.5-3B', 2.5)['marker_composite']:.1f}",
          lambda v: _rx(rf"and {v}") + r"\\?%\s+under the"),
    Check("grader gap, phrase list at 4.5",
          lambda s: f"{100 * _gate_row(s, 'Qwen2.5-3B', 4.5)['marker_composite']:.1f}",
          lambda v: _rx(rf"the phrase list reports {v}") + r"\\?%"),
    Check("grader gap, judge at 4.5",
          lambda s: f"{100 * _gate_row(s, 'Qwen2.5-3B', 4.5)['judge_composite']:.1f}",
          lambda v: _rx(rf"against the judge's {v}") + r"\\?%"),
    Check("grader ratio at 4.5 bits",
          lambda s: _fmt(_gate_row(s, "Qwen2.5-3B", 4.5)["marker_composite"]
                         / _gate_row(s, "Qwen2.5-3B", 4.5)["judge_composite"], 1),
          lambda v: _rx(rf"a factor\s+of {v}")),
    # ---- what the refusal class contains ---------------------------------
    Check("audit: new refusals",
          lambda s: str(s["refusal_class_audit"]["Qwen2.5-3B"]["n_new_refusals"]),
          lambda v: _rx(rf"We read the {v}")),
    Check("audit: with marker",
          lambda s: str(s["refusal_class_audit"]["Qwen2.5-3B"]
                        ["n_new_refusals_with_marker"]),
          # Case-insensitive on the first word: the sentence has been reworded
          # once already, and a checker that fails on capitalisation trains its
          # reader to ignore it.
          lambda v: _rx(rf"[Oo]nly {v} of the 32 contain a refusal marker")),
    # The marginal-vs-average comparison. Three numbers that only mean anything
    # together, so all three are pinned: a reworded sentence that keeps one and
    # drops another would still read plausibly.
    Check("audit: marginal marker share",
          lambda s: _fmt(100 * s["refusal_class_audit"]["Qwen2.5-3B"]
                         ["n_new_refusals_with_marker"]
                         / s["refusal_class_audit"]["Qwen2.5-3B"]["n_new_refusals"], 1),
          lambda v: _rx(rf"That is {v}\\%, against")),
    Check("audit: FP16 class marker share",
          lambda s: _fmt(100 * s["refusal_class_audit"]["Qwen2.5-3B"]
                         ["fp16_marker_share"], 1),
          lambda v: _rx(rf"{v}\\% for the class those 32 are joining")),
    Check("audit: rung class marker share",
          lambda s: _fmt(100 * s["refusal_class_audit"]["Qwen2.5-3B"]
                         ["rung_refusals_with_marker"]
                         / s["refusal_class_audit"]["Qwen2.5-3B"]["rung_refusals"], 1),
          lambda v: _rx(rf"{v}\\% for the class at this rung")),
    Check("audit: mean length new refusals",
          lambda s: _fmt(s["refusal_class_audit"]["Qwen2.5-3B"]
                         ["mean_len_new_refusals"], 0),
          lambda v: _rx(rf"mean length is {v} characters")),
    Check("audit: FP16 marker share",
          lambda s: _fmt(100 * s["refusal_class_audit"]["Qwen2.5-3B"]
                         ["fp16_marker_share"], 0),
          lambda v: _rx(rf"{v}") + r"\\?%\s+of\s+it\s+contains\s+a"),
    Check("audit: rung marker share",
          lambda s: _fmt(100 * s["refusal_class_audit"]["Qwen2.5-3B"]
                         ["rung_marker_share"], 0),
          lambda v: _rx(rf"against {v}") + r"\\?%\s+at\s+4\.5\s+bits"),
    Check("audit: Phi marker share shift",
          lambda s: _fmt(100 * s["refusal_class_audit"]["Phi-3.5-mini"]
                         ["rung_marker_share"], 0),
          lambda v: _rx("rises from 16") + r"\\?%\s+to\s+" + v + r"\\?%"),
    # ---- gate ablation ---------------------------------------------------
    Check("gate: alpha rule at 2.5 bits",
          lambda s: _fmt(100 * next(
              r["fires_per_rule"]["alpha"]
              for r in s["gate_ablation"]["Qwen2.5-3B"]["rows"]
              if r["bits"] == 2.5), 1),
          lambda v: _rx(rf"fires on {v}") + r"\\?% of"),
    Check("gate: token-share rule at 2.5 bits",
          lambda s: _fmt(100 * next(
              r["fires_per_rule"]["token_share"]
              for r in s["gate_ablation"]["Qwen2.5-3B"]["rows"]
              if r["bits"] == 2.5), 1),
          lambda v: _rx(rf"share rule on {v}") + r"\\?%"),
    Check("gate: nll rule at 2.5 bits",
          lambda s: _fmt(100 * next(
              r["fires_per_rule"]["nll"]
              for r in s["gate_ablation"]["Qwen2.5-3B"]["rows"]
              if r["bits"] == 2.5), 1),
          lambda v: _rx(rf"on only {v}") + r"\\?%"),
    Check("gate: without nll at 2.5 bits",
          lambda s: _fmt(100 * next(
              r["without_nll"] for r in s["gate_ablation"]["Qwen2.5-3B"]["rows"]
              if r["bits"] == 2.5), 1),
          lambda v: _rx(rf"leaves that rung at {v}") + r"\\?%"),
    # ---- capability -------------------------------------------------------
    Check("GSM8K FP16 accuracy",
          lambda s: f"{100 * s['gsm8k']['Qwen2.5-3B']['fp16_accuracy']:.1f}",
          lambda v: _rx(rf"from {v}") + r"\\?%"),
    Check("GSM8K 4.5 accuracy",
          lambda s: f"{100 * _gsm_row(s, 'Qwen2.5-3B', 4.5)['accuracy']:.1f}",
          lambda v: _rx(rf"accuracy falls from 18\.5\\% to {v}") + r"\\?%"),
    Check("GSM8K questions lost",
          lambda s: str(_gsm_row(s, "Qwen2.5-3B", 4.5)["lost"]),
          lambda v: _rx(rf"{v} questions lost against")),
    Check("GSM8K questions gained",
          lambda s: str(_gsm_row(s, "Qwen2.5-3B", 4.5)["gained"]),
          lambda v: _rx(rf"questions lost against {v} gained")),
    Check("GSM8K relative loss",
          lambda s: _relative_loss(s),
          lambda v: _rx(rf"{v}") + r"\\?%\s+of its baseline"),
    Check("GSM8K paired p",
          lambda s: f"{_gsm_row(s, 'Qwen2.5-3B', 4.5)['mcnemar_p']:.3f}",
          lambda v: _rx(rf"McNemar test gives \$p={v}\$")),
    Check("GSM8K Holm p, per-model family",
          lambda s: f"{_gsm_row(s, 'Qwen2.5-3B', 4.5)['mcnemar_p_holm']:.3f}",
          lambda v: _rx(rf"p_\{{7\}}={v}\$ within this model")),
    Check("GSM8K Holm p, all cells",
          lambda s: _fmt(_gsm_row(s, "Qwen2.5-3B", 4.5)["mcnemar_p_holm_all_cells"]),
          lambda v: _rx(rf"p_\{{21\}}={v}\$ over every cell")),
    # ---- probe ------------------------------------------------------------
    Check("probe band floor",
          lambda s: f"{100 * min(r['retained_mean'] for b in s['probe'].values() for r in b['rows'] if 4.5 <= r['bits'] <= 8.5):.0f}",
          lambda v: _rx(rf"between {v}") + r"\\?%\s+and 100"),
    Check("probe band span, Qwen",
          lambda s: (lambda vals: f"{100 * (max(vals) - min(vals)):.1f}")(
              [r["retained_mean"] for r in s["probe"]["Qwen2.5-3B"]["rows"]
               if 4.5 <= r["bits"] <= 8.5]),
          lambda v: _rx(rf"by {v} points on Qwen2\.5-3B")),
    Check("Qwen 4.5 retention",
          lambda s: f"{100 * _probe_row(s, 'Qwen2.5-3B', 4.5)['retained_mean']:.1f}",
          lambda v: _rx(rf"sits at {v}") + r"\\?%\s+at\s+4\.5\s+bits"),
    Check("probe absolute d', Qwen",
          lambda s: _fmt(s["probe"]["Qwen2.5-3B"]["fp16_absolute_dprime"], 2),
          lambda v: _rx(rf"only \${v}\$ for Qwen2\.5-3B")),
    Check("probe absolute d', Phi",
          lambda s: _fmt(s["probe"]["Phi-3.5-mini"]["fp16_absolute_dprime"], 2),
          lambda v: _rx(rf"\${v}\$ for Phi-3\.5-mini, which is a strong")),
    Check("Phi 3.5 retention",
          lambda s: f"{100 * _probe_row(s, 'Phi-3.5-mini', 3.5)['retained_mean']:.0f}",
          lambda v: _rx(rf"fallen to {v}") + r"\\?%"),
    Check("Qwen 3.5 retention",
          lambda s: f"{100 * _probe_row(s, 'Qwen2.5-3B', 3.5)['retained_mean']:.0f}",
          lambda v: _rx(rf"Qwen2.5-3B to {v}") + r"\\?%"),
)


class UniqueCheck(NamedTuple):
    """A quantity the paper must state ONE value for, wherever it states it.

    Every check above asks whether the right number appears somewhere. None asks
    whether a *wrong* number for the same quantity appears somewhere else, and
    that gap was not hypothetical: Phi-3.5-mini's 3.5-bit probe retention was
    written as 57% in two places and 63% in two others. Both figures are real --
    57% is the judge-label variant the section reports, 63% the marker-label
    variant an earlier revision used -- so the paper quietly carried numbers from
    two different estimands under one name, and the "expected 57 in context"
    check passed the whole time because 57 was indeed present.

    `pattern` captures the number in group 1. Every occurrence must agree with
    the recomputed value.
    """

    label: str
    value: Callable[[dict[str, Any]], str]
    pattern: str


MIN_GRADED = 400          # matches build_paper_tables and build_paper_figures


_WORD_NUMBER = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8",
                "nine": "9", "ten": "10"}


def _digits(text: str) -> str:
    """Normalise `three of four` and `3 of 4` to one form.

    Prose spells small numbers out; a recomputed tally does not. Without this
    the consistency check reports a contradiction between two ways of writing
    the same fact, which trains a reader to ignore it.
    """
    return " ".join(_WORD_NUMBER.get(w.lower(), w) for w in text.split())


def _direction_tally(agreement: dict[str, Any]) -> str:
    """How many grader x model comparisons agree on the direction, and of how many.

    Guarded because the paper contradicted itself here: three passages claimed
    the shift was "consistent in sign under every grader we tried" while the
    grader section said three of four agreed. One comparison
    (Llama-3.3-70B on Qwen2.5-3B) runs the other way, 1 toward refusal against
    2 toward compliance. A claim of unanimity is the most damaging kind of
    error a replication section can contain, so it is recomputed rather than
    trusted.

    Two exclusions, both the same ones the agreement range applies. A sweep
    below MIN_GRADED completions cannot support a paired comparison at all, and
    a tie -- no transitions in either direction -- has neither agreed nor
    disagreed with anything. Under those, the tally is 3 of 4; without the
    floor it would read 4 of 5, and quoting that would count two four-completion
    sweeps as evidence.
    """
    toward = total = 0
    for models in agreement.values():
        for block in models.values():
            if block.get("n_compared", 0) < MIN_GRADED:
                continue
            for row in block.get("reproduction") or []:
                if row["to_refusal"] == row["to_compliance"]:
                    continue
                total += 1
                toward += row["to_refusal"] > row["to_compliance"]
    return f"{toward} of {total}"


def _agreement_range(agreement: dict[str, Any]) -> str:
    """Agreement span over the sweeps large enough to support a comparison.

    The floor is the whole point. Quoting the range over every sweep pulled the
    upper end to 89.3%, from a grader that returned 375 completions, never
    covered both rungs, and therefore contributed no paired result -- the most
    flattering number in the set, produced by the least complete work.
    """
    values = [b["agreement"] for models in agreement.values()
              for b in models.values()
              if b.get("n_compared", 0) >= MIN_GRADED]
    return f"{100 * min(values):.1f}--{100 * max(values):.1f}"


UNIQUE_CHECKS: tuple[UniqueCheck, ...] = (
    UniqueCheck(
        # Catches the self-contradiction directly: any sentence of the form
        # "<n> of <m> grader comparisons" must agree with the recomputed tally.
        # The unanimity phrasings that prompted this ("consistent in sign under
        # every grader we tried") state no numbers at all, so they are caught by
        # the separate no-unanimity test rather than here.
        "grader direction tally stated consistently",
        lambda s: s["_direction_tally"],
        r"(\w+ of (?:\w+)) grader comparisons",
    ),
    UniqueCheck(
        "Phi 3.5 retention stated consistently",
        lambda s: f"{100 * _probe_row(s, 'Phi-3.5-mini', 3.5)['retained_mean']:.0f}",
        r"(?:retention (?:is|has)\s+already\s+(?:fallen\s+to\s+)?|fallen\s+to\s+)(\d+)\\?%",
    ),
    UniqueCheck(
        "Qwen 4.5 retention stated consistently",
        lambda s: f"{100 * _probe_row(s, 'Qwen2.5-3B', 4.5)['retained_mean']:.1f}",
        r"Qwen2\.5-3B\s+sits\s+at\s+([\d.]+)\\?%\s+at\s+4\.5\s+bits",
    ),
    UniqueCheck(
        "probe band floor stated consistently",
        lambda s: f"{100 * min(r['retained_mean'] for b in s['probe'].values() for r in b['rows'] if 4.5 <= r['bits'] <= 8.5):.0f}",
        r"(?:retains|retention[^.]{0,40}?)\s*(?:is\s+near-flat\s+at\s+)?(\d+)--100\\?%",
    ),
    UniqueCheck(
        "judge agreement range stated consistently",
        lambda s: s["_agreement_range"],
        r"agreement with the 7B judge (?:is|runs)\s+([\d.]+--[\d.]+)\\?%",
    ),
)


class DocCheck(NamedTuple):
    """A headline number that also appears outside the manuscript.

    The README and the claims ledger restate the paper's main figures, and
    nothing regenerates them, so they are exactly as liable to drift as the
    prose was -- and more visible, since the README is the first thing a reader
    sees. These are the numbers that would mislead if they went stale.
    """

    label: str
    path: Path
    value: Callable[[dict[str, Any]], str]
    pattern: Callable[[str], str]


README = Path("README.md")
LEDGER = Path("docs/claims_and_evidence.md")


def _pooled(stats: dict[str, Any], key: str) -> str:
    return f"{stats['drift']['_pooled'][key]:.2f}"


DOC_CHECKS: tuple[DocCheck, ...] = (
    DocCheck("README pooled kappa", README,
             lambda s: _pooled(s, "estimate"),
             lambda v: _rx(rf"\*\*{v} points per bit removed\*\*")),
    DocCheck("README kappa interval", README,
             lambda s: f"[{_pooled(s, 'ci_low')}, {_pooled(s, 'ci_high')}]",
             lambda v: _rx("95% CI " + re.escape(v).replace(",\\ ", ", "))),
    DocCheck("README max transition rate", README,
             lambda s: f"{100 * max(r['rate_itt'] for r in _transitions(s)):.1f}",
             lambda v: _rx(rf"Never above \*\*{v}%\*\*")),
    DocCheck("README simultaneous bound", README,
             lambda s: f"{100 * max(r['upper95_simultaneous'] for r in _transitions(s)):.2f}",
             lambda v: _rx(rf"upper bound \*\*{v}%\*\*")),
    DocCheck("ledger pooled kappa", LEDGER,
             lambda s: _pooled(s, "estimate"),
             lambda v: _rx(rf"\*\*{v} points/bit\*\*")),
    DocCheck("ledger simultaneous bound", LEDGER,
             lambda s: f"{100 * max(r['upper95_simultaneous'] for r in _transitions(s)):.2f}",
             lambda v: _rx(rf"\*\*{v}%\*\* simultaneous")),
    DocCheck("ledger max transition rate", LEDGER,
             lambda s: f"{100 * max(r['rate_itt'] for r in _transitions(s)):.1f}",
             lambda v: _rx(rf"stay \*\*.{{0,4}}{v}%\*\*")),
)


def _marker_spread(data: dict[str, Any], model: str) -> str:
    """Largest ratio between marker-list variants at any rung.

    Sourced from data.json rather than review_stats.json: the marker-variant
    sweep is produced by the judge run, not by the re-analysis. Rungs where the
    minimum is zero are skipped -- the ratio is unbounded there and says nothing
    about sensitivity.
    """
    variants = data["behavioural"][model]["marker_variants"]
    ratios = []
    for scheme in next(iter(variants.values())):
        values = [v[scheme] for v in variants.values()]
        if min(values) > 0:
            ratios.append(max(values) / min(values))
    return f"{max(ratios):.2f}"


DATA_CHECKS: tuple[Check, ...] = (
    Check("marker spread, Qwen2.5-3B",
          lambda d: _marker_spread(d, "Qwen2.5-3B"),
          lambda v: _rx(rf"up to \${v}\\times\$ on Qwen2\.5-3B")),
    Check("marker spread, Phi-3.5-mini",
          lambda d: _marker_spread(d, "Phi-3.5-mini"),
          lambda v: _rx(rf"\${v}\\times\$ on Phi-3\.5-mini")),
)


# ---------------------------------------------------------------------------
# the labelled-suite subsection (round 3)
# ---------------------------------------------------------------------------


def _recall(labelled: dict[str, Any], model: str) -> str:
    return f"{100 * labelled[model]['recall']:.1f}"


def _bound(k: int, n: int, family: int) -> str:
    """One-sided Clopper-Pearson, Bonferroni-adjusted for a family of `family`.

    Recomputed here rather than read from a file, because the three bounds the
    subsection quotes differ only in their family size and mixing them up is the
    exact error the sentence exists to prevent.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.review_reanalysis import clopper_pearson_upper

    return f"{100 * clopper_pearson_upper(k, n, 1 - 0.05 / family):.2f}"


def _flagged() -> str:
    """Withheld-harmful completions the exploratory pattern matches at FP16.

    Read from `leakage_stats.json` rather than `labelled_paper_stats.json`,
    because this quantity is deliberately kept out of the tables: it is a count
    of string matches by an in-sample rule, not a measurement, and the paper
    says so where it states it.
    """
    path = Path("docs/paper/leakage_stats.json")
    blocks = json.loads(path.read_text(encoding="utf-8"))
    by_model = {b["model"]: b["fp16_sensitive_comply"] for b in blocks}
    smol = next(v for k, v in by_model.items() if "SmolLM2" in k)
    qwen = next(v for k, v in by_model.items() if "Qwen2.5-3B" in k)
    phi = next(v for k, v in by_model.items() if "Phi-3.5" in k)
    return f"{smol} {qwen} {phi}"


TABLE_ORDER = ("Qwen2.5-3B", "Phi-3.5-mini", "SmolLM2-1.7B")

# Regex-safe spellings; the dots in a model name are otherwise wildcards, and
# `Phi-3.5-mini` unescaped would happily match `Phi-3X5-mini`.
MODEL_TEX = {
    "Qwen2.5-3B": r"Qwen2\.5-3B",
    "Phi-3.5-mini": r"Phi-3\.5-mini",
    "SmolLM2-1.7B": r"SmolLM2-1\.7B",
}


def _per_model(label: str, value, template: str) -> tuple[Check, ...]:
    r"""One check per model, with the number anchored to that model's name.

    The section used to quote its per-model quantities as bare triples --
    "137, 65 and 10" -- leaving the reader to map position onto table rows.
    Two different orderings were in use at once (the phrase-list paragraphs ran
    descending by coverage, everything else ran in table order), so half the
    triples pointed at the wrong model. Naming the model beside each number
    fixed the prose; checking each number *against that name* is what stops it
    coming back, because a positional check cannot tell a reordering from a
    correct list.
    """
    return tuple(
        Check(f"{label}, {model}",
              (lambda d, model=model: value(d, model)),
              (lambda v, model=model:
               _rx(template.format(v=v, m=MODEL_TEX[model]))))
        for model in TABLE_ORDER)


LABELLED_CHECKS: tuple[Check, ...] = (
    # "coverage relative to the judge", not "recall": the denominator is the
    # instrument this paper indicts, and the prose says so rather than implying
    # accuracy against a ground truth it does not have.
    Check("labelled coverage, Qwen2.5-3B",
          lambda d: _recall(d, "Qwen2.5-3B"),
          lambda v: _rx(rf"is {v}\\%, 23\.5\\% and 3\.6\\%")),
    Check("labelled coverage, SmolLM2",
          lambda d: _recall(d, "SmolLM2-1.7B"),
          lambda v: _rx(rf"is 49\.3\\%, {v}\\% and 3\.6\\%")),
    Check("labelled coverage, Phi-3.5-mini",
          lambda d: _recall(d, "Phi-3.5-mini"),
          lambda v: _rx(rf"is 49\.3\\%, 23\.5\\% and {v}\\%")),
    *_per_model("truncation at full precision",
                lambda d, m: f"{100 * d[m]['at_cap_by_scheme']['FP16']:.1f}",
                r"{v}\\% of {m}'s"),
    *_per_model("phrase-list count",
                lambda d, m: str(d[m]["marker"]),
                r"{v} of {m}'s"),
    *_per_model("benign compliance",
                lambda d, m: str(d[m]["benign_comply"]),
                r"{v} (?:times )?on {m}"),
    # The harmful-deflection counts behind the stratified truncation claim.
    # Quoted in prose only, so nothing else would catch them drifting.
    Check("harmful deflections, Phi and SmolLM2",
          lambda d: " ".join(str(d[m]["harmful_deflect"]) for m in
                             ("Phi-3.5-mini", "SmolLM2-1.7B")),
          lambda v: _rx(r"\({} of 150\s+for Phi-3\.5-mini, {} for "
                        r"SmolLM2-1\.7B\)".format(*v.split()))),
    # Cells quoted only in the caption of the full-precision matrix figure.
    # Captions drift more easily than body text because nothing else reads
    # them, so they are checked the same way the prose is. Both triples run in
    # the table's row order, which is what the caption's reader will assume.
    Check("FP16 harmful refusals",
          lambda d: " ".join(str(d[m]["harmful_refuse"]) for m in TABLE_ORDER),
          lambda v: _rx(r"{}, {} and {} outright refusals".format(*v.split()))),
    Check("FP16 harmful deflections",
          lambda d: " ".join(str(d[m]["harmful_deflect"]) for m in TABLE_ORDER),
          lambda v: _rx(r"against {}, {} and {} deflections".format(*v.split()))),
    Check("marker precision, all three",
          lambda d: f"{min(m['precision'] for m in d.values()):.3f}",
          lambda v: _rx(rf"precision\s+against the judge is exactly \${v}\$")),
    *_per_model("completions the list misses",
                lambda d, m: str(d[m]["missed"]),
                r"{v} (?:completions )?on {m}"),
    *_per_model("kappa",
                lambda d, m: f"{d[m]['kappa']:.3f}",
                r"\${v}\$ on {m}"),
    # The exploratory pattern counts. Guarded precisely because they are the
    # numbers most tempting to leave stale: they are not carried into any table
    # or bound, so nothing else would catch them drifting.
    Check("pattern-flagged withheld, all three",
          lambda d: _flagged(),
          lambda v: _rx(r"matches {} of SmolLM2's 150\s+withheld harmful "
                        r"completions at full precision, {} of Qwen2\.5-3B's "
                        r"and {} of\s+Phi-3\.5-mini's".format(*v.split()))),
    Check("bound, one cell",
          lambda d: _bound(0, 150, 1),
          lambda v: _rx(rf"is {v}\\% for a single")),
    Check("bound, seven rungs",
          lambda d: _bound(0, 150, 7),
          lambda v: _rx(rf"{v}\\% simultaneously across one model")),
    Check("bound, twenty-one cells",
          lambda d: _bound(0, 150, 21),
          lambda v: _rx(rf"and {v}\\% across all 21")),
)


def _matched(cov: dict[str, Any], model: str, grader: str,
             side: str) -> str:
    """Transitions on a coverage-matched subset, external or 7B.

    These are the numbers that separate grader disagreement from coverage
    disagreement, and the paper now leads with them. They are also the easiest
    numbers in the manuscript to get subtly wrong by hand, because each one is
    a pair of counts restricted to a subset defined by a different file.
    """
    row = next(r for r in cov[model]["rows"] if r["grader"] == grader)
    block = row["own"] if side == "external" else row["local_on_same"]
    return f"{block['to_refusal']} {block['to_compliance']}"


COVERAGE_CHECKS: tuple[Check, ...] = (
    Check("Qwen/Claude, external",
          lambda c: _matched(c, "Qwen2.5-3B", "agent_claude", "external"),
          lambda v: _rx(r"{} against {}".format(*v.split()))),
    Check("Qwen/Claude, 7B matched",
          lambda c: _matched(c, "Qwen2.5-3B", "agent_claude", "local"),
          lambda v: _rx(r"{} against {}".format(*v.split()))),
    Check("Phi/Haiku, external",
          lambda c: _matched(c, "Phi-3.5-mini", "agent_haiku", "external"),
          lambda v: _rx(r"{} against {}".format(*v.split()))),
    Check("Phi/Haiku, 7B matched",
          lambda c: _matched(c, "Phi-3.5-mini", "agent_haiku", "local"),
          lambda v: _rx(r"{} against {}".format(*v.split()))),
    Check("Qwen 7B full coverage",
          lambda c: (f"{c['Qwen2.5-3B']['local_all']['to_refusal']} "
                     f"{c['Qwen2.5-3B']['local_all']['to_compliance']}"),
          lambda v: _rx(r"{} transitions toward refusal against {}".format(
              *v.split()))),
)


_MATRIX_KEY = {"Qwen2.5-3B": "qwen3b", "Phi-3.5-mini": "phi35",
               "SmolLM2-1.7B": "smol17"}


def _degenerate_share(matrix: dict[str, Any], model: str, scheme: str) -> str:
    """What fraction of the usefulness lost at `scheme` is degenerate output.

    The distinction the figure exists to draw: a benign prompt the model stops
    answering because it decided to decline is a different finding from one it
    stops answering because it stopped producing language. At 3.5 bits the two
    are mixed, and mixed *differently* per family, which is the only reason the
    caption quotes three numbers instead of one.
    """
    block = next((v for k, v in matrix.items() if _MATRIX_KEY[model] in k), None)
    row = next(r for r in block["paired"] if r["scheme"] == scheme)
    lost = (row["utility_rate"] or 0) * row["n_benign"]
    share = 100 * row["utility_lost_by_class"]["degenerate"] / lost
    # "100" reads better than "100.0" in a caption, and the paper writes it
    # that way; match the rendered form rather than forcing the prose.
    return f"{share:.1f}".removesuffix(".0")


MATRIX_CHECKS: tuple[Check, ...] = tuple(
    Check(f"degenerate share at 3.5 bits, {model}",
          (lambda m, model=model: _degenerate_share(m, model, "RTN_3B")),
          (lambda v, model=model:
           _rx(rf"{v}\\% (?:degenerate )?on {MODEL_TEX[model]}")))
    for model in TABLE_ORDER)


def live_tex(source: str) -> str:
    r"""Strip what the reader never sees, so a check cannot be satisfied by it.

    Matching against raw source means a commented-out or \iffalse-disabled
    sentence carrying the right number passes the assertion while the rendered
    paper says something else, or nothing. Percent signs escaped as \% are
    not comments and must survive.
    """
    without_inactive = re.sub(r"\\iffalse\\b.*?\\fi\\b", "", source, flags=re.S)
    return re.sub(r"(?<!\\)%.*", "", without_inactive)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", type=Path,
                    default=Path("docs/paper/cliff_artifact.tex"))
    ap.add_argument("--stats", type=Path,
                    default=Path("docs/paper/review_stats.json"))
    ap.add_argument("--data", type=Path,
                    default=Path("docs/paper/data.json"))
    ap.add_argument("--agreement", type=Path,
                    default=Path("docs/paper/judge_agreement.json"))
    ap.add_argument("--coverage", type=Path,
                    default=Path("docs/paper/grader_coverage.json"),
                    help="coverage-matched grader transitions; the numbers that "
                         "separate grader disagreement from coverage disagreement")
    ap.add_argument("--matrix", type=Path,
                    default=Path("docs/paper/matrix_stats.json"),
                    help="paired per-rung blocks; source of the degenerate "
                         "share quoted in the utility figure's caption")
    ap.add_argument("--labelled", type=Path,
                    default=Path("docs/paper/labelled_paper_stats.json"),
                    help="round-3 measurements; skipped when absent, because "
                         "the labelled subsection is only in the paper when "
                         "those runs are")
    args = ap.parse_args()

    # The manuscript and its consolidated measurement files are not published
    # from this repository -- results are released deliberately, not as a side
    # effect of a commit. This checker is therefore an author's tool, and on a
    # fresh clone it has nothing to check. Say that, rather than raising a
    # FileNotFoundError that reads like the repository is broken.
    missing = [p for p in (args.tex, args.stats, args.data) if not p.exists()]
    if missing:
        print("nothing to check: " + ", ".join(str(p) for p in missing)
              + " not present.\n"
              "The manuscript and its data files are kept out of this "
              "repository. Regenerate them from run directories with:\n"
              "  python scripts/build_paper_data.py\n"
              "  python scripts/review_reanalysis.py --gsm8k <test.jsonl>")
        return 0

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    # Folded in so the uniqueness checks read one object. The agreement range is
    # recomputed here rather than stored, because the floor that defines it is a
    # decision this file has to be able to state.
    if args.agreement.exists():
        agreement = json.loads(args.agreement.read_text(encoding="utf-8"))
        stats["_agreement_range"] = _agreement_range(agreement)
        stats["_direction_tally"] = _direction_tally(agreement)
    data = json.loads(args.data.read_text(encoding="utf-8"))
    text = live_tex(args.tex.read_text(encoding="utf-8"))

    failures: list[str] = []
    print(f"{'quantity':34s} {'expected':16s} status")
    print("-" * 64)
    for check in CHECKS:
        value = check.value(stats)
        found = re.search(check.pattern(value), text) is not None
        print(f"{check.label:34s} {value:16s} {'ok' if found else 'MISSING'}")
        if not found:
            failures.append(f"  {check.label}: expected {value!r} in context "
                            f"/{check.pattern(value)}/")

    for check in DATA_CHECKS:
        value = check.value(data)
        found = re.search(check.pattern(value), text) is not None
        print(f"{check.label:34s} {value:16s} {'ok' if found else 'MISSING'}")
        if not found:
            failures.append(f"  {check.label}: expected {value!r} in context "
                            f"/{check.pattern(value)}/")

    # Round 3 travels with its own measurement file. If the subsection is in the
    # manuscript, its numbers are checked; if the runs are absent, so is the
    # subsection, and there is nothing to check rather than something broken.
    if args.labelled.exists():
        labelled = json.loads(args.labelled.read_text(encoding="utf-8"))
        for check in LABELLED_CHECKS:
            value = check.value(labelled)
            found = re.search(check.pattern(value), text) is not None
            print(f"{check.label:34s} {value:16s} {'ok' if found else 'MISSING'}")
            if not found:
                failures.append(f"  {check.label}: expected {value!r} in "
                                f"context /{check.pattern(value)}/")
    elif "sec:labelled" in text:
        failures.append(
            "  the manuscript contains the labelled subsection but "
            f"{args.labelled} is missing, so none of its numbers are checked. "
            "Run scripts/build_labelled_tables.py.")

    if args.coverage.exists():
        cov = json.loads(args.coverage.read_text(encoding="utf-8"))
        for check in COVERAGE_CHECKS:
            value = check.value(cov)
            found = re.search(check.pattern(value), text) is not None
            print(f"{check.label:34s} {value:16s} {'ok' if found else 'MISSING'}")
            if not found:
                failures.append(f"  {check.label}: expected {value!r} in "
                                f"context /{check.pattern(value)}/")

    # The utility figure's caption is the only place the degenerate SHARE of a
    # loss is quoted, and it comes from the paired blocks rather than from the
    # labelled summary, so it needs its own source.
    if args.matrix.exists():
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        for check in MATRIX_CHECKS:
            value = check.value(matrix)
            found = re.search(check.pattern(value), text) is not None
            print(f"{check.label:34s} {value:16s} {'ok' if found else 'MISSING'}")
            if not found:
                failures.append(f"  {check.label}: expected {value!r} in "
                                f"context /{check.pattern(value)}/")

    print()
    for unique in UNIQUE_CHECKS:
        value = unique.value(stats)
        found = re.findall(unique.pattern, text)
        # Compared on the digit-normalised form, so "three of four" and
        # "3 of 4" are one statement rather than a false contradiction.
        wrong = sorted({f for f in found if _digits(f) != _digits(value)})
        status = ("NOT STATED" if not found
                  else f"CONTRADICTED by {wrong}" if wrong
                  else f"ok ({len(found)}x)")
        print(f"{unique.label:34s} {value:16s} {status}")
        if not found:
            failures.append(f"  {unique.label}: no occurrence matched "
                            f"/{unique.pattern}/; the claim may have been "
                            "reworded out of the checker's reach")
        elif wrong:
            failures.append(
                f"  {unique.label}: the paper states {wrong} as well as "
                f"{value!r} for one quantity. Both may be real numbers from "
                "different estimands, which is worse than a typo, not better.")

    print()
    for doc in DOC_CHECKS:
        if not doc.path.exists():
            continue
        value = doc.value(stats)
        body = doc.path.read_text(encoding="utf-8")
        found = re.search(doc.pattern(value), body) is not None
        print(f"{doc.label:34s} {value:16s} {'ok' if found else 'MISSING'}")
        if not found:
            failures.append(f"  {doc.label}: expected {value!r} in "
                            f"{doc.path} /{doc.pattern(value)}/")

    if failures:
        print("\nprose disagrees with the measurements:")
        print("\n".join(failures))
        return 1
    # The labelled group is conditional, so the total is computed rather than
    # written down -- a fixed number would quietly under-report the moment
    # round 3's measurements were present.
    n_checked = (len(CHECKS) + len(DATA_CHECKS) + len(DOC_CHECKS)
                 + (len(LABELLED_CHECKS) if args.labelled.exists() else 0)
                 + (len(MATRIX_CHECKS) if args.matrix.exists() else 0)
                 + (len(COVERAGE_CHECKS) if args.coverage.exists() else 0))
    print(f"\nall {n_checked} "
          f"quoted quantities match {args.stats} in context, and "
          f"{len(UNIQUE_CHECKS)} of them are stated consistently everywhere "
          "they appear")
    return 0


if __name__ == "__main__":
    sys.exit(main())
