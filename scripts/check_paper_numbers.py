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


CHECKS: tuple[Check, ...] = (
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
          lambda v: _rx(rf"Only {v} of the 32 contain a refusal marker")),
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
    Check("probe band span",
          lambda s: (lambda vals: f"{100 * (max(vals) - min(vals)):.1f}")(
              [r["retained_mean"] for r in s["probe"]["Qwen2.5-3B"]["rows"]
               if 4.5 <= r["bits"] <= 8.5]),
          lambda v: _rx(rf"probe moves by {v} points")),
    Check("Qwen 5.5 retention",
          lambda s: f"{100 * _probe_row(s, 'Qwen2.5-3B', 5.5)['retained_mean']:.1f}",
          lambda v: _rx(rf"sits at {v}") + r"\\?%\s+at 5.5 bits"),
    Check("Phi 3.5 retention",
          lambda s: f"{100 * _probe_row(s, 'Phi-3.5-mini', 3.5)['retained_mean']:.0f}",
          lambda v: _rx(rf"fallen to {v}") + r"\\?%"),
    Check("Qwen 3.5 retention",
          lambda s: f"{100 * _probe_row(s, 'Qwen2.5-3B', 3.5)['retained_mean']:.0f}",
          lambda v: _rx(rf"Qwen2.5-3B to {v}") + r"\\?%"),
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


def live_tex(source: str) -> str:
    """Strip what the reader never sees, so a check cannot be satisfied by it.

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
    args = ap.parse_args()

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
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
    print(f"\nall {len(CHECKS) + len(DATA_CHECKS) + len(DOC_CHECKS)} "
          f"quoted quantities match "
          f"{args.stats} in context")
    return 0


if __name__ == "__main__":
    sys.exit(main())
