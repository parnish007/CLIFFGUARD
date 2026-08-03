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
    Check("Qwen 4.5 Holm p",
          lambda s: f"{_row(s, 'Qwen2.5-3B', 4.5)['mcnemar_p_holm']:.3f}",
          lambda v: _rx(rf"Holm-adjusted \$p={v}\$")),
    Check("Phi 4.5 to-refuse count",
          lambda s: str(_row(s, "Phi-3.5-mini", 4.5)["to_refusal"]),
          lambda v: _rx(rf"Phi-3.5-mini (?:shows )?{v} against")),
    Check("Phi 4.5 Holm p",
          lambda s: f"{_row(s, 'Phi-3.5-mini', 4.5)['mcnemar_p_holm']:.3f}",
          lambda v: _rx(rf"Phi-3\.5-mini 21 against 4 \(\$p={v}\$\)")),
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
    Check("GSM8K Holm p",
          lambda s: f"{_gsm_row(s, 'Qwen2.5-3B', 4.5)['mcnemar_p_holm']:.3f}",
          lambda v: _rx(rf"p_{{\\text{{Holm}}}}={v}\$")),
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", type=Path,
                    default=Path("docs/paper/cliff_artifact.tex"))
    ap.add_argument("--stats", type=Path,
                    default=Path("docs/paper/review_stats.json"))
    args = ap.parse_args()

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    text = args.tex.read_text(encoding="utf-8")

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

    if failures:
        print("\nprose disagrees with the measurements:")
        print("\n".join(failures))
        return 1
    print(f"\nall {len(CHECKS)} quoted quantities match {args.stats} in context")
    return 0


if __name__ == "__main__":
    sys.exit(main())
