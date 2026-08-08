"""LaTeX table bodies for the labelled-suite section of the manuscript.

Only the quantities the paper actually argues from are emitted here. The
labelled runs measure a good deal more than this -- a full five-way transition
matrix, a benign-arm capability collapse, an exploratory leakage queue -- and
most of it either duplicates a section the paper already has or opens a question
the paper does not answer. Putting those in the manuscript would pad it, so they
stay in the local write-up and out of here.

What survives the filter is the two things that bear on the paper's own thesis:

  1. how differently one refusal-phrase list reads three model families, which
     is the paper's indictment of that instrument extended to a family it was
     never tuned on
  2. what the judge that replaces it does with a 48-token completion, which is
     the same indictment turned on this paper's own instrument

Usage:
  python scripts/build_labelled_tables.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_behavioural_ladder import has_refusal_marker

RUNS = {
    "Qwen2.5-3B": "20260808-154229_069d48d_lab-qwen3b-xstest",
    "Phi-3.5-mini": "20260808-162254_069d48d_lab-phi35-xstest",
    "SmolLM2-1.7B": "20260808-164828_069d48d_lab-smol17-xstest",
}
# Token counts are only meaningful under the tokenizer that produced the text.
# Measuring all three with Qwen's reported Phi-3.5 as 0% truncated when it is
# 100%, so the mapping is explicit rather than assumed.
TOKENIZER_OF = {
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}
DECLINING = ("refusal", "deflection", "disclaimer")


def measure(results: Path) -> dict[str, Any]:
    manifest = json.loads((results.parent / "manifest.json").read_text(
        encoding="utf-8"))
    tax = json.loads((results / "completion_taxonomy.json").read_text(
        encoding="utf-8"))
    harm = json.loads((results / "prompts.json").read_text(
        encoding="utf-8"))["harm_label"]
    labels = tax["resolved"]["FP16"]
    completions = json.loads((results / "completions_FP16.json").read_text(
        encoding="utf-8"))["completions"]

    declining = {i for i, lab in enumerate(labels) if lab in DECLINING}
    flagged = {i for i, c in enumerate(completions) if has_refusal_marker(c)}

    # Recall alone would presuppose the judge is the standard. Precision and the
    # miss count do not: they say the two label sets NEST, and that the phrase
    # list's shortfall is one-directional and family-dependent. That claim
    # survives without assuming the judge is right, so it is what the paper
    # argues from and it is recorded here beside the recall it qualifies.
    n = len(labels)
    tp = len(flagged & declining)
    fp = len(flagged - declining)
    fn = len(declining - flagged)
    tn = n - tp - fp - fn
    observed = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2

    def cell(cls: str, label: str) -> int:
        return sum(1 for lab, h in zip(labels, harm) if h == cls and lab == label)

    return {
        # Provenance travels with the numbers. Without this the stats file names
        # no run, and a reader tracing a paper claim back to a run directory has
        # to guess from the model id.
        "run": results.parent.name,
        "model_id": manifest.get("model_id"),
        "judge_model": tax.get("judge_model"),
        "judge_loaded_in_4bit": tax.get("judge_loaded_in_4bit"),
        "max_new_tokens": manifest.get("max_new_tokens"),
        "n_prompts": len(harm),
        "schemes": tax.get("schemes"),
        "marker": len(flagged),
        "declines": len(declining),
        "recall": tp / len(declining) if declining else None,
        "precision": tp / len(flagged) if flagged else None,
        "missed": fn,
        # Reported so its absence cannot be read as concealment. Under a 92-94%
        # base rate it is dominated by prevalence and reads as near-total
        # disagreement between two scorers that never contradict each other,
        # which is why the paper does not lead with it.
        "kappa": ((observed - expected) / (1 - expected)
                  if expected != 1 else None),
        "harmful_comply": cell("harmful", "compliance"),
        "harmful_deflect": cell("harmful", "deflection"),
        "benign_comply": cell("benign", "compliance"),
        "benign_deflect": cell("benign", "deflection"),
        # The refusal column is not in the table, which reports only the two
        # cells the safety question turns on. It is quoted in the caption of
        # the full-precision matrix figure, so it is emitted here rather than
        # read off a rendered PNG.
        "harmful_refuse": cell("harmful", "refusal"),
        "benign_refuse": cell("benign", "refusal"),
    }


def at_cap_by_scheme(results: Path, tokenizer_id: str, budget: int,
                     cached: dict[str, float] | None = None) -> dict[str, float]:
    """Fraction of completions that ran into the budget, per scheme.

    Stored rather than recomputed at plot time so that the figure and the prose
    read the same number. An earlier version of the figure used a character
    threshold as a stand-in for token length; it disagreed with the prose by
    tens of points at the collapsed rungs, which is exactly the drift these
    generated artefacts exist to prevent.

    This is the one quantity here that needs each model's own tokenizer, and so
    the one that cannot be recomputed in an environment without `transformers`
    -- which includes the machine this manuscript is usually edited on. Rather
    than fail the whole build, or worse substitute a character proxy, reuse the
    value already recorded in the stats file and say so. Refusing to carry a
    cached value forward is only correct if the alternative is recomputing it;
    here the alternative is having no number at all.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        if cached is None:
            raise
        print("  ! transformers unavailable; keeping the recorded "
              "at_cap_by_scheme for this model")
        return cached

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    out: dict[str, float] = {}
    for path in sorted(results.glob("completions_*.json")):
        scheme = path.stem.removeprefix("completions_")
        texts = json.loads(path.read_text(encoding="utf-8"))["completions"]
        n = sum(1 for t in texts
                if len(tok.encode(t, add_special_tokens=False)) >= budget - 1)
        out[scheme] = n / len(texts)
    return out


def table(rows: dict[str, dict[str, Any]]) -> str:
    """One table carrying both halves of the argument.

    The left block is the two scorers disagreeing; the right block is what the
    judge did with prompts of each class. The harmful-compliance column is the
    one the reader is meant to stop at: it is zero everywhere, which is why the
    paired safety statistic on these runs is a property of the instrument.
    """
    out = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
           r"& \multicolumn{3}{c}{scorers disagree}"
           r"& \multicolumn{2}{c}{harmful (150)}"
           r"& \multicolumn{2}{c}{benign (150)} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
           r"model & phrase & judge & recall & deflect & comply "
           r"& deflect & comply \\",
           r"\midrule"]
    for name, m in rows.items():
        recall = "--" if m["recall"] is None else f"{100 * m['recall']:.1f}"
        out.append(
            f"{name} & {m['marker']} & {m['declines']} & {recall} & "
            f"{m['harmful_deflect']} & \\textbf{{{m['harmful_comply']}}} & "
            f"{m['benign_deflect']} & {m['benign_comply']} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="artifacts/runs")
    ap.add_argument("--out", type=Path, default=Path("docs/paper/tables"))
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    missing = [m for m, r in RUNS.items()
               if not (repo / args.runs / r / "results").exists()]
    if missing:
        raise SystemExit(
            f"no run directory for {missing}. This table is built from the "
            "labelled XSTest runs; unzip the Colab archive at the repository "
            "root first.")

    # Read back before overwriting, so the tokenizer-dependent measurement can
    # survive a rebuild on a machine that cannot recompute it.
    stats_path = repo / "docs/paper/labelled_paper_stats.json"
    previous: dict[str, Any] = {}
    if stats_path.exists():
        previous = json.loads(stats_path.read_text(encoding="utf-8"))

    rows = {}
    for name, run in RUNS.items():
        results = repo / args.runs / run / "results"
        row = measure(results)
        row["at_cap_by_scheme"] = at_cap_by_scheme(
            results, TOKENIZER_OF[name], row["max_new_tokens"],
            cached=previous.get(name, {}).get("at_cap_by_scheme"))
        rows[name] = row
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tab_labelled.tex").write_text(table(rows), encoding="utf-8")
    print(f"  {args.out / 'tab_labelled.tex'}")

    # The same numbers as JSON, so `check_paper_numbers.py` can verify the prose
    # against them rather than against a value retyped into the manuscript.
    stats_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
