"""How many prompts did each external grader actually pair, and does the
direction survive on the intersection they all covered?

The judge-agreement table reports `n_compared`, the number of completions a
grader returned. That is not the quantity a paired test uses. A paired McNemar
needs prompts graded at BOTH ends -- full precision and the quantized rung --
and the API graders did not all return the same prompts, or the same number of
them, or both ends for every model. Reporting one number that looks like a
sample size while the test consumes a different, smaller one invites exactly the
question a reviewer should ask.

So this reports three counts per grader, which is what the paper should have
said in the first place:

    returned    parsable verdicts the grader produced, across both ends
    paired      prompts it graded at BOTH ends -- the matched sample
    common      prompts EVERY qualifying grader paired, the intersection

`paired` is the matched sample, not literally the exact-McNemar denominator:
that test conditions on the discordant pairs alone, which here are 15, 3, 19, 9
and 1. The matched sample is what a reader needs to size the comparison, and it
is the quantity the agreement table's `n_compared` is most likely to be mistaken
for. `n_compared` is a third number again -- it counts cache entries including
parse failures, so Llama-3.3-70B on Qwen2.5-3B reads 580 there, 508 here, and
pairs 26.

and then re-runs the direction comparison twice: once on each grader's own
paired set, and once restricted to `common`, where every grader sees an
identical prompt set and coverage cannot explain a disagreement.

That last analysis is the point. The four grader comparisons disagree about the
direction three-to-one, and unequal coverage is a live alternative explanation
for the odd one out. Testing on the intersection removes it -- or fails to,
which would be worth knowing.

Usage:
  python scripts/grader_coverage.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.review_reanalysis import exact_mcnemar

# The rung the external graders were pointed at. Round two re-graded one rung,
# not the ladder, so this is the only paired comparison they support.
RUNG = "RTN_4B"
BASELINE = "FP16"

RUNS = {
    "Qwen2.5-3B": "20260803-165047_0d3f097_colab-behavioural-qwen3b",
    "Phi-3.5-mini": "20260803-174622_0d3f097_colab-behavioural-phi35",
}
# A verdict outside the label set is not a grading, it is a parse failure, and
# it must not be paired with anything.
VALID = {"REFUSE", "COMPLY", "UNCLEAR"}


def load_cache(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items() if v in VALID}


def graders_in(results: Path) -> list[str]:
    names = set()
    for path in results.glob("judge_api_*_FP16.json"):
        names.add(re.sub(r"^judge_api_|_FP16$", "", path.stem))
    return sorted(names)


def coverage(results: Path, grader: str) -> dict[str, Any]:
    base = load_cache(results / f"judge_api_{grader}_{BASELINE}.json")
    rung = load_cache(results / f"judge_api_{grader}_{RUNG}.json")
    paired = sorted(set(base) & set(rung))
    return {
        "grader": grader,
        "returned": len(base) + len(rung),
        "at_baseline": len(base),
        "at_rung": len(rung),
        "paired": len(paired),
        "paired_ids": paired,
        "base": base,
        "rung": rung,
    }


def transitions(cov: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    """Refusal-direction transitions over exactly `ids`."""
    to_refusal = sum(1 for i in ids
                     if cov["base"][i] == "COMPLY" and cov["rung"][i] == "REFUSE")
    to_compliance = sum(1 for i in ids
                        if cov["base"][i] == "REFUSE" and cov["rung"][i] == "COMPLY")
    return {
        "n": len(ids),
        "to_refusal": to_refusal,
        "to_compliance": to_compliance,
        "direction": ("toward refusal" if to_refusal > to_compliance
                      else "toward compliance" if to_compliance > to_refusal
                      else "tie"),
        "p": exact_mcnemar(to_refusal, to_compliance),
    }


def local_judge(results: Path) -> dict[str, dict[int, str]] | None:
    """The 7B judge's own per-prompt verdicts, keyed like the API caches.

    Stored as a positional list, one entry per prompt in corpus order, while
    the API caches are dicts keyed by that same index. Converting to the dict
    form lets both be restricted to an identical prompt subset.
    """
    hits = sorted(p for p in results.glob(f"judge_*_{BASELINE}.json")
                  if "_api_" not in p.name and "collapsed" not in p.name)
    if not hits:
        return None
    fingerprint = hits[0].stem[len("judge_"):-len(f"_{BASELINE}")]
    out: dict[str, dict[int, str]] = {}
    for end in (BASELINE, RUNG):
        path = results / f"judge_{fingerprint}_{end}.json"
        if not path.exists():
            return None
        verdicts = json.loads(path.read_text(encoding="utf-8"))
        out[end] = {i: v for i, v in enumerate(verdicts) if v in VALID}
    return out


def matched_local(local: dict[str, dict[int, str]],
                  ids: list[int]) -> dict[str, Any] | None:
    """What the 7B judge says about exactly the prompts an API grader paired.

    This is the comparison that separates the two explanations the paper has
    been conflating. A difference between the 7B judge over 500 prompts and an
    external grader over the 200 it happened to return is a difference of
    grader AND of sample. Restricting the 7B judge to those same 200 removes
    the sample, and whatever remains is the grader.
    """
    usable = [i for i in ids if i in local[BASELINE] and i in local[RUNG]]
    if not usable:
        return None
    shim = {"base": local[BASELINE], "rung": local[RUNG]}
    return transitions(shim, usable)


def analyse(results: Path) -> dict[str, Any] | None:
    covs = {g: coverage(results, g) for g in graders_in(results)}
    covs = {g: c for g, c in covs.items() if c["paired"]}
    if not covs:
        return None
    local = local_judge(results)

    # The intersection every grader that paired anything also paired.
    common = set.intersection(*(set(c["paired_ids"]) for c in covs.values()))
    common_ids = sorted(common)

    rows = []
    for g, c in covs.items():
        rows.append({
            "grader": g,
            "returned": c["returned"],
            "at_baseline": c["at_baseline"],
            "at_rung": c["at_rung"],
            "paired": c["paired"],
            "own": transitions(c, c["paired_ids"]),
            "common": transitions(c, common_ids) if common_ids else None,
            # The 7B judge over the SAME prompts, so the two rows differ by
            # grader alone.
            "local_on_same": (matched_local(local, c["paired_ids"])
                              if local else None),
        })
    return {"rows": rows, "n_common": len(common_ids),
            "local_all": (matched_local(local, sorted(local[BASELINE]))
                          if local else None)}


def report(model: str, block: dict[str, Any]) -> None:
    print("=" * 78)
    print(f"{model}   baseline {BASELINE} vs {RUNG}")
    print("=" * 78)
    print(f"{'grader':34s} {'ret':>5s} {'@fp16':>6s} {'@rung':>6s} "
          f"{'PAIRED':>7s}")
    print("-" * 62)
    for r in block["rows"]:
        print(f"{r['grader']:34s} {r['returned']:5d} {r['at_baseline']:6d} "
              f"{r['at_rung']:6d} {r['paired']:7d}")
    print(f"\ncommon intersection: {block['n_common']} prompts\n")

    print(f"{'grader':34s} | {'on its own paired set':^30s} | "
          f"{'on the common set':^30s}")
    print(f"{'':34s} | {'n':>5s} {'->ref':>5s} {'->cmp':>5s} {'p':>6s} | "
          f"{'n':>5s} {'->ref':>5s} {'->cmp':>5s} {'p':>6s}")
    print("-" * 100)
    for r in block["rows"]:
        o, c = r["own"], r["common"]
        line = (f"{r['grader']:34s} | {o['n']:5d} {o['to_refusal']:5d} "
                f"{o['to_compliance']:5d} {o['p']:6.3f} | ")
        line += (f"{c['n']:5d} {c['to_refusal']:5d} {c['to_compliance']:5d} "
                 f"{c['p']:6.3f}" if c else f"{'--':>5s}")
        print(line)

    # Coverage held fixed: the same prompts, scored by the external grader and
    # by the 7B judge. Any remaining difference is the grader, and only this
    # column supports a statement of the form "grader choice moves the result".
    print(f"\n{'grader':34s} | {'external':^19s} | {'7B on the SAME prompts':^24s}")
    print(f"{'':34s} | {'n':>5s} {'->ref':>5s} {'->cmp':>5s} | "
          f"{'n':>5s} {'->ref':>5s} {'->cmp':>5s} {'p':>6s}")
    print("-" * 86)
    for r in block["rows"]:
        o, m = r["own"], r["local_on_same"]
        line = (f"{r['grader']:34s} | {o['n']:5d} {o['to_refusal']:5d} "
                f"{o['to_compliance']:5d} | ")
        line += (f"{m['n']:5d} {m['to_refusal']:5d} {m['to_compliance']:5d} "
                 f"{m['p']:6.3f}" if m else f"{'--':>5s}")
        print(line)
    if block.get("local_all"):
        a = block["local_all"]
        print(f"{'7B judge, full coverage':34s} | {'':19s} | "
              f"{a['n']:5d} {a['to_refusal']:5d} {a['to_compliance']:5d} "
              f"{a['p']:6.3f}")

    own_dirs = [r["own"]["direction"] for r in block["rows"]]
    com_dirs = [r["common"]["direction"] for r in block["rows"] if r["common"]]
    print(f"\n  own paired sets : {own_dirs.count('toward refusal')} of "
          f"{len(own_dirs)} point toward refusal")
    if com_dirs:
        print(f"  common set      : {com_dirs.count('toward refusal')} of "
              f"{len(com_dirs)} point toward refusal")
    print()


GRADER_LABEL = {
    "agent_claude": "Claude",
    "agent_haiku": "Claude Haiku",
    "groq_llama-3-3-70b-versatile": "Llama-3.3-70B",
    "gemini_gemini-2-0-flash": "Gemini 2.0 Flash",
    "nvidia_meta-llama-3-1-70b-instruct": "Llama-3.1-70B",
}


def latex_table(payload: dict[str, Any]) -> str:
    """Returned against paired, and the direction on each grader's own set.

    The two count columns are the whole point of the table: a reader who sees
    only `returned` will size the paired test by it, and be wrong by a factor
    of up to twenty.
    """
    out = [r"\begin{tabular}{llrrrrl}", r"\toprule",
           r"grader & model & returned & \textbf{paired} & $\to$refuse "
           r"& $\to$comply & direction \\",
           r"\midrule"]
    for model, block in payload.items():
        for r in block["rows"]:
            own = r["own"]
            out.append(
                f"{GRADER_LABEL.get(r['grader'], r['grader'])} & {model} & "
                f"{r['returned']} & \\textbf{{{r['paired']}}} & "
                f"{own['to_refusal']} & {own['to_compliance']} & "
                f"{own['direction']} \\\\")
        out.append(r"\addlinespace")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    return "\n".join(out) + "\n"


def latex_matched(payload: dict[str, Any]) -> str:
    """Each external grader beside the 7B judge on the SAME prompts.

    The table that separates the two explanations. Reading the external column
    against the 7B judge's full-coverage result confounds grader with sample;
    reading it against the adjacent column does not, because both scored the
    same prompts.
    """
    out = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"& & & \multicolumn{2}{c}{external grader} "
           r"& \multicolumn{3}{c}{7B judge, same prompts} \\",
           r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
           r"grader & model & paired & $\to$ref & $\to$cmp "
           r"& $\to$ref & $\to$cmp & $p$ \\",
           r"\midrule"]
    for model, block in payload.items():
        if not isinstance(block, dict) or "rows" not in block:
            continue
        for r in block["rows"]:
            matched = r.get("local_on_same")
            if not matched:
                continue
            own = r["own"]
            out.append(
                f"{GRADER_LABEL.get(r['grader'], r['grader'])} & {model} & "
                f"{own['n']} & {own['to_refusal']} & {own['to_compliance']} & "
                f"{matched['to_refusal']} & {matched['to_compliance']} & "
                f"{matched['p']:.3f} \\\\")
        full = block.get("local_all")
        if full:
            out.append(
                f"\\emph{{7B, full coverage}} & {model} & {full['n']} & "
                f"--- & --- & {full['to_refusal']} & {full['to_compliance']} & "
                f"{full['p']:.3f} \\\\")
        out.append(r"\addlinespace")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="artifacts/runs")
    ap.add_argument("--out", type=Path,
                    default=Path("docs/paper/grader_coverage.json"))
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    payload: dict[str, Any] = {}
    for model, run in RUNS.items():
        results = repo / args.runs / run / "results"
        if not results.exists():
            continue
        block = analyse(results)
        if block:
            payload[model] = {
                "n_common": block["n_common"],
                # The 7B judge over its whole 500. Carried so a reader can see
                # what the matched columns are being compared against, and why
                # the Qwen gap between 32/11 and 8/5 is coverage rather than
                # grader.
                "local_all": block.get("local_all"),
                "rows": [{k: v for k, v in r.items()} for r in block["rows"]],
            }
            report(model, block)

    if not payload:
        print("no external-grader caches found.")
        return 1

    print("`returned` is what the agreement table reports; `paired` is what the "
          "test consumes.\nThe gap between them is the reviewer's question, and "
          "on the common set no grader\nhas a coverage advantage over any other.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    table = repo / "docs/paper/tables/tab_grader_coverage.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(latex_table(payload), encoding="utf-8")
    print(f"wrote {table}")

    matched = repo / "docs/paper/tables/tab_coverage_matched.tex"
    matched.write_text(latex_matched(payload), encoding="utf-8")
    print(f"wrote {matched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
