"""Answer the four questions the round-2 runs were generated to answer.

`review_reanalysis.py` fits a ladder. It assumes one behavioural run per model
spanning enough rungs to regress on, and it refuses to proceed when two runs
describe the same model, because which one won would otherwise depend on
directory order. Both assumptions are right for round one and wrong for round
two, which produces:

  r2-behavioural-qwen15b   a full 7-rung ladder, the model round one excluded
  r2-long256-<model>       FP16 + two rungs at a 256-token budget
  r2-deployed-<model>      FP16 + AWQ + GPTQ, no RTN rungs at all
  r2-gsm8k-<model>         the capability ladder at 496 questions

Two of those describe Qwen2.5-3B, so review_reanalysis stops before it starts;
the deployed run has no ordinal axis to fit; and a slope through two rungs is
not a slope. None of that is a defect in either script. These runs pose paired
comparisons, not dose-response questions, so this script runs the paired
machinery -- exact McNemar, Holm, exact one-sided bounds -- and keys everything
by run LABEL rather than by model, which is what makes two runs of one model a
normal situation instead of an ambiguity.

The four questions, and what answers each:

  1. Was excluding Qwen2.5-1.5B from the refusal arm the right call, or an
     artifact of grading it with a 1.5B self-judge that returned REFUSE for
     100% of full-precision completions?          -> transitions, full ladder
  2. Does the contested Qwen2.5-3B 4.5-bit capability drop survive more
     questions? At n=200 it was p=0.020 raw, 0.100 within model.  -> gsm8k
  3. Is the refusal shift an artifact of a 48-token budget cutting answers off
     before they commit?              -> the same rungs at 256 tokens, paired
                                         against the 48-token run directly
  4. Does the direction hold under a quantizer anyone deploys?  -> AWQ / GPTQ

Usage:
  python scripts/analyse_round2.py --gsm8k <gsm8k test.jsonl>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_paper_data import MODEL_LABELS, bits_of, clopper_pearson
from scripts.reanalyse_runs import load_run, select_runs
from scripts.review_reanalysis import (
    clopper_pearson_upper,
    exact_mcnemar,
    holm,
    label_matrix,
    transition_matrix,
)
from scripts.run_sector_ladder import extract_gold, is_correct


def describe(run: dict[str, Any]) -> dict[str, Any]:
    """Identity of a run, from its own manifest rather than from its name."""
    manifest = run["manifest"]
    model = manifest.get("model_id", "?")
    return {
        "label": manifest.get("label", run["path"].name),
        "run": run["path"].name,
        "model": MODEL_LABELS.get(model, model),
        "model_id": model,
        "n_prompts": int(manifest.get("n_prompts", 0)),
        "max_new_tokens": int(manifest.get("max_new_tokens", 0)),
        "schemes": list(manifest.get("schemes")
                        or ["FP16"] + [f"RTN_{b}B" for b in manifest.get("bits", [])]),
        "deployed": manifest.get("deployed", {}),
    }


def paired_rows(labels: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Every scheme against this run's own FP16, paired, with the full 3x3.

    Holm runs over the rungs of this run only. That is the right family here:
    each round-2 run is one question asked once, not a ladder swept for the
    smallest p-value.
    """
    base = labels["FP16"]
    n = len(base)
    rows, raw_p = [], []
    for scheme, cur in labels.items():
        if scheme == "FP16":
            continue
        matrix = transition_matrix(base, cur)
        to_compliance = matrix["refusal"]["compliance"]
        to_refusal = matrix["compliance"]["refusal"]
        gradable = int((np.isin(base, ["refusal", "compliance"])
                        & np.isin(cur, ["refusal", "compliance"])).sum())
        p = exact_mcnemar(to_compliance, to_refusal)
        rows.append({
            "scheme": scheme, "bits": bits_of(scheme),
            "on_rtn_axis": bool(np.isfinite(bits_of(scheme))),
            "to_compliance": to_compliance, "to_refusal": to_refusal,
            "discordant": to_compliance + to_refusal,
            "n_gradable": gradable, "n_itt": n,
            "refusal_rate": float((cur == "refusal").mean()),
            "degenerate_rate": float((cur == "degenerate").mean()),
            "rate_itt": to_compliance / n,
            "upper95_itt": clopper_pearson_upper(to_compliance, n),
            "mcnemar_p": p, "matrix": matrix,
        })
        raw_p.append(p)
    for row, adjusted in zip(rows, holm(raw_p)):
        row["mcnemar_p_holm_within_run"] = adjusted
    return rows


def behavioural(runs: list[Path]) -> dict[str, Any]:
    """Every behavioural run, keyed by label so two runs of one model coexist."""
    out: dict[str, Any] = {}
    for run_dir in runs:
        run = load_run(run_dir)
        if run is None or not run["judge_raw"]:
            continue
        info = describe(run)
        if info["n_prompts"] < 150:
            print(f"  [skip] {info['label']}: n={info['n_prompts']} is a smoke run")
            continue
        if info["label"] in out:
            raise SystemExit(
                f"two runs carry the label {info['label']!r} "
                f"({out[info['label']]['run']} and {info['run']}). Labels key every "
                "comparison below; delete the older directory or relabel it.")
        info["rows"] = paired_rows(label_matrix(run, "composite"))
        info["fp16_refusal_rate"] = float(
            (label_matrix(run, "composite")["FP16"] == "refusal").mean())
        out[info["label"]] = info
    return out


def token_budget(blocks: dict[str, Any], runs: list[Path]) -> dict[str, Any]:
    """Does the refusal shift survive a longer generation budget?

    48 new tokens is enough for a refusal but often not for substantive
    compliance, so the design is biased toward observing refusal intact. The
    answer is not a 256-token run read on its own -- it is the SAME rungs and the
    SAME prompts at both budgets, compared directly.

    The prompt lists are verified equal rather than assumed equal. Both runs
    rebuild the corpus from the same files with the same count, so they should
    match; if they ever do not, the pairing is meaningless and comparing the two
    would silently pair prompt i against a different prompt i.
    """
    by_run = {}
    for run_dir in runs:
        run = load_run(run_dir)
        if run is not None and run["judge_raw"]:
            by_run[describe(run)["label"]] = run

    out: dict[str, Any] = {}
    for label, block in blocks.items():
        if block["max_new_tokens"] <= 48:
            continue
        rungs = {r["scheme"] for r in block["rows"]}
        # Sharing the model, the prompt count and a shorter budget is not
        # enough: the deployed run satisfies all three and has no RTN rung in
        # common, so pairing against it would compare an empty set of schemes
        # and report that as an answer. The baseline must share a rung.
        candidates = [
            other for other in blocks.values()
            if other["model_id"] == block["model_id"]
            and other["max_new_tokens"] <= 48
            and other["n_prompts"] == block["n_prompts"]
            and other["label"] != label
            and rungs & {r["scheme"] for r in other["rows"]}
        ]
        if len(candidates) != 1:
            out[label] = {
                "comparable_baseline": None,
                "candidates": [c["label"] for c in candidates],
                "note": (
                    "no 48-token run of this model at the same prompt count shares "
                    "a rung with it; the truncation question needs exactly one"
                    if not candidates else
                    f"{len(candidates)} runs qualify as the 48-token baseline "
                    f"({[c['label'] for c in candidates]}); which one is used cannot "
                    "be left to directory order"),
            }
            continue
        short = candidates[0]
        long_prompts = json.loads(
            (Path(by_run[label]["path"]) / "results/prompts.json")
            .read_text(encoding="utf-8"))["prompts"]
        short_prompts = json.loads(
            (Path(by_run[short["label"]]["path"]) / "results/prompts.json")
            .read_text(encoding="utf-8"))["prompts"]
        if long_prompts != short_prompts:
            out[label] = {"comparable_baseline": short["label"],
                          "note": "PROMPT LISTS DIFFER; the two runs are not paired"}
            continue

        shared = [r["scheme"] for r in block["rows"]
                  if r["scheme"] in {s["scheme"] for s in short["rows"]}]
        short_run = by_run.get(short["label"])
        long_run = by_run.get(label)
        rows: list[dict[str, Any]] = []
        for scheme in shared:
            a = next(r for r in short["rows"] if r["scheme"] == scheme)
            b = next(r for r in block["rows"] if r["scheme"] == scheme)
            rows.append({
                "scheme": scheme,
                "tokens_48": {"to_refusal": a["to_refusal"],
                              "to_compliance": a["to_compliance"],
                              "p": a["mcnemar_p"]},
                "tokens_256": {"to_refusal": b["to_refusal"],
                               "to_compliance": b["to_compliance"],
                               "p": b["mcnemar_p"]},
                # Kept, but demoted. Two aggregates pointing the same way is a
                # weak statement and it used to be the whole answer here.
                "same_direction": (b["to_refusal"] > b["to_compliance"])
                                  == (a["to_refusal"] > a["to_compliance"]),
                "per_prompt": (per_prompt_budget(short_run, long_run, scheme)
                               if short_run and long_run else None),
            })
        out[label] = {"comparable_baseline": short["label"],
                      "baseline_run": short["run"], "rows": rows,
                      "n_prompts": block["n_prompts"],
                      "flip_status": (flip_status_budget(short_run, long_run,
                                                         shared)
                                      if short_run and long_run else None),
                      "note": f"{len(rows)} shared rung(s): {shared}"}
    return out


def per_prompt_budget(short_run: dict[str, Any], long_run: dict[str, Any],
                      scheme: str) -> dict[str, Any]:
    """How one scheme's own verdicts move when the window widens.

    The comparison this file used to make was between two AGGREGATES: it took
    each run's own FP16-versus-rung imbalance and asked whether the two pointed
    the same way. That is not a test of the truncation question. It cannot see
    a prompt whose verdict changed, only whether two totals happened to keep
    their sign, and two totals can agree while every prompt underneath them
    moves.

    This asks the question directly, on the same prompt at the same rung under
    two budgets, and reports the full transition table rather than a summary,
    because which way the changes go is the interesting part.
    """
    short_v = short_run["judge_raw"].get(scheme)
    long_v = long_run["judge_raw"].get(scheme)
    if not short_v or not long_v or len(short_v) != len(long_v):
        return {"note": "verdicts absent or different lengths; not comparable"}

    table: dict[str, int] = {}
    for a, b in zip(short_v, long_v):
        table[f"{a}->{b}"] = table.get(f"{a}->{b}", 0) + 1
    changed = sum(n for k, n in table.items() if k.split("->")[0] != k.split("->")[1])

    # The two directions that matter, tested against each other. A window that
    # simply adds noise moves prompts both ways in similar numbers; a window
    # that was hiding compliance moves them one way.
    to_comply = sum(1 for a, b in zip(short_v, long_v)
                    if a != "COMPLY" and b == "COMPLY")
    to_refuse = sum(1 for a, b in zip(short_v, long_v)
                    if a == "COMPLY" and b != "COMPLY")
    return {
        "n": len(short_v),
        "changed": changed,
        "changed_rate": changed / len(short_v),
        "became_comply": to_comply,
        "left_comply": to_refuse,
        "p": exact_mcnemar(to_comply, to_refuse),
        "transitions": dict(sorted(table.items(), key=lambda kv: -kv[1])),
    }


def flip_status_budget(short_run: dict[str, Any], long_run: dict[str, Any],
                       schemes: list[str]) -> dict[str, Any]:
    """Does a prompt's unsafe-flip status survive the wider window?

    This is the test the truncation question actually needs, and the one thing
    the previous analysis could not do even in principle. Each budget defines an
    unsafe flip against ITS OWN full-precision label, so if the window changes
    the FP16 verdict the two budgets are not scoring the same event -- a prompt
    can leave the long-window baseline without anything being reported.

    So two numbers are produced. First, how stable the baseline is: the count of
    prompts whose FP16 verdict itself changed with the budget. If that is large,
    nothing below it is interpretable and the honest answer is that the
    endpoint is budget-dependent. Second, the paired test on flip status
    itself, over the prompts where the baseline held.
    """
    rung = next((s for s in schemes if s != "FP16"), None)
    if rung is None or "FP16" not in schemes:
        return {"note": "needs FP16 and one rung in common"}
    for run in (short_run, long_run):
        if not all(run["judge_raw"].get(s) for s in ("FP16", rung)):
            return {"note": "verdicts missing for FP16 or the rung"}

    s_fp, s_rn = short_run["judge_raw"]["FP16"], short_run["judge_raw"][rung]
    l_fp, l_rn = long_run["judge_raw"]["FP16"], long_run["judge_raw"][rung]
    if not len({len(s_fp), len(s_rn), len(l_fp), len(l_rn)}) == 1:
        return {"note": "verdict lists differ in length; not paired"}

    baseline_moved = sum(1 for a, b in zip(s_fp, l_fp) if a != b)
    stable = [i for i in range(len(s_fp)) if s_fp[i] == l_fp[i]]

    def flip(fp: list[str], rn: list[str], i: int) -> bool:
        return fp[i] == "REFUSE" and rn[i] == "COMPLY"

    only_short = sum(1 for i in stable
                     if flip(s_fp, s_rn, i) and not flip(l_fp, l_rn, i))
    only_long = sum(1 for i in stable
                    if flip(l_fp, l_rn, i) and not flip(s_fp, s_rn, i))
    return {
        "rung": rung,
        "n": len(s_fp),
        "fp16_verdict_changed": baseline_moved,
        "fp16_verdict_changed_rate": baseline_moved / len(s_fp),
        "n_baseline_stable": len(stable),
        "flip_only_at_48": only_short,
        "flip_only_at_256": only_long,
        "p": exact_mcnemar(only_long, only_short),
        "note": ("the baseline itself moves with the budget, so a flip-status "
                 "comparison rests on shifting ground"
                 if baseline_moved > 0.1 * len(s_fp) else
                 "baseline stable enough for the flip comparison to mean "
                 "something"),
    }


def deployed(blocks: dict[str, Any]) -> dict[str, Any]:
    """FP16 against each pre-quantized checkpoint, paired.

    Reported entirely separately from the RTN rungs. These checkpoints have a
    bit budget and no position on the ordinal axis -- they change the
    quantization algorithm as well as the width -- so the comparison they support
    is "does the direction hold under a quantizer people ship", not "where on the
    curve does this sit".
    """
    out: dict[str, Any] = {}
    for label, block in blocks.items():
        rows = [r for r in block["rows"] if not r["on_rtn_axis"]]
        if not rows:
            continue
        for row in rows:
            meta = block["deployed"].get(row["scheme"], {})
            row["repo"] = meta.get("repo")
            row["quantization_config"] = meta.get("quantization_config", {})
        out[label] = {"model": block["model"], "run": block["run"], "rows": rows}
    return out


def gsm8k(runs: list[Path], gsm8k_path: Path, min_n: int) -> dict[str, Any]:
    """Paired exact McNemar per rung, at whatever question count the run used.

    Correctness is answer extraction alone with no degeneracy gate, matching
    review_reanalysis: gating here would censor on an outcome quantization
    causes.

    `min_n` decides the correction family, which is why it is not cosmetic. A
    smoke run's cells would otherwise enter the all-cells Holm adjustment and
    inflate every real p-value in it. So would round one's 200-question runs --
    and those are worse, because they measure the SAME model x rung cells this
    arm was regenerated to re-measure, so including both counts each cell twice.
    The default sits above 200 and below 496 for exactly that reason.
    """
    rows_by_run: dict[str, Any] = {}
    raw = gsm8k_path.read_bytes()
    all_golds = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        gold = extract_gold(json.loads(line)["answer"])
        if gold is not None:
            all_golds.append(gold)

    for run_dir in runs:
        sector = run_dir / "results" / "sector_gsm8k.json"
        if not sector.exists():
            continue
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        label = manifest.get("label", run_dir.name)
        n = int(manifest["n_problems"])
        if n < min_n:
            print(f"  [skip] {label}: n={n} is below --gsm8k-min-n={min_n}")
            continue
        if len(all_golds) < n:
            raise SystemExit(
                f"{label} scored {n} questions but the supplied file yields only "
                f"{len(all_golds)} with a parseable gold answer")
        golds = all_golds[:n]
        stored = json.loads(sector.read_text(encoding="utf-8"))
        correct = {}
        for scheme in stored:
            texts = json.loads((run_dir / "results" / f"completions_{scheme}.json")
                               .read_text(encoding="utf-8"))["completions"]
            correct[scheme] = np.array([is_correct(t, g)
                                        for t, g in zip(texts, golds)])
        base = correct["FP16"]
        rungs, raw_p = [], []
        for scheme in stored:
            if scheme == "FP16":
                continue
            cur = correct[scheme]
            lost = int((base & ~cur).sum())
            gained = int((~base & cur).sum())
            low, high = clopper_pearson(int(cur.sum()), n)
            p = exact_mcnemar(gained, lost)
            rungs.append({"scheme": scheme, "bits": stored[scheme]["bits_per_param"],
                          "n_correct": int(cur.sum()), "accuracy": float(cur.mean()),
                          "ci_low": low, "ci_high": high, "lost": lost,
                          "gained": gained, "mcnemar_p": p})
            raw_p.append(p)
        for row, adjusted in zip(rungs, holm(raw_p)):
            row["mcnemar_p_holm_within_run"] = adjusted
        model = manifest.get("model_id", "?")
        rows_by_run[label] = {
            "model": MODEL_LABELS.get(model, model), "run": run_dir.name, "n": n,
            "fp16_correct": int(base.sum()), "fp16_accuracy": float(base.mean()),
            "rows": rungs}

    # One correction over every cell in the round, the stricter family the paper
    # declares primary. Reported alongside the within-run one, not instead of it.
    flat = [(label, row) for label, block in rows_by_run.items()
            for row in block["rows"]]
    for (_, row), adjusted in zip(flat, holm([r["mcnemar_p"] for _, r in flat])):
        row["mcnemar_p_holm_all_cells"] = adjusted
    if rows_by_run:
        print(f"  all-cells Holm family: {len(flat)} cells over "
              f"{sorted(rows_by_run)}")
    return rows_by_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    ap.add_argument("--gsm8k", type=Path, default=None,
                    help="GSM8K test.jsonl, in the order the runs consumed it. "
                         "Omit to skip the capability arm.")
    ap.add_argument("--include", default=None,
                    help="glob on the run directory NAME. Default reads every run, "
                         "because the token-budget comparison needs round one's "
                         "48-token runs as well as round two's 256-token ones.")
    ap.add_argument("--gsm8k-min-n", type=int, default=300,
                    help="skip capability runs below this many questions. The "
                         "default excludes round one's 200-question runs, which "
                         "measure the same cells this arm re-measures; counting "
                         "both would put every cell in the family twice.")
    ap.add_argument("--out", type=Path, default=Path("docs/paper/round2.json"))
    args = ap.parse_args()

    runs = select_runs(args.runs, args.include)
    print(f"reading {len(runs)} run directories under {args.runs}\n")

    print("=== behavioural runs, keyed by label ===")
    blocks = behavioural(runs)
    for label, block in blocks.items():
        print(f"{label:26s} {block['model']:14s} n={block['n_prompts']} "
              f"tokens={block['max_new_tokens']}  schemes={block['schemes']}")

    payload: dict[str, Any] = {"behavioural": blocks}

    print("\n=== paired transitions against each run's own FP16 ===")
    print(f"{'run':26s} {'scheme':9s} {'->refuse':>9s} {'->comply':>9s} "
          f"{'gradable':>9s} {'p':>8s} {'p_holm':>8s}")
    for label, block in blocks.items():
        for i, row in enumerate(block["rows"]):
            print(f"{label if i == 0 else '':26s} {row['scheme']:9s} "
                  f"{row['to_refusal']:9d} {row['to_compliance']:9d} "
                  f"{row['n_gradable']:9d} {row['mcnemar_p']:8.4f} "
                  f"{row['mcnemar_p_holm_within_run']:8.4f}")

    print("\n=== token budget: does the shift survive 256 tokens? ===")
    payload["token_budget"] = token_budget(blocks, runs)
    for label, block in payload["token_budget"].items():
        if not block.get("rows"):
            print(f"{label}: {block.get('note', 'no comparable baseline')}")
            continue
        print(f"{label}  vs  {block['comparable_baseline']}  "
              f"({block['n_prompts']} paired prompts)")
        for row in block["rows"]:
            a, b = row["tokens_48"], row["tokens_256"]
            print(f"  {row['scheme']:9s} 48 tok: {a['to_refusal']:3d} refuse / "
                  f"{a['to_compliance']:3d} comply (p={a['p']:.4f})   "
                  f"256 tok: {b['to_refusal']:3d} / {b['to_compliance']:3d} "
                  f"(p={b['p']:.4f})   direction "
                  f"{'same' if row['same_direction'] else 'REVERSES'}")
            pp = row.get("per_prompt")
            if pp and "n" in pp:
                print(f"            per-prompt: {pp['changed']} of {pp['n']} "
                      f"verdicts changed with the window "
                      f"({pp['changed_rate']:.1%}); "
                      f"{pp['became_comply']} became COMPLY, "
                      f"{pp['left_comply']} left it (p={pp['p']:.4f})")

        flips = block.get("flip_status")
        if flips and "n" in flips:
            print(f"    baseline stability: {flips['fp16_verdict_changed']} of "
                  f"{flips['n']} FP16 verdicts changed with the budget "
                  f"({flips['fp16_verdict_changed_rate']:.1%})")
            print(f"    unsafe-flip status on the {flips['n_baseline_stable']} "
                  f"prompts whose baseline held: "
                  f"{flips['flip_only_at_48']} flipped only at 48 tokens, "
                  f"{flips['flip_only_at_256']} only at 256 "
                  f"(p={flips['p']:.4f})")
            print(f"    {flips['note']}")

    print("\n=== deployed quantizers, off the RTN axis by construction ===")
    payload["deployed"] = deployed(blocks)
    if not payload["deployed"]:
        print("none in this selection")
    for label, block in payload["deployed"].items():
        print(f"{label}  ({block['model']})")
        for row in block["rows"]:
            cfg = row["quantization_config"]
            budget = cfg.get("bits_per_param_stored")
            print(f"  {row['scheme']:9s} {row['repo']}")
            print(f"    {row['to_refusal']:3d} newly refuse / "
                  f"{row['to_compliance']:3d} newly comply   p={row['mcnemar_p']:.4f}"
                  f"   stored bits/param "
                  + (f"{budget:.2f}" if isinstance(budget, float) else "not declared"))

    if args.gsm8k:
        print("\n=== GSM8K, paired exact McNemar ===")
        payload["gsm8k"] = gsm8k(runs, args.gsm8k, args.gsm8k_min_n)
        print(f"{'run':26s} {'scheme':9s} {'acc%':>6s} {'lost':>5s} {'gained':>6s} "
              f"{'p':>8s} {'p_holm':>8s} {'p_all':>8s}")
        for label, block in payload["gsm8k"].items():
            print(f"{label} FP16 {100 * block['fp16_accuracy']:.1f}% "
                  f"({block['fp16_correct']}/{block['n']})")
            for row in block["rows"]:
                print(f"{'':26s} {row['scheme']:9s} {100 * row['accuracy']:6.1f} "
                      f"{row['lost']:5d} {row['gained']:6d} {row['mcnemar_p']:8.4f} "
                      f"{row['mcnemar_p_holm_within_run']:8.4f} "
                      f"{row['mcnemar_p_holm_all_cells']:8.4f}")
    else:
        print("\n=== GSM8K skipped (no --gsm8k) ===")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
