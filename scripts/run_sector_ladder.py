"""Does the cliff sit at the same bit-width for reasoning as for refusal?

Every measurement in this project so far has been about one behaviour: refusal.
That makes it a result about safety, not the general law it set out to be. A law
of the form "quantization costs capability X at bit-width b" only becomes a law
once X varies.

This script runs the identical RTN ladder against GSM8K arithmetic reasoning and
reports the same three quantities the refusal arm reports:

  accuracy(q)        fraction of problems solved
  degenerate_rate(q) fraction of completions that are not language
  collapse point     the bit-width where accuracy falls below half of FP16's

Comparing that collapse point against refusal's is the first real test of
Corollary 2.1 (sector ordering): the claim that different behaviours break at
different, predictable bit-widths. Two sectors is not a law either, but it is the
difference between one data point and a slope.

Why GSM8K specifically: the answer is a number, so scoring needs no judge model
and no execution sandbox, which keeps the measurement as transparent as the
refusal-marker heuristic it sits beside. Chain-of-thought is allowed and only the
final number is scored.

Usage:
  python scripts/run_sector_ladder.py                    # 200 problems, 8..2 bits
  python scripts/run_sector_ladder.py --n 24 --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliffguard.eval.storage import new_run, record_corpus, record_environment
from scripts.run_behavioural_ladder import (
    DEGENERACY_NLL_MULTIPLE,
    generate_batched,
    score_nll,
)
import scripts.run_local_ladder as ladder
from scripts.run_local_ladder import (
    load_fp16_model,
    load_rtn_model,
    rtn_bits_per_parameter,
)

FloatArray = Any

# GSM8K gold answers are written as "#### 42" at the end of the rationale.
GOLD_PATTERN = re.compile(r"####\s*([-+]?[\d,]*\.?\d+)")
# Any number in the model's output; the LAST one is taken as the answer, which is
# the standard GSM8K convention and tolerates chain-of-thought arithmetic.
NUMBER_PATTERN = re.compile(r"[-+]?[\d,]*\.?\d+")


def parse_number(text: str) -> float | None:
    cleaned = text.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_gold(answer: str) -> float | None:
    match = GOLD_PATTERN.search(answer)
    return parse_number(match.group(1)) if match else None


def extract_predicted(completion: str) -> float | None:
    """Last number in the completion.

    Standard for GSM8K: models state intermediate arithmetic and finish with the
    result. Taking the last number is more robust than requiring a fixed answer
    format, which a quantized model is exactly the sort of thing to stop obeying.
    """
    matches = NUMBER_PATTERN.findall(completion)
    for candidate in reversed(matches):
        value = parse_number(candidate)
        if value is not None:
            return value
    return None


def is_correct(completion: str, gold: float, tolerance: float = 1e-4) -> bool:
    predicted = extract_predicted(completion)
    return predicted is not None and abs(predicted - gold) <= tolerance


def load_gsm8k(n: int) -> tuple[list[str], list[float]]:
    """Load GSM8K test problems with parseable gold answers."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("datasets not installed: uv pip install datasets") from exc

    rows = load_dataset("openai/gsm8k", "main", split="test")
    questions: list[str] = []
    golds: list[float] = []
    for row in rows:
        gold = extract_gold(row["answer"])
        if gold is None:
            continue
        questions.append(row["question"].strip())
        golds.append(gold)
        if len(questions) >= n:
            break
    if len(questions) < 4:
        raise SystemExit(f"only {len(questions)} usable GSM8K problems loaded")
    return questions, golds


def main() -> int:  # noqa: C901 - linear experiment script
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=ladder.MODEL_ID, help="HF checkpoint under test")
    ap.add_argument("--n", type=int, default=200, help="GSM8K problems")
    ap.add_argument("--bits", type=int, nargs="*", default=[8, 7, 6, 5, 4, 3, 2])
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache", type=Path, default=Path("artifacts/sector_cache"))
    ap.add_argument("--label", default="sector-ladder-gsm8k")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    ladder.MODEL_ID = args.model          # loaders read the module global
    args.cache = args.cache / "".join(
        c if (c.isalnum() or c in "-_.") else "-" for c in args.model
    )
    args.cache.mkdir(parents=True, exist_ok=True)

    print(f"model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    questions, golds = load_gsm8k(args.n)
    print(f"GSM8K: {len(questions)} problems")

    # Chain-of-thought is invited but the format is not over-constrained: a
    # quantized model losing instruction-following would otherwise be scored as
    # losing arithmetic, confounding two different capabilities.
    prompts = [
        f"{q}\n\nSolve this step by step, then give the final numeric answer."
        for q in questions
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    schemes = ["FP16"] + [f"RTN_{b}B" for b in args.bits]
    completions: dict[str, list[str]] = {}

    for name in schemes:
        cache = args.cache / f"gsm8k_{name}_n{len(prompts)}_t{args.max_new_tokens}.json"
        if cache.exists():
            print(f"[{name}] cache hit", flush=True)
            completions[name] = json.loads(cache.read_text(encoding="utf-8"))
            continue
        started = time.time()
        print(f"[{name}] loading ...", flush=True)
        if name == "FP16":
            model = load_fp16_model()
        else:
            code_bits = int(name.split("_")[1][:-1])
            model, _ = load_rtn_model(code_bits, args.group, [])
        model.eval()
        print(f"[{name}] generating {args.max_new_tokens} tokens x {len(prompts)} ...", flush=True)
        texts = generate_batched(
            model, tokenizer, prompts, args.max_new_tokens, args.batch_size, name
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        cache.write_text(json.dumps(texts, indent=1), encoding="utf-8")
        completions[name] = texts
        print(f"[{name}] done in {time.time() - started:.0f}s", flush=True)

    # Degeneracy on the same footing as the refusal arm: one FP16 reference.
    print("\n=== scoring completions under the FP16 reference ===")
    nll_cache = args.cache / f"nll_gsm8k_n{len(prompts)}_t{args.max_new_tokens}.json"
    if nll_cache.exists():
        nll = {k: np.asarray(v, dtype=np.float64)
               for k, v in json.loads(nll_cache.read_text(encoding="utf-8")).items()}
        print("cache hit")
    else:
        reference = load_fp16_model()
        reference.eval()
        nll = {n: score_nll(reference, tokenizer, completions[n], args.batch_size, n)
               for n in schemes}
        del reference
        gc.collect()
        torch.cuda.empty_cache()
        nll_cache.write_text(json.dumps({k: v.tolist() for k, v in nll.items()}), encoding="utf-8")

    finite_fp16 = nll["FP16"][np.isfinite(nll["FP16"])]
    threshold = float(np.median(finite_fp16) * DEGENERACY_NLL_MULTIPLE)
    print(f"FP16 median NLL {np.median(finite_fp16):.3f} -> degeneracy threshold {threshold:.3f}")

    print("\n=== SECTOR LADDER: GSM8K REASONING ===")
    print(f"{'scheme':10s} {'bits':>6s} {'accuracy':>9s} {'degen':>7s} {'medNLL':>7s} "
          f"{'vs FP16':>8s}")
    print("-" * 52)
    results: dict[str, Any] = {}
    fp16_accuracy = None
    for name in schemes:
        texts = completions[name]
        degenerate = np.array(
            [(not t.strip()) or (not np.isfinite(v)) or v > threshold
             for t, v in zip(texts, nll[name])]
        )
        # Degenerate completions are scored as incorrect, not excluded. Losing the
        # ability to produce language IS losing the ability to solve the problem;
        # excluding them would flatter the low-precision rungs by dropping their
        # worst cases from the denominator.
        correct = np.array([
            (not degenerate[i]) and is_correct(texts[i], golds[i]) for i in range(len(texts))
        ])
        accuracy = float(correct.mean())
        if name == "FP16":
            fp16_accuracy = accuracy
        finite = nll[name][np.isfinite(nll[name])]
        bits = 16.0 if name == "FP16" else rtn_bits_per_parameter(
            int(name.split("_")[1][:-1]), args.group
        )
        results[name] = {
            "accuracy": accuracy, "n_correct": int(correct.sum()),
            "degenerate_rate": float(degenerate.mean()),
            "median_nll": float(np.median(finite)) if finite.size else float("inf"),
            "bits_per_param": bits,
            # A zero FP16 baseline means the task is out of reach for this model at
            # this token budget, not that the ladder did anything. Reported as such
            # rather than as a nan ratio.
            "relative_to_fp16": (accuracy / fp16_accuracy) if fp16_accuracy else None,
        }
        relative = results[name]["relative_to_fp16"]
        print(f"{name:10s} {bits:6.2f} {accuracy:9.3f} {degenerate.mean():7.3f} "
              f"{results[name]['median_nll']:7.2f} "
              + (f"{relative:8.2f}" if relative is not None else "     n/a"))

    # Collapse point: first rung at or below half of FP16 accuracy, walking down.
    collapse_bits = None
    for name in schemes[1:]:
        if fp16_accuracy and results[name]["accuracy"] <= 0.5 * fp16_accuracy:
            collapse_bits = results[name]["bits_per_param"]
            break
    print(f"\nFP16 accuracy {fp16_accuracy:.3f}")
    if collapse_bits is not None:
        print(f"reasoning collapse (accuracy <= 50% of FP16) first at {collapse_bits:.2f} "
              "bits/param")
    else:
        print("accuracy never fell to half of FP16 on this ladder")
    print("\nCompare against the refusal arm's collapse point (run_behavioural_ladder.py).")
    print("Corollary 2.1 predicts different sectors break at different bit-widths; two")
    print("sectors is not a law, but it is the difference between a point and a slope.")

    run = new_run(
        args.label,
        model_id=args.model,
        extra={
            "stage": "sector_ladder", "sector": "reasoning_gsm8k",
            "dataset": "openai/gsm8k main test", "n_problems": len(prompts),
            "bits": args.bits, "group": args.group,
            "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
            "degeneracy_nll_threshold": threshold,
            "fp16_accuracy": fp16_accuracy, "collapse_bits": collapse_bits,
            "gpu": torch.cuda.get_device_name(0),
            "scoring_note": (
                "Last number in the completion versus the GSM8K gold answer, tolerance 1e-4. "
                "Chain-of-thought is invited but the output format is not constrained, so a "
                "model that loses instruction-following is not scored as losing arithmetic. "
                "Degenerate completions count as INCORRECT rather than being excluded: dropping "
                "them would flatter the low-precision rungs by removing their worst cases."
            ),
        },
    )
    record_environment(run)
    record_corpus(run, "gsm8k_questions", questions)
    for name in schemes:
        run.save_json(f"completions_{name}", {"completions": completions[name]})
    run.save_json("sector_gsm8k", results)
    run.write_manifest()
    summary = (
        f"GSM8K: FP16 {fp16_accuracy:.3f}"
        + (f"; collapse at {collapse_bits:.2f} bits" if collapse_bits else "; no collapse")
    )
    run.append_to_index(summary)
    print(f"\nrun: {run.path}\n{summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
