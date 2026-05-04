"""End-to-end dry run — see development.md Task 34.

Exercises the full CLIFFGUARD request pipeline using only Phase A
stub components — no real models, no GPU, no corpus files required.

Pipeline exercised:
  1. ATTEST-WH: synthetic manifest hash (always ALLOW in dry run).
  2. LADDER: route gates for the requested tier and mode.
  3. VESTIBULE gates: run LZ and PS on a synthetic prompt.
  4. PROBE gates (if tier >= B): run RM, MT, HD on synthetic arrays.
  5. TRIPWIRE gates: run H and R on synthetic entropy/logprob arrays.
  6. LOOKOUT gates: run CT on synthetic output; skip JG (requires LLM).
  7. B-PROBE gates (if black_box=True): run LOGIT and CONSISTENCY.
  8. CONDUCTOR: build context vector, select weights, aggregate verdict.
  9. Print a structured summary of all gate verdicts and the final
     block decision.

Exit code 0 = pipeline completed without error (regardless of block
decision). Exit code 1 = any unhandled exception.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path  # noqa: F401

import numpy as np

from cliffguard.attest.wh import AttestResult
from cliffguard.bprobe.consistency import evaluate as cons_eval
from cliffguard.bprobe.logit import evaluate as logit_eval
from cliffguard.conductor.bandit import Conductor
from cliffguard.conductor.context import CONTEXT_DIM, build_context
from cliffguard.ladder.router import route
from cliffguard.lookout.ct import evaluate as ct_eval
from cliffguard.probe.hd import evaluate as hd_eval
from cliffguard.probe.mt import evaluate as mt_eval
from cliffguard.probe.rm import evaluate as rm_eval
from cliffguard.tripwire.h import evaluate as th_eval
from cliffguard.tripwire.r import evaluate as tr_eval
from cliffguard.types import CalibrationTable, GateVerdict, QuantScheme, Tier
from cliffguard.vestibule.lz import evaluate as lz_eval
from cliffguard.vestibule.ps import evaluate as ps_eval

SYNTHETIC_PROMPT = (
    "Explain the history of the Roman Empire in three paragraphs."
)
SYNTHETIC_OUTPUT = (
    "The Roman Empire began with Augustus Caesar in 27 BC..."
)


def make_calibration(
    scheme: QuantScheme,
    threshold: float = 0.5,
) -> CalibrationTable:
    """Return a CalibrationTable with threshold for the given scheme."""
    return CalibrationTable(
        primitive="DRY-RUN",
        thresholds={scheme: threshold},
    )


def run_dry_run(
    tier: Tier = Tier.A,
    scheme: QuantScheme = QuantScheme.FP16,
    black_box: bool = False,
    hidden_dim: int = 64,
    seed: int = 42,
) -> dict[str, object]:
    """Run the full dry-run pipeline.
    Returns dict with keys:
      tier, scheme, black_box,
      gates_run: list[str],
      verdicts: list[dict] (gate, fired, score, threshold),
      block_decision: bool,
      context_dim: int.
    Prints a summary table to stdout."""
    rng = np.random.default_rng(seed)
    cal = make_calibration(scheme)

    # LADDER: ordered gate list for this tier/mode.
    routed_gates = route(tier, black_box=black_box)

    gates_run: list[str] = []
    gate_verdicts: list[GateVerdict] = []
    attest_result = AttestResult.ALLOW

    for gate_name in routed_gates:
        if gate_name == "LOOKOUT-JG":
            # Requires LLM inference — skip in dry run.
            continue

        gates_run.append(gate_name)

        if gate_name == "ATTEST-WH":
            # Synthetic manifest hash — always ALLOW.
            gate_verdicts.append(
                GateVerdict(
                    gate="ATTEST-WH",
                    fired=False,
                    score=1.0,
                    threshold=0.5,
                    tier=tier,
                )
            )
            attest_result = AttestResult.ALLOW

        elif gate_name == "VESTIBULE-LZ":
            gate_verdicts.append(lz_eval(SYNTHETIC_PROMPT, cal, scheme, tier))

        elif gate_name == "VESTIBULE-PS":
            gate_verdicts.append(ps_eval(SYNTHETIC_PROMPT, cal, scheme, tier))

        elif gate_name == "PROBE-RM":
            h = rng.standard_normal(hidden_dim).astype(np.float64)
            r_dir = rng.standard_normal(hidden_dim).astype(np.float64)
            _, verdict = rm_eval(h, r_dir, cal, scheme, tier)
            gate_verdicts.append(verdict)

        elif gate_name == "PROBE-MT":
            margins = rng.standard_normal(5).astype(np.float64)
            _, verdict = mt_eval(margins, cal, scheme, tier)
            gate_verdicts.append(verdict)

        elif gate_name == "PROBE-HD":
            h2 = rng.standard_normal(hidden_dim).astype(np.float64)
            harm_dir = rng.standard_normal(hidden_dim).astype(np.float64)
            _, verdict = hd_eval(h2, harm_dir, cal, scheme, tier)
            gate_verdicts.append(verdict)

        elif gate_name == "TRIPWIRE-H":
            entropies = np.abs(rng.standard_normal(20)).astype(np.float64)
            gate_verdicts.append(th_eval(entropies, cal, scheme, tier))

        elif gate_name == "TRIPWIRE-R":
            input_lp = rng.standard_normal(20).astype(np.float64)
            benign_lp = rng.standard_normal(20).astype(np.float64)
            gate_verdicts.append(tr_eval(input_lp, benign_lp, cal, scheme, tier))

        elif gate_name == "LOOKOUT-CT":
            gate_verdicts.append(ct_eval(SYNTHETIC_OUTPUT, [], cal, scheme, tier))

        elif gate_name == "B-PROBE-LOGIT":
            k = 10
            logprobs = rng.standard_normal(k).astype(np.float64)
            logit_weights = rng.standard_normal(k).astype(np.float64)
            _, verdict = logit_eval(logprobs, logit_weights, cal, scheme, tier)
            gate_verdicts.append(verdict)

        elif gate_name == "B-PROBE-CONSISTENCY":
            lpm = rng.standard_normal((2, 10)).astype(np.float64)
            _, verdict = cons_eval(lpm, cal, scheme, tier)
            gate_verdicts.append(verdict)

    # CONDUCTOR: assemble context, select weights, aggregate verdict.
    ctx = build_context(gate_verdicts, tier, attest_result=attest_result)
    conductor = Conductor(d=CONTEXT_DIM)
    weights = conductor.select_weights(ctx)
    block_decision = conductor.aggregate_verdict(gate_verdicts, weights)

    # Serialise verdicts for the return dict.
    verdict_dicts: list[dict[str, object]] = [
        {
            "gate": v.gate,
            "fired": v.fired,
            "score": v.score,
            "threshold": v.threshold,
        }
        for v in gate_verdicts
    ]

    # Print structured summary (use typed gate_verdicts to avoid object cast).
    print(
        f"--- CLIFFGUARD dry run | tier={tier.value}"
        f" scheme={scheme.value} black_box={black_box} ---"
    )
    print(f"{'Gate':<28} {'Fired':<6} {'Score':>10} {'Threshold':>12}")
    print("-" * 60)
    for v in gate_verdicts:
        fired_str = "YES" if v.fired else "no"
        print(f"{v.gate:<28} {fired_str:<6} {v.score:>10.4f} {v.threshold:>12.4f}")
    print("-" * 60)
    print(f"Gates run: {len(gates_run)} | Context dim: {CONTEXT_DIM}")
    print(f"Block decision: {'BLOCK' if block_decision else 'ALLOW'}")

    return {
        "tier": tier.value,
        "scheme": scheme.value,
        "black_box": black_box,
        "gates_run": gates_run,
        "verdicts": verdict_dicts,
        "block_decision": block_decision,
        "context_dim": CONTEXT_DIM,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse --tier, --scheme, --black-box, --hidden-dim, --seed.
    Call run_dry_run and return 0 on success, 1 on exception."""
    parser = argparse.ArgumentParser(
        description="CLIFFGUARD end-to-end dry run (Phase A stubs, no GPU)."
    )
    parser.add_argument(
        "--tier",
        default="A",
        choices=["A", "B", "C", "C_PLUS"],
        help="Deployment tier (default: A).",
    )
    parser.add_argument(
        "--scheme",
        default="FP16",
        help="Quantization scheme name (default: FP16).",
    )
    parser.add_argument(
        "--black-box",
        action="store_true",
        default=False,
        help="Use black-box mode (remove white-box-only gates).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for synthetic arrays (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data (default: 42).",
    )
    args = parser.parse_args(argv)

    try:
        tier = Tier(args.tier)
        scheme = QuantScheme.from_string(args.scheme)
        run_dry_run(
            tier=tier,
            scheme=scheme,
            black_box=args.black_box,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
