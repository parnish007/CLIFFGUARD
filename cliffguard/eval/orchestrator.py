"""Request-cycle orchestrator — see blueprint §4.1, §6.

Wires the LADDER router, gate evaluators, and CONDUCTOR into a single
callable per-request function. This is the top-level integration point
for Phase B: replace the stub gate calls with real engine-backed calls
to get end-to-end evaluation.

Request cycle (per blueprint §4.1):
  1. ATTEST-WH (boot-time, cached — not re-run per request)
  2. LADDER.route() → ordered gate list for tier + observability mode
  3. For each active gate: evaluate, collect GateVerdict
  4. CONDUCTOR.build_context() → context vector
  5. CONDUCTOR.select_weights() → per-gate weights
  6. CONDUCTOR.aggregate_verdict() → block decision (bool)
  7. (Async) CONDUCTOR.update() on incident feedback

Phase A: gate evaluators are stubs. Orchestrator accepts pre-computed
GateVerdict lists to decouple from real inference.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cliffguard.attest.wh import AttestResult
from cliffguard.conductor.bandit import Conductor
from cliffguard.conductor.context import build_context
from cliffguard.ladder.router import route
from cliffguard.types import GateVerdict, QuantScheme, Tier


def run_request_cycle(
    verdicts: list[GateVerdict],
    tier: Tier,
    scheme: QuantScheme,
    conductor: Conductor,
    attest_result: AttestResult = AttestResult.ALLOW,
    probe_mt_rho_ddot: float | None = None,
    black_box: bool = False,
) -> tuple[bool, dict[str, float], npt.NDArray[np.float64]]:
    """Run one request cycle given pre-computed gate verdicts.

    Steps:
      1. Build context vector via build_context().
      2. Select weights via conductor.select_weights(context).
      3. Filter weights to only gates active for this tier + mode
         (use route(tier, black_box) to get active gate list).
      4. Aggregate verdict via conductor.aggregate_verdict(
           verdicts, active_weights).
      5. Return (block_decision, active_weights, context_vector).

    active_weights is the subset of weights dict for active gates.
    block_decision is True if the request should be blocked."""
    context = build_context(
        verdicts,
        tier,
        attest_result=attest_result,
        probe_mt_rho_ddot=probe_mt_rho_ddot,
    )
    all_weights = conductor.select_weights(context)
    active_gates = route(tier, black_box=black_box)
    active_weights = {gate: all_weights[gate] for gate in active_gates if gate in all_weights}
    block_decision = conductor.aggregate_verdict(verdicts, active_weights)
    return block_decision, active_weights, context


def update_conductor(
    conductor: Conductor,
    gate_name: str,
    context: npt.NDArray[np.float64],
    reward: float,
) -> None:
    """Update CONDUCTOR with incident feedback.
    Thin wrapper around conductor.update() for a single arm."""
    conductor.update(gate_name, context, reward)
