"""CONDUCTOR context-vector builder — see blueprint §6.3.

Assembles the LinUCB context vector x ∈ ℝ^d from the gate verdicts
and margins produced by one request cycle. The context vector is the
input to Conductor.select_weights() and Conductor.update().

Feature dimensions (d = 14):
  0:  VESTIBULE-LZ score (compression ratio)
  1:  VESTIBULE-PS score (signal score)
  2:  PROBE-RM margin (refusal direction cosine)
  3:  PROBE-MT rho_dot (first derivative of margin trajectory)
  4:  PROBE-MT rho_ddot (second derivative — surfaced here per
      Phase A gate decision: rho_ddot computed in PROBE-MT but
      deferred to this module for CONDUCTOR feature vector)
  5:  PROBE-HD harmfulness margin
  6:  TRIPWIRE-H final CUSUM statistic
  7:  TRIPWIRE-R log-likelihood ratio
  8:  LOOKOUT-CT canary count
  9:  LOOKOUT-JG compliance rate
  10: B-PROBE-LOGIT logistic score
  11: B-PROBE-CONSISTENCY JSD score
  12: ATTEST-WH result (1.0=ALLOW, 0.5=DEGRADED, 0.0=BLOCK)
  13: tier indicator (0.0=A, 0.33=B, 0.67=C, 1.0=C_PLUS)

Missing gates (not run for this tier) are filled with 0.0.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cliffguard.attest.wh import AttestResult
from cliffguard.types import GateVerdict, Tier

CONTEXT_DIM: int = 14

FEATURE_INDEX: dict[str, int] = {
    "VESTIBULE-LZ": 0,
    "VESTIBULE-PS": 1,
    "PROBE-RM": 2,
    "PROBE-MT-rho_dot": 3,
    "PROBE-MT-rho_ddot": 4,
    "PROBE-HD": 5,
    "TRIPWIRE-H": 6,
    "TRIPWIRE-R": 7,
    "LOOKOUT-CT": 8,
    "LOOKOUT-JG": 9,
    "B-PROBE-LOGIT": 10,
    "B-PROBE-CONSISTENCY": 11,
    "ATTEST-WH": 12,
    "tier": 13,
}

TIER_INDICATOR: dict[Tier, float] = {
    Tier.A: 0.0,
    Tier.B: 0.33,
    Tier.C: 0.67,
    Tier.C_PLUS: 1.0,
}

ATTEST_SCORE: dict[AttestResult, float] = {
    AttestResult.ALLOW: 1.0,
    AttestResult.DEGRADED: 0.5,
    AttestResult.BLOCK: 0.0,
}


def build_context(
    verdicts: list[GateVerdict],
    tier: Tier,
    attest_result: AttestResult | None = None,
    probe_mt_rho_ddot: float | None = None,
) -> npt.NDArray[np.float64]:
    """Assemble the CONTEXT_DIM-dimensional context vector.

    For each GateVerdict in verdicts, place verdict.score at the
    index given by FEATURE_INDEX[verdict.gate].
    Special cases:
      PROBE-MT: verdict.score goes to index 3 (rho_dot).
                probe_mt_rho_ddot goes to index 4 if provided,
                else 0.0.
      ATTEST-WH: use ATTEST_SCORE[attest_result] at index 12;
                 if attest_result is None, use 1.0 (ALLOW assumed).
    Always set index 13 to TIER_INDICATOR[tier].
    Missing gates (gate name not in FEATURE_INDEX) are silently
    skipped — callers may pass partial verdict sets for lower tiers.
    Returns array of shape (CONTEXT_DIM,) dtype float64."""
    x: npt.NDArray[np.float64] = np.zeros(CONTEXT_DIM, dtype=np.float64)

    for verdict in verdicts:
        gate = verdict.gate
        if gate == "PROBE-MT":
            x[FEATURE_INDEX["PROBE-MT-rho_dot"]] = verdict.score
            x[FEATURE_INDEX["PROBE-MT-rho_ddot"]] = (
                probe_mt_rho_ddot if probe_mt_rho_ddot is not None else 0.0
            )
        elif gate == "ATTEST-WH":
            # score from verdict is ignored; attest_result drives index 12
            pass
        elif gate in FEATURE_INDEX:
            x[FEATURE_INDEX[gate]] = verdict.score
        # unknown gate names are silently skipped

    # Index 12 is always driven by attest_result, not verdict.score
    x[FEATURE_INDEX["ATTEST-WH"]] = (
        ATTEST_SCORE[attest_result] if attest_result is not None else 1.0
    )
    x[FEATURE_INDEX["tier"]] = TIER_INDICATOR[tier]
    return x


def context_dim() -> int:
    """Return CONTEXT_DIM. Utility for external callers."""
    return CONTEXT_DIM
