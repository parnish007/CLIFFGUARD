"""LADDER tier router — tier detection and gate selection.
See blueprint §10.

LADDER selects which gates run, at what depth, and with what budgets,
based on detected hardware tier and observability mode (white-box /
black-box). This is configuration, not learning — CONDUCTOR handles
the online weight adaptation.

Tier gate sets per blueprint §10:
  Tier A: all 11 primitives + ATTEST-WH
  Tier B: all except LOOKOUT-JG (too slow for Pi 5 CPU)
  Tier C: VESTIBULE-LZ, VESTIBULE-PS, ATTEST-WH only (narrow scope)
  Tier C+: VESTIBULE-LZ, VESTIBULE-PS, ATTEST-WH + B-PROBE-LOGIT
           (PromptGuard-2-22M-INT4 replaces full PROBE stack)
"""

from cliffguard.types import Tier

TIER_GATES: dict[Tier, list[str]] = {
    Tier.A: [
        "VESTIBULE-LZ", "VESTIBULE-PS",
        "PROBE-RM", "PROBE-MT", "PROBE-HD",
        "TRIPWIRE-H", "TRIPWIRE-R",
        "LOOKOUT-CT", "LOOKOUT-JG",
        "B-PROBE-LOGIT", "B-PROBE-CONSISTENCY",
        "ATTEST-WH",
    ],
    Tier.B: [
        "VESTIBULE-LZ", "VESTIBULE-PS",
        "PROBE-RM", "PROBE-MT", "PROBE-HD",
        "TRIPWIRE-H", "TRIPWIRE-R",
        "LOOKOUT-CT",
        "B-PROBE-LOGIT", "B-PROBE-CONSISTENCY",
        "ATTEST-WH",
    ],
    Tier.C: [
        "VESTIBULE-LZ", "VESTIBULE-PS",
        "ATTEST-WH",
    ],
    Tier.C_PLUS: [
        "VESTIBULE-LZ", "VESTIBULE-PS",
        "B-PROBE-LOGIT",
        "ATTEST-WH",
    ],
}


def gates_for_tier(tier: Tier) -> list[str]:
    """Return the ordered list of gate names active for this tier."""
    return list(TIER_GATES[tier])


def is_gate_active(gate_name: str, tier: Tier) -> bool:
    """Return True if gate_name is in the active set for tier."""
    return gate_name in TIER_GATES[tier]
