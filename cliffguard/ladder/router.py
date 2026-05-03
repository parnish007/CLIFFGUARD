"""LADDER router — dispatches gate execution by tier.
See blueprint §10.

The router does not execute gates itself — it returns the ordered
list of gate names that should be executed for a given tier and
observability mode, respecting the never-disable constraint from
CONDUCTOR (TRIPWIRE-R and ATTEST-WH always included if available
for the tier).
"""

from cliffguard.types import Tier
from cliffguard.ladder.tier import gates_for_tier

_WHITE_BOX_ONLY: frozenset[str] = frozenset({"PROBE-RM", "PROBE-MT", "PROBE-HD"})
_NEVER_DISABLE: frozenset[str] = frozenset({"TRIPWIRE-R", "ATTEST-WH"})


def route(
    tier: Tier,
    black_box: bool = False,
) -> list[str]:
    """Return ordered gate list for tier.
    If black_box=True, remove white-box-only gates:
      PROBE-RM, PROBE-MT, PROBE-HD (require residual stream access).
    TRIPWIRE-R and ATTEST-WH are always retained (never-disable).
    """
    gates = gates_for_tier(tier)
    if black_box:
        gates = [
            g for g in gates
            if g not in _WHITE_BOX_ONLY or g in _NEVER_DISABLE
        ]
    return gates


def gate_count(tier: Tier, black_box: bool = False) -> int:
    """Return the number of active gates for tier and mode."""
    return len(route(tier, black_box=black_box))
