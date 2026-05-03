"""CONDUCTOR — LinUCB contextual bandit gate orchestrator.
See blueprint §6.

Maintains per-arm (per-gate) LinUCB state and selects gate weights
online from sparse incident feedback. Arms correspond to the eleven
primitives plus ATTEST-WH. Reward is +1 for a correct block, -1 for
a miss (harmful output passed), -0.2 for a false positive.

Two firing-direction groups (established across Tasks 5-15):
  Fires-HIGH: VESTIBULE-LZ, VESTIBULE-PS, PROBE-HD, TRIPWIRE-H,
              LOOKOUT-CT, LOOKOUT-JG, B-PROBE-LOGIT
  Fires-LOW:  PROBE-RM, PROBE-MT, TRIPWIRE-R, B-PROBE-CONSISTENCY

CONDUCTOR must handle both groups when assembling the feature vector.
The never-disable list (TRIPWIRE-R, ATTEST-WH) always has minimum
weight MIN_WEIGHT = 0.1 regardless of bandit updates.

Per blueprint §6.2: LinUCB regret bound O~(sqrt(dT)).
Per blueprint §6.4: ADWIN drift detection triggers weight reset.
"""

import numpy as np
import numpy.typing as npt

from cliffguard.types import GateVerdict

ARMS: list[str] = [
    "VESTIBULE-LZ", "VESTIBULE-PS",
    "PROBE-RM", "PROBE-MT", "PROBE-HD",
    "TRIPWIRE-H", "TRIPWIRE-R",
    "LOOKOUT-CT", "LOOKOUT-JG",
    "B-PROBE-LOGIT", "B-PROBE-CONSISTENCY",
    "ATTEST-WH",
]

NEVER_DISABLE: frozenset[str] = frozenset({"TRIPWIRE-R", "ATTEST-WH"})
MIN_WEIGHT: float = 0.1
REWARD_CORRECT_BLOCK: float = 1.0
REWARD_MISS: float = -1.0
REWARD_FALSE_POSITIVE: float = -0.2


class LinUCBArm:
    """Per-arm LinUCB state (A matrix, b vector).
    d = feature dimension."""

    def __init__(self, d: int, alpha: float = 1.0) -> None:
        """Initialise A = I_d, b = 0_d."""
        self._d = d
        self._alpha = alpha
        self.A: npt.NDArray[np.float64] = np.eye(d, dtype=np.float64)
        self.b: npt.NDArray[np.float64] = np.zeros(d, dtype=np.float64)

    def update(self, x: npt.NDArray[np.float64], reward: float) -> None:
        """Update A and b given context vector x and scalar reward."""
        self.A += np.outer(x, x)
        self.b += reward * x

    def ucb_score(self, x: npt.NDArray[np.float64]) -> float:
        """Return the UCB score: x^T A^{-1} b + alpha * sqrt(x^T A^{-1} x).
        Uses np.linalg.solve rather than explicit inverse."""
        A_inv_x = np.linalg.solve(self.A, x).astype(np.float64)
        theta_hat = float(x @ np.linalg.solve(self.A, self.b))
        exploration = self._alpha * float(np.sqrt(x @ A_inv_x))
        return theta_hat + exploration


class Conductor:
    """LinUCB CONDUCTOR. Manages one LinUCBArm per gate arm."""

    def __init__(self, d: int, alpha: float = 1.0) -> None:
        """Initialise one LinUCBArm per arm in ARMS."""
        self._d = d
        self._alpha = alpha
        self._arms: dict[str, LinUCBArm] = {
            name: LinUCBArm(d, alpha) for name in ARMS
        }

    def select_weights(self, context: npt.NDArray[np.float64]) -> dict[str, float]:
        """Return a weight dict mapping each arm name to its UCB score,
        clipped to [MIN_WEIGHT, inf) for never-disable arms."""
        weights: dict[str, float] = {}
        for name, arm in self._arms.items():
            score = arm.ucb_score(context)
            if name in NEVER_DISABLE:
                score = max(score, MIN_WEIGHT)
            weights[name] = score
        return weights

    def update(
        self,
        arm_name: str,
        context: npt.NDArray[np.float64],
        reward: float,
    ) -> None:
        """Update the LinUCBArm for arm_name with context and reward.
        Raises ValueError if arm_name not in ARMS."""
        if arm_name not in self._arms:
            raise ValueError(
                f"Unknown arm '{arm_name}'. Valid arms: {ARMS}"
            )
        self._arms[arm_name].update(context, reward)

    def aggregate_verdict(
        self,
        verdicts: list[GateVerdict],
        weights: dict[str, float],
    ) -> bool:
        """Weighted vote across verdicts. Each fired gate contributes
        its weight to the total fired score. Returns True (block) if
        fired_score / total_weight > 0.5."""
        fired_score = 0.0
        total_weight = 0.0
        for verdict in verdicts:
            w = weights.get(verdict.gate, 1.0)
            total_weight += w
            if verdict.fired:
                fired_score += w
        if total_weight == 0.0:
            return False
        return (fired_score / total_weight) > 0.5

    def reset_weights(self) -> None:
        """Re-initialise all LinUCBArm states (ADWIN drift reset)."""
        for name in ARMS:
            self._arms[name] = LinUCBArm(self._d, self._alpha)
