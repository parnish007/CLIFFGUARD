"""B-PROBE-LOGIT black-box probe — see blueprint §5.10.

Operates on the top-k log-probability vector ℓ(q) ∈ ℝᵏ observable
from closed-weight API endpoints (OpenAI, Anthropic, Gemini — top-k
typically ≤ 20 as of 2025-2026). Applies a logistic head
φ_θ: ℝᵏ → ℝ trained on (prompt, refused/complied) pairs to produce
a compliance-risk score.

Per Corollary 14.2 (FPR-decoupling, black-box): FPR is portable across
quantization schemes via per-scheme calibration. TPR is NOT decoupled —
the cliff regime may shrink the harmful-prompt margin toward the benign
distribution, destroying detection power. This is what H3 measures.

In scaffolding mode (Phase A), the logistic head weights are synthetic.
Phase B (Task 23) fits real weights from Fold A data.
"""

import numpy as np
import numpy.typing as npt

from cliffguard.types import CalibrationTable, GateVerdict, Margin, QuantScheme, Tier


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid: 1 / (1 + exp(-x)).
    Clamp x to [-500, 500] to avoid overflow."""
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + float(np.exp(-x)))


def logistic_score(
    logprobs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    bias: float = 0.0,
) -> float:
    """Apply the logistic head: sigmoid(dot(weights, logprobs) + bias).
    Returns a score in (0.0, 1.0).
    Raises ValueError if logprobs and weights have different shapes."""
    if logprobs.shape != weights.shape:
        raise ValueError(
            f"Shape mismatch: logprobs {logprobs.shape} vs weights {weights.shape}"
        )
    return sigmoid(float(np.dot(weights, logprobs)) + bias)


def evaluate(
    logprobs: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    calibration: CalibrationTable,
    scheme: QuantScheme,
    tier: Tier,
    bias: float = 0.0,
) -> tuple[Margin, GateVerdict]:
    """Evaluate the B-PROBE-LOGIT gate.
    score = logistic_score(logprobs, weights, bias).
    fired = True if score > calibration.tau(scheme).
    (High score = high compliance risk = fire.)
    gate name is "B-PROBE-LOGIT".
    Margin.primitive is "B-PROBE-LOGIT".
    Margin.layer is None.
    Margin.value is the logistic score."""
    score = logistic_score(logprobs, weights, bias)
    threshold = calibration.tau(scheme)
    margin = Margin(value=score, scheme=scheme, primitive="B-PROBE-LOGIT", layer=None)
    verdict = GateVerdict(
        gate="B-PROBE-LOGIT",
        fired=score > threshold,
        score=score,
        threshold=threshold,
        tier=tier,
        threat_model=None,
    )
    return margin, verdict
