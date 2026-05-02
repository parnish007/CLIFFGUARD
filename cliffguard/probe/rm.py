"""PROBE-RM refusal-margin gate — see blueprint §5.1.

Computes the scalar refusal margin m_r = r̂ · z_l(t_post-inst) where
r̂ is the refusal direction in the residual stream (Arditi et al.,
arXiv:2406.11717) and z_l is the hidden state at layer l, position
t_post-inst (the last instruction token).

In scaffolding mode (Phase A), this module operates on synthetic
hidden states (numpy arrays). The real residual-stream extraction via
forward hooks is implemented in Phase B (cliffguard/engines/).

FPR decoupling: per Theorem 14.1, the FPR of this gate is independent
of quantization scheme up to the calibration map P_q. TPR is NOT
decoupled and may collapse in the cliff regime — that is what H2 and
H1 measure respectively.
"""

import numpy as np
import numpy.typing as npt

from cliffguard.types import CalibrationTable, GateVerdict, Margin, QuantScheme, Tier


def compute_margin(
    hidden_state: npt.NDArray[np.float64],
    refusal_direction: npt.NDArray[np.float64],
) -> float:
    """Return the scalar refusal margin: dot(r̂, z) / (||r̂|| * ||z||).

    This is the cosine projection of the hidden state onto the
    refusal direction.
    Raises ValueError if either array is zero-norm.
    """
    norm_z = float(np.linalg.norm(hidden_state))
    norm_r = float(np.linalg.norm(refusal_direction))
    if norm_z == 0.0:
        raise ValueError("hidden_state has zero norm — cannot compute refusal margin")
    if norm_r == 0.0:
        raise ValueError("refusal_direction has zero norm — direction is undefined")
    return float(np.dot(hidden_state, refusal_direction)) / (norm_z * norm_r)


def evaluate(
    hidden_state: npt.NDArray[np.float64],
    refusal_direction: npt.NDArray[np.float64],
    calibration: CalibrationTable,
    scheme: QuantScheme,
    tier: Tier,
) -> tuple[Margin, GateVerdict]:
    """Compute margin and evaluate the gate.

    Returns both the Margin (for CONDUCTOR feature vector) and the
    GateVerdict.
    fired = True if margin < calibration.tau(scheme).
    (Low margin = hidden state far from refusal direction = risk.)
    gate name is "PROBE-RM".
    Margin.primitive is "PROBE-RM".
    Margin.layer is None in scaffolding mode.
    """
    margin_value = compute_margin(hidden_state, refusal_direction)
    threshold = calibration.tau(scheme)
    margin = Margin(
        value=margin_value,
        scheme=scheme,
        primitive="PROBE-RM",
        layer=None,
    )
    verdict = GateVerdict(
        gate="PROBE-RM",
        fired=margin_value < threshold,
        score=margin_value,
        threshold=threshold,
        tier=tier,
        threat_model=None,
    )
    return margin, verdict
