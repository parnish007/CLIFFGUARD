"""PROBE-HD harmfulness-direction gate — see blueprint §5.3.

Computes the scalar harmfulness margin m_h = ĥ · z_l(t_inst) where
ĥ is the harmfulness direction in the residual stream (Zhao et al.,
arXiv:2507.11878) and z_l is the hidden state at layer l, position
t_inst (the last token of the instruction itself, before the model
begins responding).

Complements PROBE-RM: PROBE-RM reads the model's refusal posture at
t_post-inst; PROBE-HD reads the input's harmfulness signal at t_inst.
High m_h with low PROBE-RM margin is the highest-risk configuration.

In scaffolding mode (Phase A), operates on synthetic hidden states.
"""

import numpy as np
import numpy.typing as npt

from cliffguard.types import CalibrationTable, GateVerdict, Margin, QuantScheme, Tier


def compute_harmfulness_margin(
    hidden_state: npt.NDArray[np.float64],
    harmfulness_direction: npt.NDArray[np.float64],
) -> float:
    """Return the cosine projection of the hidden state at t_inst
    onto the harmfulness direction ĥ.

    Raises ValueError if either array is zero-norm.
    Identical in structure to PROBE-RM's compute_margin but
    semantically distinct: high score = high harmfulness signal.
    """
    norm_z = float(np.linalg.norm(hidden_state))
    norm_h = float(np.linalg.norm(harmfulness_direction))
    if norm_z == 0.0:
        raise ValueError("hidden_state has zero norm — cannot compute harmfulness margin")
    if norm_h == 0.0:
        raise ValueError("harmfulness_direction has zero norm — direction is undefined")
    return float(np.dot(hidden_state, harmfulness_direction)) / (norm_z * norm_h)


def evaluate(
    hidden_state: npt.NDArray[np.float64],
    harmfulness_direction: npt.NDArray[np.float64],
    calibration: CalibrationTable,
    scheme: QuantScheme,
    tier: Tier,
) -> tuple[Margin, GateVerdict]:
    """Evaluate the HD gate.

    fired = True if compute_harmfulness_margin(...) > calibration.tau(scheme).
    (High harmfulness margin = input is in the harmful region — fire.)
    gate name is "PROBE-HD".
    Margin.value is the harmfulness margin.
    Margin.primitive is "PROBE-HD".
    Margin.layer is None.

    Firing direction is OPPOSITE to PROBE-RM:
      PROBE-RM fires when margin is LOW  (model not in refusal posture).
      PROBE-HD fires when margin is HIGH (input in harmful region).
    """
    margin_value = compute_harmfulness_margin(hidden_state, harmfulness_direction)
    threshold = calibration.tau(scheme)
    margin = Margin(
        value=margin_value,
        scheme=scheme,
        primitive="PROBE-HD",
        layer=None,
    )
    verdict = GateVerdict(
        gate="PROBE-HD",
        fired=margin_value > threshold,
        score=margin_value,
        threshold=threshold,
        tier=tier,
        threat_model=None,
    )
    return margin, verdict
