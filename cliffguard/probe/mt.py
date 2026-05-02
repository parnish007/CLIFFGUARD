"""PROBE-MT margin-trajectory gate — see blueprint §5.2.

Tracks the first and second derivatives of the refusal margin across
decoding steps: ρ̇ = Δmargin/Δt and ρ̈ = Δ²margin/Δt².
A sustained negative ρ̇ (margin falling during generation) is a signal
that the model is drifting toward compliance mid-response.

References: Zhao et al. arXiv:2507.11878; arXiv:2509.09708.
In scaffolding mode (Phase A), operates on synthetic margin sequences.
"""

import numpy as np
import numpy.typing as npt

from cliffguard.types import CalibrationTable, GateVerdict, Margin, QuantScheme, Tier


def compute_trajectory(
    margins: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Given a 1-D array of scalar margin values over decoding steps,
    return (rho_dot, rho_ddot):
      rho_dot  = mean of first differences (margins[1:] - margins[:-1])
      rho_ddot = mean of second differences
    Raises ValueError if len(margins) < 3.
    """
    if len(margins) < 3:
        raise ValueError(
            f"margins must have at least 3 elements to compute trajectory, got {len(margins)}"
        )
    first_diffs = np.diff(margins)
    second_diffs = np.diff(first_diffs)
    return float(np.mean(first_diffs)), float(np.mean(second_diffs))


def evaluate(
    margins: npt.NDArray[np.float64],
    calibration: CalibrationTable,
    scheme: QuantScheme,
    tier: Tier,
) -> tuple[Margin, GateVerdict]:
    """Evaluate the MT gate.

    score = rho_dot (first derivative of margin trajectory).
    fired = True if rho_dot < calibration.tau(scheme).
    (Negative rho_dot means margin is falling — model drifting toward
    compliance.)
    gate name is "PROBE-MT".
    Margin.value is rho_dot.
    Margin.primitive is "PROBE-MT".
    Margin.layer is None.
    """
    rho_dot, _ = compute_trajectory(margins)
    threshold = calibration.tau(scheme)
    margin = Margin(
        value=rho_dot,
        scheme=scheme,
        primitive="PROBE-MT",
        layer=None,
    )
    verdict = GateVerdict(
        gate="PROBE-MT",
        fired=rho_dot < threshold,
        score=rho_dot,
        threshold=threshold,
        tier=tier,
        threat_model=None,
    )
    return margin, verdict
