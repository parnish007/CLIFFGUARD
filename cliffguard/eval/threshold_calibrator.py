"""Per-quantization threshold calibrator — see blueprint §12.4, §14.1.

Fits the calibration threshold tau_q for each primitive and each
quantization scheme by finding the score percentile that achieves the
target FPR on Fold A benign prompts.

Per Theorem 14.1 (FPR-decoupling): the FPR of the gate is independent
of quantization scheme up to the calibration map P_q. This calibrator
constructs P_q: it takes the empirical score distribution on benign
Fold A prompts under scheme q, and sets tau_q = percentile(scores,
(1 - fpr_target) * 100).

Per blueprint §14.4 (Sensitivity Corollary): |C| < 100 inflates the
KS estimation error to O(1/sqrt(|C|)); a warning is emitted when the
calibration corpus is smaller than 100 samples.
"""

import warnings

import numpy as np
import numpy.typing as npt

from cliffguard.types import CalibrationTable, QuantScheme

MIN_RELIABLE_SIZE: int = 100


def calibrate_threshold(
    scores: npt.NDArray[np.float64],
    fpr_target: float = 0.05,
) -> float:
    """Find tau_q such that FPR <= fpr_target on the given scores.
    tau_q = np.percentile(scores, (1 - fpr_target) * 100).
    Emits a UserWarning if len(scores) < MIN_RELIABLE_SIZE, citing
    blueprint §14.4.
    Raises ValueError if scores is empty.
    Raises ValueError if fpr_target not in (0.0, 1.0)."""
    if len(scores) == 0:
        raise ValueError("scores must be non-empty")
    if not (0.0 < fpr_target < 1.0):
        raise ValueError(
            f"fpr_target must be in (0.0, 1.0), got {fpr_target}"
        )
    if len(scores) < MIN_RELIABLE_SIZE:
        warnings.warn(
            f"Calibration corpus has only {len(scores)} samples "
            f"(< {MIN_RELIABLE_SIZE}). Per blueprint §14.4, KS estimation "
            f"error is O(1/sqrt(|C|)); the FPR portability band will be wider.",
            UserWarning,
            stacklevel=2,
        )
    return float(np.percentile(scores, (1.0 - fpr_target) * 100.0))


def build_calibration_table(
    primitive: str,
    scores_by_scheme: dict[QuantScheme, npt.NDArray[np.float64]],
    fpr_target: float = 0.05,
) -> CalibrationTable:
    """Build a CalibrationTable for a primitive by calibrating each
    scheme independently.
    Calls calibrate_threshold for each scheme in scores_by_scheme.
    Returns a CalibrationTable with all schemes populated.
    Raises ValueError if scores_by_scheme is empty."""
    if not scores_by_scheme:
        raise ValueError("scores_by_scheme must not be empty")
    thresholds: dict[QuantScheme, float] = {
        scheme: calibrate_threshold(scores, fpr_target)
        for scheme, scores in scores_by_scheme.items()
    }
    return CalibrationTable(
        primitive=primitive,
        thresholds=thresholds,
        fpr_target=fpr_target,
    )


def empirical_fpr(
    scores: npt.NDArray[np.float64],
    threshold: float,
    fires_high: bool = True,
) -> float:
    """Compute the empirical FPR on a set of benign scores.
    If fires_high: FPR = mean(scores > threshold).
    If not fires_high: FPR = mean(scores < threshold).
    Returns a float in [0.0, 1.0].
    Raises ValueError if scores is empty."""
    if len(scores) == 0:
        raise ValueError("scores must be non-empty")
    if fires_high:
        return float(np.mean(scores > threshold))
    return float(np.mean(scores < threshold))
