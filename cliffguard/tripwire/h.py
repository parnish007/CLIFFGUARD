"""TRIPWIRE-H streaming entropy change-point monitor — see blueprint §5.4.

Applies a CUSUM (cumulative sum) control chart to the per-token Shannon
entropy H(y_t) of the model's output distribution. A sustained drop in
token entropy signals that the model has locked onto a low-entropy
(compliant / scripted) response trajectory — the signature of a
successful injection.

Per blueprint §5.4: ARL_0 ≈ 1000 (expected false-alarm run length on
benign traffic). CUSUM threshold h and reference value k are calibrated
per-quantization scheme.

In scaffolding mode (Phase A), operates on synthetic entropy sequences.
"""

import numpy as np
import numpy.typing as npt

from cliffguard.types import CalibrationTable, GateVerdict, QuantScheme, Tier


def token_entropy(logprobs: npt.NDArray[np.float64]) -> float:
    """Compute Shannon entropy H = -sum(p * log2(p)) from a 1-D array
    of log-probabilities (base e). Probabilities are recovered via
    exp(logprobs). Values of zero probability are skipped (0 * log 0
    = 0 by convention). Returns entropy in bits.
    """
    probs = np.exp(logprobs)
    nonzero = probs > 0.0
    if not nonzero.any():
        return 0.0
    p = probs[nonzero]
    return float(-np.sum(p * np.log2(p)))


def cusum_statistic(
    entropies: npt.NDArray[np.float64],
    k: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Compute the one-sided lower CUSUM statistic sequence S_t:
      S_0 = 0
      S_t = max(0, S_{t-1} - (H_t - k))
    A rising S_t signals a sustained drop in entropy below reference k.
    Returns the full sequence as a numpy array of the same length as
    entropies.
    """
    n = len(entropies)
    s = np.zeros(n, dtype=np.float64)
    for i in range(n):
        prev = s[i - 1] if i > 0 else 0.0
        s[i] = max(0.0, prev - (float(entropies[i]) - k))
    return s


def evaluate(
    entropies: npt.NDArray[np.float64],
    calibration: CalibrationTable,
    scheme: QuantScheme,
    tier: Tier,
    k: float = 0.5,
) -> GateVerdict:
    """Evaluate the TRIPWIRE-H gate.

    score = final value of cusum_statistic(entropies, k).
    fired = True if score > calibration.tau(scheme).
    gate name is "TRIPWIRE-H".
    threat_model is None.
    """
    if len(entropies) == 0:
        raise ValueError("entropies must be non-empty")
    cusum = cusum_statistic(entropies, k)
    score = float(cusum[-1])
    threshold = calibration.tau(scheme)
    return GateVerdict(
        gate="TRIPWIRE-H",
        fired=score > threshold,
        score=score,
        threshold=threshold,
        tier=tier,
        threat_model=None,
    )
