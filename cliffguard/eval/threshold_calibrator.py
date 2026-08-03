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
    fires_high: bool = True,
) -> float:
    """Find tau_q such that FPR <= fpr_target on the given benign scores.

    The tail depends on the gate's firing direction, and getting it wrong
    inverts the FPR (a 0.05 target becomes a realised 0.95):

      fires_high=True  (fired = score > tau):
          tau = percentile(scores, (1 - fpr_target) * 100)   [upper tail]
      fires_high=False (fired = score < tau):
          tau = percentile(scores, fpr_target * 100)         [lower tail]

    Fires-HIGH: VESTIBULE-LZ, VESTIBULE-PS, PROBE-HD, TRIPWIRE-H,
                LOOKOUT-CT, LOOKOUT-JG, B-PROBE-LOGIT.
    Fires-LOW:  PROBE-RM, PROBE-MT, TRIPWIRE-R, B-PROBE-CONSISTENCY.

    Tie/interpolation convention. Discontinuous order-statistic quantiles are
    used rather than linear interpolation, because the gate rules are strict
    (`>` / `<`) and linear interpolation can land strictly between adjacent
    order statistics, firing on ceil((n-1)*alpha) observations and OVERSHOOTING
    the target (independently measured: 0.30 realised at n=10, alpha=0.25).
    With `method="lower"` for fires-LOW and `method="higher"` for fires-HIGH,
    each fires on at most floor((n-1)*alpha) observations, so the realised
    same-sample FPR is <= fpr_target.

    Scope of the guarantee. This bounds the SAME-SAMPLE realised rate only. It
    is NOT a population or held-out guarantee — that needs an order-statistic
    tolerance bound or an independent confidence interval. If the score
    distribution has an atom straddling the target quantile, the target may be
    structurally unattainable by any deterministic threshold (randomised
    tie-breaking would be required). Always verify with empirical_fpr() on a
    HELD-OUT benign set.

    Emits a UserWarning if len(scores) < MIN_RELIABLE_SIZE.
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
    if fires_high:
        return float(np.percentile(scores, (1.0 - fpr_target) * 100.0, method="higher"))
    return float(np.percentile(scores, fpr_target * 100.0, method="lower"))


"""Primitives whose gate fires when the score falls BELOW tau.

Calibrating these against the upper tail inverts the FPR — see D0 in
docs/research_audit_2026-08.md."""
FIRES_LOW_PRIMITIVES: frozenset[str] = frozenset(
    {"PROBE-RM", "PROBE-MT", "TRIPWIRE-R", "B-PROBE-CONSISTENCY"}
)

"""Primitives whose gate fires when the score rises ABOVE tau."""
FIRES_HIGH_PRIMITIVES: frozenset[str] = frozenset(
    {
        "VESTIBULE-LZ",
        "VESTIBULE-PS",
        "PROBE-HD",
        "TRIPWIRE-H",
        "LOOKOUT-CT",
        "LOOKOUT-JG",
        "B-PROBE-LOGIT",
    }
)


def fires_high_for(primitive: str) -> bool:
    """Return the firing direction for a named primitive.

    FAIL-CLOSED: unknown primitives raise rather than defaulting. A silent
    majority default (fires-HIGH) would turn a typo or a newly added
    lower-tail readout straight back into defect D0, where the realised FPR
    is 1 - target instead of target. Register every new readout explicitly.

    Raises KeyError if the primitive is not registered."""
    if primitive in FIRES_LOW_PRIMITIVES:
        return False
    if primitive in FIRES_HIGH_PRIMITIVES:
        return True
    known = sorted(FIRES_LOW_PRIMITIVES | FIRES_HIGH_PRIMITIVES)
    raise KeyError(
        f"unknown primitive {primitive!r}: firing direction is not registered, "
        f"so the calibration tail cannot be chosen safely. Add it to "
        f"FIRES_LOW_PRIMITIVES or FIRES_HIGH_PRIMITIVES, or pass fires_high "
        f"explicitly. Known primitives: {known}"
    )


def build_calibration_table(
    primitive: str,
    scores_by_scheme: dict[QuantScheme, npt.NDArray[np.float64]],
    fpr_target: float = 0.05,
    fires_high: bool | None = None,
) -> CalibrationTable:
    """Build a CalibrationTable for a primitive by calibrating each
    scheme independently.

    fires_high=None (default) resolves the firing direction from the
    primitive name via fires_high_for(), so PROBE-RM and the other
    fires-LOW readouts are calibrated against the correct tail.
    Pass an explicit bool to override.

    Raises ValueError if scores_by_scheme is empty."""
    if not scores_by_scheme:
        raise ValueError("scores_by_scheme must not be empty")
    resolved = fires_high_for(primitive) if fires_high is None else fires_high
    thresholds: dict[QuantScheme, float] = {
        scheme: calibrate_threshold(scores, fpr_target, fires_high=resolved)
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


def conservative_threshold(
    scores: npt.NDArray[np.float64],
    fpr_target: float = 0.05,
    fires_high: bool = True,
    confidence: float = 0.95,
) -> float:
    """Threshold with a one-sided POPULATION FPR guarantee, not just a
    same-sample one.

    Motivation. `calibrate_threshold`
    is conservative *on the calibration sample*, but it is only a point
    estimator of the population quantile. Measured at n_cal=400, benign N(0,1),
    fpr_target=0.05: held-out mean FPR 0.050, 95% range [0.030, 0.073], and
    **45 % of trials exceeded the 0.05 target**. That is fine for an estimate
    and wrong for a guarantee.

    This function instead picks the largest order statistic k such that
    P(Binomial(n, fpr_target) >= k) <= 1 - confidence, i.e. the realised
    population FPR exceeds fpr_target with probability at most 1 - confidence.
    At n=400 and alpha=0.05 that is about the 13th order statistic (expected
    tail ~3.24 %) rather than the 20th, so the price of the guarantee is a
    visibly stricter threshold and lower TPR.

    Use this when an FPR budget must actually hold on unseen traffic. Use
    `calibrate_threshold` when reporting a calibrated operating point and
    quoting a held-out interval alongside it.

    Raises ValueError on an empty sample or out-of-range parameters."""
    from scipy.stats import binom

    n = len(scores)
    if n == 0:
        raise ValueError("scores must be non-empty")
    if not (0.0 < fpr_target < 1.0):
        raise ValueError(f"fpr_target must be in (0.0, 1.0), got {fpr_target}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0.0, 1.0), got {confidence}")

    # If tau is the k-th most extreme observation, the population exceedance
    # probability p satisfies p ~ Beta(k, n-k+1), so
    #     P(p > fpr_target) = P(Bin(n, fpr_target) <= k-1) = binom.cdf(k-1, ...)
    # which is INCREASING in k. We therefore want the LARGEST k whose
    # cdf stays within the risk budget: bigger k means a less extreme (weaker)
    # threshold, so this picks the loosest threshold that still guarantees.
    budget = 1.0 - confidence
    k = 0
    for candidate in range(1, n + 1):
        if float(binom.cdf(candidate - 1, n, fpr_target)) <= budget:
            k = candidate
        else:
            break
    if k == 0:
        # n is too small for any order statistic to attain the guarantee.
        # Use the single most extreme observation and let empirical_fpr on a
        # held-out set reveal the shortfall rather than hiding it.
        k = 1

    ordered = np.sort(scores)
    if fires_high:
        # Fire above tau; allow at most k-1 of n benign scores to exceed it.
        idx = max(0, n - k)
        return float(ordered[min(idx, n - 1)])
    # Fire below tau; allow at most k-1 benign scores beneath it.
    return float(ordered[max(0, k - 1)])
