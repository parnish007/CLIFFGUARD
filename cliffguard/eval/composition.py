"""Composition rules and collapse-bit-width prediction — docs/theorems.md §2-§3.

Theorem 1 gives the shared mechanism: discriminability decays as
d'(q) = d'_0 (1 + eta_q)^(-1/2). What differs between behavioural domains is
not the mechanism but the COMPOSITION RULE by which per-instance decisions
aggregate into the observed behaviour:

  THRESHOLD  single decision            TPR = Phi(d' - z_{1-alpha})
  CHAIN(T)   T-fold conjunction         P_T = p^T,  p = Phi(d' - z)
  TAIL(p0)   rare event at depth Delta  log p_q / log p_0 -> 1/(1+eta)

Composing a smooth, monotone d'(b) with a sigmoidal readout, and with
quantization noise growing exponentially in removed bits (eta ~ beta^(4-b)),
produces an apparently sharp cliff whose location is predictable:

    b* = 4 - log_beta( ((d'_0 / z_{1-alpha})^2 - 1) / eta_4 )

Two consequences this module exists to compute:

  * Corollary 1.2 — b*(4T) - b*(T) -> 1 bit. Each 4x increase in reasoning
    chain length costs one additional bit of precision.
  * Corollary 2.1 — the ORDER in which sectors collapse as precision falls.
    Long chains and rare failures break before single-shot decisions do.

SCOPE. Everything here inherits assumptions A1-A3 from docs/theorems.md. In
particular CHAIN assumes conditionally independent, comparable steps, which is
false for real chain-of-thought (self-correction, correlated errors); and TAIL
requires genuine tail depth (Delta >~ 2-3). Report where these bend rather
than extrapolating through them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from scipy.stats import norm

DEFAULT_BASE: float = 4.0
"""Noise growth per bit removed. 4.0 holds for uniform non-saturating scalar
quantization. It is FITTED, never assumed — see eval/noise_spectrum."""

SEARCH_LO_BITS: float = 1.0
SEARCH_HI_BITS: float = 16.0
BISECTION_BITS_TOLERANCE: float = 1e-10


class Composition(Enum):
    """How per-instance decisions aggregate into an observed behaviour."""

    THRESHOLD = "THRESHOLD"
    CHAIN = "CHAIN"
    TAIL = "TAIL"


def eta_at_bits(bits: float, eta_4: float, base: float = DEFAULT_BASE) -> float:
    """Noise-to-signal ratio at effective bit-width `bits`.

    eta(b) = eta_4 * base^(4 - b), anchored at 4 bits. Assumption A3."""
    if eta_4 < 0.0:
        raise ValueError(f"eta_4 must be >= 0, got {eta_4}")
    if base <= 1.0:
        raise ValueError(f"base must be > 1, got {base}")
    return float(eta_4 * (base ** (4.0 - bits)))


def d_prime_at_bits(
    bits: float, d_prime_0: float, eta_4: float, base: float = DEFAULT_BASE
) -> float:
    """Theorem 1 evaluated along a bit-width ladder."""
    if d_prime_0 <= 0.0:
        raise ValueError(f"d_prime_0 must be > 0, got {d_prime_0}")
    return d_prime_0 / math.sqrt(1.0 + eta_at_bits(bits, eta_4, base))


# ---------------------------------------------------------------------------
# Composition readouts
# ---------------------------------------------------------------------------


def threshold_performance(d_prime: float, alpha: float = 0.05) -> float:
    """Corollary 1.1: TPR at a fixed operating FPR."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(norm.cdf(d_prime - norm.ppf(1.0 - alpha)))


def chain_performance(d_prime: float, n_steps: int, alpha: float = 0.05) -> float:
    """Corollary 1.2: probability a T-step conjunction succeeds.

    P_T = p^T with p = Phi(d' - z). The product law itself is textbook
    series-system reliability and is NOT a contribution; the content is its
    coupling to the noise model via d'(eta)."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    return float(threshold_performance(d_prime, alpha) ** n_steps)


def tail_rate(p0: float, eta: float) -> float:
    """Corollary 1.3: inflated rate of a rare failure under quantization.

    Uses the exact Gaussian form rather than the asymptotic power law, so it
    stays correct for moderate p0. The asymptotic statement
    log p_q / log p_0 -> 1/(1+eta) is recovered as p0 -> 0."""
    if not (0.0 < p0 < 1.0):
        raise ValueError(f"p0 must be in (0, 1), got {p0}")
    if eta < 0.0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    delta = norm.ppf(1.0 - p0)
    return float(norm.sf(delta / math.sqrt(1.0 + eta)))


def tail_inflation(p0: float, eta: float) -> float:
    """Fold-change p_q / p_0. Grows with rarity at fixed eta — the mechanism
    behind 'quality is not a safety proxy under quantization'."""
    return tail_rate(p0, eta) / p0


# ---------------------------------------------------------------------------
# Collapse prediction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollapsePrediction:
    """Predicted bit-width at which a behaviour collapses."""

    bits: float
    composition: Composition
    label: str
    d_prime_0: float
    eta_4: float
    base: float
    retention: float

    def summary(self) -> str:
        if math.isnan(self.bits):
            return (
                f"{self.label:28s} no collapse crossing in "
                f"[{SEARCH_LO_BITS:g}, {SEARCH_HI_BITS:g}] bits "
                f"({self.composition.value}, d'_0={self.d_prime_0:.2f}, "
                f"eta_4={self.eta_4:.2f}, base={self.base:g})"
            )
        return (
            f"{self.label:28s} b* = {self.bits:5.2f} bits "
            f"({self.composition.value}, d'_0={self.d_prime_0:.2f}, "
            f"eta_4={self.eta_4:.2f}, base={self.base:g})"
        )


def collapse_bits_threshold_closed_form(
    d_prime_0: float, alpha: float = 0.05, eta_4: float = 0.30, base: float = DEFAULT_BASE
) -> float:
    """Theorem 2 in closed form: the bit-width where d'(b) = z_{1-alpha}.

    b* = 4 - log_base( ((d'_0/z)^2 - 1) / eta_4 )

    Raises ValueError if d'_0 <= z_{1-alpha}, i.e. the behaviour is already at
    or below its operating threshold at full precision, so no crossing exists."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if base <= 1.0:
        raise ValueError(f"base must be > 1, got {base}")
    z = float(norm.ppf(1.0 - alpha))
    ratio = (d_prime_0 / z) ** 2 - 1.0
    if ratio <= 0.0:
        raise ValueError(
            f"d_prime_0={d_prime_0} is at or below z_{{1-alpha}}={z:.3f}; "
            "the behaviour never crosses its threshold, so b* is undefined"
        )
    if eta_4 <= 0.0:
        raise ValueError(f"eta_4 must be > 0, got {eta_4}")
    return 4.0 - math.log(ratio / eta_4) / math.log(base)


def _bisect_collapse(
    performance_at_bits: Callable[[float], float],
    target: float,
    base_hi: float = SEARCH_HI_BITS,
) -> float:
    """Find the crossing where performance rises through ``target``.

    ``performance_at_bits`` must be non-decreasing in b (more bits, no
    worse). A finite value is returned only when the target is bracketed
    strictly inside the search interval. Flat curves and crossings outside
    the interval return NaN instead of fabricating a boundary at 1 or 16 bits.
    """
    lo, hi = SEARCH_LO_BITS, base_hi
    if not math.isfinite(target):
        raise ValueError(f"target must be finite, got {target}")
    value_lo = performance_at_bits(lo)
    value_hi = performance_at_bits(hi)
    if not math.isfinite(value_lo) or not math.isfinite(value_hi):
        raise ValueError("performance curve must be finite at the search bounds")

    scale = max(1.0, abs(value_lo), abs(value_hi), abs(target))
    value_tolerance = 32.0 * math.ulp(scale)
    if value_hi < value_lo - value_tolerance:
        raise ValueError(
            "performance_at_bits must be non-decreasing over the search interval"
        )
    if abs(value_hi - value_lo) <= value_tolerance:
        return float("nan")
    if target <= value_lo + value_tolerance:
        return float("nan")  # crossing is below the search range (or absent)
    if target > value_hi + value_tolerance:
        return float("nan")  # never reaches the target at high precision

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        value_mid = performance_at_bits(mid)
        if not math.isfinite(value_mid):
            raise ValueError("performance curve returned a non-finite value")
        if value_mid < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= BISECTION_BITS_TOLERANCE:
            break
    return 0.5 * (lo + hi)


def predict_collapse(
    composition: Composition,
    d_prime_0: float,
    eta_4: float = 0.30,
    base: float = DEFAULT_BASE,
    alpha: float = 0.05,
    retention: float = 0.5,
    n_steps: int = 1,
    p0: float = 1e-3,
    label: str = "",
) -> CollapsePrediction:
    """Predict the collapse bit-width for a behaviour under its composition rule.

    `retention` is the fraction of the behaviour's own FULL-PRECISION value that
    defines collapse. A relative criterion is required rather than an absolute
    one: for large n_steps an absolute bar becomes unreachable from ceiling
    effects alone, which would confound chain length with quantization.

    For TAIL the criterion is inverted — collapse is where the failure rate has
    grown by 1/retention (default: doubled)."""
    if not isinstance(composition, Composition):
        raise ValueError(
            f"composition must be a Composition value, got {composition!r}"
        )
    if not (0.0 < retention < 1.0):
        raise ValueError(f"retention must be in (0, 1), got {retention}")

    if composition is Composition.THRESHOLD:

        def threshold_curve(b: float) -> float:
            return threshold_performance(d_prime_at_bits(b, d_prime_0, eta_4, base), alpha)

        ref = threshold_curve(SEARCH_HI_BITS)
        bits = _bisect_collapse(threshold_curve, retention * ref)
    elif composition is Composition.CHAIN:

        def chain_curve(b: float) -> float:
            return chain_performance(
                d_prime_at_bits(b, d_prime_0, eta_4, base), n_steps, alpha
            )

        ref = chain_curve(SEARCH_HI_BITS)
        bits = _bisect_collapse(chain_curve, retention * ref)
    else:
        # TAIL — the failure RATE grows as bits fall, so negate it to keep the
        # bisection helper's "non-decreasing in b" contract.
        def tail_curve(b: float) -> float:
            return -tail_rate(p0, eta_at_bits(b, eta_4, base))

        bits = _bisect_collapse(tail_curve, -(p0 / retention))

    return CollapsePrediction(
        bits=bits,
        composition=composition,
        label=label or composition.value,
        d_prime_0=d_prime_0,
        eta_4=eta_4,
        base=base,
        retention=retention,
    )


def chain_bit_cost(n_steps: int, base: float = DEFAULT_BASE) -> float:
    """Corollary 1.2: additional bits required for an n_steps chain relative to
    a single-shot decision, in the asymptotic regime.

    log_base(T). For base 4 this is +1 bit per 4x chain length. The exact
    relation carries an O(1) offset from the linearisation (measured ~ +0.23);
    the INCREMENTS are what converge exactly, so prefer comparing
    b*(4T) - b*(T) against 1.0 rather than absolute values."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if base <= 1.0:
        raise ValueError(f"base must be > 1, got {base}")
    return math.log(n_steps) / math.log(base)


def sector_ordering(
    predictions: list[CollapsePrediction],
) -> list[CollapsePrediction]:
    """Corollary 2.1: sectors sorted by the order in which they collapse as
    precision falls (highest b* first = breaks earliest).

    This ordering is the headline testable prediction: one GGUF ladder, several
    sectors, check whether the measured order matches."""
    return sorted(
        [p for p in predictions if not math.isnan(p.bits)],
        key=lambda p: p.bits,
        reverse=True,
    )
