"""Discriminability (d-prime) for behavioural readouts — replaces Delta_cliff.

Defect D1 in docs/research_audit_2026-08.md: Delta_cliff is defined three
mutually inconsistent ways across the blueprint, docs/math.md, the code, and
docs/paper/main.tex, and the paper reports a value its own equation does not
produce. This module supplies the replacement.

d-prime is the standard signal-detection separation between two
class-conditional score distributions:

    d' = (mu_pos - mu_neg) / sqrt((var_pos + var_neg) / 2)

It is well defined, comparable across behaviours and quantization schemes,
and is the quantity the discriminability-decay theorem is stated in:

    d'(q) = d'(FP16) * (1 + eta_q)^(-1/2)

Sign convention. d' is reported so that LARGER means BETTER separated,
independent of the readout's firing direction. For a fires-LOW readout such
as PROBE-RM (harmful prompts sit at LOWER margin), pass fires_high=False and
the sign is handled internally.

Invariance, stated correctly. d' is invariant under a common POSITIVE AFFINE
rescaling of both classes, ``s -> a*s + b`` with ``a > 0``, because mean and
standard deviation scale together and their ratio is unchanged.

It is NOT invariant under an arbitrary strictly increasing transform. That
stronger property belongs to the ROC and hence to AUC: a monotone map moves
every threshold to an image threshold and preserves ranks, but it distorts
means and variances, so ``(mu_pos - mu_neg) / sigma_pooled`` changes. An
earlier version of this docstring claimed the stronger property; it was wrong,
and the error mattered because it was used to argue that the norm-normalised
margin ``<a, r> / ||a||`` could not affect d'. Per-example division by
``||a(x)||`` is not a common affine map — the divisor varies by example — so
that argument does not hold and the normalisation had to be tested empirically
instead (see scripts/analyse_margin_normalisation.py).

What survives, and is what the project actually needs: threshold
recalibration cannot restore lost d', because moving a single threshold slides
along the ROC rather than changing it. Fixing defect D0 is therefore necessary
but not sufficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

FloatArray = npt.NDArray[np.float64]

DEFAULT_N_BOOTSTRAP: int = 2000


@dataclass(frozen=True)
class Discriminability:
    """d-prime with a bootstrap confidence interval and derived quantities."""

    d_prime: float
    ci_low: float
    ci_high: float
    n_pos: int
    n_neg: int
    fires_high: bool

    @property
    def auc(self) -> float:
        """AUC implied by the equal-variance Gaussian model: Phi(d'/sqrt(2))."""
        return float(norm.cdf(self.d_prime / math.sqrt(2.0)))

    def tpr_at_fpr(self, fpr: float) -> float:
        """Predicted TPR at a fixed operating FPR: Phi(d' - z_{1-fpr}).

        This is the composition rule for a single-threshold behaviour."""
        if not (0.0 < fpr < 1.0):
            raise ValueError(f"fpr must be in (0, 1), got {fpr}")
        return float(norm.cdf(self.d_prime - norm.ppf(1.0 - fpr)))

    def summary(self) -> str:
        return (
            f"d' = {self.d_prime:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}] "
            f"(n_pos={self.n_pos}, n_neg={self.n_neg}), "
            f"AUC={self.auc:.3f}, TPR@5%FPR={self.tpr_at_fpr(0.05):.3f}"
        )


def d_prime(
    pos_scores: FloatArray,
    neg_scores: FloatArray,
    fires_high: bool = True,
) -> float:
    """Pooled-variance d-prime between two score samples.

    pos_scores: scores for the class the readout should detect (harmful).
    neg_scores: scores for the benign class.
    fires_high: True if the detected class scores HIGHER; False if it scores
        LOWER (e.g. PROBE-RM refusal margin). The returned d' is positive
        when the classes are separated in the expected direction.

    Raises ValueError if either sample has fewer than 2 elements (variance
    is undefined) or if the pooled variance is zero."""
    if pos_scores.size < 2 or neg_scores.size < 2:
        raise ValueError(
            f"need >= 2 samples per class, got pos={pos_scores.size}, "
            f"neg={neg_scores.size}"
        )
    # ddof=1: unbiased sample variance.
    var_pos = float(np.var(pos_scores, ddof=1))
    var_neg = float(np.var(neg_scores, ddof=1))
    pooled = math.sqrt((var_pos + var_neg) / 2.0)
    if pooled == 0.0:
        raise ValueError("pooled standard deviation is zero — d' undefined")
    diff = float(np.mean(pos_scores) - np.mean(neg_scores))
    return (diff if fires_high else -diff) / pooled


def d_prime_with_ci(
    pos_scores: FloatArray,
    neg_scores: FloatArray,
    fires_high: bool = True,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 0,
) -> Discriminability:
    """d-prime with a percentile bootstrap confidence interval.

    The bootstrap resamples each class independently with replacement, which
    is the right structure here: the two samples are unpaired.

    A CI is mandatory rather than decorative — the headline claim of the
    project is that d' declines along a quantization ladder, and without
    intervals a monotone-looking sequence of point estimates is not evidence."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    point = d_prime(pos_scores, neg_scores, fires_high=fires_high)

    rng = np.random.default_rng(seed)
    n_p, n_n = pos_scores.size, neg_scores.size
    draws: list[float] = []
    for _ in range(n_bootstrap):
        bp = pos_scores[rng.integers(0, n_p, n_p)]
        bn = neg_scores[rng.integers(0, n_n, n_n)]
        try:
            draws.append(d_prime(bp, bn, fires_high=fires_high))
        except ValueError:
            # Degenerate resample (zero pooled variance) — skip it rather
            # than let it poison the interval.
            continue
    if not draws:
        raise ValueError("all bootstrap resamples were degenerate")

    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return Discriminability(
        d_prime=point,
        ci_low=lo,
        ci_high=hi,
        n_pos=int(n_p),
        n_neg=int(n_n),
        fires_high=fires_high,
    )


def empirical_auc(
    pos_scores: FloatArray,
    neg_scores: FloatArray,
    fires_high: bool = True,
) -> float:
    """Nonparametric AUC (Mann-Whitney U statistic), ties counted as 0.5.

    Reported alongside the Gaussian-model AUC: a large gap between the two is
    direct evidence that the equal-variance Gaussian assumption behind the
    theorem does not hold for this readout, which is exactly the kind of
    assumption failure the plan requires us to surface rather than hide."""
    if pos_scores.size == 0 or neg_scores.size == 0:
        raise ValueError("both score arrays must be non-empty")
    pos = pos_scores if fires_high else -pos_scores
    neg = neg_scores if fires_high else -neg_scores
    # Rank-based U: mean over all pairs of 1[pos > neg] + 0.5 * 1[pos == neg].
    comparisons = pos[:, None] - neg[None, :]
    wins = float(np.sum(comparisons > 0.0))
    ties = float(np.sum(comparisons == 0.0))
    return (wins + 0.5 * ties) / float(pos.size * neg.size)


def gaussianity_gap(
    pos_scores: FloatArray,
    neg_scores: FloatArray,
    fires_high: bool = True,
) -> float:
    """|empirical AUC - Gaussian-model AUC|.

    This is a descriptive model-mismatch statistic, not a calibrated
    Gaussianity verdict. The documented value 0.05 is a practical effect-size
    reference whose Type I error and power depend strongly on sample size and
    the alternative. In particular, the statistic is mathematically blind to
    unequal-variance Gaussian classes: both the empirical AUC and
    ``Phi(d'/sqrt(2))`` depend on ``var_pos + var_neg``, even though unequal
    variances violate A2.

    A large gap is useful evidence against the closed-form ROC model. A small
    gap cannot validate A2; inspect the raw distributions and separately test
    variance equality before trusting closed-form TPR predictions.
    """
    emp = empirical_auc(pos_scores, neg_scores, fires_high=fires_high)
    dp = d_prime(pos_scores, neg_scores, fires_high=fires_high)
    model = float(norm.cdf(dp / math.sqrt(2.0)))
    return abs(emp - model)


def predict_d_prime(d_prime_fp16: float, eta: float) -> float:
    """The discriminability-decay law: d'(q) = d'_0 * (1 + eta)^(-1/2).

    eta is the quantization noise-to-signal ratio on the behavioural
    subspace, measured from weights alone (see eval/noise_spectrum.py)."""
    if eta < 0.0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    return d_prime_fp16 / math.sqrt(1.0 + eta)


def implied_eta(d_prime_fp16: float, d_prime_q: float) -> float:
    """Invert the decay law: eta = (d'_0 / d'_q)^2 - 1.

    Lets a measured d' pair be compared against an independently measured
    eta from the weights. Agreement between the two is the core validation
    of the mechanism; disagreement falsifies it."""
    if d_prime_q <= 0.0:
        raise ValueError(f"d_prime_q must be > 0, got {d_prime_q}")
    if d_prime_fp16 <= 0.0:
        raise ValueError(f"d_prime_fp16 must be > 0, got {d_prime_fp16}")
    return (d_prime_fp16 / d_prime_q) ** 2 - 1.0


def held_out_d_prime(
    pos_activations: FloatArray,
    neg_activations: FloatArray,
    n_splits: int = 20,
    fires_high: bool = True,
    seed: int = 0,
) -> tuple[float, float]:
    """d-prime with the direction fitted on one half and scored on the other.

    Fitting the direction on the same prompts it then scores makes d' optimistically
    biased. Measured on Qwen2.5-1.5B-Instruct at layer 14, n=200/class: in-sample
    d' = 0.642 shrinks to 0.507 +/- 0.115 held out. That shrinkage is the
    overfitting of the direction estimator, and it is large enough to change
    conclusions, so held-out is the default for any reported d'.

    Both inputs are (n, D) activation matrices for the two classes. The direction
    is the difference-in-means of the fitted half, so the detected class scores
    HIGH on it (a harmfulness-style direction); pass fires_high accordingly.

    Returns (mean, std) of d' over `n_splits` random half-splits.
    Raises ValueError if either class has fewer than 4 rows."""
    if pos_activations.shape[0] < 4 or neg_activations.shape[0] < 4:
        raise ValueError("need >= 4 samples per class for a held-out split")
    if pos_activations.shape[1] != neg_activations.shape[1]:
        raise ValueError(
            f"dimension mismatch: {pos_activations.shape[1]} vs {neg_activations.shape[1]}"
        )
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")

    def _margins(acts: FloatArray, direction: FloatArray) -> FloatArray:
        d = direction / np.linalg.norm(direction)
        return np.asarray((acts @ d) / np.linalg.norm(acts, axis=1), dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_p, n_n = pos_activations.shape[0], neg_activations.shape[0]
    half_p, half_n = n_p // 2, n_n // 2
    values: list[float] = []
    for _ in range(n_splits):
        pp, pn = rng.permutation(n_p), rng.permutation(n_n)
        fit_p, sc_p = pp[:half_p], pp[half_p : 2 * half_p]
        fit_n, sc_n = pn[:half_n], pn[half_n : 2 * half_n]
        direction = pos_activations[fit_p].mean(axis=0) - neg_activations[fit_n].mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            continue
        try:
            values.append(
                d_prime(
                    _margins(pos_activations[sc_p], direction),
                    _margins(neg_activations[sc_n], direction),
                    fires_high=fires_high,
                )
            )
        except ValueError:
            continue
    if not values:
        raise ValueError("all held-out splits were degenerate")
    return float(np.mean(values)), float(np.std(values))
