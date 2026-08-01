"""Uncertainty controls for difference-in-means directions — Stage 0.

This module answers the question that gates the entire project (defect D2 in
docs/research_audit_2026-08.md):

    Is the measured FP16 -> NF4 rotation of the refusal direction a real
    quantization effect, or is it just finite-sample estimation noise?

The Stage 0 gate is the paired FP16-to-quantized direction shift: prompt indices
are resampled synchronously across schemes, and PASS means its BCa interval
excludes zero. The within-FP16 split-half angle is retained as a diagnostic of
estimator instability; it is not the cross-scheme decision.

Sample-size correction. A split-half control uses n/2 prompts per class, so
its noise floor is inflated relative to the full-n estimate under test. The
delta method gives E[angle^2] ~= K/n, and there are TWO distinct corrections
depending on the comparison being made: divide the chord by sqrt(2) for two
independent full-n estimators against each other, or by 2 for one full-n
estimator against the population direction. See `NoiseFloorResult` for the
derivation.

WHICH CONTROL TO USE. The split-half floor is a within-FP16 diagnostic of
estimator instability. It is NOT the right control for the cross-scheme
statistic, because FP16 and the quantized model score the SAME prompts, so
their estimator errors are correlated and the relevant variance carries a
-2 Cov(FP16, q) cross term that no fixed rescaling reproduces. For the
cross-scheme question use `paired_direction_shift()`, which resamples prompt
indices synchronously across schemes and preserves that covariance.

Both controls are provided and the notebook reports both. See
docs/build_log.md (Claude Entry 1 doubts D-b/C3, adjudicated by Codex Entry 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DEFAULT_N_SPLITS: int = 50


def unit(v: FloatArray) -> FloatArray:
    """Return v / ||v||. Raises ValueError on a zero-norm vector."""
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero-norm vector")
    return v / norm


def difference_in_means(
    harmful: FloatArray,
    harmless: FloatArray,
) -> FloatArray:
    """Unit-norm difference-in-means direction (Arditi et al. arXiv:2406.11717).

    harmful and harmless are (n_i, D) activation matrices. Returns the
    normalised mean(harmful) - mean(harmless) vector, shape (D,).
    Raises ValueError if either input is empty or the dimensions disagree."""
    if harmful.ndim != 2 or harmless.ndim != 2:
        raise ValueError("harmful and harmless must be 2-D (n_samples, D)")
    if harmful.shape[0] == 0 or harmless.shape[0] == 0:
        raise ValueError("harmful and harmless must each be non-empty")
    if harmful.shape[1] != harmless.shape[1]:
        raise ValueError(
            f"dimension mismatch: harmful D={harmful.shape[1]} "
            f"vs harmless D={harmless.shape[1]}"
        )
    return unit(harmful.mean(axis=0) - harmless.mean(axis=0))


def angle_between(a: FloatArray, b: FloatArray) -> float:
    """Angle in DEGREES between two vectors. Robust to floating-point
    overshoot of the cosine outside [-1, 1]."""
    cos = float(np.dot(unit(a), unit(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def chord_distance(a: FloatArray, b: FloatArray) -> float:
    """||a_hat - b_hat|| for unit-normalised inputs.

    This is the raw Euclidean distance between unit vectors, in [0, 2].
    Reported unnormalised on purpose: the sqrt(2)-normalised variant used
    elsewhere in this repo (`eval/cliff_metrics.geometric_cliff`) is one of
    three mutually inconsistent definitions of Delta_cliff (defect D1), and
    is being retired in favour of d-prime."""
    return float(np.linalg.norm(unit(a) - unit(b)))


@dataclass(frozen=True)
class NoiseFloorResult:
    """Distribution of the split-half estimator noise floor."""

    angles_deg: list[float] = field(default_factory=list)
    n_per_class_used: int = 0
    n_splits: int = 0
    dimension: int = 0

    @property
    def median_deg(self) -> float:
        return float(np.median(self.angles_deg))

    @property
    def mean_deg(self) -> float:
        return float(np.mean(self.angles_deg))

    def quantile_deg(self, q: float) -> float:
        """Empirical quantile of the noise-floor angle distribution."""
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"q must be in [0, 1], got {q}")
        return float(np.quantile(self.angles_deg, q))

    @property
    def corrected_median_deg(self) -> float:
        """Median split-half angle scaled to a TWO-FULL-N comparison.

        Scaling derivation (delta method). With delta = mu_H - mu_B,
        g = ||delta||, P = I - r r^T and A = Sigma_H + Sigma_B, the normalised
        estimator satisfies r_hat_n - r ~= P e_n / g with Cov(e_n) = A/n, so
        E[angle(r_hat_n, r)^2] ~= K/n for K = tr(P A P)/g^2. Two independent
        estimators of sizes n1, n2 have leading squared angle K(1/n1 + 1/n2).

        With N per class and each split using N/2, the raw split-pair squared
        angle is 4K/N. Hence there are TWO different corrections:

          * two independent full-N estimators vs each other -> raw / sqrt(2)
          * one full-N estimator vs the population direction -> raw / 2

        This property applies the sqrt(2) factor, i.e. it is the floor for the
        FIRST comparison. Use `corrected_vs_population_deg` for the second.
        The factor is applied to the chord and converted back, not treated as
        an exact angular identity.

        IMPORTANT CAVEAT. Neither correction is the right control for the
        cross-scheme statistic, because FP16 and the quantized scheme score the
        SAME prompts and their estimator errors are therefore correlated: the
        relevant variance is A_FP16 + A_q - 2 Cov(FP16, q). Use
        `paired_direction_shift()` for that. This split-half floor remains a
        valid within-FP16 diagnostic of estimator instability."""
        return self._scaled(self.median_deg, math.sqrt(2.0))

    def corrected_quantile_deg(self, q: float) -> float:
        """Two-full-N corrected quantile of the noise-floor angle."""
        return self._scaled(self.quantile_deg(q), math.sqrt(2.0))

    @property
    def corrected_vs_population_deg(self) -> float:
        """Median split-half angle scaled to ONE full-N estimator versus the
        population direction (divide by 2, not sqrt(2))."""
        return self._scaled(self.median_deg, 2.0)

    @staticmethod
    def _scaled(angle_deg: float, factor: float) -> float:
        chord = 2.0 * math.sin(math.radians(angle_deg) / 2.0)
        scaled = chord / factor
        return math.degrees(2.0 * math.asin(max(-1.0, min(1.0, scaled / 2.0))))

    def exceeds_floor(self, observed_deg: float, q: float = 0.95) -> bool:
        """Compare an angle with the corrected split-half diagnostic floor.

        This method is retained for descriptive/backward-compatible use. It is
        not the Stage 0 gate because it discards the cross-scheme covariance.
        Stage 0 is decided by ``paired_direction_shift(...).excludes_zero``.
        """
        return observed_deg > self.corrected_quantile_deg(q)

    def summary(self) -> str:
        return (
            f"split-half noise floor over {self.n_splits} splits "
            f"(n={self.n_per_class_used}/class, D={self.dimension}): "
            f"median {self.median_deg:.2f} deg "
            f"(corrected to full-n: {self.corrected_median_deg:.2f} deg), "
            f"95th pct {self.quantile_deg(0.95):.2f} deg "
            f"(corrected: {self.corrected_quantile_deg(0.95):.2f} deg)"
        )


def split_half_noise_floor(
    harmful: FloatArray,
    harmless: FloatArray,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = 0,
) -> NoiseFloorResult:
    """Estimate the estimator noise floor by repeated disjoint split-half
    re-extraction of the difference-in-means direction.

    For each split: partition the harmful rows into two disjoint halves and
    the harmless rows likewise, extract a direction from (harmful_A,
    harmless_A) and another from (harmful_B, harmless_B), and record the
    angle between them. Quantization is held fixed throughout, so the only
    source of variation is finite-sample noise.

    Repeated splits are used because a single split is one draw from a random
    variable. The resulting distribution is a within-scheme diagnostic, not
    the Stage 0 cross-scheme gate.

    Raises ValueError if either class has fewer than 4 samples (each half
    needs at least 2 to have any within-half variation)."""
    if harmful.shape[0] < 4 or harmless.shape[0] < 4:
        raise ValueError(
            f"need >= 4 samples per class for a split-half control, got "
            f"harmful={harmful.shape[0]}, harmless={harmless.shape[0]}"
        )
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")

    rng = np.random.default_rng(seed)
    angles: list[float] = []
    n_h, n_l = harmful.shape[0], harmless.shape[0]
    half_h, half_l = n_h // 2, n_l // 2

    for _ in range(n_splits):
        perm_h = rng.permutation(n_h)
        perm_l = rng.permutation(n_l)
        a = difference_in_means(
            harmful[perm_h[:half_h]], harmless[perm_l[:half_l]]
        )
        b = difference_in_means(
            harmful[perm_h[half_h : 2 * half_h]],
            harmless[perm_l[half_l : 2 * half_l]],
        )
        angles.append(angle_between(a, b))

    return NoiseFloorResult(
        angles_deg=angles,
        n_per_class_used=min(half_h, half_l),
        n_splits=n_splits,
        dimension=int(harmful.shape[1]),
    )


@dataclass(frozen=True)
class PairedShiftResult:
    """Percentile and BCa intervals for the paired direction shift.

    This is the CORRECT control for the cross-scheme statistic. The split-half
    floor is a within-FP16 diagnostic of estimator instability, but it does not
    isolate the sampling contribution to the *paired* FP16-vs-NF4 angle: it uses
    half-sized estimators and, more importantly, discards the positive
    covariance induced by scoring the SAME prompts under both schemes. The
    relevant variance is Sigma_FP16 + Sigma_q - 2 Cov(FP16, q); no fixed
    rescaling of a within-FP16 floor captures that cross term.

    Here prompt indices are resampled SYNCHRONOUSLY across schemes, so the
    covariance is preserved and the resulting interval is for the quantity
    actually being interpreted.

    ``ci_low_deg`` and ``ci_high_deg`` retain the original percentile interval.
    ``bca_ci_low_deg`` and ``bca_ci_high_deg`` add bias correction and
    jackknife acceleration. The Stage 0 ``excludes_zero`` decision uses BCa;
    the percentile interval remains exposed and is never silently replaced.
    Neither bootstrap construction is forced to bracket the point estimate.
    """

    observed_angle_deg: float
    ci_low_deg: float
    ci_high_deg: float
    bca_ci_low_deg: float
    bca_ci_high_deg: float
    n_bootstrap: int
    n_per_class: int

    @property
    def percentile_ci_low_deg(self) -> float:
        """Explicit alias for the original percentile lower endpoint."""
        return self.ci_low_deg

    @property
    def percentile_ci_high_deg(self) -> float:
        """Explicit alias for the original percentile upper endpoint."""
        return self.ci_high_deg

    @property
    def excludes_zero(self) -> bool:
        """Stage 0 gate: true exactly when the paired BCa CI excludes zero."""
        return self.bca_ci_low_deg > 0.0

    @property
    def percentile_excludes_zero(self) -> bool:
        """Whether the retained percentile interval excludes zero."""
        return self.ci_low_deg > 0.0

    def summary(self) -> str:
        return (
            f"paired FP16->quantized rotation {self.observed_angle_deg:.2f} deg "
            f"(percentile [{self.ci_low_deg:.2f}, {self.ci_high_deg:.2f}], "
            f"BCa [{self.bca_ci_low_deg:.2f}, {self.bca_ci_high_deg:.2f}]) "
            f"({self.n_bootstrap} paired resamples, n={self.n_per_class}/class)"
        )


def paired_direction_shift(
    harmful_fp16: FloatArray,
    harmless_fp16: FloatArray,
    harmful_q: FloatArray,
    harmless_q: FloatArray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedShiftResult:
    """Paired percentile and BCa intervals for the cross-scheme rotation.

    All four activation matrices must be row-aligned: row i of `harmful_fp16`
    and row i of `harmful_q` must be the SAME prompt scored under the two
    schemes. Prompt indices are resampled synchronously, preserving the
    cross-scheme covariance.

    The BCa bias correction is computed from the paired bootstrap distribution;
    its acceleration is computed by leave-one-prompt-out jackknifing within
    each class. Raises ValueError if the paired matrices disagree in shape or
    either class has fewer than two prompts.
    """
    if harmful_fp16.shape != harmful_q.shape:
        raise ValueError(
            f"harmful matrices must be row-aligned: {harmful_fp16.shape} "
            f"vs {harmful_q.shape}"
        )
    if harmless_fp16.shape != harmless_q.shape:
        raise ValueError(
            f"harmless matrices must be row-aligned: {harmless_fp16.shape} "
            f"vs {harmless_q.shape}"
        )
    if harmful_fp16.shape[0] < 2 or harmless_fp16.shape[0] < 2:
        raise ValueError("paired BCa intervals require at least two prompts per class")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    observed = _paired_angle(harmful_fp16, harmless_fp16, harmful_q, harmless_q)

    rng = np.random.default_rng(seed)
    n_h, n_l = harmful_fp16.shape[0], harmless_fp16.shape[0]
    draws: list[float] = []
    for _ in range(n_bootstrap):
        ih = rng.integers(0, n_h, n_h)
        il = rng.integers(0, n_l, n_l)
        try:
            draws.append(
                _paired_angle(
                    harmful_fp16[ih],
                    harmless_fp16[il],
                    harmful_q[ih],
                    harmless_q[il],
                )
            )
        except ValueError:
            continue
    if not draws:
        raise ValueError("all paired bootstrap resamples were degenerate")

    jackknife: list[float] = []
    for index in range(n_h):
        keep = np.arange(n_h) != index
        try:
            jackknife.append(
                _paired_angle(
                    harmful_fp16[keep],
                    harmless_fp16,
                    harmful_q[keep],
                    harmless_q,
                )
            )
        except ValueError:
            continue
    for index in range(n_l):
        keep = np.arange(n_l) != index
        try:
            jackknife.append(
                _paired_angle(
                    harmful_fp16,
                    harmless_fp16[keep],
                    harmful_q,
                    harmless_q[keep],
                )
            )
        except ValueError:
            continue
    if not jackknife:
        raise ValueError("all paired jackknife samples were degenerate")

    percentile_low = float(np.quantile(draws, alpha / 2.0))
    percentile_high = float(np.quantile(draws, 1.0 - alpha / 2.0))
    bca_low, bca_high = _bca_interval(
        observed,
        np.asarray(draws, dtype=np.float64),
        np.asarray(jackknife, dtype=np.float64),
        alpha,
    )

    return PairedShiftResult(
        observed_angle_deg=observed,
        ci_low_deg=percentile_low,
        ci_high_deg=percentile_high,
        bca_ci_low_deg=bca_low,
        bca_ci_high_deg=bca_high,
        n_bootstrap=len(draws),
        n_per_class=int(min(n_h, n_l)),
    )


def _paired_angle(
    harmful_fp16: FloatArray,
    harmless_fp16: FloatArray,
    harmful_q: FloatArray,
    harmless_q: FloatArray,
) -> float:
    return angle_between(
        difference_in_means(harmful_fp16, harmless_fp16),
        difference_in_means(harmful_q, harmless_q),
    )


def _bca_interval(
    observed: float,
    bootstrap: FloatArray,
    jackknife: FloatArray,
    alpha: float,
) -> tuple[float, float]:
    """Return a bias-corrected and accelerated bootstrap interval."""
    normal = NormalDist()
    n_bootstrap = int(bootstrap.size)
    proportion_below = float(np.count_nonzero(bootstrap < observed)) / n_bootstrap
    # Finite-sample clipping keeps inv_cdf defined when every draw lies on the
    # same side of the observed statistic, including the exact-zero boundary.
    tail_clip = 0.5 / n_bootstrap
    proportion_below = min(1.0 - tail_clip, max(tail_clip, proportion_below))
    bias_correction = normal.inv_cdf(proportion_below)

    jackknife_mean = float(np.mean(jackknife))
    centered = jackknife_mean - jackknife
    squared_sum = float(np.dot(centered, centered))
    acceleration = 0.0
    if squared_sum > 0.0:
        acceleration = float(
            np.sum(centered**3) / (6.0 * squared_sum ** 1.5)
        )

    def adjusted_probability(probability: float) -> float:
        z_alpha = normal.inv_cdf(probability)
        combined = bias_correction + z_alpha
        denominator = 1.0 - acceleration * combined
        if abs(denominator) < np.finfo(np.float64).eps:
            denominator = math.copysign(np.finfo(np.float64).eps, denominator)
        adjusted = normal.cdf(bias_correction + combined / denominator)
        return min(1.0, max(0.0, adjusted))

    low_probability = adjusted_probability(alpha / 2.0)
    high_probability = adjusted_probability(1.0 - alpha / 2.0)
    return (
        float(np.quantile(bootstrap, low_probability)),
        float(np.quantile(bootstrap, high_probability)),
    )


@dataclass(frozen=True)
class GateResult:
    """Stage 0 decision: is the cross-scheme rotation larger than what the
    estimator produces when there is NO quantization effect at all?

    WHY `excludes_zero` WAS WRONG (recorded so it is not reinstated). The gate
    was briefly defined as "the paired CI for the cross-scheme angle excludes
    zero". That is unfalsifiable: an angle is a NON-NEGATIVE statistic, so a
    noisy estimate is essentially never zero and its interval essentially never
    contains zero. Null-calibrated, that rule fired on 40/40 trials in which
    the quantized activations differed from FP16 by pure exchangeable noise and
    no systematic rotation. The question is not "is the rotation non-zero?" but
    "is it bigger than the estimator's own noise?", which needs a null
    REFERENCE, not a null VALUE."""

    observed_angle_deg: float
    null_quantile_deg: float
    null_median_deg: float
    alpha: float
    n_null: int
    n_per_class: int

    @property
    def passes(self) -> bool:
        """True if the observed rotation exceeds the same-scheme null."""
        return self.observed_angle_deg > self.null_quantile_deg

    @property
    def ratio_to_null_median(self) -> float:
        if self.null_median_deg == 0.0:
            raise ValueError("null median is zero — cannot form a ratio")
        return self.observed_angle_deg / self.null_median_deg

    def summary(self) -> str:
        verdict = "PASS" if self.passes else "NULL"
        return (
            f"Stage 0 {verdict}: observed {self.observed_angle_deg:.2f} deg vs "
            f"same-scheme null median {self.null_median_deg:.2f} deg, "
            f"{100 * (1 - self.alpha):.0f}th pct {self.null_quantile_deg:.2f} deg "
            f"({self.n_null} draws, n={self.n_per_class}/class); "
            f"ratio {self.ratio_to_null_median:.2f}x"
        )


def same_scheme_null_angles(
    harmful: FloatArray,
    harmless: FloatArray,
    n_null: int = 1000,
    seed: int = 0,
) -> list[float]:
    """Null distribution of the direction angle when there is NO quantization
    effect, at the SAME sample size as the statistic under test.

    Two INDEPENDENT bootstrap resamples are drawn from the same single-scheme
    activations and a direction is estimated from each; the angle between them
    is recorded. Because each resample uses all n rows, this null is directly
    comparable with a cross-scheme angle computed at n — unlike the split-half
    floor, which uses n/2 and needs a scaling correction.

    Raises ValueError if either class has fewer than 2 rows."""
    if harmful.shape[0] < 2 or harmless.shape[0] < 2:
        raise ValueError("need >= 2 samples per class for a bootstrap null")
    if n_null < 1:
        raise ValueError(f"n_null must be >= 1, got {n_null}")

    rng = np.random.default_rng(seed)
    n_h, n_l = harmful.shape[0], harmless.shape[0]
    angles: list[float] = []
    for _ in range(n_null):
        a = difference_in_means(
            harmful[rng.integers(0, n_h, n_h)], harmless[rng.integers(0, n_l, n_l)]
        )
        b = difference_in_means(
            harmful[rng.integers(0, n_h, n_h)], harmless[rng.integers(0, n_l, n_l)]
        )
        angles.append(angle_between(a, b))
    return angles


def stage0_gate(
    harmful_fp16: FloatArray,
    harmless_fp16: FloatArray,
    harmful_q: FloatArray,
    harmless_q: FloatArray,
    alpha: float = 0.05,
    n_null: int = 1000,
    seed: int = 0,
) -> GateResult:
    """The Stage 0 decision (defect D2).

    Compares the observed FP16 -> quantized rotation against the same-scheme
    bootstrap null at matched sample size. PASS means the rotation is larger
    than the estimator produces with no quantization effect; NULL means the
    published 13.56 deg / Delta_cliff = 0.167 is consistent with sampling noise
    and premise P1 is unproven.

    Note the null is computed on the FP16 activations only, so it inherits no
    information from the quantized scheme."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    observed = angle_between(
        difference_in_means(harmful_fp16, harmless_fp16),
        difference_in_means(harmful_q, harmless_q),
    )
    null = same_scheme_null_angles(harmful_fp16, harmless_fp16, n_null=n_null, seed=seed)
    return GateResult(
        observed_angle_deg=observed,
        null_quantile_deg=float(np.quantile(null, 1.0 - alpha)),
        null_median_deg=float(np.median(null)),
        alpha=alpha,
        n_null=len(null),
        n_per_class=int(min(harmful_fp16.shape[0], harmless_fp16.shape[0])),
    )


@dataclass(frozen=True)
class ReplicationResult:
    """Does the FP16 -> quantized rotation REPLICATE across disjoint prompts?

    This is the correct Stage 0 test, arrived at after two wrong designs:

      1. `exceeds_floor` (split-half floor). Wrong sample size, and it ignores
         that FP16 and the quantized model score the SAME prompts, so their
         estimator errors are correlated.
      2. `excludes_zero` (paired CI vs 0). Unfalsifiable — an angle is
         non-negative, so its interval essentially never contains zero. It
         fired on 40/40 null trials.

    The insight both missed: at FIXED prompts, prompt-sampling noise is COMMON
    to both schemes and largely cancels, so the observed rotation is not
    explained by sampling noise in the first place. The real question is
    whether the rotation is SYSTEMATIC (a property of the quantizer) or
    IDIOSYNCRATIC (a property of these particular prompts).

    Test: split the prompts into disjoint halves; on each half compute the
    tangent shift — the component of (r_q - r_FP16) orthogonal to r_FP16 — and
    measure the alignment of the two shift vectors.

      cosine ~ 1  =>  the same rotation is found on independent prompts.
                      The effect is systematic. PASS.
      cosine ~ 0  =>  the halves disagree on WHERE the direction moved.
                      The rotation is prompt-idiosyncratic noise. NULL.

    In D dimensions two unrelated vectors have cosine ~ 0 with standard
    deviation ~1/sqrt(D), so the null is sharp for large D."""

    cosines: list[float] = field(default_factory=list)
    dimension: int = 0
    n_splits: int = 0

    @property
    def median_cosine(self) -> float:
        return float(np.median(self.cosines))

    @property
    def null_sd(self) -> float:
        """SD of the cosine between unrelated vectors in D dimensions."""
        if self.dimension < 2:
            raise ValueError("dimension must be >= 2")
        return 1.0 / math.sqrt(self.dimension)

    @property
    def z_score(self) -> float:
        """Median cosine in units of the random-alignment null SD."""
        return self.median_cosine / self.null_sd

    def passes(self, z_threshold: float = 3.0) -> bool:
        """PASS if the rotation replicates well above chance alignment."""
        return self.z_score > z_threshold

    def summary(self) -> str:
        verdict = "PASS" if self.passes() else "NULL"
        return (
            f"Stage 0 {verdict}: rotation replicates across disjoint prompt "
            f"halves with median cosine {self.median_cosine:+.4f} "
            f"(chance SD {self.null_sd:.4f}, z = {self.z_score:+.1f}) "
            f"over {self.n_splits} splits, D={self.dimension}"
        )


def rotation_replication(
    harmful_fp16: FloatArray,
    harmless_fp16: FloatArray,
    harmful_q: FloatArray,
    harmless_q: FloatArray,
    n_splits: int = 50,
    seed: int = 0,
) -> ReplicationResult:
    """Stage 0: test whether the cross-scheme rotation replicates on disjoint
    prompt halves.

    All four matrices must be row-aligned (row i is the same prompt under both
    schemes). Returns the distribution of cosines between the tangent shifts
    computed on independent halves.

    Raises ValueError on shape mismatch or too few prompts."""
    if harmful_fp16.shape != harmful_q.shape:
        raise ValueError(
            f"harmful matrices must be row-aligned: {harmful_fp16.shape} vs {harmful_q.shape}"
        )
    if harmless_fp16.shape != harmless_q.shape:
        raise ValueError(
            f"harmless matrices must be row-aligned: {harmless_fp16.shape} vs {harmless_q.shape}"
        )
    if harmful_fp16.shape[0] < 4 or harmless_fp16.shape[0] < 4:
        raise ValueError("need >= 4 prompts per class to split into halves")

    def tangent_shift(hf: FloatArray, lf: FloatArray, hq: FloatArray, lq: FloatArray) -> FloatArray:
        r0 = difference_in_means(hf, lf)
        r1 = difference_in_means(hq, lq)
        delta = r1 - r0
        return delta - float(np.dot(delta, r0)) * r0      # orthogonal component

    rng = np.random.default_rng(seed)
    n_h, n_l = harmful_fp16.shape[0], harmless_fp16.shape[0]
    half_h, half_l = n_h // 2, n_l // 2
    cosines: list[float] = []
    for _ in range(n_splits):
        ph, pl = rng.permutation(n_h), rng.permutation(n_l)
        ah, bh = ph[:half_h], ph[half_h : 2 * half_h]
        al, bl = pl[:half_l], pl[half_l : 2 * half_l]
        sa = tangent_shift(
            harmful_fp16[ah], harmless_fp16[al], harmful_q[ah], harmless_q[al]
        )
        sb = tangent_shift(
            harmful_fp16[bh], harmless_fp16[bl], harmful_q[bh], harmless_q[bl]
        )
        na, nb = float(np.linalg.norm(sa)), float(np.linalg.norm(sb))
        if na == 0.0 or nb == 0.0:
            continue
        cosines.append(float(np.dot(sa, sb) / (na * nb)))
    if not cosines:
        raise ValueError("all splits produced a degenerate (zero) tangent shift")

    return ReplicationResult(
        cosines=cosines, dimension=int(harmful_fp16.shape[1]), n_splits=len(cosines)
    )


def theoretical_floor_deg(dimension: int, n_per_class: int, snr: float) -> float:
    """Closed-form small-angle prediction of the split-half noise floor.

    For a difference-in-means estimator with per-class SNR
    g = ||mu_h - mu_l|| / sigma, the angular error of a single estimate is
    approximately asin(sqrt(2 D / n) / g); two independent estimates differ by
    approximately sqrt(2) times that.

    Provided as a sanity check on the empirical estimate — if the two disagree
    badly, the isotropic-equal-covariance assumption behind the theorem is
    already suspect. Returns 90.0 when the estimator is unresolved."""
    if dimension < 1 or n_per_class < 1:
        raise ValueError("dimension and n_per_class must be >= 1")
    if snr <= 0.0:
        raise ValueError(f"snr must be > 0, got {snr}")
    s = math.sqrt(2.0 * dimension / n_per_class) / snr
    single = math.asin(min(1.0, s)) if s < 1.0 else math.pi / 2.0
    pair = min(1.0, math.sin(single) * math.sqrt(2.0))
    return math.degrees(math.asin(pair))
