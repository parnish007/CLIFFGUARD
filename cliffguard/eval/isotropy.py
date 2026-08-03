"""Isotropy test for quantization perturbations — the Stage 1 experiment.

Premise P1 of the discriminability-decay theorem is that quantization acts as
approximately ISOTROPIC noise on the residual stream: the perturbation is
zero-mean and uncorrelated with the behavioural direction, so it degrades
discriminability by inflating variance rather than by rotating or selectively
destroying the safety-critical channels.

The refined claim (docs/theory_panel_2026-08.md §4) is that isotropy is a
property of the QUANTIZER, not of quantization:

  * Round-to-nearest / NF4 / vanilla GGUF k-quants have no reason to correlate
    their error with any behavioural direction  -> isotropic -> cliff.
  * Activation-aware schemes (AWQ, GPTQ) explicitly protect high-salience
    channels                                    -> anisotropic -> no cliff.

If that holds, it mechanistically resolves the contradiction between
arXiv:2606.10154 (refusal falls 12-68 pp under quantization) and
arXiv:2606.29581 (AWQ INT4 stays within ~1.6 pp of FP16 for 7 of 8 models).

Method. Given the FP16 and quantized estimates of a behavioural direction,
compare the observed perturbation against a matched-magnitude ISOTROPIC null:
draw random perturbations of the same norm, apply them to the FP16 direction,
renormalise, and compute the same statistics. Report z-scores.

SCOPE OF THE TEST. Failure to reject this null is not evidence that A1 holds.
The input contains one direction difference, so the test can detect coordinate
concentration but cannot establish zero mean, equal projected variance across
behavioural directions, or distinguish a fixed dense target from one draw of
isotropic noise. The public status is therefore "concentration null not
rejected", never a positive "isotropic" verdict.

CAVEAT (defect D2). Finite-sample estimation noise is ITSELF isotropic. A
perturbation that looks isotropic is therefore consistent with BOTH "the
quantizer adds isotropic noise" AND "we measured nothing but sampling error".
Always run eval/noise_floor.split_half_noise_floor first and report the
result alongside; this module refuses to interpret its own output without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]

DEFAULT_N_NULL: int = 400
DEFAULT_Z_THRESHOLD: float = 3.0
CONCENTRATION_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10)


def _unit(v: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero-norm vector")
    return v / norm


def concentration(delta: FloatArray, fraction: float) -> float:
    """Fraction of squared perturbation energy carried by the top `fraction`
    of coordinates by magnitude.

    A targeted perturbation (a few critical channels) concentrates; an
    isotropic one does not. Returns a value in [0, 1]."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    sq = delta**2
    total = float(sq.sum())
    if total == 0.0:
        raise ValueError("perturbation is identically zero")
    k = max(1, int(round(delta.size * fraction)))
    top = np.sort(sq)[::-1][:k]
    return float(top.sum() / total)


def parallel_orthogonal_split(
    delta: FloatArray, reference: FloatArray
) -> tuple[float, float]:
    """Decompose a perturbation into components parallel and orthogonal to a
    reference direction.

    The parallel component is a systematic BIAS along the behavioural
    direction and is correctable by affine recalibration. The orthogonal
    component is input-dependent and is NOT correctable post hoc (the
    data-processing inequality bites there). Returns (parallel, orthogonal).

    On the repo's own FP16 -> NF4 artifacts this gives (-0.0279, 0.2345):
    ~99 % of the damage sits in the irrecoverable component."""
    ref = _unit(reference)
    par = float(np.dot(delta, ref))
    total_sq = float(np.dot(delta, delta))
    orth = math.sqrt(max(total_sq - par**2, 0.0))
    return par, orth


@dataclass(frozen=True)
class IsotropyResult:
    """Observed perturbation statistics against a matched isotropic null."""

    angle_deg: float
    cosine: float
    excess_kurtosis_delta: float
    excess_kurtosis_reference: float
    parallel: float
    orthogonal: float
    concentration_observed: dict[float, float]
    concentration_null_mean: dict[float, float]
    concentration_null_std: dict[float, float]
    n_null: int
    dimension: int

    def concentration_z(self, fraction: float) -> float:
        """z-score of observed concentration against the isotropic null.

        |z| small means only that this concentration statistic does not reject
        the matched null. A large positive value means damage is concentrated
        on a few channels; a large negative value means it is unusually
        diffuse."""
        sd = self.concentration_null_std[fraction]
        if sd == 0.0:
            raise ValueError("null standard deviation is zero")
        return (self.concentration_observed[fraction] - self.concentration_null_mean[fraction]) / sd

    @property
    def max_abs_z(self) -> float:
        return max(abs(self.concentration_z(f)) for f in self.concentration_observed)

    @property
    def irrecoverable_fraction(self) -> float:
        """Orthogonal share of the chord between two unit directions.

        CARRIES NO INFORMATION BEYOND THE ANGLE. For unit vectors at angle
        theta this is identically ``cos(theta/2)**2``:

            parallel   = cos(theta) - 1
            orthogonal = sin(theta)
            orthogonal**2 / ||delta||**2 = (1 + cos theta) / 2 = cos(theta/2)**2

        Verified against measurement: at 88.69 deg it returns 0.5114, and
        ``cos(44.345 deg)**2`` is 0.5114. Any apparent trend in this quantity
        across a ladder is a restatement of the angle trend. A results write-up
        once reported "irrecoverable damage falls from 1.000 to 0.511 as bits
        drop" as a counterintuitive finding; it is trigonometry.

        The name is retained for compatibility but the recoverability reading
        is NOT justified. This decomposes the difference between two aggregate
        unit directions, not per-example score noise, so nothing here shows the
        parallel part is removable by affine recalibration or that the
        orthogonal part is not. Use it as chord geometry, and prefer
        ``angle_deg`` which says the same thing without the implication.
        """
        denom = self.parallel**2 + self.orthogonal**2
        if denom == 0.0:
            raise ValueError("perturbation is identically zero")
        return self.orthogonal**2 / denom

    def rejects_concentration_null(
        self, z_threshold: float = DEFAULT_Z_THRESHOLD
    ) -> bool:
        """Whether the concentration statistics reject the matched null.

        The default threshold is an empirically conservative family-wise rule
        over the three correlated concentration fractions. It is not a test of
        all of A1; in particular, a dense targeted direction is unidentifiable
        from a single isotropic-looking draw.
        """
        if not math.isfinite(z_threshold) or z_threshold <= 0.0:
            raise ValueError(
                f"z_threshold must be finite and positive, got {z_threshold}"
            )
        return self.max_abs_z >= z_threshold

    def concentration_null_not_rejected(
        self, z_threshold: float = DEFAULT_Z_THRESHOLD
    ) -> bool:
        """Negation of :meth:`rejects_concentration_null`.

        This is a fail-to-reject result, not a positive isotropy decision.
        """
        return not self.rejects_concentration_null(z_threshold)

    def is_isotropic(self, z_threshold: float = DEFAULT_Z_THRESHOLD) -> bool:
        """Backward-compatible alias for a fail-to-reject diagnostic.

        Despite the historical name, ``True`` does *not* establish isotropy.
        New decision logic must use ``rejects_concentration_null`` and report
        the z-scores themselves.
        """
        return self.concentration_null_not_rejected(z_threshold)

    def summary(self) -> str:
        lines = [
            f"rotation {self.angle_deg:.2f} deg (cos {self.cosine:.4f}), D={self.dimension}",
            f"excess kurtosis: perturbation {self.excess_kurtosis_delta:.2f}, "
            f"reference direction {self.excess_kurtosis_reference:.2f}",
            f"parallel {self.parallel:+.5f} / orthogonal {self.orthogonal:.5f} "
            f"({100 * self.irrecoverable_fraction:.0f}% irrecoverable)",
        ]
        for f in sorted(self.concentration_observed):
            lines.append(
                f"top {100 * f:4.0f}%: observed {100 * self.concentration_observed[f]:5.1f}% "
                f"vs null {100 * self.concentration_null_mean[f]:5.1f}% "
                f"+/- {100 * self.concentration_null_std[f]:.1f}  "
                f"z={self.concentration_z(f):+.1f}"
            )
        if self.rejects_concentration_null():
            status = "REJECTED (coordinate concentration is anisotropic)"
        else:
            status = "NOT REJECTED (not positive evidence of isotropy)"
        lines.append(
            f"concentration null: {status} (max |z| = {self.max_abs_z:.1f})"
        )
        return "\n".join(lines)


def isotropy_test(
    direction_fp16: FloatArray,
    direction_quantized: FloatArray,
    n_null: int = DEFAULT_N_NULL,
    seed: int = 0,
    fractions: tuple[float, ...] = CONCENTRATION_FRACTIONS,
) -> IsotropyResult:
    """Compare an observed direction perturbation against a matched-magnitude
    isotropic null.

    The null preserves the perturbation's NORM and the unit-norm constraint on
    the resulting direction, so the only thing being tested is the perturbation's
    SHAPE (concentration and tail behaviour), not its size.

    Raises ValueError on shape mismatch or zero-norm inputs."""
    if direction_fp16.shape != direction_quantized.shape:
        raise ValueError(
            f"shape mismatch: {direction_fp16.shape} vs {direction_quantized.shape}"
        )
    if n_null < 2:
        raise ValueError(f"n_null must be >= 2, got {n_null}")

    a = _unit(direction_fp16)
    b = _unit(direction_quantized)
    delta = b - a
    magnitude = float(np.linalg.norm(delta))
    if magnitude == 0.0:
        raise ValueError("directions are identical — nothing to test")

    cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
    par, orth = parallel_orthogonal_split(delta, a)
    observed = {f: concentration(delta, f) for f in fractions}

    rng = np.random.default_rng(seed)
    null_draws: dict[float, list[float]] = {f: [] for f in fractions}
    dim = a.size
    for _ in range(n_null):
        noise = rng.normal(size=dim)
        noise = _unit(noise) * magnitude
        b_null = _unit(a + noise)
        d_null = b_null - a
        for f in fractions:
            null_draws[f].append(concentration(d_null, f))

    return IsotropyResult(
        angle_deg=math.degrees(math.acos(cos)),
        cosine=cos,
        excess_kurtosis_delta=float(stats.kurtosis(delta)),
        excess_kurtosis_reference=float(stats.kurtosis(a)),
        parallel=par,
        orthogonal=orth,
        concentration_observed=observed,
        concentration_null_mean={f: float(np.mean(v)) for f, v in null_draws.items()},
        concentration_null_std={f: float(np.std(v)) for f, v in null_draws.items()},
        n_null=n_null,
        dimension=int(dim),
    )


def random_direction_distance(dimension: int, n: int = 2000, seed: int = 0) -> float:
    """Mean chord distance between independent random unit vectors in R^d.

    The saturation reference: two unrelated directions in high dimension sit at
    chord distance ~sqrt(2) (angle ~90 deg). Any observed cross-scheme distance
    must be read against this ceiling — at D=3072 a chord of 0.236 (13.6 deg)
    means the directions are STRONGLY aligned, not badly damaged."""
    if dimension < 2:
        raise ValueError(f"dimension must be >= 2, got {dimension}")
    rng = np.random.default_rng(seed)
    dists: list[float] = []
    for _ in range(n):
        u = _unit(rng.normal(size=dimension))
        v = _unit(rng.normal(size=dimension))
        dists.append(float(np.linalg.norm(u - v)))
    return float(np.mean(dists))
