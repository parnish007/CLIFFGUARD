"""CPU-only readiness audit for the pivoted CLIFFGUARD measurement stack.

The default invocation runs theorem identities, independent null/power
calibrations, real saved-direction checks, the GGUF no-file self-check, the
unit suite, and strict mypy. It never downloads a model, imports torch, or
installs a package. Checks that require unavailable activations, model files,
or GPU dependencies are reported as SKIP with an explicit reason.

PASS means the implementation satisfies its audited contract. For rules that
were downgraded to diagnostics, PASS means the runner reproduced both the
useful behavior and the pinned limitation; it does not promote the diagnostic
back into a scientific verdict.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import norm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cliffguard.eval.composition import (  # noqa: E402
    Composition,
    chain_performance,
    collapse_bits_threshold_closed_form,
    d_prime_at_bits,
    eta_at_bits,
    predict_collapse,
    tail_rate,
    threshold_performance,
)
from cliffguard.eval.discriminability import gaussianity_gap  # noqa: E402
from cliffguard.eval.isotropy import isotropy_test  # noqa: E402
from cliffguard.eval.noise_spectrum import (  # noqa: E402
    EtaMeasurement,
    fit_eta_vs_bits_report,
)
from cliffguard.eval.threshold_calibrator import (  # noqa: E402
    calibrate_threshold,
    empirical_fpr,
)


FloatArray = npt.NDArray[np.float64]
Status = Literal["PASS", "FAIL", "SKIP"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str


class Audit:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def check(
        self, name: str, function: Callable[[], tuple[bool, str]]
    ) -> None:
        try:
            passed, detail = function()
        except Exception as exc:  # readiness runner must report, not abort early
            self.results.append(
                CheckResult(name, "FAIL", f"{type(exc).__name__}: {exc}")
            )
            return
        self.results.append(CheckResult(name, "PASS" if passed else "FAIL", detail))

    def skip(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name, "SKIP", detail))

    def print_table(self, elapsed_seconds: float) -> None:
        status_width = 6
        name_width = max(5, *(len(result.name) for result in self.results))
        print()
        print(
            f"{'STATUS':<{status_width}} | {'CHECK':<{name_width}} | DETAIL"
        )
        print(f"{'-' * status_width}-+-{'-' * name_width}-+-{'-' * 72}")
        for result in self.results:
            detail = " ".join(result.detail.splitlines())
            print(
                f"{result.status:<{status_width}} | "
                f"{result.name:<{name_width}} | {detail}"
            )
        counts = {
            status: sum(result.status == status for result in self.results)
            for status in ("PASS", "FAIL", "SKIP")
        }
        print()
        print(
            f"Summary: {counts['PASS']} PASS, {counts['FAIL']} FAIL, "
            f"{counts['SKIP']} SKIP in {elapsed_seconds:.1f}s"
        )

    @property
    def failed(self) -> bool:
        return any(result.status == "FAIL" for result in self.results)


def _unit(vector: FloatArray) -> FloatArray:
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        raise ValueError("zero vector")
    return np.asarray(vector / length, dtype=np.float64)


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """95% Wilson score interval without another optional dependency."""
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return center - half, center + half


# ---------------------------------------------------------------------------
# Theorem identities
# ---------------------------------------------------------------------------


def _check_kappa_identity() -> tuple[bool, str]:
    d0 = 2.5
    alpha = 0.05
    z_alpha = float(norm.ppf(1.0 - alpha))
    closed = d0 / 2.0 * float(norm.pdf(d0 - z_alpha) / norm.cdf(d0 - z_alpha))

    def log_performance(eta: float) -> float:
        return math.log(float(norm.cdf(d0 / math.sqrt(1.0 + eta) - z_alpha)))

    step = 1e-6
    numerical = -(
        log_performance(step) - log_performance(-step)
    ) / (2.0 * step)
    passed = abs(closed - numerical) < 1e-8 and abs(closed - 0.430424) < 1e-6
    return passed, f"closed={closed:.9f}, derivative={numerical:.9f}"


def _check_chain_increment_identity() -> tuple[bool, str]:
    bit_widths = {
        steps: predict_collapse(
            Composition.CHAIN,
            d_prime_0=2.5,
            eta_4=0.30,
            alpha=0.05,
            retention=0.5,
            n_steps=steps,
        ).bits
        for steps in (4, 8, 16, 32, 64)
    }
    increments = (
        bit_widths[16] - bit_widths[4],
        bit_widths[32] - bit_widths[8],
        bit_widths[64] - bit_widths[16],
    )
    passed = all(abs(value - 1.0) < 0.03 for value in increments)
    rendered = ", ".join(f"{value:.3f}" for value in increments)
    return passed, f"b*(4T)-b*(T)={rendered} bits (target 1)"


def _check_tail_identity() -> tuple[bool, str]:
    eta = 0.25
    p0_values = (1e-2, 1e-4, 1e-8, 1e-16)
    ratios = [
        math.log(tail_rate(p0, eta)) / math.log(p0) for p0 in p0_values
    ]
    target = 1.0 / (1.0 + eta)
    passed = all(
        later < earlier for earlier, later in zip(ratios, ratios[1:])
    ) and abs(ratios[-1] - target) < 0.02
    return passed, (
        f"ratios={[round(value, 3) for value in ratios]}, target={target:.3f}"
    )


def _check_closed_form_collapse() -> tuple[bool, str]:
    bits = collapse_bits_threshold_closed_form(2.5, 0.05, 0.30, 4.0)
    return abs(bits - 2.94) < 0.01, f"b*={bits:.6f} bits (documented 2.94)"


def _check_disagreement_bound() -> tuple[bool, str]:
    rng = np.random.default_rng(710_001)
    margins = rng.normal(size=500_000)
    unit_errors = rng.uniform(-1.0, 1.0, size=margins.size)
    threshold = 0.7
    details: list[str] = []
    passed = True
    for epsilon in (0.001, 0.01, 0.05, 0.2):
        corrected = margins + epsilon * unit_errors
        disagreements = float(
            np.mean((margins >= threshold) != (corrected >= threshold))
        )
        empirical_envelope = float(np.mean(np.abs(margins - threshold) <= epsilon))
        analytic_bound = float(
            norm.cdf(threshold + epsilon) - norm.cdf(threshold - epsilon)
        )
        passed = passed and disagreements <= empirical_envelope + 1e-15
        passed = passed and disagreements <= analytic_bound + 0.001
        details.append(f"eps={epsilon:g}: {disagreements:.5f}<={analytic_bound:.5f}")
    return passed, "; ".join(details)


# ---------------------------------------------------------------------------
# Independent Stage 0 calibration (does not call noise_floor.py)
# ---------------------------------------------------------------------------


RotationData = tuple[FloatArray, FloatArray, FloatArray, FloatArray]


def _rotation_data(
    shift: float,
    seed: int,
    *,
    n_per_class: int = 200,
    dimension: int = 128,
    quantized_noise: float = 0.5,
    separation: float = 3.0,
) -> RotationData:
    rng = np.random.default_rng(seed)
    class_shift = np.zeros(dimension, dtype=np.float64)
    class_shift[0] = separation
    harmful_fp16 = rng.normal(size=(n_per_class, dimension)) + class_shift
    harmless_fp16 = rng.normal(size=(n_per_class, dimension))
    harmful_q = harmful_fp16 + rng.normal(
        scale=quantized_noise, size=harmful_fp16.shape
    )
    harmless_q = harmless_fp16 + rng.normal(
        scale=quantized_noise, size=harmless_fp16.shape
    )
    harmful_q[:, 1] += shift
    return harmful_fp16, harmless_fp16, harmful_q, harmless_q


def _mean_direction(harmful: FloatArray, harmless: FloatArray) -> FloatArray:
    return _unit(np.mean(harmful, axis=0) - np.mean(harmless, axis=0))


def _parallel_transport(
    vector: FloatArray, source: FloatArray, destination: FloatArray
) -> FloatArray:
    """Transport a tangent vector along the shortest unit-sphere geodesic."""
    denominator = 1.0 + float(np.dot(source, destination))
    if denominator <= 1e-12:
        raise ValueError("antipodal references have no unique short transport")
    return np.asarray(
        vector
        - (float(np.dot(vector, destination)) / denominator)
        * (source + destination),
        dtype=np.float64,
    )


def _independent_replication_z(
    data: RotationData,
    *,
    split_seed: int,
    n_splits: int = 40,
    variant: Literal["tangent", "raw", "transport"] = "tangent",
) -> float:
    """Independent implementation of the complementary-half statistic."""
    harmful_fp16, harmless_fp16, harmful_q, harmless_q = data
    n_harmful, dimension = harmful_fp16.shape
    n_harmless = harmless_fp16.shape[0]
    half_harmful = n_harmful // 2
    half_harmless = n_harmless // 2
    rng = np.random.default_rng(split_seed)
    cosines: list[float] = []

    for _ in range(n_splits):
        harmful_order = rng.permutation(n_harmful)
        harmless_order = rng.permutation(n_harmless)
        halves = (
            (
                harmful_order[:half_harmful],
                harmless_order[:half_harmless],
            ),
            (
                harmful_order[half_harmful : 2 * half_harmful],
                harmless_order[half_harmless : 2 * half_harmless],
            ),
        )
        shifts: list[FloatArray] = []
        references: list[FloatArray] = []
        for harmful_indices, harmless_indices in halves:
            reference = _mean_direction(
                harmful_fp16[harmful_indices], harmless_fp16[harmless_indices]
            )
            quantized = _mean_direction(
                harmful_q[harmful_indices], harmless_q[harmless_indices]
            )
            difference = quantized - reference
            if variant == "raw":
                shift_vector = difference
            else:
                shift_vector = difference - float(np.dot(difference, reference)) * reference
            shifts.append(np.asarray(shift_vector, dtype=np.float64))
            references.append(reference)

        first, second = shifts
        if variant == "transport":
            first = _parallel_transport(first, references[0], references[1])
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        if first_norm > 0.0 and second_norm > 0.0:
            cosines.append(float(np.dot(first, second) / (first_norm * second_norm)))
    if not cosines:
        raise ValueError("all independent replication splits were degenerate")

    # Match the production threshold convention. A tangent vector has D-1
    # degrees of freedom, so sqrt(D-1) is the exact single-cosine scale; at
    # D=128 the production sqrt(D) convention differs by only 0.39%.
    return float(np.median(cosines) * math.sqrt(dimension))


def _stage0_null_and_power() -> tuple[bool, str]:
    null_trials = 300
    alternative_trials = 120
    null_z = [
        _independent_replication_z(
            _rotation_data(0.0, 810_000 + trial),
            split_seed=910_000 + trial,
        )
        for trial in range(null_trials)
    ]
    null_fires = sum(value > 3.0 for value in null_z)
    powers: dict[float, float] = {}
    median_z: dict[float, float] = {}
    for shift in (0.10, 0.25, 0.50, 0.75):
        values = [
            _independent_replication_z(
                _rotation_data(shift, 820_000 + 10_000 * int(shift * 100) + trial),
                split_seed=920_000 + trial,
            )
            for trial in range(alternative_trials)
        ]
        powers[shift] = sum(value > 3.0 for value in values) / alternative_trials
        median_z[shift] = float(np.median(values))
    power_values = [powers[shift] for shift in sorted(powers)]
    passed = (
        null_fires / null_trials <= 0.05
        and power_values == sorted(power_values)
        and 0.40 <= powers[0.50] <= 0.85
        and powers[0.75] >= 0.90
    )
    _, null_upper = _wilson_interval(null_fires, null_trials)
    curve = ", ".join(
        f"{shift:.2f}:{powers[shift]:.1%} (median z={median_z[shift]:.2f})"
        for shift in sorted(powers)
    )
    return passed, (
        f"Type I={null_fires}/{null_trials} (95% Wilson upper {null_upper:.1%}); "
        f"power shift->rate {curve}"
    )


def _stage0_sample_size_power() -> tuple[bool, str]:
    trials = 160
    powers: dict[int, float] = {}
    for n_per_class in (200, 225, 250):
        fired = 0
        for trial in range(trials):
            value = _independent_replication_z(
                _rotation_data(
                    0.50,
                    930_000 + 10_000 * n_per_class + trial,
                    n_per_class=n_per_class,
                ),
                split_seed=940_000 + trial,
            )
            fired += int(value > 3.0)
        powers[n_per_class] = fired / trials
    population_angle = math.degrees(math.atan2(0.50, 3.0))
    passed = powers[200] < 0.80 and powers[250] >= 0.80
    return passed, (
        f"shift 0.50 = population rotation {population_angle:.2f}deg; "
        + ", ".join(f"n={n}:{power:.1%}" for n, power in powers.items())
        + "; planning recommendation n=250/class"
    )


def _stage0_shift_definition() -> tuple[bool, str]:
    trials = 120
    variants = ("tangent", "raw", "transport")
    nominal_power: dict[str, float] = {}
    stress_type_i: dict[str, float] = {}
    for variant in variants:
        nominal_power[variant] = sum(
            _independent_replication_z(
                _rotation_data(0.50, 950_000 + trial),
                split_seed=960_000 + trial,
                variant=variant,  # type: ignore[arg-type]
            )
            > 3.0
            for trial in range(trials)
        ) / trials
        stress_type_i[variant] = sum(
            _independent_replication_z(
                _rotation_data(
                    0.0,
                    970_000 + trial,
                    quantized_noise=4.0,
                ),
                split_seed=980_000 + trial,
                variant=variant,  # type: ignore[arg-type]
            )
            > 3.0
            for trial in range(trials)
        ) / trials
    passed = (
        stress_type_i["tangent"] <= 0.05
        and stress_type_i["transport"] <= 0.05
        and stress_type_i["raw"] > 0.05
        and abs(nominal_power["tangent"] - nominal_power["transport"]) <= 0.05
    )
    power_text = ", ".join(
        f"{key}={value:.1%}" for key, value in nominal_power.items()
    )
    type_i_text = ", ".join(
        f"{key}={value:.1%}" for key, value in stress_type_i.items()
    )
    return passed, (
        f"nominal shift-0.5 power: {power_text}; high-noise null Type I: "
        f"{type_i_text}. Tangent projection removes radial normalization bias; "
        "parallel transport is numerically equivalent here."
    )


# ---------------------------------------------------------------------------
# Part 2 decision-rule calibrations
# ---------------------------------------------------------------------------


def _isotropy_calibration() -> tuple[bool, str]:
    trials = 60
    dimension = 3072
    n_null = 160

    def rejection_rate(scale_multiplier: float, seed_offset: int) -> float:
        rejected = 0
        for trial in range(trials):
            rng = np.random.default_rng(seed_offset + trial)
            reference = _unit(rng.normal(size=dimension))
            scales = np.ones(dimension, dtype=np.float64)
            scales[: dimension // 20] = scale_multiplier
            perturbation = _unit(rng.normal(size=dimension) * scales) * 0.236
            quantized = _unit(reference + perturbation)
            result = isotropy_test(
                reference,
                quantized,
                n_null=n_null,
                seed=seed_offset + 100_000 + trial,
            )
            rejected += int(result.rejects_concentration_null())
        return rejected / trials

    type_i = rejection_rate(1.0, 1_010_000)
    sparse_1_5 = rejection_rate(1.5, 1_020_000)
    sparse_2 = rejection_rate(2.0, 1_030_000)
    # A dense Gaussian-looking target has the same coordinate-shape law as the
    # null from this statistic's perspective; its rejection rate estimates the
    # unavoidable Type II limitation rather than a useful alternative power.
    dense_target = rejection_rate(1.0, 1_040_000)
    passed = (
        type_i <= 0.05
        and sparse_2 >= 0.80
        and dense_target <= 0.10
        and sparse_1_5 <= sparse_2
    )
    return passed, (
        f"isotropic-null rejection={type_i:.1%}; power when 5% of coordinates "
        f"have 1.5x/2x SD={sparse_1_5:.1%}/{sparse_2:.1%}; dense-target "
        f"rejection={dense_target:.1%}. z<3 is a fail-to-reject diagnostic."
    )


def _gaussianity_gap_calibration() -> tuple[bool, str]:
    trials = 800
    n_per_class = 200
    rates: dict[str, float] = {}

    for kind in ("null", "unequal", "student", "contaminated"):
        flags = 0
        for trial in range(trials):
            rng = np.random.default_rng(1_100_000 + 10_000 * len(kind) + trial)
            negative = rng.normal(0.0, 1.0, n_per_class)
            if kind == "null":
                positive = rng.normal(1.2, 1.0, n_per_class)
            elif kind == "unequal":
                positive = rng.normal(1.2, 2.0, n_per_class)
            elif kind == "student":
                positive = 1.2 + rng.standard_t(3, n_per_class) / math.sqrt(3.0)
            else:
                positive = rng.normal(1.2, 1.0, n_per_class)
                contaminated = rng.random(n_per_class) < 0.05
                positive[contaminated] = rng.normal(
                    0.0, 10.0, int(np.count_nonzero(contaminated))
                )
            flags += int(gaussianity_gap(positive, negative) > 0.05)
        rates[kind] = flags / trials

    passed = (
        rates["null"] <= 0.02
        and rates["contaminated"] >= 0.80
        and rates["unequal"] <= 0.10
        and rates["student"] < 0.30
    )
    return passed, (
        f"P(gap>0.05): Gaussian null={rates['null']:.1%}, 5% 10-sigma "
        f"contamination={rates['contaminated']:.1%}, t3={rates['student']:.1%}, "
        f"unequal-variance Gaussian={rates['unequal']:.1%}. Threshold is an "
        "effect-size diagnostic, not an A2 verdict."
    )


def _eta_observations(bits: FloatArray, log_eta: FloatArray) -> dict[str, EtaMeasurement]:
    return {
        f"scheme-{index}": EtaMeasurement(
            bits_per_param_wholefile=float(bit_width + 0.5),
            bits_per_param_payload=float(bit_width),
            eta=float(math.exp(value)),
        )
        for index, (bit_width, value) in enumerate(zip(bits, log_eta))
    }


def _exponent_ci_calibration() -> tuple[bool, str]:
    trials = 1_500
    bits = np.asarray([8.0, 6.0, 5.0, 4.0, 3.0, 2.0], dtype=np.float64)
    x = 4.0 - bits
    rng = np.random.default_rng(1_200_000)

    def rejection_rate(exponent: float, errors: FloatArray) -> float:
        rejected = 0
        for row in errors:
            log_eta = math.log(0.30) + x * math.log(exponent) + row
            report = fit_eta_vs_bits_report(_eta_observations(bits, log_eta))
            rejected += int(
                bool(report.model_conditional_exponent_ci_excludes_four)
            )
        return rejected / len(errors)

    independent_errors = rng.normal(scale=0.15, size=(trials, bits.size))
    indices = np.arange(bits.size)
    covariance = 0.15**2 * 0.6 ** np.abs(indices[:, None] - indices[None, :])
    correlated_errors = rng.multivariate_normal(
        np.zeros(bits.size), covariance, size=trials
    )
    type_i_independent = rejection_rate(4.0, independent_errors)
    type_i_correlated = rejection_rate(4.0, correlated_errors)
    power_3_5 = rejection_rate(3.5, independent_errors)
    passed = (
        0.025 <= type_i_independent <= 0.08
        and type_i_correlated >= 0.10
        and power_3_5 >= 0.75
    )
    return passed, (
        f"beta=4 Type I: iid log-error={type_i_independent:.1%}, AR(1) "
        f"rho=0.6={type_i_correlated:.1%}; beta=3.5 power={power_3_5:.1%}. "
        "OLS CI is model-conditional and not a ladder verdict without "
        "covariance-aware repeated estimates; the 25% band is heuristic only."
    )


def _composition_calibration() -> tuple[bool, str]:
    null_predictions = [
        predict_collapse(
            composition,
            d_prime_0=d_prime,
            eta_4=0.0,
            retention=retention,
            n_steps=16,
            p0=1e-3,
        )
        for composition in Composition
        for d_prime in (2.0, 2.5, 3.2, 4.0)
        for retention in (0.25, 0.5, 0.75)
    ]
    false_crossings = sum(
        math.isfinite(prediction.bits) for prediction in null_predictions
    )

    alternatives = [
        (Composition.THRESHOLD, dict(d_prime_0=2.5, eta_4=0.30)),
        (Composition.CHAIN, dict(d_prime_0=3.2, eta_4=0.30, n_steps=4)),
        (Composition.CHAIN, dict(d_prime_0=3.2, eta_4=0.30, n_steps=16)),
        (Composition.CHAIN, dict(d_prime_0=3.2, eta_4=0.30, n_steps=64)),
        (Composition.TAIL, dict(d_prime_0=2.5, eta_4=0.30, p0=1e-3)),
        (Composition.TAIL, dict(d_prime_0=2.5, eta_4=0.30, p0=1e-5)),
    ]
    resolved = 0
    residuals: list[float] = []
    for composition, keyword_arguments in alternatives:
        prediction = predict_collapse(composition, retention=0.5, **keyword_arguments)
        if not math.isfinite(prediction.bits):
            continue
        resolved += 1
        d_prime_0 = float(keyword_arguments["d_prime_0"])
        eta_4 = float(keyword_arguments["eta_4"])
        if composition is Composition.THRESHOLD:
            reference = threshold_performance(d_prime_at_bits(16.0, d_prime_0, eta_4))
            actual = threshold_performance(
                d_prime_at_bits(prediction.bits, d_prime_0, eta_4)
            )
            target = 0.5 * reference
        elif composition is Composition.CHAIN:
            n_steps = int(keyword_arguments["n_steps"])
            reference = chain_performance(
                d_prime_at_bits(16.0, d_prime_0, eta_4), n_steps
            )
            actual = chain_performance(
                d_prime_at_bits(prediction.bits, d_prime_0, eta_4), n_steps
            )
            target = 0.5 * reference
        else:
            p0 = float(keyword_arguments["p0"])
            actual = tail_rate(p0, eta_at_bits(prediction.bits, eta_4))
            target = p0 / 0.5
        residuals.append(abs(actual - target))
    passed = (
        false_crossings == 0
        and resolved == len(alternatives)
        and max(residuals, default=math.inf) < 1e-9
    )
    return passed, (
        f"flat/no-crossing Type I={false_crossings}/{len(null_predictions)}; "
        f"resolved in-range alternatives={resolved}/{len(alternatives)}, "
        f"max target residual={max(residuals, default=math.inf):.2e}"
    )


def _threshold_calibration() -> tuple[bool, str]:
    rng = np.random.default_rng(1_300_000)
    target = 0.05
    same_sample_ok = True
    for fires_high in (False, True):
        for scores in (
            rng.normal(size=5_000),
            rng.standard_t(2, size=5_000),
            np.round(rng.normal(size=5_000), 1),
        ):
            threshold = calibrate_threshold(scores, target, fires_high=fires_high)
            same_sample_ok = same_sample_ok and (
                empirical_fpr(scores, threshold, fires_high=fires_high)
                <= target + 1e-12
            )

    trials = 1_000
    held_out_fprs: list[float] = []
    harmful_tprs: list[float] = []
    for _ in range(trials):
        calibration = rng.normal(size=400)
        held_out = rng.normal(size=4_000)
        harmful = rng.normal(-2.5, 1.0, size=4_000)
        threshold = calibrate_threshold(calibration, target, fires_high=False)
        held_out_fprs.append(empirical_fpr(held_out, threshold, fires_high=False))
        harmful_tprs.append(empirical_fpr(harmful, threshold, fires_high=False))
    fprs = np.asarray(held_out_fprs)
    mean_fpr = float(np.mean(fprs))
    exceedance = float(np.mean(fprs > target))
    interval = np.quantile(fprs, [0.025, 0.975])
    mean_power = float(np.mean(harmful_tprs))
    passed = (
        same_sample_ok
        and 0.045 <= mean_fpr <= 0.055
        and 0.30 <= exceedance <= 0.60
        and 0.75 <= mean_power <= 0.85
    )
    return passed, (
        f"same-sample conservative={same_sample_ok}; n_cal=400 held-out mean "
        f"FPR={mean_fpr:.3f}, 95% range [{interval[0]:.3f}, {interval[1]:.3f}], "
        f"P(FPR>0.05)={exceedance:.1%}; d'=2.5 TPR={mean_power:.1%}. "
        "Default is a point calibrator, not a population-FPR guarantee."
    )


# ---------------------------------------------------------------------------
# Real artifacts, CLI harness, and project-wide static checks
# ---------------------------------------------------------------------------


def _real_direction_artifacts() -> tuple[bool, str]:
    folder = REPO_ROOT / "notebooks" / "fold_a"
    fp16_path = folder / "r_hat_Llama-3.2-3B-Instruct_FP16.npz"
    nf4_path = folder / "r_hat_Llama-3.2-3B-Instruct_NF4.npz"
    with np.load(fp16_path, allow_pickle=False) as archive:
        fp16 = np.asarray(archive["direction"], dtype=np.float64)
    with np.load(nf4_path, allow_pickle=False) as archive:
        nf4 = np.asarray(archive["direction"], dtype=np.float64)
    result = isotropy_test(fp16, nf4, n_null=400, seed=0)
    passed = (
        abs(result.angle_deg - 13.5624) < 0.01
        and abs(result.excess_kurtosis_delta - 0.1990) < 0.01
        and abs(result.excess_kurtosis_reference - 19.3438) < 0.02
        and result.irrecoverable_fraction >= 0.985
    )
    return passed, (
        f"angle={result.angle_deg:.2f}deg; kurtosis delta/reference="
        f"{result.excess_kurtosis_delta:.2f}/{result.excess_kurtosis_reference:.2f}; "
        f"irrecoverable={100 * result.irrecoverable_fraction:.2f}%; "
        f"max|z|={result.max_abs_z:.2f} (concentration null not rejected)"
    )


def _notebook_verdict_contract() -> tuple[bool, str]:
    notebook_path = REPO_ROOT / "notebooks" / "stage0_noise_floor_and_isotropy.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )
    required = (
        "rotation_replication(",
        '"passes_gate": bool(replication.passes())',
        '"diagnostics_not_decisions"',
        '"exceeds_floor": bool(exceeds_floor_diagnostic)',
        '"paired_excludes_zero": bool(paired.excludes_zero)',
    )
    missing = [fragment for fragment in required if fragment not in source]
    forbidden = '"passes_gate": bool(paired.excludes_zero)'
    passed = not missing and forbidden not in source
    return passed, (
        "rotation_replication is the sole persisted gate; exceeds_floor and "
        "paired intervals are nested under diagnostics_not_decisions"
        if passed
        else f"missing={missing}, obsolete_gate_present={forbidden in source}"
    )


def _run_subprocess(command: list[str], timeout: int) -> tuple[bool, str]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    combined = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
    meaningful = [line.strip() for line in combined if line.strip()]
    detail = meaningful[-1] if meaningful else "no output"
    return completed.returncode == 0, detail


def _gguf_self_check() -> tuple[bool, str]:
    return _run_subprocess(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_gguf_pair.py"), "--self-check"],
        timeout=30,
    )


def _pytest_check() -> tuple[bool, str]:
    return _run_subprocess(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        timeout=180,
    )


def _mypy_check() -> tuple[bool, str]:
    return _run_subprocess(
        [sys.executable, "-m", "mypy", "--strict", "cliffguard"],
        timeout=120,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-suite",
        action="store_true",
        help="skip the full pytest and strict-mypy subprocesses",
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    audit = Audit()

    audit.check("Theorem 1 kappa_1 identity", _check_kappa_identity)
    audit.check("Corollary 1.2 chain increment", _check_chain_increment_identity)
    audit.check("Corollary 1.3 tail limit", _check_tail_identity)
    audit.check("Theorem 2 closed-form b*", _check_closed_form_collapse)
    audit.check("Theorem 7 disagreement bound", _check_disagreement_bound)

    audit.check("Stage 0 independent null/power", _stage0_null_and_power)
    audit.check("Stage 0 MDE sample-size power", _stage0_sample_size_power)
    audit.check("Stage 0 tangent/raw/transport", _stage0_shift_definition)
    audit.check("Isotropy concentration rule", _isotropy_calibration)
    audit.check("gaussianity_gap diagnostic", _gaussianity_gap_calibration)
    audit.check("Noise exponent OLS interval", _exponent_ci_calibration)
    audit.check("Collapse bisection", _composition_calibration)
    audit.check("Threshold held-out calibration", _threshold_calibration)

    audit.check("Fold A saved-direction pipeline", _real_direction_artifacts)
    audit.check("Notebook verdict contract", _notebook_verdict_contract)
    audit.skip(
        "Fold A d-prime",
        "r_hat NPZ files contain only a direction vector; raw per-prompt margins are absent",
    )
    audit.skip(
        "Fold A noise floor/replication",
        "r_hat NPZ files contain no row-aligned per-prompt FP16/NF4 activations",
    )
    audit.check("GGUF harness no-file self-check", _gguf_self_check)

    torch_present = importlib.util.find_spec("torch") is not None
    transformers_present = importlib.util.find_spec("transformers") is not None
    audit.skip(
        "Transformers/NF4 model checks",
        f"torch installed={torch_present}, transformers installed={transformers_present}; "
        "no model download/cache is authorized and CUDA activations are absent",
    )
    audit.skip(
        "Real F16/Q_K GGUF pair",
        "no multi-GB matched GGUF pair and optional gguf package is unavailable",
    )
    audit.skip(
        "Colab-class inference experiments",
        "requires 8-12 GB runtime VRAM for a 3B FP16 reference; local GPU has 6 GB",
    )
    audit.skip(
        "Legacy Delta_cliff decisions",
        "KAPPA=0.25 and cliff-boundary booleans are uncalibrated and retired by the pivot plan",
    )
    audit.skip(
        "Legacy CONDUCTOR majority rule",
        "the >0.5 weighted vote is an archived operational policy, not a statistical claim",
    )

    if arguments.skip_suite:
        audit.skip("pytest -q", "disabled by --skip-suite")
        audit.skip("mypy --strict cliffguard", "disabled by --skip-suite")
    else:
        audit.check("pytest -q", _pytest_check)
        audit.check("mypy --strict cliffguard", _mypy_check)

    elapsed = time.perf_counter() - started
    audit.print_table(elapsed)
    return 1 if audit.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
