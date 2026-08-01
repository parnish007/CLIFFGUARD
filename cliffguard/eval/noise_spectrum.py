"""Weight-space quantization-noise spectrum for the pivot-plan scaling law.

The assumption-light primary output is the per-matrix spectrum of dequantized
weight perturbations projected onto a behavioural direction.  The additive
summary is deliberately named ``eta_proxy``: summing layer contributions
assumes that they are independent and that layer inputs are isotropic,
unit-variance activations.  Both assumptions are known to be false in real
networks, so the proxy is not a behavioural variance estimate.

``eta_empirical`` supplies the separate number needed for behavioural-margin
variance by applying one matrix perturbation to real FP16 benign activations.
This uses no harmful prompts, judge, or behavioural labels.  The resulting
claim is therefore "no behavioural labels", not "weights only".

The pure NumPy core is backend-independent.  Thin optional adapters handle
loaded transformers/bitsandbytes pairs and GGUF files associated with
``LlamaCppAdapter`` instances.  Missing models or optional GPU/GGUF packages
produce an unavailable report rather than an import-time failure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from typing import Any, cast
import warnings

import numpy as np
import numpy.typing as npt
from scipy.stats import t as student_t


FloatArray = npt.NDArray[np.float64]

EXPECTED_EXPONENT: float = 4.0
EXPONENT_RELATIVE_TOLERANCE: float = 0.25
POOR_FIT_R_SQUARED: float = 0.90


class ExponentDeviationWarning(UserWarning):
    """Warning for a model-conditional OLS interval that excludes four."""


class PoorEtaFitWarning(UserWarning):
    """Warning emitted when the log-linear bit law describes the data poorly."""


@dataclass(frozen=True)
class EtaMeasurement:
    """One scheme's measured eta and two measured storage rates.

    ``bits_per_param_payload`` is the tensor payload alone and is the regressor
    used by :func:`fit_eta_vs_bits`; constant container overhead would otherwise
    bias the fitted exponent on small artifacts. ``bits_per_param_wholefile``
    includes headers, metadata, and tokenizer data and is the deployed figure.
    """

    bits_per_param_wholefile: float
    bits_per_param_payload: float
    eta: float


@dataclass(frozen=True)
class EtaFitReport:
    """Diagnostics for ``eta = eta_4 * exponent ** (4 - bits)``.

    ``r_squared`` and ``rmse_log`` are computed in log-eta space, which is the
    space in which the exponential model is linear. The exponent confidence
    interval is the ordinary two-sided Student-t interval for the log-space OLS
    slope, transformed back to the exponent scale. It is unavailable with only
    two positive observations because the residual degrees of freedom are zero.

    The ordinary OLS interval is correctly calibrated only when log-eta errors
    are independent, Gaussian, and homoskedastic. Measurements from a matched
    ladder reuse the same weights, directions, and activation sample, so that
    independence is not established. The explicitly named
    ``model_conditional_exponent_ci_excludes_four`` field is therefore a
    diagnostic, not a pre-registered verdict. ``exponent_ci_excludes_four`` is
    retained as a compatibility alias. A defensible project-level decision
    requires repeated estimates and a covariance-aware or synchronized block
    bootstrap interval.

    ``exceeds_25_percent_development_band`` retains the arbitrary 25 percent
    development heuristic as a plainly named convenience field, not an
    inferential decision. ``practically_deviates_from_four`` is a compatibility
    alias.
    """

    eta_4: float
    exponent: float
    r_squared: float
    rmse_log: float
    n_points: int
    exponent_ci_low: float | None
    exponent_ci_high: float | None
    confidence_level: float
    model_conditional_exponent_ci_excludes_four: bool | None
    exceeds_25_percent_development_band: bool
    skipped_zero_schemes: tuple[str, ...]

    @property
    def exponent_ci_excludes_four(self) -> bool | None:
        """Compatibility alias for the model-conditional OLS diagnostic."""
        return self.model_conditional_exponent_ci_excludes_four

    @property
    def practically_deviates_from_four(self) -> bool:
        """Compatibility alias for the arbitrary 25 percent development band."""
        return self.exceeds_25_percent_development_band


@dataclass(frozen=True)
class WeightNoiseMeasurement:
    """Projected perturbation variance and eta for one named weight matrix."""

    weight_name: str
    sigma_squared: float
    eta: float


@dataclass(frozen=True)
class NoiseSpectrumReport:
    """Result of measuring one full-precision/quantized weight pair.

    ``measurements`` is the assumption-light primary result. Per-matrix
    variances are also summed for ``sigma_squared`` and ``eta_proxy``. That
    aggregate assumes independent layer contributions and isotropic,
    unit-variance activations; both assumptions are known to be false.
    """

    backend: str
    available: bool
    measurements: tuple[WeightNoiseMeasurement, ...]
    sigma_squared: float | None
    s_squared: float
    eta_proxy: float | None
    bits_per_param_wholefile: float | None
    bits_per_param_payload: float | None
    skipped: tuple[str, ...]
    reason: str | None = None


def projected_perturbation_variance(
    W_fp16: FloatArray,
    W_q: FloatArray,
    direction: FloatArray,
) -> float:
    """Return the variance of a projected weight perturbation.

    ``W_fp16`` and dequantized ``W_q`` must be matching 2-D matrices.
    ``direction`` is normalised internally, making the result invariant to its
    scale.  If its length matches the first matrix axis, the projection is
    ``direction @ (W_fp16 - W_q)`` (the usual ``[out, in]`` layout).  If only
    the second axis matches, the projection is ``(W_fp16 - W_q) @ direction``.
    For a square matrix, the first-axis convention wins.

    The population variance (``ddof=0``) is used because the projected entries
    are the complete weight perturbation being summarized, not a sample from
    an unobserved finite list.

    Raises ValueError for invalid dimensions, mismatched shapes, non-finite
    values, or a zero direction.
    """
    projected = _projected_weight_perturbation(W_fp16, W_q, direction)
    return float(np.var(projected, dtype=np.float64))


def _projected_weight_perturbation(
    W_fp16: FloatArray,
    W_q: FloatArray,
    direction: FloatArray,
) -> FloatArray:
    """Return the input-space vector induced by projecting ``W_fp16 - W_q``."""
    fp = np.asarray(W_fp16, dtype=np.float64)
    quantized = np.asarray(W_q, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64)

    if fp.ndim != 2 or quantized.ndim != 2:
        raise ValueError(
            f"W_fp16 and W_q must be 2-D, got {fp.shape} and {quantized.shape}"
        )
    if fp.shape != quantized.shape:
        raise ValueError(
            f"weight shape mismatch: W_fp16 has {fp.shape}, W_q has {quantized.shape}"
        )
    if vector.ndim != 1:
        raise ValueError(f"direction must be 1-D, got shape {vector.shape}")
    if not np.all(np.isfinite(fp)) or not np.all(np.isfinite(quantized)):
        raise ValueError("weight matrices must contain only finite values")
    if not np.all(np.isfinite(vector)):
        raise ValueError("direction must contain only finite values")

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("direction must have non-zero norm")
    unit = vector / norm
    perturbation = fp - quantized

    if unit.size == fp.shape[0]:
        projected: FloatArray = np.asarray(unit @ perturbation, dtype=np.float64)
    elif unit.size == fp.shape[1]:
        projected = np.asarray(perturbation @ unit, dtype=np.float64)
    else:
        raise ValueError(
            f"direction length {unit.size} matches neither weight axis {fp.shape}"
        )

    return projected


def eta_from_weights(
    W_fp16: FloatArray,
    W_q: FloatArray,
    direction: FloatArray,
    s_squared: float,
) -> float:
    """Return ``eta_q`` for one weight matrix and an external FP16 ``s^2``.

    ``s_squared`` must be finite and strictly positive.  The denominator is
    deliberately supplied by the caller: this module obtains the numerator
    from weights and does not inspect quantized behaviour observations.
    """
    _validate_s_squared(s_squared)
    return projected_perturbation_variance(W_fp16, W_q, direction) / s_squared


def eta_empirical(
    W_fp16: FloatArray,
    W_q: FloatArray,
    direction: FloatArray,
    activations: FloatArray,
) -> float:
    """Return ``Var_x[r^T (W_fp16 - W_q) a(x)]`` on benign activations.

    ``activations`` must contain real FP16 benign layer inputs with shape
    ``(n_samples, input_dimension)``. Population variance (``ddof=0``) is used
    over those supplied prompts. Unlike ``eta_proxy``, this calculation keeps
    the empirical activation covariance and makes no isotropy assumption.

    Supplying benign activations does not introduce behavioural supervision:
    no harmful prompts, judge, or labels are used. It supports a "no
    behavioural labels" prediction, but should not be described as
    "weights-only".
    """
    projected = _projected_weight_perturbation(W_fp16, W_q, direction)
    benign = np.asarray(activations, dtype=np.float64)
    if benign.ndim != 2:
        raise ValueError(f"activations must be 2-D, got shape {benign.shape}")
    if benign.shape[0] < 2:
        raise ValueError("activations must contain at least two benign samples")
    if benign.shape[1] != projected.size:
        raise ValueError(
            f"activation width {benign.shape[1]} does not match weight input "
            f"dimension {projected.size}"
        )
    if not np.all(np.isfinite(benign)):
        raise ValueError("activations must contain only finite values")
    perturbations = np.asarray(benign @ projected, dtype=np.float64)
    return float(np.var(perturbations, dtype=np.float64))


def wholefile_bits_per_parameter(file_path: Path, parameter_count: int) -> float:
    """Return ``8 * file_bytes / stored_parameter_count`` for an artifact.

    The complete file size is used, including scales, super-block metadata,
    tokenizer data, and format overhead. This is the deployed storage figure.

    Raises FileNotFoundError if ``file_path`` is absent and ValueError if the
    parameter count is not positive or the path is not a regular file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found: {path}")
    if not path.is_file():
        raise ValueError(f"model artifact must be a file: {path}")
    if parameter_count <= 0:
        raise ValueError(f"parameter_count must be positive, got {parameter_count}")
    return 8.0 * float(path.stat().st_size) / float(parameter_count)


def payload_bits_per_parameter(payload_bytes: int, parameter_count: int) -> float:
    """Return tensor-payload bits per stored parameter, excluding the container."""
    if payload_bytes < 0:
        raise ValueError(f"payload_bytes must be non-negative, got {payload_bytes}")
    if parameter_count <= 0:
        raise ValueError(f"parameter_count must be positive, got {parameter_count}")
    return 8.0 * float(payload_bytes) / float(parameter_count)


def effective_bit_width(file_path: Path, parameter_count: int) -> float:
    """Backward-compatible alias for :func:`wholefile_bits_per_parameter`.

    New reports use the unambiguous ``bits_per_param_wholefile`` and
    ``bits_per_param_payload`` names.
    """
    return wholefile_bits_per_parameter(file_path, parameter_count)


def fit_eta_vs_bits_report(
    eta_by_scheme: Mapping[str, EtaMeasurement],
    *,
    confidence_level: float = 0.95,
) -> EtaFitReport:
    """Fit and report the measured exponential relation between eta and bits.

    The fitted model is

        ``eta(bits) = eta_4 * exponent ** (4 - bits)``.

    Both ``eta_4`` and ``exponent`` are fitted by ordinary least squares in
    log space, using tensor-payload bits per parameter as the regressor. Exact
    zero values (normally the FP16 self-comparison) are
    omitted because their logarithm is undefined; negative values are errors.
    At least two positive-eta observations at distinct effective bit-widths
    are required.
    """
    if len(eta_by_scheme) < 2:
        raise ValueError("eta_by_scheme must contain at least two schemes")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    bits: list[float] = []
    etas: list[float] = []
    skipped_zero: list[str] = []
    for scheme, measurement in eta_by_scheme.items():
        bit_width = float(measurement.bits_per_param_payload)
        eta = float(measurement.eta)
        if not math.isfinite(bit_width) or bit_width <= 0.0:
            raise ValueError(
                f"payload bits/parameter for {scheme!r} must be finite and positive"
            )
        if not math.isfinite(eta) or eta < 0.0:
            raise ValueError(f"eta for {scheme!r} must be finite and non-negative")
        if eta == 0.0:
            skipped_zero.append(scheme)
            continue
        bits.append(bit_width)
        etas.append(eta)

    if len(etas) < 2:
        raise ValueError("at least two strictly positive eta values are required")

    x = 4.0 - np.asarray(bits, dtype=np.float64)
    y = np.log(np.asarray(etas, dtype=np.float64))
    x_centered = x - float(np.mean(x))
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0.0:
        raise ValueError("effective bit-widths must contain at least two distinct values")

    y_mean = float(np.mean(y))
    slope = float(np.dot(x_centered, y - y_mean) / denominator)
    intercept = y_mean - slope * float(np.mean(x))
    fitted = intercept + slope * x
    residual = y - fitted
    sum_squared_error = float(np.dot(residual, residual))
    total_sum_squares = float(np.dot(y - y_mean, y - y_mean))
    r_squared = (
        1.0 - sum_squared_error / total_sum_squares
        if total_sum_squares > 0.0
        else 1.0
    )
    rmse_log = math.sqrt(sum_squared_error / float(len(etas)))
    eta_4 = math.exp(intercept)
    exponent = math.exp(slope)
    if not math.isfinite(eta_4) or not math.isfinite(exponent):
        raise ValueError("fitted eta curve overflowed; inspect the input measurements")

    relative_deviation = abs(exponent - EXPECTED_EXPONENT) / EXPECTED_EXPONENT
    exponent_ci_low: float | None = None
    exponent_ci_high: float | None = None
    exponent_ci_excludes_four: bool | None = None
    degrees_of_freedom = len(etas) - 2
    if degrees_of_freedom > 0:
        residual_variance = max(0.0, sum_squared_error / float(degrees_of_freedom))
        slope_standard_error = math.sqrt(residual_variance / denominator)
        critical = float(
            student_t.ppf(0.5 + confidence_level / 2.0, degrees_of_freedom)
        )
        margin = critical * slope_standard_error
        exponent_ci_low = math.exp(slope - margin)
        exponent_ci_high = math.exp(slope + margin)
        numerical_tolerance = 1e-12 * EXPECTED_EXPONENT
        exponent_ci_excludes_four = bool(
            exponent_ci_high < EXPECTED_EXPONENT - numerical_tolerance
            or exponent_ci_low > EXPECTED_EXPONENT + numerical_tolerance
        )
    return EtaFitReport(
        eta_4=eta_4,
        exponent=exponent,
        r_squared=r_squared,
        rmse_log=rmse_log,
        n_points=len(etas),
        exponent_ci_low=exponent_ci_low,
        exponent_ci_high=exponent_ci_high,
        confidence_level=confidence_level,
        model_conditional_exponent_ci_excludes_four=exponent_ci_excludes_four,
        exceeds_25_percent_development_band=(
            relative_deviation > EXPONENT_RELATIVE_TOLERANCE
        ),
        skipped_zero_schemes=tuple(skipped_zero),
    )


def fit_eta_vs_bits(
    eta_by_scheme: Mapping[str, EtaMeasurement],
) -> tuple[float, float]:
    """Return ``(eta_4, exponent)`` fitted from measured scheme observations.

    Goodness-of-fit is available from :func:`fit_eta_vs_bits_report`.  This
    convenience function warns when log-space R-squared is below 0.90 or when
    the model-conditional fitted exponent confidence interval excludes four.
    Warning messages include the interval and fitted R-squared. Because a
    matched ladder can have correlated log-eta errors, the warning is a fit
    diagnostic rather than a project verdict. The separate 25 percent
    practical flag remains available from :func:`fit_eta_vs_bits_report`.
    """
    report = fit_eta_vs_bits_report(eta_by_scheme)
    if report.model_conditional_exponent_ci_excludes_four:
        assert report.exponent_ci_low is not None
        assert report.exponent_ci_high is not None
        warnings.warn(
            f"Under independent homoskedastic log-error OLS, the fitted eta "
            f"exponent is {report.exponent:.6g}; its model-conditional "
            f"{report.confidence_level:.0%} CI "
            f"[{report.exponent_ci_low:.6g}, {report.exponent_ci_high:.6g}] "
            f"excludes 4; "
            f"log-space R^2={report.r_squared:.6f}.",
            ExponentDeviationWarning,
            stacklevel=2,
        )
    if report.r_squared < POOR_FIT_R_SQUARED:
        warnings.warn(
            f"The exponential eta-vs-bits fit is poor: log-space "
            f"R^2={report.r_squared:.6f}, RMSE={report.rmse_log:.6g}.",
            PoorEtaFitWarning,
            stacklevel=2,
        )
    return (report.eta_4, report.exponent)


def measure_weight_mappings(
    fp16_weights: Mapping[str, FloatArray],
    quantized_weights: Mapping[str, FloatArray],
    directions: Mapping[str, FloatArray],
    s_squared: float,
    *,
    backend: str = "numpy",
    bits_per_param_wholefile: float | None = None,
    bits_per_param_payload: float | None = None,
) -> NoiseSpectrumReport:
    """Measure a matched pair of named, already-dequantized weight mappings.

    ``directions`` selects the matrices to measure and supplies the behavioural
    direction for each.  Missing or incompatible matrices are recorded in
    ``skipped``.  The report is unavailable only when no selected matrix could
    be measured.
    """
    _validate_s_squared(s_squared)
    measurements: list[WeightNoiseMeasurement] = []
    skipped: list[str] = []

    for name, direction in directions.items():
        if name not in fp16_weights:
            skipped.append(f"{name}: missing from FP16 weights")
            continue
        if name not in quantized_weights:
            skipped.append(f"{name}: missing from quantized weights")
            continue
        try:
            sigma_squared = projected_perturbation_variance(
                fp16_weights[name], quantized_weights[name], direction
            )
        except ValueError as exc:
            skipped.append(f"{name}: {exc}")
            continue
        measurements.append(
            WeightNoiseMeasurement(
                weight_name=name,
                sigma_squared=sigma_squared,
                eta=sigma_squared / s_squared,
            )
        )

    if not measurements:
        return _unavailable_report(
            backend,
            s_squared,
            "no matched, direction-compatible weight matrices were available",
            skipped=tuple(skipped),
            bits_per_param_wholefile=bits_per_param_wholefile,
            bits_per_param_payload=bits_per_param_payload,
        )

    sigma_total = math.fsum(item.sigma_squared for item in measurements)
    return NoiseSpectrumReport(
        backend=backend,
        available=True,
        measurements=tuple(measurements),
        sigma_squared=sigma_total,
        s_squared=s_squared,
        eta_proxy=sigma_total / s_squared,
        bits_per_param_wholefile=bits_per_param_wholefile,
        bits_per_param_payload=bits_per_param_payload,
        skipped=tuple(skipped),
    )


def measure_transformers_pair(
    fp16_model: object | None,
    quantized_model: object | None,
    directions: Mapping[str, FloatArray],
    s_squared: float,
) -> NoiseSpectrumReport:
    """Measure a loaded transformers FP16/bitsandbytes model pair.

    Parameter names must match the keys in ``directions``.  NF4 ``Params4bit``
    values are dequantized with their own ``quant_state``.  Ordinary torch
    tensors and NumPy-backed test doubles are also supported.

    If either model, a required optional package, or every requested weight is
    unavailable, returns ``available=False`` with a reason instead of raising.
    Invalid ``s_squared`` remains a caller error and raises ValueError.
    """
    _validate_s_squared(s_squared)
    if fp16_model is None or quantized_model is None:
        missing = "FP16" if fp16_model is None else "quantized"
        return _unavailable_report(
            "transformers-bnb", s_squared, f"{missing} model is unavailable"
        )

    try:
        fp16_weights, fp16_skipped = _extract_transformers_weights(
            fp16_model, frozenset(directions)
        )
        quantized_weights, quantized_skipped = _extract_transformers_weights(
            quantized_model, frozenset(directions)
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        return _unavailable_report(
            "transformers-bnb", s_squared, f"could not read model weights: {exc}"
        )

    report = measure_weight_mappings(
        fp16_weights,
        quantized_weights,
        directions,
        s_squared,
        backend="transformers-bnb",
    )
    combined_skipped = fp16_skipped + quantized_skipped + report.skipped
    return NoiseSpectrumReport(
        backend=report.backend,
        available=report.available,
        measurements=report.measurements,
        sigma_squared=report.sigma_squared,
        s_squared=report.s_squared,
        eta_proxy=report.eta_proxy,
        bits_per_param_wholefile=report.bits_per_param_wholefile,
        bits_per_param_payload=report.bits_per_param_payload,
        skipped=combined_skipped,
        reason=report.reason,
    )


def measure_gguf_pair(
    fp16_source: object | None,
    quantized_source: object | None,
    directions: Mapping[str, FloatArray],
    s_squared: float,
) -> NoiseSpectrumReport:
    """Measure a matched F16/quantized GGUF pair used by llama.cpp.

    A source may be a path, a path-like string, or a loaded
    ``LlamaCppAdapter``/llama.cpp object exposing ``model_path``.  Tensor bytes
    are read and dequantized through llama.cpp's optional ``gguf`` Python
    package. Whole-file and tensor-payload bits per parameter are reported
    separately; the payload measure excludes GGUF headers, metadata, and
    tokenizer data.

    Missing sources, files, or the optional ``gguf`` package return an
    unavailable report rather than breaking environments that only have the
    transformers pair.
    """
    _validate_s_squared(s_squared)
    fp16_path = _resolve_gguf_path(fp16_source)
    quantized_path = _resolve_gguf_path(quantized_source)
    if fp16_path is None or quantized_path is None:
        missing = "FP16" if fp16_path is None else "quantized"
        return _unavailable_report("gguf", s_squared, f"{missing} GGUF source is unavailable")
    if not fp16_path.is_file() or not quantized_path.is_file():
        missing_path = fp16_path if not fp16_path.is_file() else quantized_path
        return _unavailable_report(
            "gguf", s_squared, f"GGUF artifact is unavailable: {missing_path}"
        )

    try:
        fp16_weights, fp16_count, _fp16_payload, fp16_skipped = _extract_gguf_weights(
            fp16_path, frozenset(directions)
        )
        (
            quantized_weights,
            quantized_count,
            quantized_payload,
            quantized_skipped,
        ) = _extract_gguf_weights(quantized_path, frozenset(directions))
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _unavailable_report(
            "gguf", s_squared, f"could not read GGUF weights: {exc}"
        )

    parameter_count = quantized_count if quantized_count > 0 else fp16_count
    wholefile_bits = (
        wholefile_bits_per_parameter(quantized_path, parameter_count)
        if parameter_count > 0
        else None
    )
    payload_bits = (
        payload_bits_per_parameter(quantized_payload, parameter_count)
        if parameter_count > 0
        else None
    )
    report = measure_weight_mappings(
        fp16_weights,
        quantized_weights,
        directions,
        s_squared,
        backend="gguf",
        bits_per_param_wholefile=wholefile_bits,
        bits_per_param_payload=payload_bits,
    )
    combined_skipped = fp16_skipped + quantized_skipped + report.skipped
    return NoiseSpectrumReport(
        backend=report.backend,
        available=report.available,
        measurements=report.measurements,
        sigma_squared=report.sigma_squared,
        s_squared=report.s_squared,
        eta_proxy=report.eta_proxy,
        bits_per_param_wholefile=report.bits_per_param_wholefile,
        bits_per_param_payload=report.bits_per_param_payload,
        skipped=combined_skipped,
        reason=report.reason,
    )


def _validate_s_squared(s_squared: float) -> None:
    if not math.isfinite(s_squared) or s_squared <= 0.0:
        raise ValueError(f"s_squared must be finite and positive, got {s_squared}")


def _unavailable_report(
    backend: str,
    s_squared: float,
    reason: str,
    *,
    skipped: tuple[str, ...] = (),
    bits_per_param_wholefile: float | None = None,
    bits_per_param_payload: float | None = None,
) -> NoiseSpectrumReport:
    return NoiseSpectrumReport(
        backend=backend,
        available=False,
        measurements=(),
        sigma_squared=None,
        s_squared=s_squared,
        eta_proxy=None,
        bits_per_param_wholefile=bits_per_param_wholefile,
        bits_per_param_payload=bits_per_param_payload,
        skipped=skipped,
        reason=reason,
    )


def _extract_transformers_weights(
    model: object,
    wanted: frozenset[str],
) -> tuple[dict[str, FloatArray], tuple[str, ...]]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model does not expose named_parameters()")

    raw_parameters = cast(Iterable[tuple[str, object]], named_parameters())
    weights: dict[str, FloatArray] = {}
    skipped: list[str] = []
    for name, parameter in raw_parameters:
        if name not in wanted:
            continue
        try:
            weights[name] = _tensor_to_numpy(parameter)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            skipped.append(f"{name}: could not dequantize tensor: {exc}")
    return weights, tuple(skipped)


def _tensor_to_numpy(tensor: object) -> FloatArray:
    value = tensor
    quant_state = getattr(value, "quant_state", None)
    if quant_state is not None:
        functional = importlib.import_module("bitsandbytes.functional")
        dequantize_4bit = getattr(functional, "dequantize_4bit", None)
        if not callable(dequantize_4bit):
            raise ImportError("bitsandbytes.functional.dequantize_4bit is unavailable")
        packed = getattr(value, "data", value)
        value = dequantize_4bit(packed, quant_state=quant_state)
    elif bool(getattr(value, "is_quantized", False)):
        dequantize = getattr(value, "dequantize", None)
        if not callable(dequantize):
            raise TypeError("quantized tensor does not expose dequantize()")
        value = dequantize()

    for method_name in ("detach", "float", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    array: FloatArray = np.asarray(value, dtype=np.float64)
    return array


def _resolve_gguf_path(source: object | None) -> Path | None:
    if source is None:
        return None
    if isinstance(source, Path):
        return source
    if isinstance(source, str):
        return Path(source)
    model_path = getattr(source, "model_path", None)
    if isinstance(model_path, (str, Path)):
        return Path(model_path)
    return None


def _extract_gguf_weights(
    path: Path,
    wanted: frozenset[str],
) -> tuple[dict[str, FloatArray], int, int, tuple[str, ...]]:
    gguf = importlib.import_module("gguf")
    reader_type = getattr(gguf, "GGUFReader", None)
    dequantize = getattr(gguf, "dequantize", None)
    if not callable(reader_type) or not callable(dequantize):
        raise ImportError("gguf.GGUFReader and gguf.dequantize are required")

    reader = reader_type(str(path))
    tensors = cast(Iterable[Any], getattr(reader, "tensors"))
    weights: dict[str, FloatArray] = {}
    skipped: list[str] = []
    parameter_count = 0
    payload_bytes = 0
    for tensor in tensors:
        parameter_count += int(tensor.n_elements)
        stored_bytes = getattr(tensor, "n_bytes", None)
        if stored_bytes is None:
            stored_bytes = np.asarray(tensor.data).nbytes
        payload_bytes += int(stored_bytes)
        name = str(tensor.name)
        if name not in wanted:
            continue
        try:
            raw = dequantize(tensor.data, tensor.tensor_type)
            weights[name] = np.asarray(raw, dtype=np.float64)
        except (RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
            skipped.append(f"{name}: could not dequantize GGUF tensor: {exc}")
    return weights, parameter_count, payload_bytes, tuple(skipped)
