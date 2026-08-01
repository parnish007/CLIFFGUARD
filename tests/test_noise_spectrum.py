from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from cliffguard.eval.noise_spectrum import (
    EtaMeasurement,
    ExponentDeviationWarning,
    effective_bit_width,
    eta_empirical,
    eta_from_weights,
    fit_eta_vs_bits,
    fit_eta_vs_bits_report,
    measure_gguf_pair,
    measure_transformers_pair,
    measure_weight_mappings,
    payload_bits_per_parameter,
    projected_perturbation_variance,
    wholefile_bits_per_parameter,
)


def test_projected_variance_uses_output_axis_and_population_variance() -> None:
    fp16 = np.array(
        [[0.0, 1.0, 2.0, 3.0], [10.0, 10.0, 10.0, 10.0]],
        dtype=np.float64,
    )
    quantized = np.zeros_like(fp16)
    direction = np.array([1.0, 0.0], dtype=np.float64)
    assert projected_perturbation_variance(fp16, quantized, direction) == pytest.approx(1.25)


def test_projected_variance_normalises_direction() -> None:
    fp16 = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 8.0]], dtype=np.float64)
    quantized = np.zeros_like(fp16)
    direction = np.array([1.0, -1.0], dtype=np.float64)
    scaled = projected_perturbation_variance(fp16, quantized, 7.0 * direction)
    unscaled = projected_perturbation_variance(fp16, quantized, direction)
    assert scaled == pytest.approx(unscaled)


def test_projected_variance_supports_direction_on_second_axis() -> None:
    fp16 = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    quantized = np.zeros_like(fp16)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    assert projected_perturbation_variance(fp16, quantized, direction) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("fp16", "quantized", "direction", "message"),
    [
        (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            "shape mismatch",
        ),
        (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            "non-zero norm",
        ),
        (
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            np.ones(4, dtype=np.float64),
            "matches neither",
        ),
    ],
)
def test_projected_variance_rejects_invalid_inputs(
    fp16: np.ndarray,
    quantized: np.ndarray,
    direction: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        projected_perturbation_variance(fp16, quantized, direction)


def test_eta_from_weights_divides_by_fp16_variance() -> None:
    fp16 = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    quantized = np.zeros_like(fp16)
    direction = np.array([1.0, 0.0], dtype=np.float64)
    assert eta_from_weights(fp16, quantized, direction, s_squared=0.25) == pytest.approx(5.0)


@pytest.mark.parametrize("s_squared", [0.0, -1.0, float("nan"), float("inf")])
def test_eta_from_weights_requires_positive_finite_denominator(s_squared: float) -> None:
    weight = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="s_squared"):
        eta_from_weights(weight, weight, np.ones(2, dtype=np.float64), s_squared)


def test_eta_empirical_recovers_isotropic_covariance_value() -> None:
    rng = np.random.default_rng(7)
    fp16 = np.array([[1.0, -1.0], [0.0, 0.0]], dtype=np.float64)
    quantized = np.zeros_like(fp16)
    direction = np.array([1.0, 0.0], dtype=np.float64)
    activations = rng.normal(size=(200_000, 2))

    # For covariance I and projected perturbation [1, -1], the analytic value
    # is [1, -1] I [1, -1]^T = 2.
    assert eta_empirical(fp16, quantized, direction, activations) == pytest.approx(
        2.0, rel=0.015
    )


def test_eta_empirical_exposes_anisotropy_that_eta_proxy_misses() -> None:
    rng = np.random.default_rng(11)
    fp16 = np.array([[1.0, -1.0], [0.0, 0.0]], dtype=np.float64)
    quantized = np.zeros_like(fp16)
    direction = np.array([1.0, 0.0], dtype=np.float64)
    activations = rng.normal(size=(200_000, 2)) * np.array([10.0, 1.0])

    empirical = eta_empirical(fp16, quantized, direction, activations)
    report = measure_weight_mappings(
        {"layer": fp16},
        {"layer": quantized},
        {"layer": direction},
        s_squared=1.0,
    )
    # Covariance diag(100, 1) gives the analytic empirical variance 101. The
    # weight-only proxy is 1 and therefore disagrees by construction.
    assert empirical == pytest.approx(101.0, rel=0.015)
    assert report.eta_proxy == pytest.approx(1.0)
    assert empirical > 90.0 * report.eta_proxy


def test_eta_empirical_rejects_activation_width_mismatch() -> None:
    weight = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="activation width"):
        eta_empirical(
            weight,
            weight,
            np.ones(2, dtype=np.float64),
            np.ones((10, 4), dtype=np.float64),
        )


def test_measure_weight_mappings_reports_primary_spectrum_and_eta_proxy() -> None:
    fp16 = {
        "a": np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]]),
        "b": np.array([[0.0, 2.0, 4.0, 6.0], [0.0, 0.0, 0.0, 0.0]]),
    }
    quantized = {name: np.zeros_like(weight) for name, weight in fp16.items()}
    directions = {
        "a": np.array([1.0, 0.0], dtype=np.float64),
        "b": np.array([1.0, 0.0], dtype=np.float64),
    }
    report = measure_weight_mappings(fp16, quantized, directions, s_squared=0.5)
    assert report.available is True
    assert len(report.measurements) == 2
    assert report.sigma_squared == pytest.approx(6.25)
    assert report.eta_proxy == pytest.approx(12.5)


def test_measure_weight_mappings_records_missing_tensors() -> None:
    direction = {"missing": np.ones(2, dtype=np.float64)}
    report = measure_weight_mappings({}, {}, direction, s_squared=1.0)
    assert report.available is False
    assert report.eta_proxy is None
    assert report.reason is not None
    assert "missing from FP16" in report.skipped[0]


def _eta_law(eta_4: float, exponent: float) -> dict[str, EtaMeasurement]:
    return {
        f"scheme-{bits}": EtaMeasurement(
            bits_per_param_wholefile=float(30 - bits),
            bits_per_param_payload=float(bits),
            eta=eta_4 * exponent ** (4.0 - float(bits)),
        )
        for bits in (8, 6, 5, 4, 3, 2)
    }


def test_fit_eta_vs_bits_recovers_eta4_and_exponent() -> None:
    eta_4, exponent = fit_eta_vs_bits(_eta_law(eta_4=0.3, exponent=4.0))
    assert eta_4 == pytest.approx(0.3)
    assert exponent == pytest.approx(4.0)


def test_fit_eta_report_exposes_goodness_of_fit() -> None:
    report = fit_eta_vs_bits_report(_eta_law(eta_4=0.2, exponent=4.0))
    assert report.r_squared == pytest.approx(1.0)
    assert report.rmse_log == pytest.approx(0.0, abs=1e-14)
    assert report.n_points == 6
    assert report.exponent_ci_low == pytest.approx(4.0)
    assert report.exponent_ci_high == pytest.approx(4.0)
    assert report.model_conditional_exponent_ci_excludes_four is False
    assert report.exponent_ci_excludes_four is False
    assert report.exceeds_25_percent_development_band is False
    assert report.practically_deviates_from_four is False


def test_fit_eta_vs_bits_flags_exponent_far_from_four() -> None:
    observations = _eta_law(eta_4=0.2, exponent=2.0)
    with pytest.warns(ExponentDeviationWarning, match=r"R\^2"):
        eta_4, exponent = fit_eta_vs_bits(observations)
    assert eta_4 == pytest.approx(0.2)
    assert exponent == pytest.approx(2.0)
    report = fit_eta_vs_bits_report(observations)
    assert report.model_conditional_exponent_ci_excludes_four is True
    assert report.exponent_ci_excludes_four is True
    assert report.exceeds_25_percent_development_band is True
    assert report.practically_deviates_from_four is True


def test_exponent_ci_is_primary_even_without_practical_deviation() -> None:
    observations = _eta_law(eta_4=0.2, exponent=3.5)
    report = fit_eta_vs_bits_report(observations)
    assert report.model_conditional_exponent_ci_excludes_four is True
    assert report.exponent_ci_excludes_four is True
    assert report.practically_deviates_from_four is False


def test_fit_eta_report_skips_exact_fp16_zero() -> None:
    observations = _eta_law(eta_4=0.3, exponent=4.0)
    observations["FP16"] = EtaMeasurement(
        bits_per_param_wholefile=17.0,
        bits_per_param_payload=16.0,
        eta=0.0,
    )
    report = fit_eta_vs_bits_report(observations)
    assert report.skipped_zero_schemes == ("FP16",)
    assert report.exponent == pytest.approx(4.0)


def test_fit_eta_requires_distinct_bit_widths() -> None:
    observations = {
        "a": EtaMeasurement(5.0, 4.0, 0.1),
        "b": EtaMeasurement(9.0, 4.0, 0.2),
    }
    with pytest.raises(ValueError, match="distinct"):
        fit_eta_vs_bits(observations)


def test_effective_bit_width_uses_actual_file_size(tmp_path: Path) -> None:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"x" * 100)
    assert effective_bit_width(artifact, parameter_count=50) == pytest.approx(16.0)
    assert wholefile_bits_per_parameter(artifact, parameter_count=50) == pytest.approx(16.0)


def test_payload_bits_per_parameter_excludes_container_bytes() -> None:
    assert payload_bits_per_parameter(payload_bytes=60, parameter_count=100) == pytest.approx(4.8)


class _FakeModel:
    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.weights = weights

    def named_parameters(self) -> Iterable[tuple[str, object]]:
        return self.weights.items()


def test_transformers_pair_accepts_loaded_model_pair() -> None:
    fp16_weight = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    quantized_weight = np.zeros_like(fp16_weight)
    direction = {"layer.weight": np.array([1.0, 0.0], dtype=np.float64)}
    report = measure_transformers_pair(
        _FakeModel({"layer.weight": fp16_weight}),
        _FakeModel({"layer.weight": quantized_weight}),
        direction,
        s_squared=1.0,
    )
    assert report.available is True
    assert report.backend == "transformers-bnb"
    assert report.eta_proxy == pytest.approx(2.0 / 3.0)


class _FakeParams4bit:
    def __init__(self, packed: np.ndarray, dequantized: np.ndarray) -> None:
        self.data = packed
        self.quant_state = {"dequantized": dequantized}


def test_transformers_pair_dequantizes_params4bit_with_quant_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp16_weight = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    quantized_weight = np.zeros_like(fp16_weight)
    packed = np.array([1, 2, 3], dtype=np.uint8)
    params4bit = _FakeParams4bit(packed, quantized_weight)

    functional = ModuleType("bitsandbytes.functional")

    def fake_dequantize(data: object, *, quant_state: object) -> np.ndarray:
        assert data is packed
        assert isinstance(quant_state, dict)
        return np.asarray(quant_state["dequantized"], dtype=np.float64)

    functional.dequantize_4bit = fake_dequantize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", functional)

    report = measure_transformers_pair(
        _FakeModel({"layer.weight": fp16_weight}),
        _FakeModel({"layer.weight": params4bit}),
        {"layer.weight": np.array([1.0, 0.0], dtype=np.float64)},
        s_squared=1.0,
    )
    assert report.available is True
    assert report.eta_proxy == pytest.approx(2.0 / 3.0)


def test_transformers_pair_degrades_gracefully_when_one_model_is_missing() -> None:
    report = measure_transformers_pair(None, _FakeModel({}), {}, s_squared=1.0)
    assert report.available is False
    assert report.eta_proxy is None
    assert report.reason == "FP16 model is unavailable"


@dataclass
class _FakeLlamaAdapter:
    model_path: str


@dataclass
class _FakeGgufTensor:
    name: str
    data: np.ndarray
    n_elements: int
    tensor_type: str = "F32"
    n_bytes: int | None = None


def test_gguf_pair_reads_adapter_paths_and_measures_actual_bits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fp16_path = tmp_path / "model-f16.gguf"
    quantized_path = tmp_path / "model-q4.gguf"
    fp16_path.write_bytes(b"f" * 160)
    quantized_path.write_bytes(b"q" * 40)

    fp16_weight = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    quantized_weight = np.zeros_like(fp16_weight)
    tensors_by_path = {
        str(fp16_path): [_FakeGgufTensor("layer.weight", fp16_weight, 6, n_bytes=12)],
        str(quantized_path): [
            _FakeGgufTensor("layer.weight", quantized_weight, 6, n_bytes=6)
        ],
    }

    class FakeReader:
        def __init__(self, path: str) -> None:
            self.tensors = tensors_by_path[path]

    fake_gguf = ModuleType("gguf")
    fake_gguf.GGUFReader = FakeReader  # type: ignore[attr-defined]
    fake_gguf.dequantize = lambda data, _tensor_type: data  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)

    report = measure_gguf_pair(
        _FakeLlamaAdapter(str(fp16_path)),
        _FakeLlamaAdapter(str(quantized_path)),
        {"layer.weight": np.array([1.0, 0.0], dtype=np.float64)},
        s_squared=1.0,
    )
    assert report.available is True
    assert report.backend == "gguf"
    assert report.eta_proxy == pytest.approx(2.0 / 3.0)
    assert report.bits_per_param_wholefile == pytest.approx(40.0 * 8.0 / 6.0)
    assert report.bits_per_param_payload == pytest.approx(6.0 * 8.0 / 6.0)


def test_gguf_pair_degrades_gracefully_when_one_source_is_missing(tmp_path: Path) -> None:
    present = tmp_path / "present.gguf"
    present.write_bytes(b"GGUF")
    report = measure_gguf_pair(None, present, {}, s_squared=1.0)
    assert report.available is False
    assert report.reason == "FP16 GGUF source is unavailable"
