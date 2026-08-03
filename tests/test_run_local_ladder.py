"""Tests for the RTN ladder runner.

The quantizer is the instrument the whole ladder is measured with, so its
properties are pinned here rather than assumed. A silently-wrong quantizer would
not crash; it would produce a smooth, plausible, meaningless eta curve.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.run_local_ladder import (  # noqa: E402
    load_prompts,
    rtn_bits_per_parameter,
    rtn_quantize_dequantize,
    salience_scaled_quantize_dequantize,
)


def _weight(rows: int = 32, cols: int = 256, seed: int = 0) -> object:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator, dtype=torch.float16)


# ---------------------------------------------------------------------------
# shape, dtype, and the degenerate cases
# ---------------------------------------------------------------------------


def test_preserves_shape_and_dtype() -> None:
    w = _weight()
    out = rtn_quantize_dequantize(w, bits=4, group=64)
    assert out.shape == w.shape
    assert out.dtype == w.dtype


def test_handles_group_not_dividing_input_width() -> None:
    """Real models have odd intermediate sizes; padding must not corrupt the tail."""
    w = _weight(cols=200)          # 200 % 64 != 0
    out = rtn_quantize_dequantize(w, bits=8, group=64)
    assert out.shape == w.shape
    assert torch.isfinite(out.float()).all()


def test_constant_group_reconstructs_exactly() -> None:
    """A group with zero range would divide by zero if the scale were not guarded."""
    w = torch.full((4, 64), 0.25, dtype=torch.float16)
    out = rtn_quantize_dequantize(w, bits=2, group=64)
    assert torch.allclose(out.float(), w.float())


def test_all_zero_weight_survives() -> None:
    w = torch.zeros(4, 128, dtype=torch.float16)
    out = rtn_quantize_dequantize(w, bits=2, group=64)
    assert torch.equal(out, w)


# ---------------------------------------------------------------------------
# the property the ladder depends on
# ---------------------------------------------------------------------------


def test_error_grows_monotonically_as_bits_fall() -> None:
    """The ordinal dose axis is only meaningful if more bits means less damage."""
    w = _weight()
    errors = [
        float((rtn_quantize_dequantize(w, bits=b, group=64).float() - w.float()).pow(2).mean())
        for b in (8, 7, 6, 5, 4, 3, 2)
    ]
    assert errors == sorted(errors), f"error not monotone in bits: {errors}"


def test_error_falls_as_group_shrinks() -> None:
    """Smaller groups fit the local range better, at the cost of more scale overhead."""
    w = _weight()
    coarse = float((rtn_quantize_dequantize(w, 4, 256).float() - w.float()).pow(2).mean())
    fine = float((rtn_quantize_dequantize(w, 4, 32).float() - w.float()).pow(2).mean())
    assert fine < coarse


def test_reconstruction_stays_inside_the_original_range() -> None:
    """Asymmetric RTN interpolates; it must never extrapolate past the group's min/max."""
    w = _weight()
    out = rtn_quantize_dequantize(w, bits=3, group=64).float()
    assert out.max() <= w.float().max() + 1e-3
    assert out.min() >= w.float().min() - 1e-3


def test_error_scales_with_the_quantization_step() -> None:
    """RTN error should track 4^-b, the assumption A3 the ladder is built to test.

    Checked as an order-of-magnitude property rather than an equality: the
    per-group min/max fit makes the constant depend on the weight distribution.
    """
    w = _weight(rows=64, cols=512, seed=3)
    mse = {
        b: float((rtn_quantize_dequantize(w, bits=b, group=64).float() - w.float()).pow(2).mean())
        for b in (6, 4)
    }
    ratio = mse[4] / mse[6]
    assert 4.0 < ratio < 64.0, f"two fewer bits changed MSE by {ratio:.1f}x, expected ~16x"


def test_eight_bits_is_near_lossless() -> None:
    w = _weight()
    out = rtn_quantize_dequantize(w, bits=8, group=64).float()
    relative = float((out - w.float()).norm() / w.float().norm())
    assert relative < 0.01, f"8-bit RTN relative error {relative:.4f} is too large"


def test_two_bits_is_visibly_lossy() -> None:
    """If the bottom rung were not damaging, the ladder could not show a cliff."""
    w = _weight()
    out = rtn_quantize_dequantize(w, bits=2, group=64).float()
    relative = float((out - w.float()).norm() / w.float().norm())
    assert relative > 0.05, f"2-bit RTN relative error {relative:.4f} is implausibly small"


def test_quantized_values_are_drawn_from_a_small_alphabet() -> None:
    """b bits must yield at most 2**b distinct reconstructions per group."""
    w = _weight(rows=1, cols=64)
    out = rtn_quantize_dequantize(w, bits=3, group=64)
    assert len(torch.unique(out)) <= 8


# ---------------------------------------------------------------------------
# the bit accounting that becomes the regressor
# ---------------------------------------------------------------------------


def test_bits_per_parameter_includes_group_overhead() -> None:
    """fp16 scale + fp16 zero per group = 32 bits amortised over `group` weights."""
    assert rtn_bits_per_parameter(4, 64) == pytest.approx(4.5)
    assert rtn_bits_per_parameter(8, 128) == pytest.approx(8.25)
    assert rtn_bits_per_parameter(2, 32) == pytest.approx(3.0)


def test_bits_per_parameter_is_monotone_in_both_arguments() -> None:
    assert rtn_bits_per_parameter(3, 64) < rtn_bits_per_parameter(4, 64)
    assert rtn_bits_per_parameter(4, 128) < rtn_bits_per_parameter(4, 64)


def test_bits_per_parameter_ordering_matches_measured_error() -> None:
    """The regressor must order the rungs the same way the actual damage does.

    A cheaper-looking rung that is in fact more damaged would invert the fitted
    exponent's sign without any error being raised.
    """
    w = _weight(rows=64, cols=512, seed=7)
    rungs = [(8, 64), (6, 64), (4, 64), (3, 64), (2, 64)]
    by_bits = sorted(rungs, key=lambda bg: -rtn_bits_per_parameter(*bg))
    errors = [
        float((rtn_quantize_dequantize(w, b, g).float() - w.float()).pow(2).mean())
        for b, g in by_bits
    ]
    assert errors == sorted(errors), f"bit accounting disagrees with measured error: {errors}"


# ---------------------------------------------------------------------------
# salience-aware quantization — the controlled arm of claim C5
# ---------------------------------------------------------------------------


def _scale(cols: int = 256, seed: int = 1) -> object:
    """A spiky activation profile: a few channels far above the rest."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(cols, generator=generator) * 0.1 + 0.01
    base[::32] *= 40.0
    return base


def test_alpha_zero_is_exactly_plain_rtn() -> None:
    """The control must be bit-identical, or the comparison measures two changes."""
    w = _weight()
    assert torch.equal(
        salience_scaled_quantize_dequantize(w, 4, 64, _scale(), alpha=0.0),
        rtn_quantize_dequantize(w, 4, 64),
    )


def test_salience_scaling_changes_the_reconstruction() -> None:
    w = _weight()
    assert not torch.equal(
        salience_scaled_quantize_dequantize(w, 4, 64, _scale(), alpha=0.5),
        rtn_quantize_dequantize(w, 4, 64),
    )


def test_salient_channels_are_protected_at_equal_bit_budget() -> None:
    """The whole point: error on high-activation channels must fall.

    This is the property the C5 arm rests on. If it did not hold, the observed
    reduction in direction rotation would have some other cause.
    """
    w = _weight(rows=64, cols=256, seed=5)
    scale = _scale()
    salient = torch.arange(0, 256, 32)          # the channels boosted in _scale

    def channel_mse(out: object) -> object:
        return (out.float() - w.float()).pow(2).mean(dim=0)

    plain = channel_mse(rtn_quantize_dequantize(w, 3, 64))
    aware = channel_mse(salience_scaled_quantize_dequantize(w, 3, 64, scale, alpha=0.5))
    assert float(aware[salient].mean()) < float(plain[salient].mean())


def test_protection_is_paid_for_in_total_error() -> None:
    """Salience-awareness is a reallocation, not a free lunch.

    Measured on the real ladder: 1.5-2.0x more total weight perturbation buying
    22-26% less rotation of the behavioural direction. The sign of that trade is
    pinned here so a future change cannot silently turn it into a strict
    improvement, which would mean the scaling is no longer doing what it claims.
    """
    w = _weight(rows=64, cols=256, seed=5)
    plain = float((rtn_quantize_dequantize(w, 3, 64).float() - w.float()).pow(2).mean())
    aware = float(
        (salience_scaled_quantize_dequantize(w, 3, 64, _scale(), alpha=0.5).float()
         - w.float()).pow(2).mean()
    )
    assert aware > plain


def test_flat_activation_profile_is_nearly_a_no_op() -> None:
    """With no salience to exploit, the transform should barely move anything."""
    w = _weight()
    flat = torch.ones(256)
    aware = salience_scaled_quantize_dequantize(w, 5, 64, flat, alpha=0.5)
    plain = rtn_quantize_dequantize(w, 5, 64)
    assert torch.allclose(aware.float(), plain.float(), atol=1e-2)


def test_protection_has_an_interior_optimum_in_alpha() -> None:
    """Protection is NOT monotone in alpha, and that is why alpha is searched.

    Measured here: salient-channel MSE goes 0.00637 (alpha=0) -> 0.00541 (0.25)
    -> 0.00247 (0.5) -> 0.00391 (1.0). Past some point the boosted channels
    dominate their group's min/max, so the shared affine range stretches to cover
    them and every channel in the group -- including the salient ones -- loses
    resolution. Cranking alpha to 1 is therefore worse than 0.5.

    This is exactly why AWQ grid-searches the exponent instead of maximising it,
    and it justifies 0.5 as the default the C5 arm runs at. A future change that
    made this monotone would mean the scaling had stopped interacting with the
    group range, i.e. it was no longer doing what it claims.
    """
    w = _weight(rows=64, cols=256, seed=5)
    scale = _scale()
    salient = torch.arange(0, 256, 32)

    def salient_mse(alpha: float) -> float:
        out = salience_scaled_quantize_dequantize(w, 3, 64, scale, alpha=alpha)
        return float((out.float() - w.float()).pow(2).mean(dim=0)[salient].mean())

    errors = {alpha: salient_mse(alpha) for alpha in (0.0, 0.25, 0.5, 1.0)}
    assert errors[0.5] < errors[0.0], "alpha=0.5 must beat the unprotected control"
    assert errors[0.5] < errors[0.25], "protection must still be improving at 0.5"
    assert errors[1.0] > errors[0.5], "over-scaling must degrade protection"


def test_salience_preserves_shape_dtype_and_finiteness() -> None:
    w = _weight()
    out = salience_scaled_quantize_dequantize(w, 2, 64, _scale(), alpha=0.5)
    assert out.shape == w.shape and out.dtype == w.dtype
    assert torch.isfinite(out.float()).all()


def test_salience_survives_zero_activation_channels() -> None:
    """A dead channel must not produce a division by zero or a NaN weight."""
    w = _weight()
    scale = _scale()
    scale[:8] = 0.0
    out = salience_scaled_quantize_dequantize(w, 4, 64, scale, alpha=0.5)
    assert torch.isfinite(out.float()).all()


# ---------------------------------------------------------------------------
# corpus loading
# ---------------------------------------------------------------------------


def test_load_prompts_refuses_to_invent_a_corpus(tmp_path, monkeypatch) -> None:
    """A missing corpus must stop the run, not silently produce placeholder prompts."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="Fold A corpus not found"):
        load_prompts(10)


def test_load_prompts_dedupes_within_and_across_classes(tmp_path, monkeypatch) -> None:
    """Repeats fake precision; a prompt in both classes is a label contradiction."""
    import json

    fold = tmp_path / "data" / "folds" / "fold_a"
    fold.mkdir(parents=True)
    refused = ["a", "b", "a", "c", "d"]
    benign = ["c", "e", "e", "f", "g", "h"]      # "c" also appears as refused
    for name, rows in (("anthropic_hh_refused", refused), ("anthropic_hh_benign", benign)):
        (fold / f"{name}.jsonl").write_text(
            "\n".join(json.dumps({"prompt": p, "source": "test"}) for p in rows),
            encoding="utf-8",
        )
    monkeypatch.chdir(tmp_path)

    harmful, harmless = load_prompts(4)
    assert harmful == ["a", "b", "c", "d"]
    assert "c" not in harmless
    assert len(set(harmless)) == len(harmless)
    assert not set(harmful) & set(harmless)


def test_load_prompts_rejects_a_corpus_that_is_too_small(tmp_path, monkeypatch) -> None:
    import json

    fold = tmp_path / "data" / "folds" / "fold_a"
    fold.mkdir(parents=True)
    for name in ("anthropic_hh_refused", "anthropic_hh_benign"):
        (fold / f"{name}.jsonl").write_text(
            json.dumps({"prompt": "only one", "source": "test"}), encoding="utf-8"
        )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="corpus too small"):
        load_prompts(10)


def test_quantizer_is_deterministic() -> None:
    """Round-to-nearest has no randomness; two calls must agree bit for bit."""
    w = _weight()
    assert torch.equal(
        rtn_quantize_dequantize(w, 4, 64), rtn_quantize_dequantize(w, 4, 64)
    )


def test_numpy_roundtrip_matches_torch_reference() -> None:
    """Guards against a silent dtype or reshape change in the group folding."""
    w = _weight(rows=2, cols=128, seed=11)
    out = rtn_quantize_dequantize(w, bits=4, group=32).float().numpy()
    reference = np.empty_like(out)
    raw = w.float().numpy()
    for row in range(raw.shape[0]):
        for start in range(0, raw.shape[1], 32):
            block = raw[row, start : start + 32]
            lo, hi = block.min(), block.max()
            scale = (hi - lo) / 15.0 if hi > lo else 1.0
            codes = np.clip(np.round((block - lo) / scale), 0, 15)
            reference[row, start : start + 32] = codes * scale + lo
    assert np.allclose(out, reference, atol=1e-3)
