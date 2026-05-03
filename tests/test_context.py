import numpy as np
import pytest

from cliffguard.attest.wh import AttestResult
from cliffguard.types import GateVerdict, Tier
from cliffguard.conductor.context import (
    CONTEXT_DIM,
    TIER_INDICATOR,
    build_context,
    context_dim,
)


def _verdict(gate: str, score: float, fired: bool = False) -> GateVerdict:
    return GateVerdict(gate=gate, score=score, fired=fired, threshold=0.5, tier=Tier.A)


# ---------------------------------------------------------------------------
# context_dim
# ---------------------------------------------------------------------------


def test_context_dim_returns_14() -> None:
    assert context_dim() == 14


# ---------------------------------------------------------------------------
# build_context shape and dtype
# ---------------------------------------------------------------------------


def test_build_context_shape() -> None:
    x = build_context([], Tier.A)
    assert x.shape == (CONTEXT_DIM,)


def test_build_context_dtype_float64() -> None:
    x = build_context([], Tier.A)
    assert x.dtype == np.float64


# ---------------------------------------------------------------------------
# Tier indicator (index 13)
# ---------------------------------------------------------------------------


def test_build_context_tier_a() -> None:
    x = build_context([], Tier.A)
    assert x[13] == pytest.approx(TIER_INDICATOR[Tier.A])


def test_build_context_tier_b() -> None:
    x = build_context([], Tier.B)
    assert x[13] == pytest.approx(TIER_INDICATOR[Tier.B])


def test_build_context_tier_c() -> None:
    x = build_context([], Tier.C)
    assert x[13] == pytest.approx(TIER_INDICATOR[Tier.C])


def test_build_context_tier_c_plus() -> None:
    x = build_context([], Tier.C_PLUS)
    assert x[13] == pytest.approx(TIER_INDICATOR[Tier.C_PLUS])


# ---------------------------------------------------------------------------
# ATTEST-WH (index 12)
# ---------------------------------------------------------------------------


def test_build_context_attest_none_defaults_to_allow() -> None:
    x = build_context([], Tier.A, attest_result=None)
    assert x[12] == pytest.approx(1.0)


def test_build_context_attest_allow() -> None:
    x = build_context([], Tier.A, attest_result=AttestResult.ALLOW)
    assert x[12] == pytest.approx(1.0)


def test_build_context_attest_degraded() -> None:
    x = build_context([], Tier.A, attest_result=AttestResult.DEGRADED)
    assert x[12] == pytest.approx(0.5)


def test_build_context_attest_block() -> None:
    x = build_context([], Tier.A, attest_result=AttestResult.BLOCK)
    assert x[12] == pytest.approx(0.0)


def test_build_context_attest_wh_verdict_score_ignored() -> None:
    # Even if ATTEST-WH appears in verdicts with score=0.9, index 12
    # must be driven by attest_result, not verdict.score.
    verdicts = [_verdict("ATTEST-WH", 0.9)]
    x = build_context(verdicts, Tier.A, attest_result=AttestResult.BLOCK)
    assert x[12] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Individual gate scores
# ---------------------------------------------------------------------------


def test_build_context_vestibule_lz_at_index_0() -> None:
    x = build_context([_verdict("VESTIBULE-LZ", 0.42)], Tier.A)
    assert x[0] == pytest.approx(0.42)


def test_build_context_vestibule_ps_at_index_1() -> None:
    x = build_context([_verdict("VESTIBULE-PS", 0.77)], Tier.A)
    assert x[1] == pytest.approx(0.77)


def test_build_context_probe_rm_at_index_2() -> None:
    x = build_context([_verdict("PROBE-RM", 0.55)], Tier.A)
    assert x[2] == pytest.approx(0.55)


def test_build_context_probe_mt_rho_dot_at_index_3() -> None:
    x = build_context([_verdict("PROBE-MT", 0.33)], Tier.A)
    assert x[3] == pytest.approx(0.33)


def test_build_context_probe_mt_rho_ddot_at_index_4_when_provided() -> None:
    x = build_context([_verdict("PROBE-MT", 0.33)], Tier.A, probe_mt_rho_ddot=0.11)
    assert x[4] == pytest.approx(0.11)


def test_build_context_probe_mt_rho_ddot_zero_when_not_provided() -> None:
    x = build_context([_verdict("PROBE-MT", 0.33)], Tier.A)
    assert x[4] == pytest.approx(0.0)


def test_build_context_probe_hd_at_index_5() -> None:
    x = build_context([_verdict("PROBE-HD", 0.6)], Tier.A)
    assert x[5] == pytest.approx(0.6)


def test_build_context_tripwire_h_at_index_6() -> None:
    x = build_context([_verdict("TRIPWIRE-H", 1.5)], Tier.A)
    assert x[6] == pytest.approx(1.5)


def test_build_context_tripwire_r_at_index_7() -> None:
    x = build_context([_verdict("TRIPWIRE-R", 2.3)], Tier.A)
    assert x[7] == pytest.approx(2.3)


def test_build_context_lookout_ct_at_index_8() -> None:
    x = build_context([_verdict("LOOKOUT-CT", 3.0)], Tier.A)
    assert x[8] == pytest.approx(3.0)


def test_build_context_lookout_jg_at_index_9() -> None:
    x = build_context([_verdict("LOOKOUT-JG", 0.9)], Tier.A)
    assert x[9] == pytest.approx(0.9)


def test_build_context_bprobe_logit_at_index_10() -> None:
    x = build_context([_verdict("B-PROBE-LOGIT", 0.85)], Tier.A)
    assert x[10] == pytest.approx(0.85)


def test_build_context_bprobe_consistency_at_index_11() -> None:
    x = build_context([_verdict("B-PROBE-CONSISTENCY", 0.12)], Tier.A)
    assert x[11] == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# Missing gates filled with 0.0
# ---------------------------------------------------------------------------


def test_build_context_missing_gates_are_zero() -> None:
    # Only VESTIBULE-LZ provided; all others (except 12=ALLOW, 13=tier) zero.
    x = build_context([_verdict("VESTIBULE-LZ", 0.5)], Tier.A)
    for i in range(1, 12):
        assert x[i] == pytest.approx(0.0), f"index {i} should be 0.0"


# ---------------------------------------------------------------------------
# Unknown gate names silently skipped
# ---------------------------------------------------------------------------


def test_build_context_unknown_gate_silently_skipped() -> None:
    x_with = build_context([_verdict("UNKNOWN-GATE", 99.0)], Tier.A)
    x_without = build_context([], Tier.A)
    np.testing.assert_array_equal(x_with, x_without)


# ---------------------------------------------------------------------------
# Multiple verdicts at once
# ---------------------------------------------------------------------------


def test_build_context_multiple_verdicts() -> None:
    verdicts = [
        _verdict("VESTIBULE-LZ", 0.1),
        _verdict("PROBE-MT", 0.2),
        _verdict("TRIPWIRE-H", 0.3),
    ]
    x = build_context(verdicts, Tier.B, probe_mt_rho_ddot=0.05)
    assert x[0] == pytest.approx(0.1)
    assert x[3] == pytest.approx(0.2)
    assert x[4] == pytest.approx(0.05)
    assert x[6] == pytest.approx(0.3)
    assert x[13] == pytest.approx(TIER_INDICATOR[Tier.B])
