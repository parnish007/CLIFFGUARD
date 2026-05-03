import numpy as np
import pytest

from cliffguard.attest.wh import AttestResult
from cliffguard.conductor.bandit import Conductor
from cliffguard.conductor.context import CONTEXT_DIM
from cliffguard.types import GateVerdict, QuantScheme, Tier
from cliffguard.eval.orchestrator import run_request_cycle, update_conductor

_D = CONTEXT_DIM  # 14


def _make_conductor() -> Conductor:
    return Conductor(d=_D)


def _verdict(gate: str, fired: bool, score: float = 0.8) -> GateVerdict:
    return GateVerdict(gate=gate, score=score, fired=fired, threshold=0.5, tier=Tier.A)


# ---------------------------------------------------------------------------
# Return type and shape
# ---------------------------------------------------------------------------


def test_run_request_cycle_returns_tuple() -> None:
    block, weights, ctx = run_request_cycle([], Tier.A, QuantScheme.FP16, _make_conductor())
    assert isinstance(block, bool)
    assert isinstance(weights, dict)
    assert isinstance(ctx, np.ndarray)


def test_run_request_cycle_context_shape() -> None:
    _, _, ctx = run_request_cycle([], Tier.A, QuantScheme.FP16, _make_conductor())
    assert ctx.shape == (_D,)


# ---------------------------------------------------------------------------
# active_weights filtered to tier gates
# ---------------------------------------------------------------------------


def test_run_request_cycle_tier_c_active_weights_keys() -> None:
    _, weights, _ = run_request_cycle([], Tier.C, QuantScheme.FP16, _make_conductor())
    assert set(weights.keys()) == {"VESTIBULE-LZ", "VESTIBULE-PS", "ATTEST-WH"}


def test_run_request_cycle_tier_a_active_weights_has_all_gates() -> None:
    _, weights, _ = run_request_cycle([], Tier.A, QuantScheme.FP16, _make_conductor())
    # Tier A has 12 gates
    assert len(weights) == 12


def test_run_request_cycle_tier_c_plus_active_weights_keys() -> None:
    _, weights, _ = run_request_cycle([], Tier.C_PLUS, QuantScheme.FP16, _make_conductor())
    assert set(weights.keys()) == {
        "VESTIBULE-LZ", "VESTIBULE-PS", "B-PROBE-LOGIT", "ATTEST-WH"
    }


# ---------------------------------------------------------------------------
# block_decision correctness
# ---------------------------------------------------------------------------


def test_run_request_cycle_blocks_when_all_fired() -> None:
    verdicts = [
        _verdict("VESTIBULE-LZ", fired=True),
        _verdict("VESTIBULE-PS", fired=True),
        _verdict("ATTEST-WH", fired=True),
    ]
    block, _, _ = run_request_cycle(verdicts, Tier.C, QuantScheme.FP16, _make_conductor())
    assert block is True


def test_run_request_cycle_passes_when_none_fired() -> None:
    verdicts = [
        _verdict("VESTIBULE-LZ", fired=False),
        _verdict("VESTIBULE-PS", fired=False),
        _verdict("ATTEST-WH", fired=False),
    ]
    block, _, _ = run_request_cycle(verdicts, Tier.C, QuantScheme.FP16, _make_conductor())
    assert block is False


def test_run_request_cycle_passes_when_no_verdicts() -> None:
    block, _, _ = run_request_cycle([], Tier.C, QuantScheme.FP16, _make_conductor())
    assert block is False


# ---------------------------------------------------------------------------
# black_box mode excludes white-box-only gates
# ---------------------------------------------------------------------------


def test_run_request_cycle_black_box_excludes_white_box_gates() -> None:
    _, weights, _ = run_request_cycle(
        [], Tier.A, QuantScheme.FP16, _make_conductor(), black_box=True
    )
    for gate in ("PROBE-RM", "PROBE-MT", "PROBE-HD"):
        assert gate not in weights


def test_run_request_cycle_black_box_retains_never_disable() -> None:
    _, weights, _ = run_request_cycle(
        [], Tier.A, QuantScheme.FP16, _make_conductor(), black_box=True
    )
    assert "TRIPWIRE-R" in weights
    assert "ATTEST-WH" in weights


# ---------------------------------------------------------------------------
# attest_result propagates to context index 12
# ---------------------------------------------------------------------------


def test_run_request_cycle_attest_block_sets_context_index_12() -> None:
    _, _, ctx = run_request_cycle(
        [], Tier.A, QuantScheme.FP16, _make_conductor(),
        attest_result=AttestResult.BLOCK,
    )
    assert ctx[12] == pytest.approx(0.0)


def test_run_request_cycle_attest_degraded_sets_context_index_12() -> None:
    _, _, ctx = run_request_cycle(
        [], Tier.A, QuantScheme.FP16, _make_conductor(),
        attest_result=AttestResult.DEGRADED,
    )
    assert ctx[12] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# update_conductor
# ---------------------------------------------------------------------------


def test_update_conductor_no_error() -> None:
    conductor = _make_conductor()
    ctx = np.zeros(_D, dtype=np.float64)
    update_conductor(conductor, "VESTIBULE-LZ", ctx, reward=1.0)


def test_update_conductor_raises_for_unknown_arm() -> None:
    conductor = _make_conductor()
    ctx = np.zeros(_D, dtype=np.float64)
    with pytest.raises(ValueError, match="Unknown arm"):
        update_conductor(conductor, "NOT-A-GATE", ctx, reward=1.0)
