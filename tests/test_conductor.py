import numpy as np
import pytest

from cliffguard.types import GateVerdict, QuantScheme, Tier
from cliffguard.conductor.bandit import (
    ARMS,
    MIN_WEIGHT,
    NEVER_DISABLE,
    Conductor,
    LinUCBArm,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_D = 4
_CTX = np.ones(_D, dtype=np.float64)
_CTX_UNIT = _CTX / np.linalg.norm(_CTX)


def _make_verdict(gate: str, fired: bool) -> GateVerdict:
    return GateVerdict(
        gate=gate,
        fired=fired,
        score=1.0 if fired else 0.0,
        threshold=0.5,
        tier=Tier.A,
        threat_model=None,
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_arms_has_twelve_members() -> None:
    assert len(ARMS) == 12


def test_never_disable_is_subset_of_arms() -> None:
    assert NEVER_DISABLE <= set(ARMS)


def test_arms_contains_expected_primitives() -> None:
    expected = {
        "VESTIBULE-LZ", "VESTIBULE-PS",
        "PROBE-RM", "PROBE-MT", "PROBE-HD",
        "TRIPWIRE-H", "TRIPWIRE-R",
        "LOOKOUT-CT", "LOOKOUT-JG",
        "B-PROBE-LOGIT", "B-PROBE-CONSISTENCY",
        "ATTEST-WH",
    }
    assert set(ARMS) == expected


# ---------------------------------------------------------------------------
# LinUCBArm
# ---------------------------------------------------------------------------


def test_linucb_arm_initialises_A_as_identity() -> None:
    arm = LinUCBArm(_D)
    np.testing.assert_array_equal(arm.A, np.eye(_D))


def test_linucb_arm_initialises_b_as_zeros() -> None:
    arm = LinUCBArm(_D)
    np.testing.assert_array_equal(arm.b, np.zeros(_D))


def test_linucb_arm_ucb_score_returns_float() -> None:
    arm = LinUCBArm(_D)
    score = arm.ucb_score(_CTX_UNIT)
    assert isinstance(score, float)


def test_linucb_arm_ucb_score_non_negative_for_unit_context() -> None:
    # With A=I, b=0, score = 0 + alpha * sqrt(x^T x) >= 0.
    arm = LinUCBArm(_D, alpha=1.0)
    assert arm.ucb_score(_CTX_UNIT) >= 0.0


def test_linucb_arm_update_changes_A() -> None:
    arm = LinUCBArm(_D)
    A_before = arm.A.copy()
    arm.update(_CTX_UNIT, reward=1.0)
    assert not np.allclose(arm.A, A_before)


def test_linucb_arm_update_changes_b() -> None:
    arm = LinUCBArm(_D)
    arm.update(_CTX_UNIT, reward=1.0)
    assert not np.allclose(arm.b, np.zeros(_D))


def test_linucb_arm_update_A_increases_by_outer_product() -> None:
    arm = LinUCBArm(_D)
    arm.update(_CTX_UNIT, reward=0.5)
    expected_A = np.eye(_D) + np.outer(_CTX_UNIT, _CTX_UNIT)
    np.testing.assert_allclose(arm.A, expected_A)


def test_linucb_arm_update_b_by_reward_times_x() -> None:
    arm = LinUCBArm(_D)
    arm.update(_CTX_UNIT, reward=2.0)
    np.testing.assert_allclose(arm.b, 2.0 * _CTX_UNIT)


# ---------------------------------------------------------------------------
# Conductor
# ---------------------------------------------------------------------------


def test_conductor_select_weights_returns_all_12_arms() -> None:
    c = Conductor(_D)
    w = c.select_weights(_CTX_UNIT)
    assert set(w.keys()) == set(ARMS)
    assert len(w) == 12


def test_conductor_select_weights_tripwire_r_ge_min_weight() -> None:
    c = Conductor(_D)
    # Apply a large negative reward to try to push TRIPWIRE-R weight below MIN_WEIGHT.
    for _ in range(20):
        c.update("TRIPWIRE-R", _CTX_UNIT, reward=-10.0)
    w = c.select_weights(_CTX_UNIT)
    assert w["TRIPWIRE-R"] >= MIN_WEIGHT


def test_conductor_select_weights_attest_wh_ge_min_weight() -> None:
    c = Conductor(_D)
    for _ in range(20):
        c.update("ATTEST-WH", _CTX_UNIT, reward=-10.0)
    w = c.select_weights(_CTX_UNIT)
    assert w["ATTEST-WH"] >= MIN_WEIGHT


def test_conductor_update_raises_for_unknown_arm() -> None:
    c = Conductor(_D)
    with pytest.raises(ValueError, match="Unknown arm"):
        c.update("NONEXISTENT-GATE", _CTX_UNIT, reward=1.0)


def test_conductor_update_changes_arm_state() -> None:
    c = Conductor(_D)
    b_before = c._arms["VESTIBULE-LZ"].b.copy()
    c.update("VESTIBULE-LZ", _CTX_UNIT, reward=1.0)
    assert not np.allclose(c._arms["VESTIBULE-LZ"].b, b_before)


def test_conductor_aggregate_verdict_true_when_all_fired() -> None:
    c = Conductor(_D)
    w = c.select_weights(_CTX_UNIT)
    verdicts = [_make_verdict(arm, fired=True) for arm in ARMS]
    assert c.aggregate_verdict(verdicts, w) is True


def test_conductor_aggregate_verdict_false_when_none_fired() -> None:
    c = Conductor(_D)
    w = c.select_weights(_CTX_UNIT)
    verdicts = [_make_verdict(arm, fired=False) for arm in ARMS]
    assert c.aggregate_verdict(verdicts, w) is False


def test_conductor_aggregate_verdict_empty_list_returns_false() -> None:
    c = Conductor(_D)
    w = c.select_weights(_CTX_UNIT)
    assert c.aggregate_verdict([], w) is False


def test_conductor_aggregate_verdict_majority_fires() -> None:
    c = Conductor(_D)
    # Equal weights (fresh conductor) — 7 fired out of 12 > 50%.
    w = {arm: 1.0 for arm in ARMS}
    fired = [_make_verdict(arm, fired=True) for arm in ARMS[:7]]
    not_fired = [_make_verdict(arm, fired=False) for arm in ARMS[7:]]
    assert c.aggregate_verdict(fired + not_fired, w) is True


def test_conductor_aggregate_verdict_minority_does_not_fire() -> None:
    c = Conductor(_D)
    w = {arm: 1.0 for arm in ARMS}
    fired = [_make_verdict(arm, fired=True) for arm in ARMS[:5]]
    not_fired = [_make_verdict(arm, fired=False) for arm in ARMS[5:]]
    assert c.aggregate_verdict(fired + not_fired, w) is False


def test_conductor_reset_weights_restores_A_to_identity() -> None:
    c = Conductor(_D)
    # Dirty all arms first.
    for arm in ARMS:
        c.update(arm, _CTX_UNIT, reward=1.0)
    c.reset_weights()
    for arm in ARMS:
        np.testing.assert_array_equal(c._arms[arm].A, np.eye(_D))


def test_conductor_reset_weights_restores_b_to_zeros() -> None:
    c = Conductor(_D)
    for arm in ARMS:
        c.update(arm, _CTX_UNIT, reward=1.0)
    c.reset_weights()
    for arm in ARMS:
        np.testing.assert_array_equal(c._arms[arm].b, np.zeros(_D))
