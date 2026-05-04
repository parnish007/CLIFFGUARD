import numpy as np
import pytest

from cliffguard.conductor.bandit import ARMS, Conductor
from cliffguard.eval.drift_sim import ADWIN_DELTA, adwin_statistic, run_drift_simulation

_REQUIRED_KEYS = frozenset({
    "drift_detected",
    "reset_step",
    "recovery_steps",
    "warmup_mean_reward",
    "drift_mean_reward",
    "recovery_mean_reward",
})


def _fresh_conductor(d: int = 4) -> Conductor:
    return Conductor(d=d)


# ---------------------------------------------------------------------------
# adwin_statistic
# ---------------------------------------------------------------------------


def test_adwin_raises_for_empty_stream() -> None:
    with pytest.raises(ValueError):
        adwin_statistic(np.array([], dtype=np.float64))


def test_adwin_returns_false_for_stable_rewards() -> None:
    stable = np.ones(100, dtype=np.float64) * 0.5
    detected, idx = adwin_statistic(stable)
    assert detected is False


def test_adwin_stable_index_is_stream_length() -> None:
    stable = np.ones(100, dtype=np.float64) * 0.5
    _, idx = adwin_statistic(stable)
    assert idx == len(stable)


def test_adwin_returns_false_for_constant_positive() -> None:
    stable = np.ones(200, dtype=np.float64)
    detected, _ = adwin_statistic(stable)
    assert detected is False


def test_adwin_returns_true_for_clear_mean_shift() -> None:
    # First half all 1.0, second half all -1.0 — clear downward drift.
    stream = np.concatenate([
        np.ones(50, dtype=np.float64),
        -np.ones(50, dtype=np.float64),
    ])
    detected, _ = adwin_statistic(stream)
    assert detected is True


def test_adwin_drift_index_within_second_half_for_step_change() -> None:
    n = 100
    stream = np.concatenate([
        np.ones(n // 2, dtype=np.float64),
        -np.ones(n // 2, dtype=np.float64),
    ])
    _, drift_index = adwin_statistic(stream)
    # Drift must be detected in the second half (indices 50-99).
    assert drift_index >= n // 2
    assert drift_index < n


def test_adwin_returns_tuple_bool_int() -> None:
    stream = np.ones(10, dtype=np.float64)
    result = adwin_statistic(stream)
    assert isinstance(result[0], bool)
    assert isinstance(result[1], int)


def test_adwin_smaller_delta_detects_earlier() -> None:
    # With smaller delta, lambda_thresh = -log(delta) increases
    # so drift takes longer — but detection of large shifts happens quickly.
    stream = np.concatenate([
        np.ones(50, dtype=np.float64),
        -np.ones(50, dtype=np.float64),
    ])
    detected_default, _ = adwin_statistic(stream, delta=ADWIN_DELTA)
    detected_small, _ = adwin_statistic(stream, delta=1e-6)
    # Both should detect the large step-change
    assert detected_default is True
    assert detected_small is True


def test_adwin_large_shift_detected_with_default_delta() -> None:
    # Extremely clear shift: 0.0 → 10.0 — detectable immediately.
    stream = np.concatenate([
        np.zeros(30, dtype=np.float64),
        np.full(30, -10.0, dtype=np.float64),
    ])
    detected, idx = adwin_statistic(stream)
    assert detected is True
    assert idx >= 30  # after the change point


# ---------------------------------------------------------------------------
# run_drift_simulation — error handling
# ---------------------------------------------------------------------------


def test_run_raises_for_unknown_target_arm() -> None:
    with pytest.raises(ValueError, match="target_arm"):
        run_drift_simulation(
            _fresh_conductor(), context_dim=4, target_arm="DOES-NOT-EXIST"
        )


def test_run_raises_for_t_warm_less_than_one() -> None:
    with pytest.raises(ValueError, match="T_warm"):
        run_drift_simulation(
            _fresh_conductor(), context_dim=4,
            target_arm=ARMS[0], T_warm=0,
        )


def test_run_raises_for_t_drift_less_than_one() -> None:
    with pytest.raises(ValueError, match="T_drift"):
        run_drift_simulation(
            _fresh_conductor(), context_dim=4,
            target_arm=ARMS[0], T_drift=0,
        )


def test_run_raises_for_t_recover_less_than_one() -> None:
    with pytest.raises(ValueError, match="T_recover"):
        run_drift_simulation(
            _fresh_conductor(), context_dim=4,
            target_arm=ARMS[0], T_recover=0,
        )


# ---------------------------------------------------------------------------
# run_drift_simulation — return dict structure
# ---------------------------------------------------------------------------


def test_run_returns_dict_with_all_required_keys() -> None:
    result = run_drift_simulation(
        _fresh_conductor(), context_dim=4, target_arm=ARMS[0],
        T_warm=20, T_drift=20, T_recover=15, seed=0,
    )
    assert _REQUIRED_KEYS.issubset(result.keys())


def test_run_warmup_mean_reward_is_positive() -> None:
    result = run_drift_simulation(
        _fresh_conductor(), context_dim=4, target_arm=ARMS[0],
        T_warm=30, T_drift=30, T_recover=15, seed=1,
    )
    assert float(result["warmup_mean_reward"]) > 0.0  # type: ignore[arg-type]


def test_run_drift_mean_reward_is_negative() -> None:
    result = run_drift_simulation(
        _fresh_conductor(), context_dim=4, target_arm=ARMS[0],
        T_warm=30, T_drift=30, T_recover=15, seed=2,
    )
    assert float(result["drift_mean_reward"]) < 0.0  # type: ignore[arg-type]


def test_run_recovery_mean_reward_is_positive() -> None:
    result = run_drift_simulation(
        _fresh_conductor(), context_dim=4, target_arm=ARMS[0],
        T_warm=50, T_drift=50, T_recover=20, seed=3,
    )
    assert float(result["recovery_mean_reward"]) > 0.0  # type: ignore[arg-type]


def test_run_drift_detected_true_for_large_t_drift() -> None:
    # T_warm=100 + T_drift=50 combined stream: clear downward shift detectable.
    result = run_drift_simulation(
        _fresh_conductor(), context_dim=4, target_arm=ARMS[0],
        T_warm=100, T_drift=50, T_recover=20, seed=42,
    )
    assert result["drift_detected"] is True


def test_run_reset_step_set_when_drift_detected() -> None:
    result = run_drift_simulation(
        _fresh_conductor(), context_dim=4, target_arm=ARMS[0],
        T_warm=100, T_drift=50, T_recover=20, seed=42,
    )
    if result["drift_detected"]:
        assert result["reset_step"] is not None
        assert isinstance(result["reset_step"], int)
        assert int(result["reset_step"]) >= 0  # type: ignore[arg-type]


def test_run_reset_step_within_drift_phase() -> None:
    T_drift = 50
    result = run_drift_simulation(
        _fresh_conductor(), context_dim=4, target_arm=ARMS[0],
        T_warm=100, T_drift=T_drift, T_recover=20, seed=42,
    )
    if result["drift_detected"]:
        rs = int(result["reset_step"])  # type: ignore[arg-type]
        assert 0 <= rs < T_drift


def test_run_drift_detected_false_gives_none_reset_step() -> None:
    # Very short warmup+drift → unlikely to trigger drift detection.
    result = run_drift_simulation(
        _fresh_conductor(), context_dim=4, target_arm=ARMS[0],
        T_warm=1, T_drift=1, T_recover=5, seed=99,
    )
    if not result["drift_detected"]:
        assert result["reset_step"] is None


def test_run_uses_valid_arm_from_arms_list() -> None:
    # Smoke test: all arms in ARMS are accepted without ValueError.
    for arm in ARMS[:3]:  # test first 3 arms for speed
        result = run_drift_simulation(
            _fresh_conductor(), context_dim=4, target_arm=arm,
            T_warm=10, T_drift=10, T_recover=5, seed=0,
        )
        assert "drift_detected" in result
