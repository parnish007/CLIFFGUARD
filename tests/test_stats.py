import pytest
import numpy as np

from cliffguard.eval.stats import (
    ALPHA_CORRECTED,
    ALPHA_UNCORRECTED,
    BONFERRONI_N,
    ks_test_fpr_decoupling,
    realized_power,
    required_n,
    test_h1_cliff_existence as h1_cliff_existence,
    test_h4_composition_gain as h4_composition_gain,
    test_h5_tier_c_weakness as h5_tier_c_weakness,
    wilcoxon_test,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_alpha_corrected_equals_0_01() -> None:
    assert ALPHA_CORRECTED == pytest.approx(0.01)


def test_alpha_corrected_is_uncorrected_divided_by_bonferroni_n() -> None:
    assert ALPHA_CORRECTED == pytest.approx(ALPHA_UNCORRECTED / BONFERRONI_N)


def test_bonferroni_n_is_five() -> None:
    assert BONFERRONI_N == 5


# ---------------------------------------------------------------------------
# required_n
# ---------------------------------------------------------------------------


def test_required_n_raises_for_effect_size_zero() -> None:
    with pytest.raises(ValueError, match="effect_size"):
        required_n(0.0)


def test_required_n_raises_for_negative_effect_size() -> None:
    with pytest.raises(ValueError, match="effect_size"):
        required_n(-0.5)


def test_required_n_raises_for_power_equals_one() -> None:
    with pytest.raises(ValueError, match="power"):
        required_n(0.5, power=1.0)


def test_required_n_raises_for_power_above_one() -> None:
    with pytest.raises(ValueError, match="power"):
        required_n(0.5, power=1.1)


def test_required_n_raises_for_power_zero() -> None:
    with pytest.raises(ValueError, match="power"):
        required_n(0.5, power=0.0)


def test_required_n_raises_for_alpha_zero() -> None:
    with pytest.raises(ValueError, match="alpha"):
        required_n(0.5, alpha=0.0)


def test_required_n_returns_int() -> None:
    result = required_n(0.5)
    assert isinstance(result, int)


def test_required_n_returns_at_least_two() -> None:
    # Very large effect size might give n < 2 without the floor
    assert required_n(100.0) >= 2


def test_required_n_valid_returns_positive_integer() -> None:
    n = required_n(0.5, alpha=0.01, power=0.80)
    assert n >= 2


def test_required_n_larger_for_smaller_effect_size() -> None:
    n_small = required_n(0.2)
    n_large = required_n(0.8)
    assert n_small > n_large


def test_required_n_larger_for_higher_power() -> None:
    n_high = required_n(0.5, power=0.90)
    n_low = required_n(0.5, power=0.70)
    assert n_high > n_low


def test_required_n_plausible_value() -> None:
    # effect_size=0.5, alpha=0.01, power=0.80 → expect n in roughly [40, 80]
    n = required_n(0.5, alpha=0.01, power=0.80)
    assert 20 <= n <= 200


# ---------------------------------------------------------------------------
# realized_power
# ---------------------------------------------------------------------------


def test_realized_power_raises_for_n_less_than_2() -> None:
    with pytest.raises(ValueError, match="n"):
        realized_power(1, effect_size=0.5)


def test_realized_power_raises_for_n_zero() -> None:
    with pytest.raises(ValueError, match="n"):
        realized_power(0, effect_size=0.5)


def test_realized_power_raises_for_effect_size_zero() -> None:
    with pytest.raises(ValueError, match="effect_size"):
        realized_power(50, effect_size=0.0)


def test_realized_power_raises_for_negative_effect_size() -> None:
    with pytest.raises(ValueError, match="effect_size"):
        realized_power(50, effect_size=-0.3)


def test_realized_power_returns_float() -> None:
    result = realized_power(50, effect_size=0.5)
    assert isinstance(result, float)


def test_realized_power_in_open_unit_interval() -> None:
    result = realized_power(50, effect_size=0.5)
    assert 0.0 < result < 1.0


def test_realized_power_increases_with_n() -> None:
    p_small = realized_power(10, effect_size=0.5)
    p_large = realized_power(200, effect_size=0.5)
    assert p_large > p_small


def test_realized_power_increases_with_effect_size() -> None:
    p_small = realized_power(50, effect_size=0.2)
    p_large = realized_power(50, effect_size=0.8)
    assert p_large > p_small


def test_realized_power_consistency_with_required_n() -> None:
    # n returned by required_n should give realized power >= target power (approx)
    n = required_n(0.5, alpha=0.01, power=0.80)
    p = realized_power(n, effect_size=0.5, alpha=0.01)
    assert p >= 0.75  # allow some approximation slack


# ---------------------------------------------------------------------------
# wilcoxon_test
# ---------------------------------------------------------------------------


def test_wilcoxon_raises_for_different_length_arrays() -> None:
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="length"):
        wilcoxon_test(x, y)


def test_wilcoxon_raises_for_length_one() -> None:
    x = np.array([1.0], dtype=np.float64)
    y = np.array([2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="length"):
        wilcoxon_test(x, y)


def test_wilcoxon_raises_for_empty_arrays() -> None:
    x = np.array([], dtype=np.float64)
    y = np.array([], dtype=np.float64)
    with pytest.raises(ValueError):
        wilcoxon_test(x, y)


def test_wilcoxon_returns_tuple_of_two_floats() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    y = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=np.float64)
    result = wilcoxon_test(x, y)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


def test_wilcoxon_p_value_close_to_one_for_identical_arrays() -> None:
    x = np.ones(20, dtype=np.float64) * 0.5
    y = np.ones(20, dtype=np.float64) * 0.5
    _, p_value = wilcoxon_test(x, y)
    assert p_value == pytest.approx(1.0)


def test_wilcoxon_small_p_value_for_clearly_different_arrays() -> None:
    # x is all zeros, y is all ones — large consistent shift
    x = np.zeros(50, dtype=np.float64)
    y = np.ones(50, dtype=np.float64)
    _, p_value = wilcoxon_test(x, y)
    assert p_value < 0.001


def test_wilcoxon_p_value_in_unit_interval() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal(30).astype(np.float64)
    y = rng.standard_normal(30).astype(np.float64)
    _, p_value = wilcoxon_test(x, y)
    assert 0.0 <= p_value <= 1.0


def test_wilcoxon_statistic_nonnegative() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    stat, _ = wilcoxon_test(x, y)
    assert stat >= 0.0


# ---------------------------------------------------------------------------
# ks_test_fpr_decoupling
# ---------------------------------------------------------------------------


def test_ks_raises_for_missing_reference_scheme() -> None:
    fpr = {"NF4": np.array([0.04, 0.05, 0.06], dtype=np.float64)}
    with pytest.raises(ValueError, match="FP16"):
        ks_test_fpr_decoupling(fpr, reference_scheme="FP16")


def test_ks_raises_for_custom_missing_reference() -> None:
    fpr = {"FP16": np.array([0.05, 0.05], dtype=np.float64)}
    with pytest.raises(ValueError, match="MISSING"):
        ks_test_fpr_decoupling(fpr, reference_scheme="MISSING")


def test_ks_returns_dict_with_correct_keys() -> None:
    fpr = {
        "FP16": np.array([0.04, 0.05, 0.06, 0.05], dtype=np.float64),
        "NF4": np.array([0.04, 0.05, 0.07, 0.05], dtype=np.float64),
        "GGUF_Q3_K_M": np.array([0.03, 0.06, 0.08, 0.04], dtype=np.float64),
    }
    result = ks_test_fpr_decoupling(fpr, reference_scheme="FP16")
    assert set(result.keys()) == {"NF4", "GGUF_Q3_K_M"}


def test_ks_reference_scheme_not_in_result() -> None:
    fpr = {
        "FP16": np.array([0.05, 0.05, 0.05], dtype=np.float64),
        "NF4": np.array([0.05, 0.05, 0.05], dtype=np.float64),
    }
    result = ks_test_fpr_decoupling(fpr)
    assert "FP16" not in result


def test_ks_values_are_stat_p_tuples() -> None:
    fpr = {
        "FP16": np.array([0.04, 0.05, 0.06], dtype=np.float64),
        "NF4": np.array([0.04, 0.05, 0.06], dtype=np.float64),
    }
    result = ks_test_fpr_decoupling(fpr)
    ks_stat, p_val = result["NF4"]
    assert isinstance(ks_stat, float)
    assert isinstance(p_val, float)
    assert 0.0 <= ks_stat <= 1.0
    assert 0.0 <= p_val <= 1.0


def test_ks_identical_distributions_high_p_value() -> None:
    arr = np.linspace(0.01, 0.09, 50, dtype=np.float64)
    fpr = {"FP16": arr.copy(), "NF4": arr.copy()}
    result = ks_test_fpr_decoupling(fpr)
    _, p_val = result["NF4"]
    assert p_val > 0.05


# ---------------------------------------------------------------------------
# h1_cliff_existence
# ---------------------------------------------------------------------------


def test_h1_returns_true_when_two_families_detected() -> None:
    families = {"llama-3": True, "qwen-2.5": True, "mistral": False}
    accepted, summary = h1_cliff_existence(families)
    assert accepted is True
    assert isinstance(summary, str)


def test_h1_returns_true_when_all_three_detected() -> None:
    families = {"llama-3": True, "qwen-2.5": True, "mistral": True}
    accepted, _ = h1_cliff_existence(families)
    assert accepted is True


def test_h1_returns_false_when_only_one_detected() -> None:
    families = {"llama-3": True, "qwen-2.5": False, "mistral": False}
    accepted, _ = h1_cliff_existence(families)
    assert accepted is False


def test_h1_returns_false_when_none_detected() -> None:
    families = {"llama-3": False, "qwen-2.5": False}
    accepted, _ = h1_cliff_existence(families)
    assert accepted is False


def test_h1_summary_contains_accepted_or_rejected() -> None:
    _, summary = h1_cliff_existence({"a": True, "b": True})
    assert "Accepted" in summary or "Rejected" in summary


def test_h1_custom_min_families() -> None:
    families = {"a": True, "b": False, "c": False}
    accepted_1, _ = h1_cliff_existence(families, min_families=1)
    accepted_2, _ = h1_cliff_existence(families, min_families=2)
    assert accepted_1 is True
    assert accepted_2 is False


# ---------------------------------------------------------------------------
# h4_composition_gain
# ---------------------------------------------------------------------------


def test_h4_returns_three_tuple() -> None:
    full = np.ones(20, dtype=np.float64) * 0.8
    single = np.ones(20, dtype=np.float64) * 0.4
    result = h4_composition_gain(full, single)
    assert len(result) == 3


def test_h4_accepted_true_for_clearly_better_full_stack() -> None:
    # full stack blocks 90% of attacks; best single blocks 40%
    rng = np.random.default_rng(7)
    full = np.clip(rng.normal(0.90, 0.02, 60).astype(np.float64), 0.0, 1.0)
    single = np.clip(rng.normal(0.40, 0.02, 60).astype(np.float64), 0.0, 1.0)
    _, p_value, accepted = h4_composition_gain(full, single)
    assert accepted is True
    assert p_value < ALPHA_CORRECTED


def test_h4_accepted_false_for_identical_scores() -> None:
    arr = np.ones(20, dtype=np.float64) * 0.5
    _, _, accepted = h4_composition_gain(arr, arr.copy())
    assert accepted is False


def test_h4_accepted_false_when_full_stack_worse() -> None:
    full = np.ones(50, dtype=np.float64) * 0.3
    single = np.ones(50, dtype=np.float64) * 0.9
    _, _, accepted = h4_composition_gain(full, single)
    assert accepted is False


def test_h4_p_value_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    full = rng.uniform(0.5, 1.0, 30).astype(np.float64)
    single = rng.uniform(0.0, 0.5, 30).astype(np.float64)
    _, p_value, _ = h4_composition_gain(full, single)
    assert 0.0 <= p_value <= 1.0


# ---------------------------------------------------------------------------
# h5_tier_c_weakness
# ---------------------------------------------------------------------------


def test_h5_returns_all_required_keys() -> None:
    arr = np.ones(20, dtype=np.float64) * 0.1
    result = h5_tier_c_weakness(arr, arr + 0.5, arr)
    required_keys = {
        "tier_c_stat", "tier_c_p", "tier_c_accepted",
        "tier_c_plus_stat", "tier_c_plus_p", "tier_c_plus_accepted",
    }
    assert required_keys.issubset(result.keys())


def test_h5_tier_c_accepted_true_for_identical_tier_c_and_baseline() -> None:
    baseline = np.ones(30, dtype=np.float64) * 0.1
    tier_c = np.ones(30, dtype=np.float64) * 0.1   # identical to baseline
    tier_c_plus = np.ones(30, dtype=np.float64) * 0.8  # clearly better
    result = h5_tier_c_weakness(tier_c, tier_c_plus, baseline)
    assert result["tier_c_accepted"] is True
    assert float(result["tier_c_p"]) >= ALPHA_CORRECTED  # type: ignore[arg-type]


def test_h5_tier_c_plus_accepted_true_for_clearly_better() -> None:
    rng = np.random.default_rng(13)
    baseline = np.clip(rng.normal(0.10, 0.01, 60).astype(np.float64), 0.0, 1.0)
    tier_c = np.clip(rng.normal(0.10, 0.01, 60).astype(np.float64), 0.0, 1.0)
    tier_c_plus = np.clip(rng.normal(0.85, 0.01, 60).astype(np.float64), 0.0, 1.0)
    result = h5_tier_c_weakness(tier_c, tier_c_plus, baseline)
    assert result["tier_c_plus_accepted"] is True


def test_h5_p_values_in_unit_interval() -> None:
    arr = np.linspace(0.05, 0.15, 25, dtype=np.float64)
    result = h5_tier_c_weakness(arr, arr + 0.3, arr)
    assert 0.0 <= float(result["tier_c_p"]) <= 1.0  # type: ignore[arg-type]
    assert 0.0 <= float(result["tier_c_plus_p"]) <= 1.0  # type: ignore[arg-type]
