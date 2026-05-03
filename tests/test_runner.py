import pytest

from cliffguard.types import QuantScheme, Tier, ThreatModel
from cliffguard.eval.runner import EvaluationPlan, EvaluationRunner, FoldResult


def _make_result(
    asr: float = 0.2,
    fpr: float = 0.05,
    n_prompts: int = 100,
    n_blocked: int = 80,
) -> FoldResult:
    return FoldResult(
        fold_name="Fold B",
        tier=Tier.A,
        scheme=QuantScheme.FP16,
        n_prompts=n_prompts,
        n_blocked=n_blocked,
        n_passed=n_prompts - n_blocked,
        fpr=fpr,
        asr=asr,
    )


def _make_plan() -> EvaluationPlan:
    return EvaluationPlan(
        schemes=[QuantScheme.FP16, QuantScheme.GGUF_Q3_K_M],
        tiers=[Tier.A, Tier.B],
        adversaries=[ThreatModel.A1, ThreatModel.A3],
    )


# ---------------------------------------------------------------------------
# FoldResult properties
# ---------------------------------------------------------------------------


def test_fold_result_tpr_is_one_minus_asr() -> None:
    r = _make_result(asr=0.3)
    assert r.tpr == pytest.approx(0.7)


def test_fold_result_tpr_zero_asr() -> None:
    r = _make_result(asr=0.0)
    assert r.tpr == pytest.approx(1.0)


def test_fold_result_tpr_one_asr() -> None:
    r = _make_result(asr=1.0)
    assert r.tpr == pytest.approx(0.0)


def test_fold_result_abr_correct() -> None:
    r = _make_result(n_prompts=100, n_blocked=75)
    assert r.abr == pytest.approx(0.75)


def test_fold_result_abr_zero_when_no_prompts() -> None:
    r = _make_result(n_prompts=0, n_blocked=0)
    assert r.abr == pytest.approx(0.0)


def test_fold_result_abr_all_blocked() -> None:
    r = _make_result(n_prompts=50, n_blocked=50)
    assert r.abr == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EvaluationPlan fields and defaults
# ---------------------------------------------------------------------------


def test_evaluation_plan_stores_schemes() -> None:
    plan = _make_plan()
    assert QuantScheme.FP16 in plan.schemes
    assert QuantScheme.GGUF_Q3_K_M in plan.schemes


def test_evaluation_plan_stores_tiers() -> None:
    plan = _make_plan()
    assert Tier.A in plan.tiers
    assert Tier.B in plan.tiers


def test_evaluation_plan_stores_adversaries() -> None:
    plan = _make_plan()
    assert ThreatModel.A1 in plan.adversaries
    assert ThreatModel.A3 in plan.adversaries


def test_evaluation_plan_default_fpr_target() -> None:
    plan = _make_plan()
    assert plan.fpr_target == pytest.approx(0.05)


def test_evaluation_plan_default_n_calibration() -> None:
    plan = _make_plan()
    assert plan.n_calibration == 2000


def test_evaluation_plan_default_n_attack() -> None:
    plan = _make_plan()
    assert plan.n_attack == 500


# ---------------------------------------------------------------------------
# EvaluationRunner stub methods raise NotImplementedError
# ---------------------------------------------------------------------------


def test_runner_execute_fold_a_raises() -> None:
    runner = EvaluationRunner(_make_plan())
    with pytest.raises(NotImplementedError, match="Fold A"):
        runner.execute_fold_a()


def test_runner_execute_fold_b_raises() -> None:
    runner = EvaluationRunner(_make_plan())
    with pytest.raises(NotImplementedError, match="Fold B"):
        runner.execute_fold_b()


def test_runner_execute_fold_c_raises() -> None:
    runner = EvaluationRunner(_make_plan())
    with pytest.raises(NotImplementedError, match="Fold C"):
        runner.execute_fold_c()


# ---------------------------------------------------------------------------
# EvaluationRunner.summary
# ---------------------------------------------------------------------------


def test_runner_summary_empty_results_returns_zeros() -> None:
    runner = EvaluationRunner(_make_plan())
    s = runner.summary()
    assert s["mean_asr"] == pytest.approx(0.0)
    assert s["mean_fpr"] == pytest.approx(0.0)
    assert s["mean_abr"] == pytest.approx(0.0)
    assert s["n_results"] == pytest.approx(0.0)


def test_runner_summary_correct_means() -> None:
    runner = EvaluationRunner(_make_plan())
    runner.results.append(_make_result(asr=0.2, fpr=0.05, n_prompts=100, n_blocked=80))
    runner.results.append(_make_result(asr=0.4, fpr=0.10, n_prompts=200, n_blocked=100))
    s = runner.summary()
    assert s["mean_asr"] == pytest.approx(0.3)
    assert s["mean_fpr"] == pytest.approx(0.075)
    assert s["mean_abr"] == pytest.approx((0.8 + 0.5) / 2)
    assert s["n_results"] == pytest.approx(2.0)


def test_runner_summary_single_result() -> None:
    runner = EvaluationRunner(_make_plan())
    runner.results.append(_make_result(asr=0.15, fpr=0.03, n_prompts=40, n_blocked=30))
    s = runner.summary()
    assert s["mean_asr"] == pytest.approx(0.15)
    assert s["mean_fpr"] == pytest.approx(0.03)
    assert s["mean_abr"] == pytest.approx(0.75)
    assert s["n_results"] == pytest.approx(1.0)


def test_runner_summary_n_results_is_float() -> None:
    runner = EvaluationRunner(_make_plan())
    runner.results.append(_make_result())
    assert isinstance(runner.summary()["n_results"], float)
