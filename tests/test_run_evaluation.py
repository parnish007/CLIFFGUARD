import pytest

from cliffguard.types import QuantScheme, Tier
from scripts.run_full_evaluation import build_plan, main, parse_args


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_tier_a_and_scheme_fp16() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    assert args.tier == "A"
    assert args.schemes == ["FP16"]


def test_parse_args_multiple_schemes() -> None:
    args = parse_args(["--tier", "B", "--schemes", "FP16", "GGUF_Q3_K_M"])
    assert args.schemes == ["FP16", "GGUF_Q3_K_M"]


def test_parse_args_default_folds() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    assert set(args.folds) == {"A", "B", "C"}


def test_parse_args_custom_folds() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16", "--folds", "A", "B"])
    assert set(args.folds) == {"A", "B"}


def test_parse_args_default_fpr_target() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    assert args.fpr_target == pytest.approx(0.05)


def test_parse_args_default_n_calibration() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    assert args.n_calibration == 2000


def test_parse_args_default_n_attack() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    assert args.n_attack == 500


def test_parse_args_dry_run_flag() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16", "--dry-run"])
    assert args.dry_run is True


def test_parse_args_dry_run_default_false() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    assert args.dry_run is False


def test_parse_args_unknown_tier_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--tier", "Z", "--schemes", "FP16"])


def test_parse_args_c_plus_tier() -> None:
    args = parse_args(["--tier", "C_PLUS", "--schemes", "FP16"])
    assert args.tier == "C_PLUS"


# ---------------------------------------------------------------------------
# build_plan
# ---------------------------------------------------------------------------


def test_build_plan_tier_is_enum() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    plan = build_plan(args)
    assert plan.tiers == [Tier.A]


def test_build_plan_scheme_is_enum() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16", "GGUF_Q3_K_M"])
    plan = build_plan(args)
    assert QuantScheme.FP16 in plan.schemes
    assert QuantScheme.GGUF_Q3_K_M in plan.schemes


def test_build_plan_all_adversaries_present() -> None:
    from cliffguard.types import ThreatModel
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    plan = build_plan(args)
    assert set(plan.adversaries) == set(ThreatModel)


def test_build_plan_fpr_target_propagated() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16", "--fpr-target", "0.10"])
    plan = build_plan(args)
    assert plan.fpr_target == pytest.approx(0.10)


def test_build_plan_invalid_scheme_raises_value_error() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    args.schemes = ["NOT_A_SCHEME"]
    with pytest.raises(ValueError, match="Unknown QuantScheme"):
        build_plan(args)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_dry_run_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--tier", "A", "--schemes", "FP16", "--dry-run"])
    assert code == 0


def test_main_dry_run_prints_plan(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--tier", "B", "--schemes", "FP16", "GGUF_Q3_K_M", "--dry-run"])
    out = capsys.readouterr().out
    assert "tier" in out
    assert "FP16" in out


def test_main_without_dry_run_returns_0_despite_not_implemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["--tier", "A", "--schemes", "FP16", "--folds", "A"])
    assert code == 0


def test_main_phase_a_prints_phase_b_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--tier", "A", "--schemes", "FP16", "--folds", "A", "B", "C"])
    out = capsys.readouterr().out
    assert "Phase B" in out


def test_main_prints_summary(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--tier", "A", "--schemes", "FP16", "--folds", "A"])
    out = capsys.readouterr().out
    assert "Summary" in out
