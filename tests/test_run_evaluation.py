import pytest

from cliffguard.types import QuantScheme, Tier
from scripts.run_full_evaluation import build_config, main, parse_args


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


def test_parse_args_default_folds_all_five() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    assert set(args.folds) == {"A", "B", "C", "D", "E"}


def test_parse_args_custom_folds() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16", "--folds", "A", "B"])
    assert set(args.folds) == {"A", "B"}


def test_parse_args_default_fpr_target() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    assert args.fpr_target == pytest.approx(0.05)


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


def test_parse_args_artifacts_dir_default() -> None:
    from pathlib import Path
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    assert args.artifacts_dir == Path("artifacts/results/")


def test_parse_args_data_dir_default() -> None:
    from pathlib import Path
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    assert args.data_dir == Path("data/")


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------


def test_build_config_tier_is_enum() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    config = build_config(args)
    assert config.tiers == [Tier.A]


def test_build_config_scheme_is_enum() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16", "GGUF_Q3_K_M"])
    config = build_config(args)
    assert QuantScheme.FP16 in config.schemes
    assert QuantScheme.GGUF_Q3_K_M in config.schemes


def test_build_config_fpr_target_propagated() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16", "--fpr-target", "0.10"])
    config = build_config(args)
    assert config.fpr_target == pytest.approx(0.10)


def test_build_config_invalid_scheme_raises_value_error() -> None:
    args = parse_args(["--tier", "A", "--schemes", "FP16"])
    args.schemes = ["NOT_A_SCHEME"]
    with pytest.raises(ValueError, match="Unknown QuantScheme"):
        build_config(args)


def test_build_config_artifacts_dir_propagated() -> None:
    from pathlib import Path
    args = parse_args(["--tier", "A", "--schemes", "FP16", "--artifacts-dir", "out/"])
    config = build_config(args)
    assert config.artifacts_dir == Path("out/")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_dry_run_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--tier", "A", "--schemes", "FP16", "--dry-run"])
    assert code == 0


def test_main_dry_run_prints_config(capsys: pytest.CaptureFixture[str]) -> None:
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


def test_main_fold_b_skipped_when_fold_a_not_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--tier", "A", "--schemes", "FP16", "--folds", "B"])
    out = capsys.readouterr().out
    assert "prerequisite" in out or "Fold A" in out


def test_main_all_folds_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--tier", "A", "--schemes", "FP16"])
    assert code == 0
