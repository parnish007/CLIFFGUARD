import pytest
from pathlib import Path

from cliffguard.eval.five_fold_orchestrator import (
    FiveFoldOrchestrator,
    FoldAResults,
    OrchestratorConfig,
)
from cliffguard.types import CalibrationTable, QuantScheme, Tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    schemes: list[QuantScheme] | None = None,
    tiers: list[Tier] | None = None,
) -> OrchestratorConfig:
    return OrchestratorConfig(
        data_dir=data_dir or Path("data/"),
        artifacts_dir=artifacts_dir or Path("artifacts/"),
        schemes=schemes or [QuantScheme.FP16, QuantScheme.GGUF_Q3_K_M],
        tiers=tiers or [Tier.A],
    )


def _stub_fold_a() -> FoldAResults:
    return FoldAResults(
        refusal_directions={},
        calibration_tables={},
        kenlm_paths={},
        fp16_behavior=[("test prompt", True)],
        fold_a_hashes=frozenset(["abc123"]),
    )


# ---------------------------------------------------------------------------
# OrchestratorConfig — defaults and storage
# ---------------------------------------------------------------------------


def test_orchestrator_config_default_fpr_target() -> None:
    assert _config().fpr_target == 0.05


def test_orchestrator_config_default_n_calibration() -> None:
    assert _config().n_calibration == 2000


def test_orchestrator_config_default_n_attack() -> None:
    assert _config().n_attack == 500


def test_orchestrator_config_kenlm_order_tier_ab_default() -> None:
    assert _config().kenlm_order_tier_ab == 5


def test_orchestrator_config_kenlm_order_tier_c_default() -> None:
    assert _config().kenlm_order_tier_c == 3


def test_orchestrator_config_stores_data_dir() -> None:
    cfg = _config(data_dir=Path("my/data"))
    assert cfg.data_dir == Path("my/data")


def test_orchestrator_config_stores_artifacts_dir() -> None:
    cfg = _config(artifacts_dir=Path("out/"))
    assert cfg.artifacts_dir == Path("out/")


def test_orchestrator_config_stores_schemes() -> None:
    schemes = [QuantScheme.GGUF_Q4_K_M, QuantScheme.GGUF_Q3_K_M]
    assert _config(schemes=schemes).schemes == schemes


def test_orchestrator_config_stores_tiers() -> None:
    tiers = [Tier.B, Tier.C]
    assert _config(tiers=tiers).tiers == tiers


def test_orchestrator_config_custom_kenlm_order_tier_ab() -> None:
    cfg = OrchestratorConfig(
        data_dir=Path("d/"),
        artifacts_dir=Path("a/"),
        schemes=[QuantScheme.FP16],
        tiers=[Tier.A],
        kenlm_order_tier_ab=7,
    )
    assert cfg.kenlm_order_tier_ab == 7


def test_orchestrator_config_custom_kenlm_order_tier_c() -> None:
    cfg = OrchestratorConfig(
        data_dir=Path("d/"),
        artifacts_dir=Path("a/"),
        schemes=[QuantScheme.FP16],
        tiers=[Tier.C],
        kenlm_order_tier_c=4,
    )
    assert cfg.kenlm_order_tier_c == 4


# ---------------------------------------------------------------------------
# FiveFoldOrchestrator — initialisation
# ---------------------------------------------------------------------------


def test_orchestrator_init_stores_config() -> None:
    cfg = _config()
    orch = FiveFoldOrchestrator(cfg)
    assert orch.config is cfg


def test_orchestrator_fold_a_results_none_initially() -> None:
    assert FiveFoldOrchestrator(_config()).fold_a_results is None


def test_orchestrator_fold_a_results_can_be_assigned() -> None:
    orch = FiveFoldOrchestrator(_config())
    stub = _stub_fold_a()
    orch.fold_a_results = stub
    assert orch.fold_a_results is stub


# ---------------------------------------------------------------------------
# execute_fold_a
# ---------------------------------------------------------------------------


def test_execute_fold_a_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        FiveFoldOrchestrator(_config()).execute_fold_a()


# ---------------------------------------------------------------------------
# execute_fold_b
# ---------------------------------------------------------------------------


def test_execute_fold_b_raises_runtime_error_when_fold_a_none() -> None:
    with pytest.raises(RuntimeError):
        FiveFoldOrchestrator(_config()).execute_fold_b()


def test_execute_fold_b_runtime_error_mentions_fold_a() -> None:
    with pytest.raises(RuntimeError, match="[Ff]old [Aa]"):
        FiveFoldOrchestrator(_config()).execute_fold_b()


def test_execute_fold_b_raises_not_implemented_when_fold_a_set() -> None:
    orch = FiveFoldOrchestrator(_config())
    orch.fold_a_results = _stub_fold_a()
    with pytest.raises(NotImplementedError):
        orch.execute_fold_b()


# ---------------------------------------------------------------------------
# execute_fold_c
# ---------------------------------------------------------------------------


def test_execute_fold_c_raises_runtime_error_when_fold_a_none() -> None:
    with pytest.raises(RuntimeError):
        FiveFoldOrchestrator(_config()).execute_fold_c()


def test_execute_fold_c_runtime_error_mentions_fold_a() -> None:
    with pytest.raises(RuntimeError, match="[Ff]old [Aa]"):
        FiveFoldOrchestrator(_config()).execute_fold_c()


def test_execute_fold_c_raises_not_implemented_when_fold_a_set() -> None:
    orch = FiveFoldOrchestrator(_config())
    orch.fold_a_results = _stub_fold_a()
    with pytest.raises(NotImplementedError):
        orch.execute_fold_c()


# ---------------------------------------------------------------------------
# execute_fold_d
# ---------------------------------------------------------------------------


def test_execute_fold_d_raises_runtime_error_when_fold_a_none() -> None:
    with pytest.raises(RuntimeError):
        FiveFoldOrchestrator(_config()).execute_fold_d()


def test_execute_fold_d_runtime_error_mentions_fold_a() -> None:
    with pytest.raises(RuntimeError, match="[Ff]old [Aa]"):
        FiveFoldOrchestrator(_config()).execute_fold_d()


def test_execute_fold_d_raises_not_implemented_when_fold_a_set() -> None:
    orch = FiveFoldOrchestrator(_config())
    orch.fold_a_results = _stub_fold_a()
    with pytest.raises(NotImplementedError):
        orch.execute_fold_d()


# ---------------------------------------------------------------------------
# execute_fold_e
# ---------------------------------------------------------------------------


def test_execute_fold_e_raises_runtime_error_when_fold_a_none() -> None:
    with pytest.raises(RuntimeError):
        FiveFoldOrchestrator(_config()).execute_fold_e()


def test_execute_fold_e_runtime_error_mentions_fold_a() -> None:
    with pytest.raises(RuntimeError, match="[Ff]old [Aa]"):
        FiveFoldOrchestrator(_config()).execute_fold_e()


def test_execute_fold_e_raises_not_implemented_when_fold_a_set() -> None:
    orch = FiveFoldOrchestrator(_config())
    orch.fold_a_results = _stub_fold_a()
    with pytest.raises(NotImplementedError):
        orch.execute_fold_e()


# ---------------------------------------------------------------------------
# FoldAResults — storage
# ---------------------------------------------------------------------------


def test_fold_a_results_fp16_behavior_stored() -> None:
    r = FoldAResults(
        refusal_directions={},
        calibration_tables={},
        kenlm_paths={},
        fp16_behavior=[("prompt A", True), ("prompt B", False)],
        fold_a_hashes=frozenset(),
    )
    assert len(r.fp16_behavior) == 2
    assert r.fp16_behavior[0] == ("prompt A", True)
    assert r.fp16_behavior[1] == ("prompt B", False)


def test_fold_a_results_fold_a_hashes_stored() -> None:
    hashes: frozenset[str] = frozenset(["hash1", "hash2"])
    r = FoldAResults(
        refusal_directions={},
        calibration_tables={},
        kenlm_paths={},
        fp16_behavior=[],
        fold_a_hashes=hashes,
    )
    assert r.fold_a_hashes == hashes


def test_fold_a_results_calibration_tables_stored() -> None:
    table = CalibrationTable(
        primitive="probe-rm",
        thresholds={QuantScheme.FP16: 0.5},
        fpr_target=0.05,
    )
    r = FoldAResults(
        refusal_directions={},
        calibration_tables={"probe-rm": table},
        kenlm_paths={},
        fp16_behavior=[],
        fold_a_hashes=frozenset(),
    )
    assert "probe-rm" in r.calibration_tables


def test_fold_a_results_kenlm_paths_stored() -> None:
    r = FoldAResults(
        refusal_directions={},
        calibration_tables={},
        kenlm_paths={"fp16": Path("models/fp16.arpa")},
        fp16_behavior=[],
        fold_a_hashes=frozenset(),
    )
    assert r.kenlm_paths["fp16"] == Path("models/fp16.arpa")


def test_fold_a_results_refusal_directions_stored() -> None:
    r = FoldAResults(
        refusal_directions={"llama-3:FP16": [0.1, 0.2]},
        calibration_tables={},
        kenlm_paths={},
        fp16_behavior=[],
        fold_a_hashes=frozenset(),
    )
    assert "llama-3:FP16" in r.refusal_directions
