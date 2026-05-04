"""FiveFoldOrchestrator — see blueprint §12 (five-fold evaluation plan).

Coordinates execution of all five evaluation folds, enforcing:
  - Fold A (calibration) must complete before Folds B–E.
  - Fold E (BCN-2 construction) uses only fp16_behavior from Fold A,
    NOT calibration thresholds or geometric scores (blueprint §12.2).

Fold roles (blueprint §12.2):
  A — Calibration: fits r̂, ĥ, τ_q thresholds, KenLM corpora.
  B — Cliff measurement: tests H1 (geometric + behavioral agreement).
  C — Defense composition: measures ABR/FPR per primitive and full stack.
  D — Bandit/online drift: tests CONDUCTOR under A8 + non-stationary drift.
  E — BCN-2 construction: builds below-cliff naturals using Fold A
      FP16 behavioral output and a cross-family paraphraser.

Phase A: all execute_fold_* methods raise NotImplementedError.
Phase B: replace stubs with real inference calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cliffguard.types import CalibrationTable, QuantScheme, Tier


@dataclass
class FoldAResults:
    """Artifacts produced by Fold A (calibration) — blueprint §12.2.

    Data flow discipline:
      - refusal_directions, calibration_tables, kenlm_paths → Folds B/C/D.
      - fp16_behavior, fold_a_hashes → Fold E ONLY (no thresholds, no geometry).
    """

    refusal_directions: dict[str, Any]               # "family:scheme" → r̂ unit vector
    calibration_tables: dict[str, CalibrationTable]  # primitive name → CalibrationTable
    kenlm_paths: dict[str, Path]                     # scheme_name → ARPA/binary path
    fp16_behavior: list[tuple[str, bool]]            # (prompt_text, fp16_refused)
    fold_a_hashes: frozenset[str]                    # SHA-256 of all Fold A prompts


@dataclass
class OrchestratorConfig:
    """Configuration for FiveFoldOrchestrator — blueprint §12."""

    data_dir: Path
    artifacts_dir: Path
    schemes: list[QuantScheme]
    tiers: list[Tier]
    fpr_target: float = 0.05
    n_calibration: int = 2000   # blueprint §12.2 minimum
    n_attack: int = 500         # prompts per adversary per scheme
    kenlm_order_tier_ab: int = 5  # blueprint §5.5: 5-gram for Tier A/B full deployment
    kenlm_order_tier_c: int = 3   # decisions_log C25: trigram for Tier C/C+ budget


class FiveFoldOrchestrator:
    """Coordinates the five evaluation folds per blueprint §12.

    Usage:
        orch = FiveFoldOrchestrator(config)
        fold_a = orch.execute_fold_a()   # Phase B: real calibration
        orch.fold_a_results = fold_a
        orch.execute_fold_b()            # Phase B: cliff measurement
        orch.execute_fold_c()            # Phase B: defense composition
        orch.execute_fold_d()            # Phase B: bandit drift
        orch.execute_fold_e()            # Phase B: BCN-2 construction

    Folds B–E raise RuntimeError if fold_a_results is None (Fold A
    must complete first). All folds raise NotImplementedError in Phase A.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.fold_a_results: FoldAResults | None = None

    def execute_fold_a(self) -> FoldAResults:
        """Run Fold A: calibration (blueprint §12.2, Week 1).

        Fits r̂_(M,q), ĥ_(M,q), τ_q thresholds, μ_0^(q) entropy baseline,
        and builds KenLM n-gram corpora for TRIPWIRE.
        Raises NotImplementedError in Phase A.
        """
        raise NotImplementedError(
            "Fold A requires Phase B wiring: real inference on calibration corpus."
        )

    def execute_fold_b(self) -> None:
        """Run Fold B: cliff measurement (blueprint §12.2, Week 2).

        Tests H1: Δ_cliff and Δ_B-cliff exhibit κ ≥ 0.25 jump at the same
        quantization boundary for ≥ 2 of 3 model families.
        Raises RuntimeError if Fold A has not been run.
        Raises NotImplementedError in Phase A.
        """
        if self.fold_a_results is None:
            raise RuntimeError(
                "Fold B requires Fold A results. Call execute_fold_a() first "
                "and assign the return value to self.fold_a_results."
            )
        raise NotImplementedError(
            "Fold B requires Phase B wiring: real cliff measurement inference."
        )

    def execute_fold_c(self) -> None:
        """Run Fold C: defense composition (blueprint §12.2, Week 3).

        Measures ABR/FPR per primitive and full stack across attack families.
        Requires Fold A calibration tables and thresholds.
        Raises RuntimeError if Fold A has not been run.
        Raises NotImplementedError in Phase A.
        """
        if self.fold_a_results is None:
            raise RuntimeError(
                "Fold C requires Fold A results. Call execute_fold_a() first "
                "and assign the return value to self.fold_a_results."
            )
        raise NotImplementedError(
            "Fold C requires Phase B wiring: real defense composition evaluation."
        )

    def execute_fold_d(self) -> None:
        """Run Fold D: bandit/online drift (blueprint §12.2, Week 4).

        Tests CONDUCTOR (LinUCB) under A8 + non-stationary attack drift.
        Requires Fold A calibration outputs for CONDUCTOR initialization.
        Raises RuntimeError if Fold A has not been run.
        Raises NotImplementedError in Phase A.
        """
        if self.fold_a_results is None:
            raise RuntimeError(
                "Fold D requires Fold A results. Call execute_fold_a() first "
                "and assign the return value to self.fold_a_results."
            )
        raise NotImplementedError(
            "Fold D requires Phase B wiring: real bandit drift simulation."
        )

    def execute_fold_e(self) -> None:
        """Run Fold E: BCN-2 non-circular dataset construction (blueprint §12.2).

        Uses ONLY fp16_behavior from Fold A (which prompts the FP16 model
        refused) — no calibration thresholds or geometric scores.
        The paraphraser must be a different model family than the test family.
        Raises RuntimeError if Fold A has not been run.
        Raises NotImplementedError in Phase A.
        """
        if self.fold_a_results is None:
            raise RuntimeError(
                "Fold E requires Fold A results (fp16_behavior only). "
                "Call execute_fold_a() first and assign the return value "
                "to self.fold_a_results."
            )
        raise NotImplementedError(
            "Fold E requires Phase B wiring: real paraphraser inference for BCN-2."
        )
