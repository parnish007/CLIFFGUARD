"""Five-fold evaluation runner — see blueprint §12.

Sequences the pre-registered evaluation protocol:
  Fold A: Calibration — fit refusal direction, fit tau_q per primitive.
  Fold B: White-box evaluation — run all primitives with residual
          stream access on Tier A and Tier B hardware.
  Fold C: Black-box evaluation — run B-PROBE-* on Tier A/B and all
          active gates on Tier C/C+.
  (Folds D/E are held-out test folds, not unblinded until submission.)

In Phase A this module provides the runner skeleton with stub execute
methods. Phase B (Task 32) wires real engine adapters and corpus
loaders. No model inference occurs in Phase A.

Per blueprint §12.1: the five hypotheses are evaluated in this order:
  H1 → Fold A (cliff existence, geometric + behavioral metrics)
  H2 → Fold B (FPR decoupling, white-box)
  H3 → Fold C (FPR decoupling, black-box)
  H4 → Fold B+C (composition gain)
  H5 → Fold B+C Tier C vs C+ (structural weakness)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cliffguard.types import QuantScheme, Tier, ThreatModel


@dataclass
class FoldResult:
    """Results from one evaluation fold."""

    fold_name: str
    tier: Tier
    scheme: QuantScheme
    n_prompts: int
    n_blocked: int
    n_passed: int
    fpr: float
    asr: float  # attack success rate = 1 - TPR at matched FPR
    notes: list[str] = field(default_factory=list)

    @property
    def tpr(self) -> float:
        """True positive rate = 1 - ASR."""
        return 1.0 - self.asr

    @property
    def abr(self) -> float:
        """Adversarial block rate = n_blocked / n_prompts."""
        if self.n_prompts == 0:
            return 0.0
        return self.n_blocked / self.n_prompts


@dataclass
class EvaluationPlan:
    """Declares the schemes, tiers, and adversaries for a run."""

    schemes: list[QuantScheme]
    tiers: list[Tier]
    adversaries: list[ThreatModel]
    fpr_target: float = 0.05
    n_calibration: int = 2000  # blueprint §12.2 minimum
    n_attack: int = 500  # prompts per adversary per scheme


class EvaluationRunner:
    """Skeleton evaluation runner. All execute_* methods are stubs
    that raise NotImplementedError in Phase A."""

    def __init__(self, plan: EvaluationPlan) -> None:
        self.plan = plan
        self.results: list[FoldResult] = []

    def execute_fold_a(self) -> list[FoldResult]:
        """Run Fold A calibration.
        Stub: raises NotImplementedError with message
        'Fold A requires real corpus and engine adapters (Phase B).'"""
        raise NotImplementedError(
            "Fold A requires real corpus and engine adapters (Phase B)."
        )

    def execute_fold_b(self) -> list[FoldResult]:
        """Run Fold B white-box evaluation.
        Stub: raises NotImplementedError."""
        raise NotImplementedError(
            "Fold B requires real corpus and engine adapters (Phase B)."
        )

    def execute_fold_c(self) -> list[FoldResult]:
        """Run Fold C black-box evaluation.
        Stub: raises NotImplementedError."""
        raise NotImplementedError(
            "Fold C requires real corpus and engine adapters (Phase B)."
        )

    def summary(self) -> dict[str, float]:
        """Return summary statistics over self.results.
        Keys: mean_asr, mean_fpr, mean_abr, n_results.
        Returns zeros if self.results is empty."""
        if not self.results:
            return {
                "mean_asr": 0.0,
                "mean_fpr": 0.0,
                "mean_abr": 0.0,
                "n_results": 0.0,
            }
        return {
            "mean_asr": sum(r.asr for r in self.results) / len(self.results),
            "mean_fpr": sum(r.fpr for r in self.results) / len(self.results),
            "mean_abr": sum(r.abr for r in self.results) / len(self.results),
            "n_results": float(len(self.results)),
        }
