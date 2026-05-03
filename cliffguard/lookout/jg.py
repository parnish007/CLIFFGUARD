"""LOOKOUT-JG output judge gate — see blueprint §5.9.

Applies SmoothLLM-style paraphrase consistency checking to the model's
output: the output is paraphrased N times (N ∈ {2, 3} per blueprint)
and the compliance rate across paraphrases is measured. A high
compliance rate on paraphrases of a borderline response is evidence
that the model has genuinely complied with a harmful instruction rather
than producing a superficially compliant-looking but safe response.

In scaffolding mode (Phase A), the paraphraser is a stub that returns
pre-specified outputs. The real paraphraser integration is Phase B.
Per blueprint §5.9: N ∈ {2, 3}; Tier A only (too slow for Tier B/C).
"""

from typing import Protocol

from cliffguard.types import CalibrationTable, GateVerdict, QuantScheme, Tier


class Paraphraser(Protocol):
    """Stub interface for a paraphraser.
    Phase B will provide a real implementation backed by a small LLM."""

    def paraphrase(self, text: str, n: int) -> list[str]:
        """Return n paraphrases of text."""
        ...


class JudgeStack(Protocol):
    """Stub interface for a compliance judge.
    Phase B will provide StrongREJECT + Llama-Guard-3-8B."""

    def is_compliant(self, response: str) -> bool:
        """Return True if response is compliant with a harmful
        instruction (i.e. the model did NOT refuse)."""
        ...


def compliance_rate(
    response: str,
    paraphraser: Paraphraser,
    judge: JudgeStack,
    n: int = 3,
) -> float:
    """Paraphrase response n times, judge each paraphrase, return
    the fraction judged compliant. Range [0.0, 1.0].
    Raises ValueError if n < 1."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    paraphrases = paraphraser.paraphrase(response, n)
    compliant = sum(1 for p in paraphrases if judge.is_compliant(p))
    return compliant / n


def evaluate(
    response: str,
    paraphraser: Paraphraser,
    judge: JudgeStack,
    calibration: CalibrationTable,
    scheme: QuantScheme,
    tier: Tier,
    n: int = 3,
) -> GateVerdict:
    """Evaluate the JG gate.
    score = compliance_rate(response, paraphraser, judge, n).
    fired = True if score > calibration.tau(scheme).
    gate name is "LOOKOUT-JG".
    threat_model is None.
    Raises ValueError if n < 1."""
    score = compliance_rate(response, paraphraser, judge, n)
    threshold = calibration.tau(scheme)
    return GateVerdict(
        gate="LOOKOUT-JG",
        fired=score > threshold,
        score=score,
        threshold=threshold,
        tier=tier,
        threat_model=None,
    )
