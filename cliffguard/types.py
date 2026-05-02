"""Foundational typed enums for CLIFFGUARD. See blueprint §2.2 (adversaries), §10 (tiers), §8 (quantization schemes). Margin, CalibrationTable, and GateVerdict are the runtime data types flowing between primitives and CONDUCTOR — see blueprint §6."""

from enum import Enum

from pydantic import BaseModel, field_validator

_THREAT_DESCRIPTIONS: dict[str, str] = {
    "A1": "Direct injector — DAN, persuasive jailbreaks, role-play",
    "A2": "Indirect injector / poisoned-weight attacker — RAG/tool injection or Egashira-style",
    "A3": "Optimizer — GCG, AutoDAN, AmpleGCG",
    "A4": "Iterator — PAIR, TAP, Crescendo",
    "A5": "Scaler — best-of-N, many-shot, randomized augmentation",
    "A6": "Encoder — ArtPrompt, low-resource language, bijection learning",
    "A7": "Quantization-cliff exploiter — natural-language prompts that flip at low bit-width",
    "A8": "Defender-aware adversary — Kerckhoffs assumption on bandit and calibration tables",
    "A9": "Closed-weight black-box endpoint adversary — top-k logprobs only",
}

_TIER_DESCRIPTIONS: dict[str, str] = {
    "A": "RTX 5060 8 GB — full stack, 7-9B NF4/AWQ-INT4",
    "B": "Pi 5 8 GB CPU — Q4_K_M 1.5B-3B, all primitives except LOOKOUT-JG",
    "C": "2 GB embedded — Q3_K_M ≤1.5B, narrow scope, no bandit",
    "C_PLUS": "2 GB embedded with PromptGuard-2-22M-INT4 — modest scope, static weights",
}

_CLIFF_CANDIDATES: frozenset[str] = frozenset(
    {"GGUF_Q3_K_M", "GGUF_IQ3_XXS", "GGUF_Q2_K", "GGUF_IQ2_XXS"}
)


class ThreatModel(Enum):
    """Nine-adversary threat model — blueprint §2.2."""

    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"
    A6 = "A6"
    A7 = "A7"
    A8 = "A8"
    A9 = "A9"

    def description(self) -> str:
        return _THREAT_DESCRIPTIONS[self.value]


class Tier(Enum):
    """Four hardware deployment tiers — blueprint §10."""

    A = "A"
    B = "B"
    C = "C"
    C_PLUS = "C_PLUS"

    def description(self) -> str:
        return _TIER_DESCRIPTIONS[self.value]


class QuantScheme(Enum):
    """Quantization schemes covered in the blueprint — blueprint §8."""

    FP16 = "FP16"
    INT8 = "INT8"
    NF4 = "NF4"
    AWQ_INT4 = "AWQ_INT4"
    GGUF_Q6_K = "GGUF_Q6_K"
    GGUF_Q5_K_M = "GGUF_Q5_K_M"
    GGUF_Q4_K_M = "GGUF_Q4_K_M"
    GGUF_Q3_K_M = "GGUF_Q3_K_M"
    GGUF_IQ3_XXS = "GGUF_IQ3_XXS"
    GGUF_Q2_K = "GGUF_Q2_K"
    GGUF_IQ2_XXS = "GGUF_IQ2_XXS"
    RKNN_W8A8 = "RKNN_W8A8"

    def is_cliff_candidate(self) -> bool:
        return self.value in _CLIFF_CANDIDATES

    @classmethod
    def from_string(cls, s: str) -> "QuantScheme":
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(f"Unknown QuantScheme: {s!r}") from None


class Margin(BaseModel):
    """A single refusal-margin observation from any PROBE variant — blueprint §5.1–5.3, §5.10."""

    value: float
    scheme: QuantScheme
    primitive: str
    layer: int | None = None

    @field_validator("primitive")
    @classmethod
    def _primitive_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("primitive must not be empty")
        return v

    @property
    def is_cliff_regime(self) -> bool:
        return self.scheme.is_cliff_candidate()


class CalibrationTable(BaseModel):
    """Per-quantization thresholds (tau_q) for one primitive — blueprint §5.1, §14."""

    primitive: str
    thresholds: dict[QuantScheme, float]
    fpr_target: float = 0.05

    def tau(self, scheme: QuantScheme) -> float:
        if scheme not in self.thresholds:
            raise KeyError(
                f"No threshold calibrated for scheme {scheme.value!r}"
                f" in primitive {self.primitive!r}"
            )
        return self.thresholds[scheme]

    def schemes_covered(self) -> list[QuantScheme]:
        return sorted(self.thresholds.keys(), key=lambda s: s.name)


class GateVerdict(BaseModel):
    """Output of one gate evaluation, forwarded to CONDUCTOR — blueprint §6, §9."""

    gate: str
    fired: bool
    score: float
    threshold: float
    tier: Tier
    threat_model: ThreatModel | None = None

    @property
    def margin_to_threshold(self) -> float:
        return self.threshold - self.score
