"""Foundational typed enums for CLIFFGUARD. See blueprint §2.2 (adversaries), §10 (tiers), §8 (quantization schemes)."""

from enum import Enum

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
