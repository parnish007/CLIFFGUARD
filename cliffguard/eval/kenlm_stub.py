"""KenLM stub interface — see blueprint §5.5, §12.5.

KenLM (Heafield, 2011) provides fast n-gram language model scoring
used by TRIPWIRE-R to supply benign_logprobs. In Phase A this module
provides a stub that returns synthetic benign log-probability vectors.
Phase B (Task 30) wires the real kenlm Python binding.

The stub is deterministic given a prompt string (hash-seeded), making
tests reproducible without a real KenLM model file.
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt


class KenLMScorer(Protocol):
    """Protocol for a KenLM-backed per-token log-probability scorer."""

    def score_tokens(self, text: str, k: int = 20) -> npt.NDArray[np.float64]:
        """Return per-token log-probabilities for text.
        Shape: (num_tokens, k) — top-k logprobs per token.
        In scaffolding mode k is treated as the vocabulary slice size."""
        ...


class SyntheticKenLMScorer:
    """Deterministic synthetic scorer for Phase A scaffolding.
    Returns hash-seeded Gaussian log-prob vectors — not linguistically
    meaningful but reproducible and shaped correctly for testing."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def score_tokens(
        self,
        text: str,
        k: int = 20,
    ) -> npt.NDArray[np.float64]:
        """Return a (max(1, len(text.split())), k) array of
        synthetic log-probabilities in [-10.0, 0.0].
        Seed is combined with hash(text) for reproducibility."""
        tokens = text.split()
        num_tokens = max(1, len(tokens))
        # Combine instance seed with a stable hash of the text.
        combined_seed = (self.seed ^ (hash(text) & 0xFFFFFFFF)) & 0xFFFFFFFF
        rng = np.random.default_rng(combined_seed)
        # Raw Gaussian, then clip to [-10.0, 0.0].
        raw = rng.standard_normal((num_tokens, k)) * 3.0 - 5.0
        result: npt.NDArray[np.float64] = np.clip(raw, -10.0, 0.0)
        return result
