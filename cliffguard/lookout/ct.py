"""LOOKOUT-CT per-session canary-token Bloom-filter gate — see blueprint §5.8.

Injects per-session canary tokens into the system prompt and monitors the
model's output for their reappearance. A model that leaks a canary token in
its output has been coerced into echoing injected content — the signature of
a successful prompt-injection attack (A1: direct override, A3: indirect
retrieval-augmented injection).

BloomFilter is used as a fast pre-filter before the exact substring match
to avoid O(n*m) scanning on every token.
"""

import hashlib

from cliffguard.types import CalibrationTable, GateVerdict, QuantScheme, Tier


class BloomFilter:
    """Fixed-size Bloom filter backed by a bytearray bit-array."""

    def __init__(self, m: int = 256, k: int = 3) -> None:
        self._m = m
        self._k = k
        self._bits = bytearray(m)

    def _bit_positions(self, item: str) -> list[int]:
        positions: list[int] = []
        for i in range(self._k):
            salted = f"{i}:{item}".encode()
            digest = hashlib.sha256(salted).hexdigest()
            positions.append(int(digest, 16) % self._m)
        return positions

    def add(self, item: str) -> None:
        for pos in self._bit_positions(item):
            self._bits[pos] = 1

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        return all(self._bits[pos] for pos in self._bit_positions(item))


def build_canary_filter(canary_tokens: list[str]) -> BloomFilter:
    """Build a BloomFilter pre-loaded with all canary tokens."""
    bf: BloomFilter = BloomFilter()
    for token in canary_tokens:
        bf.add(token)
    return bf


def check_output(
    output_text: str,
    canary_filter: BloomFilter,
    canary_tokens: list[str],
) -> list[str]:
    """Return the list of canary tokens found in output_text.

    Uses the Bloom filter as a pre-filter; exact substring match confirms.
    """
    matched: list[str] = []
    for token in canary_tokens:
        if token in canary_filter and token in output_text:
            matched.append(token)
    return matched


def evaluate(
    output_text: str,
    canary_tokens: list[str],
    calibration: CalibrationTable,
    scheme: QuantScheme,
    tier: Tier,
) -> GateVerdict:
    """Evaluate the LOOKOUT-CT gate.

    score = number of canary tokens found in output_text.
    fired = True if score > calibration.tau(scheme).
    gate name is "LOOKOUT-CT".
    threat_model is None.
    """
    canary_filter = build_canary_filter(canary_tokens)
    matched = check_output(output_text, canary_filter, canary_tokens)
    score = float(len(matched))
    threshold = calibration.tau(scheme)
    return GateVerdict(
        gate="LOOKOUT-CT",
        fired=score > threshold,
        score=score,
        threshold=threshold,
        tier=tier,
        threat_model=None,
    )
