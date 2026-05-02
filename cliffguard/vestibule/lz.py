"""VESTIBULE-LZ compression-ratio gate — see blueprint §5.6.

Computes the LZ-based compression ratio of the input string as a proxy
for lexical entropy. High-entropy (high-compression-ratio) inputs are
characteristic of adversarial suffixes (A3: GCG, AutoDAN). Low-entropy
inputs compressed to near-zero size are characteristic of repetitive
injection patterns.

Per blueprint §5.6: LZ4/zlib ratio, ~20 µs / 1 KB cost.
"""

import zlib

from cliffguard.types import CalibrationTable, GateVerdict, QuantScheme, Tier


def compression_ratio(text: str, encoding: str = "utf-8") -> float:
    """Return len(compressed) / len(raw_bytes).

    Uses zlib.compress at default level.
    Returns 1.0 if raw_bytes is empty (degenerate case).
    """
    raw = text.encode(encoding)
    if not raw:
        return 1.0
    return len(zlib.compress(raw)) / len(raw)


def evaluate(
    text: str,
    calibration: CalibrationTable,
    scheme: QuantScheme,
    tier: Tier,
    encoding: str = "utf-8",
) -> GateVerdict:
    """Evaluate the LZ compression gate.

    fired = True if compression_ratio(text) > calibration.tau(scheme).
    gate name is "VESTIBULE-LZ".
    score is the compression ratio.
    threshold is calibration.tau(scheme).
    threat_model is None (this gate is not adversary-specific).
    """
    ratio = compression_ratio(text, encoding)
    threshold = calibration.tau(scheme)
    return GateVerdict(
        gate="VESTIBULE-LZ",
        fired=ratio > threshold,
        score=ratio,
        threshold=threshold,
        tier=tier,
        threat_model=None,
    )
