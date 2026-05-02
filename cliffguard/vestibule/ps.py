"""VESTIBULE-PS prompt-structure / spotlighting gate — see blueprint §5.7.

Detects structural injection patterns: separator tokens, role-override
phrases, and abnormal instruction-boundary markers embedded in user
input. Implements the spotlighting defence concept of Hines et al.
(arXiv:2403.14720): legitimate system-prompt delimiters are
distinguishable from injected ones by their position and encoding.

This gate operates on the raw input string before tokenisation.
"""

import re

from cliffguard.types import CalibrationTable, GateVerdict, QuantScheme, Tier

# ---------------------------------------------------------------------------
# Compiled patterns — built once at import time.
# ---------------------------------------------------------------------------

# Signal class 1: role-override phrases (case-insensitive).
ROLE_OVERRIDE_RE: re.Pattern[str] = re.compile(
    r"ignore previous instructions"
    r"|ignore all previous"
    r"|disregard"
    r"|you are now"
    r"|new persona"
    r"|act as"
    r"|pretend you are"
    r"|forget your instructions",
    re.IGNORECASE,
)

# Signal class 2: separator / delimiter injection — 4+ repetitions of the
# *same* punctuation character (backreference ensures identity, not just class).
SEPARATOR_RE: re.Pattern[str] = re.compile(
    r"([-=#+`*_.~|])\1{3,}"
)

# Signal class 3: prompt-boundary / chat-template markers.
BOUNDARY_RE: re.Pattern[str] = re.compile(
    r"\[/?INST\]"
    r"|<\|(?:im_start|im_end|system|user|assistant)\|>"
    r"|###\s+(?:Instruction|Response|System)"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_signals(text: str) -> dict[str, int]:
    """Return a dict with keys 'role_override', 'separator', 'boundary',
    each mapped to the match count in text."""
    return {
        "role_override": sum(1 for _ in ROLE_OVERRIDE_RE.finditer(text)),
        "separator": sum(1 for _ in SEPARATOR_RE.finditer(text)),
        "boundary": sum(1 for _ in BOUNDARY_RE.finditer(text)),
    }


def signal_score(text: str) -> float:
    """Aggregate signal score: weighted sum of match counts.

    Weights: role_override=1.0, separator=0.5, boundary=1.0.
    Score is not capped — callers compare to a calibrated threshold.
    """
    counts = count_signals(text)
    return (
        counts["role_override"] * 1.0
        + counts["separator"] * 0.5
        + counts["boundary"] * 1.0
    )


def evaluate(
    text: str,
    calibration: CalibrationTable,
    scheme: QuantScheme,
    tier: Tier,
) -> GateVerdict:
    """Evaluate the PS gate.

    fired = True if signal_score(text) > calibration.tau(scheme).
    gate name is "VESTIBULE-PS".
    score is signal_score(text).
    threshold is calibration.tau(scheme).
    threat_model is None.
    """
    score = signal_score(text)
    threshold = calibration.tau(scheme)
    return GateVerdict(
        gate="VESTIBULE-PS",
        fired=score > threshold,
        score=score,
        threshold=threshold,
        tier=tier,
        threat_model=None,
    )
