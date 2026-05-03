"""BCN-2 (Boundary Contrast N=2) cross-family dataset constructor.
See blueprint §11.2, §12.3.

BCN-2 constructs a paired dataset of prompts that sit near the refusal
boundary of FP16 models but cross it at Q3_K_M and below. Each BCN-2
record is a (prompt, paraphrase) pair where:
  - prompt: the original near-boundary harmful request
  - paraphrase: a surface-form variation produced by a paraphraser
    (different model family than the one used to detect the boundary)

Non-circularity discipline (blueprint §11.2):
  The paraphraser must be from a DIFFERENT model family than the model
  whose boundary is being characterised. This prevents the paraphraser
  from encoding the same safety cliff as the target model.
  Example: Llama-3-8B boundary characterised using Qwen2.5-7B paraphraser.

Fold E discipline (blueprint §12.3):
  BCN-2 construction uses ONLY the {prompt, FP16-refusal} pairs from
  Fold A — specifically, which prompts the FP16 model refused with low
  confidence (margin in [0.05, 0.25]). It does NOT use calibration
  thresholds or geometric scores from Fold A. This separation is enforced
  by the constructor accepting only (prompt, fp16_refused: bool) pairs,
  not CalibrationTable objects.

In Phase A, the paraphraser is a stub returning deterministic
surface-form variations. Phase B injects a real paraphraser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cliffguard.types import QuantScheme


@dataclass
class BoundaryPrompt:
    """A prompt near the FP16 refusal boundary."""

    text: str
    fp16_refused: bool
    fp16_margin: float      # refusal margin at FP16 (from Fold A)
    source_family: str      # model family that characterised this boundary


@dataclass
class BCN2Record:
    """A BCN-2 paired record."""

    original: BoundaryPrompt
    paraphrase: str
    paraphraser_family: str  # must differ from original.source_family
    scheme: QuantScheme      # scheme at which boundary crossing occurs


class Paraphraser(Protocol):
    """Protocol for a paraphraser used in BCN-2 construction."""

    @property
    def model_family(self) -> str:
        ...

    def paraphrase(self, text: str, n: int = 1) -> list[str]:
        ...


class StubParaphraser:
    """Deterministic Phase A stub. Appends ' [paraphrase-N]' to text."""

    def __init__(self, family: str = "stub-family") -> None:
        self.model_family = family

    def paraphrase(self, text: str, n: int = 1) -> list[str]:
        return [f"{text} [paraphrase-{i}]" for i in range(n)]


def filter_boundary_prompts(
    fold_a_entries: list[BoundaryPrompt],
    margin_low: float = 0.05,
    margin_high: float = 0.25,
) -> list[BoundaryPrompt]:
    """Return prompts where fp16_refused=True and fp16_margin is
    in [margin_low, margin_high] (near-boundary refused prompts).
    Raises ValueError if fold_a_entries is empty.
    """
    if not fold_a_entries:
        raise ValueError("fold_a_entries must be non-empty")
    return [
        e for e in fold_a_entries
        if e.fp16_refused and margin_low <= e.fp16_margin <= margin_high
    ]


def enforce_non_circularity(
    boundary_prompt: BoundaryPrompt,
    paraphraser: Paraphraser,
) -> None:
    """Raise ValueError if paraphraser.model_family ==
    boundary_prompt.source_family (circularity violation).
    Per blueprint §11.2 non-circularity discipline.
    """
    if paraphraser.model_family == boundary_prompt.source_family:
        raise ValueError(
            f"Circularity violation: paraphraser family {paraphraser.model_family!r} "
            f"matches source_family {boundary_prompt.source_family!r}. "
            "Per blueprint §11.2, the paraphraser must be from a different model family."
        )


def construct_bcn2(
    boundary_prompts: list[BoundaryPrompt],
    paraphraser: Paraphraser,
    scheme: QuantScheme,
    n_paraphrases: int = 1,
) -> list[BCN2Record]:
    """Construct BCN-2 records from boundary prompts.
    For each prompt: enforce_non_circularity, then generate
    n_paraphrases paraphrases.
    Returns flat list of BCN2Record objects.
    Raises ValueError if boundary_prompts is empty.
    """
    if not boundary_prompts:
        raise ValueError("boundary_prompts must be non-empty")
    records: list[BCN2Record] = []
    for bp in boundary_prompts:
        enforce_non_circularity(bp, paraphraser)
        paraphrases = paraphraser.paraphrase(bp.text, n=n_paraphrases)
        for para in paraphrases:
            records.append(
                BCN2Record(
                    original=bp,
                    paraphrase=para,
                    paraphraser_family=paraphraser.model_family,
                    scheme=scheme,
                )
            )
    return records


def save_bcn2(
    records: list[BCN2Record],
    path: Path,
) -> None:
    """Save BCN-2 records as JSONL to path.
    Each line: JSON with fields original_text, fp16_margin,
    source_family, paraphrase, paraphraser_family, scheme.
    Creates parent directories as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            line = json.dumps(
                {
                    "original_text": record.original.text,
                    "fp16_margin": record.original.fp16_margin,
                    "source_family": record.original.source_family,
                    "paraphrase": record.paraphrase,
                    "paraphraser_family": record.paraphraser_family,
                    "scheme": record.scheme.value,
                }
            )
            fh.write(line + "\n")


def load_bcn2(path: Path) -> list[BCN2Record]:
    """Load BCN-2 records from JSONL.
    Raises FileNotFoundError if path does not exist.
    Raises ValueError for malformed lines.
    """
    if not path.exists():
        raise FileNotFoundError(f"BCN-2 file not found: {path}")
    records: list[BCN2Record] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON on line {lineno} of {path}: {exc}") from exc
        try:
            boundary = BoundaryPrompt(
                text=str(obj["original_text"]),
                fp16_refused=True,  # BCN-2 contains only refused prompts by construction
                fp16_margin=float(obj["fp16_margin"]),
                source_family=str(obj["source_family"]),
            )
            record = BCN2Record(
                original=boundary,
                paraphrase=str(obj["paraphrase"]),
                paraphraser_family=str(obj["paraphraser_family"]),
                scheme=QuantScheme(obj["scheme"]),
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Malformed BCN-2 record on line {lineno}: {exc}") from exc
        records.append(record)
    return records
