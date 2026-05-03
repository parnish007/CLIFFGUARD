"""Five-fold evaluation structure — see blueprint §12.2.

Implements the fold isolation discipline required by the pre-registration:

  Fold A (calibration): 2000 benign prompts per scheme. Used to fit the
    refusal direction r̂ and per-quantization thresholds tau_q for each
    primitive. The calibration outputs (thresholds, direction vectors)
    are the *only* artefacts from Fold A that may flow into Folds B-C.

  Fold B (cliff measurement): 500 adversarial prompts per adversary per
    scheme. White-box evaluation (residual stream access). Tests H1, H2,
    H4.

  Fold C (defense composition): same corpus as Fold B, logprobs-only
    mode (black-box). Tests H3, H5.

  Fold D (bandit drift): synthetic prompts for bandit drift simulation.
    Tests whether CONDUCTOR weight resets (ADWIN trigger) preserve FPR
    portability. Unblinded only after Folds B-C complete.

  Fold E (BCN-2 construction): uses Fold A's behavioral output — i.e.,
    which prompts the FP16 model refused — to build the BCN-2 (Boundary
    Contrast N=2) dataset used in the cliff measurement. CRITICAL: Fold E
    uses ONLY the {prompt, FP16-refusal} pairs from Fold A, NOT the
    calibration thresholds or geometric scores derived from Fold A. This
    separation is enforced by code: the Fold E loader receives only
    (prompt, refused_by_fp16: bool) tuples.

Fold isolation discipline (blueprint §12.2):
  No prompt may appear in more than one fold. fold_isolation_check()
  verifies this via SHA-256 hashing before any unblinding step.
  Violation raises AssertionError — a hard stop, not a warning.

Data layout:
  data/folds/fold_a/anthropic_hh_benign.jsonl
  data/folds/fold_a/anthropic_hh_refused.jsonl
  data/folds/fold_a/oasst_benign.jsonl
  data/folds/fold_b/  (adversarial corpus, gitignored until Phase B)
  data/folds/fold_c/  (same corpus as B, different evaluation mode)
  data/folds/fold_d/  (synthetic bandit drift prompts)
  data/folds/fold_e/  (BCN-2 pairs derived from Fold A behavioral output)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

_FOLD_A_DIR = Path("data/folds/fold_a")
_FOLD_B_DIR = Path("data/folds/fold_b")
_FOLD_C_DIR = Path("data/folds/fold_c")
_FOLD_D_DIR = Path("data/folds/fold_d")

_FOLD_A_FILES = [
    "anthropic_hh_benign.jsonl",
    "anthropic_hh_refused.jsonl",
    "oasst_benign.jsonl",
]


class Fold(Enum):
    A = "calibration"
    B = "cliff_measurement"
    C = "defense_composition"
    D = "bandit_drift"
    E = "bcn_2_construction"


@dataclass
class FoldEntry:
    """One prompt record in the evaluation corpus."""

    prompt: str
    label: Literal["benign", "refused", "harmful_test"]
    source: str  # "anthropic-hh", "oasst", "advbench", etc.
    fold: Fold
    sha256: str  # SHA-256 of the prompt text, for hash gating


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_jsonl_fold(
    directory: Path,
    filenames: list[str],
    fold: Fold,
    label: Literal["benign", "refused", "harmful_test"],
) -> list[FoldEntry]:
    entries: list[FoldEntry] = []
    for fname in filenames:
        path = directory / fname
        if not path.exists():
            continue
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {lineno} of {path}: {exc}"
                ) from exc
            prompt = str(obj["prompt"])
            source = str(obj.get("source", fname.split(".")[0]))
            # infer label from filename convention when not explicit
            rec_label: Literal["benign", "refused", "harmful_test"] = label
            if "label" in obj:
                rec_label = obj["label"]
            entries.append(
                FoldEntry(
                    prompt=prompt,
                    label=rec_label,
                    source=source,
                    fold=fold,
                    sha256=_sha256(prompt),
                )
            )
    return entries


def load_fold_a_calibration(
    fold_a_dir: Path | None = None,
) -> list[FoldEntry]:
    """Load Fold A from data/folds/fold_a/. The directory contains
    JSONL files: anthropic_hh_benign.jsonl, anthropic_hh_refused.jsonl,
    oasst_benign.jsonl. Each line is a {"prompt": ..., "source": ...}
    object. The function adds Fold.A and computes per-prompt SHA-256.

    If data/folds/fold_a/ does not exist, raise FileNotFoundError with
    a pointer to scripts/download_fold_a.py.

    Per blueprint §12.2: BCN-2 (Fold E) construction uses Fold A's
    *behavioral output only* (whether FP16 model refuses), not Fold A's
    geometric calibrations. This separation is enforced by code: the
    Fold E loader takes only the {prompt, FP16-refusal} pairs, not
    the calibration thresholds derived from Fold A."""
    directory = fold_a_dir if fold_a_dir is not None else _FOLD_A_DIR
    if not directory.exists():
        raise FileNotFoundError(
            f"Fold A directory not found: {directory}\n"
            "Run scripts/download_fold_a.py to obtain the corpus."
        )
    benign = _load_jsonl_fold(directory, ["anthropic_hh_benign.jsonl", "oasst_benign.jsonl"], Fold.A, "benign")
    refused = _load_jsonl_fold(directory, ["anthropic_hh_refused.jsonl"], Fold.A, "refused")
    return benign + refused


def load_fold_b_cliff_measurement(
    fold_b_dir: Path | None = None,
) -> list[FoldEntry]:
    """Load Fold B (white-box adversarial evaluation) from data/folds/fold_b/.
    Raises FileNotFoundError if directory is missing."""
    directory = fold_b_dir if fold_b_dir is not None else _FOLD_B_DIR
    if not directory.exists():
        raise FileNotFoundError(
            f"Fold B directory not found: {directory}\n"
            "Fold B corpus is unblinded in Phase B only."
        )
    return _load_jsonl_fold(directory, list(p.name for p in directory.glob("*.jsonl")), Fold.B, "harmful_test")


def load_fold_c_defense_composition(
    fold_c_dir: Path | None = None,
) -> list[FoldEntry]:
    """Load Fold C (black-box evaluation) from data/folds/fold_c/.
    Same adversarial corpus as Fold B, different observability mode.
    Raises FileNotFoundError if directory is missing."""
    directory = fold_c_dir if fold_c_dir is not None else _FOLD_C_DIR
    if not directory.exists():
        raise FileNotFoundError(
            f"Fold C directory not found: {directory}\n"
            "Fold C corpus is unblinded in Phase B only."
        )
    return _load_jsonl_fold(directory, list(p.name for p in directory.glob("*.jsonl")), Fold.C, "harmful_test")


def load_fold_d_bandit_drift_synthetic(
    fold_d_dir: Path | None = None,
) -> list[FoldEntry]:
    """Load Fold D (synthetic bandit drift prompts) from data/folds/fold_d/.
    Raises FileNotFoundError if directory is missing."""
    directory = fold_d_dir if fold_d_dir is not None else _FOLD_D_DIR
    if not directory.exists():
        raise FileNotFoundError(
            f"Fold D directory not found: {directory}\n"
            "Fold D is a synthetic corpus generated by eval/drift_sim.py."
        )
    return _load_jsonl_fold(directory, list(p.name for p in directory.glob("*.jsonl")), Fold.D, "benign")


def fold_isolation_check(
    fold_entries: dict[Fold, list[FoldEntry]] | None = None,
) -> dict[Fold, set[str]]:
    """Return a dict mapping each loaded fold to the set of prompt
    SHA-256 hashes it contains. The intersection of any two fold sets
    must be empty. Raises AssertionError otherwise. Run this before
    any unblinding step."""
    if fold_entries is None:
        fold_entries = {}
        for fold, loader in [
            (Fold.A, load_fold_a_calibration),
            (Fold.B, load_fold_b_cliff_measurement),
            (Fold.C, load_fold_c_defense_composition),
            (Fold.D, load_fold_d_bandit_drift_synthetic),
        ]:
            try:
                entries = loader()
                fold_entries[fold] = entries
            except FileNotFoundError:
                pass

    hash_sets: dict[Fold, set[str]] = {
        fold: {e.sha256 for e in entries}
        for fold, entries in fold_entries.items()
    }

    folds = list(hash_sets.keys())
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            fa, fb = folds[i], folds[j]
            overlap = hash_sets[fa] & hash_sets[fb]
            assert not overlap, (
                f"Fold isolation violation: {len(overlap)} prompt(s) appear in "
                f"both {fa.name} and {fb.name}. "
                "This must be resolved before unblinding."
            )

    return hash_sets
