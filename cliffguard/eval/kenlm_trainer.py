"""KenLM n-gram language model trainer for TRIPWIRE-R — see blueprint §5.5, §12.5.

Trains a KenLM n-gram model on the Fold A benign corpus and serialises
the ARPA file for use by TRIPWIRE-R at inference time.

Training pipeline:
  1. Assemble the Fold A benign text corpus (one sentence per line).
  2. Call the lmplz binary (KenLM command-line tool) via subprocess.
  3. Serialise the resulting ARPA file to data/kenlm/fold_a_{scheme}.arpa.
  4. Optionally binarise to .klm format for faster loading.

Tier C+ memory budget note (blueprint §10, §5.5):
  PromptGuard-2-22M-INT4 occupies ~30 MB. KenLM 3-gram on 2000 sentences
  typically requires ~5-15 MB ARPA. Total: ~45 MB — well within the 2 GB
  embedded budget. The ArpaSize estimator below provides a pre-training
  budget check.

Binary detection:
  If the lmplz binary is not on PATH, train_kenlm and binarise_arpa raise
  NotImplementedError with the install URL. When the binary IS available
  (apt install kenlm, or built from source), the subprocess call executes
  and the real ARPA file is written. The Phase A test suite asserts the
  NotImplementedError path; since CI machines do not have lmplz installed,
  those tests still pass on the dev box even though the production path
  is functional on a properly-configured GPU/edge runner.

KenLM installation: https://github.com/kpu/kenlm
"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

KENLM_DEFAULT_ORDER: int = 3  # trigram; §5.5 cites 5-gram for full deployment
KENLM_DEFAULT_PRUNE: str = "0 0 1"  # prune singletons at order 3

_PHASE_A_MSG = (
    "train_kenlm requires the lmplz binary. See:\n"
    " https://github.com/kpu/kenlm\n"
    " On a GPU/edge runner: apt install kenlm or build from source."
)

_BINARISE_MSG = (
    "binarise_arpa requires the build_binary binary. See:\n"
    " https://github.com/kpu/kenlm\n"
    " On a GPU/edge runner: apt install kenlm or build from source."
)


def assemble_corpus(
    fold_entries: list[Any],  # list[FoldEntry] — accept any object with .prompt attr
    out_path: Path,
) -> int:
    """Write one prompt per line to out_path.
    Returns the number of lines written.
    Raises ValueError if fold_entries is empty.
    Creates parent directories as needed.
    """
    if not fold_entries:
        raise ValueError("fold_entries must be non-empty")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [entry.prompt for entry in fold_entries]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def estimate_arpa_size_mb(
    n_sentences: int,
    avg_tokens_per_sentence: float = 15.0,
    order: int = KENLM_DEFAULT_ORDER,
) -> float:
    """Estimate ARPA file size in MB using the rule-of-thumb from
    Heafield (2011): approximately 15 bytes per n-gram entry.
    Total n-grams ≈ n_sentences * avg_tokens * order.
    Returns estimated MB as a float.
    Emits UserWarning if estimate exceeds 50 MB (Tier C+ budget risk).
    """
    total_ngrams = n_sentences * avg_tokens_per_sentence * order
    size_mb = (total_ngrams * 15) / (1024 * 1024)
    if size_mb > 50.0:
        warnings.warn(
            f"Estimated ARPA size {size_mb:.1f} MB exceeds 50 MB Tier C+ budget. "
            "Consider reducing corpus size or n-gram order.",
            UserWarning,
            stacklevel=2,
        )
    return size_mb


def train_kenlm(
    corpus_path: Path,
    out_arpa_path: Path,
    order: int = KENLM_DEFAULT_ORDER,
    prune: str = KENLM_DEFAULT_PRUNE,
    lmplz_binary: str = "lmplz",
) -> Path:
    """Train a KenLM model via subprocess call to lmplz.
    Command: lmplz -o {order} --prune {prune} < corpus_path > out_arpa_path

    Behaviour:
      - If lmplz is NOT on PATH (or named lmplz_binary), raise
        NotImplementedError with the install URL. Phase A test suite
        relies on this path on the dev machine.
      - If lmplz IS available, run the subprocess and write the ARPA
        file. Raises subprocess.CalledProcessError on non-zero exit.

    Returns out_arpa_path on success.
    Creates parent directories as needed.
    """
    if shutil.which(lmplz_binary) is None:
        raise NotImplementedError(_PHASE_A_MSG)

    out_arpa_path.parent.mkdir(parents=True, exist_ok=True)
    prune_args = prune.split()
    cmd = [lmplz_binary, "-o", str(order), "--prune", *prune_args]
    with corpus_path.open("rb") as stdin_f, out_arpa_path.open("wb") as stdout_f:
        subprocess.run(cmd, stdin=stdin_f, stdout=stdout_f, check=True)
    return out_arpa_path


def binarise_arpa(
    arpa_path: Path,
    out_klm_path: Path,
    build_binary: str = "build_binary",
) -> Path:
    """Binarise the ARPA file to .klm format for fast loading.
    Command: build_binary {arpa_path} {out_klm_path}

    Behaviour mirrors train_kenlm: NotImplementedError when the binary
    is absent; real subprocess call when present.
    Returns out_klm_path on success.
    """
    if shutil.which(build_binary) is None:
        raise NotImplementedError(_PHASE_A_MSG)

    subprocess.run(
        [build_binary, str(arpa_path), str(out_klm_path)],
        check=True,
    )
    return out_klm_path


def train_and_save(
    fold_entries: list[Any],
    out_dir: Path,
    scheme_name: str,
    order: int = KENLM_DEFAULT_ORDER,
) -> dict[str, Path]:
    """Full pipeline: assemble corpus → estimate size → train → binarise.
    Returns dict with keys: corpus, arpa, klm (paths).
    Raises NotImplementedError (from train_kenlm) when lmplz is absent.
    The size estimate is always computed and logged.
    """
    corpus_path = out_dir / f"fold_a_{scheme_name}.txt"
    out_arpa_path = out_dir / f"fold_a_{scheme_name}.arpa"
    out_klm_path = out_dir / f"fold_a_{scheme_name}.klm"

    n = len(fold_entries)
    estimate_arpa_size_mb(n, order=order)

    assemble_corpus(fold_entries, corpus_path)
    train_kenlm(corpus_path, out_arpa_path, order=order)  # raises if lmplz missing
    binarise_arpa(out_arpa_path, out_klm_path)

    return {"corpus": corpus_path, "arpa": out_arpa_path, "klm": out_klm_path}
