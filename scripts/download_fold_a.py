"""Download and assemble the Fold A calibration corpus — blueprint §12.2.

Two modes:

  Default (no flags):
    Prints step-by-step manual instructions for obtaining the three Fold A
    subsets. Use this mode when license review is required or when running
    in an air-gapped environment.

  --download:
    Calls the HuggingFace `datasets` library to automatically download and
    write the JSONL files to data/folds/fold_a/. Requires:
      - The `datasets` package installed (`uv add datasets` if not present).
      - Network access and any required HF auth tokens (set HF_TOKEN env var
        before running for gated datasets).
      - Acceptance of each dataset's license at first access. The script
        prints license URLs before attempting download.

Fold A corpus composition (blueprint §12.2):
  anthropic_hh_benign.jsonl   — benign turns from Anthropic HH-RLHF
  anthropic_hh_refused.jsonl  — refused turns from Anthropic HH-RLHF
  oasst_benign.jsonl          — benign turns from OpenAssistant OASST1

Target sizes (blueprint §12.2 minimum, total >= 2000 prompts):
  Benign:   >= 1200 prompts (60%)
  Refused:  >=  800 prompts (40%)

JSONL format (one JSON object per line):
  {"prompt": "<text>", "source": "<dataset-name>"}

Usage:
  python scripts/download_fold_a.py                  # print instructions
  python scripts/download_fold_a.py --download       # actually download
  python scripts/download_fold_a.py --download --max 500   # smaller cap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_FOLD_A_DIR = Path("data/folds/fold_a")
_REFUSAL_MARKERS = (
    "I cannot", "I can't", "I'm sorry", "I am sorry", "I won't", "I will not",
    "I am unable", "I am not able", "I do not", "as an AI",
    "against my", "I refuse",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _classify_refusal(text: str) -> bool:
    """Heuristic FP16 refusal detector — first 200 chars contain a refusal marker."""
    head = text[:200]
    return any(m.lower() in head.lower() for m in _REFUSAL_MARKERS)


def download_anthropic_hh(
    target_dir: Path,
    n_benign: int = 1200,
    n_refused: int = 800,
) -> None:
    """Download Anthropic HH-RLHF and write benign + refused JSONL files.

    License: MIT (see https://huggingface.co/datasets/Anthropic/hh-rlhf)
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets library not installed. Run: uv add datasets"
        ) from exc

    print("[download_fold_a] Loading Anthropic/hh-rlhf ...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")

    benign: list[dict[str, str]] = []
    refused: list[dict[str, str]] = []

    for row in ds:
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")
        # Extract the first human turn.
        parts = chosen.split("\n\nHuman:")
        if len(parts) < 2:
            continue
        try:
            human_turn = parts[1].split("\n\nAssistant:")[0].strip()
        except IndexError:
            continue
        if not human_turn:
            continue

        # Classify by whether the rejected response looks like a refusal.
        if _classify_refusal(rejected) and len(refused) < n_refused:
            refused.append({"prompt": human_turn, "source": "anthropic-hh"})
        elif not _classify_refusal(chosen) and len(benign) < n_benign:
            benign.append({"prompt": human_turn, "source": "anthropic-hh"})

        if len(benign) >= n_benign and len(refused) >= n_refused:
            break

    _write_jsonl(target_dir / "anthropic_hh_benign.jsonl", benign)
    _write_jsonl(target_dir / "anthropic_hh_refused.jsonl", refused)
    print(f"[download_fold_a]   benign: {len(benign)} prompts written")
    print(f"[download_fold_a]   refused: {len(refused)} prompts written")


def download_oasst(target_dir: Path, n_benign: int = 600) -> None:
    """Download OASST1 first-turn English prompts.

    License: Apache 2.0 (see https://huggingface.co/datasets/OpenAssistant/oasst1)
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets library not installed. Run: uv add datasets"
        ) from exc

    print("[download_fold_a] Loading OpenAssistant/oasst1 ...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    prompts: list[dict[str, str]] = []
    for row in ds:
        if row.get("role") != "prompter":
            continue
        if row.get("lang") != "en":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        prompts.append({"prompt": text, "source": "oasst"})
        if len(prompts) >= n_benign:
            break

    _write_jsonl(target_dir / "oasst_benign.jsonl", prompts)
    print(f"[download_fold_a]   oasst_benign: {len(prompts)} prompts written")


def verify(target_dir: Path) -> bool:
    expected = {
        "anthropic_hh_benign.jsonl": 1200,
        "anthropic_hh_refused.jsonl": 800,
        "oasst_benign.jsonl": 600,
    }
    all_ok = True
    print()
    print("Verification:")
    for fname, target in expected.items():
        path = target_dir / fname
        if path.exists():
            count = sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
            status = "OK" if count >= target else f"WARN (got {count}, need >= {target})"
            sha = _sha256_file(path)
            print(f"  {fname}: {count} lines — {status}")
            print(f"    SHA-256: {sha}")
        else:
            print(f"  {fname}: MISSING")
            all_ok = False
    return all_ok


def print_instructions() -> None:
    print("=" * 72)
    print("CLIFFGUARD — Fold A Calibration Corpus Assembly")
    print("=" * 72)
    print()
    print("To AUTO-DOWNLOAD with the datasets library:")
    print("  uv add datasets")
    print("  python scripts/download_fold_a.py --download")
    print()
    print("Target directory:", _FOLD_A_DIR.resolve())
    print()
    print("Sources and licenses:")
    print("  Anthropic HH-RLHF  — https://huggingface.co/datasets/Anthropic/hh-rlhf (MIT)")
    print("  OpenAssistant     — https://huggingface.co/datasets/OpenAssistant/oasst1 (Apache 2.0)")
    print()
    print("If you need to download manually (air-gapped):")
    print("  1. Visit each URL above and accept license terms.")
    print("  2. Use a workstation with network to run the --download flow.")
    print("  3. Copy data/folds/fold_a/ to the air-gapped machine.")
    print()
    print("After download, verify with:")
    print("  uv run python -c \"from cliffguard.eval.folds import load_fold_a_calibration; "
          "print(len(load_fold_a_calibration()), 'entries loaded')\"")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download or describe the Fold A calibration corpus."
    )
    parser.add_argument("--download", action="store_true",
                        help="Actually download datasets via the HF datasets library.")
    parser.add_argument("--max", type=int, default=None,
                        help="Optional cap on prompts per category (default uses blueprint minima).")
    parser.add_argument("--target-dir", type=Path, default=_FOLD_A_DIR,
                        help="Output directory (default: data/folds/fold_a)")
    args = parser.parse_args(argv)

    if not args.download:
        print_instructions()
        return 0

    n_benign_hh = args.max if args.max else 1200
    n_refused = args.max if args.max else 800
    n_benign_oa = args.max if args.max else 600

    try:
        download_anthropic_hh(args.target_dir, n_benign=n_benign_hh, n_refused=n_refused)
        download_oasst(args.target_dir, n_benign=n_benign_oa)
    except ImportError as exc:
        print(f"[download_fold_a] ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[download_fold_a] ERROR while downloading: {exc}", file=sys.stderr)
        return 3

    ok = verify(args.target_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
