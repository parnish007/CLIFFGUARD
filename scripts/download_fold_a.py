"""Download and assemble Fold A calibration corpus — see blueprint §12.2.

This script prints step-by-step instructions for obtaining the three
subsets that compose Fold A and writing them to data/folds/fold_a/.
It does NOT automatically download datasets because each source has
its own license terms that require explicit human acceptance.

Fold A corpus composition (blueprint §12.2):
  anthropic_hh_benign.jsonl   — benign turns from Anthropic-HH (HH-RLHF)
  anthropic_hh_refused.jsonl  — refused turns from Anthropic-HH (HH-RLHF)
  oasst_benign.jsonl          — benign turns from OpenAssistant (OASST1)

Target sizes (blueprint §12.2 minimum):
  Total Fold A: >= 2000 prompts
  Benign:       >= 1200 prompts (60%)
  Refused:      >= 800 prompts  (40%)

JSONL format (one JSON object per line):
  {"prompt": "<text>", "source": "<dataset-name>"}

Usage:
  python scripts/download_fold_a.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_FOLD_A_DIR = Path("data/folds/fold_a")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("=" * 72)
    print("CLIFFGUARD — Fold A Calibration Corpus Assembly")
    print("=" * 72)
    print()
    print("Target directory:", _FOLD_A_DIR.resolve())
    print()
    print("STEP 1 — Create the target directory")
    print("-" * 40)
    print("  mkdir -p data/folds/fold_a")
    print()
    print("STEP 2 — Download Anthropic-HH (HH-RLHF)")
    print("-" * 40)
    print("  License: MIT (check https://huggingface.co/datasets/Anthropic/hh-rlhf)")
    print("  Source:  https://huggingface.co/datasets/Anthropic/hh-rlhf")
    print()
    print("  Python extraction (run once, requires `datasets` package):")
    print("""
    from datasets import load_dataset
    import json, pathlib

    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    out_dir = pathlib.Path("data/folds/fold_a")

    benign, refused = [], []
    for row in ds:
        chosen = row["chosen"]
        rejected = row["rejected"]
        # Extract the human turn (first Human: ... segment)
        human_turn = chosen.split("\\n\\nHuman:")[1].split("\\n\\nAssistant:")[0].strip()
        label = row.get("label", "benign")
        if label == "refused" or "I cannot" in row["chosen"][:200]:
            refused.append({"prompt": human_turn, "source": "anthropic-hh"})
        else:
            benign.append({"prompt": human_turn, "source": "anthropic-hh"})

    with open(out_dir / "anthropic_hh_benign.jsonl", "w") as f:
        for r in benign[:1200]:
            f.write(json.dumps(r) + "\\n")

    with open(out_dir / "anthropic_hh_refused.jsonl", "w") as f:
        for r in refused[:800]:
            f.write(json.dumps(r) + "\\n")
    """)
    print()
    print("STEP 3 — Download OASST1 (OpenAssistant)")
    print("-" * 40)
    print("  License: Apache 2.0 (check https://huggingface.co/datasets/OpenAssistant/oasst1)")
    print("  Source:  https://huggingface.co/datasets/OpenAssistant/oasst1")
    print()
    print("  Python extraction:")
    print("""
    from datasets import load_dataset
    import json, pathlib

    ds = load_dataset("OpenAssistant/oasst1", split="train")
    out_dir = pathlib.Path("data/folds/fold_a")

    prompts = [
        {"prompt": row["text"], "source": "oasst"}
        for row in ds
        if row["role"] == "prompter" and row["lang"] == "en"
    ]

    with open(out_dir / "oasst_benign.jsonl", "w") as f:
        for r in prompts[:600]:
            f.write(json.dumps(r) + "\\n")
    """)
    print()
    print("STEP 4 — Verify line counts")
    print("-" * 40)
    expected = {
        "anthropic_hh_benign.jsonl": 1200,
        "anthropic_hh_refused.jsonl": 800,
        "oasst_benign.jsonl": 600,
    }
    all_ok = True
    for fname, target in expected.items():
        path = _FOLD_A_DIR / fname
        if path.exists():
            count = sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
            status = "OK" if count >= target else f"WARN (got {count}, need >= {target})"
            sha = _sha256_file(path)
            print(f"  {fname}: {count} lines — {status}")
            print(f"    SHA-256: {sha}")
        else:
            print(f"  {fname}: MISSING")
            all_ok = False
    print()
    if all_ok:
        print("Fold A corpus assembled. Run:")
        print("  uv run python -c \"from cliffguard.eval.folds import load_fold_a_calibration; "
              "print(len(load_fold_a_calibration()), 'entries loaded')\"")
    else:
        print("Some files are missing. Complete Steps 2-3 before continuing.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
