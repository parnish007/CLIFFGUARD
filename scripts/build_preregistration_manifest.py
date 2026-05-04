"""CLI: build and save a CLIFFGUARD reproducibility manifest.

Usage:
  uv run python scripts/build_preregistration_manifest.py \\
      --tier A --schemes FP16 NF4 GGUF_Q3_K_M \\
      [--repo-root .] [--artifacts-dir artifacts/] [--data-dir data/]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without the package installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliffguard.eval.repro import build_manifest, save_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CLIFFGUARD reproducibility manifest."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Root of the git repository (default: current directory).",
    )
    parser.add_argument(
        "--tier",
        type=str,
        required=True,
        help="Evaluation tier (e.g. A, B, C, C+).",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=[],
        help="Quantization scheme names (e.g. FP16 NF4 GGUF_Q3_K_M).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where artifacts live and manifest will be saved.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing data files to hash.",
    )
    parser.add_argument(
        "--hardware-description",
        type=str,
        default="unspecified",
        help="Hardware description string.",
    )
    return parser.parse_args(argv)


def collect_files(directory: Path, patterns: list[str]) -> list[Path]:
    if not directory.exists():
        return []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(directory.glob(pattern)))
    return files


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = args.repo_root.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    data_dir = args.data_dir.resolve()

    data_files = collect_files(data_dir, ["**/*.jsonl"])
    artifact_files = collect_files(
        artifacts_dir, ["**/*.json", "**/*.npy", "**/*.arpa"]
    )

    manifest = build_manifest(
        repo_root=repo_root,
        tier=args.tier,
        schemes=args.schemes,
        data_files=data_files,
        artifact_files=artifact_files,
        hardware_description=args.hardware_description,
    )

    out_path = save_manifest(manifest, artifacts_dir)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
