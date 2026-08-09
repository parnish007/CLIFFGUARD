"""Fail early when a round-2 Colab run cannot produce paired, durable results.

The 256-token round takes three to four hours on a free T4. Its most important
comparison is against the existing 48-token runs, so it is worth spending a few
CPU seconds proving that their prompt rows still mean the same thing before a
session spends its limited GPU allocation. This script also catches missing
notebook entry points, an unsuitable runtime, and a destination that cannot
hold the results.

Usage:
  python scripts/preflight_round2.py
  python scripts/preflight_round2.py --skip-gpu
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import NamedTuple


REPO = Path(__file__).resolve().parents[1]
EXPECTED_PROMPT_SHA = "7da25bf88ee0409ce4900a12052e15849a2898ed01cfdcdfe6409bbfc11bd9b5"

# This is deliberately data rather than control flow: the notebook contract is
# visible in one place, and adding a command cannot quietly evade the gate.
SCRIPT_FLAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("run_behavioural_ladder.py", (
        "--model", "--n", "--bits", "--max-new-tokens", "--seed", "--cache",
        "--label", "--deployed", "--no-activations",
    )),
    ("run_sector_ladder.py", ("--model", "--n", "--bits", "--cache", "--label")),
    # --no-activations: the behavioural script grew this flag after the
    # notebook was written. Without it every scheme pays for a second full
    # forward with output_hidden_states=True, at batch 8, to collect
    # activations no round-2 step reads -- the largest avoidable VRAM spike
    # in the session, and the one path with no OOM batch-halving fallback.
    # --completion-chars: how much of a completion the judge is shown. A
    # 256-token run graded through the old 600-character window would be a
    # ~117-token measurement wearing a 256-token label.
    ("classify_completions_judge.py", ("--judge-model", "--judge-4bit",
                                       "--completion-chars")),
)

FLAG_RE = re.compile(r'add_argument\(\s*[\'\"](?P<flag>--[^\'\"]+)[\'\"]')
GIB = 1024 ** 3


class Result(NamedTuple):
    """One preflight finding and the recovery instruction if it blocks a run.

    Four verdicts, and the distinction between two of them is the whole point.
    `ok` means the check ran and passed. `skip` means it could not run here --
    no GPU on the author's laptop, no `statvfs` on Windows. Reporting a skipped
    check as `ok` would make this script's green light mean "nothing looked
    wrong OR nothing looked", which is precisely the reassurance a preflight
    must never give: the checks most likely to be skipped locally are the ones
    that only matter on Colab.
    """

    name: str
    verdict: str
    detail: str
    remedy: str | None = None


def _emit(result: Result) -> None:
    print(f"{result.name:<20} {result.verdict:<4} {result.detail}")


def _corpus_check(n: int, expected_sha: str) -> Result:
    """Hash the exact list used by the Fold A behavioural path."""
    try:
        sys.path.insert(0, str(REPO))
        from cliffguard.eval.storage import prompt_manifest_hash
        from scripts.run_local_ladder import load_prompts

        harmful, benign = load_prompts(n)
        prompt_sha = prompt_manifest_hash(harmful + benign)
    except (Exception, SystemExit) as exc:
        # `load_prompts` correctly raises SystemExit when the corpus is absent;
        # a preflight must turn that into a finding rather than exit before the
        # remaining, independent runtime checks can help the user.
        return Result(
            "corpus pairing", "FAIL", f"could not build Fold A prompt list: {exc}",
            "Restore the Fold A corpus from Drive. Do not rebuild it from HuggingFace: "
            "download_fold_a.py uses an unpinned upstream revision.",
        )

    recorded, unreadable, without_combined_hash = _recorded_prompt_hashes()
    matching_runs = [run for run, sha in recorded if sha == prompt_sha]
    recorded_detail = (
        "matching runs: " + ", ".join(matching_runs)
        if recorded else "no recorded runs present"
    )
    nonmatching = len(recorded) - len(matching_runs)
    if nonmatching:
        recorded_detail += f"; {nonmatching} recorded hash(es) differ"
    if without_combined_hash:
        recorded_detail += f"; {without_combined_hash} without a combined prompts hash"
    if unreadable:
        recorded_detail += f"; {len(unreadable)} manifest(s) unreadable"

    if prompt_sha != expected_sha:
        return Result(
            "corpus pairing", "FAIL",
            f"sha256 {prompt_sha} != expected {expected_sha}; {recorded_detail}",
            "Restore the corpus from Drive instead of rebuilding it. The round-2 "
            "256-token run is meaningful only when row i is the same prompt as "
            "row i of the 48-token runs. Fold A is re-downloaded from HuggingFace "
            "without a revision pin: download_fold_a.py calls "
            "load_dataset(\"Anthropic/hh-rlhf\", split=\"train\"). An upstream "
            "change silently breaks that pairing and wastes the whole session.",
        )
    if unreadable:
        return Result(
            "corpus pairing", "FAIL",
            f"sha256 matches expected; {recorded_detail}",
            "Repair or remove the unreadable run manifests before relying on their "
            "provenance. Restore the corpus from Drive rather than rebuilding it.",
        )
    return Result(
        "corpus pairing", "ok",
        f"sha256 matches expected; {recorded_detail}",
    )


def _recorded_prompt_hashes() -> tuple[list[tuple[str, str]], list[Path], int]:
    """Read existing behavioural manifests without requiring that any exist."""
    recorded: list[tuple[str, str]] = []
    unreadable: list[Path] = []
    without_combined_hash = 0
    for manifest_path in sorted((REPO / "artifacts" / "runs").glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable.append(manifest_path)
            continue
        try:
            sha = manifest["corpora"]["prompts"]["sha256_ordered"]
        except (KeyError, TypeError):
            # Older local ladders record their two arms separately. They remain
            # useful provenance, but cannot confirm the combined row order.
            without_combined_hash += 1
            continue
        if not isinstance(sha, str):
            unreadable.append(manifest_path)
            continue
        recorded.append((manifest_path.parent.name, sha))
    return recorded, unreadable, without_combined_hash


def _scripts_check() -> Result:
    """Confirm the notebook's commands still parse without importing torch."""
    problems: list[str] = []
    for script, required_flags in SCRIPT_FLAGS:
        path = REPO / "scripts" / script
        if not path.exists():
            problems.append(f"{script} missing")
            continue
        source = path.read_text(encoding="utf-8")
        found = set(FLAG_RE.findall(source))
        missing = [flag for flag in required_flags if flag not in found]
        if missing:
            problems.append(f"{script}: missing {', '.join(missing)}")
    if problems:
        return Result(
            "scripts and flags", "FAIL", "; ".join(problems),
            "Restore the notebook command contract: add the missing script or "
            "argument, or update this preflight and the notebook together.",
        )
    return Result("scripts and flags", "ok", "all notebook commands and flags found")


def _gpu_check(skip_gpu: bool) -> Result:
    if skip_gpu:
        return Result("gpu", "skip", "not checked (--skip-gpu)")
    try:
        import torch
    except ImportError:
        return Result("gpu", "skip", "not checked: torch is not installed")
    try:
        available = torch.cuda.is_available()
    except Exception as exc:
        return Result(
            "gpu", "FAIL", f"could not query CUDA: {exc}",
            "Restart the Colab GPU runtime and rerun this check before beginning the ladder.",
        )
    if not available:
        return Result(
            "gpu", "FAIL", "torch.cuda.is_available() is False",
            "Start a Colab GPU runtime and rerun this check before beginning the ladder.",
        )
    try:
        properties = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        return Result(
            "gpu", "FAIL", f"could not inspect CUDA device: {exc}",
            "Restart the Colab GPU runtime and rerun this check before beginning the ladder.",
        )
    vram_gib = properties.total_memory / GIB
    detail = f"{device_name}; {vram_gib:.1f} GiB"
    if vram_gib < 15:
        return Result(
            "gpu", "WARN", f"{detail}; below the 15 GiB recommendation",
        )
    return Result("gpu", "ok", detail)


def _disk_check() -> Result:
    if not hasattr(os, "statvfs"):
        return Result("disk", "skip",
                      "not checked: os.statvfs is unavailable here")
    try:
        stats = os.statvfs(REPO)
    except OSError as exc:
        return Result(
            "disk", "FAIL", f"could not read free space: {exc}",
            "Use a writable runtime disk with at least 25 GiB free before downloading models.",
        )
    free_gib = stats.f_frsize * stats.f_bavail / GIB
    if free_gib < 25:
        return Result(
            "disk", "FAIL", f"{free_gib:.1f} GiB free; need at least 25 GiB",
            "Free space or choose a larger runtime disk. The checkpoints need about 32 GB.",
        )
    if free_gib < 40:
        return Result("disk", "WARN", f"{free_gib:.1f} GiB free; 40 GiB is recommended")
    return Result("disk", "ok", f"{free_gib:.1f} GiB free")


def _writability_check() -> Result:
    runs = REPO / "artifacts" / "runs"
    probe = runs / ".preflight_round2_write_probe"
    try:
        runs.mkdir(parents=True, exist_ok=True)
        probe.write_text("preflight\n", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Result(
            "artifacts writable", "FAIL", f"{runs}: {exc}",
            "Mount or select a writable Drive-backed artifacts directory before running.",
        )
    return Result("artifacts writable", "ok", str(runs.relative_to(REPO)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=250, help="prompts per source class")
    ap.add_argument("--expect-sha", default=EXPECTED_PROMPT_SHA,
                    help="expected order-sensitive Fold A prompt hash")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="run the non-GPU checks on a CPU-only machine")
    args = ap.parse_args()

    results = (
        _corpus_check(args.n, args.expect_sha),
        _scripts_check(),
        _gpu_check(args.skip_gpu),
        _disk_check(),
        _writability_check(),
    )
    for result in results:
        _emit(result)

    failures = [r for r in results if r.verdict == "FAIL"]
    warnings = sum(r.verdict == "WARN" for r in results)
    skipped = [r.name for r in results if r.verdict == "skip"]
    tail = f"{warnings} warning(s), {len(skipped)} not checked"

    if failures:
        print(f"{'summary':<20} FAIL {len(failures)} failure(s), {tail}")
        print("\nWHAT TO DO BEFORE STARTING THE NOTEBOOK")
        for result in failures:
            print(f"\n{result.name}: {result.remedy}")
        return 1

    print(f"{'summary':<20} ok   {tail}")
    # Named rather than merely counted. A run that passes here on a laptop has
    # not checked the GPU or the disk, and those are exactly the two things
    # that fail on Colab -- so the operator has to be told which reassurance
    # they are not getting, not just how many.
    if skipped:
        print(f"{'':<20}      not checked here: {', '.join(skipped)}. "
              "Rerun without --skip-gpu on the Colab runtime itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
