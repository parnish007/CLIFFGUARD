"""Reproducibility manifest builder — see blueprint §15.

Builds a machine-readable JSON manifest recording everything needed
to reproduce a CLIFFGUARD evaluation run:

  - Git commit hash of the repo at evaluation time.
  - Hashes of all data files consumed (Fold A/B/C corpora).
  - Hashes of all artifact files produced (directions, calibration
    tables, ARPA files, result JSONLs).
  - Python environment snapshot (uv pip freeze equivalent).
  - Hardware description (tier, device name, quantization schemes run).
  - Timestamp (ISO 8601 UTC).
  - Pre-registration document hash (docs/preregistration.md SHA-256).

The manifest is written to artifacts/manifest_{timestamp}.json.
A manifest is considered valid if: (1) git hash is present and not
'UNKNOWN', (2) preregistration_hash matches the current
docs/preregistration.md SHA-256, (3) all listed data_files exist
on disk at the time of verification.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def get_git_hash(repo_root: Path) -> str:
    """Return the current HEAD commit hash via git rev-parse HEAD.
    Returns 'UNKNOWN' if git is unavailable or repo_root is not
    a git repository. Never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def hash_file_sha256(path: Path) -> str:
    """Return SHA-256 hex digest of a file.
    Raises FileNotFoundError if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_preregistration(repo_root: Path) -> str:
    """Return SHA-256 of docs/preregistration.md.
    Raises FileNotFoundError if file does not exist."""
    return hash_file_sha256(repo_root / "docs" / "preregistration.md")


def build_manifest(
    repo_root: Path,
    tier: str,
    schemes: list[str],
    data_files: list[Path],
    artifact_files: list[Path],
    hardware_description: str = "unspecified",
) -> dict[str, object]:
    """Build the manifest dict.
    Fields: git_hash, timestamp (ISO UTC), tier, schemes,
    hardware_description, preregistration_hash,
    data_files (list of {path, sha256}),
    artifact_files (list of {path, sha256} — skip if not found,
    record as {path, sha256: 'MISSING'}).
    Returns the dict."""
    git_hash = get_git_hash(repo_root)
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        prereg_hash = hash_preregistration(repo_root)
    except FileNotFoundError:
        prereg_hash = "MISSING"

    data_entries: list[dict[str, str]] = []
    for p in data_files:
        try:
            sha = hash_file_sha256(p)
        except FileNotFoundError:
            sha = "MISSING"
        data_entries.append({"path": str(p), "sha256": sha})

    artifact_entries: list[dict[str, str]] = []
    for p in artifact_files:
        try:
            sha = hash_file_sha256(p)
        except FileNotFoundError:
            sha = "MISSING"
        artifact_entries.append({"path": str(p), "sha256": sha})

    return {
        "git_hash": git_hash,
        "timestamp": timestamp,
        "tier": tier,
        "schemes": schemes,
        "hardware_description": hardware_description,
        "preregistration_hash": prereg_hash,
        "data_files": data_entries,
        "artifact_files": artifact_entries,
    }


def save_manifest(
    manifest: dict[str, object],
    artifacts_dir: Path,
) -> Path:
    """Save manifest as JSON to artifacts_dir/manifest_{timestamp}.json.
    Timestamp taken from manifest['timestamp'].
    Creates parent directories. Returns the written path."""
    timestamp = str(manifest["timestamp"])
    # Sanitize timestamp for use in a filename (replace colons/plusses).
    safe_ts = timestamp.replace(":", "-").replace("+", "p")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / f"manifest_{safe_ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path


def verify_manifest(
    manifest: dict[str, object],
    repo_root: Path,
) -> tuple[bool, list[str]]:
    """Verify a loaded manifest dict.
    Checks: (1) git_hash != 'UNKNOWN', (2) preregistration_hash
    matches current docs/preregistration.md SHA-256, (3) all
    data_files listed exist on disk.
    Returns (valid: bool, issues: list[str]).
    issues is empty when valid=True."""
    issues: list[str] = []

    # Check 1: git_hash present and not UNKNOWN.
    git_hash = manifest.get("git_hash", "")
    if not git_hash or git_hash == "UNKNOWN":
        issues.append("git_hash is UNKNOWN or missing")

    # Check 2: preregistration_hash matches current file.
    stored_hash = str(manifest.get("preregistration_hash", ""))
    try:
        current_hash = hash_preregistration(repo_root)
        if stored_hash != current_hash:
            issues.append(
                f"preregistration_hash mismatch: stored={stored_hash!r} "
                f"current={current_hash!r}"
            )
    except FileNotFoundError:
        issues.append("docs/preregistration.md not found during verification")

    # Check 3: all data_files exist on disk.
    data_files = manifest.get("data_files", [])
    if isinstance(data_files, list):
        for entry in data_files:
            if isinstance(entry, dict):
                p = Path(str(entry.get("path", "")))
                if not p.exists():
                    issues.append(f"data_file missing on disk: {p}")

    return (len(issues) == 0, issues)
