import json
from pathlib import Path

import pytest

from cliffguard.eval.repro import (
    build_manifest,
    get_git_hash,
    hash_file_sha256,
    hash_preregistration,
    save_manifest,
    verify_manifest,
)

# Real repo root (two levels up from this test file).
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# get_git_hash
# ---------------------------------------------------------------------------


def test_get_git_hash_returns_non_empty_for_real_repo() -> None:
    result = get_git_hash(REPO_ROOT)
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_git_hash_returns_known_hash_for_real_repo() -> None:
    result = get_git_hash(REPO_ROOT)
    # Must be either a valid 40-char hex hash or 'UNKNOWN'.
    assert result == "UNKNOWN" or (len(result) == 40 and all(c in "0123456789abcdef" for c in result))


def test_get_git_hash_returns_unknown_for_non_git_dir(tmp_path: Path) -> None:
    result = get_git_hash(tmp_path)
    assert result == "UNKNOWN"


def test_get_git_hash_never_raises(tmp_path: Path) -> None:
    # Should not raise even for a completely empty directory.
    result = get_git_hash(tmp_path)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# hash_file_sha256
# ---------------------------------------------------------------------------


def test_hash_file_sha256_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_file_sha256(tmp_path / "nonexistent.txt")


def test_hash_file_sha256_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "data.txt"
    f.write_text("hello cliffguard", encoding="utf-8")
    h1 = hash_file_sha256(f)
    h2 = hash_file_sha256(f)
    assert h1 == h2


def test_hash_file_sha256_returns_64_char_hex(tmp_path: Path) -> None:
    f = tmp_path / "sample.bin"
    f.write_bytes(b"\x00\x01\x02\x03")
    h = hash_file_sha256(f)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_file_sha256_changes_with_content(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("content A", encoding="utf-8")
    h1 = hash_file_sha256(f)
    f.write_text("content B", encoding="utf-8")
    h2 = hash_file_sha256(f)
    assert h1 != h2


def test_hash_file_sha256_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")
    h = hash_file_sha256(f)
    assert len(h) == 64


# ---------------------------------------------------------------------------
# hash_preregistration
# ---------------------------------------------------------------------------


def test_hash_preregistration_returns_64_char_hex() -> None:
    h = hash_preregistration(REPO_ROOT)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_preregistration_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_preregistration(tmp_path)


def test_hash_preregistration_is_deterministic() -> None:
    h1 = hash_preregistration(REPO_ROOT)
    h2 = hash_preregistration(REPO_ROOT)
    assert h1 == h2


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {
    "git_hash",
    "timestamp",
    "tier",
    "schemes",
    "hardware_description",
    "preregistration_hash",
    "data_files",
    "artifact_files",
}


def test_build_manifest_returns_dict_with_all_required_fields() -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=["FP16"], data_files=[], artifact_files=[])
    assert _REQUIRED_FIELDS.issubset(m.keys())


def test_build_manifest_tier_stored_correctly() -> None:
    m = build_manifest(REPO_ROOT, tier="C+", schemes=[], data_files=[], artifact_files=[])
    assert m["tier"] == "C+"


def test_build_manifest_schemes_stored_correctly() -> None:
    schemes = ["FP16", "NF4", "GGUF_Q3_K_M"]
    m = build_manifest(REPO_ROOT, tier="B", schemes=schemes, data_files=[], artifact_files=[])
    assert m["schemes"] == schemes


def test_build_manifest_hardware_description_default() -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    assert m["hardware_description"] == "unspecified"


def test_build_manifest_hardware_description_custom() -> None:
    m = build_manifest(
        REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[],
        hardware_description="Raspberry Pi 5 8GB",
    )
    assert m["hardware_description"] == "Raspberry Pi 5 8GB"


def test_build_manifest_timestamp_is_iso_string() -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    ts = str(m["timestamp"])
    assert "T" in ts  # ISO 8601 format contains 'T' separator


def test_build_manifest_records_missing_for_nonexistent_artifact(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.npy"
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[missing])
    entries = m["artifact_files"]
    assert isinstance(entries, list)
    assert len(entries) == 1
    assert entries[0]["sha256"] == "MISSING"  # type: ignore[index]


def test_build_manifest_hashes_existing_data_file(tmp_path: Path) -> None:
    f = tmp_path / "corpus.jsonl"
    f.write_text('{"prompt": "hello"}\n', encoding="utf-8")
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[f], artifact_files=[])
    entries = m["data_files"]
    assert isinstance(entries, list)
    assert len(entries) == 1
    assert entries[0]["sha256"] != "MISSING"  # type: ignore[index]
    assert len(entries[0]["sha256"]) == 64  # type: ignore[index]


def test_build_manifest_preregistration_hash_is_64_char_hex() -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    h = str(m["preregistration_hash"])
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# save_manifest
# ---------------------------------------------------------------------------


def test_save_manifest_writes_valid_json_file(tmp_path: Path) -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=["FP16"], data_files=[], artifact_files=[])
    out = save_manifest(m, tmp_path)
    assert out.exists()
    with open(out, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["tier"] == "A"


def test_save_manifest_returns_path_inside_artifacts_dir(tmp_path: Path) -> None:
    m = build_manifest(REPO_ROOT, tier="B", schemes=[], data_files=[], artifact_files=[])
    out = save_manifest(m, tmp_path)
    assert out.parent == tmp_path


def test_save_manifest_filename_contains_manifest(tmp_path: Path) -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    out = save_manifest(m, tmp_path)
    assert "manifest" in out.name


def test_save_manifest_creates_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "artifacts"
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    out = save_manifest(m, nested)
    assert out.exists()


# ---------------------------------------------------------------------------
# verify_manifest
# ---------------------------------------------------------------------------


def test_verify_manifest_valid_for_fresh_build_with_no_data_files() -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=["FP16"], data_files=[], artifact_files=[])
    valid, issues = verify_manifest(m, REPO_ROOT)
    assert valid is True
    assert issues == []


def test_verify_manifest_returns_false_when_git_hash_unknown() -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    m["git_hash"] = "UNKNOWN"
    valid, issues = verify_manifest(m, REPO_ROOT)
    assert valid is False
    assert len(issues) > 0


def test_verify_manifest_issues_mention_git_hash_when_unknown() -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    m["git_hash"] = "UNKNOWN"
    _, issues = verify_manifest(m, REPO_ROOT)
    assert any("git_hash" in issue for issue in issues)


def test_verify_manifest_returns_false_when_preregistration_hash_wrong() -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    m["preregistration_hash"] = "a" * 64  # wrong hash
    valid, issues = verify_manifest(m, REPO_ROOT)
    assert valid is False
    assert len(issues) > 0


def test_verify_manifest_issues_mention_preregistration_when_mismatch() -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    m["preregistration_hash"] = "b" * 64
    _, issues = verify_manifest(m, REPO_ROOT)
    assert any("preregistration" in issue for issue in issues)


def test_verify_manifest_returns_false_when_data_file_missing(tmp_path: Path) -> None:
    phantom = tmp_path / "phantom.jsonl"
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    m["data_files"] = [{"path": str(phantom), "sha256": "abc123"}]
    valid, issues = verify_manifest(m, REPO_ROOT)
    assert valid is False
    assert any("phantom" in issue for issue in issues)


def test_verify_manifest_issues_empty_when_valid() -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    _, issues = verify_manifest(m, REPO_ROOT)
    # If valid, issues must be empty (checked via valid flag too).
    valid, issues2 = verify_manifest(m, REPO_ROOT)
    if valid:
        assert issues2 == []


def test_verify_manifest_both_issues_when_git_and_prereg_wrong() -> None:
    m = build_manifest(REPO_ROOT, tier="A", schemes=[], data_files=[], artifact_files=[])
    m["git_hash"] = "UNKNOWN"
    m["preregistration_hash"] = "c" * 64
    valid, issues = verify_manifest(m, REPO_ROOT)
    assert valid is False
    assert len(issues) >= 2
