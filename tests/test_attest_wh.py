import hashlib
import json

import pytest

from cliffguard.types import Tier
from cliffguard.attest.wh import AttestResult, attest, hash_file, load_manifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYNTHETIC_BYTES = b"\x00\x01\x02\x03" * 1024  # 4 KiB synthetic binary


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# hash_file
# ---------------------------------------------------------------------------


def test_hash_file_raises_for_nonexistent_path(tmp_path: pytest.TempPathFactory) -> None:
    missing = tmp_path / "no_such_file.gguf"  # type: ignore[operator]
    with pytest.raises(FileNotFoundError):
        hash_file(missing)  # type: ignore[arg-type]


def test_hash_file_returns_64_char_hex(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "model.gguf"  # type: ignore[operator]
    p.write_bytes(_SYNTHETIC_BYTES)  # type: ignore[union-attr]
    digest = hash_file(p)  # type: ignore[arg-type]
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_hash_file_is_deterministic(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "model.gguf"  # type: ignore[operator]
    p.write_bytes(_SYNTHETIC_BYTES)  # type: ignore[union-attr]
    assert hash_file(p) == hash_file(p)  # type: ignore[arg-type]


def test_hash_file_matches_known_digest(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "model.gguf"  # type: ignore[operator]
    p.write_bytes(_SYNTHETIC_BYTES)  # type: ignore[union-attr]
    assert hash_file(p) == _sha256(_SYNTHETIC_BYTES)  # type: ignore[arg-type]


def test_hash_file_differs_for_different_content(tmp_path: pytest.TempPathFactory) -> None:
    p1 = tmp_path / "a.gguf"  # type: ignore[operator]
    p2 = tmp_path / "b.gguf"  # type: ignore[operator]
    p1.write_bytes(b"AAA")  # type: ignore[union-attr]
    p2.write_bytes(b"BBB")  # type: ignore[union-attr]
    assert hash_file(p1) != hash_file(p2)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


def test_load_manifest_raises_for_nonexistent_path(tmp_path: pytest.TempPathFactory) -> None:
    missing = tmp_path / "no_manifest.json"  # type: ignore[operator]
    with pytest.raises(FileNotFoundError):
        load_manifest(missing)  # type: ignore[arg-type]


def test_load_manifest_raises_for_malformed_json(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "manifest.json"  # type: ignore[operator]
    p.write_text("{ not valid json }", encoding="utf-8")  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="[Mm]alformed"):
        load_manifest(p)  # type: ignore[arg-type]


def test_load_manifest_raises_for_json_list(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "manifest.json"  # type: ignore[operator]
    p.write_text(json.dumps(["a", "b"]), encoding="utf-8")  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        load_manifest(p)  # type: ignore[arg-type]


def test_load_manifest_returns_correct_dict(tmp_path: pytest.TempPathFactory) -> None:
    data = {"model.gguf": "abc123", "other.gguf": "def456"}
    p = tmp_path / "manifest.json"  # type: ignore[operator]
    p.write_text(json.dumps(data), encoding="utf-8")  # type: ignore[union-attr]
    assert load_manifest(p) == data  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# attest
# ---------------------------------------------------------------------------


def _write_model_and_manifest(
    tmp_path: pytest.TempPathFactory,  # type: ignore[type-arg]
    *,
    content: bytes = _SYNTHETIC_BYTES,
    correct_hash: bool = True,
) -> tuple:  # type: ignore[type-arg]
    model = tmp_path / "model.gguf"  # type: ignore[operator]
    model.write_bytes(content)  # type: ignore[union-attr]
    digest = _sha256(content) if correct_hash else "0" * 64
    manifest = tmp_path / "manifest.json"  # type: ignore[operator]
    manifest.write_text(json.dumps({"model.gguf": digest}), encoding="utf-8")  # type: ignore[union-attr]
    return model, manifest


def test_attest_returns_allow_on_hash_match(tmp_path: pytest.TempPathFactory) -> None:
    model, manifest = _write_model_and_manifest(tmp_path, correct_hash=True)
    assert attest(model, manifest, Tier.A) == AttestResult.ALLOW  # type: ignore[arg-type]


def test_attest_returns_block_for_tier_a_on_mismatch(tmp_path: pytest.TempPathFactory) -> None:
    model, manifest = _write_model_and_manifest(tmp_path, correct_hash=False)
    assert attest(model, manifest, Tier.A) == AttestResult.BLOCK  # type: ignore[arg-type]


def test_attest_returns_block_for_tier_b_on_mismatch(tmp_path: pytest.TempPathFactory) -> None:
    model, manifest = _write_model_and_manifest(tmp_path, correct_hash=False)
    assert attest(model, manifest, Tier.B) == AttestResult.BLOCK  # type: ignore[arg-type]


def test_attest_returns_degraded_for_tier_c_on_mismatch(tmp_path: pytest.TempPathFactory) -> None:
    model, manifest = _write_model_and_manifest(tmp_path, correct_hash=False)
    assert attest(model, manifest, Tier.C) == AttestResult.DEGRADED  # type: ignore[arg-type]


def test_attest_returns_degraded_for_tier_c_plus_on_mismatch(tmp_path: pytest.TempPathFactory) -> None:
    model, manifest = _write_model_and_manifest(tmp_path, correct_hash=False)
    assert attest(model, manifest, Tier.C_PLUS) == AttestResult.DEGRADED  # type: ignore[arg-type]


def test_attest_returns_degraded_when_model_not_in_manifest(tmp_path: pytest.TempPathFactory) -> None:
    model = tmp_path / "unlisted.gguf"  # type: ignore[operator]
    model.write_bytes(_SYNTHETIC_BYTES)  # type: ignore[union-attr]
    manifest = tmp_path / "manifest.json"  # type: ignore[operator]
    manifest.write_text(json.dumps({"other_model.gguf": "abc123"}), encoding="utf-8")  # type: ignore[union-attr]
    assert attest(model, manifest, Tier.A) == AttestResult.DEGRADED  # type: ignore[arg-type]


def test_attest_raises_for_nonexistent_model_path(tmp_path: pytest.TempPathFactory) -> None:
    missing = tmp_path / "ghost.gguf"  # type: ignore[operator]
    manifest = tmp_path / "manifest.json"  # type: ignore[operator]
    manifest.write_text(json.dumps({"ghost.gguf": "abc123"}), encoding="utf-8")  # type: ignore[union-attr]
    with pytest.raises(FileNotFoundError):
        attest(missing, manifest, Tier.A)  # type: ignore[arg-type]
