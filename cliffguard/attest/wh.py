"""ATTEST-WH boot-time weight-hash attestation — see blueprint §5.12.

At boot time, hashes the on-disk GGUF / safetensors blob with SHA-256
and compares against a signed vendor manifest. A hash mismatch signals
Egashira-style poisoned weights (A2): the model is benign at FP16 but
malicious at a specific quantization. Outputs ALLOW, DEGRADED, or BLOCK
to the CONDUCTOR.

Per blueprint §4.3: vendor manifest is signed; ATTEST runs once at boot,
not per request. A mismatch does not necessarily mean the system halts
— CONDUCTOR may choose DEGRADED mode (reduced capability, heightened
alertness) rather than full BLOCK, depending on tier policy.
"""

import hashlib
import json
from enum import Enum
from pathlib import Path

from cliffguard.types import Tier


class AttestResult(Enum):
    ALLOW = "ALLOW"
    DEGRADED = "DEGRADED"
    BLOCK = "BLOCK"


def hash_file(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 of a file by reading in chunks.
    Returns the hex digest string.
    Raises FileNotFoundError if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path) -> dict[str, str]:
    """Load a vendor manifest JSON file mapping filename → expected
    SHA-256 hex digest.
    Returns the dict.
    Raises FileNotFoundError if manifest_path does not exist.
    Raises ValueError if the JSON is malformed or not a dict."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed manifest JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Manifest must be a JSON object (dict), got {type(data).__name__}"
        )
    return dict(data)


_EMBEDDED_TIERS = {Tier.C, Tier.C_PLUS}


def attest(
    model_path: Path,
    manifest_path: Path,
    tier: Tier,
) -> AttestResult:
    """Run boot-time attestation.
    1. Load manifest via load_manifest(manifest_path).
    2. Look up model_path.name in the manifest.
       If not found: return AttestResult.DEGRADED.
    3. Hash model_path via hash_file(model_path).
    4. Compare hash to manifest entry.
       If match: return AttestResult.ALLOW.
       If mismatch: return AttestResult.BLOCK for Tier A and B;
                    return AttestResult.DEGRADED for Tier C and C_PLUS
                    (embedded boards may not have signed manifests).
    Raises FileNotFoundError if model_path does not exist."""
    manifest = load_manifest(manifest_path)
    if model_path.name not in manifest:
        return AttestResult.DEGRADED
    actual = hash_file(model_path)
    expected = manifest[model_path.name]
    if actual == expected:
        return AttestResult.ALLOW
    return AttestResult.DEGRADED if tier in _EMBEDDED_TIERS else AttestResult.BLOCK
