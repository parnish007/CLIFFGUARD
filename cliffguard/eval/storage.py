"""Organised, provenance-stamped artifact storage.

Every experimental run gets its own immutable directory keyed by a run id that
encodes when it ran and which commit produced it:

    artifacts/runs/<YYYYmmdd-HHMMSS>_<git-sha7>_<label>/
        manifest.json          provenance: git sha, model, corpus hashes, env
        activations/           raw residual-stream matrices (.npy)
        directions/            fitted direction vectors (.npz)
        margins/               per-prompt margin scores (.npy)
        results/               analysis outputs (.json)

Why this shape:

  * **Raw before summary.** Defect D3 happened partly because only thresholds
    were persisted, so `Delta_B-cliff = 0.000` could not be audited afterwards.
    Margins and activations are saved so any later analysis can be redone
    without a GPU.
  * **Corpus hashes in the manifest.** A result is meaningless without knowing
    exactly which prompts produced it, in which order.
  * **Immutability.** Runs are never mutated in place; a re-run makes a new
    directory. `artifacts/runs/INDEX.md` accumulates one line per run.

Nothing here depends on torch, so results can be re-analysed in the project
venv even though model inference needs a separate GPU environment.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

ARTIFACTS_ROOT = Path("artifacts")
RUNS_ROOT = ARTIFACTS_ROOT / "runs"
INDEX_FILE = RUNS_ROOT / "INDEX.md"

_SUBDIRS = ("activations", "directions", "margins", "results")


def git_sha(short: bool = True) -> str:
    """Current commit, or 'nogit' when unavailable. Never raises."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip().split()[0]
    except Exception:
        return "nogit"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_manifest_hash(prompts: list[str]) -> str:
    """Order-sensitive hash of a prompt list.

    Order matters: the paired analyses require row i to be the same prompt
    under every scheme, so a reordering invalidates the comparison and must
    produce a different hash."""
    return sha256_text("\n".join(prompts))


@dataclass
class RunDir:
    """An immutable, provenance-stamped run directory."""

    path: Path
    run_id: str
    manifest: dict[str, Any] = field(default_factory=dict)

    # -- paths -------------------------------------------------------------
    def sub(self, kind: str) -> Path:
        if kind not in _SUBDIRS:
            raise ValueError(f"unknown subdir {kind!r}; expected one of {_SUBDIRS}")
        p = self.path / kind
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- writers -----------------------------------------------------------
    def save_array(self, kind: str, name: str, arr: FloatArray) -> Path:
        """Persist an array and record its shape and hash in the manifest."""
        p = self.sub(kind) / f"{name}.npy"
        np.save(p, arr)
        self.manifest.setdefault("arrays", {})[f"{kind}/{name}"] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": sha256_file(p),
        }
        return p

    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        p = self.sub("results") / f"{name}.json"
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return p

    def write_manifest(self) -> Path:
        p = self.path / "manifest.json"
        p.write_text(json.dumps(self.manifest, indent=2, default=str), encoding="utf-8")
        return p

    # -- index -------------------------------------------------------------
    def append_to_index(self, summary: str) -> None:
        """Add one line to artifacts/runs/INDEX.md."""
        INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not INDEX_FILE.exists():
            INDEX_FILE.write_text(
                "# Run index\n\n"
                "One line per experimental run. Runs are immutable; a re-run "
                "creates a new directory.\n\n"
                "| run id | label | summary |\n|---|---|---|\n",
                encoding="utf-8",
            )
        label = self.manifest.get("label", "")
        with INDEX_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"| `{self.run_id}` | {label} | {summary} |\n")


def new_run(
    label: str,
    model_id: str | None = None,
    extra: dict[str, Any] | None = None,
    root: Path = RUNS_ROOT,
) -> RunDir:
    """Create a fresh run directory with a provenance manifest.

    `label` is a short slug (e.g. 'stage0-qwen1.5b-nf4') that appears in the
    directory name and the index."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in label)
    run_id = f"{stamp}_{git_sha()}_{safe}"
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    for d in _SUBDIRS:
        (path / d).mkdir(exist_ok=True)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "label": label,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": git_sha(short=False),
        "model_id": model_id,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
    }
    if extra:
        manifest.update(extra)
    return RunDir(path=path, run_id=run_id, manifest=manifest)


def record_corpus(run: RunDir, name: str, prompts: list[str]) -> None:
    """Record a prompt list's size and order-sensitive hash in the manifest."""
    run.manifest.setdefault("corpora", {})[name] = {
        "n": len(prompts),
        "sha256_ordered": prompt_manifest_hash(prompts),
    }


def record_environment(run: RunDir) -> None:
    """Record versions of the packages whose behaviour affects results.

    transformers 5.x changed `torch_dtype` -> `dtype` and made
    `apply_chat_template` return a BatchEncoding; silently running under a
    different major version can change what is extracted, so the version is
    part of the provenance."""
    env: dict[str, str] = {}
    for mod in ("numpy", "scipy", "torch", "transformers", "bitsandbytes"):
        try:
            m = __import__(mod)
            env[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            env[mod] = "absent"
    try:
        import torch

        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["vram_gb"] = str(
                round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            )
    except Exception:
        pass
    run.manifest["environment"] = env


def list_runs(root: Path = RUNS_ROOT) -> list[Path]:
    """All run directories, newest first."""
    if not root.exists():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()),
        reverse=True,
    )


def load_run(run_id: str, root: Path = RUNS_ROOT) -> dict[str, Any]:
    """Load a run's manifest by id. Raises FileNotFoundError if absent."""
    p = root / run_id / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"no manifest at {p}")
    loaded: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return loaded
