import hashlib
import json
from pathlib import Path

import pytest

from cliffguard.eval.folds import (
    Fold,
    FoldEntry,
    fold_isolation_check,
    load_fold_a_calibration,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, records: list[dict]) -> None:  # type: ignore[type-arg]
    path.write_text(
        "\n".join(json.dumps(r) for r in records),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# load_fold_a_calibration — missing directory
# ---------------------------------------------------------------------------


def test_load_fold_a_raises_file_not_found_when_dir_missing(tmp_path: pytest.TempPathFactory) -> None:
    missing = tmp_path / "no_such_fold_a"  # type: ignore[operator]
    with pytest.raises(FileNotFoundError, match="download_fold_a.py"):
        load_fold_a_calibration(fold_a_dir=missing)  # type: ignore[arg-type]


def test_load_fold_a_error_message_mentions_directory(tmp_path: pytest.TempPathFactory) -> None:
    missing = tmp_path / "absent"  # type: ignore[operator]
    with pytest.raises(FileNotFoundError) as exc_info:
        load_fold_a_calibration(fold_a_dir=missing)  # type: ignore[arg-type]
    assert "fold_a" in str(exc_info.value).lower() or "absent" in str(exc_info.value)


# ---------------------------------------------------------------------------
# load_fold_a_calibration — correct parsing
# ---------------------------------------------------------------------------


def test_load_fold_a_returns_correct_count(tmp_path: pytest.TempPathFactory) -> None:
    fold_dir = tmp_path / "fold_a"  # type: ignore[operator]
    fold_dir.mkdir()  # type: ignore[union-attr]
    records = [{"prompt": f"Prompt {i}", "source": "anthropic-hh"} for i in range(10)]
    _write_jsonl(fold_dir / "anthropic_hh_benign.jsonl", records)  # type: ignore[operator]
    entries = load_fold_a_calibration(fold_a_dir=fold_dir)  # type: ignore[arg-type]
    assert len(entries) == 10


def test_load_fold_a_returns_fold_entry_objects(tmp_path: pytest.TempPathFactory) -> None:
    fold_dir = tmp_path / "fold_a"  # type: ignore[operator]
    fold_dir.mkdir()  # type: ignore[union-attr]
    _write_jsonl(
        fold_dir / "anthropic_hh_benign.jsonl",  # type: ignore[operator]
        [{"prompt": "Hello", "source": "anthropic-hh"}],
    )
    entries = load_fold_a_calibration(fold_a_dir=fold_dir)  # type: ignore[arg-type]
    assert all(isinstance(e, FoldEntry) for e in entries)


def test_load_fold_a_sets_fold_enum(tmp_path: pytest.TempPathFactory) -> None:
    fold_dir = tmp_path / "fold_a"  # type: ignore[operator]
    fold_dir.mkdir()  # type: ignore[union-attr]
    _write_jsonl(
        fold_dir / "anthropic_hh_benign.jsonl",  # type: ignore[operator]
        [{"prompt": "Test prompt", "source": "anthropic-hh"}],
    )
    entries = load_fold_a_calibration(fold_a_dir=fold_dir)  # type: ignore[arg-type]
    assert all(e.fold == Fold.A for e in entries)


def test_load_fold_a_computes_correct_sha256(tmp_path: pytest.TempPathFactory) -> None:
    fold_dir = tmp_path / "fold_a"  # type: ignore[operator]
    fold_dir.mkdir()  # type: ignore[union-attr]
    prompt_text = "What is the capital of France?"
    _write_jsonl(
        fold_dir / "anthropic_hh_benign.jsonl",  # type: ignore[operator]
        [{"prompt": prompt_text, "source": "anthropic-hh"}],
    )
    entries = load_fold_a_calibration(fold_a_dir=fold_dir)  # type: ignore[arg-type]
    assert entries[0].sha256 == _sha256(prompt_text)


def test_load_fold_a_loads_multiple_files(tmp_path: pytest.TempPathFactory) -> None:
    fold_dir = tmp_path / "fold_a"  # type: ignore[operator]
    fold_dir.mkdir()  # type: ignore[union-attr]
    benign = [{"prompt": f"Benign {i}", "source": "anthropic-hh"} for i in range(4)]
    refused = [{"prompt": f"Refused {i}", "source": "anthropic-hh"} for i in range(3)]
    oasst = [{"prompt": f"OASST {i}", "source": "oasst"} for i in range(3)]
    _write_jsonl(fold_dir / "anthropic_hh_benign.jsonl", benign)  # type: ignore[operator]
    _write_jsonl(fold_dir / "anthropic_hh_refused.jsonl", refused)  # type: ignore[operator]
    _write_jsonl(fold_dir / "oasst_benign.jsonl", oasst)  # type: ignore[operator]
    entries = load_fold_a_calibration(fold_a_dir=fold_dir)  # type: ignore[arg-type]
    assert len(entries) == 10


def test_load_fold_a_skips_blank_lines(tmp_path: pytest.TempPathFactory) -> None:
    fold_dir = tmp_path / "fold_a"  # type: ignore[operator]
    fold_dir.mkdir()  # type: ignore[union-attr]
    path = fold_dir / "anthropic_hh_benign.jsonl"  # type: ignore[operator]
    path.write_text(  # type: ignore[union-attr]
        json.dumps({"prompt": "A", "source": "hh"}) + "\n\n"
        + json.dumps({"prompt": "B", "source": "hh"}) + "\n",
        encoding="utf-8",
    )
    entries = load_fold_a_calibration(fold_a_dir=fold_dir)  # type: ignore[arg-type]
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# fold_isolation_check
# ---------------------------------------------------------------------------


def _make_entries(prompts: list[str], fold: Fold) -> list[FoldEntry]:
    return [
        FoldEntry(
            prompt=p,
            label="benign",
            source="test",
            fold=fold,
            sha256=_sha256(p),
        )
        for p in prompts
    ]


def test_fold_isolation_check_passes_disjoint_folds() -> None:
    fold_entries = {
        Fold.A: _make_entries(["prompt-a-1", "prompt-a-2"], Fold.A),
        Fold.B: _make_entries(["prompt-b-1", "prompt-b-2"], Fold.B),
    }
    result = fold_isolation_check(fold_entries)
    assert Fold.A in result
    assert Fold.B in result


def test_fold_isolation_check_returns_hash_sets() -> None:
    fold_entries = {
        Fold.A: _make_entries(["hello", "world"], Fold.A),
    }
    result = fold_isolation_check(fold_entries)
    assert result[Fold.A] == {_sha256("hello"), _sha256("world")}


def test_fold_isolation_check_raises_on_overlap() -> None:
    shared = "this prompt appears in both folds"
    fold_entries = {
        Fold.A: _make_entries([shared, "unique-a"], Fold.A),
        Fold.B: _make_entries([shared, "unique-b"], Fold.B),
    }
    with pytest.raises(AssertionError, match="isolation violation"):
        fold_isolation_check(fold_entries)


def test_fold_isolation_check_empty_folds_pass() -> None:
    result = fold_isolation_check({})
    assert result == {}


def test_fold_isolation_check_single_fold_always_passes() -> None:
    fold_entries = {Fold.A: _make_entries(["x", "y", "z"], Fold.A)}
    result = fold_isolation_check(fold_entries)
    assert len(result[Fold.A]) == 3
