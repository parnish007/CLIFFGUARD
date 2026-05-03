import json

import pytest

from cliffguard.types import QuantScheme
from cliffguard.eval.calibration import (
    load_fold_a,
    load_jsonl,
    make_synthetic_corpus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: object, records: list[dict]) -> None:  # type: ignore[type-arg]
    from pathlib import Path
    p = Path(str(path))  # type: ignore[arg-type]
    p.write_text(
        "\n".join(json.dumps(r) for r in records),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------


def test_load_jsonl_raises_for_nonexistent_path(tmp_path: pytest.TempPathFactory) -> None:
    missing = tmp_path / "no_file.jsonl"  # type: ignore[operator]
    with pytest.raises(FileNotFoundError):
        load_jsonl(missing)  # type: ignore[arg-type]


def test_load_jsonl_raises_for_invalid_json(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "bad.jsonl"  # type: ignore[operator]
    p.write_text('{"text": "ok"}\nnot json\n', encoding="utf-8")  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl(p)  # type: ignore[arg-type]


def test_load_jsonl_raises_for_missing_text_field(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "no_field.jsonl"  # type: ignore[operator]
    p.write_text('{"other": "value"}\n', encoding="utf-8")  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="Missing field"):
        load_jsonl(p)  # type: ignore[arg-type]


def test_load_jsonl_returns_correct_list(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "good.jsonl"  # type: ignore[operator]
    records = [{"text": "hello"}, {"text": "world"}, {"text": "foo"}]
    _write_jsonl(p, records)
    result = load_jsonl(p)  # type: ignore[arg-type]
    assert result == ["hello", "world", "foo"]


def test_load_jsonl_custom_text_field(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "custom.jsonl"  # type: ignore[operator]
    records = [{"prompt": "ask me anything"}]
    _write_jsonl(p, records)
    result = load_jsonl(p, text_field="prompt")  # type: ignore[arg-type]
    assert result == ["ask me anything"]


def test_load_jsonl_skips_blank_lines(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "blanks.jsonl"  # type: ignore[operator]
    p.write_text('{"text": "a"}\n\n{"text": "b"}\n', encoding="utf-8")  # type: ignore[union-attr]
    assert load_jsonl(p) == ["a", "b"]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# load_fold_a
# ---------------------------------------------------------------------------


def _make_corpus_file(path: object, n: int) -> None:  # type: ignore[type-arg]
    from pathlib import Path
    records = [{"text": f"prompt {i}"} for i in range(n)]
    _write_jsonl(Path(str(path)), records)  # type: ignore[arg-type]


def test_load_fold_a_raises_when_no_file_exists(tmp_path: pytest.TempPathFactory) -> None:
    with pytest.raises(FileNotFoundError):
        load_fold_a(tmp_path, QuantScheme.FP16, min_size=1)  # type: ignore[arg-type]


def test_load_fold_a_raises_when_corpus_too_small(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "fold_a.jsonl"  # type: ignore[operator]
    _make_corpus_file(p, n=3)
    with pytest.raises(ValueError, match="§14.4"):
        load_fold_a(tmp_path, QuantScheme.FP16, min_size=5)  # type: ignore[arg-type]


def test_load_fold_a_raises_mentions_actual_vs_required(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "fold_a.jsonl"  # type: ignore[operator]
    _make_corpus_file(p, n=3)
    with pytest.raises(ValueError) as exc_info:
        load_fold_a(tmp_path, QuantScheme.FP16, min_size=5)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "3" in msg and "5" in msg


def test_load_fold_a_returns_corpus_when_size_met(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "fold_a.jsonl"  # type: ignore[operator]
    _make_corpus_file(p, n=10)
    corpus = load_fold_a(tmp_path, QuantScheme.FP16, min_size=5)  # type: ignore[arg-type]
    assert len(corpus) == 10


def test_load_fold_a_uses_scheme_specific_file(tmp_path: pytest.TempPathFactory) -> None:
    scheme_file = tmp_path / "fold_a_fp16.jsonl"  # type: ignore[operator]
    fallback = tmp_path / "fold_a.jsonl"  # type: ignore[operator]
    _make_corpus_file(scheme_file, n=6)
    _make_corpus_file(fallback, n=6)
    # scheme-specific file has distinct content
    scheme_file.write_text(  # type: ignore[union-attr]
        "\n".join(json.dumps({"text": f"scheme-specific {i}"}) for i in range(6)),
        encoding="utf-8",
    )
    corpus = load_fold_a(tmp_path, QuantScheme.FP16, min_size=5)  # type: ignore[arg-type]
    assert all("scheme-specific" in p for p in corpus)


def test_load_fold_a_falls_back_to_generic_jsonl(tmp_path: pytest.TempPathFactory) -> None:
    fallback = tmp_path / "fold_a.jsonl"  # type: ignore[operator]
    _make_corpus_file(fallback, n=7)
    # No scheme-specific file → must use fallback
    corpus = load_fold_a(tmp_path, QuantScheme.FP16, min_size=5)  # type: ignore[arg-type]
    assert len(corpus) == 7


# ---------------------------------------------------------------------------
# make_synthetic_corpus
# ---------------------------------------------------------------------------


def test_make_synthetic_corpus_returns_correct_length() -> None:
    assert len(make_synthetic_corpus(n=50)) == 50


def test_make_synthetic_corpus_is_deterministic() -> None:
    assert make_synthetic_corpus(n=20, seed=7) == make_synthetic_corpus(n=20, seed=7)


def test_make_synthetic_corpus_different_seeds_differ() -> None:
    assert make_synthetic_corpus(n=20, seed=1) != make_synthetic_corpus(n=20, seed=2)


def test_make_synthetic_corpus_n_zero_returns_empty() -> None:
    assert make_synthetic_corpus(n=0) == []


def test_make_synthetic_corpus_returns_strings() -> None:
    corpus = make_synthetic_corpus(n=10)
    assert all(isinstance(p, str) for p in corpus)


def test_make_synthetic_corpus_default_length() -> None:
    assert len(make_synthetic_corpus()) == 200
