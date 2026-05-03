import pytest

from cliffguard.eval.bcn2 import (
    BCN2Record,
    BoundaryPrompt,
    StubParaphraser,
    construct_bcn2,
    enforce_non_circularity,
    filter_boundary_prompts,
    load_bcn2,
    save_bcn2,
)
from cliffguard.types import QuantScheme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bp(
    text: str = "harmful request",
    fp16_refused: bool = True,
    fp16_margin: float = 0.1,
    source_family: str = "llama-3",
) -> BoundaryPrompt:
    return BoundaryPrompt(
        text=text,
        fp16_refused=fp16_refused,
        fp16_margin=fp16_margin,
        source_family=source_family,
    )


_STUB = StubParaphraser(family="qwen-2.5")
_SCHEME = QuantScheme.GGUF_Q3_K_M


# ---------------------------------------------------------------------------
# StubParaphraser
# ---------------------------------------------------------------------------


def test_stub_paraphraser_model_family_returns_constructor_value() -> None:
    stub = StubParaphraser(family="mistral-7b")
    assert stub.model_family == "mistral-7b"


def test_stub_paraphraser_default_family() -> None:
    stub = StubParaphraser()
    assert stub.model_family == "stub-family"


def test_stub_paraphraser_returns_list_of_length_n() -> None:
    stub = StubParaphraser()
    result = stub.paraphrase("hello", n=3)
    assert len(result) == 3


def test_stub_paraphraser_n_equals_1_returns_single_element() -> None:
    stub = StubParaphraser()
    result = stub.paraphrase("hello", n=1)
    assert len(result) == 1


def test_stub_paraphraser_output_contains_original_text() -> None:
    stub = StubParaphraser()
    result = stub.paraphrase("my text", n=2)
    assert all("my text" in r for r in result)


def test_stub_paraphraser_zero_n_returns_empty() -> None:
    stub = StubParaphraser()
    assert stub.paraphrase("anything", n=0) == []


# ---------------------------------------------------------------------------
# filter_boundary_prompts
# ---------------------------------------------------------------------------


def test_filter_returns_refused_prompts_in_margin_range() -> None:
    entries = [
        _bp(text="a", fp16_refused=True, fp16_margin=0.1),
        _bp(text="b", fp16_refused=True, fp16_margin=0.2),
    ]
    result = filter_boundary_prompts(entries)
    assert len(result) == 2


def test_filter_excludes_non_refused_prompts() -> None:
    entries = [
        _bp(text="refused", fp16_refused=True, fp16_margin=0.1),
        _bp(text="complied", fp16_refused=False, fp16_margin=0.1),
    ]
    result = filter_boundary_prompts(entries)
    assert len(result) == 1
    assert result[0].text == "refused"


def test_filter_excludes_prompts_below_margin_low() -> None:
    entries = [
        _bp(fp16_refused=True, fp16_margin=0.03),  # below 0.05
    ]
    result = filter_boundary_prompts(entries)
    assert result == []


def test_filter_excludes_prompts_above_margin_high() -> None:
    entries = [
        _bp(fp16_refused=True, fp16_margin=0.30),  # above 0.25
    ]
    result = filter_boundary_prompts(entries)
    assert result == []


def test_filter_includes_prompts_at_boundary_margins() -> None:
    entries = [
        _bp(fp16_refused=True, fp16_margin=0.05),  # exactly at low
        _bp(fp16_refused=True, fp16_margin=0.25),  # exactly at high
    ]
    result = filter_boundary_prompts(entries)
    assert len(result) == 2


def test_filter_raises_for_empty_input() -> None:
    with pytest.raises(ValueError):
        filter_boundary_prompts([])


def test_filter_returns_empty_list_when_no_entries_qualify() -> None:
    entries = [_bp(fp16_refused=False, fp16_margin=0.5)]
    result = filter_boundary_prompts(entries)
    assert result == []


# ---------------------------------------------------------------------------
# enforce_non_circularity
# ---------------------------------------------------------------------------


def test_enforce_raises_when_families_match() -> None:
    bp = _bp(source_family="llama-3")
    stub = StubParaphraser(family="llama-3")
    with pytest.raises(ValueError, match="[Cc]ircularity"):
        enforce_non_circularity(bp, stub)


def test_enforce_passes_when_families_differ() -> None:
    bp = _bp(source_family="llama-3")
    stub = StubParaphraser(family="qwen-2.5")
    enforce_non_circularity(bp, stub)  # should not raise


def test_enforce_error_message_names_both_families() -> None:
    bp = _bp(source_family="llama-3")
    stub = StubParaphraser(family="llama-3")
    with pytest.raises(ValueError) as exc_info:
        enforce_non_circularity(bp, stub)
    msg = str(exc_info.value)
    assert "llama-3" in msg


# ---------------------------------------------------------------------------
# construct_bcn2
# ---------------------------------------------------------------------------


def test_construct_raises_for_empty_boundary_prompts() -> None:
    with pytest.raises(ValueError):
        construct_bcn2([], _STUB, _SCHEME)


def test_construct_raises_via_enforce_when_family_matches() -> None:
    bp = _bp(source_family="qwen-2.5")
    stub = StubParaphraser(family="qwen-2.5")
    with pytest.raises(ValueError, match="[Cc]ircularity"):
        construct_bcn2([bp], stub, _SCHEME)


def test_construct_returns_n_paraphrases_per_prompt() -> None:
    bps = [_bp(text=f"prompt-{i}") for i in range(3)]
    records = construct_bcn2(bps, _STUB, _SCHEME, n_paraphrases=2)
    assert len(records) == 6  # 3 prompts × 2 paraphrases


def test_construct_n_paraphrases_1_returns_one_per_prompt() -> None:
    bps = [_bp()]
    records = construct_bcn2(bps, _STUB, _SCHEME, n_paraphrases=1)
    assert len(records) == 1


def test_construct_records_have_correct_scheme() -> None:
    records = construct_bcn2([_bp()], _STUB, _SCHEME)
    assert all(r.scheme == _SCHEME for r in records)


def test_construct_records_have_correct_paraphraser_family() -> None:
    records = construct_bcn2([_bp()], _STUB, _SCHEME)
    assert all(r.paraphraser_family == _STUB.model_family for r in records)


def test_construct_records_contain_original_prompt() -> None:
    bp = _bp(text="my test prompt")
    records = construct_bcn2([bp], _STUB, _SCHEME)
    assert all(r.original.text == "my test prompt" for r in records)


# ---------------------------------------------------------------------------
# save_bcn2 / load_bcn2 round-trip
# ---------------------------------------------------------------------------


def _sample_records(n: int = 2) -> list[BCN2Record]:
    bps = [_bp(text=f"test prompt {i}", fp16_margin=0.1 + i * 0.05) for i in range(n)]
    return construct_bcn2(bps, _STUB, _SCHEME, n_paraphrases=1)


def test_save_load_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "bcn2.jsonl"  # type: ignore[operator]
    records = _sample_records(3)
    save_bcn2(records, path)  # type: ignore[arg-type]
    loaded = load_bcn2(path)  # type: ignore[arg-type]
    assert len(loaded) == 3
    assert loaded[0].original.text == records[0].original.text
    assert loaded[0].paraphrase == records[0].paraphrase
    assert loaded[0].scheme == records[0].scheme
    assert loaded[0].paraphraser_family == records[0].paraphraser_family


def test_save_creates_parent_directories(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "nested" / "dir" / "bcn2.jsonl"  # type: ignore[operator]
    save_bcn2(_sample_records(), path)  # type: ignore[arg-type]
    assert path.exists()  # type: ignore[union-attr]


def test_load_raises_file_not_found(tmp_path: pytest.TempPathFactory) -> None:
    missing = tmp_path / "no_such_file.jsonl"  # type: ignore[operator]
    with pytest.raises(FileNotFoundError):
        load_bcn2(missing)  # type: ignore[arg-type]


def test_load_raises_for_malformed_json(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "bad.jsonl"  # type: ignore[operator]
    path.write_text("not valid json\n", encoding="utf-8")  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="[Mm]alformed"):
        load_bcn2(path)  # type: ignore[arg-type]


def test_load_raises_for_missing_required_field(tmp_path: pytest.TempPathFactory) -> None:
    import json as _json
    path = tmp_path / "missing_field.jsonl"  # type: ignore[operator]
    # Omit "scheme" field
    line = _json.dumps({
        "original_text": "test",
        "fp16_margin": 0.1,
        "source_family": "llama-3",
        "paraphrase": "test para",
        "paraphraser_family": "qwen",
        # scheme intentionally omitted
    })
    path.write_text(line + "\n", encoding="utf-8")  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        load_bcn2(path)  # type: ignore[arg-type]


def test_load_restores_fp16_margin(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "bcn2.jsonl"  # type: ignore[operator]
    bp = _bp(fp16_margin=0.17)
    records = construct_bcn2([bp], _STUB, _SCHEME)
    save_bcn2(records, path)  # type: ignore[arg-type]
    loaded = load_bcn2(path)  # type: ignore[arg-type]
    assert loaded[0].original.fp16_margin == pytest.approx(0.17)


def test_load_sets_fp16_refused_true(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "bcn2.jsonl"  # type: ignore[operator]
    save_bcn2(_sample_records(), path)  # type: ignore[arg-type]
    loaded = load_bcn2(path)  # type: ignore[arg-type]
    assert all(r.original.fp16_refused is True for r in loaded)
