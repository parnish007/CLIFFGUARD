import warnings

import pytest

from cliffguard.eval.kenlm_trainer import (
    KENLM_DEFAULT_ORDER,
    assemble_corpus,
    binarise_arpa,
    estimate_arpa_size_mb,
    train_and_save,
    train_kenlm,
)


# ---------------------------------------------------------------------------
# Minimal stub for fold_entries — any object with .prompt attribute
# ---------------------------------------------------------------------------


class _Entry:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt


def _entries(n: int) -> list[_Entry]:
    return [_Entry(f"Prompt number {i}") for i in range(n)]


# ---------------------------------------------------------------------------
# assemble_corpus
# ---------------------------------------------------------------------------


def test_assemble_corpus_writes_correct_line_count(tmp_path: pytest.TempPathFactory) -> None:
    out = tmp_path / "corpus.txt"  # type: ignore[operator]
    n = assemble_corpus(_entries(7), out)  # type: ignore[arg-type]
    assert n == 7
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln]  # type: ignore[union-attr]
    assert len(lines) == 7


def test_assemble_corpus_content_matches_prompts(tmp_path: pytest.TempPathFactory) -> None:
    entries = [_Entry("hello world"), _Entry("foo bar")]
    out = tmp_path / "corpus.txt"  # type: ignore[operator]
    assemble_corpus(entries, out)  # type: ignore[arg-type]
    lines = out.read_text(encoding="utf-8").splitlines()  # type: ignore[union-attr]
    assert lines[0] == "hello world"
    assert lines[1] == "foo bar"


def test_assemble_corpus_raises_for_empty(tmp_path: pytest.TempPathFactory) -> None:
    out = tmp_path / "corpus.txt"  # type: ignore[operator]
    with pytest.raises(ValueError):
        assemble_corpus([], out)  # type: ignore[arg-type]


def test_assemble_corpus_creates_parent_directories(tmp_path: pytest.TempPathFactory) -> None:
    out = tmp_path / "nested" / "deep" / "corpus.txt"  # type: ignore[operator]
    assemble_corpus(_entries(3), out)  # type: ignore[arg-type]
    assert out.exists()  # type: ignore[union-attr]


def test_assemble_corpus_returns_count(tmp_path: pytest.TempPathFactory) -> None:
    out = tmp_path / "corpus.txt"  # type: ignore[operator]
    n = assemble_corpus(_entries(10), out)  # type: ignore[arg-type]
    assert n == 10


# ---------------------------------------------------------------------------
# estimate_arpa_size_mb
# ---------------------------------------------------------------------------


def test_estimate_arpa_size_mb_returns_positive_float() -> None:
    result = estimate_arpa_size_mb(2000)
    assert isinstance(result, float)
    assert result > 0.0


def test_estimate_arpa_size_mb_no_warning_for_small_n() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = estimate_arpa_size_mb(100)
    assert result > 0.0


def test_estimate_arpa_size_mb_warns_when_exceeds_50mb() -> None:
    # With order=3, avg_tokens=15: need n such that n*15*3*15/1e6 > 50
    # n > 50*1024*1024 / (15*3*15) ≈ 77672
    large_n = 100_000
    with pytest.warns(UserWarning, match="50 MB"):
        estimate_arpa_size_mb(large_n)


def test_estimate_arpa_size_mb_scales_linearly() -> None:
    s1 = estimate_arpa_size_mb(1000)
    s2 = estimate_arpa_size_mb(2000)
    assert s2 == pytest.approx(2 * s1, rel=1e-9)


def test_estimate_arpa_size_mb_uses_order_parameter() -> None:
    s3 = estimate_arpa_size_mb(1000, order=3)
    s5 = estimate_arpa_size_mb(1000, order=5)
    assert s5 == pytest.approx(s3 * 5 / 3, rel=1e-9)


def test_estimate_arpa_size_mb_default_order_is_kenlm_default() -> None:
    s_default = estimate_arpa_size_mb(1000)
    s_explicit = estimate_arpa_size_mb(1000, order=KENLM_DEFAULT_ORDER)
    assert s_default == pytest.approx(s_explicit, rel=1e-9)


# ---------------------------------------------------------------------------
# train_kenlm — Phase A: NotImplementedError
# ---------------------------------------------------------------------------


def test_train_kenlm_raises_not_implemented(tmp_path: pytest.TempPathFactory) -> None:
    corpus = tmp_path / "corpus.txt"  # type: ignore[operator]
    arpa = tmp_path / "out.arpa"  # type: ignore[operator]
    with pytest.raises(NotImplementedError):
        train_kenlm(corpus, arpa)  # type: ignore[arg-type]


def test_train_kenlm_error_message_contains_kenlm_url(tmp_path: pytest.TempPathFactory) -> None:
    corpus = tmp_path / "corpus.txt"  # type: ignore[operator]
    arpa = tmp_path / "out.arpa"  # type: ignore[operator]
    with pytest.raises(NotImplementedError, match="https://github.com/kpu/kenlm"):
        train_kenlm(corpus, arpa)  # type: ignore[arg-type]


def test_train_kenlm_error_message_mentions_lmplz(tmp_path: pytest.TempPathFactory) -> None:
    corpus = tmp_path / "corpus.txt"  # type: ignore[operator]
    arpa = tmp_path / "out.arpa"  # type: ignore[operator]
    with pytest.raises(NotImplementedError, match="lmplz"):
        train_kenlm(corpus, arpa)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# binarise_arpa — Phase A: NotImplementedError
# ---------------------------------------------------------------------------


def test_binarise_arpa_raises_not_implemented(tmp_path: pytest.TempPathFactory) -> None:
    arpa = tmp_path / "model.arpa"  # type: ignore[operator]
    klm = tmp_path / "model.klm"  # type: ignore[operator]
    with pytest.raises(NotImplementedError):
        binarise_arpa(arpa, klm)  # type: ignore[arg-type]


def test_binarise_arpa_error_message_contains_kenlm_url(tmp_path: pytest.TempPathFactory) -> None:
    arpa = tmp_path / "model.arpa"  # type: ignore[operator]
    klm = tmp_path / "model.klm"  # type: ignore[operator]
    with pytest.raises(NotImplementedError, match="https://github.com/kpu/kenlm"):
        binarise_arpa(arpa, klm)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# train_and_save — raises NotImplementedError but computes size estimate first
# ---------------------------------------------------------------------------


def test_train_and_save_raises_not_implemented(tmp_path: pytest.TempPathFactory) -> None:
    with pytest.raises(NotImplementedError):
        train_and_save(_entries(5), tmp_path, "nf4")  # type: ignore[arg-type]


def test_train_and_save_assembles_corpus_before_raising(tmp_path: pytest.TempPathFactory) -> None:
    # assemble_corpus runs before train_kenlm, so the corpus file should exist
    with pytest.raises(NotImplementedError):
        train_and_save(_entries(4), tmp_path, "q4_k_m")  # type: ignore[arg-type]
    corpus_path = tmp_path / "fold_a_q4_k_m.txt"
    assert corpus_path.exists()


def test_train_and_save_computes_size_estimate_for_large_n(
    tmp_path: pytest.TempPathFactory,
) -> None:
    # A large corpus triggers UserWarning from estimate_arpa_size_mb even in Phase A
    large_entries = _entries(100_000)
    with pytest.warns(UserWarning, match="50 MB"):
        with pytest.raises(NotImplementedError):
            train_and_save(large_entries, tmp_path, "q3_k_m")  # type: ignore[arg-type]
