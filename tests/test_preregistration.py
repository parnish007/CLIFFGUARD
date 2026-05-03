from pathlib import Path

_DOC = Path(__file__).parent.parent / "docs" / "preregistration.md"


def _text() -> str:
    return _DOC.read_text(encoding="utf-8")


def test_preregistration_file_exists() -> None:
    assert _DOC.exists(), "docs/preregistration.md not found"


def test_preregistration_contains_kappa_threshold() -> None:
    assert "kappa >= 0.25" in _text()


def test_preregistration_contains_epsilon_threshold() -> None:
    assert "epsilon = 0.02" in _text()


def test_preregistration_contains_significance_level() -> None:
    assert "p < 0.05" in _text()


def test_preregistration_contains_bonferroni() -> None:
    assert "Bonferroni" in _text()


def test_preregistration_contains_h1() -> None:
    assert "H1" in _text()


def test_preregistration_contains_h2() -> None:
    assert "H2" in _text()


def test_preregistration_contains_h3() -> None:
    assert "H3" in _text()


def test_preregistration_contains_h4() -> None:
    assert "H4" in _text()


def test_preregistration_contains_h5() -> None:
    assert "H5" in _text()


def test_preregistration_status_line() -> None:
    assert "Pre-registered" in _text()
