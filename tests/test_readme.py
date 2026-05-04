"""Smoke tests that README.md and configs/example.yaml are well-formed."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
EXAMPLE_YAML = REPO_ROOT / "configs" / "example.yaml"


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------


def test_readme_exists() -> None:
    assert README.exists()


def test_readme_is_non_empty() -> None:
    assert README.stat().st_size > 0


def test_readme_contains_cliffguard() -> None:
    assert "CLIFFGUARD" in README.read_text(encoding="utf-8")


def test_readme_contains_h1() -> None:
    assert "H1" in README.read_text(encoding="utf-8")


def test_readme_contains_h2() -> None:
    assert "H2" in README.read_text(encoding="utf-8")


def test_readme_contains_h3() -> None:
    assert "H3" in README.read_text(encoding="utf-8")


def test_readme_contains_h4() -> None:
    assert "H4" in README.read_text(encoding="utf-8")


def test_readme_contains_h5() -> None:
    assert "H5" in README.read_text(encoding="utf-8")


def test_readme_contains_tier_a() -> None:
    assert "Tier A" in README.read_text(encoding="utf-8")


def test_readme_contains_tier_b() -> None:
    assert "Tier B" in README.read_text(encoding="utf-8")


def test_readme_contains_tier_c() -> None:
    assert "Tier C" in README.read_text(encoding="utf-8")


def test_readme_contains_tier_c_plus() -> None:
    assert "Tier C+" in README.read_text(encoding="utf-8")


def test_readme_contains_preregistration() -> None:
    assert "preregistration" in README.read_text(encoding="utf-8")


def test_readme_contains_bibtex_marker() -> None:
    assert "@article" in README.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# configs/example.yaml
# ---------------------------------------------------------------------------


def test_example_yaml_exists() -> None:
    assert EXAMPLE_YAML.exists()


def test_example_yaml_is_non_empty() -> None:
    assert EXAMPLE_YAML.stat().st_size > 0


def test_example_yaml_contains_fpr_target() -> None:
    assert "fpr_target" in EXAMPLE_YAML.read_text(encoding="utf-8")


def test_example_yaml_contains_kenlm() -> None:
    assert "kenlm" in EXAMPLE_YAML.read_text(encoding="utf-8")


def test_example_yaml_contains_order_tier_ab() -> None:
    assert "order_tier_ab" in EXAMPLE_YAML.read_text(encoding="utf-8")


def test_example_yaml_contains_tier_field() -> None:
    assert "tier:" in EXAMPLE_YAML.read_text(encoding="utf-8")


def test_example_yaml_contains_schemes_field() -> None:
    assert "schemes:" in EXAMPLE_YAML.read_text(encoding="utf-8")


def test_example_yaml_contains_folds_field() -> None:
    assert "folds:" in EXAMPLE_YAML.read_text(encoding="utf-8")
