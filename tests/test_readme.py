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


def test_readme_names_the_project() -> None:
    assert "CliffGuard" in README.read_text(encoding="utf-8")


def test_readme_leads_with_the_finding() -> None:
    """The headline is the direction of the effect, not the tooling."""
    text = README.read_text(encoding="utf-8")
    assert "more" in text and "refusal" in text.lower()
    assert "1.15" in text, "the drift coefficient should be quotable from the README"


def test_readme_states_the_transition_ceiling_with_its_bound() -> None:
    """A rate without its upper bound invites reading it as an equivalence."""
    text = README.read_text(encoding="utf-8")
    assert "2.2%" in text
    assert "4.6%" in text, "the simultaneous upper bound must appear beside the rate"


def test_readme_documents_all_six_protocol_steps() -> None:
    text = README.read_text(encoding="utf-8")
    for step in ("Pair", "Gate", "Grade", "Transition",
                 "Stress the scorer", "Separate capability from fluency"):
        assert step in text, f"protocol step missing from README: {step}"


def test_readme_explains_why_perplexity_alone_fails() -> None:
    """The 2.5-bit rung being MORE predictable than 3.5 is the whole reason the
    gate needs surface statistics; a reader who misses it will reimplement the
    broken version."""
    text = README.read_text(encoding="utf-8")
    assert "4.13" in text and "5.80" in text


def test_readme_carries_the_negative_results() -> None:
    """A README that lists only what the work establishes is a sales document."""
    text = README.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "does not establish" in lowered
    for gap in ("unvalidated", "48 tokens", "greedy"):
        assert gap in lowered, f"limitation missing from README: {gap}"


def test_readme_avoids_the_unlicensed_safety_vocabulary() -> None:
    """No harmfulness labels exist for these prompts, so the README must not
    claim harmful compliance was measured."""
    text = README.read_text(encoding="utf-8").lower()
    assert "harmful compliance" not in text.replace(
        'never "harmful compliance"', "")


def test_readme_points_at_the_paper_and_the_claims_ledger() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/paper/cliff_artifact.pdf" in text
    assert "docs/claims_and_evidence.md" in text


def test_readme_reproduction_commands_name_real_scripts() -> None:
    text = README.read_text(encoding="utf-8")
    for script in ("build_paper_data.py", "review_reanalysis.py",
                   "build_paper_figures.py", "build_paper_tables.py"):
        assert script in text, f"README omits {script}"
        assert (REPO_ROOT / "scripts" / script).exists(), (
            f"README advertises scripts/{script}, which does not exist")


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
