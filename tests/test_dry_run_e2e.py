"""End-to-end dry run tests — development.md Task 34."""

from cliffguard.types import QuantScheme, Tier

# Import via scripts path — the script is not a package module.
import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dry_run.py"
_spec = importlib.util.spec_from_file_location("dry_run", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_dry_run_mod = importlib.util.module_from_spec(_spec)
sys.modules["dry_run"] = _dry_run_mod
_spec.loader.exec_module(_dry_run_mod)  # type: ignore[union-attr]

run_dry_run = _dry_run_mod.run_dry_run
main = _dry_run_mod.main


_REQUIRED_KEYS = frozenset({
    "tier",
    "scheme",
    "black_box",
    "gates_run",
    "verdicts",
    "block_decision",
    "context_dim",
})

# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


def test_run_returns_dict_with_all_required_keys() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert _REQUIRED_KEYS.issubset(result.keys())


def test_run_context_dim_equals_14() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert result["context_dim"] == 14


def test_run_block_decision_is_bool() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert isinstance(result["block_decision"], bool)


def test_run_tier_stored_as_string() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert result["tier"] == "A"


def test_run_scheme_stored_as_string() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert result["scheme"] == "FP16"


def test_run_black_box_stored_correctly() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16, black_box=False)
    assert result["black_box"] is False


# ---------------------------------------------------------------------------
# Tier A gate coverage
# ---------------------------------------------------------------------------


def test_tier_a_gates_run_contains_vestibule_lz() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert "VESTIBULE-LZ" in result["gates_run"]


def test_tier_a_gates_run_contains_vestibule_ps() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert "VESTIBULE-PS" in result["gates_run"]


def test_tier_a_gates_run_contains_tripwire_h() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert "TRIPWIRE-H" in result["gates_run"]


def test_tier_a_gates_run_contains_tripwire_r() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert "TRIPWIRE-R" in result["gates_run"]


def test_tier_a_gates_run_contains_probe_rm() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert "PROBE-RM" in result["gates_run"]


def test_tier_a_gates_run_excludes_lookout_jg() -> None:
    # LOOKOUT-JG requires LLM — skipped in dry run.
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert "LOOKOUT-JG" not in result["gates_run"]


def test_tier_a_gates_run_is_nonempty() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert len(result["gates_run"]) > 0


# ---------------------------------------------------------------------------
# Tier C gate set (narrow scope)
# ---------------------------------------------------------------------------

_TIER_C_EXPECTED = {"VESTIBULE-LZ", "VESTIBULE-PS", "ATTEST-WH"}


def test_tier_c_gates_run_contains_exactly_tier_c_gates() -> None:
    result = run_dry_run(Tier.C, QuantScheme.GGUF_Q3_K_M)
    assert set(result["gates_run"]) == _TIER_C_EXPECTED


def test_tier_c_gates_run_length_is_three() -> None:
    result = run_dry_run(Tier.C, QuantScheme.GGUF_Q3_K_M)
    assert len(result["gates_run"]) == 3


def test_tier_c_gates_run_excludes_probe_rm() -> None:
    result = run_dry_run(Tier.C, QuantScheme.GGUF_Q3_K_M)
    assert "PROBE-RM" not in result["gates_run"]


# ---------------------------------------------------------------------------
# Black-box mode
# ---------------------------------------------------------------------------


def test_black_box_true_excludes_probe_rm() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16, black_box=True)
    assert "PROBE-RM" not in result["gates_run"]


def test_black_box_true_excludes_probe_mt() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16, black_box=True)
    assert "PROBE-MT" not in result["gates_run"]


def test_black_box_true_excludes_probe_hd() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16, black_box=True)
    assert "PROBE-HD" not in result["gates_run"]


def test_black_box_false_includes_probe_rm_for_tier_a() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16, black_box=False)
    assert "PROBE-RM" in result["gates_run"]


# ---------------------------------------------------------------------------
# Verdict list consistency
# ---------------------------------------------------------------------------


def test_verdicts_length_equals_gates_run_length_tier_a() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    assert len(result["verdicts"]) == len(result["gates_run"])


def test_verdicts_length_equals_gates_run_length_tier_c() -> None:
    result = run_dry_run(Tier.C, QuantScheme.GGUF_Q3_K_M)
    assert len(result["verdicts"]) == len(result["gates_run"])


def test_verdicts_length_equals_gates_run_length_black_box() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16, black_box=True)
    assert len(result["verdicts"]) == len(result["gates_run"])


def test_each_verdict_has_required_fields() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    for vd in result["verdicts"]:
        assert "gate" in vd
        assert "fired" in vd
        assert "score" in vd
        assert "threshold" in vd


def test_verdict_fired_is_bool() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    for vd in result["verdicts"]:
        assert isinstance(vd["fired"], bool)


def test_verdict_gate_names_match_gates_run() -> None:
    result = run_dry_run(Tier.A, QuantScheme.FP16)
    verdict_gates = [vd["gate"] for vd in result["verdicts"]]
    assert verdict_gates == result["gates_run"]


# ---------------------------------------------------------------------------
# main() exit codes
# ---------------------------------------------------------------------------


def test_main_tier_a_fp16_returns_0() -> None:
    assert main(["--tier", "A", "--scheme", "FP16"]) == 0


def test_main_tier_c_gguf_returns_0() -> None:
    assert main(["--tier", "C", "--scheme", "GGUF_Q3_K_M"]) == 0


def test_main_tier_b_nf4_returns_0() -> None:
    assert main(["--tier", "B", "--scheme", "NF4"]) == 0


def test_main_black_box_flag_returns_0() -> None:
    assert main(["--tier", "A", "--scheme", "FP16", "--black-box"]) == 0


def test_main_invalid_scheme_returns_1() -> None:
    assert main(["--tier", "A", "--scheme", "NOT_A_SCHEME"]) == 1
