"""Which schemes sit on the bit-width axis, which are merely present, and why.

The RTN ladder is an ordinal dose: one quantizer family, one group size, and
bit-width as the only thing that varies. A deployed AWQ or GPTQ checkpoint has a
known bit budget and no place on that axis, because it changes the quantization
algorithm at the same time -- a line drawn through both is not a dose-response
curve.

Both halves of that distinction were broken at once, in opposite directions:

  * `bits_of` accepted any `<word>_<digits>B` name, so `AWQ_4B` was placed at
    4.5 bits, one rung of the RTN ladder, and entered the drift fit.
  * `ordered_schemes` kept only names starting with `RTN_`, so the same
    deployed schemes were dropped from every table the analysis produced.

So the run generated the completions, the judge never graded them, no table
showed them, and if any of that had worked the result would have been folded
into a regression it does not belong in. These tests hold both edges.
"""

from __future__ import annotations

import json
import math

from scripts.build_paper_data import bits_of
from scripts.reanalyse_runs import ordered_schemes, select_runs


# ---------------------------------------------------------------------------
# the ordinal axis
# ---------------------------------------------------------------------------


def test_rtn_rungs_carry_their_group_overhead() -> None:
    assert bits_of("RTN_4B") == 4.5           # 4 code bits + 32/64 for scale+zero
    assert bits_of("RTN_2B") == 2.5
    assert bits_of("FP16") == 16.0


def test_salience_ladder_shares_the_axis() -> None:
    """SAL_* is the same code path at the same bit budget, run as its own
    ladder. It is an ordinal dose within itself, so it keeps a position."""
    assert bits_of("SAL_4B") == 4.5


def test_deployed_checkpoints_have_no_position_on_the_axis() -> None:
    """AWQ_4B is a 4-bit checkpoint and still not 4.5 RTN bits.

    This is the assertion that keeps a different quantizer out of the drift
    slope. Before it, the name alone was enough to earn a rung.
    """
    for scheme in ("AWQ_4B", "GPTQ_4B", "NF4", "GGUF_Q4_K_M"):
        assert math.isnan(bits_of(scheme)), scheme


def test_group_size_changes_the_position() -> None:
    assert bits_of("RTN_4B", group=128) == 4.25


# ---------------------------------------------------------------------------
# scheme ordering
# ---------------------------------------------------------------------------


def test_ladder_is_ordered_fp16_then_descending_bits() -> None:
    completions = {"RTN_2B": [], "FP16": [], "RTN_8B": [], "RTN_4B": []}
    assert ordered_schemes(completions) == ["FP16", "RTN_8B", "RTN_4B", "RTN_2B"]


def test_deployed_schemes_appear_after_the_ladder() -> None:
    """Present, so they can be graded and tabulated; last, because they are not
    a rung and must not read as one."""
    completions = {"FP16": [], "RTN_4B": [], "AWQ_4B": [], "GPTQ_4B": []}
    assert ordered_schemes(completions) == ["FP16", "RTN_4B", "AWQ_4B", "GPTQ_4B"]


def test_a_run_with_only_deployed_schemes_still_lists_them() -> None:
    """The --deployed run passes an empty --bits, which is exactly the case
    that used to reduce to ['FP16'] and grade nothing."""
    completions = {"FP16": [], "AWQ_4B": [], "GPTQ_4B": []}
    assert ordered_schemes(completions) == ["FP16", "AWQ_4B", "GPTQ_4B"]


# ---------------------------------------------------------------------------
# run selection
# ---------------------------------------------------------------------------


def _make_run(root, name):
    path = root / name
    (path / "results").mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"label": name}), encoding="utf-8")
    return path


def test_selection_defaults_to_every_run(tmp_path) -> None:
    for name in ("20260101-000000_abc_behavioural", "20260804-000000_abc_r2-behavioural"):
        _make_run(tmp_path, name)
    assert len(select_runs(tmp_path)) == 2


def test_include_and_exclude_split_the_two_rounds(tmp_path) -> None:
    """Two rounds share one artifacts tree. Every analysis downstream refuses to
    choose between two runs of the same model, so the caller has to say which
    round they mean -- and both directions must work."""
    _make_run(tmp_path, "20260101-000000_abc_behavioural-qwen3b")
    _make_run(tmp_path, "20260804-000000_abc_r2-long256-qwen3b")

    included = [p.name for p in select_runs(tmp_path, include="*r2-*")]
    excluded = [p.name for p in select_runs(tmp_path, exclude="*r2-*")]
    assert included == ["20260804-000000_abc_r2-long256-qwen3b"]
    assert excluded == ["20260101-000000_abc_behavioural-qwen3b"]


def test_selection_skips_loose_files(tmp_path) -> None:
    _make_run(tmp_path, "20260101-000000_abc_behavioural")
    (tmp_path / "INDEX.md").write_text("not a run", encoding="utf-8")
    (tmp_path / "ROUND2_STATUS.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in select_runs(tmp_path)] == ["20260101-000000_abc_behavioural"]


def test_selection_is_ordered_by_name(tmp_path) -> None:
    """Run ids begin with a UTC timestamp, so name order is chronological and
    'the last match' means 'the newest run'."""
    for name in ("20260804-120000_a_x", "20260101-000000_a_x", "20260601-000000_a_x"):
        _make_run(tmp_path, name)
    assert [p.name for p in select_runs(tmp_path)] == [
        "20260101-000000_a_x", "20260601-000000_a_x", "20260804-120000_a_x"]
