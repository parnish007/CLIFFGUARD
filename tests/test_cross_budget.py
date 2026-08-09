"""The cross-budget comparison must test prompts, not totals.

The truncation question is whether a result measured at 48 generated tokens
survives at 256. The analysis used to answer it by taking each run's own
full-precision-versus-rung imbalance and checking whether the two imbalances
kept their sign. That cannot see a prompt whose verdict changed -- only whether
two totals happened to agree -- and two totals can agree while every prompt
underneath them moves.

Worse, each budget defines an unsafe flip against ITS OWN full-precision label.
If widening the window changes the full-precision verdict, the two budgets stop
scoring the same event and a prompt can leave the long-window baseline with
nothing reported.

These tests use hand-built verdict lists where the right answer is known by
construction, because no 256-token run exists yet and a statistic that is only
exercised once the data arrives is a statistic nobody has checked.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def analysis():
    spec = importlib.util.spec_from_file_location(
        "_round2_under_test", REPO / "scripts" / "analyse_round2.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run(**schemes: list[str]) -> dict[str, object]:
    return {"judge_raw": dict(schemes)}


def test_identical_verdicts_report_no_movement() -> None:
    mod = analysis()
    verdicts = ["REFUSE", "COMPLY", "REFUSE", "UNCLEAR"]
    got = mod.per_prompt_budget(run(RTN_4B=verdicts), run(RTN_4B=verdicts),
                                "RTN_4B")
    assert got["changed"] == 0
    assert got["became_comply"] == 0 and got["left_comply"] == 0


def test_aggregates_can_agree_while_every_prompt_moves() -> None:
    """The exact blind spot of the old sign comparison.

    Two verdict lists with the same totals, on completely different prompts.
    A comparison of aggregates sees nothing; a per-prompt one sees four
    changes.
    """
    mod = analysis()
    short = ["REFUSE", "REFUSE", "COMPLY", "COMPLY"]
    long = ["COMPLY", "COMPLY", "REFUSE", "REFUSE"]
    got = mod.per_prompt_budget(run(RTN_4B=short), run(RTN_4B=long), "RTN_4B")
    assert got["changed"] == 4, "a per-prompt test must see all four moves"
    assert got["became_comply"] == 2 and got["left_comply"] == 2


def test_a_window_that_reveals_compliance_is_detected() -> None:
    """Three prompts that read as refusal at 48 tokens and comply at 256."""
    mod = analysis()
    short = ["REFUSE"] * 10
    long = ["COMPLY", "COMPLY", "COMPLY"] + ["REFUSE"] * 7
    got = mod.per_prompt_budget(run(RTN_4B=short), run(RTN_4B=long), "RTN_4B")
    assert got["became_comply"] == 3 and got["left_comply"] == 0
    assert got["p"] < 0.30, "three one-directional changes should not read as noise"


def test_a_moving_baseline_is_reported_not_hidden() -> None:
    """If full precision itself moves, the flip comparison rests on nothing.

    Built so that every full-precision verdict changes with the budget. The
    function must say so rather than quietly compare flip statuses defined
    against two different baselines.
    """
    mod = analysis()
    short = run(FP16=["REFUSE"] * 10, RTN_4B=["COMPLY"] * 10)
    long = run(FP16=["COMPLY"] * 10, RTN_4B=["COMPLY"] * 10)
    got = mod.flip_status_budget(short, long, ["FP16", "RTN_4B"])
    assert got["fp16_verdict_changed"] == 10
    assert got["n_baseline_stable"] == 0
    assert "shifting ground" in got["note"]


def test_flip_status_is_compared_only_where_the_baseline_held() -> None:
    """Prompts whose baseline moved are excluded, and counted separately."""
    mod = analysis()
    # Prompt 0: baseline moves. Prompts 1-9: baseline stable, and prompt 1
    # stops being an unsafe flip once the window widens.
    short = run(FP16=["REFUSE"] * 10,
                RTN_4B=["COMPLY", "COMPLY"] + ["REFUSE"] * 8)
    long = run(FP16=["COMPLY"] + ["REFUSE"] * 9,
               RTN_4B=["COMPLY", "REFUSE"] + ["REFUSE"] * 8)
    got = mod.flip_status_budget(short, long, ["FP16", "RTN_4B"])
    assert got["fp16_verdict_changed"] == 1
    assert got["n_baseline_stable"] == 9
    assert got["flip_only_at_48"] == 1, "prompt 1 flips at 48 and not at 256"
    assert got["flip_only_at_256"] == 0


def test_mismatched_lengths_refuse_to_compare() -> None:
    """Two runs of different sizes are not paired, and must not be treated as."""
    mod = analysis()
    got = mod.per_prompt_budget(run(RTN_4B=["REFUSE"] * 5),
                                run(RTN_4B=["REFUSE"] * 6), "RTN_4B")
    assert "note" in got and "not comparable" in got["note"]
