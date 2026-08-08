"""The round-3 table, and the statistic the paper actually argues from.

The manuscript's cross-family claim rests on two numbers computed here: that the
phrase list's precision against the judge is 1.000 on every model, and that what
it misses varies by family. Precision is the load-bearing one -- recall alone
would presuppose the judge is the standard, while precision 1.000 says the two
label sets NEST, which is true whichever scorer one believes.

The run-dependent checks skip on a clean checkout, where `artifacts/runs/` is
absent by design; the arithmetic ones always run.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_labelled_tables import RUNS, measure, table

STATS = REPO / "docs" / "paper" / "labelled_paper_stats.json"
present = all((REPO / "artifacts/runs" / r / "results").exists()
              for r in RUNS.values())
needs_runs = pytest.mark.skipif(
    not present, reason="the labelled runs are not in this checkout")


@needs_runs
@pytest.mark.parametrize("model", list(RUNS))
def test_the_two_label_sets_nest(model: str) -> None:
    """Precision 1.000: the list never flags what the judge calls compliance.

    If this ever fails, the paper's phrasing must change -- the disagreement
    would no longer be one-directional and "the list misses" would become "the
    two scorers contradict each other", a different and weaker claim.
    """
    m = measure(REPO / "artifacts/runs" / RUNS[model] / "results")
    assert m["precision"] == 1.0, (
        f"{model}: {m['marker'] - int(m['recall'] * m['declines'])} flagged "
        "completions fall outside the judge's declining classes")


@needs_runs
@pytest.mark.parametrize(
    "model,recall,missed",
    [("Qwen2.5-3B", 0.493, 141),
     ("SmolLM2-1.7B", 0.235, 212),
     ("Phi-3.5-mini", 0.036, 271)])
def test_the_shortfall_is_family_dependent(model: str, recall: float,
                                           missed: int) -> None:
    m = measure(REPO / "artifacts/runs" / RUNS[model] / "results")
    assert round(m["recall"], 3) == recall
    assert m["missed"] == missed


@needs_runs
def test_the_spread_across_families_is_about_fourteenfold() -> None:
    """The number the subsection's headline claim quotes."""
    recalls = [measure(REPO / "artifacts/runs" / r / "results")["recall"]
               for r in RUNS.values()]
    assert round(max(recalls) / min(recalls)) == 14


@needs_runs
@pytest.mark.parametrize("model", list(RUNS))
def test_provenance_travels_with_the_numbers(model: str) -> None:
    """A stats file that names no run cannot be traced back to one."""
    m = measure(REPO / "artifacts/runs" / RUNS[model] / "results")
    assert m["run"] == RUNS[model]
    assert m["max_new_tokens"] == 48
    assert m["n_prompts"] == 300
    assert m["judge_loaded_in_4bit"] is True
    assert len(m["schemes"]) == 8


@needs_runs
def test_the_harmful_compliance_cell_is_empty_everywhere() -> None:
    """The claim the whole subsection turns on."""
    for model, run in RUNS.items():
        m = measure(REPO / "artifacts/runs" / run / "results")
        assert m["harmful_comply"] == 0, f"{model} complied on a harmful prompt"


def test_the_table_marks_the_empty_safety_cell() -> None:
    """The reader is meant to stop at that column, so it is bold in the source.

    Planted counts, so this tests the rendering and not the run.
    """
    rows = {"Fake-1B": {"marker": 10, "declines": 100, "recall": 0.1,
                        "harmful_deflect": 90, "harmful_comply": 0,
                        "benign_deflect": 80, "benign_comply": 20}}
    out = table(rows)
    assert r"\textbf{0}" in out
    assert "Fake-1B & 10 & 100 & 10.0 &" in out
    assert out.startswith(r"\begin{tabular}") and out.rstrip().endswith(
        r"\end{tabular}")


def test_a_model_with_no_declines_reports_no_recall() -> None:
    """`--` rather than a division by zero or a misleading 0.0."""
    rows = {"Fake-1B": {"marker": 0, "declines": 0, "recall": None,
                        "harmful_deflect": 0, "harmful_comply": 0,
                        "benign_deflect": 0, "benign_comply": 0}}
    assert "& -- &" in table(rows)


@pytest.mark.skipif(not STATS.exists(), reason="round-3 stats not present")
def test_the_stats_file_matches_what_measure_computes() -> None:
    """The file the manuscript is checked against must not go stale."""
    stored = json.loads(STATS.read_text(encoding="utf-8"))
    if not present:
        pytest.skip("runs absent; cannot recompute")
    for model, run in RUNS.items():
        fresh = measure(REPO / "artifacts/runs" / run / "results")
        # `at_cap_by_scheme` is attached by the builder rather than by
        # `measure`, because it needs a tokenizer download; compare what
        # `measure` is responsible for, and assert the rest is merely present.
        assert {k: stored[model][k] for k in fresh} == fresh, (
            f"{model}: labelled_paper_stats.json is stale — "
            "re-run scripts/build_labelled_tables.py")
        assert "at_cap_by_scheme" in stored[model], (
            f"{model}: truncation rates missing — the figure and the prose "
            "both read them, so a stale file would desynchronise them")
