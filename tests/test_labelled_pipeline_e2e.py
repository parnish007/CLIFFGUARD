"""Run the labelled pipeline's analysis scripts for real, on a synthesised run.

Unit tests cover the counting. What they cannot cover is the contract between
the run directory a GPU session writes and the analyses a later session reads:
file names, the manifest's `schemes` list, where the harmfulness labels live,
which gate the taxonomy applied. Every one of those has already broken this
project at least once, always by producing a plausible-looking answer rather
than an error -- a judge that graded nothing because `schemes` was reconstructed
as `["FP16"]`, an analysis that skipped every run because a key moved.

So this builds a run directory byte-for-byte like the real thing, with the four
matrix cells planted at known counts, and invokes `analyse_labelled.py` and
`analyse_matrix.py` as subprocesses -- the way the notebook does. If the scripts
and the layout drift apart, this fails on a laptop in two seconds instead of on
a GPU after an hour.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]

SCHEMES = ["FP16", "RTN_4B"]
N_HARMFUL = 30
N_BENIGN = 30
N = N_HARMFUL + N_BENIGN

# The counts this fixture plants, asserted below rather than recomputed.
SAFETY_LOST = 5          # harmful, FP16 refusal -> rung compliance
OVER_REFUSAL = 7         # benign,  FP16 compliance -> rung refusal
BENIGN_DEFLECTIONS = 4   # benign,  FP16 compliance -> rung deflection (broad only)

COMPLIANT_TEXT = "Sure. Here is how you would approach that, step by step: first, "
REFUSAL_TEXT = "I can't help with that request."
DEFLECT_TEXT = "That touches on a sensitive area. Consider speaking to a professional."


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A behavioural run with both axes, laid out exactly as the ladder writes it."""
    runs = tmp_path / "runs"
    run = runs / "20260101-000000_abc1234_lab-fake-xstest"
    results = run / "results"

    harm = ["harmful"] * N_HARMFUL + ["benign"] * N_BENIGN
    prompts = [f"prompt {i}" for i in range(N)]

    # FP16: refuses every harmful prompt, answers every benign one -- the
    # well-behaved baseline the regressions are measured against.
    fp16 = [REFUSAL_TEXT] * N_HARMFUL + [COMPLIANT_TEXT] * N_BENIGN
    fp16_tax = ["REFUSE"] * N_HARMFUL + ["COMPLY"] * N_BENIGN

    rung = list(fp16)
    rung_tax = list(fp16_tax)
    for i in range(SAFETY_LOST):                       # harmful: withheld -> answered
        rung[i], rung_tax[i] = COMPLIANT_TEXT, "COMPLY"
    for i in range(N_HARMFUL, N_HARMFUL + OVER_REFUSAL):
        rung[i], rung_tax[i] = REFUSAL_TEXT, "REFUSE"  # benign: answered -> refused
    start = N_HARMFUL + OVER_REFUSAL
    for i in range(start, start + BENIGN_DEFLECTIONS):
        rung[i], rung_tax[i] = DEFLECT_TEXT, "DEFLECT" # benign: answered -> deflected

    _write(run / "manifest.json", {
        "label": "lab-fake-xstest", "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "schemes": SCHEMES, "bits": [4], "n_prompts": N,
        "prompt_corpus_labelled": "data/eval_suites/xstest.jsonl",
        "prompt_corpus": "data/eval_suites/xstest.jsonl",
    })
    _write(results / "prompts.json", {"prompts": prompts, "harm_label": harm})
    _write(results / "completions_FP16.json", {"completions": fp16})
    _write(results / "completions_RTN_4B.json", {"completions": rung})
    # Every completion comfortably below any degeneracy threshold, so the gate is
    # inert here and the planted counts are what the analyses must report.
    _write(results / "completion_nll.json",
           {"FP16": [1.0] * N, "RTN_4B": [1.0] * N})
    # The three-way judge's per-scheme cache, named as the grader writes it.
    _write(results / "judge_deadbeefdeadbeef_FP16.json",
           ["REFUSE" if t == REFUSAL_TEXT else "COMPLY" for t in fp16])
    _write(results / "judge_deadbeefdeadbeef_RTN_4B.json",
           ["REFUSE" if t == REFUSAL_TEXT else "COMPLY" for t in rung])

    from scripts.classify_completion_taxonomy import CLASS_OF

    _write(results / "completion_taxonomy.json", {
        "judge_model": "Qwen/Qwen2.5-7B-Instruct",
        "schemes": SCHEMES,
        "labels": list(CLASS_OF),
        "class_of": CLASS_OF,
        "degeneracy_threshold": 5.0,
        "per_scheme": {s: {"margin_median": 4.0} for s in SCHEMES},
        "resolved_gate": "composite",
        "resolved": {"FP16": [CLASS_OF[v] for v in fp16_tax],
                     "RTN_4B": [CLASS_OF[v] for v in rung_tax]},
        "verdicts": {"FP16": fp16_tax, "RTN_4B": rung_tax},
        "margins": {s: [4.0] * N for s in SCHEMES},
    })
    return runs


def _invoke(script: str, runs: Path, out: Path) -> str:
    proc = subprocess.run(
        [sys.executable, f"scripts/{script}", "--runs", str(runs),
         "--min-n", "10", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def test_analyse_labelled_reads_the_run_and_finds_the_planted_cells(
        run_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "labelled.json"
    _invoke("analyse_labelled.py", run_dir, out)
    block = next(iter(json.loads(out.read_text()).values()))
    row = block["rows"][0]
    assert row["safety_lost"] == SAFETY_LOST
    assert row["over_refusal"] == OVER_REFUSAL
    assert block["baseline"]["harmful"]["refusal_rate"] == 1.0
    assert block["baseline"]["benign"]["refusal_rate"] == 0.0


def test_analyse_matrix_reads_both_axes_and_separates_the_readings(
        run_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "matrix.json"
    stdout = _invoke("analyse_matrix.py", run_dir, out)
    assert "SAFETY REGRESSION" in stdout and "OVER-REFUSAL" in stdout

    block = next(iter(json.loads(out.read_text()).values()))
    assert block["n_harmful"] == N_HARMFUL and block["n_benign"] == N_BENIGN

    strict = block["paired"]["strict"][0]
    broad = block["paired"]["broad"][0]
    assert strict["safety_lost"] == SAFETY_LOST
    assert strict["over_refusal"] == OVER_REFUSAL
    # The deflections are the whole point of having two readings: invisible to
    # the strict count, and counted by the broad one.
    assert broad["over_refusal"] == OVER_REFUSAL + BENIGN_DEFLECTIONS
    assert broad["safety_lost"] == SAFETY_LOST

    spread = block["reading_sensitivity"]["over_refusal"]["RTN_4B"]
    assert spread["by_reading"] == {"strict": OVER_REFUSAL,
                                    "broad": OVER_REFUSAL + BENIGN_DEFLECTIONS}
    assert spread["spread"] == BENIGN_DEFLECTIONS


def test_the_full_contingency_is_written_for_every_scheme(
        run_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "matrix.json"
    _invoke("analyse_matrix.py", run_dir, out)
    block = next(iter(json.loads(out.read_text()).values()))
    assert set(block["contingency"]) == set(SCHEMES)
    fp16 = block["contingency"]["FP16"]
    assert fp16["harmful"]["refusal"] == N_HARMFUL
    assert fp16["benign"]["compliance"] == N_BENIGN
    rung = block["contingency"]["RTN_4B"]
    assert rung["benign"]["deflection"] == BENIGN_DEFLECTIONS
    for prompt_class in ("harmful", "benign"):
        assert sum(rung[prompt_class].values()) == (
            N_HARMFUL if prompt_class == "harmful" else N_BENIGN)


def test_the_gate_is_re_applied_from_raw_verdicts_not_trusted(
        run_dir: Path, tmp_path: Path) -> None:
    """Which gate ran is a reported axis here, so it must stay the analysis's choice.

    The stored `resolved` array is deliberately falsified. If the analysis reads
    it instead of re-deriving from `verdicts`, the planted counts vanish and this
    fails -- which is the point: a grader that resolves the gate itself removes
    an axis the rest of the project crosses on purpose.
    """
    run = next(run_dir.iterdir())
    path = run / "results" / "completion_taxonomy.json"
    payload = json.loads(path.read_text())
    payload["resolved"] = {s: ["degenerate"] * N for s in SCHEMES}
    path.write_text(json.dumps(payload), encoding="utf-8")

    out = tmp_path / "matrix.json"
    _invoke("analyse_matrix.py", run_dir, out)
    block = next(iter(json.loads(out.read_text()).values()))
    assert block["gate"] == "composite"
    assert block["paired"]["strict"][0]["safety_lost"] == SAFETY_LOST


def test_an_old_run_cannot_be_reported_under_a_gate_it_never_ran(
        run_dir: Path, tmp_path: Path) -> None:
    """Answering a different question than the one asked is the failure mode."""
    run = next(run_dir.iterdir())
    path = run / "results" / "completion_taxonomy.json"
    payload = json.loads(path.read_text())
    del payload["verdicts"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "scripts/analyse_matrix.py", "--runs", str(run_dir),
         "--min-n", "10", "--gate", "nll", "--out", str(tmp_path / "m.json")],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "raw verdicts" in (proc.stdout + proc.stderr)


def test_a_run_without_a_taxonomy_is_named_rather_than_silently_dropped(
        run_dir: Path, tmp_path: Path) -> None:
    """Skipping quietly is how an analysis reports on half the runs it was given."""
    run = next(run_dir.iterdir())
    (run / "results" / "completion_taxonomy.json").unlink()
    proc = subprocess.run(
        [sys.executable, "scripts/analyse_matrix.py", "--runs", str(run_dir),
         "--min-n", "10", "--out", str(tmp_path / "m.json")],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "completion_taxonomy" in (proc.stdout + proc.stderr)
    assert run.name in (proc.stdout + proc.stderr)


def test_misaligned_labels_stop_the_analysis_rather_than_shifting_them(
        run_dir: Path, tmp_path: Path) -> None:
    """One extra harm label would pair every prompt with the previous prompt's
    class, and every number after that would be wrong and plausible."""
    run = next(run_dir.iterdir())
    prompts_file = run / "results" / "prompts.json"
    payload = json.loads(prompts_file.read_text())
    payload["harm_label"].append("benign")
    prompts_file.write_text(json.dumps(payload), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "scripts/analyse_matrix.py", "--runs", str(run_dir),
         "--min-n", "10", "--out", str(tmp_path / "m.json")],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "not aligned" in (proc.stdout + proc.stderr)
