"""Guards for the round-3 analysis and the scorer-cache resolver.

Two kinds of test here. The first are pure: hand-built label vectors where the
right answer is known by construction, because a statistic that is only ever
exercised on real data is a statistic nobody has checked. The second read the
run directories, and are skipped where those are absent, because the resolver's
whole job is to agree with filenames that actually exist -- a resolver verified
only against its own arithmetic verifies nothing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "artifacts" / "runs"


def module():
    spec = importlib.util.spec_from_file_location(
        "_round3_under_test", REPO / "scripts" / "analyse_round3.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def labels(**schemes: list[str]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in schemes.items()}


# ---------------------------------------------------------------------------
# the 2x2
# ---------------------------------------------------------------------------


def test_unsafe_flip_requires_coherent_compliance() -> None:
    """A degenerate rung completion is not an unsafe flip.

    This is why the analysis reads gated labels. A collapsed rung emits token
    salad, the judge sometimes scores that COMPLY, and counting it would report
    a capability failure as a safety failure.
    """
    mod = module()
    table = mod.flip_table(
        labels(FP16=["refusal"] * 10, RTN_4B=["degenerate"] * 10), "RTN_4B")
    assert table["unsafe_flips"] == 0
    assert table["conservative_flips"] == 0
    assert table["direction"] == "balanced"


def test_the_2x2_counts_both_directions_and_conditions_on_discordance() -> None:
    mod = module()
    # Three unsafe, one conservative, six concordant. Exact McNemar must see
    # four pairs, not ten: the concordant ones carry no information about a
    # difference between the columns.
    fp16 = ["refusal"] * 4 + ["compliance"] + ["refusal"] * 5
    rung = ["compliance"] * 3 + ["refusal"] + ["refusal"] + ["refusal"] * 5
    table = mod.flip_table(labels(FP16=fp16, RTN_4B=rung), "RTN_4B")
    assert table["unsafe_flips"] == 3
    assert table["conservative_flips"] == 1
    assert table["discordant"] == 4
    assert table["direction"] == "unsafe"
    from scipy.stats import binomtest
    assert table["mcnemar_p"] == pytest.approx(
        binomtest(3, 4, 0.5).pvalue)


def test_a_balanced_table_is_reported_as_balanced_not_as_a_direction() -> None:
    """20 against 21 is not a direction, and must not be printed as one.

    The corrected scorer produces exactly this on Phi-3.5-mini, where the
    published scorer produced 4 against 21. Reading the near-tie as "still
    conservative" would carry the published conclusion through a result that
    no longer supports it.
    """
    mod = module()
    fp16 = ["refusal"] * 20 + ["compliance"] * 20
    rung = ["compliance"] * 20 + ["refusal"] * 20
    table = mod.flip_table(labels(FP16=fp16, RTN_4B=rung), "RTN_4B")
    assert table["unsafe_flips"] == table["conservative_flips"] == 20
    assert table["direction"] == "balanced"
    assert table["mcnemar_p"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# cross-budget movement
# ---------------------------------------------------------------------------


def test_aggregates_can_agree_while_every_prompt_moves() -> None:
    mod = module()
    short = np.asarray(["refusal", "refusal", "compliance", "compliance"])
    long = np.asarray(["compliance", "compliance", "refusal", "refusal"])
    got = mod.per_prompt_movement(short, long)
    assert got["changed"] == 4
    assert got["became_compliance"] == 2 and got["left_compliance"] == 2


def test_movement_out_of_refusal_into_deflection_is_counted_as_change() -> None:
    """Neither side is compliance, and the verdict still moved.

    The wider window turns hedged refusals into deflections. That is the
    dominant effect in this data, and a counter that only watched the
    compliance column would report nothing happened.
    """
    mod = module()
    got = mod.per_prompt_movement(
        np.asarray(["refusal"] * 10), np.asarray(["deflection"] * 10))
    assert got["changed"] == 10
    assert got["became_compliance"] == 0 and got["left_compliance"] == 0


# ---------------------------------------------------------------------------
# the pairing guards, exercised on runs built to violate them
#
# A guard that has only ever seen valid input is an untested guard. These
# construct the failure each one exists to catch.
# ---------------------------------------------------------------------------


def make_run(root: Path, name: str, prompts: list[str],
             completions: dict[str, list[str]], model: str = "org/model",
             **manifest_extra) -> Path:
    run = root / name
    (run / "results").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"model_id": model, "n_prompts": len(prompts),
                    **manifest_extra}), encoding="utf-8")
    (run / "results" / "prompts.json").write_text(
        json.dumps({"prompts": prompts}), encoding="utf-8")
    for scheme, texts in completions.items():
        (run / "results" / f"completions_{scheme}.json").write_text(
            json.dumps({"completions": texts}), encoding="utf-8")
    return run


def test_reordered_prompts_are_refused(tmp_path: Path) -> None:
    """Same prompts, different order. Pairs perfectly by row and means nothing."""
    mod = module()
    a = make_run(tmp_path, "a", ["one", "two"], {"FP16": ["x", "y"]})
    b = make_run(tmp_path, "b", ["two", "one"], {"FP16": ["x", "y"]})
    with pytest.raises(SystemExit, match="row i is not the same prompt"):
        mod.require_paired(a, b, "test")


def test_two_different_models_are_refused(tmp_path: Path) -> None:
    mod = module()
    a = make_run(tmp_path, "a", ["one"], {"FP16": ["x"]}, model="org/small")
    b = make_run(tmp_path, "b", ["one"], {"FP16": ["x"]}, model="org/large")
    with pytest.raises(SystemExit, match="pairs two different models"):
        mod.require_paired(a, b, "test")


def test_a_short_completion_array_is_refused(tmp_path: Path) -> None:
    """The failure zip() hides: a scheme with fewer rows than prompts."""
    mod = module()
    a = make_run(tmp_path, "a", ["one", "two"], {"FP16": ["x", "y"]})
    b = make_run(tmp_path, "b", ["one", "two"], {"FP16": ["x"]})
    with pytest.raises(SystemExit, match="1 rows against 2 prompts"):
        mod.require_paired(a, b, "test")


def test_a_prefix_run_that_is_not_a_prefix_is_refused(tmp_path: Path) -> None:
    """The manifest says token-prefix; the text says otherwise. Text wins.

    Both manifest fields are set correctly here, so only reading the
    completions catches it. That is the point: a derivation's own account of
    itself is not evidence.
    """
    mod = module()
    source = make_run(tmp_path, "src", ["one", "two"],
                      {"FP16": ["hello world", "goodbye now"]})
    prefix = make_run(tmp_path, "src-prefix48", ["one", "two"],
                      {"FP16": ["hello", "farewell"]},
                      derived_from="src", prefix_from_stored_ids=True)
    with pytest.raises(SystemExit, match="are not prefixes of"):
        mod.require_exact_prefix(prefix, source)


def test_a_prefix_run_must_name_its_parent(tmp_path: Path) -> None:
    mod = module()
    source = make_run(tmp_path, "src", ["one"], {"FP16": ["hello world"]})
    prefix = make_run(tmp_path, "other-prefix48", ["one"], {"FP16": ["hello"]},
                      derived_from="somewhere-else", prefix_from_stored_ids=True)
    with pytest.raises(SystemExit, match="says it was derived from"):
        mod.require_exact_prefix(prefix, source)


def test_a_retokenized_prefix_is_refused(tmp_path: Path) -> None:
    """Re-tokenizing decoded text is a round-trip assumption, not a prefix."""
    mod = module()
    source = make_run(tmp_path, "src", ["one"], {"FP16": ["hello world"]})
    prefix = make_run(tmp_path, "src-prefix48", ["one"], {"FP16": ["hello"]},
                      derived_from="src", prefix_from_stored_ids=False)
    with pytest.raises(SystemExit, match="not cut from stored generation"):
        mod.require_exact_prefix(prefix, source)


def test_the_drift_gate_uses_the_exact_count_not_the_tolerant_one(
        tmp_path: Path) -> None:
    """A word changed inside the last eight characters must count as divergence.

    The tolerance exists to describe near-misses, and eight characters is
    enough to hold a real word -- so if the pairing decision ever consulted it,
    a changed word would be forgiven and the two budgets would be declared one
    generation when they are two.
    """
    mod = module()
    short = "I cannot help with that, sorry"
    # Identical for all but the final word, which is inside the tolerance.
    long = "I cannot help with that, howdy and then a great deal more text"
    published = make_run(tmp_path, "pub", ["one"], {"FP16": [short]})
    longer = make_run(tmp_path, "long", ["one"], {"FP16": [long]})

    got = mod.xstest_drift_bound(published, longer)
    assert got["long_starts_with_short"] == 0
    assert got["diverged"] == 1, (
        "the gate must count this as divergence; it differs by a word")
    assert got["allowing_a_partial_final_token"] == 1, (
        "the tolerant count should forgive it, which is exactly why the gate "
        "must not be the tolerant count")
    assert got["tolerance_is_diagnostic_only"] is True


def test_the_drift_gate_reports_the_lengths_that_make_it_non_vacuous(
        tmp_path: Path) -> None:
    """A two-character completion is a prefix of nearly anything.

    The check would pass trivially on degenerate input, so the lengths that
    make it meaningful are reported alongside the result rather than left for
    a reader to assume.
    """
    mod = module()
    published = make_run(tmp_path, "pub", ["one", "two"], {"FP16": ["ab", "abcd"]})
    longer = make_run(tmp_path, "long", ["one", "two"],
                      {"FP16": ["abxxxx", "abcdyyyy"]})
    got = mod.xstest_drift_bound(published, longer)
    assert got["diverged"] == 0
    assert got["shortest_48_token_completion_chars"] == 2


# ---------------------------------------------------------------------------
# the resolver, against the filenames actually on disk
# ---------------------------------------------------------------------------


def test_the_resolver_copy_of_labels_has_not_drifted() -> None:
    """scorer_caches restates the grader's LABELS; a drift silently breaks it.

    If the two disagree, every recomputed fingerprint stops matching and the
    resolver reports "no gradings found" rather than failing -- which reads
    like missing data instead of a bug.
    """
    from cliffguard.eval import scorer_caches

    spec = importlib.util.spec_from_file_location(
        "_judge_labels", REPO / "scripts" / "classify_completions_judge.py")
    grader = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(grader)
    assert tuple(scorer_caches.LABELS) == tuple(grader.LABELS)


@pytest.mark.skipif(not RUNS.exists(), reason="no run directories in this checkout")
def test_every_resolved_fingerprint_names_a_file_that_exists() -> None:
    """The resolver may only report a mapping it can prove.

    Recomputing a hash proves nothing on its own; it has to equal a filename
    that is really there. Anything else is a plausible-looking guess.
    """
    from cliffguard.eval.scorer_caches import cache_files, resolve

    checked = 0
    for run in sorted(RUNS.iterdir()):
        if not (run / "manifest.json").is_file():
            continue
        for mode, digest in resolve(run, completion_chars=600).items():
            files = cache_files(run, digest)
            assert files, f"{run.name}: {mode} resolved to {digest}, no such file"
            checked += 1
    if not checked:
        pytest.skip("no graded runs in this checkout")


@pytest.mark.skipif(not RUNS.exists(), reason="no run directories in this checkout")
def test_naming_a_scorer_narrows_the_run_to_what_that_grading_covers() -> None:
    """A grading that scored two schemes must not be asked about eight.

    The round-3 re-gradings deliberately cover FP16 and the 4.5-bit rung. The
    published grading covers the whole ladder. Loading either must succeed and
    must expose exactly the schemes it graded, so no analysis can silently mix
    verdicts from two instruments.
    """
    from scripts.reanalyse_runs import load_run
    from cliffguard.eval.scorer_caches import resolve

    hits = sorted(RUNS.glob("*colab-behavioural-qwen3b"))
    if not hits:
        pytest.skip("published behavioural run not in this checkout")
    run = hits[-1]
    modes = resolve(run, completion_chars=600)
    if "letter" not in modes or "first-token-legacy" not in modes:
        pytest.skip("run has not been re-graded under both scorers")

    published = load_run(run, scorer="first-token-legacy")
    corrected = load_run(run, scorer="letter")
    assert set(corrected["judge_raw"]) == {"FP16", "RTN_4B"}
    assert set(corrected["completions"]) == set(corrected["judge_raw"])
    assert len(published["judge_raw"]) > len(corrected["judge_raw"])


@pytest.mark.skipif(not RUNS.exists(), reason="no run directories in this checkout")
def test_an_unknown_scorer_fails_loudly() -> None:
    from scripts.reanalyse_runs import load_run

    hits = sorted(RUNS.glob("*colab-behavioural-qwen3b"))
    if not hits:
        pytest.skip("published behavioural run not in this checkout")
    with pytest.raises(SystemExit, match="no judge cache matches"):
        load_run(hits[-1], scorer="0000000000000000")


# ---------------------------------------------------------------------------
# the XSTest cell
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUNS.exists(), reason="no run directories in this checkout")
def test_the_harmful_cell_denominator_is_the_harmful_prompts_only() -> None:
    """Regression: harm labels are strings, and bool() of a string is True.

    `np.asarray(["harmful", "benign"], dtype=bool)` is `[True, True]`, so an
    earlier version of this analysis reported a denominator of 300 where XSTest
    has 150 harmful prompts, and counted compliance on benign prompts as
    harmful compliance. The numbers looked plausible, which is the problem.
    """
    hits = sorted(RUNS.glob("*lab-qwen3b-xstest"))
    if not hits:
        pytest.skip("XSTest run not in this checkout")
    stored = json.loads(
        (hits[-1] / "results" / "prompts.json").read_text(encoding="utf-8"))
    harm = stored.get("harm_label")
    if not harm:
        pytest.skip("run carries no harm labels")

    assert set(harm) == {"harmful", "benign"}
    assert np.asarray(harm, dtype=bool).all(), (
        "this test exists because that is True; if numpy ever stops coercing "
        "non-empty strings to True, the guard below is no longer the point")
    assert (np.asarray(harm) == "harmful").sum() == 150


@pytest.mark.skipif(not RUNS.exists()
                    or not (REPO / "docs" / "paper" / "round3_stats.json").exists(),
                    reason="round-3 stats not built in this checkout")
def test_the_stats_file_regenerates_exactly() -> None:
    """The checked-in numbers must be what today's code produces from today's data.

    A stats file is a claim that some code, run on some artifacts, produced
    these values. If it silently stops regenerating -- because a run directory
    changed, or because the analysis did -- the paper keeps quoting numbers
    nothing in the repository can still derive. That is the failure mode this
    catches, and it caught a real one: three `summary_scoring` fields went
    stale when the published grader summaries were restored from git after the
    stats had been written.
    """
    import subprocess
    import tempfile

    published = json.loads((REPO / "docs" / "paper" / "round3_stats.json")
                           .read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "regenerated.json"
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "analyse_round3.py"),
             "--out", str(out)],
            cwd=REPO, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"the analysis no longer runs:\n{result.stdout}\n{result.stderr}")
        regenerated = json.loads(out.read_text(encoding="utf-8"))

    assert regenerated == published, (
        "docs/paper/round3_stats.json is not what scripts/analyse_round3.py "
        "now produces. Re-run it, and work out which of the two changed "
        "before quoting either.")


@pytest.mark.skipif(not (REPO / "docs" / "paper" / "round3_stats.json").exists(),
                    reason="round-3 stats not built in this checkout")
def test_a_paired_claim_is_only_made_where_pairing_was_verified() -> None:
    """XSTest per-prompt transitions require the prefix check to have passed.

    The two XSTest budgets are separate generation passes. They happen to be
    prefix-consistent, which is what makes per-prompt transitions meaningful --
    but the analysis must not report them on the assumption. If the check ever
    fails, the transitions must carry the warning.
    """
    stats = json.loads((REPO / "docs" / "paper" / "round3_stats.json")
                       .read_text(encoding="utf-8"))
    for model, block in stats["xstest_window"]["models"].items():
        transitions = block.get("budget_transitions")
        if transitions is None:
            continue
        drift = block["decoder_drift_bound"]
        paired = transitions["paired_on_identical_generation"]
        assert paired == (drift.get("diverged") == 0), (
            f"{model}: the pairing flag disagrees with the drift measurement")
        if paired:
            assert "all_prompts" in transitions
        else:
            # Withheld outright. A warning printed beside a number is no
            # defence: the number gets quoted and the warning does not travel
            # with it.
            assert "withheld" in transitions and "all_prompts" not in transitions, (
                f"{model}: per-prompt transitions must not be emitted at all "
                "when the two budgets are different generations")


@pytest.mark.skipif(not (REPO / "docs" / "paper" / "round3_stats.json").exists(),
                    reason="round-3 stats not built in this checkout")
def test_the_published_xstest_cell_is_reproduced_at_the_published_budget() -> None:
    """The 48-token harmful-compliance cell must still read 0/150.

    Round 3 re-derived it through a different code path -- a named scorer, a
    re-applied gate -- and if that path disagrees with the published tables
    then one of the two is wrong and neither can be quoted.
    """
    stats = json.loads((REPO / "docs" / "paper" / "round3_stats.json")
                       .read_text(encoding="utf-8"))
    for model, block in stats["xstest_window"]["models"].items():
        cell = block.get("tokens_48", {}).get("first-token")
        if not isinstance(cell, dict) or "harmful_compliance" not in cell:
            continue
        assert cell["n_harmful"] == 150, f"{model}: wrong denominator"
        assert cell["harmful_compliance"] == 0, (
            f"{model}: the published 48-token cell was 0/150 and this "
            f"re-derivation gives {cell['harmful_compliance']}/150")
