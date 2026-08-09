"""The round-2 analysis, tested against the shape of the runs it will receive.

Written after discovering that the command the notebook told the author to run
after their GPU session -- `review_reanalysis.py --include '*r2-*'` -- exits
immediately on the tree round two produces, because two of its four behavioural
runs describe Qwen2.5-3B. That was found by fabricating the tree and running the
command, not by reading the code, so the tests here work the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyse_round2 import describe, paired_rows, token_budget

N = 200


def _labels(schemes: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    """Label arrays with planted transition counts.

    Each scheme is given as (newly_refuse, newly_comply) against an FP16
    baseline that refuses the first half and complies with the second, so the
    counts a test asserts on are the counts it planted.
    """
    base = np.array(["refusal"] * (N // 2) + ["compliance"] * (N // 2))
    out = {"FP16": base}
    for scheme, (to_refuse, to_comply) in schemes.items():
        cur = base.copy()
        cur[:to_comply] = "compliance"                    # refusal -> compliance
        cur[N // 2 : N // 2 + to_refuse] = "refusal"      # compliance -> refusal
        out[scheme] = cur
    return out


def test_transition_counts_are_read_off_the_pairing() -> None:
    rows = paired_rows(_labels({"RTN_4B": (21, 4)}))
    assert len(rows) == 1
    assert rows[0]["to_refusal"] == 21
    assert rows[0]["to_compliance"] == 4
    assert rows[0]["mcnemar_p"] < 0.01


def test_deployed_schemes_are_paired_but_flagged_off_axis() -> None:
    """The comparison is FP16 against one deployed quantizer. It is a paired
    test like any other; what it is not is a point on the RTN dose axis."""
    rows = {r["scheme"]: r for r in paired_rows(_labels({"RTN_4B": (5, 1),
                                                        "AWQ_4B": (7, 2)}))}
    assert rows["RTN_4B"]["on_rtn_axis"] is True
    assert rows["RTN_4B"]["bits"] == 4.5
    assert rows["AWQ_4B"]["on_rtn_axis"] is False
    assert rows["AWQ_4B"]["bits"] != rows["AWQ_4B"]["bits"]      # NaN
    assert rows["AWQ_4B"]["to_refusal"] == 7


def test_holm_family_is_the_run_not_the_study() -> None:
    """Each round-2 run is one question asked once. Correcting across runs would
    penalise the deployed comparison for the token-budget one existing."""
    rows = paired_rows(_labels({"RTN_5B": (3, 2), "RTN_4B": (21, 4)}))
    for row in rows:
        assert row["mcnemar_p_holm_within_run"] <= min(2 * row["mcnemar_p"], 1.0)


# ---------------------------------------------------------------------------
# the token-budget comparison
# ---------------------------------------------------------------------------


def _block(label, model, tokens, schemes, n=500):
    return {
        "label": label, "run": f"20260101-000000_abc_{label}", "model": model,
        "model_id": model, "n_prompts": n, "max_new_tokens": tokens,
        "schemes": ["FP16"] + schemes, "deployed": {},
        "rows": [{"scheme": s, "to_refusal": 10, "to_compliance": 2,
                  "mcnemar_p": 0.02} for s in schemes],
    }


def _run_dir(tmp_path, block, prompts):
    path = tmp_path / block["run"]
    (path / "results").mkdir(parents=True)
    (path / "results/prompts.json").write_text(
        json.dumps({"prompts": prompts}), encoding="utf-8")
    (path / "manifest.json").write_text(json.dumps({
        "label": block["label"], "model_id": block["model_id"],
        "n_prompts": block["n_prompts"],
        "max_new_tokens": block["max_new_tokens"],
        "schemes": block["schemes"], "bits": []}), encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Patch load_run so the comparison can be driven without real artifacts."""
    import scripts.analyse_round2 as mod

    prompts = {}

    def fake_load_run(path: Path):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        return {"path": path, "manifest": manifest, "judge_raw": {"FP16": [1]},
                "nll": {}, "completions": {}, "judge_meta": {}}

    monkeypatch.setattr(mod, "load_run", fake_load_run)
    return tmp_path, prompts


def test_pairs_the_long_run_against_the_short_one(tree) -> None:
    tmp_path, _ = tree
    prompts = [f"p{i}" for i in range(500)]
    short = _block("behavioural-qwen3b", "Q3B", 48, ["RTN_5B", "RTN_4B"])
    long_ = _block("r2-long256-qwen3b", "Q3B", 256, ["RTN_5B", "RTN_4B"])
    runs = [_run_dir(tmp_path, short, prompts), _run_dir(tmp_path, long_, prompts)]

    out = token_budget({b["label"]: b for b in (short, long_)}, runs)
    assert out["r2-long256-qwen3b"]["comparable_baseline"] == "behavioural-qwen3b"
    assert len(out["r2-long256-qwen3b"]["rows"]) == 2
    # Renamed from `direction_holds`: two aggregates keeping their sign is a
    # weak statement, and it used to be the only one this function made. It is
    # still reported, alongside the per-prompt comparison that replaced it as
    # the answer to the truncation question.
    assert all(r["same_direction"] for r in out["r2-long256-qwen3b"]["rows"])
    assert all("per_prompt" in r for r in out["r2-long256-qwen3b"]["rows"])


def test_a_deployed_run_is_not_a_token_budget_baseline(tree) -> None:
    """It has the same model, the same prompt count and a shorter budget, and
    shares no rung. Accepting it produced an empty comparison reported as an
    answer -- and then a KeyError, which is the only reason it was noticed."""
    tmp_path, _ = tree
    prompts = [f"p{i}" for i in range(500)]
    deployed = _block("r2-deployed-qwen3b", "Q3B", 48, ["AWQ_4B", "GPTQ_4B"])
    long_ = _block("r2-long256-qwen3b", "Q3B", 256, ["RTN_5B", "RTN_4B"])
    runs = [_run_dir(tmp_path, deployed, prompts), _run_dir(tmp_path, long_, prompts)]

    out = token_budget({b["label"]: b for b in (deployed, long_)}, runs)
    entry = out["r2-long256-qwen3b"]
    assert entry["comparable_baseline"] is None
    assert not entry.get("rows")
    assert "shares" in entry["note"]


def test_refuses_to_pair_runs_with_different_prompts(tree) -> None:
    """Both runs rebuild the corpus independently. If they ever disagree, prompt
    i is a different question in each and every count would be meaningless."""
    tmp_path, _ = tree
    short = _block("behavioural-qwen3b", "Q3B", 48, ["RTN_4B"])
    long_ = _block("r2-long256-qwen3b", "Q3B", 256, ["RTN_4B"])
    runs = [_run_dir(tmp_path, short, [f"p{i}" for i in range(500)]),
            _run_dir(tmp_path, long_, [f"other{i}" for i in range(500)])]

    out = token_budget({b["label"]: b for b in (short, long_)}, runs)
    assert "PROMPT LISTS DIFFER" in out["r2-long256-qwen3b"]["note"]
    assert not out["r2-long256-qwen3b"].get("rows")


def test_ambiguous_baseline_is_refused_rather_than_picked(tree) -> None:
    tmp_path, _ = tree
    prompts = [f"p{i}" for i in range(500)]
    a = _block("behavioural-qwen3b", "Q3B", 48, ["RTN_4B"])
    b = _block("behavioural-qwen3b-again", "Q3B", 48, ["RTN_4B"])
    long_ = _block("r2-long256-qwen3b", "Q3B", 256, ["RTN_4B"])
    runs = [_run_dir(tmp_path, x, prompts) for x in (a, b, long_)]

    out = token_budget({x["label"]: x for x in (a, b, long_)}, runs)
    assert out["r2-long256-qwen3b"]["comparable_baseline"] is None
    assert "2 runs qualify" in out["r2-long256-qwen3b"]["note"]


def test_describe_prefers_the_manifest_scheme_list(tmp_path) -> None:
    """A --deployed run has an empty `bits`, so reconstructing schemes from it
    yields ['FP16'] -- the bug that meant step four graded nothing."""
    run = {"path": tmp_path, "manifest": {
        "label": "r2-deployed-qwen3b", "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "n_prompts": 500, "max_new_tokens": 48, "bits": [],
        "schemes": ["FP16", "AWQ_4B", "GPTQ_4B"]}}
    assert describe(run)["schemes"] == ["FP16", "AWQ_4B", "GPTQ_4B"]
    assert describe(run)["model"] == "Qwen2.5-3B"


def test_describe_falls_back_for_runs_predating_the_scheme_list(tmp_path) -> None:
    run = {"path": tmp_path, "manifest": {
        "label": "behavioural-qwen3b", "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "n_prompts": 500, "max_new_tokens": 48, "bits": [5, 4]}}
    assert describe(run)["schemes"] == ["FP16", "RTN_5B", "RTN_4B"]
