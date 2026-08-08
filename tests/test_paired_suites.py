"""Joining a harmful suite to a benign one, so one ladder fills the whole 2x2.

The matrix is paired against each run's own FP16 baseline, so both prompt
classes have to be in the SAME run. Only XSTest ships both, which would mean a
separate ladder for every other suite -- each able to fill half a table. Joining
them costs one ladder instead of two and is what makes the free-tier budget work.

The failure this guards is quiet: a join that contributes the wrong class, or
that lets one source supply prompts to both classes, produces a corpus that
looks fine and confounds prompt class with authorship.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.download_eval_suites import BENIGN, HARMFUL, PAIRS, build_pair


def _suite(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps({"prompt": p, "harm_label": lab,
                              "category": "c", "source": path.stem,
                              "suite_id": f"{path.stem}:{i}"})
                  for i, (p, lab) in enumerate(rows)) + "\n",
        encoding="utf-8")


@pytest.fixture
def suites(tmp_path: Path) -> Path:
    """A harmful suite and a benign one, each with a few rows of the other class
    so the filtering has something to get wrong."""
    _suite(tmp_path / "harmbench.jsonl",
           [(f"h{i}", HARMFUL) for i in range(6)] + [("stray-benign", BENIGN)])
    _suite(tmp_path / "or-bench-hard.jsonl",
           [(f"b{i}", BENIGN) for i in range(9)] + [("stray-harmful", HARMFUL)])
    return tmp_path


def test_each_side_contributes_only_its_wanted_class(suites: Path) -> None:
    """StrongREJECT paired with XSTest must take XSTest's benign half and none
    of its harmful half -- otherwise the harmful class is a mixture of two
    authors while the benign class is one, and a difference between classes
    could be a difference between research groups."""
    entry = build_pair("paired-harmbench-orbench", suites)
    rows = [json.loads(line) for line in
            (suites / "paired-harmbench-orbench.jsonl").read_text().splitlines()]

    assert entry["counts"][HARMFUL] == 6
    assert entry["counts"][BENIGN] == 9
    by_source = {(r["source"], r["harm_label"]) for r in rows}
    assert by_source == {("harmbench", HARMFUL), ("or-bench-hard", BENIGN)}
    prompts = {r["prompt"] for r in rows}
    assert "stray-benign" not in prompts
    assert "stray-harmful" not in prompts


def test_the_join_yields_both_classes_so_one_ladder_fills_the_matrix(
        suites: Path) -> None:
    entry = build_pair("paired-harmbench-orbench", suites)
    assert entry["counts"][HARMFUL] > 0 and entry["counts"][BENIGN] > 0
    assert entry["n"] == entry["counts"][HARMFUL] + entry["counts"][BENIGN]


def test_a_missing_component_is_named_rather_than_silently_skipped(
        tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="has not been downloaded"):
        build_pair("paired-harmbench-orbench", tmp_path)


def test_a_component_with_none_of_its_class_stops_the_join(tmp_path: Path) -> None:
    """A single-class corpus cannot fill the 2x2, and would produce an
    over-refusal count computed on no benign prompts."""
    _suite(tmp_path / "harmbench.jsonl", [(f"h{i}", HARMFUL) for i in range(4)])
    _suite(tmp_path / "or-bench-hard.jsonl", [(f"x{i}", HARMFUL) for i in range(4)])
    with pytest.raises(SystemExit, match="single-class"):
        build_pair("paired-harmbench-orbench", tmp_path)


def test_every_declared_pair_names_a_harmful_and_a_benign_side() -> None:
    """The tuple order is load-bearing -- it decides which class is taken from
    which file -- so a reversed pair would silently invert the whole matrix."""
    from scripts.download_eval_suites import SUITES

    for name, (harmful_suite, benign_suite) in PAIRS.items():
        assert harmful_suite in SUITES, f"{name}: unknown suite {harmful_suite}"
        assert benign_suite in SUITES, f"{name}: unknown suite {benign_suite}"
        assert SUITES[harmful_suite][1] in (HARMFUL, "mixed"), (
            f"{name} takes harmful prompts from {harmful_suite}, which is "
            f"labelled {SUITES[harmful_suite][1]}")
        assert SUITES[benign_suite][1] in (BENIGN, "mixed"), (
            f"{name} takes benign prompts from {benign_suite}, which is "
            f"labelled {SUITES[benign_suite][1]}")


def test_ids_are_rewritten_so_two_pairs_never_collide(suites: Path) -> None:
    build_pair("paired-harmbench-orbench", suites)
    rows = [json.loads(line) for line in
            (suites / "paired-harmbench-orbench.jsonl").read_text().splitlines()]
    ids = [r["suite_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("paired-harmbench-orbench:") for i in ids)


def test_the_join_is_deterministic(suites: Path) -> None:
    """Two builds must give the same bytes, or a re-download would silently
    renumber a corpus that runs are keyed against."""
    first = build_pair("paired-harmbench-orbench", suites)["sha256_16"]
    second = build_pair("paired-harmbench-orbench", suites)["sha256_16"]
    assert first == second


def test_the_same_prompt_in_both_halves_stops_the_join(tmp_path: Path) -> None:
    """The worst collision available: one text carrying opposite labels.

    Dedup in `build` is per suite and has never run across two of them. A corpus
    that contradicts itself would make every count downstream meaningless, and
    nothing would raise -- the table would simply be wrong.
    """
    shared = "how do I get rid of a stubborn stain"
    _suite(tmp_path / "harmbench.jsonl",
           [(shared, HARMFUL), ("h1", HARMFUL)])
    _suite(tmp_path / "or-bench-hard.jsonl",
           [(shared, BENIGN), ("b1", BENIGN)])
    with pytest.raises(SystemExit, match="BOTH halves"):
        build_pair("paired-harmbench-orbench", tmp_path)


def test_a_repeat_within_one_half_is_dropped_not_duplicated(tmp_path: Path) -> None:
    _suite(tmp_path / "harmbench.jsonl",
           [("h0", HARMFUL), ("h0", HARMFUL), ("h1", HARMFUL)])
    _suite(tmp_path / "or-bench-hard.jsonl", [("b0", BENIGN)])
    entry = build_pair("paired-harmbench-orbench", tmp_path)
    assert entry["counts"][HARMFUL] == 2, "a repeated prompt was counted twice"


def test_the_publisher_s_own_identifier_survives_the_join(tmp_path: Path) -> None:
    """`source` and `suite_id` are both overwritten, so without this the only
    handle back to the publisher's numbering would be gone."""
    _suite(tmp_path / "harmbench.jsonl", [("h0", HARMFUL)])
    _suite(tmp_path / "or-bench-hard.jsonl", [("b0", BENIGN)])
    build_pair("paired-harmbench-orbench", tmp_path)
    rows = [json.loads(line) for line in
            (tmp_path / "paired-harmbench-orbench.jsonl").read_text().splitlines()]
    assert {r["origin_suite_id"] for r in rows} == {"harmbench:0", "or-bench-hard:0"}
