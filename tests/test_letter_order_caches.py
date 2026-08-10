"""A permuted grading must land in its own cache, and be findable again.

`test_letter_order` covers the prompt. This covers what happens to it after
that: the fingerprint it produces, the file that fingerprint names, and whether
the resolver asked for one assignment returns that assignment's file.

Both directions matter and both fail quietly. A fingerprint that does not move
under permutation means a replicate is served the canonical cache and reports
perfect agreement -- the experiment's null result, manufactured. A resolver
that cannot reproduce the grader's fingerprint means a grading that took GPU
time is reported as never run, and the honest response to that is to run it
again.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from cliffguard.eval import scorer_caches

N_PROMPTS = 8
SCHEMES = ("FP16", "RTN_4B")


@pytest.fixture()
def run(tmp_path: Path) -> Path:
    """The smallest thing `resolve` will read: a manifest and some caches."""
    run_dir = tmp_path / "20260101-000000_abc1234_fake-behavioural"
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"n_prompts": N_PROMPTS}), encoding="utf-8")
    return run_dir


def digest_for(order: tuple[str, ...] | None) -> str:
    module = scorer_caches._judge_module()
    template = (module.letter_template(order) if order
                else module.LETTER_TEMPLATE)
    return scorer_caches.fingerprint(scorer_caches._identity(
        scorer_caches.DEFAULT_JUDGE, True, template, "letter", N_PROMPTS, None))


def write_cache(run_dir: Path, digest: str, verdict: str) -> None:
    for scheme in SCHEMES:
        (run_dir / "results" / f"judge_{digest}_{scheme}.json").write_text(
            json.dumps([verdict] * N_PROMPTS), encoding="utf-8")


def test_every_assignment_gets_its_own_fingerprint() -> None:
    labels = scorer_caches._judge_module().LABELS
    seen = {digest_for(o) for o in itertools.permutations(labels)}
    assert len(seen) == len(list(itertools.permutations(labels)))


def test_the_canonical_order_and_no_order_are_the_same_grading() -> None:
    """Passing the published order explicitly must not fork the cache.

    The notebook passes nothing for the canonical pass and names an order for
    the replicates. If those two spellings disagreed, the canonical grading
    would be recomputed under a second fingerprint and the comparison would run
    against a cache the manuscript never used.
    """
    labels = scorer_caches._judge_module().LABELS
    assert digest_for(None) == digest_for(tuple(labels))


def test_the_resolver_finds_each_assignment_and_not_another(run: Path) -> None:
    labels = tuple(scorer_caches._judge_module().LABELS)
    orders = [labels, ("COMPLY", "UNCLEAR", "REFUSE"),
              ("UNCLEAR", "COMPLY", "REFUSE")]
    digests = {o: digest_for(o) for o in orders}
    for order, digest in digests.items():
        write_cache(run, digest, order[0])

    for order, digest in digests.items():
        asked = None if order == labels else order
        found = scorer_caches.resolve(run, completion_chars=600,
                                      letter_order=asked)
        assert found.get("letter") == digest, order


def test_an_ungraded_permutation_resolves_to_nothing(run: Path) -> None:
    """Not to whatever else is on disk.

    This is the failure that would look like a finished experiment: the
    replicate never ran, the resolver fell back to the canonical cache, and the
    comparison reported 100% agreement.
    """
    labels = tuple(scorer_caches._judge_module().LABELS)
    write_cache(run, digest_for(labels), "REFUSE")
    found = scorer_caches.resolve(run, completion_chars=600,
                                  letter_order=("COMPLY", "UNCLEAR", "REFUSE"))
    assert "letter" not in found


def test_the_analysis_separates_a_letter_preference_from_a_judgement(
        run: Path) -> None:
    """A judge that always answers A: letters frozen, classes following the order.

    Written as the pathology rather than as the clean case, because a
    comparison that cannot detect this one is not worth running.
    """
    from scripts.analyse_letter_order import compare

    labels = tuple(scorer_caches._judge_module().LABELS)
    letters = scorer_caches._judge_module().LETTERS
    permuted_order = ("COMPLY", "UNCLEAR", "REFUSE")
    canonical = [labels[0]] * N_PROMPTS
    permuted = [permuted_order[0]] * N_PROMPTS

    row = compare(canonical, permuted, letters, labels, permuted_order)
    assert row["agreement"] == 0.0
    assert row["moved"] == N_PROMPTS
    assert row["letters"]["canonical"] == row["letters"]["permuted"] == {"A": N_PROMPTS}
    assert row["classes"]["canonical"] != row["classes"]["permuted"]


def test_gradings_of_different_lengths_are_refused_not_zipped(run: Path) -> None:
    from scripts.analyse_letter_order import compare

    labels = tuple(scorer_caches._judge_module().LABELS)
    letters = scorer_caches._judge_module().LETTERS
    with pytest.raises(SystemExit):
        compare(["REFUSE"] * 5, ["REFUSE"] * 4, letters, labels, labels)


def test_the_taxonomy_batch_size_is_part_of_its_identity() -> None:
    """Round 3 graded at batch 4 and round 4 nearly graded at 8.

    The taxonomy fingerprint includes the batch size, which is what makes that
    mistake visible rather than silent -- the caches simply do not match. This
    pins the behaviour, because the notebook now depends on it: its readout
    passes the batch size explicitly so a successful grading is not reported as
    zero rungs.
    """
    module = scorer_caches._taxonomy_module()
    base = {
        "judge": scorer_caches.DEFAULT_JUDGE, "four_bit": True,
        "labels": list(module.LABELS), "template": module.LETTER_TEMPLATE,
        "n_prompts": 8, "content": "deadbeef",
    }

    def digest(batch: int) -> str:
        return scorer_caches.fingerprint({
            **base,
            "policy": {"prompt_chars": module.PROMPT_CHARS,
                       "completion_chars": 2000, "max_length": 2560,
                       "padding_side": "left", "batch_size": batch,
                       "scoring": "letter"}})

    assert digest(4) != digest(8)
