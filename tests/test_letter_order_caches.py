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


# The order the ladder is written to a manifest in, which is descending by
# bit width and so is NOT its sorted order. The grader takes its scheme list
# from the manifest and hashes the completions in that sequence, so a fixture
# that sorts instead reproduces a digest the grader would never write.
MANIFEST_ORDER = ("FP16", "RTN_8B", "RTN_7B", "RTN_6B",
                  "RTN_5B", "RTN_4B", "RTN_3B", "RTN_2B")


def taxonomy_run(tmp_path: Path, gradings: dict[str, tuple[str, ...]],
                 manifest_order: tuple[str, ...] = MANIFEST_ORDER) -> Path:
    """A run carrying one taxonomy grading per entry, over the schemes named.

    Digests are computed the way the grader computes them, from the stored text,
    so the resolver has to reproduce them rather than be handed them. That
    includes the sequence: `taxonomy_content_hash` absorbs one scheme after
    another, so hashing a subset in sorted order rather than the manifest's
    produces a different fingerprint for the same grading.
    """
    module = scorer_caches._taxonomy_module()
    run_dir = tmp_path / "20260101-000000_abc1234_fake-xstest"
    results = run_dir / "results"
    results.mkdir(parents=True)
    prompts = [f"prompt {i}" for i in range(N_PROMPTS)]
    (results / "prompts.json").write_text(
        json.dumps({"prompts": prompts}), encoding="utf-8")
    all_schemes = sorted({s for schemes in gradings.values() for s in schemes})
    (run_dir / "manifest.json").write_text(
        json.dumps({"n_prompts": N_PROMPTS,
                    "schemes": [s for s in manifest_order
                                if s in set(all_schemes)]}), encoding="utf-8")
    completions = {s: [f"{s} completion {i}" for i in range(N_PROMPTS)]
                   for s in all_schemes}
    for scheme, texts in completions.items():
        (results / f"completions_{scheme}.json").write_text(
            json.dumps({"completions": texts}), encoding="utf-8")

    rank = {name: i for i, name in enumerate(manifest_order)}
    digests: dict[str, str] = {}
    for name, schemes in gradings.items():
        subset = sorted(schemes, key=lambda s: (rank.get(s, len(rank)), s))
        digest = scorer_caches.fingerprint({
            "judge": scorer_caches.DEFAULT_JUDGE, "four_bit": True,
            "labels": list(module.LABELS), "template": module.LETTER_TEMPLATE,
            "n_prompts": N_PROMPTS,
            "content": scorer_caches.taxonomy_content_hash(
                prompts, completions, subset),
            "policy": {"prompt_chars": module.PROMPT_CHARS,
                       "completion_chars": 2000, "max_length": 2560,
                       "padding_side": "left", "batch_size": 4,
                       "scoring": "letter"}})
        digests[name] = digest
        for scheme in subset:
            (results / f"taxonomy_{digest}_{scheme}.json").write_text(
                json.dumps({"verdicts": ["REFUSE"] * N_PROMPTS,
                            "margins": [1.0] * N_PROMPTS}), encoding="utf-8")
    return run_dir, digests


def test_the_widest_grading_wins_when_two_cover_different_rungs(
        tmp_path: Path) -> None:
    """Round 4 leaves both on disk, and only one of them is the ladder.

    Step 1 grades full precision alone so the option-order replicates have
    something to compare against; step 2 grades all eight rungs. The taxonomy
    fingerprint hashes the graded text of the scheme set it was handed, so those
    are two different fingerprints and both reproduce.

    `corrected_by_scheme` takes the ONE digest the resolver returns and reads
    the files carrying it. Returning the narrow grading would report an
    eight-rung re-grade as covering one rung -- silently, since a partial
    corrected grading is a legitimate state this project has actually been in.
    Every ladder-wide labelled quantity would then be built from full precision
    and nothing would say so.
    """
    ladder = ("FP16", "RTN_2B", "RTN_3B", "RTN_4B",
              "RTN_5B", "RTN_6B", "RTN_7B", "RTN_8B")
    run_dir, digests = taxonomy_run(
        tmp_path, {"audit": ("FP16",), "ladder": ladder})
    assert digests["audit"] != digests["ladder"]

    found = scorer_caches.resolve_taxonomy(run_dir)
    assert found.get("letter") == digests["ladder"]
    # What `corrected_by_scheme` reads: the files carrying that one digest.
    covered = {p.stem.split("_", 2)[-1] for p in
               (run_dir / "results").glob(f"taxonomy_{found['letter']}_*.json")}
    assert covered == set(ladder)


def test_the_ladder_resolves_when_the_manifest_is_not_in_sorted_order(
        tmp_path: Path) -> None:
    """The order the grader hashed in is the manifest's, not the alphabet's.

    Round 4 graded every rung with `--schemes FP16 RTN_8B ... RTN_2B`, and the
    grader takes its sequence from the manifest regardless of what the flag
    named. The resolver looked the same grading up as FP16,RTN_2B...RTN_8B, and
    a subset hashed in a different sequence is a different fingerprint, so the
    eight-rung re-grade matched nothing on disk.

    What made that quiet rather than loud is the fallback: a one-scheme grading
    reproduces under either ordering, because a single element is already
    sorted. So resolution did not fail -- it succeeded onto the full-precision
    audit cache from step 1, and `corrected_by_scheme` reported a completed
    ladder as covering one rung. Both eight-rung XSTest re-grades in the round 4
    archive were invisible this way.
    """
    ladder = ("FP16", "RTN_2B", "RTN_3B", "RTN_4B",
              "RTN_5B", "RTN_6B", "RTN_7B", "RTN_8B")
    assert tuple(sorted(ladder)) != MANIFEST_ORDER, (
        "this test is only meaningful while the two orderings differ")
    run_dir, digests = taxonomy_run(
        tmp_path, {"audit": ("FP16",), "ladder": ladder})

    found = scorer_caches.resolve_taxonomy(run_dir)
    assert found.get("letter") == digests["ladder"], (
        "resolved to the one-rung audit cache, which is what the sorted-order "
        "lookup did while reporting success")
    covered = {p.stem.split("_", 2)[-1] for p in
               (run_dir / "results").glob(f"taxonomy_{found['letter']}_*.json")}
    assert covered == set(ladder)


def test_an_interrupted_grading_does_not_win_on_the_width_it_never_reached(
        tmp_path: Path) -> None:
    """Hashed over eight rungs is not the same as having eight rungs on disk.

    The candidate scheme sets are pooled across every digest in the run, so a
    set contributed by one grading is tried against every scorer. A complete
    first-token pass over the whole ladder therefore offers the eight-rung set
    to the letter scorer as well -- and the content hash is over the
    completions, which are all present, not over the verdict files, which after
    an interruption are not.

    So a letter pass that died having written only FP16 still reproduces its
    eight-rung fingerprint, beats a complete one-rung grading on width, and is
    returned as an eight-rung digest carrying one file. `corrected_by_scheme`
    would then build the labelled tables from full precision alone while the
    notebook printed that every rung was graded.
    """
    ladder = ("FP16", "RTN_2B", "RTN_3B", "RTN_4B",
              "RTN_5B", "RTN_6B", "RTN_7B", "RTN_8B")
    run_dir, digests = taxonomy_run(
        tmp_path, {"audit": ("FP16",), "interrupted": ladder})
    results = run_dir / "results"

    # A COMPLETE grading by the other scorer over the whole ladder. This is
    # what keeps the eight-rung set alive as a candidate after the letter pass
    # is truncated below -- the subsets are pooled across digests, so the
    # first-token grading's coverage is offered to the letter scorer too.
    # Without it the wide set simply stops being tried and the bug is invisible.
    module = scorer_caches._taxonomy_module()
    prompts = json.loads((results / "prompts.json").read_text(
        encoding="utf-8"))["prompts"]
    completions = {
        s: json.loads((results / f"completions_{s}.json").read_text(
            encoding="utf-8"))["completions"] for s in MANIFEST_ORDER}
    ordered = [s for s in MANIFEST_ORDER if s in set(ladder)]
    first_token = scorer_caches.fingerprint({
        "judge": scorer_caches.DEFAULT_JUDGE, "four_bit": True,
        "labels": list(module.LABELS), "template": module.TAXONOMY_TEMPLATE,
        "n_prompts": N_PROMPTS,
        "content": scorer_caches.taxonomy_content_hash(
            prompts, completions, ordered),
        "policy": {"prompt_chars": module.PROMPT_CHARS,
                   "completion_chars": 2000, "max_length": 2560,
                   "padding_side": "left", "batch_size": 4,
                   "scoring": "first-token"}})
    for scheme in ladder:
        (results / f"taxonomy_{first_token}_{scheme}.json").write_text(
            json.dumps({"verdicts": ["REFUSE"] * N_PROMPTS,
                        "margins": [1.0] * N_PROMPTS}), encoding="utf-8")

    # Delete every rung of the wide LETTER grading but full precision: the
    # files an interrupted run would have written, and nothing after them.
    for path in sorted(results.glob(f"taxonomy_{digests['interrupted']}_*.json")):
        if not path.name.endswith("_FP16.json"):
            path.unlink()

    found = scorer_caches.resolve_taxonomy(run_dir)
    assert found.get("letter") == digests["audit"], (
        "resolved to a grading hashed over eight rungs that has one on disk")

    covered = {p.stem.split("_", 2)[-1] for p in
               results.glob(f"taxonomy_{found['letter']}_*.json")}
    assert covered == {"FP16"}


def test_the_manifest_bits_fallback_matches_the_grader(tmp_path: Path) -> None:
    """No `schemes` key is a state the grader has its own answer for.

    `classify_completion_taxonomy.py` falls back to full precision followed by
    one rung per entry of `bits`, in that order, and hashes in that sequence.
    A resolver that returned nothing here would sort alphabetically instead,
    reproduce no multi-rung digest, and drop onto whichever one-rung grading
    happened to match -- the same silent narrowing by a different route.
    """
    ladder = ("FP16", "RTN_2B", "RTN_4B")
    run_dir, digests = taxonomy_run(
        tmp_path, {"ladder": ladder},
        manifest_order=("FP16", "RTN_4B", "RTN_2B"))
    manifest = run_dir / "manifest.json"
    manifest.write_text(json.dumps({"n_prompts": N_PROMPTS, "bits": [4, 2]}),
                        encoding="utf-8")

    assert scorer_caches._manifest_order(run_dir) == ["FP16", "RTN_4B", "RTN_2B"]
    found = scorer_caches.resolve_taxonomy(run_dir)
    assert found.get("letter") == digests["ladder"]


def test_a_run_with_no_manifest_still_resolves_a_single_grading(
        tmp_path: Path) -> None:
    """Ordering comes from the manifest, so a run without one must degrade.

    `_manifest_order` returns nothing here and every scheme falls into the
    sorted tail. A one-scheme grading is unaffected by sequence, so the state
    the repository was in before round 4 keeps resolving rather than becoming
    an error.
    """
    run_dir, digests = taxonomy_run(tmp_path, {"audit": ("FP16",)})
    (run_dir / "manifest.json").unlink()
    found = scorer_caches.resolve_taxonomy(run_dir)
    assert found.get("letter") == digests["audit"]


def test_two_gradings_of_equal_width_are_refused_not_guessed(
        tmp_path: Path) -> None:
    """Widest-first has no answer when two gradings are equally wide.

    Nothing in rounds 3 to 5 produces this, and the resolver still must not
    invent an answer: the caller takes the single digest it is given and reads
    whatever files carry it, so choosing arbitrarily hands back half a ladder
    that looks like a whole one. An exception naming both is the only honest
    return value.
    """
    run_dir, _ = taxonomy_run(tmp_path, {
        "lower": ("FP16", "RTN_2B", "RTN_3B", "RTN_4B"),
        "upper": ("RTN_5B", "RTN_6B", "RTN_7B", "RTN_8B")})
    with pytest.raises(ValueError, match="cover 4 scheme"):
        scorer_caches.resolve_taxonomy(run_dir)


def test_a_narrow_grading_alone_still_resolves(tmp_path: Path) -> None:
    """Preferring the wide set must not stop the narrow one being found.

    This is the state the repository is in today: full precision re-graded and
    nothing below it.
    """
    run_dir, digests = taxonomy_run(tmp_path, {"audit": ("FP16",)})
    found = scorer_caches.resolve_taxonomy(run_dir)
    assert found.get("letter") == digests["audit"]


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
