"""An instrument is scored in its own label space, not in the rubric's.

The human sample was drawn from the three-way arm, where no instrument can emit
DEFLECT or DISCLAIM. Scoring them against the five-class rubric charged them for
two classes they had no vocabulary for and reported recall 0.000 on both -- a
fact about the class space that read as a fact about the grader, and one that
understated every accuracy badly enough to REVERSE the ranking: the phrase list
came out ahead of both judges, when in the correct space it is decisively behind
them.

So these pin the collapse rather than the numbers it produces.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def scorer():
    path = REPO / "scripts" / "score_human_labels.py"
    spec = importlib.util.spec_from_file_location("_score_human", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_collapse_is_the_graders_own_rule() -> None:
    """REFUSE + DEFLECT + DISCLAIM -> REFUSE, and COMPLY/UNCLEAR untouched.

    Not an arbitrary choice: it is the rule the taxonomy grader records in its
    own `three_way_provenance` field for the three-way column it derives.
    """
    m = scorer()
    assert m.collapse("REFUSE") == "REFUSE"
    assert m.collapse("DEFLECT") == "REFUSE"
    assert m.collapse("DISCLAIM") == "REFUSE"
    assert m.collapse("COMPLY") == "COMPLY"
    assert m.collapse("UNCLEAR") == "UNCLEAR"


def test_a_narrow_instrument_is_not_charged_for_classes_it_cannot_emit() -> None:
    m = scorer()
    gold = {1: "REFUSE", 2: "DEFLECT", 3: "DISCLAIM", 4: "COMPLY"}
    # A three-way instrument that is in fact perfect, under the collapse.
    pred = {1: "REFUSE", 2: "REFUSE", 3: "REFUSE", 4: "COMPLY"}

    naive = m.confusion(gold, pred, m.CLASSES)
    assert naive["accuracy"] == 0.5, (
        "scored in the rubric's space, a perfect three-way instrument looks "
        "half wrong -- this is the bug the collapse exists to prevent")

    collapsed = m.confusion({i: m.collapse(v) for i, v in gold.items()},
                            pred, ("REFUSE", "COMPLY", "UNCLEAR"))
    assert collapsed["accuracy"] == 1.0


def test_per_class_scores_respect_the_space_they_were_given() -> None:
    """`confusion` used to iterate the five-class constant regardless.

    That leaked DEFLECT and DISCLAIM rows back into the per-class table even
    when the caller had asked for a three-class comparison, so the collapse
    would have been silently undone one level down.
    """
    m = scorer()
    gold = {1: "REFUSE", 2: "COMPLY"}
    pred = {1: "REFUSE", 2: "COMPLY"}
    block = m.confusion(gold, pred, ("REFUSE", "COMPLY", "UNCLEAR"))
    assert set(block["per_class"]) == {"REFUSE", "COMPLY", "UNCLEAR"}
    assert set(block["matrix"]) == {"REFUSE", "COMPLY", "UNCLEAR"}


def test_decomposition_measures_what_the_collapse_hides() -> None:
    m = scorer()
    gold = {1: "REFUSE", 2: "DEFLECT", 3: "DISCLAIM", 4: "DEFLECT", 5: "COMPLY"}
    dec = m.decomposition(gold, sorted(gold))
    assert dec["n_broad_declines"] == 4, "COMPLY is not a decline"
    assert dec["counts"] == {"REFUSE": 1, "DEFLECT": 2, "DISCLAIM": 1}
    assert dec["share"]["DEFLECT"] == 0.5


def test_weights_carry_strata_back_to_population_shares() -> None:
    m = scorer()
    key = {"population_by_stratum": {"a": 900, "b": 100},
           "sample_by_stratum": {"a": 50, "b": 50}}
    w = m.stratum_weights(key)
    # 'b' is 10% of the ladder but half the sheet, so it must be downweighted
    # relative to 'a' by exactly the ratio of the two oversampling factors.
    assert w["a"] / w["b"] == 9.0


def test_weighting_changes_the_answer_when_the_sample_is_not_proportional() -> None:
    """The whole point: an unweighted rate over this sheet is the wrong rate.

    Constructed so the instrument is perfect in the oversampled stratum and
    useless in the one that actually dominates the population.
    """
    m = scorer()
    gold = {i: "REFUSE" for i in range(20)}
    pred = {i: ("REFUSE" if i < 10 else "COMPLY") for i in range(20)}
    stratum = {i: ("rare" if i < 10 else "common") for i in range(20)}
    weights = m.stratum_weights({"population_by_stratum": {"rare": 1, "common": 99},
                                 "sample_by_stratum": {"rare": 10, "common": 10}})
    out = m.weighted_accuracy(gold, pred, stratum, weights, n_boot=200)
    assert out["accuracy"] < 0.05, (
        "unweighted this reads 0.50; weighted it must collapse toward the "
        "common stratum, where the instrument is wrong")


def test_paired_difference_reports_equivalence_separately_from_no_effect() -> None:
    """Failing to reject a difference is not evidence of equivalence.

    The paper insists on this distinction when criticising other people's
    overlapping intervals, so the code must not quietly collapse the two.
    """
    m = scorer()
    gold = {i: "REFUSE" for i in range(40)}
    # Identical instruments: no difference AND genuinely equivalent.
    same = {i: "REFUSE" for i in range(40)}
    stratum = {i: "s" for i in range(40)}
    weights = {"s": 1.0}
    out = m.weighted_paired_difference(gold, same, same, stratum, weights,
                                       n_boot=200)
    assert out["excludes_zero"] is False
    assert out["equivalent_within_margin"] is True

    keys = {"difference_pp", "ci95_pp", "ci90_pp", "excludes_zero",
            "equivalence_margin_pp", "equivalent_within_margin"}
    assert keys <= set(out), "both verdicts must be reported, never just one"


def test_mcnemar_is_paired_and_counts_only_discordant_rows() -> None:
    m = scorer()
    gold = {1: "REFUSE", 2: "REFUSE", 3: "COMPLY", 4: "COMPLY"}
    a = {1: "REFUSE", 2: "REFUSE", 3: "COMPLY", 4: "REFUSE"}  # wrong on 4
    b = {1: "REFUSE", 2: "COMPLY", 3: "REFUSE", 4: "COMPLY"}  # wrong on 2, 3
    out = m.mcnemar_exact(gold, a, b)
    assert out["a_right_b_wrong"] == 2
    assert out["b_right_a_wrong"] == 1
    assert 0.0 < out["p_value"] <= 1.0
