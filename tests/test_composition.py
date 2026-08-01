import math

import pytest

from cliffguard.eval.composition import (
    Composition,
    chain_bit_cost,
    chain_performance,
    collapse_bits_threshold_closed_form,
    d_prime_at_bits,
    eta_at_bits,
    predict_collapse,
    sector_ordering,
    tail_inflation,
    tail_rate,
    threshold_performance,
)

# ---------------------------------------------------------------------------
# eta / d' along the ladder (Assumption A3, Theorem 1)
# ---------------------------------------------------------------------------


def test_eta_is_anchored_at_four_bits() -> None:
    assert eta_at_bits(4.0, eta_4=0.30) == pytest.approx(0.30)


def test_eta_quadruples_per_bit_removed() -> None:
    """A3 with base 4: one bit removed multiplies eta by 4."""
    assert eta_at_bits(3.0, 0.30) == pytest.approx(4.0 * eta_at_bits(4.0, 0.30))


def test_eta_is_monotone_decreasing_in_bits() -> None:
    vals = [eta_at_bits(b, 0.30) for b in (2.0, 3.0, 4.0, 8.0)]
    assert vals == sorted(vals, reverse=True)


def test_eta_rejects_bad_base() -> None:
    with pytest.raises(ValueError, match="base"):
        eta_at_bits(4.0, 0.3, base=1.0)


def test_d_prime_recovers_full_precision_at_high_bits() -> None:
    assert d_prime_at_bits(16.0, 2.5, 0.30) == pytest.approx(2.5, rel=1e-3)


def test_d_prime_is_monotone_increasing_in_bits() -> None:
    vals = [d_prime_at_bits(b, 2.5, 0.30) for b in (2.0, 3.0, 4.0, 8.0)]
    assert vals == sorted(vals)


def test_d_prime_is_smooth_no_discontinuity() -> None:
    """Theorem 2: d'(b) has NO discontinuity — the cliff comes from the
    sigmoidal readout, not from anything discontinuous in the weights."""
    step = 0.01
    diffs = [
        d_prime_at_bits(b + step, 2.5, 0.30) - d_prime_at_bits(b, 2.5, 0.30)
        for b in [1.0 + i * step for i in range(1000)]
    ]
    assert max(diffs) < 0.02   # bounded, no jump


# ---------------------------------------------------------------------------
# Composition readouts
# ---------------------------------------------------------------------------


def test_threshold_performance_increases_with_d_prime() -> None:
    assert threshold_performance(3.0) > threshold_performance(1.0)


def test_threshold_performance_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        threshold_performance(2.0, alpha=1.0)


def test_chain_reduces_to_threshold_at_one_step() -> None:
    assert chain_performance(2.5, n_steps=1) == pytest.approx(threshold_performance(2.5))


def test_chain_decays_geometrically_in_steps() -> None:
    p = threshold_performance(2.5)
    assert chain_performance(2.5, n_steps=8) == pytest.approx(p**8)


def test_chain_rejects_zero_steps() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        chain_performance(2.0, n_steps=0)


def test_tail_rate_recovers_p0_at_zero_eta() -> None:
    assert tail_rate(1e-3, 0.0) == pytest.approx(1e-3, rel=1e-6)


def test_tail_rate_increases_with_eta() -> None:
    vals = [tail_rate(1e-3, e) for e in (0.0, 0.25, 1.0, 4.0)]
    assert vals == sorted(vals)


def test_tail_inflation_grows_with_rarity() -> None:
    """Corollary 1.3, the central claim: at FIXED eta, rarer events inflate
    more. This is the mechanism behind 'quality is not a safety proxy'."""
    infl = [tail_inflation(p0, 0.5) for p0 in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)]
    assert infl == sorted(infl)
    assert infl[0] == pytest.approx(1.5, abs=0.1)
    assert infl[-1] == pytest.approx(24.9, abs=1.0)


def test_tail_asymptotic_log_ratio_converges() -> None:
    """log p_q / log p_0 -> 1/(1+eta) as p0 -> 0."""
    eta = 0.25
    ratios = [
        math.log(tail_rate(p0, eta)) / math.log(p0)
        for p0 in (1e-2, 1e-4, 1e-8, 1e-16)
    ]
    assert ratios == sorted(ratios, reverse=True)      # monotone descent
    assert ratios[-1] == pytest.approx(1.0 / (1.0 + eta), abs=0.02)


def test_tail_rate_rejects_out_of_range_p0() -> None:
    with pytest.raises(ValueError, match="p0"):
        tail_rate(0.0, 0.5)


# ---------------------------------------------------------------------------
# Theorem 2 — collapse location
# ---------------------------------------------------------------------------


def test_closed_form_matches_the_documented_value() -> None:
    """docs/theorems.md §3: d'_0=2.5, alpha=0.05, eta_4=0.30 -> b* = 2.94 bits,
    i.e. Q3_K_M — where the blueprint pre-registered the cliff."""
    assert collapse_bits_threshold_closed_form(2.5, 0.05, 0.30) == pytest.approx(2.94, abs=0.01)


def test_closed_form_raises_when_no_crossing_exists() -> None:
    with pytest.raises(ValueError, match="never crosses"):
        collapse_bits_threshold_closed_form(1.0, alpha=0.05)


def test_closed_form_is_where_d_prime_equals_z() -> None:
    from scipy.stats import norm

    b = collapse_bits_threshold_closed_form(2.5, 0.05, 0.30)
    assert d_prime_at_bits(b, 2.5, 0.30) == pytest.approx(float(norm.ppf(0.95)), abs=1e-6)


def test_stronger_full_precision_signal_survives_lower_bits() -> None:
    weak = collapse_bits_threshold_closed_form(2.0, 0.05, 0.30)
    strong = collapse_bits_threshold_closed_form(4.0, 0.05, 0.30)
    assert strong < weak


# ---------------------------------------------------------------------------
# Corollary 1.2 — the reasoning-depth law
# ---------------------------------------------------------------------------


def test_chain_bit_cost_is_one_bit_per_four_times() -> None:
    assert chain_bit_cost(4) == pytest.approx(1.0)
    assert chain_bit_cost(16) == pytest.approx(2.0)
    assert chain_bit_cost(64) == pytest.approx(3.0)


def test_collapse_increments_converge_to_one_bit_per_4x() -> None:
    """Corollary 1.2 as stated: the INCREMENTS converge to 1 bit, not the
    absolute values (which carry an O(1) linearisation offset)."""
    bits = {
        T: predict_collapse(Composition.CHAIN, d_prime_0=3.2, n_steps=T).bits
        for T in (4, 16, 64)
    }
    assert bits[16] - bits[4] == pytest.approx(1.0, abs=0.15)
    assert bits[64] - bits[16] == pytest.approx(1.0, abs=0.15)


def test_longer_chains_collapse_at_higher_bit_width() -> None:
    bits = [
        predict_collapse(Composition.CHAIN, 3.2, n_steps=T).bits for T in (1, 4, 16, 64)
    ]
    assert bits == sorted(bits)


def test_predict_collapse_rejects_bad_retention() -> None:
    with pytest.raises(ValueError, match="retention"):
        predict_collapse(Composition.THRESHOLD, 2.5, retention=1.0)


@pytest.mark.parametrize(
    "composition",
    [Composition.THRESHOLD, Composition.CHAIN, Composition.TAIL],
)
def test_flat_curve_has_no_collapse_crossing(composition: Composition) -> None:
    """eta_4=0 makes performance independent of bit-width.

    The old bisection returned the lower search boundary (1 bit), fabricating
    a collapse point on this null curve.
    """
    prediction = predict_collapse(
        composition,
        d_prime_0=3.2,
        eta_4=0.0,
        n_steps=8,
        p0=1e-3,
    )
    assert math.isnan(prediction.bits)
    assert "no collapse crossing" in prediction.summary()


def test_predict_collapse_rejects_non_enum_composition() -> None:
    with pytest.raises(ValueError, match="Composition value"):
        predict_collapse("CHAIN", 3.2)  # type: ignore[arg-type]


def test_predict_collapse_root_reproduces_its_target() -> None:
    prediction = predict_collapse(
        Composition.CHAIN,
        d_prime_0=3.2,
        eta_4=0.30,
        retention=0.5,
        n_steps=16,
    )
    reference = chain_performance(d_prime_at_bits(16.0, 3.2, 0.30), 16)
    at_root = chain_performance(d_prime_at_bits(prediction.bits, 3.2, 0.30), 16)
    assert at_root == pytest.approx(0.5 * reference, abs=1e-10)


def test_closed_form_validates_alpha_and_base() -> None:
    with pytest.raises(ValueError, match="alpha"):
        collapse_bits_threshold_closed_form(2.5, alpha=1.0)
    with pytest.raises(ValueError, match="base"):
        collapse_bits_threshold_closed_form(2.5, base=1.0)


# ---------------------------------------------------------------------------
# Corollary 2.1 — sector ordering (the headline testable prediction)
# ---------------------------------------------------------------------------


def test_sector_ordering_matches_the_documented_prediction() -> None:
    preds = [
        predict_collapse(Composition.CHAIN, 3.2, n_steps=64, label="reasoning T=64"),
        predict_collapse(Composition.CHAIN, 3.2, n_steps=4, label="reasoning T=4"),
        predict_collapse(Composition.TAIL, 2.5, p0=1e-5, label="security p0=1e-5"),
        predict_collapse(Composition.THRESHOLD, 2.5, label="safety single-shot"),
    ]
    order = [p.label for p in sector_ordering(preds)]
    assert order[0] == "reasoning T=64"
    assert order[-1] == "safety single-shot"
    assert order.index("security p0=1e-5") < order.index("reasoning T=4")


def test_sector_ordering_drops_unreachable_predictions() -> None:
    ok = predict_collapse(Composition.THRESHOLD, 2.5, label="ok")
    unreachable = predict_collapse(
        Composition.THRESHOLD, 2.5, retention=0.999999, label="unreachable"
    )
    result = sector_ordering([ok, unreachable])
    assert all(not math.isnan(p.bits) for p in result)


def test_summary_reports_bits_and_composition() -> None:
    s = predict_collapse(Composition.CHAIN, 3.2, n_steps=8, label="reasoning").summary()
    assert "reasoning" in s and "CHAIN" in s and "bits" in s
