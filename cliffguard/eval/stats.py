"""Statistical analysis module — see blueprint §12.5, §13.

Implements the pre-registered hypothesis tests and power calculations:

  H1 (cliff existence): chi-square test on cliff boundary agreement
     across model families.
  H2/H3 (FPR decoupling): Kolmogorov-Smirnov test on empirical FPR
     distributions across quantization schemes.
  H4 (composition gain): Wilcoxon signed-rank test on per-prompt
     ABR: full stack vs best single primitive.
  H5 (Tier C weakness): Wilcoxon signed-rank test on per-prompt
     block decisions: Tier C vs baseline, Tier C+ vs baseline.

Power calculations use the pre-registered effect sizes and alpha
(Bonferroni-corrected: alpha = 0.05 / 5 = 0.01).

Per blueprint §14.4 (Sensitivity Corollary): all tests are two-sided.
Bonferroni correction across five hypotheses: alpha_corrected = 0.01.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy import stats as _stats

ALPHA_UNCORRECTED: float = 0.05
BONFERRONI_N: int = 5
ALPHA_CORRECTED: float = ALPHA_UNCORRECTED / BONFERRONI_N  # 0.01


def required_n(
    effect_size: float,
    alpha: float = ALPHA_CORRECTED,
    power: float = 0.80,
) -> int:
    """Estimate required sample size for a two-sided Wilcoxon
    signed-rank test using the normal approximation.
    effect_size: Cohen's d equivalent.
    Returns the integer n (minimum 2).
    Raises ValueError if effect_size <= 0, alpha <= 0, power <= 0
    or power >= 1."""
    if effect_size <= 0:
        raise ValueError(f"effect_size must be > 0, got {effect_size}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if power <= 0 or power >= 1:
        raise ValueError(f"power must be in (0, 1), got {power}")
    z_alpha2 = float(_stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(_stats.norm.ppf(power))
    n = ((z_alpha2 + z_power) / effect_size) ** 2
    return max(2, math.ceil(n))


def realized_power(
    n: int,
    effect_size: float,
    alpha: float = ALPHA_CORRECTED,
) -> float:
    """Estimate realized power for a two-sided Wilcoxon signed-rank
    test given n and effect_size.
    Returns float in (0.0, 1.0).
    Raises ValueError if n < 2 or effect_size <= 0."""
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if effect_size <= 0:
        raise ValueError(f"effect_size must be > 0, got {effect_size}")
    z_alpha2 = float(_stats.norm.ppf(1.0 - alpha / 2.0))
    z = effect_size * math.sqrt(n) - z_alpha2
    return float(_stats.norm.cdf(z))


def wilcoxon_test(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Run two-sided Wilcoxon signed-rank test on paired samples x, y.
    Returns (statistic, p_value).
    Raises ValueError if arrays have different lengths or length < 2."""
    if len(x) != len(y):
        raise ValueError(
            f"Arrays must have the same length: len(x)={len(x)}, len(y)={len(y)}"
        )
    if len(x) < 2:
        raise ValueError(f"Arrays must have length >= 2, got {len(x)}")
    differences = x - y
    if np.all(differences == 0.0):
        # Degenerate case: no differences — p-value is 1.0 by convention.
        return (0.0, 1.0)
    result = _stats.wilcoxon(differences, alternative="two-sided")
    return (float(result.statistic), float(result.pvalue))


def ks_test_fpr_decoupling(
    fpr_by_scheme: dict[str, npt.NDArray[np.float64]],
    reference_scheme: str = "FP16",
) -> dict[str, tuple[float, float]]:
    """Run KS test between each scheme's FPR distribution and the
    reference scheme (FP16).
    Returns dict mapping scheme_name -> (ks_statistic, p_value).
    Raises ValueError if reference_scheme not in fpr_by_scheme."""
    if reference_scheme not in fpr_by_scheme:
        raise ValueError(
            f"reference_scheme {reference_scheme!r} not found in fpr_by_scheme. "
            f"Available schemes: {list(fpr_by_scheme.keys())}"
        )
    ref = fpr_by_scheme[reference_scheme]
    results: dict[str, tuple[float, float]] = {}
    for scheme, dist in fpr_by_scheme.items():
        if scheme == reference_scheme:
            continue
        ks_result = _stats.ks_2samp(ref, dist)
        results[scheme] = (float(ks_result.statistic), float(ks_result.pvalue))
    return results


def test_h1_cliff_existence(
    cliff_detected_by_family: dict[str, bool],
    min_families: int = 2,
) -> tuple[bool, str]:
    """Test H1: cliff detected in >= min_families model families.
    Returns (accepted: bool, summary: str).
    Does not run a frequentist test — H1 is a counting criterion
    per blueprint §13 (cliff at same boundary in >= 2/3 families)."""
    n_detected = sum(1 for v in cliff_detected_by_family.values() if v)
    total = len(cliff_detected_by_family)
    accepted = n_detected >= min_families
    summary = (
        f"H1: cliff detected in {n_detected}/{total} model families "
        f"(threshold: >= {min_families}). "
        f"{'Accepted' if accepted else 'Rejected'}."
    )
    return (accepted, summary)


def test_h4_composition_gain(
    full_stack_scores: npt.NDArray[np.float64],
    best_single_scores: npt.NDArray[np.float64],
) -> tuple[float, float, bool]:
    """Test H4: full stack ABR > best single primitive ABR.
    Runs Wilcoxon signed-rank test.
    Returns (statistic, p_value, accepted) where accepted = p_value
    < ALPHA_CORRECTED AND mean(full_stack) > mean(best_single)."""
    stat, p_value = wilcoxon_test(full_stack_scores, best_single_scores)
    accepted = bool(
        p_value < ALPHA_CORRECTED
        and float(np.mean(full_stack_scores)) > float(np.mean(best_single_scores))
    )
    return (stat, p_value, accepted)


def test_h5_tier_c_weakness(
    tier_c_scores: npt.NDArray[np.float64],
    tier_c_plus_scores: npt.NDArray[np.float64],
    baseline_scores: npt.NDArray[np.float64],
) -> dict[str, object]:
    """Test H5: Tier C shows no significant ABR vs baseline;
    Tier C+ does.
    Runs two Wilcoxon tests: (tier_c vs baseline), (tier_c+ vs baseline).
    Returns dict with keys:
      tier_c_stat, tier_c_p, tier_c_accepted (p >= ALPHA_CORRECTED),
      tier_c_plus_stat, tier_c_plus_p, tier_c_plus_accepted (p < ALPHA_CORRECTED)."""
    tc_stat, tc_p = wilcoxon_test(tier_c_scores, baseline_scores)
    tcp_stat, tcp_p = wilcoxon_test(tier_c_plus_scores, baseline_scores)
    return {
        "tier_c_stat": tc_stat,
        "tier_c_p": tc_p,
        "tier_c_accepted": bool(tc_p >= ALPHA_CORRECTED),
        "tier_c_plus_stat": tcp_stat,
        "tier_c_plus_p": tcp_p,
        "tier_c_plus_accepted": bool(tcp_p < ALPHA_CORRECTED),
    }
