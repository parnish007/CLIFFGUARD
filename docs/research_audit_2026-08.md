# CLIFFGUARD — Research Audit and Enhancement Plan

**Date:** 2026-08-01
**Method:** Three independent adversarial reviews — Claude Opus 5, GPT-5.6-sol @ xhigh (Codex),
and Fable 5 — run blind of each other, plus direct re-analysis of the saved Fold A/B artefacts and
a fresh literature sweep with source verification.
**Status:** analysis only. No repo code or results were modified.

---

## 0. Executive summary

The engineering is real: 989 tests pass, mypy strict is clean on 54 files, and the pipeline runs
end-to-end. The science is not yet real, and all three reviewers independently reached the same
verdict: **reject in present form, but there is a strong paper inside this project.**

Three findings dominate everything else.

1. **The calibration is inverted.** PROBE-RM fires when the margin is *low*, but the calibrator
   always returns the *95th* percentile. The true false-positive rate of the headline gate is
   **95 %, not 5 %.** Every threshold, every FPR claim, and Fold B's behavioural number inherit
   this. (§1.3 **D0**)
2. **The headline metric is defined three incompatible ways**, and the paper reports a value its
   own equation does not produce. (§1.3 **D1**)
3. **The one measured effect sits inside the noise floor.** `Δ_cliff = 0.167` is a **13.6°**
   rotation — exactly what finite-sample estimation noise alone produces at n = 200. The control
   that would settle this has never been run and costs zero GPU time. (§1.3 **D2**)

But the same data contains a genuinely novel positive result. The refusal direction is
**heavy-tailed** (excess kurtosis 19.3) while the quantization damage to it is **statistically
indistinguishable from isotropic noise** (excess kurtosis 0.20). That contradicts the field's
"critical weights" narrative and points to a predictive theorem:

> **Quantization does not rotate the safety direction — it adds broadband noise around it.
> Discriminability decays as SNR, and a smooth SNR decay pushed through a decision threshold
> produces an apparent cliff. The cliff is the sigmoid, not a phase transition.**

That theorem predicts the collapse point at **2.94 bits ≈ Q3_K_M** — exactly where the blueprint
pre-registered it and where the literature reports it — using **only benign data and weight
reconstruction error**. No harmful prompts, no judge model. Nobody has this. It is §3.

---

## 1. Verified correctness audit

Everything below was executed or recomputed directly, not read off the docs.

### 1.1 What is correct

| Claim | Verdict | Evidence |
|---|---|---|
| Test suite passes | ✅ **989 passed**, 4 warnings, 28.6 s | `pytest -q` |
| mypy strict clean | ✅ **54 files, no issues** | `mypy --strict cliffguard` |
| `Δτ = 0.00085` | ✅ exact (0.00085038) | recomputed |
| `Δ_cliff = 0.167`, `Δ_W = 0.014`, `Δ_B = 0.000` | ✅ reproduce from `.npz` | recomputed |
| Both `r̂` exactly unit-norm | ✅ | `‖r̂‖ = 1.0` |
| README reports H1 as **not accepted** | ✅ | commendable — do not weaken |

### 1.2 Stale claims

| Claim | Reality |
|---|---|
| "939 tests" | **989** |
| "mypy strict 53 files" | **54** |
| "ruff clean" | **34 errors** (all in newer uncommitted files; `cliffguard/` itself is clean; 17 auto-fixable) |
| Tier A = "RTX 5060 8 GB" | actual hardware is **RTX 3050 Laptop 6 GB** |

### 1.3 Defects, by severity

#### D0 — FATAL — the calibrator uses the wrong tail

`eval/threshold_calibrator.py::calibrate_threshold` unconditionally returns
`np.percentile(scores, (1 − fpr_target)·100)` — the **95th percentile**. It has no `fires_high`
parameter, though `empirical_fpr` in the same file does.

Four primitives fire **LOW** (`fired = score < τ`): **PROBE-RM** (`probe/rm.py:70`),
**PROBE-MT** (`probe/mt.py:63`), **TRIPWIRE-R** (`tripwire/r.py:65`),
**B-PROBE-CONSISTENCY** (`bprobe/consistency.py:89`).

Demonstrated on 4000 synthetic benign margins:

```
calibrate_threshold(...)      -> tau = 0.13296   (= 95th percentile)
empirical FPR, fires_high=False (PROBE-RM's actual rule)  = 0.9500   <-- ACTUAL
empirical FPR, fires_high=True  (what the 95th pct assumes) = 0.0500
correct tau for a fires-LOW gate = 5th percentile = 0.06746 -> FPR = 0.0500
```

**Consequences.** `τ_FP16 = 0.09742` and `τ_NF4 = 0.09827` are upper-tail values used as
lower-tail thresholds. The gate as calibrated blocks ~95 % of benign traffic. The README/paper
"FPR = 5 %" claim is unsupported by the code path. Fold B's ASR — computed by thresholding
against these same τ — inherits the error, which is a plausible contributor to the exact
`Δ_B = 0.000`. H2's "2 schemes within target" is not evidence of anything.

The 989 tests do not catch this because none asserts a *semantic* FPR after calibration.

**Fix:** add an explicit `fires_high` / `fires_low` declaration per gate; select the tail
accordingly; add a test asserting empirical FPR ≈ target for both directions; recompute Fold A.

#### D1 — FATAL — `Δ_cliff` has three mutually incompatible definitions

| Source | Definition | Value on real data |
|---|---|---|
| Blueprint §11.1 | `(Δ_q* − Δ_q)/Δ_q*`, `Δ_q = med(benign) − med(harmful)` — **relative separation loss** | **never computed** |
| `docs/math.md` + `eval/cliff_metrics.py` | `‖r̂_q − r̂_FP16‖/√2` — **normalised Euclidean** | **0.16699** |
| `docs/paper/main.tex` eq. (cliff) | `1 − ⟨r̂_q, r̂_FP16⟩` — **cosine distance** | **0.02788** |

Note `(3) = (2)²` exactly. **The paper reports 0.167 while its own equation yields 0.0279** —
i.e. **11.2 % of κ**, not the "67 % of cliff threshold" claimed at `main.tex:89`.

The blueprint's original separation-loss definition is the scientifically correct one, and the
only one for which κ = 0.25 was ever justified ("cuts a quarter or more of the FP16 safety
margin"). The code silently substituted direction-rotation, for which κ = 0.25 has no grounding.

#### D2 — FATAL — the measured effect is inside the noise floor; no control exists

`r̂` is re-extracted independently per scheme by difference-in-means over n = 200 per class in
D = 3072. Angular sampling error ≈ `√(2D/n)/g`, where `g = ‖μ_H − μ_B‖/σ`:

| per-class SNR `g` | rotation from **noise alone** | implied `Δ_cliff` |
|---|---|---|
| 20 | 23.1° | 0.283 |
| 30 | 15.1° | 0.186 |
| **observed** | **13.56°** | **0.167** |
| 50 | 9.0° | 0.111 |

The observed FP16→NF4 rotation is what pure finite-sample noise gives at g ≈ 33. Unless true SNR
materially exceeds ~35, `Δ_cliff = 0.167` measures the *estimator*, not the *quantizer*.

**The missing control costs zero new GPU time:** split the 200 FP16 prompts per class into two
disjoint halves, extract `r̂` twice *within FP16*, measure the rotation. If it is also ≈ 13°, the
NF4 result is null. **This is the single most important experiment in the project.**

Codex adds a related label-validity problem: the HH-derived "refused" flag inspects refusal-like
text in the *dataset's rejected response*, not whether the *target model* refused. The vector may
therefore be a harmfulness/topic direction, not a refusal direction.

#### D3 — FATAL — `Δ_B-cliff` is not behavioural

`five_fold_live.py` computes ASR as the fraction of prompts whose **PROBE-RM margin ≥ τ_q** —
the same geometric quantity, re-thresholded. `colab_helper.py`'s own docstring concedes it is
"NOT the StrongREJECT + Llama-Guard-3-8B judge stack required by the paper §11.3."

No completions are generated at all. A harmful prompt may be refused, deflected, partially
complied with, or fully complied with; a hidden-state margin cannot distinguish these. So the
metric cannot "break circularity", which is its entire stated purpose in §11.3. And because `τ_q`
is refit per scheme, any **common-mode** shift cancels by construction — only gap *narrowing*
survives. `0.000` is the expected output, not a finding.

#### D4 — FATAL — κ = 0.25 is grounded in a misattributed citation

`docs/math.md` grounds κ in "Egashira et al. (ICLR 2025) measured ASR deltas of Δ = 88.7 %,
85.0 %, 30.1 % between FP16 and GGUF-quantized models."

Verified against source (*Mind the Gap*, arXiv:2505.23786; and *Exploiting LLM Quantization*,
arXiv:2405.18137): those deltas are the yield of an **adversarial weight-crafting attack**. The
authors "develop an attack that trains the target malicious LLM while constraining its weights
based on quantization errors." That is CLIFFGUARD's own adversary **A2 (poisoned weights)** — a
supply-chain threat. H1 measures passive PTQ degradation of an honestly-trained model.

Calibrating a passive-phenomenon threshold with an active attack's effect size is a category
error, and plausibly why κ is set far too high for what passive quantization actually does.
*Mind the Gap* was an **ICML 2025** poster, not an ICLR 2025 benign-degradation demonstration.

#### D5 — FATAL (for a systems paper) — the threat model conflates three problems

The work mixes **direct harmful requests/jailbreaks**, **indirect prompt injection** (agent/RAG
trust boundary), and **poisoned checkpoints**. These have different principals, trust boundaries,
datasets, and success conditions. Spotlighting (Hines et al.) requires a trusted/untrusted data
boundary and is not a generic harmful-request detector. The title says "prompt injection" while
H1/Fold B measure jailbreak refusal. Security reviewers will attack this first.

#### D6 — SERIOUS — marginal calibration does not compose

Even granting per-gate calibration, an OR over 11 gates each at marginal FPR 0.05 has FPR
**≈ 1 − 0.95¹¹ = 0.431** under independence (up to 0.55 by union bound), not 0.05. AND/weighted
policies depend on cross-gate dependence, which may itself shift across schemes. The
"decoupling" claim does not survive composition — and composition is the whole architecture.

Separately: **VESTIBULE-LZ and VESTIBULE-PS are input-only.** Their scores do not depend on the
model's quantization scheme at all. Per-scheme thresholds for them have no causal justification.

#### D7 — SERIOUS — n = 400 cannot resolve ε = 0.02

At n = 400 and true FPR 0.05, binomial SE ≈ 0.0109 and the 95 % interval is ≈ **±0.021** —
already wider than the pre-registered ε = 0.02. Reaching ±1 pp needs n ≈ 1 825 per scheme
pointwise (≈ 3 300 with Bonferroni across six schemes); a distribution-free DKW bound for CDF
error ≤ 0.02 at 95 % needs ≈ 4 612. The prereg itself specifies |C| ≥ 2000; Fold A used **400**.

#### D8 — SERIOUS — the statistical tests do not match the estimands

- **H4** compares paired **binary** outcomes but uses **Wilcoxon signed-rank**. Correct:
  McNemar, paired bootstrap, or conditional/mixed-effects logistic. Also, selecting the
  "best single gate" and testing against it on the same data is selection bias.
- **H5** treats non-significance as **absence of effect** — invalid. Needs an equivalence /
  non-inferiority analysis with a pre-specified margin.
- The KS helper accepts arrays described as FPR distributions, while H2's statistic is a *range*
  of scheme-level FPRs. Different objects.

#### D9 — SERIOUS — the three-metric detector is dimensionally incoherent

`detect_cliff_boundary_three_metric()` requires `wasserstein_cliff ≥ κ = 0.25`. But `Δ_W` is a
W₁ distance in **raw cosine-margin units** (measured 0.0143), while κ is a **dimensionless
fraction**. Margins live on a ~0.1 scale, so `Δ_W` cannot approach 0.25. **The detector is
structurally incapable of ever firing.**

#### D10 — SERIOUS — the LinUCB CONDUCTOR is unevaluable and mis-specified

- The pre-registered reward gives **+1** for serving clean traffic, **−1** for serving an
  injection, **−0.2** for blocking benign — and **0 for a correct security block**, despite prose
  promising "+1 correct block."
- 12 arms × 14-dim context = **168 coefficients**. At a crude 10 observations each, ≈ 1 680
  labelled outcomes; if 1 % of requests yield reliable labels, ≈ 168 000 requests.
- The code reuses raw UCB scores as aggregation weights; UCB scores may be negative, making a
  weighted aggregate nonsensical. No arm-selection policy, propensity correction, or safe
  exploration.
- Regret intuition is stochastic-bandit (`√(dT)`) while the setting is declared adversarial.

#### D11 — MODERATE — `Δ_W-cliff` is W₂ in the papers and W₁ in the code

Blueprint §11.2 and `main.tex:446` say **W₂**; `docs/math.md:45` and the implementation use
**W₁**; and `cliff_metrics.py:105` docstring says "Wasserstein-**2**" three lines above
`W_1` at :107. The published `Δ_W = 0.014` is a W₁ value under a W₂ label.

#### D12 — MODERATE — `docs/math.md` κ-intuition contradicts its own normalisation

math.md says κ = 0.25 "corresponds to a cosine similarity of `1 − 2·(0.25)² = 0.875`". That
assumes normalisation by **2**; the code normalises by **√2**, for which `cos = 1 − κ² = 0.9375`.
Stated angle 28.96° is wrong; correct is **20.36°**. *(Independently derived by two reviewers.)*

#### D13 — MODERATE — Nepali dataset quality

5 fields contain Devanagari codepoints inside supposedly-romanized text — e.g.
`illegal_002.formal_nepali` → `"Kुनai byaktilai chaot pugaauna…"` (should be `Kunai`); also
`cyber_004.formal_nepali`, `illegal_005.formal_nepali`, `illegal_002.code_switched`,
`illegal_005.code_switched`. Codex's broader script-contamination check counts 8 of 76.

`formal_nepali` is romanized, not formal Devanagari — the field name is wrong. Dataset holds
**19** prompts while `run_nepali_eval.py`'s docstring claims "20 × 5 = 100"; actual is 95.

Critically: **19 base intents = 19 independent clusters, not 95 observations.** Treating the five
variants as independent is pseudoreplication.

#### D14 — MODERATE — prereg α conflict must be an amendment, not an edit

Known (C30): prereg says p < 0.05 for H4/H5; §4 mandates Bonferroni → 0.01; `stats.py` uses 0.01.
**A pre-registration cannot be edited after data collection.** Correct action is a dated deviation
note plus a change table, preserving the original.

#### D15 — MODERATE — `nepali_safety_phase_b.ipynb` has never been executed

Verified: **0 of 12 code cells** have an execution count or outputs; no results JSON exists. This
is fully-written, ready infrastructure sitting idle, and it is the cheapest new information
available to the project.

---

## 2. Literature position — what is actually still open

All verified by direct source fetch; none from memory.

| Work | What it establishes | Impact |
|---|---|---|
| **Quality Is Not a Safety Proxy Under Quantization** (arXiv:2606.10154) | 6 models, 4 families, **7-level GGUF ladder** + AWQ/GPTQ; refusal falls **12–68 pp**; safety neurons absorb **1.39×** more quantization error (p<5e-7) | **Largely scoops H1's measurement**, at a scale a 3050 cannot match. Cite it; stop re-establishing it. |
| **LiteLMGuard** (arXiv:2505.05619, rev. Mar 2026) | On-device prompt filter *explicitly for quantization-induced SLM risks*; 97.75 % acc, >85 % defense, **135 ms** | **Occupies CLIFFGUARD's stated value proposition.** Gap left: no per-scheme calibration, English-only, static classifier. |
| **CAQ / Safety-Preserving PTQ** (arXiv:2511.07842, Wee et al.) | Preserves alignment **at the quantization step** (W4A4; Llama/Qwen/Mistral) | **Attacks the premise.** If PTQ can preserve alignment, a front-end is not *necessary*. Must be engaged head-on. |
| **From Signal Degradation to Computation Collapse** (arXiv:2604.19884) | Two quantization failure modes; notes a 2-bit "performance cliff" | Adjacent but **capability-only, no safety, no SNR model, no predictive equation** — leaves §3 open. |
| **IndicJR** (EACL 2026) | **45 216 prompts, 12 South Asian languages incl. Nepali**, mixed/romanized forms | **Largest threat to the Nepali pivot.** A 19-prompt hand-authored set is not a benchmark next to this. Use IndicJR as the external set. |
| **IndicSafe** (arXiv:2603.17915) | 12 Indic languages, 6 000 prompts, cross-language agreement **12.8 %** | Multilingual safety, but **no quantization**. |
| **Critical Weight Protection** (arXiv:2601.12033) | Quantization safety damage "especially pronounced in **non-English** settings" | **Partially occupies the compound claim.** The pivot must be sharper than "quantization hurts non-English more." |
| Arabizi transliteration bypass | Claude refuses **98.46 %** standard Arabic vs **45.58 %** romanized | Romanized-script bypass **already shown for Arabic**. Nepali alone is replication. |

**Also:** Llama-Guard-3's official card lists English, French, German, Hindi, Italian, Portuguese,
Spanish, Thai — **not Nepali**. It cannot be the sole judge for the multilingual arm.

### What remains genuinely open

1. **A predictive theory of *where* the cliff is** — nobody forecasts the collapse bit-width from
   quantities measurable without harmful prompts or a judge.
2. **The mechanism dispute** — the field says damage concentrates in critical weights; **the data
   in this repo says isotropic** (§3.2). A real, evidence-backed disagreement.
3. **Quantization × romanized/code-switched input as a tested *interaction***, with an explicit
   interaction term — not two marginal negatives stacked.

---

## 3. The major finding and the theorem to build on

### 3.1 What the artefacts actually say

Recomputed from `r_hat_*.npz`:

```
cos(r̂_FP16, r̂_NF4)             = 0.9721   -> rotation of only 13.56°
change component ∥ to r̂_FP16    = 0.0279
change component ⊥ to r̂_FP16    = 0.2345   -> 8.4 : 1 orthogonal
Δ_cliff for RANDOM directions at D=3072 ≈ 1.000 ± 0.009
```

NF4 leaves the refusal direction **almost perfectly intact**, independently reproducing the 2026
finding that the refusal direction is *structurally retained* under quantization even as
behavioural refusal degrades. **The geometric-rotation arm of H1 is therefore predicted to fail by
prior work** — continuing to chase it spends scarce GPU time confirming a null.

### 3.2 Mechanism result — the damage is isotropic, not targeted

Observed perturbation vs an isotropic-noise null (400 matched-magnitude trials):

| concentration | observed | isotropic null | z |
|---|---|---|---|
| top 1 % of coords | 9.0 % | 8.3 % ± 0.4 | **+1.8** |
| top 5 % | 29.2 % | 27.9 % ± 0.7 | **+1.9** |
| top 10 % | 45.2 % | 43.9 % ± 0.7 | **+1.7** |

| quantity | excess kurtosis |
|---|---|
| `r̂_FP16` itself | **19.34** — strongly heavy-tailed; safety *is* outlier-channel coded |
| `r̂_FP16 − r̂_NF4` | **0.20** — essentially Gaussian |

**The refusal direction is outlier-concentrated, but the damage to it is indistinguishable from
isotropic noise.** Quantization does not preferentially destroy safety-critical channels; it
sprays broadband noise across all of them.

This contradicts the blueprint's own §8.2 hypothesis ("NF4's tails — where outlier weights live —
are coarsely quantized, exactly where safety lives") **and** the prevailing critical-weight
narrative (arXiv:2601.12033, and the 1.39× claim in arXiv:2606.10154).

> ⚠️ **Must resolve D2 first.** Sampling noise is *itself* isotropic. After the split-half control
> the honest claim is either "the quantization perturbation is isotropic" **or** "the observed
> perturbation is entirely sampling noise." Both are publishable; both kill the current H1 framing.

### 3.3 Proposed Theorem — Quantization-Induced Discriminability Decay

**Setup.** Residual stream `z(x) ∈ ℝ^D` at layer ℓ; under scheme q, `z_q(x) = z(x) + ε_q(x)`.

**Premise P1** (supported by §3.2, pending D2): `ε_q` is approximately isotropic and uncorrelated
with the safety direction — `E[ε_q ε_qᵀ] ≈ σ_q² I_D`, `E⟨ε_q, r̂⟩ ≈ 0`.

Projecting onto a fixed unit direction, `⟨ε_q, r̂⟩ ~ N(0, σ_q²)`. Class-conditional margins
`N(μ_B, s²)`, `N(μ_H, s²)` at FP16 become `N(μ_·, s² + σ_q²)`. Hence:

> **Theorem.** With `η_q = σ_q²/s²` the quantization noise-to-signal ratio on the safety subspace,
>
> **d′(q) = d′(FP16) · (1 + η_q)^(−1/2)**,  and at fixed FPR α,  **TPR(q) = Φ(d′(q) − z₁₋α)**.

**(C1) The cliff is the sigmoid.** `d′` decays smoothly and monotonically in `η_q`; TPR is a
sigmoid of `d′`. The apparent discontinuity is the threshold nonlinearity — nothing in the weights
is discontinuous.

**(C2) It is sharp *in bit-width* because quantization error is exponential in bits.** With
`σ_q² ∝ 4^(−b)`, `η` **quadruples per bit removed**:

| bits | η_q | d′ | TPR@5%FPR | Δ vs prev |
|---|---|---|---|---|
| 16 | 0.000 | 2.500 | 0.804 | — |
| 8 | 0.001 | 2.499 | 0.803 | −0.000 |
| 6 | 0.019 | 2.477 | 0.797 | −0.006 |
| 5 | 0.075 | 2.411 | 0.778 | −0.019 |
| 4 | 0.300 | 2.193 | 0.708 | −0.046 |
| 3.5 | 0.600 | 1.976 | 0.630 | −0.078 |
| **3** | 1.200 | 1.685 | 0.516 | **−0.114** |
| **2.5** | 2.400 | 1.356 | 0.386 | **−0.130** |
| 2 | 4.800 | 1.038 | 0.272 | −0.114 |

**(C3) The cliff location is predictable with no safety data.** Solving `d′(q) = z₁₋α`:

> **b\* = 4 − log₄( ((d′₀/z₁₋α)² − 1) / η₄ )**

With `d′₀ = 2.5`, `α = 0.05`, `η₄ = 0.30`: **b\* = 2.94 bits ≈ Q3_K_M** — exactly the
pre-registered location. The theory *derives* it instead of assuming it. Every input (`d′₀`, `α`,
`η_q`) is measurable from **benign data and weight reconstruction error alone**.

### 3.4 Why this is the better paper

- It **explains** the cliff instead of re-measuring it (arXiv:2606.10154 measured it better).
- It **predicts** an unseen scheme's collapse point — a new capability, cheap to validate.
- It **survives the negative result**: `Δ_cliff = 0.167` becomes evidence *for* P1 (direction
  retained, noise isotropic) rather than a failed hypothesis.
- It **fits 6 GB**: one 1–3 B model across a GGUF ladder; benign prompts for prediction, one small
  harmful set for validation.
- It reframes CLIFFGUARD from "11 unvalidated gates" to "a measurement instrument that produced a
  theory" — and the instrument already exists.

---

## 4. Scope: what to cut

All three reviewers independently converged on aggressive cutting. 11 primitives × 4 tiers ×
9 adversaries × 5 hypotheses is a lab's multi-year roadmap.

**Keep:** PROBE-RM (only real-inference primitive; the instrument); VESTIBULE-LZ/PS (free, already
run, valuable as a documented negative result); **Tier A only**; **H1 reframed** as the
discriminability-decay theorem; adversary **A7** (+ **A6** if the multilingual arm stays).

**Cut:** CONDUCTOR LinUCB + Fold D (D10); LOOKOUT-JG and the full 8 B judge stack except a
small-N spot check; TRIPWIRE-H/R (needs streaming hooks + per-tier KenLM, plus an unresolved
ADWIN-vs-Page-Hinkley conflict, C31); ATTEST-WH (appendix feature, not a research question);
B-PROBE-* and H3; **H4, H5** (depend on Fold C and hardware that does not exist); Tiers B/C/C+
(need a Pi 5 / RK3588 / Jetson — H5 is untestable by construction); BCN-2; the nine-adversary
matrix.

**Minimum viable system:** `input → one small external guard → quantized model → output evaluator`,
plus one internal direction probe for diagnosis.

**Net:** 1 theorem + 1 hypothesis, 2 reported primitives, 1 tier, 6–7 schemes, 2–3 model families.

---

## 5. On the FPR-Decoupling Theorem

All three reviewers reached the same verdict independently: **as stated it is near-tautological.**

Setting `τ_q` to the empirical `(1−α)` quantile of the benign distribution at each scheme yields
FPR ≈ α at every scheme *by construction* — for **any** scoring function, including a random
number generator. The §14.2 proof cites Glivenko–Cantelli, which is exactly the statement that
empirical quantile estimators are consistent. H2/H3 as written test that the calibration code
works — and per **D0** it currently does not.

`docs/math.md` concedes this in its own "CRITICAL HONEST SCOPE" note: *"TPR is NOT decoupled…
This is precisely what H1 measures."* H1 carries all the empirical content.

**Salvageable underneath:**
1. **Sample efficiency, not portability** — how much benign data each scheme needs to hit the same
   ε band is scheme-dependent and non-trivial.
2. **Quantile degeneracy at extreme bit-width** — heavy rounding can collapse benign scores onto
   discrete atoms, creating ties at the quantile boundary so α becomes unreachable for
   *structural* rather than sampling reasons. Real, falsifiable, quantization-specific — and
   invisible in current data (Fold A scores were continuous).
3. **The black-box corollary** — `TPR(B-PROBE-LOGIT) ≤ TPR(PROBE-RM)` is a genuine
   data-processing-inequality claim; keep it, unbundled from the FPR tautology. (Note: the DPI
   bounds the *optimal* test, not a particular finite learned classifier.)

**Recommendation:** demote H2/H3 to a one-sentence stated precondition in Methods.

---

## 6. Remaining work

### 6.1 Immediate — zero new GPU time

| # | Action | Effort | Why |
|---|---|---|---|
| **1** | **Fix the calibration tail**; add `fires_high`/`fires_low` per gate + an FPR assertion test | 3 h | **D0** — invalidates every current number |
| **2** | **Noise-floor control**: split-half `r̂` within FP16 | 1 h | **D2** — decides whether any result is real |
| **3** | Reconcile `Δ_cliff` to **one** definition across all four locations | 2 h | **D1** — fatal reviewer catch |
| **4** | Fix the misattributed Egashira grounding for κ | 1 h | **D4** |
| **5** | Fix `detect_cliff_boundary_three_metric` dimensional bug | 1 h | **D9** — cannot currently fire |
| **6** | Fix W₁/W₂ labelling | 30 m | **D11** |
| **7** | Fix κ-intuition formula (`1 − κ²`, 20.36°) | 15 m | **D12** |
| **8** | Repair Nepali script contamination; fix 19-vs-20 count; rename `formal_nepali` | 2 h | **D13** |
| **9** | File a dated prereg **amendment** (never a silent edit) | 1 h | **D14** |
| **10** | `ruff --fix`; refresh README counts (989/54); correct Tier A hardware | 30 m | §1.2 |

### 6.2 Next — cheap experiments producing new information

| # | Experiment | Compute | Produces |
|---|---|---|---|
| **11** | **Save raw margin distributions**, not just thresholds | trivial | `d′₀` — required by the theorem, currently unrecoverable |
| **12** | Extend Fold B down a **matched GGUF ladder** (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K) from **one pinned checkpoint + one pinned llama.cpp commit** | ~1 day | The actual predicted cliff zone. NF4 was the scheme *least* likely to show it, and must not share an ordinal axis with GGUF k-quants. |
| **13** | Measure `η_q` directly from weight reconstruction error per scheme | 2 h | The theorem's independent variable |
| **14** | Fit `d′(q) = d′₀(1+η_q)^(−1/2)`; test predicted vs actual `b*` | 2 h | **The paper's central result** |
| **15** | **Generate real completions**; score with StrongREJECT + human audit on a stratified sample | ~4 h + GPU | A genuinely behavioural `Δ_B` (**D3**) |
| **16** | **Run `nepali_safety_phase_b.ipynb`** (written, never executed) | Colab/3050 | First quantization × language data (**D15**) |

### 6.3 Data still to collect

- **Raw margin distributions per scheme** (benign + harmful) — the biggest single gap.
- **A matched GGUF ladder** for 2–3 edge models (Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct,
  Gemma-3-4B-it). All fit 6 GB at low precision; use Colab for high-precision references.
  Mistral-7B (~14.5 GB BF16) is **not** a realistic local reference.
- **Benign: ≥ 2 000 calibration + 2 000 independent test** (prereg minimum; Fold A used 400).
  Split general benign from **hard benign** (XSTest safe split, OR-Bench) — a single scalar FPR
  hides catastrophic overblocking of medical/security/quoted-harm content.
- **Harmful: 400–500 unique held-out behaviours** after *semantic* dedup across
  HarmBench / JailbreakBench / StrongREJECT / AdvBench. Keep ≥ 200 separate for probe/layer
  development. Unit of analysis = base intent, not templated variant.
- **If the multilingual arm stays:** ≥ 300 independent intents paired across English / Devanagari
  Nepali / romanized Nepali / code-switched; **IndicJR Nepali as the external set**; two native
  annotators + adjudication. Do **not** use Llama-Guard-3 as the sole Nepali judge.

---

## 7. On the multilingual pivot

**For:** the 0/95 result is clean and honestly explainable — VESTIBULE-LZ/PS target adversarial
suffix entropy and injection scaffolding, and these are plain natural-language harmful requests.

**Against:** it is far weaker than it looks. Every language scores 0/19 **including English**, so
it demonstrates that two heuristics do not detect direct harmful requests — which is expected and
already documented. It says nothing about Nepali safety alignment. It is 19 clusters, not 95
observations. IndicJR already provides 45 216 prompts across 12 South Asian languages including
Nepali. The romanization bypass is already published for Arabic.

**Verdict — second experiment, not the thesis.** The defensible claim is not "Nepali evades
guardrails" but the **interaction**: after controlling for intent, model, and decoding, does lower
precision disproportionately increase harmful compliance for romanized/code-switched Nepali
relative to English? That needs a factorial design with an explicit **interaction term** and a
mixed-effects logistic model with random intercepts for intent — not two marginal negatives.
Do not call it "multiplicative" unless the interaction is significant and super-additive.

**The unifying prediction:** the discriminability theorem *predicts this interaction*. If
low-resource input starts at lower `d′₀`, it crosses `z₁₋α` at a **higher** bit-width — i.e.
**the cliff arrives earlier for Nepali than for English.** Sharp, novel, falsifiable, and it makes
both halves of the project one paper.

---

## 8. Honest verdict

The **safety cliff exists** in the literature — but "it exists and here is a defense" is no longer
a contribution: arXiv:2606.10154 measured it at greater scale and LiteLMGuard already ships an
on-device quantization-aware filter.

The **11-gate / 4-tier / 9-adversary architecture is engineering, not science**; per the README's
own table, 7 of 8 components have never seen real inference. It cannot be the contribution.

What is genuinely yours:

1. A **mechanism result** dissenting from the field's critical-weight narrative, from data already
   on disk (§3.2).
2. A **predictive theorem** forecasting cliff location from benign data alone (§3.3) — nobody has
   this; the nearest work is capability-only and non-predictive.
3. A **negative result** on stateless injection gates vs code-switched low-resource input (§7),
   properly scoped.

That is a real paper — smaller than currently planned, and achievable on an RTX 3050 in 3–6
months, which the current scope is not.

**Realistic venue:** not a USENIX/S&P/CCS main track on this timeline. Target an ACL-family
workshop on multilingual NLP / safety / security, or an ACL/EMNLP Findings or short-paper track
once the dataset, statistics, and artifact are solid. SaTML 2027's listed deadline (29 Sep 2026)
is too close for this rebuild.

**Do first:** fix the calibration tail (**D0**), then run the split-half noise control (**D2**).
Every other number in this project is conditional on those two.

---

*Three independent reviews. All numbers recomputed from repo artefacts; all citations fetched from
source. No repo files were modified.*
