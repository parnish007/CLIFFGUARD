# CLIFFGUARD — Theory Panel: The Theorem-Bearing Core

**Date:** 2026-08-02
**Panel:** Claude Opus 5 · GPT-5.6-sol @ xhigh (Codex) · Fable 5 — run independently, then reconciled.
**Question put to the panel:** does the cliff generalize beyond safety (security, reasoning)? Is
there a general law? Can the cliff be *cancelled*? Which theorems are actually provable?
**Predecessor:** `docs/research_audit_2026-08.md` (defects D0–D15, literature position).

---

## 0. Panel verdict in one paragraph

The seed theorem generalizes, but **not** as the sweeping law it first appeared to be. Two of the
three headline candidates were killed as trivial or false on inspection. What survived is narrower,
harder, and better: a **rate–distortion result for behavioural subspaces** — an impossibility bound
saying no post-hoc function of a frozen quantized checkpoint can restore full-precision behaviour,
paired with an achievability construction showing that a small **pre-quantization sidecar** can
preserve a chosen behaviour exactly, with an explicit bit budget. Both Codex and Fable converged on
this independently from different directions. It is the only candidate that is neither
tautological nor already occupied.

---

## 1. What the panel killed

Protecting you from repeating the FPR-Decoupling mistake was an explicit panel objective. Three
candidates failed.

| Candidate | Verdict | Why |
|---|---|---|
| **`P_T = p^T`** (chain accuracy compounds) | **Trivial (i)** | Textbook series-system reliability algebra. Already used in the quantized-reasoning literature. *Never headline this.* |
| **"Tails and compositions always cliff before means"** | **FALSE as stated** | No distribution-free ordering exists. Means are themselves aggregates of threshold decisions; arXiv:2604.19884 already reports a 2-bit *capability* cliff. Perturbations can even help. |
| **Affine/threshold recalibration cannot restore d′** | **Trivial (i), but keep as a lemma** | Any strictly increasing transform `g` maps `{m ≥ τ}` to `{g(m) ≥ g(τ)}`, so the ROC — and hence d′ — is invariant. Standard signal-detection theory. Useful only as an *impossibility* baseline. |
| **Repeated sampling gives `d′·√T`** | **FALSE** for weight quantization | See §3.1. This was my own error, caught by both panelists. |

**The `1 − (1−u)^L` "one bad token" security law** was also rejected: elementary algebra, and
tokens are not independent vulnerability events.

---

## 2. What survived — the generalization results

### 2.1 The reasoning-depth law (three independent derivations agreed)

Model a T-step chain as T thresholded readouts. Fix the target as retaining a constant *fraction*
of the chain's own FP16 success (not an absolute bar — otherwise ceiling effects dominate).
Linearizing `log p(η)` around η = 0 and solving gives tolerable noise `η*_T ∝ 1/T`, hence:

> **b\*(T) = b\*(1) + log₄(T) = b\*(1) + ½·log₂(T)**
>
> **Every 4× increase in reasoning-chain length costs exactly +1 bit of precision.**

| source | derivation | result |
|---|---|---|
| Opus 5 | numeric solve of `T·e_q ≈ 1` with `e_q = e_0^{1/(1+η)}` | +0.88 to +1.01 bits per 4× (T = 8→32) |
| Fable 5 | Mills-ratio-weighted linearization, `η*_T = log(1/c)/(Tκ₁)` | `b*_T = b*₁ + log₄T` |
| Codex | fixed relative chain retention | `b*_T = ½log₂T + O(1)` |

Three methods, one law. Worked example (Fable, using the audit's own constants): b*₁ ≈ 2.79,
b*₄ ≈ 3.79, b*₁₆ ≈ 4.79, b*₆₄ ≈ 5.79 bits. **A 64-step agentic chain needs ~3 more bits than a
single-shot judgement** to hold the same relative reliability.

**Counter-intuitive corollary (Fable).** Widening in η scales as `1/(Tκ₁)`, but `dη/db ∝ η(b)` and
`η*_T ∝ 1/T` — the two factors of T **cancel**. So the reasoning cliff is **displaced right by
log₄T but is *not* sharper**. Most people's intuition (mine included) says longer chains cliff more
abruptly. The math says: same shape, shifted. That is a falsifiable, non-obvious prediction.

**Caveats, stated up front:** holds only in the small-η linearization; near the real cliff
(η ≈ 0.3–1.2) the exact `(1+η)^(−1/2)` form deviates substantially. Assumes conditionally
independent, comparable steps — false for real CoT, which has self-correction and correlated
errors. GGUF k-quants do not form a clean scalar-bit ladder.

**Literature status.** The *qualitative* direction is thoroughly scooped — arXiv:2504.04823,
2505.11574, 2501.03035, 2606.00206, 2606.25519, 2505.20276 all report it empirically (e.g. 3-bit
AWQ takes MATH-500 from 85.6 % → 47.0 % while CoT length inflates 5.2K → 23.4K tokens; 4-bit loses
up to 59 % on long-context). **But I verified two of the leading papers directly: both are purely
empirical, with no equations, no scaling law, no noise model.** The quantitative law appears open.

### 2.2 Tail inflation (security / rare failures)

For a rare failure crossing threshold `Δ` in the tail, variance inflation `s² → s²(1+η)` gives:

> **log p_q / log p_0 → 1/(1 + η_q)** — verified numerically (ratio → 0.800 for η = 0.25 as
> p₀ → 0: 0.864, 0.839, 0.823, 0.811, …)
>
> equivalently **R(η) = p_q/p_0 ≈ √(1+η)·exp[(Δ²/2)·η/(1+η)]**

Measured inflation at η = 0.5:

| FP16 failure rate | inflation |
|---|---|
| 1e-1 | ×1.5 |
| 1e-2 | ×2.9 |
| 1e-3 | ×5.8 |
| 1e-4 | ×12.0 |
| 1e-5 | ×24.9 |

**The rarer the event, the more violently it inflates at the same η.** This is a mechanistic
explanation for "quality is not a safety proxy": perplexity/MMLU sit at the distribution centre
where relative change is small; catastrophic failures sit in the tail where the *ratio* explodes.

**Scope limit (Fable, important).** This requires genuine tail depth (Δ ≳ 2–3, baseline below
~1–5 %). The audit's own safety numbers (TPR 0.80 → 0.27) are **not** in that regime — that swing
is fully explained by the base `(1+η)^(−1/2)` decay, no Mills ratio needed. Conflating the two
would be exactly the FPR-Decoupling error again: one phenomenon dressed as two theorems.

### 2.3 The salvaged version of the "unifying law"

Both panelists replaced the false blanket claim with a **conditional margin–composition principle**:
given the joint margin distribution, the measured perturbation, and the composition rule (single
threshold / T-fold conjunction / rare tail), the degradation follows. Predictive only after those
are specified — a *framework*, not a theorem. Its decisive test is **out-of-domain / cross-quantizer
calibration transfer**, not in-sample correlation.

---

## 3. Can the cliff be cancelled?

### 3.1 The crux, resolved: quantization error is deterministic, not stochastic

For fixed input `x` and fixed quantized weights, `z_q(x)` is a **deterministic function** — same
input, same error, every time. Temperature/sampling randomizes only final token selection,
downstream of the already-corrupted representation.

> **Repeated sampling of the same prompt cannot average away ε_q. There is nothing stochastic to
> average.** Self-consistency fixes stochastic errors, not systematic bias. My `d′·√T` conjecture
> was wrong.

Population-isotropic (across inputs) and per-input-deterministic are *not* contradictory — they
describe different objects. Two consequences: (a) **paraphrase** ensembling might reduce η_eff if
ε_q decorrelates across paraphrases (untested, cheap to check); (b) because the error is
deterministic, it is **computable and therefore correctable** — which is the whole opening.

### 3.2 How much can recalibration recover? — 1 %

Measured directly on your saved `r_hat_*.npz`, decomposing the FP16→NF4 damage:

```
systematic (parallel to r̂, a BIAS)       = -0.02788   -> correctable by affine recalibration
input-dependent (orthogonal, NOISE)      =  0.23450   -> NOT correctable post-hoc
ratio                                    =  8.4 x
```

> **99 % of the damage is the irrecoverable component. Threshold recalibration can address ~1 %.**

This is the quantitative death certificate for the recalibration-based framing of CLIFFGUARD, and
it is measured from data you already have.

### 3.3 Where the DPI bites — and where it does not

With the Markov chain `Y → z(x) → z_q(x)`, the data-processing inequality gives
`I(Y; z_q) ≤ I(Y; z)`: **no function of `z_q(x)` alone** can recover the lost discriminability.

**But DPI does not bind once the corrector's input is augmented** with quantities that were never
inside that bottleneck — the weight delta `ΔW = W_FP16 − W_q` (free at build time, if you hold both
checkpoints) and the layer input `a(x)` (already computed in the forward pass). Recovery is then
bounded by `I(Y; z_q, ΔW, a(x))`, which can approach `I(Y; z)`. This is not beating information
theory; it is using side information. **This is the entire opening for mitigation.**

### 3.4 The targeted rank-k sidecar

For a layer feeding the residual position where `r̂` is read, the projected error is
`r̂ᵀε_q(x) = r̂ᵀ(W − W_q)·a(x)`. Define the **precomputable** row vector `v = r̂ᵀ(W − W_q)`:

> **m_corrected(x) = r̂ᵀz_q(x) + v·a(x) ≈ r̂ᵀz(x)**

One extra dot product at inference. No training. Storage: `O(k·d·L)` — for k=1, d=3072, L≈30 that
is ≈ 90K floats ≈ **180 KB** (Fable's correction to my "a few KB" estimate — still negligible
against a multi-billion-parameter model).

**How this differs from prior work.** LQER (arXiv:2402.02446), LoftQ, RILQ, SERQ approximate the
*entire* weight-error matrix at rank r to restore *general* fidelity. This corrects **one
task-defined direction exactly** — narrower, and therefore far cheaper. That distinction is real.

**But the practical niche is occupied.** Q-resafe (ICML 2025, arXiv:2506.20251) restores ASR from
70–80 % to 13–14 % with public code; Q-realign (arXiv:2601.08089) recovers alignment on a 7B in
40 min on one RTX 4090; CAQ (arXiv:2511.07842) preserves alignment at quantization time. All are
learned, weight-updating methods. **You cannot win on ASR-recovery numbers against an ICML defense
with a repo.** Position the patch as a **mechanism test**, not a competing product.

### 3.5 The theorem that is actually novel

Neither panelist found any single equation above to be category (iii) alone. The novel package is
their *combination*:

> **Behavioural Subspace Preservation Theorem (candidate).**
> 1. **Impossibility.** No algorithm seeing only a frozen quantized checkpoint can uniformly
>    reproduce all FP behaviours — quantization cells contain behaviourally distinct FP weights
>    (a collision argument, strengthened by DPI).
> 2. **Achievability.** If a behaviour is mediated by a causally sufficient k-dimensional
>    statistic, a rank-k sidecar saved *before* quantization preserves it, at
>    `Θ(k·d·log(R/ε))` bits.
> 3. **End-to-end bound.** Lipschitz propagation converts subspace reconstruction error to score
>    error; the FP margin CDF converts score error to behavioural decision-disagreement.

Proof program: projection → upper bound; covering of correction factors → achievability;
packing in the `k·dᵢ`-ball → matching lower bound; Lipschitz → score error; margin CDF → behaviour.

**It is only non-trivial if** upper and lower bounds use the *same* operational distortion, the
assumptions are fixed before seeing results, and causal sufficiency is empirically challenged.
Presenting `P = UUᵀ(W−Q)` and calling it a "Safety Restoration Theorem" would be dressing up linear
algebra — the exact failure mode of the FPR-Decoupling Theorem.

---

## 4. The panel's one genuine disagreement

Both converged on the rank-k sidecar as the core, but paired it differently.

**Fable — lead with the diagnosis.** The 2026 literature *contradicts itself*: arXiv:2606.10154
reports refusal falls of 12–68 pp, while arXiv:2606.29581 (verified: Prasad & Pal, 8 models,
5 families, 144 configs, ~2.0M responses, six-judge ensemble) finds **AWQ INT4 keeps ASR within
~1.6 pp of FP16 for 7 of 8 models**. The reconciling mechanism is premise P1 itself:

> **Isotropy is a property of the quantizer, not of quantization.** Naive RTN / NF4 / vanilla GGUF
> k-quants have no reason to correlate their error with any behavioural direction → isotropic →
> cliff. Activation-aware schemes (AWQ, GPTQ) explicitly protect high-salience channels → *an*isotropic
> → should **not** cliff.

I checked: the AWQ mechanism is well understood at the *capability* level, but nobody has framed it
as isotropy-vs-anisotropy **relative to a behavioural subspace**, and current sources explicitly note
that AWQ/GPTQ "require direct safety validation." **This is open.**

**Codex — lead with the theorem.** Commit to the rate–distortion result; make exactly three claims
(impossibility, achievability, empirical-efficiency-or-principled-failure); require at least one
**non-safety** behaviour to justify generality. Suggested title:
*"Behavioral Rate–Distortion for Quantized Language Models: Limits and Low-Rank Side Information."*

**Resolution — they compose.** Fable's diagnosis tells you *where the theory applies*; Codex's
theorem is the formal core; my 99 % result is the motivation for why anything beyond recalibration
is needed. One paper, three acts.

---

## 5. Recommended plan

### Stage 0 — mandatory gate (~1 h, zero GPU)
The audit's **split-half noise-floor control** (D2). Extract `r̂` twice within FP16 on disjoint
halves. If the rotation is also ≈ 13°, the isotropy premise P1 is unproven and *everything above*
must be reframed as conditional. **Nothing else should run before this.**

### Stage 1 — the cheap decisive experiment (~half a day, reuses 100 % of existing code)
Quantize the same model with **AWQ/GPTQ** alongside the existing NF4/GGUF checkpoints. Rerun the
audit's §3.2 pipeline verbatim: excess kurtosis of the perturbation, 400-trial matched-magnitude
isotropic null, top-1/5/10 % concentration z-scores.

- **Statistic:** AWQ's concentration/kurtosis z vs its own isotropic null, compared to NF4's z ≈ 1.8.
- **Falsification:** if AWQ's damage is significantly non-isotropic while NF4's is not, the
  scheme-dependent-isotropy claim is confirmed and it **mechanistically resolves a dated 2026
  literature contradiction**. If AWQ is *also* isotropic, that is equally informative — the split
  must come from judges or datasets instead.

Decisive either way, and it is the single highest value-per-hour experiment available.

### Stage 2 — the constructive test (~1 day)
Compute `v = r̂ᵀ(W_FP16 − W_q)` for the layers feeding `r̂`'s extraction point, on whichever scheme
Stage 1 confirms is isotropic. Apply `m_corrected(x) = m_q(x) + v·a_q(x)`.

- **Statistic:** recovery fraction `R = (d′_corr − d′_q)/(d′_FP16 − d′_q)`.
- **Mandatory control:** the same correction magnitude along a *random* direction of matched norm.
- **Falsification:** if `R` does not exceed the random-direction control (bootstrap CIs overlap),
  the deterministic-linear-correction model is refuted. If `R` ≫ control, the mechanism is
  validated — reported as theory validation, **not** as competing with Q-resafe on ASR.

### Stage 3 — generality (the secondary behaviour)
Codex is right that a safety-only result will not carry a rate–distortion paper. Add **controlled
reasoning depth** (GSM-Infinite or a synthetic arithmetic/program-execution generator with
controlled semantic depth; 6–8 precision points, 100–300 items per depth, greedy decoding, a step
verifier). Fit on shallow depths, then **predict deep-chain collapse without refitting** — that is
the real test of `b*(T) = b*(1) + log₄T`. Crucially, **separate semantic depth from output-token
count**, since quantized models inflate token counts independently.

### Ranked directions

| # | Direction | Novelty | Feasible on 3050+Colab | Theorem-bearing |
|---|---|---|---|---|
| 1 | **Isotropy is scheme-dependent** (resolves the 2026 split) | High | Very high — zero new eval code | High |
| 2 | **Rank-k behavioural sidecar + rate–distortion bounds** | High | High — no training, weights on disk | High |
| 3 | **Reasoning-depth law `b*(T)=b*(1)+log₄T`** | Qualitative scooped; quantitative law open | Medium — needs a depth-controlled benchmark | Medium-high |
| 4 | Security tail inflation (passive PTQ, not attack) | Medium — classical math, new application | Medium — needs executable CWE tests + adjudication | Medium |
| 5 | Frozen vs dithered quantization ensembles | Low-medium — standard statistics | High | Low |

**Commit to 1 + 2, with 3 as the generality experiment.**

### Venue
Not S&P/CCS/USENIX. If the theorem is tight, the sidecar works across multiple architectures *and*
behaviours, and comparison to LQER/QERA/Q-resafe/CAQ is serious → **ICLR/ICML/NeurIPS** is
reachable. Otherwise **TMLR** or **Findings of ACL**. A NeurIPS/ICML/ICLR workshop on efficient
inference or trustworthy ML is the natural first target — "why does this correction work" is native
framing there.

---

## 6. Honest bottom line

Your instinct to look for a general law and a cancellation mechanism was correct, and it is where
the real paper is. But two of the three obvious general laws are trivial or false, and the
cancellation is fundamentally limited: **99 % of the damage cannot be undone post-hoc, and the
1 % that can is not where the safety loss lives.**

What is left is genuinely yours and genuinely open:

- **The cliff is a property of the quantizer, not of quantization** — testable in half a day with
  code you already have, and it resolves a live contradiction in the 2026 literature.
- **The damage is deterministic, hence computable, hence correctable — but only with side
  information saved before quantization.** That converts an impossibility result into a
  constructive one, and the boundary between them is a real theorem.

Do Stage 0 first. Everything else is conditional on it.

---

*Three independent panel reviews, reconciled. All derivations recomputed; all citations fetched from
source. No repo code or results were modified.*
