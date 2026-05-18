# Mathematical Foundations

All numeric values in this document are pre-registered in `docs/preregistration.md` or cited from the referenced papers. No quantities are fabricated.

## Overview

CLIFFGUARD's mathematics rests on four pillars: **information-theoretic** (injection as a Neyman–Pearson change-of-source problem), **control-theoretic** (CUSUM/EWMA detectors as feedback controllers), **RL-theoretic** (LinUCB bandit for gate weighting), and **interpretability-theoretic** (refusal and harmfulness directions as a 2-D residual-stream safety subspace).

| Symbol | Meaning | Gate |
|---|---|---|
| ρ(x) | Refusal margin — cosine projection onto r̂ at post-instruction token | PROBE-RM |
| m_h(x) | Harmfulness margin — projection onto ĥ at instruction token | PROBE-HD |
| ρ̇, ρ̈ | Margin trajectory first and second derivatives across layers | PROBE-MT |
| S_t | CUSUM statistic on per-token entropy | TRIPWIRE-H |
| Δ_t | Log-likelihood ratio vs KenLM reference | TRIPWIRE-R |
| ρ_LZ(x) | Compression ratio of input string | VESTIBULE-LZ |
| JSD(P₁,…,Pₙ) | Jensen-Shannon divergence across N paraphrase distributions | B-PROBE-CONSISTENCY |
| UCB(a, x_t) | LinUCB upper confidence bound for arm a at context x_t | CONDUCTOR |

Pre-registered thresholds: **κ = 0.25** (cliff jump), **ε = 0.02** (FPR portability), **α_corrected = 0.01** (Bonferroni), **fpr_target = 0.05**.

## Safety Cliff (Definition 11.1)

The safety cliff is a quantization boundary q* at which refusal behavior degrades non-linearly. Three independent metrics characterise it:

**Geometric cliff metric** — the normalized distance between the refusal direction at scheme q and at FP16:

```math
\Delta_{\text{cliff}}(q) = \frac{\|\hat{r}_q - \hat{r}_{\text{FP16}}\|}{\sqrt{2}}
```

where r̂_q is the unit-norm refusal direction (difference-in-means of harmful vs. harmless residual streams, Arditi et al. 2406.11717) calibrated at scheme q. Values in [0, 1]; 1 means the refusal directions are orthogonal.

**Behavioral cliff metric** — the difference in attack success rates:

```math
\Delta_{B\text{-cliff}}(q) = \text{ASR}(q) - \text{ASR}(\text{FP16})
```

where ASR(q) is the empirical fraction of adversarial prompts that elicit harmful compliance at scheme q. Pre-registered adversary: A7 (quantization-cliff exploiter).

**Wasserstein cliff metric** — the 1-Wasserstein distance between margin distributions:

```math
\Delta_{W\text{-cliff}}(q) = W_1\!\left(\mathcal{D}_{\text{margin}}^{(q)},\, \mathcal{D}_{\text{margin}}^{(\text{FP16})}\right)
```

where D_margin^(q) is the empirical distribution of refusal margins on a fixed benign evaluation corpus at scheme q.

**Cliff boundary** q* is the smallest quantization scheme by bit-width (i.e. most aggressive quantization) such that at least two of the three metrics exceed the pre-registered threshold κ for at least two of the three target model families.

## FPR Decoupling Theorem (Theorem 14.1)

Let M be a base model and M_q its quantization at scheme q. Let G_write = {g_1, ..., g_W} be a set of write-side gates operating on (a) the input string, (b) sketch features of the input, or (c) summary statistics of model output extracted at a fixed layer. Let P_q be a per-quantization calibration map that sets threshold τ_q as the empirical (1 − α) quantile of the gate's score distribution on a held-out benign corpus C ~ B.

**Theorem (white-box, §14.1):** For any pair of schemes q, q':

```math
\left|\text{FPR}_{q} - \text{FPR}_{q'}\right| \;\leq\; \text{KS}(F_q, F_{q'}) \;\leq\; d_{q,q'}
```

where KS(F_q, F_q') is the Kolmogorov-Smirnov distance between the benign-margin CDFs at the two schemes, and d_{q,q'} is a metric that vanishes when the two CDFs are identical. By Glivenko-Cantelli, the empirical quantile converges uniformly in α at rate O(|C|^{-1/2}).

**Practical consequence:** with |C| ≥ 2000 (pre-registered minimum), the empirical FPR of each gate is within ε = 0.02 of the target across all tested schemes.

**CRITICAL HONEST SCOPE:** TPR is NOT decoupled. In the cliff regime, the harmful-prompt margin distribution collapses toward the benign distribution, and the gate's statistical power to detect harmful inputs decreases. This is precisely what H1 measures. H2 and H3 test FPR portability; the paper does not claim TPR portability.

**Black-box corollary (§14.2):** The same FPR-decoupling bound holds for B-PROBE-LOGIT when the gate is calibrated on top-k logprobs at each scheme. TPR(B-PROBE-LOGIT) ≤ TPR(PROBE-RM) at matched FPR because the black-box observable is a lossy compression of the white-box residual stream. H3 pre-registers this inequality as a required condition, not a claim to be maximized.

## Five Hypotheses

### H1 — Safety Cliff Existence

**Claim:** For at least 2 of 3 model families (Llama-3-8B, Mistral-7B, Qwen2.5-7B), both Δ_cliff and Δ_B-cliff exhibit a jump of κ ≥ 0.25 at the same quantization boundary.

**Null:** No consistent cliff boundary (fewer than 2 families exceed κ on both metrics at the same boundary).

**Metric:** `detect_cliff_boundary()` returns the same QuantScheme for ≥ 2 model families.

**Acceptance criterion:** κ = 0.25, cliff at Q3_K_M or below in ≥ 2 of 3 families.

### Intuition for κ = 0.25

κ = 0.25 is grounded in prior empirical evidence: Egashira et al. (ICLR 2025) measured attack-success-rate deltas of Δ = 88.7 % (insecure code), 85.0 % (content injection), and 30.1 % (refusal bypass) between FP16 and GGUF-quantized models. Hong et al. (arXiv:2403.15447) found a ~50-point toxicity-safety drop at GPTQ-3-bit versus FP16. For the geometric metric, κ = 0.25 corresponds to a cosine similarity of 1 − 2·(0.25)² = 0.875 between the FP16 and quantized refusal directions — a rotation large enough that the safety subspace at Q3_K_M is meaningfully misaligned with FP16. κ is pre-registered and cannot be adjusted after data collection begins.

### H2 — FPR Decoupling (White-Box)

**Claim:** PROBE-RM FPR after per-quantization calibration varies less than ε = 0.02 across {FP16, NF4, AWQ-INT4, Q4_K_M, Q3_K_M}.

**Null:** FPR range ≥ 0.02 (calibration does not achieve portability).

**Metric:** max(FPR values across schemes) − min(FPR values across schemes).

**Acceptance criterion:** FPR range < 0.02 at fpr_target = 0.05. Evaluated using the KS test in `eval/stats.py` (`ks_test_fpr_decoupling`).

### H3 — FPR Decoupling (Black-Box)

**Claim:** B-PROBE-LOGIT FPR after per-quantization calibration varies less than ε = 0.02 across the same schemes, with strictly lower TPR than PROBE-RM (confirming white-box advantage).

**Null:** FPR range ≥ 0.02 OR TPR(B-PROBE-LOGIT) ≥ TPR(PROBE-RM).

**Metric:** Same as H2 for B-PROBE-LOGIT; paired TPR comparison.

**Acceptance criterion:** FPR range < 0.02 AND TPR(B-PROBE-LOGIT) < TPR(PROBE-RM) at matched FPR.

### H4 — Composition Gain

**Claim:** The full primitive stack achieves strictly smaller ABR (attack bypass rate) than any single primitive at matched FPR.

**Null:** No significant ABR reduction from composition (Wilcoxon p ≥ α_corrected).

**Metric:** Wilcoxon signed-rank test on per-prompt block decisions (full stack vs. best single primitive, paired).

**Acceptance criterion:** p < α_corrected = 0.01 AND mean(full stack scores) > mean(best single scores). Implemented in `eval/stats.py` (`test_h4_composition_gain`).

### H5 — Tier-C Structural Weakness

**Claim:** Tier C (no dedicated classifier) shows no statistically significant ABR reduction against A7 (cliff exploiter) relative to no-defense baseline. Tier C+ (with PromptGuard-2-22M-INT4) shows significant ABR reduction.

**Null for Tier C:** Tier C achieves significant ABR reduction (p < 0.05 against baseline).

**Null for Tier C+:** Tier C+ does not achieve significant ABR reduction (p ≥ 0.05).

**Acceptance criterion:** p(Tier C) ≥ 0.05 AND p(Tier C+) < 0.05. Note that H5 Tier C uses uncorrected α = 0.05 (absence of effect is the claim). Implemented in `eval/stats.py` (`test_h5_tier_c_weakness`).

## Sensitivity Corollary (§3.4)

The sensitivity bound formalizes why the cliff is a structural consequence of quantization, not an artifact of a specific attack.

**Geometric assumption (G1).** Let z_t(q) denote the residual-stream activation at token t for prompt q. There exists a safety subspace S = span{r̂, ĥ} ⊂ ℝ^d such that in expectation over (q, q') ∈ harmful × harmless:

```math
\|\Pi_S z_{t_\text{inst}}(q) - \Pi_S z_{t_\text{inst}}(q')\| \;\ge\; \xi \cdot \|z_{t_\text{inst}}(q) - z_{t_\text{inst}}(q')\|
```

for some ξ > 0 (the safety-subspace fraction). G1 is empirically supported by Arditi et al. (arXiv:2406.11717) and Zhao et al. (arXiv:2507.11878) for instruction-tuned LLMs at 7B–72B parameters.

**Sensitivity bound.** Let q' = q + δ be an adversarial perturbation and let π(q) = Π_S z_{t_inst}(q) be the safety projection. If quantization acts as a Lipschitz operator on the residual stream with constant L_q̃:

```math
\|\pi(q) - \pi(q')\| \;\le\; L_{\tilde{q}} \cdot (\|\delta_z\|_2 + 2\,\varepsilon_{\tilde{q}})
```

where ε_q̃ is the per-quantization residual reconstruction error and δ_z is the induced activation perturbation. **The cliff is the regime where ε_q̃ inflates enough that π(q) flips sign on harmful prompts.** This is why PROBE-RM degrades at Q3_K_M and why the FPR-decoupling theorem cannot recover information lost when the per-class margin distributions overlap.

## PROBE-RM Margin

The refusal margin is the cosine projection of the post-instruction hidden state onto the unit-norm refusal direction:

```math
m_r = \frac{\hat{r} \cdot z_\ell(t_{\text{post-inst}})}{\|\hat{r}\| \cdot \|z_\ell\|}
```

where:
- r̂ is the refusal direction (difference-in-means on harmful vs. harmless residual streams at layer ℓ, normalized to unit norm)
- z_ℓ(t_post-inst) is the residual stream at layer ℓ, last instruction token position
- The gate fires (blocks) when m_r < τ_q (low margin = hidden state distant from refusal subspace)

The Bayes-optimal linear test for {harmless, harmful} under class-conditional Gaussian residuals on the 1-D subspace is exactly a threshold on m_r (Arditi et al. 2406.11717, §3). Quantization shifts the per-class means; the calibration table absorbs the shift.

## CONDUCTOR LinUCB

UCB score for arm a at context x_t:

```math
\text{UCB}(a, x_t) = x_t^\top \hat{\theta}_a + \alpha \sqrt{x_t^\top A_a^{-1} x_t}
```

where A_a = λI + Σ_{s: a_s=a} x_s x_s^T and b_a = Σ r_s x_s, so θ̂_a = A_a^{-1} b_a.

**Update rule (each request):**

```math
A_a \mathrel{+}= x_t x_t^\top, \qquad b_a \mathrel{+}= r_t \cdot x_t
```

**Pre-registered reward function:**

```math
r_t = \mathbb{1}[\text{served} \land \text{clean}] - \mathbb{1}[\text{served} \land \text{injected}] - 0.2 \cdot \mathbb{1}[\text{blocked} \land \text{benign}]
```

(+1 correct block, -1 miss, -0.2 false positive)

Regret bound in stochastic regime: Õ(√(dT)) with d = 14 (CONTEXT_DIM). Never-disable arms (TRIPWIRE-R, ATTEST-WH) have a floor weight MIN_WEIGHT = 0.1.

## TRIPWIRE-H CUSUM

Page-Hinkley one-sided CUSUM on per-token entropy, detecting sustained downward shifts:

```math
S_t = \max(0,\; S_{t-1} + (H_t - \mu_0) - k)
```

where:
- H_t = −Σ_v p_t(v) log p_t(v) is the token entropy at step t (over top-k softmax)
- μ_0 is the per-quantization benign mean entropy
- k is the reference level (sensitivity parameter, default 0.5)
- Gate fires when S_t > τ_q

An injection that pushes the model toward a forbidden completion typically shows entropy depression (high-confidence forbidden token) followed by a spike ("starts and stalls"). GCG-perturbed contexts often show abnormal early entropy. The CUSUM is initialized to 0 at session start.

## B-PROBE-CONSISTENCY JSD

Jensen-Shannon divergence across N paraphrase output distributions:

```math
\text{JSD}(P_1, \ldots, P_N) = H\!\left(\frac{1}{N}\sum_{i=1}^N P_i\right) - \frac{1}{N}\sum_{i=1}^N H(P_i)
```

where H is Shannon entropy in nats. JSD ∈ [0, log N].

**Firing direction — LOW:** the gate fires when JSD < τ_q. A refusing model produces high-variance refusal outputs across paraphrases (high JSD); a model that has been coerced into compliance produces low-variance, consistent outputs (low JSD). Low divergence = consistent compliance = suspicious.

The gate is computed on top-k log-probability vectors (one per paraphrase, vocabulary restricted to top-k tokens as returned by the API), making it compatible with the black-box endpoint path.

## PROBE-HD Harmfulness Margin

Zhao et al. (arXiv:2507.11878, NeurIPS 2025) showed that harmfulness and refusal are encoded in **separate** residual-stream directions. PROBE-HD extracts the harmfulness direction ĥ at the user-instruction token t_inst:

```math
m_h = \frac{\hat{h} \cdot z_\ell(t_{\text{inst}})}{\|\hat{h}\|}
```

where ĥ is the harmfulness direction (difference-in-means on harmful vs. harmless residual streams at the instruction token, normalized to unit norm). **Gate fires HIGH** when m_h > τ_q — high harmfulness projection at the request stage indicates the model recognises the prompt as harmful, regardless of whether it will ultimately refuse (the refusal decision is PROBE-RM's signal at t_post-inst).

Together PROBE-HD (at t_inst) and PROBE-RM (at t_post-inst) probe the full 2-D safety subspace S = span{r̂, ĥ}.

## PROBE-MT Margin Trajectory

Compute the refusal margin ρ_ℓ at layers at 25 %, 50 %, and 75 % of model depth. Define:

```math
\dot{\rho} = \rho_{\text{late}} - \rho_{\text{mid}}, \qquad \ddot{\rho} = (\rho_{\text{late}} - \rho_{\text{mid}}) - (\rho_{\text{mid}} - \rho_{\text{early}})
```

**Gate fires LOW** when ρ̇ < τ_q. A falling margin trajectory (ρ̇ < 0) indicates the model is drifting away from refusal as it processes the input — characteristic of adversarial suffixes that redirect the model over the course of the forward pass. Both ρ̇ and ρ̈ are stored in the CONDUCTOR context vector at indices 3 and 4.

Cost: K_ℓ = 3 projections, ~36 µs on GPU, ~600 µs on Pi 5.

## VESTIBULE-LZ Compression Ratio

```math
\rho_{\text{LZ}}(x) = \frac{|z(x)|}{|x|}
```

where |z(x)| is the zlib-compressed length and |x| is the raw byte length of the input string.

**Gate fires HIGH** when ρ_LZ(x) > τ_q. GCG-style adversarial suffixes consist of near-random tokens that compress poorly (high ratio). ArtPrompt ASCII art has a distinctive compression signature detectable via the calibrated threshold.

Cost: ~20 µs per 1 KB. No model access required — pure stdlib `zlib`.

## VESTIBULE-PS Signal Score

Implements the spotlighting / provenance-marking approach of Hines et al. (arXiv:2403.14720). Counts structural signals that indicate injected content within the input:

```
signal_score(x) = Σᵢ wᵢ · count_signals(x, pattern_i)
```

Marker types: instruction-density patterns (imperative verb sequences near entity references), role-assignment strings, delimiter anomalies, encoding anomalies (non-printable characters, base-N payloads, high ASCII-art density), structural-segregation signals.

**Gate fires HIGH** when signal_score(x) > τ_q. High signal score indicates injected instruction structure within retrieved content or tool outputs.

## TRIPWIRE-R Log-Likelihood Ratio

The per-token log-likelihood ratio between the protected model and a fixed KenLM reference:

```math
\Delta_t = \log P_M(y_t \mid \text{ctx}) - \log P_{\text{ref}}(y_t \mid \text{ctx})
```

Combined with TRIPWIRE-H via a linear combination fit on calibration traces:

```math
C_t = \alpha H_t + \beta \Delta_t
```

CUSUM runs on C_t. **Gate fires LOW** when the accumulated statistic falls below τ_q — the model's output is drifting away from benign-reference n-gram expectations.

**Why a fixed reference helps:** pure perplexity gates (P_M alone) must be re-tuned under quantization because P_M shifts. The KenLM reference P_ref is fixed at calibration time, so quantization affects only the P_M term. The ratio Δ_t partially cancels the drift. KenLM order: 5 for Tier A/B, 3 for Tier C/C+ (see `kenlm.order_tier_ab` / `kenlm.order_tier_c` in `configs/example.yaml`).
