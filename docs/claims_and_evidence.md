# Claims, Evidence Status, and the Path to Publication

**Date:** 2026-08-03 (was 2026-08-02)
**Purpose:** one page that says exactly what we claim, what is proven, what the code measures, and
what remains. Update the evidence column as results land; never let a claim's status drift.

---

## 0. The single most important line in this document

> **The geometry behaves exactly as the theory predicts. The behaviour does not follow it.
> Falsifier F5 has fired.**

A full 7-rung ladder now exists: `artifacts/runs/20260803-075641_dafdc44_local-ladder-rtn-qwen1.5b`,
Qwen2.5-1.5B-Instruct, n=250/class, round-to-nearest at 8→2 bits. Full write-up in
[`results_local_ladder.md`](results_local_ladder.md). In one paragraph:

The FP16→quantized rotation of the refusal direction **replicates at 8/8 rungs** (z = 15–26) and
grows smoothly from 1.77° to 88.69°, roughly doubling per bit removed. Weight-space η follows
`η₄·base^(4−b)` with **base = 4.355, 95 % CI [4.114, 4.611], R² = 0.9989** — the interval excludes
4, so A3's assumed base is rejected while the law itself is confirmed with a sharper constant. And
**held-out d′ does not move**: 0.4129 at FP16, 0.3914 at 2 bits, against a standard deviation of
0.085 — no detectable degradation at a rung where η has grown 7400× and the direction has rotated
88.7°. The η(weights)/η(d′ decay) ratio spans **374×** and falls monotonically.

The prior version of this line read "we currently have zero validated empirical results." That is
no longer true, and the replacement is not the result the theory wanted.

**The load-bearing caveat.** `d′₀ = 0.41` is a *label ceiling*, not a model property: "refused"
describes the hh-rlhf rejected response, not what this model does. The honest claim is therefore
narrow — *within the range this probe can measure*, quantization-induced rotation does not reduce
discriminability. It is **not** "quantization is safe". Deriving labels from the target model's own
completions is the highest-value next step and is not yet built.

---

## 1. The claims

### Tier 1 — the headline claims (what the paper is about)

| # | Claim | Type | Status |
|---|---|---|---|
| **C1** | Quantization degrades behaviour by **inflating variance** on the behavioural subspace, not by rotating or selectively destroying it. | Mechanism | **Rotation half supported, degradation half refuted.** The rotation is real and systematic (8/8 replicate, z = 15–26, 1.77°→88.69°) but produces no measurable d′ loss. Rotation and degradation are **decoupled**. |
| **C2** | Discriminability follows `d'(q) = d'₀(1+η_q)^(−1/2)`, with **η measurable from weights + benign activations only** — no harmful prompts, no judge, no labels. | Predictive law | **Mechanism REFUTED on Qwen2.5-1.5B (F5 fired).** η(weights) rises 7400× while η(d′) rises ~20×; their ratio spans 374× and falls monotonically. The weight→behaviour transfer is strongly compressive, not the identity the law assumes. |
| **C3** | The apparent "cliff" is a **threshold-crossing artifact**, not a phase transition. `d'(b)` is smooth; the sigmoid makes it look sharp. Predicted collapse at `b* ≈ 2.94` bits. | Shape claim | **Untestable on this run.** `b*` is undefined because `d′₀ = 0.413 < z₀.₉₅ = 1.645`. Stage 4's "3/4 within 1 sd" is a flat prediction matching a flat measurement and carries almost no information. |
| **C4** | Which sector collapses first is set by its **composition rule**: long chains and rare failures break before single-shot decisions. `b*(4T) − b*(T) → 1 bit`. | Ordering law | Derived; **not yet tested** — blocked behind C3 having a defined `b*`. |
| **C5** | **Isotropy is a property of the quantizer, not of quantization.** RTN/NF4/GGUF isotropic → cliff; AWQ/GPTQ anisotropic → no cliff. | Conjecture | **Premise false, effect real, needs restating.** RTN is *not* isotropic (null rejected 7/8 rungs), so the isotropic-vs-anisotropic framing fails. But the controlled arm settles the underlying question: at an **identical bit budget**, salience-aware quantization rotates the direction **22–26 % less** while injecting **1.5–2.0× more total weight noise**. The distinction is not *whether* damage is concentrated but **where it points**. Effect vanishes at 2 bits, where both saturate near orthogonality. |
| **A3** | `η(b) = η₄ · 4^(4−b)` — the noise-to-signal ratio quadruples per bit removed. | Assumption | **Confirmed in form, rejected in constant.** Measured base **4.355, 95 % CI [4.114, 4.611]**, R² = 0.9989, n = 7. The interval excludes 4. Theorem 2 should carry a measured base. Interval is model-conditional (rungs share weights, direction, activation sample). |

### Tier 2 — the limits (what cannot be done)

| # | Claim | Type | Status |
|---|---|---|---|
| **C6** | Threshold/affine recalibration **cannot** restore lost discriminability (ROC invariance). | Proved (i) | **Proved**; empirically corroborated (99 % of damage orthogonal) — corroboration gated on C0 |
| **C7** | No function of a **frozen quantized checkpoint alone** can uniformly recover FP16 behaviour (DPI + quantization-cell collision). | Proved (i) | **Proved** |
| **C8** | Quantization error is **deterministic** given the weights, so self-consistency / repeated sampling cannot average it away. | Proved (i) | **Proved** (refutes our own earlier `d'√T` conjecture) |

### Tier 3 — the cure (what can be recovered)

| # | Claim | Type | Status |
|---|---|---|---|
| **C9** | A **rank-k sidecar** `v = r̂ᵀ(W − W_q)` saved *before* quantization restores the projected margin exactly, at ~180 KB for k=1. DPI does not bind because ΔW and a(x) are outside the bottleneck. | Proved (i) construction | **Proved on paper; not implemented or measured** |
| **C10** | Margin error ε propagates to decision disagreement bounded by `F(τ+ε) − F(τ−ε) ≈ 2εf(τ)`. | Proved (ii) | **Proved**; numerically verified |
| **C11** | *(Conjecture 8)* A sidecar of `Θ(k·d·L·log(R/ε))` bits suffices, and nothing using `W_q` alone attains it. | **Conjecture** | Achievability sketched; **converse open — do not claim** |

**We do not claim** to beat Q-resafe / Q-realign / CAQ on safety restoration. C9 is a *mechanism
test*, not a product. Say so in the paper.

---

## 2. What the system actually measures

Concretely, module by module — this is the honest inventory of the instrument.

| Module | Input | Output | Serves |
|---|---|---|---|
| `eval/noise_floor.py` | activation matrices per class, per scheme | split-half estimator noise floor (50 splits, two scaling corrections); **paired bootstrap CI** for the cross-scheme rotation with synchronous resampling | **C0 gate**, C1 |
| `eval/isotropy.py` | two direction vectors | rotation angle, excess kurtosis of perturbation vs direction, concentration z-scores vs a matched-magnitude isotropic null, parallel/orthogonal split, irrecoverable fraction | C1, C5, C6 |
| `eval/discriminability.py` | class-conditional margin samples | `d'` with bootstrap CI, AUC (empirical + Gaussian-model), **`gaussianity_gap`** (assumption A2 check), `predict_d_prime`, `implied_eta` | C2 |
| `eval/noise_spectrum.py` | FP16 + quantized weights (+ benign activations) | per-matrix projected perturbation variance, `eta_proxy` (labelled as a proxy), `eta_empirical`, effective bits/param (payload + whole-file), fitted exponent with CI | C2, A3 |
| `eval/composition.py` | `d'₀`, `η₄`, composition rule | `predict_collapse` per sector, `chain_bit_cost`, `sector_ordering`, closed-form `b*` | C3, C4 |
| `eval/threshold_calibrator.py` | benign scores | tail-correct thresholds (fires-high/low), conservative same-sample FPR | infrastructure (D0 fixed) |
| `scripts/verify_gguf_pair.py` | two GGUF paths | tensor-name/shape correspondence, per-tensor dequantization with peak-RSS | A3 plumbing |

**Not yet built:** the sidecar (C9/C10) and the ladder builder. Stages 2 and 6 of the plan.

**What it does not measure, and must:** real model *responses*. Everything above is
representation-level. `Δ_B-cliff` was never behavioural (defect D3), and until completions are
generated and scored, no behavioural claim is supported.

---

## 3. What remains, to prove each claim

Ordered by dependency. **C0 blocks everything.**

| Step | Proves | Work | Compute | Blocks |
|---|---|---|---|---|
| **C0** | gate for all | Run `stage0_noise_floor_and_isotropy.ipynb` on the real Fold A corpus. Paired bootstrap must exclude zero. | ~1 h, T4 | **everything** |
| **S1** | C5 | Quantize the same model with AWQ/GPTQ; rerun `isotropy.py`. Compare z against NF4's. | ~½ day | C1 scope |
| **S2** | A3, C2 | Build matched GGUF ladder (F16→Q2_K, one checkpoint, one pinned llama.cpp, always from F16). Measure `η_q` per scheme; **fit** the exponent, report CI. | ~1 day | C3, C4 |
| **S3** | C2 | Save **raw margin distributions** per scheme. Compute `d'`, `gaussianity_gap`. Compare weights-measured `η` vs `implied_eta` from `d'` decay. | ~1 day | — |
| **S4** | C3 | Fit `b*` on part of the ladder; **predict the rest without refitting**. | ~2 h | — |
| **S5** | C4 | GSM-Infinite across depth × precision. Test `b*(4T) − b*(T) = 1`. Separate semantic depth from token count. | 20–60 GPU-h | — |
| **S6** | C1 behavioural | **Generate real completions**; score with StrongREJECT + human audit on a stratified sample. Replaces the D3 proxy. | ~4 h + GPU | any behavioural claim |
| **S7** | C9, C10 | Implement the sidecar; measure recovery fraction **against a random-direction matched-norm control**. | ~1 day | — |

**Critical-path minimum for a paper: C0 → S1 → S2 → S3 → S4.** That is roughly two weeks and
supports C1, C2, C3, C5. S5 adds C4 (generality). S6 is required before any behavioural language.

---

## 4. How to record

**Pre-registration first.** Before S2 runs, freeze:
- the assumption tests and their pass thresholds (A2 gap ≤ 0.05; exponent CI vs 4),
- the collapse criterion (relative retention `c`, not an absolute bar),
- the falsifiers F1–F8 from `docs/theorems.md` §7,
- the analysis plan (paired tests, CIs, multiplicity policy).

File it as a **dated amendment** to `docs/preregistration.md` with a change table — never a silent
edit (defect D14). Label all existing Fold A/B artifacts **pilot/exploratory**, not confirmatory.

**Per-run provenance.** Every run records: git SHA, model revision hash, quantizer commit,
imatrix hash, tokenizer hash, prompt-manifest hash, decoding config, seeds. `eval/repro.py` already
does most of this — extend it to the ladder.

**Raw artifacts, not summaries.** Save per-prompt margins, per-cell responses, and per-tensor
spectra. The D3 failure happened partly because only thresholds were persisted.

**Negative results in the same file as positive ones.** F1–F8 each get a row whether they fire or
not.

---

## 5. How to publish

**Title (working):** *Behavioural Rate–Distortion for Quantized Language Models: a
noise-and-composition law for where capabilities collapse.*

**Three-claim structure:** impossibility (C6–C8) → law (C2–C4) → construction (C9–C10). Safety
motivates; **at least one non-safety behaviour is required** to justify generality — that is why S5
exists.

**Must engage directly, not footnote:**
- arXiv:2508.18609 — the nearest prior work. Differentiate on *mechanism* (theirs is NLS curve
  fitting) and on *shape* (a smooth power law cannot produce a cliff).
- arXiv:2606.10154 vs arXiv:2606.29581 — the contradiction C5 resolves.
- Q-resafe / Q-realign / CAQ — position C9 as a mechanism test, not a competitor.
- LiteLMGuard, Arditi, Zhao — prior art for the components.

**Venue, honestly.** ICLR/ICML/NeurIPS is reachable **only if** the law predicts out-of-sample
across ≥2 model families and ≥2 behaviours and the comparison to the above is serious. Otherwise
TMLR or ACL Findings. A NeurIPS/ICML/ICLR workshop on efficient inference or trustworthy ML is the
right first target — "why does this correction work" is native framing there. Not S&P/CCS/USENIX:
there is no coherent systems threat model left after the scope cut, and that is fine.

**Release:** matched ladder manifests, per-prompt margins, sidecar vectors, notebooks, and the
falsifier table. The artifact is part of the contribution.

---

## 6. The failure branches, decided in advance

| If | Then |
|---|---|
| **C0 null** (rotation inside the noise floor) | Retract the geometric claim publicly. The theorem *predicts* direction retention, so this is consistent with it — but C1 becomes unproven and the paper leads with C2/C3 measured behaviourally instead. Re-test with larger n (floor scales as 1/√n). |
| **A2 fails** (gap ≫ 0.05) | Closed-form TPR predictions are void. `d'` and the ordering survive; drop the sigmoid-shape specifics. |
| **Exponent CI excludes 4** | A3 is wrong for k-quants. Report the measured base — this is a *result*, and Theorem 2's `b*` just takes the fitted value. |
| **AWQ also isotropic** | C5 refuted. The literature split must come from judges/corpora — worth publishing as a correction. |
| **η(weights) ≠ η(d′)** | **C2's mechanism is refuted.** This is the central falsifier. Report it; it bounds where noise abstractions of quantization are valid. |
| **Sidecar ≈ random control** | C9's causal-sufficiency premise fails for this readout. Delimits the whole class of "safety direction patch" proposals — genuinely useful. |

Every branch has a publishable outcome. That is the point of pre-registering them.

---

*Companion documents: `docs/theorems.md` (formal statements + falsifiers), `docs/pivot_plan_2026-08.md`
(execution), `docs/research_audit_2026-08.md` (defects D0–D15), `docs/build_log.md` (agent cross-review).*
