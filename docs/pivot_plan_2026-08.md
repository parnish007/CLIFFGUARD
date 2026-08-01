# CLIFFGUARD → Pivot Plan: from defense system to measurement instrument + theory

**Date:** 2026-08-02
**Decision:** **Repurpose, do not restart.** Keep ~30 % of the repo (the measurement
infrastructure), archive the defense architecture, build the theory core.
**Predecessors:** `docs/research_audit_2026-08.md`, `docs/theory_panel_2026-08.md`

---

## 0. The decision, and why not a fresh repo

Starting over would throw away the only genuinely valuable asset you have: a **working,
checkpointed, type-clean pipeline that extracts residual-stream directions from quantized models
on free Colab hardware and survives disconnects.** That took months and it is exactly the
instrument the new work needs.

What must go is not the code — it is the **framing**. CLIFFGUARD is currently presented as a
defense system with 11 gates. The new work is a **measurement instrument that produced a scaling
law.** Same engines, same adapters, same calibration path, same notebook. Different question,
different output.

Concretely: you keep the engines, the direction extractor, the calibration machinery, the fold
discipline, the Colab checkpointing, and the testing culture. You archive the eight-component
architecture. You add roughly six new modules that do the actual science.

> **New working title:** *Behavioural Rate–Distortion for Quantized Language Models: a
> noise-and-composition law for where capabilities collapse.*

---

## 1. The three claims the paper will make

Everything below serves these and nothing else. If a task doesn't serve one of these, cut it.

1. **Mechanism.** Quantization acts as approximately isotropic noise on the residual stream for
   noise-isotropic quantizers (RTN/NF4/GGUF k-quants), and as *anisotropic*, subspace-sparing
   noise for activation-aware quantizers (AWQ/GPTQ). **This resolves the 2026 literature
   contradiction** between arXiv:2606.10154 (12–68 pp refusal falls) and arXiv:2606.29581
   (AWQ INT4 within 1.6 pp of FP16 for 7/8 models).
2. **Law.** Given a measured noise ladder `η_q` and a measured FP16 margin `d₀`, degradation of any
   thresholded behaviour follows `d(q) = d₀(1+η_q)^(−1/2)`, and the **composition rule** determines
   the collapse point: single threshold, T-fold chain (`b*(T) = b*(1) + log₄T`), or rare tail
   (`log p_q/log p₀ = 1/(1+η)`). Validated by **out-of-sample prediction**, not in-sample fit.
3. **Limits and cure.** No post-hoc function of a frozen quantized checkpoint restores lost
   discriminability (DPI + quantization-cell collision); ~99 % of the damage is irrecoverable by
   recalibration (measured). But a **rank-k sidecar saved before quantization** — `v = r̂ᵀ(W−W_q)`,
   ~180 KB — restores a chosen behavioural projection exactly, with an explicit bit budget.

**Differentiation from the closest prior work (arXiv:2508.18609):** their task-stratified law is a
*non-linear least-squares curve fit*, smooth and monotone in `log(B_eff)` — it **structurally
cannot produce a cliff**, does not predict collapse bit-widths, and covers no safety/security
tasks. Ours is mechanistic, predicts collapse location, and explains the cliff as a
threshold-crossing artifact. Those are different, testable claims about the *shape*.

---

## 2. Code changes

### 2.1 KEEP — the instrument (fix, don't rewrite)

| Path | Action |
|---|---|
| `cliffguard/types.py` | Keep. Add `FiresDirection` enum (see D0 fix). |
| `cliffguard/engines/transformers_bnb.py` | Keep — hidden-state hooks are the core capability. |
| `cliffguard/engines/llamacpp.py` | Keep and **promote** — the GGUF ladder is now central. |
| `cliffguard/eval/refusal_direction.py` | Keep; **fix labels** (see §2.3 F3). |
| `cliffguard/eval/threshold_calibrator.py` | Keep; **fix the tail bug D0** — this is blocking. |
| `cliffguard/probe/rm.py` | Keep as the *margin readout*, not as a "gate". |
| `cliffguard/eval/folds.py`, `repro.py`, `results_writer.py` | Keep — provenance/hashing is now a selling point. |
| `notebooks/colab_helper.py` | Keep — checkpoint/resume is essential for ladder runs. |
| `tests/` | Keep the discipline; retarget to the new modules. |
| `cliffguard/vestibule/*` | Keep **only** as a documented negative result (0/95 Nepali). Not core. |

### 2.2 ARCHIVE — move to `legacy/`, do not delete history

`conductor/` (LinUCB), `lookout/`, `tripwire/`, `bprobe/`, `attest/`, `ladder/`,
`eval/bcn2.py`, `eval/drift_sim.py`, `eval/kenlm_*.py`, most of `eval/judges.py`,
and the H2–H5 machinery in `eval/stats.py`.

Keep them importable and tested so the 989 tests don't rot, but remove them from the paper's
claims entirely. A one-paragraph appendix ("the system this instrument was extracted from") is
the right amount of coverage.

### 2.3 FIX — blocking defects (from the audit)

| ID | Fix | Est. |
|---|---|---|
| **F1 = D0** | Add `fires_high`/`fires_low` per readout; select the **5th** percentile for fires-low. Add a test asserting empirical FPR ≈ target **in both directions**. Recompute Fold A. | 3 h |
| **F2 = D2** | Split-half `r̂` noise-floor control. **Gates everything.** | 1 h |
| **F3** | Refusal labels currently come from refusal-like text in the HH *dataset's rejected response*, not from whether **the target model** refused. Either derive labels from actual model responses or rename the vector a *harmfulness* direction. | 4 h |
| **F4 = D1** | Collapse `Δ_cliff` to **one** definition. Recommendation: drop it entirely — the new metric is `d′`, which is well-defined and comparable across domains. | 2 h |
| **F5** | Save **raw margin distributions**, not just thresholds. Without these `d₀` is unrecoverable. | 1 h |

### 2.4 ADD — the theory core (~6 new modules)

| New module | Purpose |
|---|---|
| `eval/ladder_builder.py` | Build a **matched** GGUF ladder from one pinned checkpoint + one pinned `llama.cpp` commit. Always quantize from F16/F32 — *requantization severely degrades quality*. Hash source, imatrix, every artifact, tokenizer, and prompt manifest. |
| `eval/noise_spectrum.py` | Measure **`η_q` from weights only**: per-layer `σ_q²` of `W − W_q`, projected onto the behavioural subspace, normalised by the FP16 within-class margin variance `s²`. This is the independent variable — no behaviour data needed. |
| `eval/isotropy.py` | Generalise the audit §3.2 test: excess kurtosis of the perturbation + matched-magnitude isotropic null (400 trials) + top-1/5/10 % concentration z-scores. Runs per scheme. **This is the Stage-1 experiment.** |
| `eval/discriminability.py` | `d′` from benign/harmful margin distributions, with bootstrap CIs; AUC; TPR@fixed-FPR; ROC. Replaces the whole `Δ_cliff` family. |
| `eval/composition.py` | The three readouts: `THRESHOLD` → `Φ(d−z)`; `CHAIN(T)` → `p^T`; `TAIL(p₀)` → `log p_q/log p₀ = 1/(1+η)`. Plus `predict_collapse_bitwidth()`. |
| `eval/sidecar.py` | `v = r̂ᵀ(W_FP16 − W_q)` per contributing layer; `m_corrected(x) = m_q(x) + v·a(x)`; storage accounting; **random-direction matched-norm control**. |
| `eval/predict.py` | Fit `d₀`, `η₄` on a **held-out-excluded** subset; predict unseen (scheme × depth) cells; report out-of-sample error. **This is the paper's headline table.** |

---

## 3. Models to test on

Constraint: **RTX 3050 Laptop 6 GB** + Colab T4 16 GB. FP16 3B ≈ 6.4 GB → does *not* fit locally
with activations; run FP16 references on Colab, quantized runs locally.

| Role | Model | FP16 | Q4 | Where |
|---|---|---|---|---|
| **Local dev / fast iteration** | `Qwen2.5-1.5B-Instruct` | ~3.1 GB | ~1.0 GB | 3050 ✔ (FP16 fits) |
| **Primary** | `Llama-3.2-3B-Instruct` | ~6.4 GB | ~2.0 GB | FP16 → Colab; quantized → 3050 ✔ |
| **Family 2** | `Qwen2.5-3B-Instruct` | ~6.2 GB | ~2.0 GB | same |
| **Family 3** | `Gemma-2-2B-it` | ~5.2 GB | ~1.6 GB | 3050 borderline; Colab safe |

Three families satisfies the "≥ 2 of 3 families" discipline. **Do not** design around Mistral-7B
(~14.5 GB BF16) — not a realistic local FP16 reference.

You already have Fold A artifacts for Llama-3.2-3B-Instruct — start there, reuse them.

### Quantization schemes

**Ordinal ladder** (one checkpoint, one pinned `llama.cpp`, always from F16):
`F16 → Q8_0 → Q6_K → Q5_K_M → Q4_K_M → Q3_K_M → Q2_K`

**Separate categorical comparison** (the isotropy test — *never* on the same ordinal axis):
`NF4 (bitsandbytes)` vs `AWQ-INT4` vs `GPTQ-INT4`

Mixing NF4/AWQ/GPTQ into the bit-width axis is a confound: different algorithms and runtimes,
not a monotone dose sequence.

---

## 4. Domains and datasets

Two domains carry the paper. The third is optional and expensive.

### D1 — Safety (composition = THRESHOLD)
- Harmful: **AdvBench + JailbreakBench** (semantic dedup; unit = base intent, not variant)
- Benign calibration: **≥ 2000** (prereg minimum; you used 400 — see audit D7)
- **Hard benign**: **XSTest** safe split + **OR-Bench** hard subset — a single scalar FPR hides
  catastrophic overblocking
- Readout: `d′`, TPR@5 %FPR, full ROC

### D2 — Reasoning (composition = CHAIN(T)) ← **the generality experiment**
- **GSM-Infinite** (arXiv:2502.05252, Infini-AI-Lab, pip-installable): synthetic, graph-based,
  **exact control of reasoning depth**.
- **Why this is the right choice:** its authors report a *"consistent sigmoid decay in performance
  with increasing complexity"* across 17 SOTA models. **That sigmoid is exactly what
  `P_T = Φ(d−z)^T` predicts.** They observed the curve; your law explains it *and* predicts how it
  shifts with bit-width. Direct engagement with an existing published curve.
- 6–8 precision points, 100–300 items per depth, greedy decoding first, step verifier.
- **Critical:** separate *semantic depth* from *output token count* — quantized models inflate
  token counts independently (arXiv:2606.25519), which would confound everything.

### D3 — Security (composition = TAIL) — **optional, defer**
Needs executable CWE tests, ≥ 2 static analyzers, and manual adjudication. The constraint is
annotation labour, not GPU. Include only if D1 + D2 land early.

---

## 5. Execution order

| Stage | What | Cost | Gate condition |
|---|---|---|---|
| **0** | F1 (tail fix) + F2 (split-half noise floor) | ~4 h, no GPU | **If split-half rotation ≈ 13°, premise P1 is unproven — stop and reframe.** |
| **1** | `isotropy.py` across NF4 vs AWQ vs GPTQ on one model | ~½ day | Decisive either way; resolves the literature split |
| **2** | `ladder_builder.py` + `noise_spectrum.py`: build the matched GGUF ladder, measure `η_q` | ~1 day | `η_q` must be measured, **not assumed** ∝ 4^(−b) — k-quants use super-block scales |
| **3** | `discriminability.py` on D1: `d₀` + margin distributions per scheme | ~1 day | Establishes the threshold-composition baseline |
| **4** | D2 on GSM-Infinite across depth × precision | ~20–60 GPU-h | Tests `b*(T) = b*(1) + log₄T` |
| **5** | `predict.py`: fit on shallow depths + high precisions, **predict deep/low-bit without refitting** | ~2 h | **The headline result** |
| **6** | `sidecar.py` + random-direction control | ~1 day | Mechanism validation, not an ASR competition |

**Total: roughly 6–10 focused weeks**, well inside your window, and the first three stages need
almost no GPU.

---

## 6. What success and failure both look like

This design is publishable either way, which is the point of pre-registering it.

- **Success:** out-of-sample prediction of collapse bit-width lands within ~0.5 bits across
  depths and schemes; AWQ shows anisotropic damage while RTN/NF4 does not. → a mechanistic
  scaling law that explains several existing empirical papers. ICLR/ICML/NeurIPS reachable if it
  transfers across families and behaviours; TMLR/Findings otherwise.
- **Failure:** the isotropic-independent abstraction breaks (correlated reasoning steps,
  non-Gaussian margins, `η ∝ 4^(−b)` violated by k-quants). → a **bounded negative result** that
  delimits where noise-based abstractions of quantization are valid. Still a real contribution,
  and honest negative results in this area are scarce.

---

## 7. What to stop doing immediately

- Building or validating gates. Seven of eight components have never seen real inference and none
  of them serves the three claims.
- Treating `Δ_cliff` as a metric. Replace with `d′` everywhere.
- Quoting κ = 0.25 — its justification is a misattributed adversarial-attack effect size (D4).
- Any H2/H3 language. Demote to one Methods sentence.
- Treating the 19-prompt Nepali set as a benchmark. It is 19 clusters, IndicJR has 45 216 prompts
  including Nepali, and 5 fields are script-corrupted. Keep it as a *negative result* only.

---

## 8. Honest risks

1. **Stage 0 may kill premise P1.** Most likely single failure. Mitigated by the fact that a null
   there still reframes the project usefully.
2. **`η ∝ 4^(−b)` may not hold for k-quants.** Super-block scales, group quantization, and outlier
   handling can break it. Mitigated by *measuring* the curve instead of assuming it.
3. **Reasoning steps are correlated**, so `p^T` will bend at large T. Report where it bends —
   that boundary is itself informative.
4. **The area moves fast.** arXiv:2508.18609 is the paper most likely to eat this contribution;
   check for a v2 or follow-up that adds a mechanism before committing.
5. **Annotation labour**, not GPU, is the binding constraint if D3 is attempted.

---

*Plan derived from three independent reviews plus direct re-analysis of repo artefacts. All model
sizes, datasets, and prior-work claims verified against sources.*
