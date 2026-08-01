# First Real Result — Stage 0 on local hardware

**Date:** 2026-08-02
**Hardware:** RTX 3050 Laptop 6 GB, torch 2.6.0+cu124, transformers 5.14.1, bitsandbytes 0.50.0
**Model:** `Qwen/Qwen2.5-1.5B-Instruct`, layer 14, D = 1536
**Corpus:** real Anthropic HH-RLHF, 200 refused / 200 benign
(`anthropic_hh_refused.jsonl` SHA-256 `d55c44f7…`, `anthropic_hh_benign.jsonl` SHA-256 `292c7ce6…`)
**Scheme pair:** FP16 vs NF4 (bitsandbytes), identical prompts in identical order
**Artifacts:** `artifacts/runs/20260801-200449_e235ed9_stage0-Qwen2.5-1.5B-Instruct-fp16-nf4/`
**Provenance:** git `e235ed9`, corpora hashed order-sensitively, environment pinned in `manifest.json`
**GPU verified:** all 1.544 B params on `cuda:0`, 3.09 GB VRAM, 68–86 % sustained utilisation, 57 ms/forward pass

> This is the project's **first empirical result on real data with a calibrated decision rule.**
> It is one model, one scheme pair, one layer. Treat it as a pilot.

---

## 1. Stage 0 — PASS

| quantity | value |
|---|---|
| observed FP16→NF4 rotation | **24.07°** (cos 0.9131) |
| median replication cosine across disjoint prompt halves | **+0.6262** |
| chance alignment SD (1/√D) | 0.0255 |
| **z** | **+24.5** |
| splits | 50 |

**The rotation replicates.** Independent halves of the prompt set agree strongly on *where* the
direction moved. This is a systematic property of the quantizer, not an artifact of these
particular prompts. Premise P1's precondition survives for this model/scheme.

### Why the gate design mattered

| rule | verdict | correct? |
|---|---|---|
| `exceeds_floor` (split-half floor 47.3° median vs observed 24.1°) | **NULL** | ✗ wrong — compares against an inflated half-sample floor |
| `excludes_zero` (paired CI [21.7, 26.0]) | **PASS** | ✗ meaningless — fires on 40/40 nulls |
| **`rotation_replication`** (z = +24.5) | **PASS** | ✓ null-calibrated: 0/30 false positives |

Both discarded designs would have given a confidently wrong answer here — one falsely NULL, one
vacuously PASS. This is the concrete payoff of null-calibrating the decision rule.

---

## 2. Isotropy — concentration null REJECTED (anisotropic)

| statistic | observed | isotropic null | z |
|---|---|---|---|
| top 1 % of coords | 11.5 % | 8.3 % ± 0.6 | **+5.7** |
| top 5 % | 33.7 % | 27.9 % ± 1.0 | **+6.0** |
| top 10 % | 49.3 % | 44.0 % ± 1.0 | **+5.2** |

| | value |
|---|---|
| excess kurtosis of perturbation | 0.86 |
| excess kurtosis of reference direction | 1.78 |
| parallel / orthogonal | −0.0869 / 0.4078 (**96 % irrecoverable**) |

**This contradicts the earlier Llama-3.2-3B artifact** (z ≈ 1.8, concentration null *not*
rejected, perturbation kurtosis 0.20 against a reference direction of kurtosis 19.34).

Two live explanations, not yet distinguished:
1. **Model-dependent.** Qwen2.5-1.5B's refusal direction is far less heavy-tailed (1.78 vs 19.34),
   so its geometry differs and NF4 damage concentrates differently.
2. **The Llama artifact is confounded.** Its noise floor was never established — that is defect D2,
   and the whole reason Stage 0 exists.

**Consequence for assumption A1:** isotropy is *not* universal. It cannot be assumed; it must be
measured per model and per scheme. Theorem 3's conjecture ("isotropy is a property of the
quantizer") needs revising to "a property of the quantizer *and the model's representation
geometry*."

---

## 3. Discriminability — essentially unchanged

| scheme | in-sample d′ | **held-out d′** (20 splits) | gaussianity gap |
|---|---|---|---|
| FP16 | 0.642 [0.449, 0.857] | **0.507 ± 0.115** | 0.003 |
| NF4 | 0.644 [0.449, 0.859] | **0.513 ± 0.119** | 0.005 |

implied η(NF4) from **held-out** d′ decay = **−0.0241** — i.e. no measurable loss; NF4 is marginally *better*,
well inside noise.

**The headline tension.** The direction rotates by 24° and the rotation is highly systematic
(z = +24.5), yet **discriminability does not degrade at all**. Rotation and degradation are
decoupled here.

This is *consistent* with the theory — Theorem 1 says d′ falls only through variance inflation
(η), and η ≈ 0 means d′ is preserved regardless of how far the direction moved. It is evidence
**against** using direction rotation as a safety-degradation metric, which is exactly what the
retired `Δ_cliff` did.

---

## 4. What this does NOT show

- **NF4 causes no measurable safety-discriminability loss** on this model. No cliff at 4 bits.
  Expected — the predicted cliff zone is Q3_K_M and below.
- **d′ ≈ 0.51 is weak** (AUC ≈ 0.68). This "refusal direction" is a mediocre classifier, and that
  is a label-validity problem: the HH `refused` label reflects the **dataset's** rejected response,
  not whether *Qwen* refused. Codex flagged this (F3); it is now empirically visible. Until labels
  come from the target model's own responses, this is a *harmfulness-ish topic* direction.
- **Sign convention.** `r̂ = mean(harmful) − mean(harmless)` makes harmful score HIGH, so it is a
  harmfulness direction (fires HIGH), not a refusal margin (fires LOW). My first analysis pass used
  the fires-LOW convention and reported d′ = −0.642. Corrected here.
- Nothing behavioural. No completions were generated.

---

## 5. Bugs found by running it

1. **Activation cache key omitted the class** — the benign call silently reused the harmful
   activations, so `mean(harmful) − mean(harmless)` was exactly zero. Caught only because
   `difference_in_means` raises on a zero-norm vector. A permissive implementation would have
   returned NaN and poisoned every downstream number.
2. **transformers 5.x API drift** — `torch_dtype` → `dtype`; `apply_chat_template` returns a
   `BatchEncoding`, not a tensor; `output_hidden_states` belongs on the forward call.
3. **In-sample d′ optimism** — 0.64 → 0.51 held out. Any d′ in this project must be held-out.

---

## 6. Immediate next steps

| # | Action | Why |
|---|---|---|
| 1 | Re-run Stage 0 on **Llama-3.2-3B** (Colab, FP16 ≈ 6.4 GB) | Decide whether §2's isotropy contradiction is model-dependent or a confound in the old artifact |
| 2 | Derive labels from the **target model's own responses** | d′ ≈ 0.51 is a label-quality ceiling, not a model property (F3) |
| 3 | Extend to **Q4_K_M → Q3_K_M → Q2_K** | 4-bit NF4 shows η ≈ 0; the predicted cliff is below it |
| 4 | Measure η from **weights** and compare to η from d′ | Falsifier F5 — the central validation, currently untested |
| 5 | ~~Make held-out d′ the default~~ **DONE** — `held_out_d_prime()` is now wired into the runner and reported first | In-sample is optimistically biased |

---

*Reproduce: `D:/AI/cliffguard-gpu/.venv/Scripts/python.exe scripts/run_local_stage0.py --n 200 --layer 14`*
*(the project `.venv` has no torch; C: has ~5.9 GB free, so the GPU stack lives on D:)*
