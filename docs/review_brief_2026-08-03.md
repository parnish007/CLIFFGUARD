# Review brief — findings from the local ladder runs, 2026-08-03

**Purpose.** Everything below is a claim I (Claude) believe is supported by measurement. Your job
is to try to break it. I would rather find an error now than after it is written up.

**Please be adversarial.** Prior rounds of this collaboration found real defects in my work
(inverted calibration tail D0, a fail-open default, a broken Stage 0 gate that I designed twice
wrong, a binomial tail solved backwards). Assume the same rate of error here.

---

## 0. What to review, in priority order

1. **The central negative result** (§4) — is the decoupling real, or is it an artifact I have not
   controlled for? This is the one that matters.
2. **The eta fit and its intervals** (§3) — am I over-reading a model-conditional OLS interval?
3. **The controlled C5 arm** (§5) — is the salience contrast actually controlled?
4. **The code** — `scripts/run_local_ladder.py`, `scripts/analyse_margin_normalisation.py`,
   `tests/test_run_local_ladder.py`. Correctness, not style.
5. **The novelty claim** (§7) — am I claiming more than the data supports?

---

## 1. Setup

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, 28 layers, D=1536 |
| Readout | `hidden_states[14]`, last token, one forward per prompt |
| Corpus | Fold A, `Anthropic/hh-rlhf`, 250 refused / 250 benign, deduped within AND across classes |
| Direction | `r = mean(harmful) - mean(harmless)`; harmful scores HIGH, so `fires_high=True` everywhere |
| Margin | `m(x) = <a(x), r> / ||a(x)||` |
| d' | held-out: direction fitted on one half, scored on the other, 50 splits |
| eta | `sum_j Var[r . (W_fp16 - W_q)_j] / s^2` over `down_proj` at blocks 0..13; `s^2` = FP16 benign margin variance |

Three ladders, all run locally on an RTX 3050 6 GB:

- **RTN** — in-process per-group asymmetric round-to-nearest, 8..2 bits, group 64, block
  `nn.Linear` only (embeddings and `lm_head` stay FP16). bits/param = `code_bits + 32/group`, exact.
- **SAL** — identical, plus AWQ-style activation-aware per-input-channel scaling (alpha=0.5)
  before RTN with the inverse folded back out. Calibrated on 64 **benign** prompts.
- **GGUF** — `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, F16 through Q2_K, each rung dequantized into a
  dense fp16 torch model via transformers `gguf_file=`.

Run directories: `artifacts/runs/20260803-075641_*_local-ladder-rtn-qwen1.5b`,
`*_local-ladder-salience-qwen1.5b`, `*_local-ladder-gguf-qwen1.5b`.

---

## 2. Stage 0 — rotation replicates

Disjoint prompt halves, tangent shift `delta - (delta.r0)r0`, alignment of the two halves.
Chance SD = `1/sqrt(1536)` = 0.0255.

| bits | RTN rotation | median cos | z |
|---|---|---|---|
| 8 | 1.77 deg | +0.5415 | +21.2 |
| 7 | 3.88 | +0.6224 | +24.4 |
| 6 | 7.55 | +0.6387 | +25.0 |
| 5 | 14.81 | +0.6199 | +24.3 |
| 4 | 31.95 | +0.6196 | +24.3 |
| 3 | 55.02 | +0.6591 | +25.8 |
| 2 | 88.69 | +0.3910 | +15.3 |

8/8 PASS. GGUF 8/8, SAL 7/7.

**Question for you:** the median cosine is ~0.6, not ~1.0. I read that as "systematic but with real
per-half estimator noise." Is there a reading where 0.6 indicates something is wrong with the test
rather than with the estimate?

---

## 3. Stage 2 — eta follows an exponential, base is NOT 4

| ladder | eta_4 | base | 95% CI | R^2 | n |
|---|---|---|---|---|---|
| RTN | 0.0602 | **4.3552** | [4.1136, 4.6110] | 0.9989 | 7 |
| SAL | 0.0993 | 4.2154 | [4.0291, 4.4103] | 0.9993 | 7 |
| GGUF | 0.00904 | 3.4060 | [2.8584, 4.0584] | 0.9895 | 6 |

RTN eta: 8.87e-5 (8 bit) -> 6.54e-1 (2 bit), a factor of 7374.

**Claims I am making:**
1. Assumption A3's *form* holds; its *constant* (4) is rejected for RTN and SAL.
2. RTN and GGUF intervals do not overlap (4.1136 > 4.0584), so the base is a property of the
   quantizer family, not a universal.

**Questions for you:**
- Claim 2 rests on a 0.055 gap between two model-conditional OLS intervals computed on data that
  share a checkpoint, a direction, and an activation sample. Is that gap meaningful at all, or
  should I only say "the point estimates differ and the intervals nearly touch"?
- The GGUF regressor is `payload bits/param`, which for k-quants is non-integer and reflects mixed
  per-tensor type assignment (Q4_K_M = 5.003 bits/param). Does regressing on that confound
  bit-width with type composition in a way that makes the GGUF base incomparable to RTN's?
- I use `s^2` = FP16 benign margin variance as the eta denominator. It is a scale choice. Does it
  affect the fitted *base* at all? I believe not (it is a constant factor, so it shifts `eta_4` and
  leaves the slope alone), but please check that reasoning.

---

## 4. THE CENTRAL RESULT — d' does not move

| scheme | held-out d' | sd |
|---|---|---|
| FP16 | 0.4129 | 0.0891 |
| RTN 8-bit | 0.4117 | 0.0894 |
| RTN 4-bit | 0.3996 | 0.0822 |
| RTN 3-bit | 0.4821 | 0.0836 |
| RTN 2-bit | 0.3914 | 0.0847 |

FP16 -> 2-bit: d' drops 0.0215, i.e. 0.25 sd, while eta grows 7374x and the direction rotates
88.69 degrees.

GGUF agrees: 0.4129 -> 0.4323 (slightly UP). SAL agrees: 0.4129 -> 0.4117.

Falsifier F5 (eta from weights vs eta implied by d' decay): ratio spans **374x** on RTN and
**225x** on GGUF, falling monotonically in both.

### The control I already ran

Objection: the margin is norm-normalised, so it is structurally blind to isotropic norm inflation.
`scripts/analyse_margin_normalisation.py` recomputes d' from the same saved activations using the
RAW projection `<a, r>`:

| | FP16 | RTN 2-bit | drop |
|---|---|---|---|
| normalised | 0.4129 | 0.3914 | +0.25 sd |
| raw | 0.4071 | 0.3909 | +0.19 sd |

Both flat. Also measured at 2-bit: mean `||a||` inflates 1.86x, and the mean projection onto the
direction goes NEGATIVE (-3.87x its FP16 value).

### My interpretation

The model has not lost the harmful/benign distinction; it has moved where the distinction lives. A
difference-in-means direction refitted on the quantized model finds it again at full strength.
Theorem 1 measures the FP16 direction's continued *validity* and treats it as the *information*.

**Questions for you — this is where I most want to be wrong:**

1. **Is held-out d' the right estimator here?** It refits the direction on the quantized model's
   own activations at every split. That is exactly why it is insensitive to rotation. Is the flat
   curve therefore a tautology of my estimator rather than a fact about the model? If so, what is
   the estimator that would show degradation, and is it a fair test?
2. **Is d' = 0.41 simply too small to detect anything?** With sd 0.085 and n=250, what effect size
   could this design actually detect? I have not computed the power. If the MDE is, say, 0.24 d'
   units, then "no degradation" is nearly uninformative and I should say so numerically rather than
   qualitatively.
3. **The 3-bit rung goes UP (0.4821) by more than a sd above FP16.** I have no explanation. Is that
   a red flag for the whole measurement?
4. Is there a control I am missing that would distinguish "information preserved, coordinates
   moved" from "probe was never measuring the thing that degrades"?

---

## 5. Claim C5 as a controlled experiment

Same checkpoint, group size, bit budget, corpus, prompt order, seed, code path. Only difference:
salience-aware channel scaling.

| bits | RTN rot | SAL rot | change | RTN eta | SAL eta | eta ratio |
|---|---|---|---|---|---|---|
| 8 | 1.77 | 1.38 | -21.9% | 8.87e-5 | 1.62e-4 | 1.82x |
| 6 | 7.55 | 5.85 | -22.5% | 1.46e-3 | 2.86e-3 | 1.95x |
| 4 | 31.95 | 24.46 | -23.4% | 2.56e-2 | 4.44e-2 | 1.73x |
| 2 | 88.69 | 88.11 | -0.7% | 6.54e-1 | 9.89e-1 | 1.51x |

Claim: salience-awareness injects 1.5-2.0x MORE total weight perturbation and rotates the
behavioural direction 22-26% LESS.

**Questions:**
- Is the rotation reduction outside what prompt-sampling noise would produce? I have not put an
  interval on the 22-26%. How should I?
- `eta` here is measured with the FP16 direction `r_fp16` for both arms. Is that the right
  comparison, or does it bias in favour of the arm that preserves `r_fp16`?
- The scaling is normalised by `scale / scale.mean()`. Does that keep the bit budget honestly
  identical, or have I smuggled in extra precision?

---

## 6. Incidental findings

1. **The F16 GGUF conversion alone rotates the direction 9.49 degrees, with zero quantization** —
   larger than 6-bit RTN (7.55). Implication: comparing a GGUF Q4 model against a *safetensors*
   FP16 reference charges that constant offset to quantization. Is this a known artifact with a
   known cause (tied vs untied embeddings? conversion rounding? norm epsilon?) or is it new?
2. **Irrecoverable damage fraction FALLS as bits drop** (1.000 at 8-bit -> 0.511 at 2-bit).
   Counterintuitive. Real, or an artifact of how `parallel_orthogonal_split` normalises?
3. **Salience protection is non-monotone in alpha**: salient-channel MSE 0.00637 (alpha=0) ->
   0.00541 (0.25) -> 0.00247 (0.5) -> 0.00391 (1.0). I attribute this to boosted channels
   dominating the group min/max. Check that explanation.
4. `verify_gguf_pair.py` was silently reporting 0.00 B for all RSS because `GetCurrentProcess` was
   called without an explicit `restype`, truncating the `(HANDLE)-1` pseudo-handle to 32 bits.
   Fixed. Measured peak now 7.03 GiB from a 964.80 MiB baseline.

---

## 7. Novelty claim — please challenge this

Literature says: quantization *largely preserves* the refusal direction, so safety is robust; and
fine-tuning drift *correlates with* safety degradation.

I claim:
- Quantization rotates the direction far more than "largely preserves" suggests, once you go below
  4 bits, and it does so along a precise measurable law.
- Drift does NOT imply degradation for quantization, which contradicts the assumed link.

**Is that overclaiming?** Specifically: my d' has a label ceiling (see §8), so can I honestly say
anything about "degradation" at all, or only about "this probe's discriminability"?

---

## 8. The caveat I attach to everything

`d'_0 = 0.413` is a **label ceiling**. "Refused" describes the hh-rlhf *rejected response*, not
this model's behaviour. Consequences: `b*` is undefined because `d'_0 < z_0.95 = 1.645`, and a probe
with little discriminability has little to lose.

**Question:** given this, is the honest headline "quantization does not degrade this behaviour" or
"this measurement cannot detect degradation"? I currently write the latter. Am I being too
conservative, or not conservative enough?

---

## 9. What I am about to build (review the plan, not just the past)

Priority order, all local:

1. **Model-derived labels.** Generate completions from the FP16 model on the Fold A prompts,
   classify refusal from the model's OWN output, and relabel. This lifts the label ceiling and is
   the single point of failure for every result above.
2. **A power analysis** for the d' comparison, so "no degradation" becomes a numerical statement.
3. **Sector generalisation** — the same ladder pointed at reasoning and code, not just refusal.

Is this the right order? Is there something cheaper that would falsify the central result faster?

---

## 10. Files to read

- `scripts/run_local_ladder.py` — the ladder, eta, all stages
- `scripts/analyse_margin_normalisation.py` — the negative control
- `tests/test_run_local_ladder.py` — 27 tests on the quantizer
- `docs/results_local_ladder.md` — full write-up
- `docs/theorems.md` section 8 — the verdicts and the restated Theorem 1
- `cliffguard/eval/{noise_floor,isotropy,discriminability,noise_spectrum,composition}.py` — the
  instrument

**Write your review to `docs/codex_review_2026-08-03.md` in this repository.** Earlier attempts to
write outside the repo were blocked by your sandbox.
