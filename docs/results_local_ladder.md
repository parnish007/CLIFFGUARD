# Local ladder results — Qwen2.5-1.5B-Instruct, RTN 8→2 bits

Run: `artifacts/runs/20260803-075641_dafdc44_local-ladder-rtn-qwen1.5b`
Command: `python scripts/run_local_ladder.py --n 250 --bits 8 7 6 5 4 3 2 --splits 50`
Hardware: RTX 3050 6 GB Laptop. Wall clock ~12 min. No downloads.

**Headline: the geometry behaves exactly as the theory predicts, and the behaviour does not
follow it. Falsifier F5 fires.**

---

## 1 Setup

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, 28 layers, D=1536 |
| Read point | `hidden_states[14]`, last token |
| Corpus | Fold A, `Anthropic/hh-rlhf`, 250 refused / 250 benign, deduped within and across classes |
| Direction | `r = mean(harmful) − mean(harmless)`, so harmful scores HIGH (`fires_high=True`) |
| Ladder | per-group asymmetric round-to-nearest, group 64, block `nn.Linear` only; embeddings and `lm_head` stay FP16 |
| Bits/param | `code_bits + 32/group` — exact, not estimated from a file format |
| η | Σ Var[r·(W_fp16 − W_q)] over `down_proj` at blocks 0–13, ÷ FP16 benign margin variance |

**Why RTN and not GGUF for the primary ladder.** RTN varies *only* bit-width. One checkpoint, one
quantizer family, one group size. That is the ordinal dose axis assumption A3 is stated over. GGUF
k-quants vary block structure and per-tensor type assignment as well, so they are a messier axis;
they are the deployment-realism arm, not the primary one.

---

## 2 Stage 0 — the rotation is systematic at every rung

Split the prompts into disjoint halves, compute the tangent shift on each, measure alignment.
Chance SD = 1/√1536 = 0.0255.

| Scheme | rotation | median cosine | z | verdict |
|---|---|---|---|---|
| NF4 | 24.05° | +0.5695 | **+22.3** | PASS |
| RTN 8-bit | 1.77° | +0.5415 | +21.2 | PASS |
| RTN 7-bit | 3.88° | +0.6224 | +24.4 | PASS |
| RTN 6-bit | 7.55° | +0.6387 | +25.0 | PASS |
| RTN 5-bit | 14.81° | +0.6199 | +24.3 | PASS |
| RTN 4-bit | 31.95° | +0.6196 | +24.3 | PASS |
| RTN 3-bit | 55.02° | +0.6591 | +25.8 | PASS |
| RTN 2-bit | 88.69° | +0.3910 | +15.3 | PASS |

**8/8 replicate.** The rotation is a property of the quantizer, not of these prompts.

The angle roughly **doubles per bit removed** (1.77 → 3.88 → 7.55 → 14.81 → 31.95), then saturates
as it approaches orthogonality. At 2 bits the refusal direction is 88.7° from its FP16 self —
essentially a different direction.

This is claim **C1, supported**, and it is the cleanest result in the run.

---

## 3 Stage 1 — the perturbation is not isotropic, and gets *less* so as bits fall

| Scheme | max abs z | irrecoverable fraction | excess kurtosis (obs / null ref) |
|---|---|---|---|
| NF4 | 5.61 | 0.957 | 0.73 / 1.39 |
| RTN 8-bit | 1.71 | 1.000 | 0.16 / 1.39 |
| RTN 7-bit | 6.12 | 0.999 | 1.53 / 1.39 |
| RTN 6-bit | **17.86** | 0.996 | 8.27 / 1.39 |
| RTN 5-bit | 3.59 | 0.983 | 0.39 / 1.39 |
| RTN 4-bit | 3.15 | 0.924 | 0.43 / 1.39 |
| RTN 3-bit | 3.41 | 0.787 | 0.68 / 1.39 |
| RTN 2-bit | 3.90 | 0.511 | 0.61 / 1.39 |

The concentration null is rejected at 7 of 8 rungs. Only 8-bit fails to reject, and that is the
rung where the perturbation is smallest and the test has least to work with.

**Irrecoverable fraction falls monotonically, 1.000 → 0.511.** At high precision essentially all
the damage is orthogonal to the direction — no rescaling can undo any of it. At 2 bits nearly half
the damage lies *along* the direction and is in principle recalibratable. This inverts the naive
expectation that low precision is where recalibration stops working.

The 6-bit kurtosis spike (8.27 against a null of 1.39) is unexplained and is flagged rather than
smoothed over. It is a single rung and may be an artifact of group boundaries at that step size.

---

## 4 Stage 2 — η follows the theory almost exactly, and the base is not 4

| Scheme | bits/param | η |
|---|---|---|
| RTN 8-bit | 8.500 | 8.87 × 10⁻⁵ |
| RTN 7-bit | 7.500 | 3.58 × 10⁻⁴ |
| RTN 6-bit | 6.500 | 1.46 × 10⁻³ |
| RTN 5-bit | 5.500 | 6.04 × 10⁻³ |
| RTN 4-bit | 4.500 | 2.56 × 10⁻² |
| RTN 3-bit | 3.500 | 1.19 × 10⁻¹ |
| RTN 2-bit | 2.500 | 6.54 × 10⁻¹ |

Fitting `η(b) = η₄ · base^(4−b)`:

```
η₄       = 0.0602
base     = 4.355     95 % CI [4.114, 4.611]
R²       = 0.9989    rmse(log) = 0.0993    n = 7
```

**The interval excludes 4.** Assumption A3's assumed base of exactly 4 is rejected at the 5 % level
— but the measured base is 4.36, close enough that A3 is *approximately* right and precisely wrong.
This is a result, not a failure: Theorem 2 should carry a measured base, and for this
quantizer on this model that base is 4.36.

The caveat the module already carries applies: the rungs share weights, direction, and activation
sample, so this OLS interval is **model-conditional**, not a pre-registered verdict. A defensible
project-level claim needs repeated estimates and a covariance-aware interval.

---

## 5 Stage 3 — d′ does not move. This is the finding.

| Scheme | held-out d′ | in-sample d′ | gaussianity gap |
|---|---|---|---|
| FP16 | **0.4129 ± 0.0891** | 0.5735 [0.397, 0.769] | 0.003 |
| NF4 | 0.4080 ± 0.0867 | 0.5610 | 0.005 |
| RTN 8-bit | 0.4117 ± 0.0894 | 0.5723 | 0.004 |
| RTN 7-bit | 0.4084 ± 0.0889 | 0.5707 | 0.003 |
| RTN 6-bit | 0.4157 ± 0.0891 | 0.5726 | 0.002 |
| RTN 5-bit | 0.4012 ± 0.0870 | 0.5488 | 0.003 |
| RTN 4-bit | 0.3996 ± 0.0822 | 0.5496 | 0.005 |
| RTN 3-bit | 0.4821 ± 0.0836 | 0.6152 | 0.002 |
| RTN 2-bit | **0.3914 ± 0.0847** | 0.7097 | 0.000 |

From FP16 to 2-bit, held-out d′ goes 0.4129 → 0.3914. That is a change of 0.02 against a standard
deviation of 0.085 — **no detectable degradation**, at a bit-width where η has grown by a factor of
7400 and the direction has rotated 88.7°.

Every gaussianity gap is ≤ 0.005, so assumption A2 is not the problem here; the equal-variance
Gaussian ROC model fits.

### Falsifier F5 fires

| Scheme | η from d′ decay | η from weights | ratio |
|---|---|---|---|
| RTN 8-bit | 5.74 × 10⁻³ | 8.87 × 10⁻⁵ | 64.7 |
| RTN 7-bit | 2.24 × 10⁻² | 3.58 × 10⁻⁴ | 62.7 |
| RTN 6-bit | — (d′ rose) | 1.46 × 10⁻³ | — |
| RTN 5-bit | 5.94 × 10⁻² | 6.04 × 10⁻³ | 9.83 |
| RTN 4-bit | 6.77 × 10⁻² | 2.56 × 10⁻² | 2.64 |
| RTN 3-bit | — (d′ rose) | 1.19 × 10⁻¹ | — |
| RTN 2-bit | 1.13 × 10⁻¹ | 6.54 × 10⁻¹ | 0.173 |

**Ratio spread 374×.** The pre-registered scale-free form of F5 is whether this ratio is constant
across rungs; it is not, it falls monotonically by more than two orders of magnitude.
**C2's mechanism is refuted on this model.**

The two accounts do not merely differ by a constant — they differ in *direction*. Weight-space η
grows 7400× down the ladder; behavioural η grows about 20×. The transfer function from weight noise
to behavioural noise is strongly compressive, not the identity the theory assumes.

---

## 6 Stage 4 — the prediction "succeeds" for the wrong reason

Fit `η₄` on the three high-precision rungs, predict the four low-precision ones with no refitting.

| Scheme | bits | predicted d′ | measured d′ | error | within 1 sd |
|---|---|---|---|---|---|
| RTN 5-bit | 5.50 | 0.4115 | 0.4012 ± 0.0870 | +0.0103 | yes |
| RTN 4-bit | 4.50 | 0.4069 | 0.3996 ± 0.0822 | +0.0073 | yes |
| RTN 3-bit | 3.50 | 0.3886 | 0.4821 ± 0.0836 | −0.0934 | **no** |
| RTN 2-bit | 2.50 | 0.3305 | 0.3914 ± 0.0847 | −0.0609 | yes |

Out-of-sample RMSE = 0.056 d′ units; 3 of 4 within one standard deviation.

**This is not evidence for C3 and must not be reported as such.** The measured curve is flat, the
predicted curve is nearly flat, and the error bars are ±0.085. A flat prediction matching a flat
measurement carries almost no information. The test has essentially no power here because there is
no decay to predict.

`b*` is undefined: `d′₀ = 0.413 < z₀.₉₅ = 1.645`. The full-precision model is already below the
operating threshold the collapse bound is defined against, so there is no crossing to locate.

---

## 6b Negative control — the flat curve is not a readout artifact

The obvious objection to §5 is that the margin is norm-normalised,
`m(x) = ⟨a(x), r̂⟩ / ‖a(x)‖`, so it would be *structurally* blind to the isotropic norm inflation
Theorem 1 is built on. `scripts/analyse_margin_normalisation.py` recomputes held-out d′ from the
same saved activations using the raw projection, no normalisation, same held-out protocol.

| Scheme | d′ normalised | d′ raw | ‖a‖ / FP16 | mean proj / FP16 |
|---|---|---|---|---|
| FP16 | 0.4129 | 0.4071 | 1.0000 | 1.000 |
| NF4 | 0.4080 | 0.4000 | 0.9725 | 2.641 |
| RTN 8-bit | 0.4117 | 0.4058 | 0.9979 | 0.907 |
| RTN 6-bit | 0.4157 | 0.4089 | 0.9997 | 2.678 |
| RTN 4-bit | 0.3996 | 0.3926 | 0.9950 | 3.553 |
| RTN 3-bit | 0.4821 | 0.4750 | 1.0111 | 2.317 |
| RTN 2-bit | 0.3914 | 0.3909 | **1.8561** | **−3.872** |

FP16 → 2-bit: normalised drop +0.25 sd, raw drop +0.19 sd. **Both flat. The normalisation is
exonerated** and the refutation in §5 stands on its own.

The last row is the more interesting one. At 2 bits the activation norm inflates by 1.86× and the
mean projection onto the readout direction **goes negative** — the direction has reoriented past
orthogonal — and d′ is still 0.39.

**Separability survives a coordinate system that has effectively reversed.** The quantized model has
not lost the harmful/benign distinction; it has moved where the distinction lives, and a
difference-in-means direction refitted on the quantized model finds it again at full strength.
Theorem 1 measures the *FP16 direction's continued validity*, which decays fast and predictably,
and treats it as the *information*, which does not decay at all. Those are different quantities.

---

## 6c Claim C5, as a controlled experiment

Run: `artifacts/runs/*_local-ladder-salience-qwen1.5b`
Command: `--salience-alpha 0.5 --salience-calib 64` — AWQ-style activation-aware per-input-channel
scaling applied before RTN, inverse folded back out.

**Why this and not a published AWQ build.** Setting our RTN against `Qwen2.5-1.5B-Instruct-AWQ`
would confound salience-awareness with that build's group size, calibration set, kernel, and
release. Here the checkpoint, group size, bit budget, corpus, prompt order, seed, and code path are
all identical to §2–§5. The *only* difference is whether the quantizer protects high-activation
input channels. The calibration uses 64 **benign** prompts — no harmful prompts, no labels.

| bits | RTN rotation | salience-aware rotation | change | RTN η | salience η | η ratio |
|---|---|---|---|---|---|---|
| 8 | 1.77° | 1.38° | **−21.9 %** | 8.87e−5 | 1.62e−4 | 1.82× |
| 7 | 3.88° | 2.87° | **−26.2 %** | 3.58e−4 | 6.26e−4 | 1.75× |
| 6 | 7.55° | 5.85° | **−22.5 %** | 1.46e−3 | 2.86e−3 | 1.95× |
| 5 | 14.81° | 11.46° | **−22.6 %** | 6.04e−3 | 1.11e−2 | 1.84× |
| 4 | 31.95° | 24.46° | **−23.4 %** | 2.56e−2 | 4.44e−2 | 1.73× |
| 3 | 55.02° | 46.10° | −16.2 % | 1.19e−1 | 1.86e−1 | 1.56× |
| 2 | 88.69° | 88.11° | −0.7 % | 6.54e−1 | 9.89e−1 | 1.51× |

**The two columns move in opposite directions.** Salience-aware quantization injects **1.5–2.0×
more total weight perturbation** and yet rotates the behavioural direction **22–26 % less**, and
that reduction is near-constant across four rungs of the ordinal axis. It spends more error overall
in order to steer error away from the direction that matters.

That is precisely the anisotropy C5 is about, now demonstrated as a controlled contrast rather than
inferred from a literature disagreement.

The effect **vanishes at 2 bits** (−0.7 %). Both quantizers saturate near orthogonality, so there
is no headroom left to protect. Salience-awareness buys rotation resistance in the regime where
rotation is still small — not in the regime where it has already been lost.

Both ladders obey the same decay law with overlapping intervals:

```
RTN               base 4.3552   95 % CI [4.1136, 4.6110]   R² = 0.9989
salience-aware    base 4.2154   95 % CI [4.0291, 4.4103]   R² = 0.9993
```

**C5 needs restating.** Its original form — "RTN/NF4/GGUF isotropic → cliff; AWQ/GPTQ anisotropic →
no cliff" — has a false premise: the concentration null is rejected for plain RTN at 7 of 8 rungs
(§3), so RTN is *not* isotropic either. The real distinction is not isotropic versus anisotropic
but **where the anisotropy points**. Both quantizers concentrate their damage; only the
salience-aware one concentrates it away from the behavioural direction.

Held-out d′ is flat for the salience ladder too (0.4129 → 0.4117 at 2 bits), which is consistent
with §5 and §6b: less rotation, same discriminability, because discriminability was never tracking
rotation in the first place.

---

## 6d The deployment-realistic arm — llama.cpp k-quants

Run: `artifacts/runs/20260803-083534_fe871f3_local-ladder-gguf-qwen1.5b`
Command: `--ladder-kind gguf`. Ladder: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, F16 → Q2_K, all seven
files converted by Qwen from one F16 checkpoint (10.1 GB, downloaded locally).

| Scheme | payload bits/param | rotation | η | held-out d′ |
|---|---|---|---|---|
| GGUF F16 | 16 | **9.49°** | — | 0.4073 |
| Q8_0 | 8.502 | 9.83° | 4.13e−5 | 0.4064 |
| Q6_K | 6.565 | 11.61° | 4.25e−4 | 0.4087 |
| Q5_K_M | 5.760 | 13.04° | 8.26e−4 | 0.4101 |
| Q4_K_M | 5.003 | 19.61° | 2.35e−3 | 0.4112 |
| Q3_K_M | 4.135 | 29.98° | 6.18e−3 | 0.4443 |
| Q2_K | 3.362 | 56.25° | 2.83e−2 | 0.4323 |

8/8 replicate (z = 22–25). Fitted `base = 3.406, 95 % CI [2.858, 4.058]`, R² = 0.9895 — a looser
fit than RTN's 0.9989, which is what a messier axis should look like. Held-out d′ is flat again
(0.4129 → 0.4323, *up* slightly), and **F5 fires again**: ratio spread 225×. The normalisation
control passes here too (−0.24 sd normalised, −0.23 sd raw).

### Two findings specific to this arm

**1. The F16 GGUF conversion alone rotates the direction 9.49°, with zero quantization.**

That is larger than *6-bit* RTN (7.55°) and far larger than 8-bit RTN (1.77°). Converting
safetensors FP16 → GGUF F16 is not behaviourally lossless with respect to the refusal direction.

This is a methodological problem for any cliff measurement built on a GGUF ladder, including the
one this project originally planned: **a constant ~9.5° offset is baked into every rung and has
nothing to do with bit-width.** Comparing a GGUF Q4 model against a *safetensors* FP16 reference —
which is the natural thing to do, and what the literature generally does — attributes that offset
to quantization. The correct reference for a GGUF ladder is the GGUF F16 member, not the original
checkpoint.

**2. The RTN and k-quant decay bases do not agree.**

```
RTN     base 4.3552   95 % CI [4.1136, 4.6110]
GGUF    base 3.4060   95 % CI [2.8584, 4.0584]
```

The intervals do not overlap (4.1136 > 4.0584), narrowly. The decay is exponential in both
families, but the base is a property of the *quantizer family*, not a universal constant — which
is another reason `collapse_bits_threshold_closed_form` must never be called with its default
`base=4.0`.

Note also that k-quant payload bits/param are not integers (8.502, 6.565, 5.760, 5.003, 4.135,
3.362) because the type assignment is mixed per tensor. Q4_K_M stores 5.003 bits/param, not 4 —
another reason it is not interchangeable with a 4-bit RTN rung.

**Stage 4 reports 3/3 within 1 sd and RMSE 0.0235**, and as in §6 that is not evidence for C3: the
measured curve is flat, so a flat prediction cannot be wrong. `b*` is undefined for the same
label-ceiling reason.

`scripts/verify_gguf_pair.py` was run on the real F16/Q4_K_M pair: all 339 tensors corresponded
and dequantized, **PASSED**. Its `UNVERIFIED-AGAINST-REAL-FILE` marker is cleared. That run also
exposed a bug in the script's own RSS instrumentation, which had been silently reporting 0.00 B —
see the file header.

---

## 7 What this does and does not establish

**Established.**

1. **C1 holds, strongly.** The FP16→quantized rotation of the refusal direction replicates across
   disjoint prompt halves at every rung (z = 15–26). It is systematic, not prompt-idiosyncratic.
2. **The rotation is a smooth, monotone function of bit-width**, roughly doubling per bit until it
   saturates near orthogonality.
3. **Weight-space η follows an exponential law with a precisely measured base of 4.355
   [4.114, 4.611]**, R² = 0.999 — a sharper statement than assumption A3's assumed 4.
4. **The perturbation is anisotropic at 7 of 8 rungs**, and the share of damage that no
   recalibration can remove *falls* as precision drops, 1.000 → 0.511.
5. **Rotation and discriminability are decoupled.** An 88.7° rotation of the refusal direction
   produced no measurable change in d′. Direction rotation is therefore **not** a valid proxy
   metric for safety degradation — a negative result with direct methodological consequences for
   any work that uses direction drift as a safety signal.
6. **F5 fires: C2's mechanism is refuted here.** Weight-space and behavioural η do not move
   together; their ratio spans 374×.

**Not established, and the honest limits.**

1. **d′₀ = 0.41 is a label ceiling, not a model property.** "Refused" describes the hh-rlhf
   *rejected response*, not what this model does. With so little discriminability to begin with,
   the claim is narrow: *within the range this probe can measure*, quantization does not reduce it.
   Deriving labels from the target model's own completions is the single highest-value next step,
   and until it is done the d′-invariance result cannot be read as "quantization is safe".
2. **No completions are generated.** Nothing here is a behavioural claim. Margin discriminability
   is a proxy for behaviour, not behaviour.
3. **η is a proxy.** Summing per-matrix projected variances assumes independent layer contributions
   and isotropic unit-variance activations; neither holds. Per-matrix values are stored in
   `results/stage2_eta.json` so any other aggregation is recomputable without re-running.
4. **One model, one layer, one language, one quantizer family.** Generality is untested here — that
   is what the Colab arms (3B, Phi-3.5-mini, GGUF k-quants) are for.

---

## 8 What this means for the theory

The pivot's central claim was that quantization damages safety behaviour through a measurable
rate-distortion channel: weight noise η rises as bits fall, η degrades d′ by
`d′(b) = d′₀/√(1+η(b))`, and the crossing point predicts collapse.

Half of that chain is now measured and holds beautifully. **The other half does not connect.**
η rises by 7400× and d′ moves by less than a quarter of one standard deviation.

Three readings are consistent with the data, and the next experiments should separate them:

1. **Label ceiling.** d′ ≈ 0.41 is so low that there is nothing to lose. Fix: model-derived labels.
2. **Wrong readout.** The last-token residual at layer 14 may not be where the behaviour lives.
   Fix: sweep layers and token positions.
3. **The mechanism is genuinely wrong.** Behavioural robustness to weight noise may be far larger
   than the linear-projection model assumes, because downstream layers renormalise. Fix: measure
   the transfer function from weight noise to margin noise directly, layer by layer — the
   compressive shape visible in the F5 ratio column is the first evidence of it.

Reading 3 is the interesting one, and the F5 ratio falling monotonically from 64.7 to 0.17 is
already a quantitative fingerprint of it. That is the finding worth chasing.
