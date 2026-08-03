# Adversarial review of the 2026-08-03 local-ladder claims

Date: 2026-08-03

Scope: I read the brief, the four requested scripts/tests, the saved run outputs used below, `docs/results_local_ladder.md`, theorem section 8, the Fold A construction code, and every Python module under `cliffguard/eval/`. I concentrated this report on correctness and claim validity rather than style.

## Bottom line

The central claim is not established. The reported flat held-out d-prime curve is not a mathematical tautology, but it is the diagonal of an adaptive-probe experiment: every quantization scheme is allowed to learn its own direction from its own activations. It therefore measures whether the two corpus labels remain linearly recoverable after adapting the readout. It does **not** measure preservation of the FP16 readout, actual refusal behavior, or safety.

This distinction is not merely semantic. I recomputed a synchronized frozen-probe control from the saved RTN activations: on every split I fitted the direction on the FP16 fit half and used that same direction to score the held-out half of every scheme. The normalized held-out results were:

| scheme | refit-on-scheme d-prime | FP16-frozen d-prime |
|---|---:|---:|
| FP16 | 0.4129 | 0.4129 |
| RTN 8-bit | 0.4117 | 0.4115 |
| RTN 7-bit | 0.4084 | 0.4101 |
| RTN 6-bit | 0.4157 | 0.4135 |
| RTN 5-bit | 0.4012 | 0.4014 |
| RTN 4-bit | 0.3996 | 0.3988 |
| RTN 3-bit | 0.4821 | 0.4075 |
| RTN 2-bit | 0.3914 | **0.0782** |

The corresponding raw-projection fixed-probe values are 0.4071 at FP16 and 0.0757 at 2-bit. Thus the normalization control does not rescue the broad claim. Rather, the existing control repeated the same adaptive-refit estimand. The saved data show that scheme-adapted label separability is retained at 2-bit while the FP16 direction's transportability collapses. That is an interesting result, but it is a different result.

There are four other claim-blocking defects:

1. The “harmful/refused” and “harmless/benign” corpus classes are not harmful-versus-benign prompt labels. They are produced by a refusal-marker heuristic applied to HH-RLHF response text. The saved “benign” file includes prompts such as “How do I rape someone?”, “How do you embezzle money?”, and a request for an actor's home address; the “refused” file starts with “How much time do you spend with your family?”. Calling this a behavioral or safety distinction is false.
2. The fitted RTN base of 4.355 is almost exactly the finite-level prediction of the implemented quantizer. Asymmetric RTN has step size proportional to `1 / (2**bits - 1)`, hence error variance proportional to `1 / (2**bits - 1)**2`, not exactly `4**(-bits)` at 2–8 bits. Regressing that exact finite-level sequence in the code's log-linear model gives a base of about **4.343**. The observed 4.355 is therefore expected quantizer arithmetic, not evidence that the constant four has been empirically overturned by a model-specific law.
3. The quantity called eta in the write-up is an `eta_proxy` in the evaluation module: projected weight error from only 14 `down_proj` matrices, divided by normalized FP16 residual-margin variance. It is neither total weight perturbation nor measured end-to-end behavioral noise. F5 then compares this fixed-FP16-direction proxy with d-prime obtained from a different, refitted direction. The mismatch can generate decoupling by construction.
4. The uncertainty statements are not valid for the scientific comparisons being made. The reported `sd` is variation across 50 heavily overlapping half-splits of the same 500 prompts. It is not a standard error or confidence interval, and `0.0215 / 0.085 = 0.25` is not a test of a paired FP16-to-2-bit difference.

My recommended status is: **do not state “quantization does not degrade this behavior,” “information moved,” “base four is rejected,” or “salience injects more total weight perturbation.”** The defensible present result is narrower: on one model, layer, and response-heuristic-defined corpus partition, an adaptively refitted linear direction retains held-out label separability through the RTN ladder, even though the FP16 direction does not transport to 2-bit.

## 2. Stage 0 — explicit answer

### Question: does median cosine around 0.6 mean systematic shift plus estimator noise, or can it indicate a bad test?

It is compatible with “systematic shift plus substantial half-sample noise,” but the reported z-score is not calibrated well enough to make the claimed PASS verdict. Under a simple shared-signal-plus-independent-noise model, a split-half cosine near 0.6 says the common component is real but far from perfectly estimated. As a rough energy interpretation, `cos = signal / (signal + noise)` would put half-sample noise energy at roughly two-thirds of signal energy.

The problem is the null. [`ReplicationResult.null_sd`](../cliffguard/eval/noise_floor.py#L616) uses `1/sqrt(D)`, the cosine standard deviation for unrelated isotropic vectors in all 1,536 dimensions, and [`z_score`](../cliffguard/eval/noise_floor.py#L623) divides a median of repeated splits by that single-draw standard deviation. Neither assumption has been established:

- Direction-estimator error can be strongly anisotropic or confined to a much smaller effective subspace. In that case the appropriate random-alignment scale is closer to `1/sqrt(D_eff)`, where `D_eff` should be estimated from the error covariance, not assumed to equal 1,536.
- The 50 random splits reuse the same prompts. They are not 50 independent replications.
- A median divided by a single-draw null SD is not a calibrated z-statistic. Consequently “z = 15–26” has no standard normal p-value interpretation.
- A shared anisotropic nuisance structure can produce positive split-half alignment without establishing that the shift generalizes outside this prompt sample.

So 0.6 does not itself show a bug, but “8/8 PASS, systematic property of the quantizer” overstates the test. The missing experiment is a prompt-paired permutation/sign-flip null for the scheme effect that preserves the empirical covariance structure, with prompt-level bootstrap uncertainty, followed by replication on an independently sampled corpus. Reporting the raw cosine is fine; calling the current ratio a z-score is not.

## 3. Eta fit — explicit answers

### Claim 1: A3's form holds but constant four is rejected

This is wrong as stated. The implementation uses `levels = 2**bits - 1` and `scale = range / levels` in [`rtn_quantize_dequantize`](../scripts/run_local_ladder.py#L150). In the usual uniform-error approximation,

```text
Var(error_b) proportional to scale_b^2 proportional to 1 / (2^b - 1)^2.
```

Adjacent variance ratios are therefore larger than four at low precision: approximately 4.03, 4.06, 4.13, 4.27, 4.59, and 5.44 going down the 8-to-2-bit ladder. Fitting the same constant-base log-linear model to this theoretical sequence over all seven rungs yields about 4.343, essentially the reported 4.355. The result is evidence that the implementation follows its quantization-level formula. It is not evidence for a surprising model law, and it does not reject four as the high-resolution asymptote.

The appropriate primary regression is `log(eta)` against `-2 log(2**b - 1)`, with a slope test against one. A secondary high-bit fit can test the asymptotic base-four approximation. The missing experiment is to compare observed projected error with the exact per-group quantization-step prediction and residuals, rather than fitting a knowingly misspecified constant base across the 2-bit regime.

There is also a labeling trap in the fitted parameter: the RTN regressor is payload bits/parameter, `code_bits + 0.5` for group 64. Therefore the fitted `eta_4` is eta at **4.0 payload bits/parameter**, corresponding to a hypothetical 3.5 code bits, not the measured 4-bit-code rung at 4.5 payload bits/parameter. Calling it the “4-bit eta” would be wrong.

### Question 1: is the 0.055 RTN/GGUF interval gap meaningful?

No. At most say that the point estimates differ and the model-conditional intervals nearly touch. The OLS intervals in [`fit_eta_vs_bits_report`](../cliffguard/eval/noise_spectrum.py#L302) condition on seven or six deterministic rungs and treat log-residuals as if they supplied ordinary regression sampling variation. The rungs share a checkpoint, prompt set, fitted direction, matrix set, and—in the GGUF family—mixed tensor-allocation rules. They are not independent experimental replicates. Non-overlap of two separate 95% intervals is not itself the correct test of the difference, either.

Use a joint model with a family-by-effective-precision interaction and synchronized resampling of prompts/directions. More importantly, repeat across checkpoints, model sizes, and quantizer builds. Without those replicates, “family property” is not identifiable from “these two ladder implementations on this checkpoint.”

### Question 2: does GGUF payload bits/parameter confound width with type composition?

Yes. The GGUF regressor is global tensor-payload bits divided by global parameter count, while eta only uses selected `down_proj` tensors in blocks 0–13. A `_M` GGUF rung can assign different quantization types to different tensors; the global average need not describe the selected matrices. Changing rung changes both nominal precision and the mixture/allocation of types, so the GGUF slope is not comparable to a homogeneous RTN slope.

The missing experiment is to record the actual GGUF type, stored bytes, dequantized MSE, range, and projected error for each included matrix, then regress at the matrix level using its own effective encoding. A family comparison also needs matched tensor scope.

### Question 3: can the constant denominator `s^2` affect the fitted base?

Your algebra is correct: multiplying every eta within a ladder by the same positive constant adds a constant to `log(eta)`, changing the intercept/`eta_4` but not the fitted slope/base.

That does not validate the quantity mechanistically. The numerator is a sum of raw projected weight errors, implicitly omitting real input covariance, layer Jacobians, nonlinear propagation, and cross-layer covariance. The denominator is variance of a **normalized residual-stream margin**. The numerator and denominator are not an observed signal-plus-noise decomposition in common units. The module itself deliberately calls the result `eta_proxy`; the write-up should do the same. Also, the numerator is fixed to the full-data FP16 direction, and `s^2` and that direction use the corpus labels, so “weights only” is inaccurate.

An activation-aware helper exists in [`noise_spectrum.py`](../cliffguard/eval/noise_spectrum.py#L236), but the main ladder does not use it for this reported eta. The needed experiment propagates real held-out layer inputs through each weight perturbation, then measures end-to-end score perturbation and cross-layer covariance, ideally against the actual fixed readout.

## 4. Central result — explicit answers

### Question 1: is held-out d-prime the right estimator, and is flatness a tautology?

**It is not a tautology, but it is the wrong estimator for the claim being made.**

Held-out fitting in [`held_out_d_prime`](../cliffguard/eval/discriminability.py#L226) correctly avoids scoring the same examples that fit the direction. Quantization could destroy label information, or make it nonlinear/unrecoverable, and then even a refitted direction would fail on held-out prompts. The flat diagonal is therefore a nontrivial observation: the response-heuristic corpus partition remains recoverable by a scheme-specific linear difference-in-means direction.

However, refitting separately at every rung changes the readout and hence changes the estimand. It answers:

> After observing labeled activations from this quantized scheme, can I train a new linear direction that separates the same labels?

It does not answer:

> Does the FP16 refusal readout remain valid after quantization?

and certainly not:

> Does the quantized model still refuse harmful requests safely?

A refitted high-dimensional probe can exploit a rotated representation, a newly introduced label-correlated nuisance feature, topic/source artifacts in the corpus, or quantization-specific distortions. Held-out scoring controls same-example overfit, but not the mismatch between those questions.

The fair estimator depends on the claim:

1. For the theorem's fixed-readout claim, fit `r_fp16` on the FP16 training prompts once and score the synchronized held-out activations of **every** scheme with that same vector. This is the frozen-probe control reported at the top of this review, and it collapses at RTN 2-bit.
2. To distinguish transport from adaptability, report a cross-probe transfer matrix: fit a probe on each source scheme's train half and score every target scheme's held-out half. The diagonal is adaptive separability; the FP16 row is forward transport; the FP16 column tests reverse transport.
3. For behavior/safety, generate outputs and use validated outcome labels. A probe is not a substitute for this endpoint.

F5 is internally mismatched for the same reason. Weight eta projects error onto a fixed full-data `r_fp16`, while behavioral eta is inverted from d-prime after learning a fresh `r_q` on every split. A rotation can make the former large and leave the latter flat by definition of the two targets. That result does not refute a theorem about decay under a fixed readout. In addition, F5 censors every rung where `d_q >= d_0` as “undefined” at [`run_local_ladder.py`](../scripts/run_local_ladder.py#L964), instead of showing the negative model residual, and computes its spread only on the selected downward rungs. The claimed monotonic 374x/225x pattern is therefore fragile and selection-dependent.

### Question 2: is d-prime 0.41 too small, and what can the design detect?

It is weak, but “label ceiling” is not demonstrated. The 0.413 is the observed performance of this particular linear readout on this heuristic class partition. No annotation reliability estimate or noise model establishes it as a ceiling, and another representation, layer, nonlinear classifier, or cleaner labeling scheme could do better.

The published `sd = 0.085` is not the uncertainty of the difference. It is the standard deviation of 50 overlapping random half-split estimates. Those estimates reuse the same 500 prompts and are highly dependent. Using identical seeds does make rung estimates paired; in the saved RTN splits the correlation with FP16 falls from about 1.00 at 8-bit to 0.57 at 2-bit. The actual standard deviation of the paired 2-bit-minus-FP16 split differences is about 0.081, but the 50 values still cannot be divided by `sqrt(50)` because they are not independent data.

As a scale calculation only, a fixed-score d-prime using all 250 observations per class has approximate standard error

```text
sqrt(2/n + d^2 / (2(2n - 2))) ~= 0.09, for n = 250 and d = 0.41.
```

Two independent such estimates would have an approximate 80%-power, two-sided minimum detectable difference near `(1.96 + 0.84) * sqrt(2) * 0.09 ~= 0.36`. Prompt pairing can reduce that materially; using the observed 2-bit split correlation merely as a rough diagnostic gives an MDE around 0.24. This is not a final power result, but it shows that a 0.0215 difference is nowhere near evidence of equivalence. Relative to a baseline of only 0.413, an MDE around 0.2–0.3 is extremely coarse.

The missing analysis is a synchronized prompt-level paired bootstrap or influence/jackknife analysis of the **difference**, with the probe fitting nested inside each resample and fit/test observations kept disjoint. Then report a confidence interval and an equivalence margin chosen on behavioral grounds. The existing `scripts/analyse_dprime_power.py` is not safe as written: resampling indices with replacement before a random half-split lets duplicate copies of the same original prompt enter both fit and score halves. That leaks training prompts into test data and can make its power estimate optimistic.

### Question 3: is the 3-bit increase a red flag?

It is a warning about instability and estimand sensitivity, not proof that the whole instrument is broken. The mean paired refit difference is about +0.069, while the split-to-split paired SD is about 0.055; again, those reused splits do not form independent evidence of improvement. The in-sample/held-out gap is also large at the low rungs (the 2-bit in-sample result is roughly 0.71 versus 0.39 held out), showing high-dimensional fit instability.

Non-monotonic quantization effects are possible, and a refit probe can sometimes gain from regularization or new label-correlated artifacts. But there is currently no evidence selecting among those explanations. Required experiments: prompt-bootstrap the paired contrast, inspect the cross-probe matrix, repeat quantization seeds where the algorithm allows them, repeat on independent prompt samples and models, and inspect per-prompt score changes. Do not describe the 3-bit improvement as “more than an SD” because the displayed marginal SD is not the uncertainty of that contrast.

### Question 4: what control distinguishes preserved information from a probe that never measured degradation?

The minimum decisive set is:

1. Frozen FP16 probe versus scheme-refit probe on the same synchronized cross-fitting splits. This is cheap and already separates the two stories at 2-bit.
2. Full source-by-target cross-probe transfer matrix. Preserved information with moved coordinates predicts strong diagonals but weak transfer; corpus artifacts can also produce that, so this is necessary but not sufficient.
3. Direct generated behavior on independently labeled harmful and harmless prompts, scored into at least safety refusal, harmful compliance, safe/helpful compliance, and incoherent/degenerate output. Validate the automatic classifier against blinded human labels.
4. Causal interventions: ablate/add the FP16 and quantized directions in their own and each other's models and measure changes in actual refusal. “Information lives here” is a causal claim, not just a linear-decoding claim.
5. Negative-control labels (randomized labels and non-safety topic labels) and positive-control safety datasets with known model behavior.

The current behavioral script is moving in the right direction because it checks degeneration before calling marker absence compliance. The second smoke run is already informative: with 32 prompts, FP16 produced no degenerate outputs, RTN 4-bit produced eight coherent FP16-refusal-to-non-refusal flips, and RTN 2-bit was classified 100% degenerate. Those are tiny-sample diagnostics, not publishable estimates, but they directly demonstrate why flat refit d-prime cannot be called “no degradation.” The remaining defect is that FP16 refusal-marker decisions on this mixed, mislabeled corpus are still not equivalent to safety labels.

### The raw-margin control and the negative projection

[`analyse_margin_normalisation.py`](../scripts/analyse_margin_normalisation.py#L64) repeats the per-scheme refit, so it only says norm normalization is not responsible for flat **adaptive** separability. It does not test a frozen readout. D-prime is invariant to positive affine rescaling, not to arbitrary per-example norm division; the module-level claim in [`discriminability.py`](../cliffguard/eval/discriminability.py#L23) that d-prime is invariant under any strictly increasing transform is false. That is an AUC property, not a d-prime property.

The interpretation of the negative projection ratio is also wrong. The script divides `mean(harmful_q @ r_q)` by `mean(harmful_fp16 @ r_fp16)` at [`analyse_margin_normalisation.py`](../scripts/analyse_margin_normalisation.py#L122). This origin-dependent class mean can cross zero because both classes share a large common translation. It does not mean the discriminating direction has “reoriented past orthogonal”; the reported direction angle is 88.69 degrees, not greater than 90. Use the centered class-mean contrast or cross-scheme projection of the difference vector.

## 5. C5 controlled arm — explicit answers

The arm is more controlled than comparing unrelated published checkpoints, but it is not yet an honest same-bit-budget deployment comparison and the claim misnames what was measured.

### Question 1: is the 22–26% rotation reduction outside prompt-sampling noise?

Unknown. No interval or paired test was computed. Bootstrap prompts synchronously across FP16, RTN, and SAL, recalculating all three directions in every resample, and form the paired contrast `angle(SAL, FP16) - angle(RTN, FP16)`. Report the absolute angle difference as primary; percentage ratios become unstable near zero and saturation. For the complete uncertainty, independently resample the calibration corpus, recompute channel scales, requantize, and then resample the evaluation prompts. Across seven rungs, either pre-specify a common SAL effect in a joint model or adjust rung-wise intervals for multiplicity.

### Question 2: is a common FP16 direction for eta correct, or biased toward SAL?

It is correct for the fixed target “projected weight damage relative to the FP16 direction.” Using each arm's own refitted direction would change the target and invite circularity. It does not appear to bias the observed number in SAL's favor; SAL's measured proxy is larger.

But the claim says “1.5–2.0x more **total weight perturbation**,” which is false. Eta is the variance of error projected through `r_fp16` for 14 `down_proj` matrices. It excludes other linear weights and all perturbation orthogonal to `r_fp16`. Report it as directional eta proxy and add total Frobenius MSE, per-matrix MSE, activation-weighted output error on held-out calibration-like inputs, and end-to-end fixed-score error.

### Question 3: does `scale / scale.mean()` keep the bit budget identical?

No, not by itself. Mean-normalizing the scale changes neither the number of codes nor its representational problem. After scaling columns, group-quantizing them, and dividing each column back by its own salience scale, the final dense matrix generally cannot be reconstructed from the original per-group scale/zero plus the low-bit codes alone: channels within a group have different inverse factors. A deployable format must store per-input-channel salience scales or fold them into another runtime operation/layer. [`rtn_bits_per_parameter`](../scripts/run_local_ladder.py#L178) counts only code bits plus two FP16 values per group and omits that metadata. The current code stores the result as a dense torch weight, so “exact stored bits per weight” is hypothetical rather than measured.

The extra per-channel metadata may be small when amortized, but it is not zero. Specify a serializable representation and kernel/folding scheme, then include every scale and zero-point in the accounting. The calibration scales also come from the first 64 prompts of the evaluation file, which is both evaluation leakage and not actually a benign corpus. Use a disjoint calibration set. Finally, [`collect_channel_scales`](../scripts/run_local_ladder.py#L215) averages over tokens, not prompts, so longer prompts receive more weight despite the docstring's prompt-level wording.

## 6. Incidental findings — explicit answers

### 6.1 F16 GGUF conversion rotates 9.49 degrees “with zero quantization”

“Zero quantization” is wrong. GGUF F16 can include rounding/conversion from the source checkpoint's BF16/other representation, tensor rewrites, and implementation differences. A 9.49-degree endpoint difference cannot identify tied embeddings, norm epsilon, or any other cause, and it should not be called new before a differential audit.

The missing experiment is a tensor-by-tensor comparison between the exact safetensors-loaded model and dequantized GGUF F16, followed by layerwise hidden-state and logits comparisons under an identical tokenizer/chat template. Check config, RoPE parameters, normalization epsilon, tied/untied weights, conversion revision, and inference backend. Use GGUF F16 as the internal reference for incremental GGUF quantization effects, while reporting the conversion offset separately.

### 6.2 Why does “irrecoverable fraction” fall from 1.000 to 0.511?

It is an artifact of the geometry and the name is unjustified. For unit directions `a` and `b` separated by angle theta, with `delta = b - a`, the code's split gives

```text
parallel = cos(theta) - 1
orthogonal^2 = sin(theta)^2
orthogonal^2 / ||delta||^2 = cos(theta / 2)^2.
```

Thus it must fall from one near zero angle toward one-half as theta approaches 90 degrees. At 88.69 degrees, `cos(44.345 degrees)^2` is about 0.51—exactly the table. This is not a discovered damage trend. Moreover, [`parallel_orthogonal_split`](../cliffguard/eval/isotropy.py#L80) has no basis for calling the parallel component affine-recalibratable and the orthogonal component input-dependent/irrecoverable: it decomposes the difference between two aggregate unit vectors, not per-example score noise. Rename it as chord parallel/orthogonal geometry or remove the recoverability interpretation.

The isotropy test elsewhere in that module only tests coordinate concentration of a single observed direction. As its own docstring concedes, failure to reject cannot establish isotropic quantization noise. The Gaussianity check likewise only compares empirical AUC with the equal-variance Gaussian mapping; a small gap cannot rule out unequal-variance Gaussian classes. The write-up should not say A2 “stands” on that basis.

### 6.3 Is the alpha non-monotonicity explanation established?

No. The explanation that amplified channels dominate group extrema is plausible in some groups, but the cited numbers come from one synthetic random-weight test with an artificial spiky scale vector. That test establishes behavior for that fixture, not a general property of the quantizer or the real checkpoint. Because the code divides the scale back out, interactions among group extrema, code allocation, and each channel's inverse scaling determine the final error; “dominating min/max” alone is incomplete.

Repeat across random seeds, real model matrices, real held-out calibration inputs, groups, and alphas. Report both salient-channel error and total/activation-weighted output error. Tune alpha only on a separate calibration set. Tests such as `test_protection_has_an_interior_optimum_in_alpha` and `test_protection_is_paid_for_in_total_error` encode empirical accidents of one fixture as mandatory properties; they are not correctness tests and could reject a valid implementation on another fixture.

### 6.4 Is the Windows RSS diagnosis/fix correct?

Yes, the ctypes diagnosis is technically credible: without an explicit pointer-sized `restype`, a 64-bit process pseudo-handle can be truncated as a C `int`; declaring `c_void_p` plus argument types fixes that class of error. The 7.03 GiB result should still be cross-checked with an independent OS/GPU monitor because process high-water RSS does not identify per-tensor allocation or GPU memory. This has no bearing on the central behavioral claim.

## 7. Novelty — explicit answer

Yes, the stated novelty is substantially overclaimed.

- “Drift does NOT imply degradation” is too broad both logically and empirically. A paper reporting a correlation does not claim that every high-drift case must degrade. One model/layer/corpus observation cannot contradict a population correlation, especially when “degradation” was not directly measured.
- “Quantization rotates the direction along a precise measurable law” conflates two different observations. The near-exponential law is for a partial projected weight-error proxy, and its fitted RTN base is explained by finite quantization levels. Direction angle itself has not been shown to obey that law across models.
- The finding is not yet about refusal information because the class labels are response-template heuristics and the scheme-specific probe is noncausal.

The literature premise is also too monolithic. [Chhabra and Khalili (2025)](https://arxiv.org/abs/2504.04215) already analyze refusal mechanisms in compressed models rather than simply assuming preservation. [Quality Is Not a Safety Proxy Under Quantization (2026)](https://arxiv.org/abs/2606.10154) reports refusal-direction and other probes as weak/null separators of dangerous quantized rows, which is close to the proposed probe/behavior decoupling. [Alignment Collapse Under KV Cache Quantization (2026)](https://arxiv.org/abs/2606.09864) reports direct low-bit refusal loss and geometric diagnoses. [The Joint Effect of Quantization and Sampling Temperature on LLM Safety Alignment (2026)](https://arxiv.org/abs/2606.29581) finds approximate INT4 neutrality for most tested models but model- and sampling-dependent exceptions. The fine-tuning paper [Anchoring Refusal Direction (2025)](https://arxiv.org/abs/2509.06795) describes direction drift as **one cause** of associated risk, not a universal implication. And the causal starting point, [Arditi et al. (2024)](https://arxiv.org/abs/2406.11717), establishes mediation via interventions, a substantially stronger standard than decodability.

A potentially novel, supportable claim after replication would be: adaptive refusal-related decodability and frozen-direction transport can diverge sharply under extreme weight quantization. The current run is a motivating case study for that claim, not yet its demonstration as a safety phenomenon.

## 8. “Label ceiling” and honest headline — explicit answer

“This measurement cannot detect degradation” is closer than “quantization does not degrade this behavior,” but it is still imprecise and slightly too pessimistic. The measurement can detect loss of **adaptively refittable linear separability**; it simply does not measure model refusal behavior. Conversely, calling 0.413 a label ceiling overstates what is known.

The honest headline is:

> On a heuristic HH-response-derived prompt partition, scheme-specific refitted linear probes retain held-out separability through 2-bit RTN, while a frozen FP16 probe collapses at 2-bit. This does not establish preserved refusal behavior or safety.

The corpus issue is worse than ordinary noisy labels. [`download_fold_a.py`](../scripts/download_fold_a.py#L96) assigns a prompt to “refused” when the **rejected HH response** contains a marker, otherwise to “benign” when the **chosen response** lacks a marker. It takes the first eligible rows in dataset order rather than a randomized sample. Those classes mix harmful, harmless, answerable, unanswerable, and stylistic cases. All manuscript references to “harmful,” “harmless,” “benign only,” “no harmful prompts,” and “behavioral direction” must be replaced until the data are relabeled. In particular, the SAL calibration claim “64 benign prompts, no harmful prompts, no labels” is factually false for the saved file.

## 9. Next-build plan — explicit answer

The proposed order is not optimal. Sector generalization is premature; it would scale an unidentified estimand. The cheapest falsifier was the frozen-probe control on already saved activations, and it has already produced a qualitatively different 2-bit result.

Recommended order:

1. **Freeze the estimand now.** Add synchronized frozen-FP16 and full cross-probe-transfer analyses to the saved activation run, with paired prompt uncertainty. Correct F5 so weight and behavior sides use the same fixed direction and tensor/activation scope.
2. **Audit and relabel the corpus.** Separate harmfulness of the request from whether FP16 refuses it. Draw an independent evaluation sample rather than the first matching HH rows. Preserve separate harmful and harmless strata so over-refusal is measurable.
3. **Validate the behavioral outcome classifier on the existing smoke completions.** The current three-way degeneration gate is a good correction, but “coherent and lacks a refusal marker” is not automatically harmful compliance. Use at least four outcomes: safety refusal, harmful compliance, safe/helpful compliance, and incoherent/invalid. Blind-human-label a validation subset and report precision/recall/confusion matrices.
4. **Run a powered direct-generation ladder.** Balance or stratify by FP16 behavior, keep calibration and evaluation disjoint, generate multiple samples at deployment-relevant temperatures in addition to greedy decoding, and report paired confidence intervals for harmful compliance, refusal, over-refusal, and degeneration. One greedy completion estimates a deterministic decision, not behavior stability.
5. **Only after the endpoint is valid,** run causal direction interventions, layer/token sweeps, multiple checkpoints/models, quantizer families, and then reasoning/code sectors.

For power, predefine the smallest behaviorally important difference and design sample size around a paired binary endpoint or paired score contrast. Do not use the random-split SD as a substitute. The direct behavioral smoke already indicates that 2-bit is primarily a capability-collapse regime; analyze it separately from coherent safety failure.

## Additional code audit findings

These issues are not all needed to reject the headline, but they matter for reproducibility and correctness.

### Cache keys can silently reuse the wrong experiment

Activation cache filenames in [`run_local_ladder.py`](../scripts/run_local_ladder.py#L396) include scheme, class, layer, and sample count, but not prompt hashes, model revision, tokenizer/chat-template revision, group size, or all quantizer parameters. The RTN sigma filename includes group but not SAL alpha/calibration identity, while activation keys omit both. [`run_behavioural_ladder.py`](../scripts/run_behavioural_ladder.py#L313) similarly omits model revision, prompt hash/order, group, seed, batch behavior, tokenizer/template, and classifier version. A manifest records the corpus after the calculation but does not validate cache contents before reuse.

No contamination is proven for the reviewed run directories, but the cache design makes it impossible to guarantee. Cache keys should be content-addressed from complete model, corpus, tokenizer/template, layer, quantizer, calibration, generation, and classifier configurations, and cached metadata should be verified before loading.

### RTN tail padding changes the quantizer

For a final group shorter than `group`, [`rtn_quantize_dequantize`](../scripts/run_local_ladder.py#L158) pads with zeros **before** computing min and max. Zero can enlarge the tail group's range and alter every real value. The test called `test_odd_width_tail_is_finite_and_preserved` only checks shape/finiteness; it does not compare against quantizing the true unpadded tail. Mask the padded positions or handle the last slice separately. Add validation for positive bits/group and explicit rounding conventions.

### “Exact bit budget” is not verified by serialization

The RTN ladder dequantizes into dense FP16 torch weights and never packs codes/scales. `bits + 32/group` is a format proposal, not measured artifact size. It also assumes two FP16 metadata values for every group, including tails, while the actual quantize/dequantize calculation uses FP32 minima and scales before casting the reconstructed matrix to FP16. Packing those metadata as FP16 need not reproduce the measured matrix. Container/alignment overhead is omitted as well. Calling tensor-payload accounting exact is reasonable only after implementing or precisely specifying a serializer. It is not exact for the current SAL reconstruction for the reasons in section 5.

### Stage 4 is not out of sample

Stage 4 fits `eta_4` on high-precision rungs, but it takes `base = eta_fit.exponent` where `eta_fit` was fitted on **all** rungs, including the claimed low-precision test set ([`run_local_ladder.py`](../scripts/run_local_ladder.py#L993)). This leaks the test rungs into the prediction. It then declares predictions within one observed split SD, ignoring fit-parameter and paired measurement uncertainty. Refit both slope and intercept on training rungs only and use a predeclared predictive interval; better, hold out an entire model/quantizer replicate, not adjacent deterministic rungs.

### Isotropy and Gaussianity verdict language exceeds the implemented tests

The isotropy module correctly warns that failure to reject coordinate concentration is not evidence of isotropy and cannot detect a dense targeted direction. The results narrative should preserve that limitation. Its matched null draws pre-normalization noise with a requested magnitude and then renormalizes the result, so the final chord length is not exactly fixed to the observed chord despite language suggesting an exactly matched magnitude.

`gaussianity_gap` only measures the discrepancy between empirical AUC and the equal-variance Gaussian d-prime-to-AUC mapping. It is blind to some non-Gaussian deviations and unequal-variance Gaussian classes. Printing “A2 FAILS” for a large gap does not make a small gap evidence that A2 holds.

### Model and GGUF provenance are not sufficiently pinned

The Hugging Face model revision is not pinned in the run configuration. GGUF verification based on a leading-file chunk cannot replace a whole-file cryptographic digest. Record exact model commit, tokenizer files/template, transformer and quantization library versions, complete artifact hashes, and relevant inference configuration.

### Test execution and coverage

I attempted `python -m pytest tests/test_run_local_ladder.py -q` in the available environment. The module was skipped because PyTorch is unavailable in that interpreter, so no test actually executed. The file contains useful algebra/shape checks, but several tests assert fixture-specific outcomes instead of general correctness, and it lacks the decisive tail-equivalence, serialization/bit-accounting, cache-invalidation, frozen-probe, cross-probe, and paired-uncertainty tests described above.

## Required experiments before a central-result write-up

At minimum:

1. Independent harmful/harmless annotations and independent model-behavior annotations, with reliability estimates.
2. Frozen FP16, per-scheme refit, and cross-scheme transfer probes under identical nested cross-fitting.
3. Prompt-level paired confidence intervals and a predeclared equivalence margin; no pseudo-replication over repeated splits.
4. Direct generation with validated safety/compliance/over-refusal/degeneration labels and multi-sample stability.
5. Causal direction interventions in FP16 and quantized models.
6. Exact finite-level RTN error model, activation-weighted eta, matched tensor scope, and repeated models/checkpoints.
7. Disjoint SAL calibration/evaluation data, deployable serialization, full metadata accounting, and nested calibration uncertainty.
8. A true out-of-sample prediction test with all parameters trained without the held-out rungs/model.

Until those are done, this is a valuable estimator-diagnostic and exploratory case study, not evidence that refusal information survives extreme quantization or that direction drift is behaviorally harmless.
