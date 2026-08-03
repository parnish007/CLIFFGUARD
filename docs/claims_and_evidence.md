# Claims and Evidence

One page stating exactly what is claimed, what the evidence is, and what is not
claimed. The paper is [`paper/cliff_artifact.pdf`](paper/cliff_artifact.tex);
every number below is generated into `paper/review_stats.json` or
`paper/data.json` by the script named beside it.

**Measurement scope for the whole table.** Round-to-nearest, group 64, applied to
transformer-block linear layers; embeddings and output head at FP16. 500 paired
first-turn HH-RLHF prompts. Greedy decoding, 48 new tokens. Refusal labels come
from Qwen2.5-7B-Instruct loaded in 4-bit, behind a composite degeneracy gate.
Comparisons are always against the *same model at FP16 on the same prompts*.

---

## Claims the evidence supports

| # | Claim | Evidence | Source |
|---|---|---|---|
| 1 | Inside the coherent quantized band, lower precision produces **more** refusal, at **1.15 points/bit** removed, 95% CI **[0.75, 1.55]** | Prompt-level cluster bootstrap, 10,000 replicates, free-intercept fit over each model's coherent band. Per model: Qwen2.5-3B 1.00 [0.44, 1.58]; Phi-3.5-mini 1.29 [0.86, 1.74] | `review_reanalysis.py` → `drift` |
| 2 | Refusal is **inert** from FP16 down to 8.5 bits | +0.4 pp (Qwen2.5-3B), +0.2 pp (Phi-3.5-mini). Extrapolating the band fit back to 16 bits undershoots the observed FP16 rate by 7.5 and 10.9 pp respectively | `review_reanalysis.py` → `drift.*.anchor_error_pp` |
| 3 | Significant paired shifts run **toward** refusal | Qwen2.5-3B at 4.5 bits: 32 new refusals vs 11 new compliances, Holm-adjusted *p* = 0.011; at 5.5 bits *p* = 0.033. Phi-3.5-mini at 4.5 bits: 21 vs 4, *p* = 0.005; at 3.5 bits 41 vs 1, *p* < 0.001 | `review_reanalysis.py` → `transitions` |
| 4 | Refusal→compliance transitions stay **≤ 2.2%**, with a **4.6%** simultaneous 95% upper bound | Exact Clopper–Pearson one-sided bounds; simultaneous version Bonferroni-adjusted over all 14 model×rung cells | `review_reanalysis.py` → `transitions.*.upper95_simultaneous` |
| 5 | Phrase-list scoring of identical completions reports up to **38.4%** | Tight marker list, perplexity-only gate, Qwen2.5-3B at 2.5 bits | `classify_completions_judge.py` |
| 6 | The artifact has **two separable causes** | Where degeneracy is 0.0% both gates admit identical completions and the gap is grader-only (10.8% vs 2.2% at 4.5 bits). At 2.5 bits the gate is the whole story (38.4% NLL-gate vs 0.2% composite-gate, same markers) | `review_reanalysis.py` → `gate_by_grader` |
| 7 | The phrase-list estimator is **non-monotone** in its marker list | It is a paired flip whose two indicators move oppositely as the list grows. Qwen2.5-3B, list 25→34 strings: FP16 refusals 223→306 while 2.5-bit compliances stay pinned at 422, so flips rise 192→263; at 4.5 bits flips go 54, 58, 71, 57 | `review_reanalysis.py` → `marker_decomposition` |
| 8 | Marker choice alone moves the estimate up to **1.38×** (Qwen2.5-3B) and **1.64×** (Phi-3.5-mini) | Four marker lists, identical completions | `data.json` → `marker_variants` |
| 9 | Perplexity alone cannot detect degeneration, and fails in the direction that matters | Qwen2.5-3B median NLL at 2.5 bits is 4.13, *lower* than 3.5 bits at 5.80. NLL-only flags 15.6% of 2.5-bit completions; composite flags 99.8% | `reanalyse_runs.py` |
| 10 | A frozen refusal probe retains **96–100%** across 8.5–4.5 bits on all three models | Difference-in-means direction fitted on one prompt half, scored on a disjoint half, 200 synchronized splits | `review_reanalysis.py` → `probe` |
| 11 | Capability collapses at a **model-specific** bit-width, one full bit apart | Qwen2.5-3B and Qwen2.5-1.5B destroyed at 3.5 bits (*p*<sub>Holm</sub> < 0.001 and 0.0003); Phi-3.5-mini indistinguishable from FP16 at 3.5 bits, collapses at 2.5 | `review_reanalysis.py` → `gsm8k` |
| 12 | The judge's three labels are separable by first-token argmax | Each label is two tokens under the Qwen2.5 tokenizer, but the first tokens are distinct: ids 38029 / 7682 / 75255. Asserted at load time | `classify_completions_judge.py` |

## Claims explicitly **not** made

| Not claimed | Why |
|---|---|
| "Harmful compliance stays below 2.2%" | The prompts carry no per-prompt harmfulness annotation. What is measured is a change in the model's own decision. A baseline refusal may be an over-refusal of a benign prompt, in which case the quantized model's compliance is a correction, not a failure |
| "Quantization does not reduce safety" | A non-significant test is not evidence of no effect. The defensible form is bounded: the transition rate is below 4.6% at every rung simultaneously |
| "Capability failure precedes fluency failure" | Qwen2.5-3B at 4.5 bits falls 18.5% → 11.5% with zero degenerate output, but paired exact McNemar gives *p* = 0.020 and *p*<sub>Holm</sub> = 0.100. Not significant at *n* = 200 |
| "κ = 1.15 is universal" / "a third model outside [0.75, 1.55] falsifies it" | Two models, and the coherent band was selected from observed degeneracy then held fixed across replicates. This is a confidence interval for these two, not a prediction interval for a new model |
| "The probe is flat across the whole drift band" | True over 8.5–4.5 bits on all three models. False for Phi-3.5-mini's full band, which reaches 3.5 bits where retention is already 63% |
| "Probe intervals are confidence intervals" | They are split-to-split dispersion over one fixed prompt set. A confidence interval would need probe refitting inside an outer prompt bootstrap |
| Anything about AWQ, GPTQ, GGUF k-quants, or mixed precision | One quantizer family was tested, deliberately, so that only bit-width varies |
| Anything about sampled decoding, long completions, or non-English prompts | Greedy, 48 tokens, English |
| That the bit-width selection rule in the paper's §9.2 improves deployments | It is stated as a testable design hypothesis. Every quantity in it is measured here; whether optimising against them beats the conventional rule is untested |

## Open work, in priority order

1. **Blinded human validation of the judge.** ~200–300 completions, stratified to
   oversample judge/phrase disagreements, the 4.5-bit rung, high-NLL outputs,
   non-English outputs, `unclear` verdicts, and completions opening with a
   disclaimer. Without this the paper compares two fallible estimators rather
   than establishing which is right. This is the single largest gap.
2. **A 256-token replication** at FP16, 5.5 and 4.5 bits. The 48-token budget
   structurally favours observing refusal intact; this is the cheapest test of
   whether the direction survives.
3. **Regrade Qwen2.5-1.5B with the 7B judge** so it joins the refusal arm. Its
   completions are stored; the blocker is that the run predates
   `completion_nll.json`, so the FP16 reference NLL must be recomputed on GPU.
4. **A second independent judge** from a different model family, with a confusion
   matrix against the first.
5. **A blinded audit of gate-removed completions**, to bound the
   outcome-dependent censoring rather than argue around it.
6. **A harmfulness-labelled suite** plus an over-refusal suite, which together
   would license the stronger safety vocabulary this paper avoids.
