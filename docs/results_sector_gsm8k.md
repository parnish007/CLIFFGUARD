# Sector arm: GSM8K reasoning under the RTN ladder

Run: `artifacts/runs/20260803-150456_281e256_sector-ladder-gsm8k`
Command: `python scripts/run_sector_ladder.py --n 200 --bits 8 7 6 5 4 3 2`
Hardware: RTX 3050 6 GB. ~20 min.

**This is the only measurement in the project that no reviewer has been able to
break.** It is worth understanding why before reading the numbers.

---

## 1. Why this arm is trustworthy when the refusal arm was not

Two rounds of adversarial review destroyed the refusal results
([`corrections_2026-08-03.md`](corrections_2026-08-03.md)). Both failures were
**labelling** failures, and neither can occur here:

| Failure mode in the refusal arm | Why GSM8K is immune |
|---|---|
| Corpus labels came from whether hh-rlhf's *rejected response* looked like a refusal — 52.4 % agreement with the model, i.e. chance | Labels are **gold arithmetic answers** shipped with the dataset |
| "Compliance" was decided by an English phrase list; the rate moved 10.4 %–48.0 % across marker variants | Correctness is `|predicted − gold| ≤ 1e-4`. No phrase list, no judge, no language dependence |
| A Chinese-language refusal was scored as harmful compliance | A correct answer is correct in any language |

The scoring rule is deliberately permissive about *format* — the last number in
the completion is taken, chain-of-thought is allowed and unconstrained. A model
that loses instruction-following would otherwise be scored as losing arithmetic,
which confounds two capabilities. Degenerate completions count as **incorrect**
rather than being excluded; dropping them would flatter the low-precision rungs
by removing their worst cases.

---

## 2. Results

| bits/param | correct / 200 | accuracy | 95 % CI | p vs FP16 | degenerate | median NLL |
|---|---|---|---|---|---|---|
| 16 (FP16) | 47 | 0.235 | [0.178, 0.300] | — | 0 % | 0.72 |
| 8.50 | 47 | 0.235 | [0.178, 0.300] | 1.00 | 0 % | 0.72 |
| 7.50 | 46 | 0.230 | [0.174, 0.295] | 0.93 | 0 % | 0.71 |
| 6.50 | 48 | 0.240 | [0.183, 0.305] | 0.87 | 0 % | 0.71 |
| 5.50 | 57 | 0.285 | [0.224, 0.353] | 0.096 | 0 % | 0.72 |
| 4.50 | 50 | 0.250 | [0.192, 0.316] | 0.62 | 0 % | 0.75 |
| **3.50** | **18** | **0.090** | **[0.054, 0.139]** | **1.6 × 10⁻⁷** | **0 %** | 1.00 |
| 2.50 | 0 | 0.000 | [0.000, 0.018] | 8.7 × 10⁻²⁴ | 100 % | 9.43 |

Clopper–Pearson intervals; two-sided exact binomial test against the FP16 rate.

### The cliff

Accuracy is **flat from 16 bits down to 4.5** — every p ≥ 0.62, and the FP16
interval contains every point. Between 4.5 and 3.5 bits/param it falls to
**38 % of full precision**, and the intervals no longer overlap.

**The collapse is not degeneracy.** At 3.5 bits the model produces **0 %
degenerate output**; its median NLL under the FP16 reference is 1.00 against
FP16's 0.72 — slightly less fluent, still language. It writes plausible
arithmetic and gets it wrong. Only at 2.5 bits does fluency itself fail (100 %
degenerate, NLL 9.43).

That separation matters: **capability collapse precedes fluency collapse by a
full rung.** A deployment check that only screens for incoherent output would
pass a 3.5-bit model that has lost two thirds of its arithmetic.

### The 5.5-bit bump

0.285 against FP16's 0.235, p = 0.096. Not significant at n=200, and I do not
have an explanation. It is reported rather than smoothed. Consistent with
quantization acting as mild regularisation on a low-accuracy task, but that is
speculation and the effect would need replication across seeds and models before
it is worth a sentence in a paper.

---

## 3. Absolute accuracy is low, and that bounds the claim

FP16 solves 23.5 % of GSM8K. Qwen2.5-1.5B-Instruct is reported around 60–70 %
with few-shot prompting and generous generation budgets. This run is zero-shot
with 192 new tokens, so many completions are truncated mid-derivation.

Consequences:

- The **relative** cliff is what is measured, and it is unambiguous.
- The absolute rate is a floor, not the model's capability.
- A harder ceiling would give more dynamic range and a sharper location for the
  collapse point. Worth re-running with 512 tokens and 8-shot prompting.

---

## 4. What this establishes

1. **A capability cliff exists at a specific bit-width**, between 4.5 and
   3.5 bits/param, measured against gold labels with p = 1.6 × 10⁻⁷.
2. **It is not a fluency artifact** — zero degenerate output at the collapse rung.
3. **Capability collapse and fluency collapse are separate events** one rung
   apart, and screening for the second misses the first.

## 5. What it does not establish

- **Two sectors is not a law.** Corollary 2.1 predicts different behaviours break
  at different, predictable bit-widths. Reasoning collapses at 3.5. Where refusal
  collapses is **not measured**, because no validated safety instrument exists in
  this repository — see [`corrections_2026-08-03.md`](corrections_2026-08-03.md)
  §R2.7. Until it does, the ordering claim has one endpoint.
- **One model, one dataset, one decoding setting**, greedy only.
- **The 3.5-bit rung is one point.** The collapse is somewhere in
  (3.5, 4.5]; locating it needs intermediate group sizes, since bits/param moves
  in steps of 1 on this ladder.

---

## 6. Next, in order

1. Re-run at 512 tokens and 8-shot to lift the FP16 ceiling and sharpen the
   collapse location.
2. Interpolate the rung spacing via group size (`--group 32/128`) to place the
   collapse point to better than one bit.
3. A second sector with gold labels — HumanEval pass@1 needs a sandbox; MMLU is
   cheaper and is multiple-choice, so it is also phrase-list-free.
4. Only then compare collapse points across sectors, which is the actual
   Corollary 2.1 test.
