# Validating the judge with a second, independent grader

Every behavioural number in the paper depends on one grader: Qwen2.5-7B-Instruct
loaded in 4-bit. That grader shares a model family with two of the three models
it judges, has never been checked against an external label, and is observably
wrong on some completions — it scored *"Stalin was the leader of the Soviet
Union from 1922 to 1953…"* as a refusal.

This is the single largest weakness in the work, and it is the cheapest one to
attack, because the completions are already stored. Grading is a pure function
of saved text, so a second opinion costs API calls and no GPU at all.

---

## What you need

One key from any of these. All are free at the volume required, and all are a
**different family** from Qwen and Phi, which is the point.

| Provider | Model | Free tier | Where |
|---|---|---|---|
| Groq | Llama-3.3-70B | ~1k requests/day, very fast | console.groq.com |
| NVIDIA NIM | Llama-3.3-70B, Mistral | free credits | build.nvidia.com |
| Google | Gemini 2.0 Flash | 1,500 requests/day | aistudio.google.com |
| OpenRouter | many families, one key | paid, but cents | openrouter.ai |

Copy `.env.example` to `.env` and paste the key in. **`.env` is gitignored and
must never be committed.** Do not paste keys into a chat or an issue; a key in a
transcript is a key that has leaked.

```bash
cp .env.example .env      # then edit
```

## Running it

```bash
# Smoke test: 20 completions, confirms the key and the parsing work
python scripts/judge_via_api.py --provider groq --limit 20

# The sweep the claim rests on: FP16 and 4.5 bits, both models
python scripts/judge_via_api.py --provider groq

# Widen it if quota allows
python scripts/judge_via_api.py --provider groq --schemes FP16 RTN_5B RTN_4B RTN_3B

# Then compare
python scripts/compare_judges.py
```

The sweep is **resumable**: every verdict is cached per completion, so an
interrupted run or an exhausted daily quota costs nothing. Re-run the same
command tomorrow and it continues.

## Why the default sweep is only two rungs

Grading all 8,000 stored completions is neither necessary nor free-tier
friendly. The paper's claim is a *paired* comparison, and the load-bearing
comparison is FP16 against 4.5 bits — the rung where both models show a
significant shift and where the degeneracy gate removes nothing. That is
500 × 2 × 2 = 2,000 calls, which fits in one day on any of the tiers above.

Widening to 5.5 and 3.5 bits strengthens the drift estimate. The 2.5-bit rung is
not worth grading: the composite gate removes 99.8% of it, so there is almost
nothing for a judge to score.

## What comes back

`scripts/compare_judges.py` writes `docs/paper/judge_agreement.json` and prints:

- **Agreement and a confusion matrix.** Not a single percentage — the shape of
  the disagreement is the informative part. If the second judge systematically
  reads the 7B judge's `REFUSE` as `COMPLY`, that is the capability-disclaimer
  problem showing up as a number rather than as an anecdote.
- **Per-class recall**, so "agrees 85%" cannot hide "agrees on compliance,
  disagrees on everything else".
- **A reproduction of the headline** using only the second judge's labels:
  the paired transition counts and an exact McNemar test. This is the decisive
  output. If the direction still runs toward refusal and survives correction,
  the finding is robust to the grader. If it does not, the paper's central claim
  was a property of one 4-bit Qwen model and must be withdrawn.
- **Disagreement examples**, saved for inspection, which is what a reviewer will
  ask to see.

## What this does and does not fix

It answers *"same family"*, *"quantized judge"*, *"no independent check"*, and
*"we only have your word that the class is applied correctly"*.

It does **not** replace blinded human labels. Two models agreeing can be two
models sharing a bias — both were trained on overlapping data with similar
instruction-tuning conventions. Agreement raises confidence; it does not
establish correctness. The honest reporting is agreement plus the disagreement
set, not a claim of validation.
