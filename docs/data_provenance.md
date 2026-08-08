<div align="center">

[← Docs index](README.md) &nbsp;|&nbsp;
[Methodology](methodology.md) &nbsp;|&nbsp;
[Claims ledger](claims_and_evidence.md) &nbsp;|&nbsp;
**Data provenance**

</div>

# Where every number came from

Three experiments, run months apart under different protocols, now appear in one
manuscript. This page exists so that no reader — including us, later — has to
guess which run produced which claim, or silently compare a number from one
round against a number from another that does not mean the same thing.

**The rule this page enforces:** a quantity may only be compared with another
quantity from the same round. Rounds differ in corpus, prompt count, grader,
class count and bit labelling, and every one of those differences moves numbers.

---

## 1. The three rounds at a glance

| | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| **When** | 2026-08-03 | 2026-08-03/04 | **2026-08-08** |
| **Models** | Qwen2.5-1.5B, Qwen2.5-3B, Phi-3.5-mini | same | Qwen2.5-3B, Phi-3.5-mini, **SmolLM2-1.7B** |
| **Families** | 2 (Qwen, Phi) | 2 | **3** (Qwen, Phi, SmolLM2) |
| **Corpus** | unlabelled paired prompts | same | **XSTest, externally labelled** |
| **Prompts** | 500 | 500 | 300 (150 harmful + 150 benign) |
| **Quantizers** | RTN 8.5→2.5 stored bits | RTN only (AWQ/GPTQ were planned, never run) | RTN 8.5→2.5 stored bits (`--bits 8…2` code bits) |
| **Grader** | Qwen2.5-7B, **3-way** | + 3 more graders | Qwen2.5-7B NF4, **5-way** |
| **Cells** | 14 model×rung | — | **21 model×rung** |
| **Budget** | 48 tokens | 48 (re-grading round 1's completions; the planned 256-token step never ran) | 48 tokens |
| **Feeds** | `review_stats.json`, `data.json` | `judge_agreement.json` | `labelled_paper_stats.json` |
| **Paper** | §Refusal, §Artifact | §Judge agreement | §[`sec:labelled`](#4-round-3) |

### The differences that make rounds non-comparable

- **Bit labelling.** The runner names schemes by *code* bits (`RTN_4B`); every
  axis and table in the paper reports *stored* bits, which adds 0.5 for the
  group scale and zero point at group 64. So **`RTN_4B` = 4.5 stored bits**, and
  the round-3 figures label their axis 8.5…2.5 to match rounds 1–2. A "4.5-bit"
  row and a `--bits 4` invocation are the same quantizer.
- **"Refusal" is not one class.** Round 1's 3-way grader folds refusing,
  deflecting, redirecting and warning into `REFUSE`. Round 3 splits them.
  Round 3's `refusal` is a *subset* of round 1's. Its comparable quantity is
  `refusal + deflection + disclaimer`, which is what this repo calls *declining*.
- **Denominators.** 500 vs 300 prompts, and round 3's are split 150/150 by an
  external label rounds 1–2 did not have.
- **Multiplicity families.** Round 1 corrects across 14 cells (`p_14`); round 3
  across 7 rungs within a model and 21 cells overall. A `p_14` and a `p_21`
  are not interchangeable.

---

## 2. Round 1 — the refusal ladder

**Run directories**

| run | model | n | tokens |
|---|---|---:|---:|
| `20260803-143410_e0eb33f_behavioural-ladder-qwen1.5b` | Qwen2.5-1.5B | 500 | 48 |
| `20260803-165047_0d3f097_colab-behavioural-qwen3b` | Qwen2.5-3B | 500 | 48 |
| `20260803-174622_0d3f097_colab-behavioural-phi35` | Phi-3.5-mini | 500 | 48 |
| `20260803-150456_281e256_sector-ladder-gsm8k` | Qwen2.5-1.5B | — | 192 |
| `20260803-171823_0d3f097_colab-gsm8k-qwen3b` | Qwen2.5-3B | — | 192 |
| `20260803-181927_0d3f097_colab-gsm8k-phi35` | Phi-3.5-mini | — | 192 |

**How collected.** `scripts/run_behavioural_ladder.py` generates greedy 48-token
completions at each rung, scores every completion's NLL under the FP16 reference,
applies the composite degeneracy gate, then classifies. GSM8K runs use
`run_sector_ladder.py` at a 192-token budget because arithmetic needs room.

**Claims it supports:** the drift coefficient (1.15 pp/bit, CI [0.75, 1.55]),
the ≤2.2% transition rate and its 4.62% simultaneous bound, the phrase-list
38.4%, the gate-vs-grader decomposition, the probe retention, the capability
thresholds.

**Consolidated into** `docs/paper/review_stats.json` and `docs/paper/data.json`
by `scripts/review_reanalysis.py` and `scripts/build_paper_data.py`.

---

## 3. Round 2 — more quantizers, more graders

Re-grades round 1's identical stored completions with three graders from two
other families. Its notebook also *plans* deployed checkpoints (AWQ, GPTQ) and a
256-token generation step; **neither was executed** — no manifest carries those
schemes or that budget (§6).

**Claims it supports:** that the *direction* of the refusal shift agrees under
**three of the four** grader comparisons that clear the 400-completion floor,
with the fourth (Llama-3.3-70B on Qwen2.5-3B, 1 toward refusal against 2 toward
compliance) running the other way; and that the *significance* replicates never.
Pairwise judge agreement is 64.9–78.4%. This is why the paper withdraws the
significance claim — and why it must not say the graders were unanimous, which
three passages once did.

**Consolidated into** `docs/paper/judge_agreement.json`.

> Round 2 re-grades round 1's stored completions. It produced no new generation
> run, which is why no separate run directory carries its name. Every completion
> it scored is in the round-1 directories above.

---

## 4. Round 3 — the labelled suites

**Run directories** — all three produced by `notebooks/colab_labelled.ipynb`,
run unmodified on a free Colab T4 on 2026-08-08.

| run | model | n | tokens | schemes |
|---|---|---:|---:|---:|
| `20260808-154229_069d48d_lab-qwen3b-xstest` | Qwen2.5-3B | 300 | 48 | 8 |
| `20260808-162254_069d48d_lab-phi35-xstest` | Phi-3.5-mini | 300 | 48 | 8 |
| `20260808-164828_069d48d_lab-smol17-xstest` | SmolLM2-1.7B | 300 | 48 | 8 |

**How collected.**

1. `scripts/download_eval_suites.py --download --suites xstest` fetches XSTest
   and writes `data/eval_suites/xstest.jsonl` with a recorded sha256.
2. `scripts/run_behavioural_ladder.py --prompts data/eval_suites/xstest.jsonl
   --n 150 --bits 8 7 6 5 4 3 2 --max-new-tokens 48 --no-activations` per model.
   The corpus fingerprint (`6dcef269` over 300 ordered prompts) is written into
   every cache key, so a run cannot silently reuse a different corpus's cache.
3. `scripts/classify_completion_taxonomy.py <run> --judge-model
   Qwen/Qwen2.5-7B-Instruct --judge-4bit --emit-three-way` grades all eight
   schemes on five classes.

**Integrity checks performed on the returned archive**

| check | result |
|---|---|
| archive vs extracted copy | 106/106 files byte-identical |
| archive vs installed copy | 105/105 identical; `INDEX.md` deliberately merged, nothing lost from either side |
| prompt file across the three runs | **byte-identical**, sha256 `1652240a…` |
| local re-analysis vs Colab's own `matrix_stats.json` | **0 differences** across all three blocks |
| notebook run vs committed notebook | identical, cell by cell |

That last pair is the important one: the analysis reproduces on a different
machine, from the stored completions, with no GPU.

**Claims it supports in the paper** (deliberately few — see §5):

- phrase-list coverage against the judge: 49.3% / 23.5% / 3.6% across the three
  families, at precision 1.000 in every case (the two label sets nest)
- the judge's compliance class is empty on harmful prompts in all 21 cells
- 92.7% / 100.0% / 96.3% of full-precision completions reach the token cap,
  measured under each model's own tokenizer, and never below 85% down to
  4.5 stored bits
- the three Clopper–Pearson bounds: 1.98% / 3.24% / 3.95%
- the label-tokenization asymmetry (`DISCLAIM` withdrawn as uninterpretable)

**Consolidated into** `docs/paper/labelled_paper_stats.json` by
`scripts/build_labelled_tables.py`, which records the run id, model, judge,
budget and scheme count beside every number.

---

## 5. What round 3 measured that is deliberately NOT in the paper

Measuring something does not entitle it to a section. Each of these is real,
reproducible and stays in `docs/results_labelled.md` (local) instead:

| measurement | why it is not in the manuscript |
|---|---|
| Full 5-way × prompt-class transition matrix per rung | descriptive; no claim rests on it, and it would add three pages |
| Benign-arm usefulness loss, significant at 3 and 2 bits | it is a **capability** collapse, and the paper already has a capability section measuring the same thing on GSM8K with a cleaner endpoint |
| The exploratory leakage queue (SmolLM2 72/150) | the detector is an **in-sample regex** against these very completions. It flags an unadjudicated population; it does not measure harm. Publishing a number that looks like a harm rate would be the exact error this paper is about |
| `safety_lost = 0` as a result | the endpoint saturated. It is reported as an instrument property, never as a safety finding |

The leakage queue is the omission most worth arguing about, because leaving it
out makes the safety null look tidier than it is. The paper handles that by
stating the null as a property of the endpoint rather than of the models, and by
naming the token budget as the cause — which is the honest version of the same
warning, without dressing a regex as a measurement.

---

## 6. What has NOT been run, and what the paper says about it

Checked by scanning every `manifest.json` for the scheme and budget in question,
not by memory.

| Planned | State | Does the paper claim it? |
|---|---|---|
| **256-token generation** (`colab_round2.ipynb`, `LONG_TOKENS = 256`) | **never executed** — no run directory has `max_new_tokens=256` | **No.** The paper calls longer-budget survival untested and lists it as the first item of future work |
| AWQ / GPTQ deployed checkpoints | never executed — no manifest carries those schemes | **No.** Explicitly disclaimed in Limitations §Scope |
| Human adjudication of completion labels | never done | **No.** Named as the largest gap in Limitations |
| Full 5-way transition matrix in the manuscript | computed, deliberately not published | **No.** Kept in `results_labelled.md` |

### Notebook execution state

| Notebook | State |
|---|---|
| `colab_labelled.ipynb` | committed as a clean template; **executed** — its output is `colab_labelled-ranned.ipynb` and the three round-3 runs |
| `colab_run.ipynb` | template; **executed** — output preserved as `colab_run_EXECUTED.ipynb` |
| `colab_round2.ipynb` | template; **partly executed.** The grader sweeps ran (they produced `judge_agreement.json`); the 256-token step did **not** |
| `cliffguard_colab.ipynb` | superseded design, kept for provenance |
| `nepali_safety_phase_b.ipynb`, `stage0_noise_floor_and_isotropy.ipynb` | never run; feed no claim |

**Nothing needs to be run for the paper as written.** Every claim in it traces to
a present run directory. The 256-token experiment is the one worth running next,
and it would *test* a claim rather than *supply* one — §7 below.

### On "we tested low tokens then high"

Worth stating plainly, because it is easy to misremember. The refusal arm has
only ever run at **48 tokens**. The 192-token runs are GSM8K, a different task
with a different endpoint, so they are not a budget comparison. No
longer-budget refusal measurement exists in this project.

The within-sample check that *is* available points the other way and is
confounded: completions that stop before the cap are graded `COMPLY` 0.5% of the
time against 7.7% for those that reach it. That is not evidence truncation is
harmless — it is the expected consequence of early-stopping completions being
mostly short refusals. It is recorded here so nobody re-derives it and mistakes
it for a result.

## 7. Tracing any number back

```bash
# every claim -> its measurement file
python scripts/check_paper_numbers.py       # fails if prose disagrees

# rebuild each measurement file from the runs
python scripts/build_paper_data.py          # -> data.json
python scripts/review_reanalysis.py         # -> review_stats.json
python scripts/build_labelled_tables.py     # -> labelled_paper_stats.json
python scripts/build_labelled_figures.py    # -> figures/fig_labelled_*
```

`labelled_paper_stats.json` carries `run`, `model_id`, `judge_model`,
`judge_loaded_in_4bit`, `max_new_tokens` and `n_prompts` for each model, so any
round-3 number can be traced to a directory without consulting this page.

## 8. Runs present but used by nothing

Fifteen directories under `artifacts/runs/` feed no claim: smoke tests, local
development ladders, and the GGUF and salience experiments. They are kept
because they are cheap and they document what was tried. **Nothing in the paper
depends on them**, and a reader should not infer otherwise from their presence.
