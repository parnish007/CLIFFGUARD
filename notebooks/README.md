# notebooks/

## Run this one

**[`colab_run.ipynb`](colab_run.ipynb)** — the only notebook you need.

1. Open in Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all`.
4. When it finishes, a zip downloads. Unzip it at the repository root.

Nothing else is required. No HuggingFace token, no editing, no manual paths.
Every model and dataset it touches is public. Expect 60–90 minutes on a T4.

> **This is a reduced replication, not the paper's configuration.** Its defaults
> are 200 behavioural prompts, 150 GSM8K questions and five rungs
> (8, 5, 4, 3, 2 code bits). The paper reports 500 prompts, 200 questions and
> seven rungs (8–2). Raise `N_PROMPTS`, `N_GSM8K` and `BITS` in the config cell
> to match the paper; that costs roughly 2.5 hours on a T4 rather than 60–90
> minutes. The reduced defaults exist so a first run finishes inside one Colab
> session, not because the paper's numbers came from them.
Each stage checkpoints, so a disconnect costs at most one stage — reconnect and
`Run all` again.

The first thing it does after installing dependencies is a **preflight**: a
CPU-only self-test of every module and entry point the run will use. If that
prints `PREFLIGHT OK`, nothing downstream can fail on an import or signature
mismatch. If it fails, stop and report it rather than letting an hour of GPU time
produce nothing.

---

## Why a hosted GPU at all

The full measurement pipeline runs on a 6 GB laptop card. Three things do not,
and they are the entire contents of the notebook.

| Arm | What it adds | Why not local |
|---|---|---|
| **A** | a **7 B judge** re-grading saved completions | this is the project's missing instrument — see below |
| **B** | `Qwen/Qwen2.5-3B-Instruct` | 6.2 GB fp16 against 5.7 GB free locally |
| **C** | `microsoft/Phi-3.5-mini-instruct` | 7.6 GB fp16 |

**Arm A is the one that matters.** A refusal measurement is only as good as its
grader, and two cheap graders are demonstrably unusable here:

- a refusal **phrase list** — the reported flip rate moves by up to 1.64x with
  the choice of strings alone, and is not even monotone in that choice;
- a 1.5 B **self-judge** — saturates at 100 % REFUSE, including on plainly
  helpful answers.

A 7 B judge is the smallest instrument with a plausible chance of doing better.
It is loaded in NF4 so it fits a T4, and that is recorded in the output manifest.
The script reports it *alongside* all four phrase-list variants and prints every
disagreement, so if this judge also fails it will be visible rather than quietly
adopted.

Arms B and C ask whether anything measured on one 1.5 B checkpoint is a property
of quantization or of that checkpoint.

---

## Running the same measurements locally

Everything the notebook does is a repo script. On a 6 GB card, with the default
1.5 B model:

```bash
python scripts/run_local_ladder.py       --n 250   # weights, eta, probe ladder
python scripts/run_behavioural_ladder.py --n 250   # generations + classification
python scripts/run_sector_ladder.py      --n 200   # GSM8K, gold-labelled
python scripts/analyse_probe_transfer.py           # frozen vs refit estimands
python scripts/analyse_dprime_power.py             # minimum detectable effects
```

Every runner takes `--model`, and `--layer` defaults to mid-depth resolved from
the model's own config, so any checkpoint works. Add `--smoke --n 24` for a fast
wiring check.

Results land in `artifacts/runs/<utc>_<git-sha>_<label>/` with a provenance
manifest, and one line is appended to `artifacts/runs/INDEX.md`.

---

## What else is in here

| File | Purpose |
|---|---|
| [`colab_run.ipynb`](colab_run.ipynb) | **Current.** The hosted-GPU runner described above. |
| [`colab_labelled.ipynb`](colab_labelled.ipynb) | **Current.** The labelled suites and the second annotation axis — see below. |
| [`colab_round2.ipynb`](colab_round2.ipynb) | **Current.** Round 2: the 256-token replication, the deployed quantizers, the 1.5B regrade. |
| [`stage0_noise_floor_and_isotropy.ipynb`](stage0_noise_floor_and_isotropy.ipynb) | Stage 0 in isolation — the rotation-replication gate and isotropy test. Diagnostic, not part of the main run. |
| [`cliffguard_colab.ipynb`](cliffguard_colab.ipynb) | Superseded notebook for an earlier design. Kept locally for provenance; its labels come from a corpus partition that agrees with model behaviour only 52.4 % of the time. Not part of the release. |
| [`colab_helper.py`](colab_helper.py) | Helpers for that superseded notebook. |

---

## `colab_labelled.ipynb` — both annotation axes

The other notebooks measure a **change in the model's own decision**, which
cannot say whether the change was good: a new refusal is either the system
working or the system becoming useless, and an unlabelled corpus reports both as
one number. This one separates them, with two independent annotations.

| axis | labels from | question |
|---|---|---|
| prompt: harmful / benign | HarmBench, AdvBench, StrongREJECT, XSTest, OR-Bench — 3,457 prompts, external to this project | should the model have helped? |
| completion: refusal / compliance / deflection / disclaimer / unclear | a 7B judge, five-way, first-token argmax | what did it actually do? |

Crossed, `harmful + compliance` is a safety failure and `benign + refusal` is an
over-refusal — opposite regressions, never summed. Splitting the completion axis
also fixes a conflation the paper admits: its three-way grader counts declining,
deflecting, redirecting *and* warning all as `REFUSE`.

The prompt axis is externally supplied; the completion axis is still a model's
opinion. That asymmetry is stated in the notebook rather than smoothed over.

**It resumes.** Every stage is a step in a journal on Drive
(`scripts/colab_pipeline.py`). A step that finished is skipped; a step whose
arguments changed is re-run and says why; a step whose output is gone is re-run;
inside a step the ladder and graders cache per scheme; and near the end of a
session the pipeline declines to *start* a long step rather than be killed inside
one. If the runtime dies, reconnect and re-run the pipeline cell.

Run directories go to Drive via `CLIFFGUARD_ARTIFACTS`, so nothing durable lives
in `/content`.

---

## Why the notebook holds no measurement logic

Every arm shells out to `scripts/`. There is no analysis code in the notebook, so
there is no second implementation to drift out of sync with the repository. This
project has already paid for that failure mode once: a notebook and the modules
it called diverged silently, and the notebook could not have produced a result on
any clone — it referenced a filename that did not exist, under a directory that
was gitignored.
