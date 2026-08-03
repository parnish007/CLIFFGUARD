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
| [`stage0_noise_floor_and_isotropy.ipynb`](stage0_noise_floor_and_isotropy.ipynb) | Stage 0 in isolation — the rotation-replication gate and isotropy test. Diagnostic, not part of the main run. |
| [`cliffguard_colab.ipynb`](cliffguard_colab.ipynb) | Superseded notebook for an earlier design. Kept locally for provenance; its labels come from a corpus partition that agrees with model behaviour only 52.4 % of the time. Not part of the release. |
| [`colab_helper.py`](colab_helper.py) | Helpers for that superseded notebook. |

---

## Why the notebook holds no measurement logic

Every arm shells out to `scripts/`. There is no analysis code in the notebook, so
there is no second implementation to drift out of sync with the repository. This
project has already paid for that failure mode once: a notebook and the modules
it called diverged silently, and the notebook could not have produced a result on
any clone — it referenced a filename that did not exist, under a directory that
was gitignored.
