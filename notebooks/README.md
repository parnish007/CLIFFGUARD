# notebooks/

Every notebook here shells out to `scripts/`. None of them contains measurement
logic, so there is no second implementation to drift out of sync with the
repository — a failure this project has already paid for once, when a notebook
and the modules it called diverged silently and the notebook could not have
produced a result on any clone.

Each is pinned to a specific commit (`REPO_COMMIT`), so a session runs the code
the notebook was verified against rather than whatever `main` happens to be that
morning.

---

## What to run now

Two rounds are built, verified, and waiting on GPU time. **Run them in order** —
round 4 audits and repairs the instrument that round 5's measurements are read
with.

### 1. [`colab_round4.ipynb`](colab_round4.ipynb) — the instrument

T4, `Run all`, about **2 h 15 m**. Generates nothing: every completion it reads
already ships with the repository.

- **Step 1, the option-order audit** (~35 min). The corrected label scorer reads
  a verdict from single-token letters A–E. That fixes the tokenization
  asymmetry, and it introduces a symbol preference nobody has tested: a judge
  that leans toward option A, whatever A says, would pass every check the
  manuscript makes. This re-grades the same text with the descriptions held word
  for word and only the letters permuted, then maps back to classes. It prints a
  verdict and carries it into the archive.
- **Step 2, the full re-grade** (~1 h 40 m). 15,200 judge pairs, putting every
  rung on the corrected scorer. Until this runs, every ladder-wide quantity in
  the paper — the drift slope, the 14-cell transition table, the simultaneous
  bound, all 21 labelled cells — is marked with a dagger and belongs to an
  instrument the paper itself shows was defective.

The audit runs **first, deliberately**: step 2 spends an hour and forty minutes
measuring with the scorer step 1 interrogates. It does not block — a ladder
graded under one fixed assignment is a coherent measurement whatever the audit
says. What the audit changes is what may be claimed about it.

### 2. [`colab_round5.ipynb`](colab_round5.ipynb) — external validity

T4 (an L4 or A100 unlocks one extra step), `Run all`, about **3 h 30 m** across
five independently resumable steps, in descending order of how much they matter.
Run against a protocol frozen in advance:
[`docs/preregistration_round5.md`](../docs/preregistration_round5.md).

| step | question | ~time |
|---|---|---|
| 1 | Does the estimator gap survive a quantizer people deploy? AWQ and GPTQ-Int4 | 30 m |
| 2 | Is the 48-token window an artifact at the quantized rung, not only at FP16? | 55 m |
| 3 | Is greedy nondeterminism a batch-size effect? One model, one corpus, two batch sizes | 12 m |
| 4 | Do the transition counts hold under sampled decoding across seeds? | 55 m |
| 5 | Does any of it hold at 7B? Skips itself, loudly, on a runtime that cannot hold an fp16 baseline | 60 m |

The preregistration states two confirmatory hypotheses with thresholds a null
result can fail, and two exploratory ones that are labelled as such and may not
be reported as replications. `scripts/analyse_deployed.py` scores them as
CONFIRMED or REFUTED by arithmetic, so the comparison between prediction and
outcome is not made by whoever writes the manuscript afterwards. The notebook
prints the protocol's SHA-256, refuses to run if the working copy differs from
the committed one, and stores the hash in the archive.

Steps 4 and 5 are the ones to drop if time runs short.

---

## Two things every session does

**Restore before validate.** `data/` is gitignored, so corpora come from Drive
or are rebuilt; Fold A's rebuild uses an unpinned HuggingFace revision, which is
why preflight hashes it immediately afterwards. A restored archive is unpacked
**only under `artifacts/runs/`** and anything else in it is refused by name —
see `tests/test_notebook_restore.py` for what that stops and, equally, for what
it does not (`extractall` does not escape the working directory; what it does
allow is a member quietly replacing a grader or installing a git hook).

**Preflight.** A CPU-only self-test of every entry point and flag the session
will use, run before any GPU time is spent. If a script was renamed or a flag
disappeared, it fails in seconds rather than two hours in — and a missing
`--letter-order` is the failure that would look most like success, since the
grader would re-derive the canonical cache under a name promising otherwise.

---

## What ran before

Kept as templates, not as instructions. Each was pinned to its own commit and
its results are already in the paper.

| Notebook | What it produced | State |
|---|---|---|
| [`colab_run.ipynb`](colab_run.ipynb) | The behavioural ladder, GSM8K and the 7B judge on Qwen2.5-3B and Phi-3.5-mini | Executed. Its reduced defaults (200 prompts, 5 rungs) are **not** the paper's configuration — see the note below |
| [`colab_labelled.ipynb`](colab_labelled.ipynb) | XSTest under both annotation axes, three model families, five-way grader | Executed |
| [`colab_round2.ipynb`](colab_round2.ipynb) | The 256-token replication and the 1.5B re-grade | Executed **in part**. Its deployed-quantizer arm never ran, which is why round 5 carries it |
| [`colab_round3.ipynb`](colab_round3.ipynb) | The scorer correction, 48-token prefix re-judging, the paired cross-budget test | Executed. Found the tokenization defect and withdrew one headline result |
| [`stage0_noise_floor_and_isotropy.ipynb`](stage0_noise_floor_and_isotropy.ipynb) | The rotation-replication gate and isotropy test | Diagnostic; not part of any main run |
| [`cliffguard_colab.ipynb`](cliffguard_colab.ipynb), [`colab_helper.py`](colab_helper.py) | An earlier design | **Superseded.** Its labels come from a corpus partition agreeing with model behaviour 52.4% of the time. Kept for provenance, not part of the release |

> **`colab_run.ipynb` is a reduced replication.** Its defaults are 200
> behavioural prompts, 150 GSM8K questions and five rungs; the paper reports 500
> prompts, 200 questions and seven. Raise `N_PROMPTS`, `N_GSM8K` and `BITS` to
> match, at roughly 2.5 hours rather than 60–90 minutes. The reduced defaults
> exist so a first run finishes inside one Colab session, not because the
> paper's numbers came from them.

Executed copies carry their outputs, which are results, so they are gitignored
rather than committed; the tracked files are clean templates.
`colab_run_EXECUTED.ipynb` predates that rule and stays tracked rather than
being rewritten out of history.

---

## Running the same measurements locally

Everything a notebook does is a repo script. On a 6 GB card, with the default
1.5B model:

```bash
python scripts/run_local_ladder.py       --n 250   # weights, eta, probe ladder
python scripts/run_behavioural_ladder.py --n 250   # generations + classification
python scripts/run_sector_ladder.py      --n 200   # GSM8K, gold-labelled
python scripts/analyse_probe_transfer.py           # frozen vs refit estimands
python scripts/analyse_dprime_power.py             # minimum detectable effects
```

Every runner takes `--model`, and `--layer` defaults to mid-depth resolved from
the model's own config, so any checkpoint works. `--smoke --n 24` is a fast
wiring check. Results land in `artifacts/runs/<utc>_<git-sha>_<label>/` with a
provenance manifest, and one line is appended to `artifacts/runs/INDEX.md`.

What needs a hosted GPU is only this: a **7B judge** (15.2 GB fp16, 4.5 GB in
NF4), **Qwen2.5-3B** (6.2 GB) and **Phi-3.5-mini** (7.6 GB). The judge is the
one that matters — a refusal measurement is only as good as its grader, and the
two cheap alternatives are demonstrably unusable here. A phrase list moves its
own answer by up to 1.64× on the choice of strings and is not monotone in that
choice; a 1.5B self-judge saturates at 100% REFUSE, including on plainly helpful
answers.

---

## What no notebook here does

**Adjudicate whether the labels are right.** Every grader in this project is
checked against another grader. `scripts/build_human_sample.py` draws the two
blinded sheets that would change that — one asking what a response *did*, on the
grader's own classes, and one asking whether it provided substantively harmful
assistance, which is a different question and the one no automated instrument
here can answer. Both ship unlabelled. Until they are labelled, this project
demonstrates instrument *disagreement*, not instrument *accuracy*.
