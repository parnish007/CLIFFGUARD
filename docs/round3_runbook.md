# Round 3: what to run, and where every input comes from

One Colab session, unattended, on a free T4. This is the operator's page: what
arrives from where, what to do before starting, and what to do with the result.

## Where each input comes from

| Input | Source | Why not somewhere else |
|---|---|---|
| Code (`scripts/`, `cliffguard/`) | Cloned from GitHub at a **pinned commit** | Following `main` would run whatever it happens to be that morning, not the code this notebook was verified against |
| Fold A corpus (`data/folds/fold_a/`) | Drive, else rebuilt from HuggingFace | `data/` is gitignored. The rebuild uses an **unpinned** dataset revision, so preflight hashes it immediately — see below |
| XSTest (`data/eval_suites/xstest.jsonl`) | Drive, else fetched from a stable URL | Same reason. Fetched from a fixed file rather than a mutable dataset, so it is the safer of the two |
| Published runs (`artifacts/runs/…`) | `MyDrive/cliffguard/prior_runs.zip` | Not in the repository. See "The one upload" |
| Model weights | HuggingFace, cached to local disk | ~32 GB; never checked in |
| Everything the run produces | Written to Drive after each step | Colab wipes local disk on disconnect |

## The one upload

Step 3 re-grades the five runs behind the published numbers under the corrected
label scorer. Those run directories are not in the repository, and they are not
in the repository on purpose: `artifacts/` is gitignored, and the completions
include model responses to harmful prompts — around 2,000 of which decline and
then begin to turn toward answering before the 48-token budget cuts them off.
None are complete instructions. Publishing them to a public, permanently-cached
repository is still a decision to take deliberately, not a side effect of
wanting a file to be reachable from Colab.

So instead:

```
python scripts/bundle_prior_runs.py          # writes prior_runs.zip, ~0.4 MB
```

Upload it to `MyDrive/cliffguard/prior_runs.zip`. It carries only what the
re-grade reads — prompts, completions, NLL and manifests, for full precision
and the 4.5-bit rung — and omits the activation tensors, which are 63–95 MB per
run and which nothing in round 3 opens.

Without it, everything except step 3 still runs, and step 3 reports itself as
not run rather than passing quietly.

## Before you start

1. Runtime → Change runtime type → **T4 GPU**.
2. Upload `prior_runs.zip` to Drive as above.
3. Run all.

The preflight cell is the gate. It runs on CPU in seconds and stops the session
before any GPU time if:

- the Fold A prompt list does not hash to the value the published 48-token runs
  were measured against — this is the check that matters most, because the
  corpus is rebuilt from an unpinned HuggingFace revision and a silent upstream
  change would leave row *i* of the new run being a different prompt from row
  *i* of the old one, which is the entire basis of the comparison;
- the XSTest list does not hash to the value the 21 published cells used;
- any script is missing a flag the notebook passes;
- there is no GPU, or too little disk.

Checks that cannot run report `skip`, never `ok`.

## What the run produces

| Step | Output | Question it addresses |
|---|---|---|
| 1 | 256-token HH-RLHF, FP16 + 4.5 bits, two models — **and** the 48-token prefix of that same text | Does the 4.5-bit direction survive a wider window, tested per prompt on identical text? |
| 2 | 256-token XSTest FP16 baselines, three families | Is the harmful-compliance cell still empty when the window is not the binding constraint? |
| 3 | The five published runs re-graded under **both** scorers | How much of the published picture is an artefact of comparing label-word prefixes against one whole word? |

Roughly 3–3.5 hours. Every step checkpoints to Drive when it finishes, so a
disconnect costs one step rather than the session, and re-running skips work it
can prove is already complete — matching model, schemes, prompt count, token
budget, seed, corpus hash, row counts, and the scorer that produced it.

## Afterwards

The notebook writes a zip. Two things to check before reading anything into it:

- the filename. It is prefixed `INCOMPLETE_` if any step failed **or never
  ran**, because a step that produced nothing is not a step that found nothing;
- `ROUND3_STATUS.json`, inside the archive rather than beside it, which lists
  every step, its exit status, and whether it was restored rather than executed.

Then:

```
python scripts/analyse_round2.py          # per-prompt cross-budget comparison
python scripts/grader_coverage.py         # coverage-matched grader table
python scripts/check_paper_numbers.py     # every quoted number, re-verified
```

## Two things the run will not settle

It tests whether a result **persists**; it does not establish the result. Two
small models, one fixed corpus, greedy decoding and a single NF4 judge are the
conditions, and a finding that survives here has survived those conditions.

And no automatic scorer validates itself. The deepest open question in this
project is whether any of these graders is right, which needs blinded human
adjudication against a substantive-harm rubric. Nothing in this session touches
that.
