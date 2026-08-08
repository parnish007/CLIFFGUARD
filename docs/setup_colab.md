<div align="center">

[← Docs index](README.md) &nbsp;|&nbsp;
[Setup](setup.md) &nbsp;|&nbsp;
**Colab** &nbsp;|&nbsp;
[Local GPU](setup_local_gpu.md) &nbsp;|&nbsp;
[Methodology](methodology.md)

</div>

# CLIFFGUARD on Google Colab

A free T4 is enough. That is a design constraint, not an aspiration.

## 1. What to run

**Open this, set the runtime to T4 GPU, and Run all:**

```
https://colab.research.google.com/github/parnish007/CLIFFGUARD/blob/main/notebooks/colab_labelled.ipynb
```

It needs **one click**: approving the Google Drive mount. Everything that makes
the run resumable — the journal, the per-scheme caches, the run directories —
lives on Drive, so the notebook stops rather than silently continuing without it.

| Notebook | Use it for | Status |
|---|---|---|
| [`colab_labelled.ipynb`](../notebooks/colab_labelled.ipynb) | **Run this.** Both annotation axes on the labelled suites, and the 2×2 | current |
| [`colab_run.ipynb`](../notebooks/colab_run.ipynb) | Round 1: 7B judge, Qwen2.5-3B and Phi-3.5-mini, refusal + GSM8K arms | already in the paper |
| [`colab_round2.ipynb`](../notebooks/colab_round2.ipynb) | Round 2: 256-token budget, AWQ/GPTQ/NF4, 1.5B regrade | already in the paper |

> `cliffguard_colab.ipynb` and `colab_helper.py` belong to an earlier design and
> are kept for provenance only — their labels come from a corpus partition that
> agrees with model behaviour 52.4% of the time, which is chance. Do not start
> there.

## 2. What it costs

Calculated from this project's own measured throughput, for the default
configuration (three models, XSTest at 150 prompts per class, seven rungs):

| Stage | Per model |
|---|---|
| Generation | ~32 min |
| Model loads + RTN construction | ~12 min |
| NLL scoring under the FP16 reference | ~2 min |
| 7B judge, five-way | ~15 min |
| **Total** | **~61 min** |

Three models ≈ **3.0 h** against a 3.5 h session cap. The notebook prints this
before anything downloads.

**The first session is longer.** It excludes ~32.5 GB of weight downloads, which
is why the cache location matters (§4).

Three choices are what make this fit a free tier:

- **`--no-activations`.** The residual stream is a second full forward over every
  prompt, and `output_hidden_states` materialises every layer to keep one. Only
  the probe arm reads it, and nothing on this path is the probe arm.
- **One judge pass.** The five-way grader also emits the three-way labels, so the
  7B model sweeps the completions once rather than twice.
- **One corpus with both classes.** The 2×2 is paired against each run's own FP16
  baseline, so both prompt classes must be in the same run.

## 3. Hardware you will actually get

Colab allocations change; this is the typical shape.

- **Free.** T4 16 GB. Reclaimed after roughly 3–4 hours, often sooner, with no
  warning. No background execution — closing the tab ends the session.
- **Pro.** Priority access to T4, L4 or A100 40 GB. Longer sessions.
- **Pro+.** A100 40/80 GB and background execution, which is the only way to do
  an unattended multi-hour run.

Set `DEADLINE_HOURS` in the config cell slightly under your real cap — 3.5 for
free, 11 for Pro. The pipeline uses it to decline to *start* a step it cannot
finish, so the runtime is reclaimed between steps rather than inside one.

### What fits

Peak VRAM measured on this project's own runs:

| Model | fp16 peak | Free T4 (16 GB) |
|---|---|---|
| Qwen2.5-0.5B | 1.5 GB | ✅ |
| SmolLM2-1.7B | 3.4 GB | ✅ |
| Qwen2.5-3B | 7.2 GB | ✅ |
| Phi-3.5-mini | 14.8 GB | ⚠️ tight |
| Llama-3.1-8B | 19.2 GB | ❌ |
| Qwen2.5-14B | 34.0 GB | ❌ A100 only |

The 7B judge is loaded in **NF4** (~4.5 GB), which is what lets it share a T4
with the rest of the run. That is recorded in the output manifest — it is a
quantized grader, and the paper says so.

Only one model is resident at a time: each scheme loads, generates, and is freed
before the next. Host RAM on a free instance is ~12.7 GB, and the loader streams
weights with `low_cpu_mem_usage` straight to the GPU, so the models do not
accumulate.

## 4. Where the weights go, and why it matters

The default three models plus the 7B judge are **32.5 GB**. A free Google Drive
is **15 GB in total**.

So the notebook picks its cache location from *measured* free space:

- Drive has room → weights go to Drive, and a reconnect reuses them.
- It does not → weights go to `/content` (~78 GB, ephemeral), and a reconnect
  re-downloads them.

Re-downloading is the cheaper mistake. Filling Drive fails the run *and* leaves
you with no space.

## 5. Resuming a dead session

**Reconnect and re-run the pipeline cell. That is the whole procedure.**

Every stage is a step in a journal on Drive. A step that finished is skipped; one
whose command, declared output or environment changed is re-run; one whose output
has vanished is re-run; and inside a step the ladder and grader cache per scheme,
so a partly-finished model resumes at the rung it reached.

Two behaviours look like failures and are not:

- **A step that never started.** Near the end of a session the pipeline refuses to
  begin a step it estimates will not fit, marks it `pending`, and stops. That is
  stopping cleanly rather than being killed mid-step.
- **A step that says `killed` and retries.** Return code −9 is the host OOM
  killer. Each attempt starts from the caches the previous one wrote, so a retry
  makes progress. It gives up after two.

A run directory is only created **after** every scheme finishes, so a kill
mid-model leaves no partial directory to confuse the next session.

## 6. Where results live

| Layer | Path | Lifetime |
|---|---|---|
| Run directories | `<Drive>/cliffguard/artifacts/runs/` | durable |
| Per-scheme caches | `<Drive>/cliffguard/cache_labelled/` | durable |
| Journal | `<Drive>/cliffguard/journal_labelled.json` | durable |
| Per-step logs | `<Drive>/cliffguard/logs_labelled/<step>.log` | durable |
| Analysis output | `<Drive>/cliffguard/matrix_stats.json`, `labelled_stats.json` | durable |
| The clone | `/content/CLIFFGUARD/` | gone on disconnect |

`CLIFFGUARD_ARTIFACTS` is what redirects run directories to Drive. Written
relative to the clone they would live in `/content` and vanish with the session,
after the GPU time that produced them was spent.

**The per-step logs matter.** Colab's cell output lives in the browser and is
lost on disconnect; the log on Drive is the only record of why something failed.

## 7. Bringing results home

The last cell packs the small files — manifests, results, analysis JSON, logs —
into one archive and offers it for download. Activations are not collected on
this path, and the per-scheme completion caches stay on Drive.

Unzip at the repository root so `artifacts/runs/<...>` lands correctly, then
re-run the analyses locally with no GPU (see [`setup.md`](setup.md)).

## 8. Troubleshooting

Full symptom table in [`methodology.md` §8](methodology.md#8-debugging-by-symptom).
The ones specific to Colab:

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: scripts.colab_pipeline` | the clone reflects what was **pushed**; the branch is stale | push, then re-run. The notebook now names the missing files up front |
| Notebook stops at the Drive mount | the authorisation was dismissed | re-run the cell and approve it — this is the one click required |
| Segfault on every `from_pretrained` | `transformers` 5.x against Colab's torch | pinned below 5 in the notebook; do not relax it |
| `rc = -9` | host OOM | lower `N_PER_CLASS` or `BATCH`; retries resume from caches |
| Disk error part-way | Drive filled | the notebook now picks the cache location from free space |
| "all steps complete" but results missing | *fixed* — the deadline path used to return success | the notebook checks `pipe.outstanding()` |

## 9. Cost

The default configuration is designed for **free** Colab. Pro (~$10/month) buys
longer sessions and better allocation, which mainly removes the need to reconnect
between models. Pro+ adds background execution, which is the only way to run
unattended.

Nothing in the default configuration requires a paid tier or a HuggingFace token.
