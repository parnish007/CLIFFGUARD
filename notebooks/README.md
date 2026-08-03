# notebooks/

Interactive entry points for hosted GPUs. The substantive code lives in
[`cliffguard/`](../cliffguard/) and [`scripts/`](../scripts/); this directory is a thin
orchestration layer over it.

## Where each measurement runs

The behavioural-rate-distortion measurement (Stages 0–4) is a repo script,
[`scripts/run_local_ladder.py`](../scripts/run_local_ladder.py), and **most of it runs on a 6 GB
laptop GPU**. Only three things genuinely need a hosted GPU, and those are what the Colab notebook
carries.

| Measurement | Where | Why |
|---|---|---|
| Qwen2.5-1.5B RTN ladder (8→2 bits), Stages 0–4 | **local** | 3.1 GB VRAM; no downloads |
| Qwen2.5-1.5B GGUF k-quant ladder | **local (slow) or Colab** | ~10 GB of downloads |
| Qwen2.5-3B RTN ladder | **Colab** | 6.2 GB FP16 exceeds a 6 GB card |
| Phi-3.5-mini RTN ladder | **Colab** | 7.6 GB FP16 |
| AWQ / GPTQ arm | **Colab** | `autoawq` / `gptqmodel` backends |

Peak VRAM is one dense FP16 copy of the model regardless of ladder length, because every rung —
RTN or GGUF — is materialised as a dense torch model rather than run through a quantized kernel.
That is what makes a measured `d'` available at *every* rung, which is what Stage 4's out-of-sample
prediction needs to exist at all.

## What's in here

| File | Purpose |
|---|---|
| [`colab_ladder_and_eta.ipynb`](colab_ladder_and_eta.ipynb) | **Current.** Runs `scripts/run_local_ladder.py` unchanged on the three arms above: scale (3B), family (Phi-3.5), and deployment realism (GGUF k-quants), plus an optional AWQ/GPTQ arm. Starts with a preflight that self-tests every repo API it will call. |
| [`stage0_noise_floor_and_isotropy.ipynb`](stage0_noise_floor_and_isotropy.ipynb) | Stage 0 only — the rotation-replication gate and the isotropy test, in isolation. |
| [`cliffguard_colab.ipynb`](cliffguard_colab.ipynb) | Pre-pivot notebook: Fold A calibration and Fold B cliff measurement for the original gate-stack design. Kept for provenance. |
| [`colab_helper.py`](colab_helper.py) | Helpers for the pre-pivot notebook and `scripts/colab_run.py`. |

## Running the current notebook

1. Open [`colab_ladder_and_eta.ipynb`](colab_ladder_and_eta.ipynb) in Colab.
2. *Runtime → Change runtime type → T4 GPU*.
3. Run top to bottom. Nothing is gated — no HuggingFace token is needed.
4. When it finishes: download the executed `.ipynb` into this directory, and unzip the results
   archive at the repo root so the run directories land in `artifacts/runs/`.

Every arm checkpoints to Drive, so a disconnect costs at most one arm.

## Running the same thing locally

```bash
# the primary ladder — no downloads, ~12 min on a 6 GB card
python scripts/run_local_ladder.py --n 250

# a fast wiring check first
python scripts/run_local_ladder.py --n 24 --smoke --bits 8 4 2

# the deployment-realistic k-quant ladder (needs the GGUF files)
python scripts/run_local_ladder.py --ladder-kind gguf
```

Results land in `artifacts/runs/<utc>_<git-sha>_<label>/` with a full provenance manifest, and one
line is appended to `artifacts/runs/INDEX.md`.

## Why a notebook *and* a script

The notebook is a **runner**, not a second implementation. Every arm shells out to
`scripts/run_local_ladder.py`, so there is no parallel copy of the measurement logic to drift out
of sync — a failure mode this project has already paid for once.

What the notebook adds over the raw script is the hosted-GPU scaffolding: repo clone, dependency
install, Drive checkpointing, corpus construction, a preflight that catches API drift before an
hour of GPU time is spent, and a single results archive to bring home.
