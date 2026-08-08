<div align="center">

[← README](../README.md) &nbsp;|&nbsp;
[Docs index](README.md) &nbsp;|&nbsp;
**Setup** &nbsp;|&nbsp;
[Methodology](methodology.md) &nbsp;|&nbsp;
[Local GPU](setup_local_gpu.md) &nbsp;|&nbsp;
[Colab](setup_colab.md)

</div>

# Setup — which guide you actually want

Short on purpose. This page routes; the guides it points at hold the measured
detail.

## Pick one

| Your situation | Go to | What you get |
|---|---|---|
| **You want to run the measurement** | [`setup_colab.md`](setup_colab.md) → `notebooks/colab_labelled.ipynb` | The full pipeline on a free T4. This is the supported path. |
| You want models on your own GPU | [`setup_local_gpu.md`](setup_local_gpu.md) | What actually limits a local run — host RAM, not VRAM — and the torch/transformers pair that works |
| You only want to re-analyse stored runs | [below](#re-analysing-without-a-gpu) | No GPU needed |

## Re-analysing without a GPU

Every completion is stored verbatim, so any grader or analysis can be re-applied
without regenerating text. Given run directories under `artifacts/runs/`:

```bash
pip install numpy scipy                    # the whole dependency set for this path

python scripts/analyse_labelled.py --runs artifacts/runs --include '*lab-*'
python scripts/analyse_matrix.py   --runs artifacts/runs --include '*lab-*'
python scripts/reanalyse_runs.py   artifacts/runs
```

Rebuilding the paper's data, tables and figures from those runs:

```bash
python scripts/build_paper_data.py
python scripts/build_paper_tables.py
python scripts/build_paper_figures.py
python scripts/check_paper_numbers.py      # fails if the prose disagrees
```

## Prerequisites for the GPU paths

- **Python 3.10+.**
- **`transformers` pinned below 5.** The 5.x line installs cleanly against older
  torch and then segfaults on every `from_pretrained` — measured, exit 139, with
  no model code involved.
- **No HuggingFace token.** Every model and suite the default configuration
  touches is ungated. Llama-3.2 and Gemma-2 are gated, so they are listed but
  commented out in the notebook rather than made a prerequisite.
- **Disk.** The default three models plus the 7B judge are about 32.5 GB of
  weights. The notebook chooses its cache location from measured free space,
  because a free Google Drive is 15 GB in total.

## What used to be on this page

Earlier versions described a different project: a prompt-injection defence with a
gate-stack architecture, hardware tiers down to a Raspberry Pi, a five-fold
orchestrator, and a `--tier` dry run. No result in this repository depends on
that design, its entry points no longer import, and the commands documenting it
would fail if pasted — `run_evaluation_3050.py --folds A` errors, because the
script has no `--folds` flag.

Removed rather than left to mislead. The history is in git.
