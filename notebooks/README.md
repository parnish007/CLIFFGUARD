# notebooks/

Interactive entry points for running CLIFFGUARD on hosted GPUs (Google Colab today; Kaggle / Lightning Studios in future). The substantive code lives in [`cliffguard/`](../cliffguard/); this directory is a thin orchestration layer over it.

## What's in here

| File | Purpose |
|---|---|
| [`cliffguard_colab.ipynb`](cliffguard_colab.ipynb) | The Colab notebook — runs Fold A calibration and Fold B cliff measurement end-to-end. Checkpoints to Google Drive after every `(model, scheme)` pair. |
| [`colab_helper.py`](colab_helper.py) | Reusable helper functions used by both the notebook and `scripts/colab_run.py`. Public API: `banner`, `symlink_datasets_from_drive`, `choose_model`, `run_fold_a_with_checkpoint`, `run_fold_b_with_checkpoint`, `sync_artifacts_to_drive`, `assemble_fold_b`. |

## Quick start

1. Open [`cliffguard_colab.ipynb`](cliffguard_colab.ipynb) in Google Colab.
2. Set runtime type to GPU (*Runtime → Change runtime type → T4 GPU*).
3. Edit cell C5 to point at your fork of the repo (replace `<owner>`).
4. Run cells top to bottom. The notebook self-checkpoints — re-running any cell after a disconnect resumes from the last completed scheme.

For the full setup walkthrough — what fits on which Colab tier, troubleshooting, wall-clock estimates — read [`docs/setup_colab.md`](../docs/setup_colab.md).

## Why a notebook *and* a script

The two entry points cover different workflows:

- **`notebooks/cliffguard_colab.ipynb`** is for interactive use. You see each cell's output, edit `config` between cells, and can pause to inspect the run directory at any time. This is the right mode when you are tuning hyperparameters or debugging a model.

- **[`scripts/colab_run.py`](../scripts/colab_run.py)** is for unattended batch execution — either via `papermill` driving the notebook, or directly via `python scripts/colab_run.py --tier A`. Same workflow, no UI. This is the right mode for an overnight Pro+ background-execution run, or for chaining multiple runs together in a single Colab session.

Both share the same `colab_helper` module, so a notebook checkpoint can be resumed by the CLI and vice versa.
