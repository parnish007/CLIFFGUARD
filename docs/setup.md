# Device Setup Guide

This guide covers cloning and running CLIFFGUARD on each hardware tier. Phase A dry run requires no GPU and no data — it exercises the full pipeline shape using synthetic arrays and completes in under a second on any machine. Phase B requires the appropriate hardware and the real corpus.

## Prerequisites (All Tiers)

- Git ≥ 2.40
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) — install with: `curl -LsSf https://astral.sh/uv/install.sh | sh` (Linux/macOS) or `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows)

**Windows — disk space:** uv caches packages on the system drive by default. If C: has limited free space, add these to your PowerShell profile (`$PROFILE`):

```powershell
$env:UV_CACHE_DIR = "D:\uv-cache"
$env:UV_PROJECT_ENVIRONMENT = "D:\cliffguard-venv"
```

This redirects the package cache and the virtual environment to a drive with more space. The `UV_CACHE_DIR` variable must be set before any `uv` command.

## Tier A — RTX 5060 8 GB (Linux recommended)

**Hardware:** RTX 5060 8 GB or equivalent (any CUDA-capable GPU with ≥ 8 GB VRAM). Linux recommended — `autoawq` and `vllm` are Linux-only. Windows works for `torch` + `bitsandbytes` NF4.

**Step-by-step:**

1. Clone the repository:
   ```bash
   git clone https://github.com/parnish007/CLIFFGUARD.git
   cd CLIFFGUARD
   ```

2. Install with GPU extras:
   ```bash
   uv sync --extra gpu
   ```
   Note: `autoawq` and `vllm` are gated by `sys_platform == 'linux'` in `pyproject.toml` and are silently skipped on Windows and macOS. `torch`, `transformers`, `bitsandbytes`, and `llama-cpp-python` install on all platforms.

3. Verify Phase A (no GPU needed):
   ```bash
   uv run python scripts/dry_run.py --tier A --scheme FP16
   ```
   Expected: all 12 gates produce verdicts, block decision printed, exit code 0.

4. Download Fold A corpus (Phase B only):
   ```bash
   uv run python scripts/download_fold_a.py
   ```
   Follow the printed instructions for any datasets requiring manual terms acceptance.

5. Full evaluation (Phase B, requires Fold A corpus and GPU):
   ```bash
   uv run python scripts/run_full_evaluation.py \
     --config configs/example.yaml --tier A
   ```

## Tier B — Raspberry Pi 5 8 GB

**Hardware:** Raspberry Pi 5 8 GB. Models run via `llama-cpp-python` on ARM64 CPU. Approximate throughput: Qwen-2.5-1.5B Q4_K_M at ~5–7 tok/s, 3B at ~3–5 tok/s (per Stratosphere Lab 2025 benchmarks).

**Step-by-step:**

1. Clone and enter the repository:
   ```bash
   git clone https://github.com/parnish007/CLIFFGUARD.git && cd CLIFFGUARD
   ```

2. Install with GPU extras (`llama-cpp-python` is the key dependency here):
   ```bash
   uv sync --extra gpu
   ```
   `llama-cpp-python` builds from source on ARM64. On Pi 5 this typically takes 5–10 minutes. The `autoawq` and `vllm` extras are silently skipped (Linux-only gate + no CUDA).

3. Verify Phase A:
   ```bash
   uv run python scripts/dry_run.py --tier B --scheme GGUF_Q4_K_M
   ```

4. Download corpus:
   ```bash
   uv run python scripts/download_fold_a.py
   ```

5. Full evaluation:
   ```bash
   uv run python scripts/run_full_evaluation.py \
     --config configs/example.yaml --tier B \
     --schemes GGUF_Q4_K_M GGUF_Q3_K_M
   ```

## Tier C — 2 GB Embedded (RK3588 / Jetson Orin Nano / Pi 4)

**Hardware:** RK3588 NPU W8A8 (10–15 tok/s on 1.1B per tinycomputers.io), Jetson Orin Nano 4 GB, or Pi 4 8 GB. Models: TinyLlama-1.1B or Qwen-2.5-0.5B/1.5B in Q3_K_M or Q4_K_M.

**HONEST SCOPE:** Tier C is not meaningfully defended against A7 (quantization-cliff exploiter). It runs precisely the models where the cliff is most likely. The minimal gate set (VESTIBULE-LZ, VESTIBULE-PS, ATTEST-WH) raises attacker cost only marginally. Suitable **only** for fixed-grammar single-task assistants with no open-domain adversarial exposure. Deployments must ship with a label: "NOT FOR OPEN-DOMAIN ADVERSARIAL USE."

**Step-by-step:**

1. Clone:
   ```bash
   git clone https://github.com/parnish007/CLIFFGUARD.git && cd CLIFFGUARD
   ```

2. Install without GPU extras:
   ```bash
   uv sync
   ```
   The RKNN runtime is board-specific (install from Rockchip's RKNN-Toolkit2 release) and is not managed by this project. For llama.cpp on Pi 4 / Jetson, install `llama-cpp-python` separately or use `uv sync --extra gpu`.

3. Verify Phase A:
   ```bash
   uv run python scripts/dry_run.py --tier C --scheme GGUF_Q3_K_M
   ```
   Expected: 3 gates run (VESTIBULE-LZ, VESTIBULE-PS, ATTEST-WH).

4. Note: No bandit is used at Tier C. Gate weights are fixed at deployment. CONDUCTOR falls back to a static expert-tuned policy with EWMA-based drift alarms only.

## Tier C+ — 2 GB Embedded with PromptGuard-2-22M-INT4

Same hardware as Tier C. Adds Meta's PromptGuard-2-22M (DeBERTa-xsmall, 22 M parameters, MIT-licensed) as the B-PROBE-LOGIT gate.

**Memory budget:** Q3_K_M base model ~1.4 GB + KV cache ~150 MB + PromptGuard-2-22M-INT4 ~30 MB + PROBE-RM final-layer projector ~50 MB ≈ 1.65 GB total. Fits under 1.8 GB.

**Step-by-step:**

1. Same steps as Tier C for clone and `uv sync`.

2. Verify Phase A:
   ```bash
   uv run python scripts/dry_run.py --tier C_PLUS --scheme GGUF_Q3_K_M
   ```
   Expected: 4 gates run (VESTIBULE-LZ, VESTIBULE-PS, B-PROBE-LOGIT, ATTEST-WH).

3. PromptGuard-2-22M-INT4 weights: download from `meta-llama/Llama-Prompt-Guard-2-22M` on Hugging Face. Phase B wiring is in `cliffguard/engines/` (stub in Phase A).

## Configuration

Copy the example configuration and edit it before running:

```bash
cp configs/example.yaml configs/my_run.yaml
```

Key fields to edit:

| Field | Default | Description |
|---|---|---|
| `tier` | `"A"` | Hardware tier: A, B, C, or C_PLUS |
| `schemes` | FP16, NF4, Q4_K_M, Q3_K_M | Quantization schemes to evaluate |
| `folds` | [A, B, C, D, E] | Evaluation folds to run (A must run first) |
| `data_dir` | `"data/"` | Path to corpus directory |
| `artifacts_dir` | `"artifacts/"` | Where results, calibration tables, figures are written |
| `fpr_target` | `0.05` | Pre-registered FPR target (do not change without amending preregistration) |
| `n_calibration` | `2000` | Fold A benign prompts (minimum per §12.2) |
| `n_attack` | `500` | Attack prompts per adversary per scheme |
| `kenlm.order_tier_ab` | `5` | KenLM n-gram order for Tier A/B (blueprint §5.5) |
| `kenlm.order_tier_c` | `3` | KenLM n-gram order for Tier C/C+ (decisions_log C25) |
| `hardware.description` | `"unspecified"` | Written into the reproducibility manifest |

Run with your config:

```bash
uv run python scripts/run_full_evaluation.py --config configs/my_run.yaml
```

## Data Acquisition

Real evaluation (Phase B) requires the Fold A calibration corpus and the adversarial evaluation corpus.

**Fold A — automatic download:**

```bash
uv run python scripts/download_fold_a.py
```

This downloads Anthropic-HH-RLHF and OpenAssistant-OASST1 (the Fold A benign calibration corpus). Data is written to `data/folds/fold_a/` and is gitignored. Follow the printed instructions for any datasets requiring manual terms acceptance.

**Folds B/C — adversarial corpus:**

The adversarial corpus requires assembly from multiple sources (blueprint §12.6): JailbreakBench, AdvBench-50, ArtPrompt, and synthetic cliff-exploiters (A7 prompts generated per the BCN-2 protocol). Phase B corpus assembly is currently manual — `scripts/download_fold_a.py` handles only Fold A. See `cliffguard/eval/attack_corpus.py` for the corpus schema and `docs/preregistration.md` for required corpus sizes.

**Folds D/E — held out:**

Fold D (bandit drift) and Fold E (BCN-2 construction) are unblinded only after Folds A/B/C complete. The scripts are scaffolded in Phase A and will be activated in Phase B.
