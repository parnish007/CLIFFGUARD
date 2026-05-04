# CLIFFGUARD

CLIFFGUARD is an edge-native, quantization-aware, black-box-tolerant, RL-adapted defense system against prompt injection at the safety cliff. The system is built around a **safety-cliff hypothesis**: post-training quantization degrades RLHF-installed safety behavior non-linearly in bit-width, and the marginal degradation between 4-bit and 3-bit dominates the marginal capability loss; therefore defenses must be placed in front of the model and must be quantization-aware in their thresholds, not their weights. CLIFFGUARD comprises eight named components — VESTIBULE, PROBE, B-PROBE, TRIPWIRE, CONDUCTOR, LOOKOUT, LADDER, and ATTEST — orchestrated by a LinUCB contextual bandit that adapts gate weights online from sparse incident feedback. The five pre-registered hypotheses driving the evaluation are: **H1** (cliff existence — a κ ≥ 0.25 refusal-margin jump at the Q4→Q3 boundary in at least two model families), **H2** (FPR decoupling, white-box — PROBE-RM FPR varies < 0.02 across schemes after per-quantization calibration), **H3** (FPR decoupling, black-box — same property for B-PROBE-LOGIT), **H4** (composition gain — the full gate stack achieves strictly smaller ABR than any single primitive at matched FPR), and **H5** (Tier-C structural weakness — Tier C without a dedicated classifier shows no statistically significant ABR reduction against the cliff exploiter; Tier C+ with PromptGuard-2-22M-INT4 does).

## Repository layout

The `cliffguard/` directory is the main Python package, containing sub-packages for every named component (vestibule, probe, bprobe, tripwire, conductor, lookout, ladder, attest) and the eval/ sub-package (stats, figures, drift simulation, five-fold orchestrator, and the reproducibility manifest builder). The `tests/` directory contains the full pytest suite covering all Phase A scaffolding. The `scripts/` directory holds standalone CLI entry points: `dry_run.py` (end-to-end pipeline smoke test), `run_full_evaluation.py` (five-fold orchestrator CLI), `build_preregistration_manifest.py` (reproducibility manifest CLI), `download_fold_a.py` (corpus download helper), and `generate_cliff_corpus.py`. The `docs/` directory holds `preregistration.md`, the pre-registered statistical analysis plan that fixes all hypotheses, thresholds, and acceptance criteria before any data is collected. The `data/` directory (gitignored) is where downloaded corpora land; see **Data acquisition** below. The `artifacts/` directory (gitignored) receives calibration tables, ARPA files, direction vectors, result JSONLs, figures, and reproducibility manifests produced during evaluation runs. The `configs/` directory holds YAML configuration files; `example.yaml` is the canonical starting point.

## Hardware tiers

### Tier A — RTX 5060 8 GB

Tier A targets an RTX 5060 8 GB consumer GPU running Qwen-2.5-7B or Llama-3-8B in NF4 (bitsandbytes) or AWQ-INT4. The memory budget fits the model (~5 GB), KV cache (~1 GB at 4 K ctx), an optional judge classifier (DeBERTa-86M FP16 or Llama Guard 3-1B-INT4 at 440 MB), and engine overhead — approximately 7 GB used with 1 GB headroom. All eleven gates are active including LOOKOUT-JG (with N=3 paraphrases); PROBE-RM, PROBE-MT, and PROBE-HD operate over full multi-layer residual streams via `output_hidden_states=True`; CONDUCTOR runs full LinUCB with |A|=16 arms.

**Quantization schemes:** FP16, INT8, NF4, AWQ_INT4, GGUF_Q6_K, GGUF_Q5_K_M, GGUF_Q4_K_M.

**Active gates:** VESTIBULE-LZ, VESTIBULE-PS, PROBE-RM, PROBE-MT, PROBE-HD, TRIPWIRE-H, TRIPWIRE-R, LOOKOUT-CT, LOOKOUT-JG, B-PROBE-LOGIT, B-PROBE-CONSISTENCY, ATTEST-WH.

**Install:**
```bash
git clone https://github.com/parnish007/CLIFFGUARD && cd CLIFFGUARD
uv sync --extra gpu
# Note: autoawq and vllm require Linux.
```

### Tier B — Raspberry Pi 5 8 GB

Tier B targets a Raspberry Pi 5 8 GB CPU running Qwen-2.5-1.5B or 3B at Q4_K_M via llama.cpp (~0.9 GB / 1.8 GB), achieving approximately 5–7 tok/s and 3–5 tok/s respectively. LOOKOUT-JG is omitted because N extra prefills would more than double per-request latency; B-PROBE-CONSISTENCY substitutes. PROBE runs with residual-stream access via llama.cpp's eval-callback or a small patch exposing intermediate residuals. CONDUCTOR is reduced to |A|=8 arms.

**Quantization schemes:** GGUF_Q4_K_M, GGUF_Q3_K_M.

**Active gates:** VESTIBULE-LZ, VESTIBULE-PS, PROBE-RM, PROBE-MT, PROBE-HD, TRIPWIRE-H, TRIPWIRE-R, LOOKOUT-CT, B-PROBE-LOGIT, B-PROBE-CONSISTENCY, ATTEST-WH.

**Install:**
```bash
uv sync --extra gpu
# Note: llama-cpp-python builds on all platforms including ARM64.
```

### Tier C — 2 GB embedded (HONEST SCOPE)

Tier C targets 2 GB-class embedded boards (RK3588 NPU W8A8, Pi 4 8 GB, Jetson Orin Nano 4 GB) running TinyLlama-1.1B or Qwen-2.5-0.5B/1.5B at Q3_K_M or Q4_K_M. Only VESTIBULE-LZ, VESTIBULE-PS, and ATTEST-WH are active; there is no bandit — gate weights are fixed at deployment because incident reward signal is too sparse to learn online. **Plain-language statement: Tier C is not meaningfully defended against A7 (the quantization-cliff exploiter)**; it is suitable only for single-task fixed-grammar assistants, narrow-domain controlled-input deployments, and read-only assistants where tool side-effects are mediated off-device.

**Quantization schemes:** GGUF_Q3_K_M, GGUF_IQ3_XXS, GGUF_Q2_K, RKNN_W8A8.

**Active gates:** VESTIBULE-LZ, VESTIBULE-PS, ATTEST-WH.

**Install:**
```bash
uv sync
# No --extra gpu needed for scaffolding or Tier C deployment.
```

### Tier C+ — 2 GB embedded with PromptGuard-2-22M

Tier C+ uses the same hardware budget as Tier C but adds Meta's PromptGuard-2-22M classifier (DeBERTa-xsmall, ~86 MB FP16 / ~25–30 MB INT4 estimated, MIT-licensed) which classifies inputs as benign or malicious via an energy-based loss for OOD robustness. The memory budget fits the Q3_K_M base model (~1.4 GB), KV cache (~150 MB), PromptGuard-2-22M-INT4 (~30 MB), and PROBE-RM final-layer projector (~50 MB) — approximately 1.65 GB total. Tier C+ reduces but does not eliminate Tier C's structural weakness; H5 is pre-registered to test this claim.

**Quantization schemes:** GGUF_Q3_K_M, GGUF_IQ3_XXS, RKNN_W8A8.

**Active gates:** VESTIBULE-LZ, VESTIBULE-PS, B-PROBE-LOGIT, ATTEST-WH.

**Install:**
```bash
uv sync
# No --extra gpu needed for scaffolding or Tier C+ deployment.
```

## Installation

**Tier A (GPU — RTX 5060 or equivalent):**
```bash
git clone https://github.com/parnish007/CLIFFGUARD && cd CLIFFGUARD
uv sync --extra gpu
```
Note: `autoawq` and `vllm` are Linux-only and are skipped automatically on other platforms. `llama-cpp-python` builds on all platforms.

**Tier B (Raspberry Pi 5 / ARM64 CPU):**
```bash
uv sync --extra gpu
```
Note: `llama-cpp-python` builds on ARM64; `torch` and `transformers` are also available for Pi 5. The `autoawq` and `vllm` extras are silently skipped on non-Linux.

**Tier C / C+ (2 GB embedded — scaffolding only):**
```bash
uv sync
```
No GPU extra is required for Phase A scaffolding or for the narrow Tier C gate set. Phase B engine integration for embedded targets (RKNN, llama.cpp ARM) requires platform-specific build steps described in blueprint §18.

## Data acquisition

The `data/` directory is gitignored and must be populated before any Phase B inference run. `scripts/download_fold_a.py` automates the download of the Anthropic-HH-RLHF dataset and OpenAssistant-OASST1, which together form the Fold A calibration corpus. Run it with `uv run python scripts/download_fold_a.py` and follow the printed instructions for any datasets that require manual acceptance of terms. The fold structure, corpus sizes, and required SHA-256 hashes are documented in blueprint §12.2. Real data is required for calibration (Fold A), cliff measurement (Fold B), defense composition (Fold C), bandit drift simulation (Fold D), and BCN-2 construction (Fold E); the Phase A dry run does not require any data files.

## Running the evaluation

**1. Dry run — no data, no GPU required:**
```bash
uv run python scripts/dry_run.py --tier A --scheme FP16
```

**2. Full evaluation — requires data and GPU:**
```bash
uv run python scripts/run_full_evaluation.py \
  --config configs/example.yaml
```

**3. Build reproducibility manifest:**
```bash
uv run python scripts/build_preregistration_manifest.py \
  --tier A --schemes FP16 NF4 GGUF_Q4_K_M GGUF_Q3_K_M
```

## Pre-registration and reproducibility

All five hypotheses (H1–H5), primitive thresholds, acceptance criteria (κ = 0.25 for H1, ε = 0.02 for H2/H3, α = 0.01 Bonferroni-corrected for H4/H5), and the statistical analysis plan are fixed in `docs/preregistration.md` before any data is collected or any model inference is run. The pre-registration document is SHA-256 hashed and recorded in every reproducibility manifest produced by `scripts/build_preregistration_manifest.py`. A manifest is considered valid if the git hash is present and not UNKNOWN, the preregistration hash matches the current `docs/preregistration.md`, and all listed data files exist on disk at verification time.

## Citation

```bibtex
@article{cliffguard2025,
  title   = {CLIFFGUARD: An Edge-Native, Quantization-Aware,
             Black-Box-Tolerant, RL-Adapted Defense System Against
             Prompt Injection at the Safety Cliff},
  author  = {[Author list — to be added at submission]},
  year    = {2025},
  note    = {Preprint}
}
```
