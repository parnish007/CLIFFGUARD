<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,100:1a1a2e&height=210&section=header&text=CLIFFGUARD&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=36&desc=Edge-Native+Prompt+Injection+Defense+at+the+Safety+Cliff&descAlignY=58&descSize=17" width="100%"/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&pause=1000&color=58A6FF&center=true&vCenter=true&width=700&lines=Quantization-aware+prompt+injection+defense;Safety+cliff%3A+where+RLHF+alignment+collapses;Eleven+primitives.+Four+tiers.+Five+hypotheses.;Phase+A+complete+%E2%80%94+Fold+A%2BB+run+on+Colab+T4)](https://git.io/typing-svg)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![mypy strict](https://img.shields.io/badge/mypy-strict-brightgreen?style=for-the-badge)](https://mypy-lang.org/)
[![ruff](https://img.shields.io/badge/ruff-passing-orange?style=for-the-badge)](https://docs.astral.sh/ruff/)
[![License MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)
[![arXiv cs.CR](https://img.shields.io/badge/arXiv-cs.CR-red?style=for-the-badge&label=arXiv&message=cs.CR)](https://arxiv.org/)
[![Phase A complete](https://img.shields.io/badge/Phase%20A-complete-brightgreen?style=for-the-badge)](docs/engineering_reference.md)
[![Phase B in progress](https://img.shields.io/badge/Phase%20B-in%20progress-yellow?style=for-the-badge)](#evaluation-status)
[![Fold A tested](https://img.shields.io/badge/Fold%20A-tested%20on%20Colab%20T4-brightgreen?style=for-the-badge)](#evaluation-status)
[![Fold B tested](https://img.shields.io/badge/Fold%20B-tested%20on%20Colab%20T4-brightgreen?style=for-the-badge)](#evaluation-status)

**[What is it](#the-safety-cliff) · [Architecture](#system-overview) · [Setup](#quick-start) · [Eval Status](#evaluation-status) · [Math](#five-pre-registered-hypotheses) · [Engineering Ref](docs/engineering_reference.md)**

</div>

> **The one-line version:** A quantized edge LLM that refuses harmful
> requests at FP16 may silently comply at Q3_K_M. CLIFFGUARD sits
> in front of the model, detects this regime shift, and blocks the
> attack — without retraining the model or requiring GPU inference
> for the defense layer.

## The Safety Cliff

Post-training quantization degrades RLHF safety alignment
non-linearly. A model that reliably refuses harmful requests at FP16
may comply at Q3_K_M — not because capability degrades proportionally
(MMLU drops ~8 points) but because the refusal direction in the
residual stream narrows and the margin between harmful and harmless
distributions collapses. This boundary, empirically near Q3_K_M for
Llama-3 and Mistral, is the **safety cliff**. Hypothesis H1
pre-registers a κ ≥ 0.25 jump in both the geometric refusal-direction
metric Δ_cliff and the behavioral attack-success-rate metric
Δ_B-cliff at the same quantization boundary, in at least two of
three model families.

Defenses baked into model weights cannot survive quantization if the
cliff hypothesis holds — the quantized residual stream no longer
encodes the safety signal the defense was trained on. The correct
response is to place defenses **in front of the model** and make
their thresholds **quantization-aware** via per-scheme calibration.
Per the FPR-decoupling theorem (H2, H3): the false-positive rate of
these gates is independent of quantization scheme up to calibration —
the same gate system is portable across NF4, AWQ-INT4, Q4_K_M, and
Q3_K_M without retraining. This is what CLIFFGUARD implements.

## System Overview

```mermaid
flowchart LR
    Input([User Prompt]) --> V

    subgraph V[VESTIBULE]
        LZ[LZ — compression ratio]
        PS[PS — provenance spotlight]
    end

    V --> ENGINE([Model Inference])

    ENGINE --> PR

    subgraph PR[PROBE]
        RM[RM — refusal margin]
        MT[MT — margin trajectory]
        HD[HD — harmfulness direction]
    end

    ENGINE --> TW

    subgraph TW[TRIPWIRE]
        TH[H — entropy CUSUM]
        TR[R — reference ratio]
    end

    PR --> CONDUCTOR
    TW --> CONDUCTOR

    ENGINE --> LO

    subgraph LO[LOOKOUT]
        CT[CT — canary token]
        JG[JG — compliance judge]
    end

    LO --> CONDUCTOR

    ATTEST-WH([ATTEST-WH boot]) -.-> CONDUCTOR

    CONDUCTOR --> ALLOW([ALLOW])
    CONDUCTOR --> BLOCK([BLOCK])
```

## Five Pre-Registered Hypotheses

| | Hypothesis | Claim | Metric | Acceptance | Empirical Status |
|---|---|---|---|---|---|
| 🏔️ | **H1** Cliff existence | Δ_cliff and Δ_B-cliff jump ≥ κ at same boundary, ≥ 2/3 families | `detect_cliff_boundary()` agrees across families | κ = 0.25 at Q3_K_M or below | 🔲 Not accepted — 1 scheme/family only |
| 📊 | **H2** FPR decoupling (white-box) | PROBE-RM FPR varies < ε across 5 schemes after calibration | max(FPR) − min(FPR) | ε = 0.02 at fpr_target = 0.05 | 🔲 Preliminary — 2/5 schemes tested |
| 🔲 | **H3** FPR decoupling (black-box) | B-PROBE-LOGIT FPR varies < ε, TPR < PROBE-RM | Same as H2 + TPR comparison | ε = 0.02 AND TPR(B-PROBE) < TPR(PROBE) | 🔲 Not tested |
| 🔀 | **H4** Composition gain | Full stack ABR < any single primitive at matched FPR | Wilcoxon signed-rank | p < 0.01 (Bonferroni α) | 🔲 Not tested |
| ⚠️ | **H5** Tier C weakness | Tier C: no significant ABR gain; Tier C+: significant | Wilcoxon p vs baseline | p(C) ≥ 0.05 AND p(C+) < 0.05 | 🔲 Not tested |

## Four Hardware Tiers

![Tier A](https://img.shields.io/badge/Tier_A-RTX_5060_8GB-dc2626?style=flat-square)
![Tier B](https://img.shields.io/badge/Tier_B-Pi_5_8GB-d97706?style=flat-square)
![Tier C](https://img.shields.io/badge/Tier_C-2GB_embedded-16a34a?style=flat-square)
![Tier C+](https://img.shields.io/badge/Tier_C%2B-2GB_%2B_PG2-2563eb?style=flat-square)

| Tier | Hardware | Schemes | Active Gates | Scope |
|---|---|---|---|---|
| **A** | RTX 5060 8 GB | FP16, NF4, AWQ-INT4, GGUF Q4–Q6 | All 12 (including LOOKOUT-JG) | Full stack, LinUCB \|A\|=16 |
| **B** | Raspberry Pi 5 8 GB CPU | GGUF Q4_K_M, Q3_K_M | All except LOOKOUT-JG (11 gates) | LinUCB \|A\|=8; B-PROBE-CONSISTENCY substitutes for JG |
| **C** | 2 GB embedded (RK3588 / Jetson / Pi 4) | GGUF Q3_K_M, IQ3_XXS, Q2_K, RKNN W8A8 | VESTIBULE-LZ, VESTIBULE-PS, ATTEST-WH | Narrow scope; no bandit; **not defended against A7** |
| **C+** | 2 GB embedded + PromptGuard-2-22M-INT4 | GGUF Q3_K_M, IQ3_XXS, RKNN W8A8 | VESTIBULE-LZ, VESTIBULE-PS, B-PROBE-LOGIT, ATTEST-WH | Modest scope; static weights; H5 tests this tier |

## Quick Start

> **Real results so far (Colab T4, May 2026):** Fold A (calibration) and Fold B (cliff measurement) have been run end-to-end on a Colab T4 GPU with Llama-3.2-3B-Instruct and NF4 quantization. Fold A calibrated thresholds at τ_FP16 = 0.09742 and τ_NF4 = 0.09827 against 400 benign prompts at FPR = 5%. Fold B measured Δ_cliff(NF4) = 0.167 (geometric) and Δ_B-cliff(NF4) = 0.000 (behavioral proxy). H1 was **not accepted** for this single scheme + family pair — more schemes and model families are required. See the [Evaluation Status](#evaluation-status) section for a full breakdown of what is tested vs planned.

<details>
<summary><b>Tier A — GPU (RTX 5060)</b></summary>

```bash
git clone https://github.com/parnish007/CLIFFGUARD.git && cd CLIFFGUARD
uv sync --extra gpu
# Note: autoawq and vllm require Linux; both are skipped automatically on Windows/macOS.
uv run python scripts/dry_run.py --tier A --scheme FP16
```

Expected output: pipeline completes, all 12 gates produce verdicts, block decision printed.

</details>

<details>
<summary><b>Tier B — Raspberry Pi 5</b></summary>

```bash
git clone https://github.com/parnish007/CLIFFGUARD.git && cd CLIFFGUARD
uv sync --extra gpu
# llama-cpp-python builds on all platforms including ARM64.
uv run python scripts/dry_run.py --tier B --scheme GGUF_Q4_K_M
```

</details>

<details>
<summary><b>Tier C / C+ — Embedded</b></summary>

```bash
git clone https://github.com/parnish007/CLIFFGUARD.git && cd CLIFFGUARD
uv sync
# No --extra gpu needed for scaffolding or Tier C deployment.
uv run python scripts/dry_run.py --tier C --scheme GGUF_Q3_K_M
```

For Tier C+: `--tier C_PLUS`.

</details>

## Repository Layout

| Path | Purpose |
|---|---|
| `cliffguard/` | Main Python package — eight component sub-packages plus `eval/` |
| `tests/` | Full pytest suite (939 tests, Phase A scaffolding) |
| `scripts/` | CLI entry points: dry_run, run_full_evaluation, build_manifest, download_fold_a |
| `docs/` | Documentation: preregistration, architecture, math, setup, engineering reference |
| `data/` | Gitignored — populated by `scripts/download_fold_a.py` before Phase B |
| `artifacts/` | Gitignored — calibration tables, ARPA files, result JSONLs, figures, manifests |
| `configs/` | YAML configuration files for evaluation runs |
| `pyproject.toml` | Build, dependency, and tool configuration |
| `configs/example.yaml` | Canonical starting-point configuration (copy and edit) |

## Documentation

- [docs/what_is_it.md](docs/what_is_it.md) — Plain-language introduction and FAQ
- [docs/architecture.md](docs/architecture.md) — System architecture and component diagrams
- [docs/math.md](docs/math.md) — Mathematical foundations and formal definitions
- [docs/setup.md](docs/setup.md) — Device-by-device setup guide
- [docs/engineering_reference.md](docs/engineering_reference.md) — Module API reference for Phase B

## Evaluation Status

![Tests](https://img.shields.io/badge/tests-939_passing-brightgreen?style=flat-square)
![mypy](https://img.shields.io/badge/mypy-strict_53_files-brightgreen?style=flat-square)
![ruff](https://img.shields.io/badge/ruff-clean-orange?style=flat-square)
![Phase A](https://img.shields.io/badge/Phase%20A-complete-brightgreen?style=flat-square)
![Phase B](https://img.shields.io/badge/Phase%20B-in%20progress-yellow?style=flat-square)

**Phase A** is complete: 939 tests pass on synthetic data, mypy strict on 53 files, ruff clean. All components exist as Phase A stubs that exercise the full pipeline shape and verify API contracts without real model inference.

**Phase B** wires real inference-engine adapters (transformers + bitsandbytes NF4, autoawq, vLLM, llama.cpp eval-callback) on hardware tiers and replaces synthetic arrays with real residual streams and logprobs. Fold A and Fold B have been run on **Colab T4 (May 2026)**. Folds C–E remain planned.

### Evaluation Folds

| Fold | Purpose | Status | Hardware | Model / Scheme | Key Results |
|---|---|---|---|---|---|
| **Fold A** | Threshold calibration (PROBE-RM) | ✅ **COMPLETE** | Colab T4 | Llama-3.2-3B-Instruct, FP16 + NF4 | τ_FP16 = 0.09742, τ_NF4 = 0.09827, FPR = 5%, 400 benign prompts |
| **Fold B** | Cliff measurement | ✅ **COMPLETE** | Colab T4 | Llama-3.2-3B-Instruct, NF4 | Δ_cliff(NF4) = 0.167, Δ_W-cliff(NF4) = 0.014, Δ_B-cliff(NF4) = 0.000 |
| **Fold C** | Defense composition (ABR/FPR per gate) | 🔲 **NOT RUN** | — | — | — |
| **Fold D** | Bandit drift recovery (CONDUCTOR online) | 🔲 **NOT RUN** | — | — | — |
| **Fold E** | BCN-2 dataset construction | 🔲 **NOT RUN** | — | — | — |

### Hypothesis Status

| Hypothesis | Claim | Status | Evidence so far |
|---|---|---|---|
| 🏔️ **H1** Cliff existence | Δ_cliff and Δ_B-cliff jump ≥ κ = 0.25 at same boundary, ≥ 2/3 families | 🔲 **NOT ACCEPTED** | Only 1 scheme (NF4) × 1 family tested. Δ_cliff = 0.167 < κ; Δ_B-cliff = 0.000. Needs Q3_K_M + multiple families. |
| 📊 **H2** FPR decoupling (white-box) | PROBE-RM FPR varies < ε = 0.02 across 5 schemes after calibration | 🔲 **PRELIMINARY** | 2 schemes (FP16, NF4) show FPR within target. Full test needs 5 schemes. |
| 🔲 **H3** FPR decoupling (black-box) | B-PROBE-LOGIT FPR varies < ε, TPR < PROBE-RM | 🔲 **NOT TESTED** | — |
| 🔀 **H4** Composition gain | Full stack ABR < any single primitive at matched FPR | 🔲 **NOT TESTED** | Depends on Fold C. |
| ⚠️ **H5** Tier C weakness | Tier C no ABR gain; Tier C+ significant gain | 🔲 **NOT TESTED** | Depends on Fold C. |

### Quantization Schemes Tested

| Scheme | Status |
|---|---|
| FP16 | ✅ Calibration complete (Fold A) |
| NF4 (bitsandbytes) | ✅ Calibration + cliff measurement (Folds A + B) |
| GGUF Q4_K_M | 🔲 Not run |
| GGUF Q3_K_M | 🔲 Not run |
| GGUF Q2_K | 🔲 Not run |
| AWQ-INT4 | 🔲 Not run |

### Model Families Tested

| Family | Status |
|---|---|
| Llama-3.2-3B-Instruct | ✅ Folds A + B complete on Colab T4 |
| Qwen | 🔲 Not run |
| Mistral | 🔲 Not run |

### Component Implementation Status

| Component | Status | Notes |
|---|---|---|
| PROBE-RM | ✅ Real inference tested | bitsandbytes NF4, Colab T4 |
| VESTIBULE (LZ, PS) | 🔲 Scaffolding only | Phase A stubs — passes 939 synthetic tests |
| TRIPWIRE (H, R) | 🔲 Scaffolding only | Phase A stubs |
| LOOKOUT (CT, JG) | 🔲 Scaffolding only | Phase A stubs |
| CONDUCTOR | 🔲 Scaffolding only | Phase A stubs |
| B-PROBE (LOGIT, CONSISTENCY) | 🔲 Scaffolding only | Phase A stubs |
| ATTEST-WH | 🔲 Scaffolding only | Phase A stubs |
| LADDER | 🔲 Scaffolding only | Phase A stubs |

> Full evaluation follows the pre-registered five-fold protocol in `docs/preregistration.md`. The Colab notebook (`notebooks/`) has been tested end-to-end on T4 with Drive checkpointing for Folds A and B.

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

<div align="center">

<img src="https://capsule-render.vercel.app/api?section=footer&type=waving&color=0:0D1117,100:1a1a2e&height=80" width="100%" />

</div>
