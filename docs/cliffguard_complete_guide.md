<div align="center">

[← what_is_it](what_is_it.md) &nbsp;|&nbsp;
[architecture](architecture.md) &nbsp;|&nbsp;
[math](math.md) &nbsp;|&nbsp;
[setup](setup.md) &nbsp;|&nbsp;
**complete_guide** &nbsp;|&nbsp;
[engineering_ref](engineering_reference.md)

</div>

# CLIFFGUARD Complete Guide

*A unified, end-to-end reference for anyone who wants to understand CLIFFGUARD — the problem it solves, the architecture that solves it, how it is implemented, what has been tested, and how to run it — without reading the 100-page blueprint.*

---

## §0 Table of Contents

### Part I — Orientation
- [§1 TL;DR](#1-tldr)
- [§2 The Safety Cliff Problem](#2-the-safety-cliff-problem)
  - [§2.1 Quantization, in 10 lines](#21-quantization-in-10-lines)
  - [§2.2 What "the cliff" actually means](#22-what-the-cliff-actually-means)
  - [§2.3 Why this matters](#23-why-this-matters)
  - [§2.4 The paint-on-foundation analogy](#24-the-paint-on-foundation-analogy)
  - [§2.5 The empirical phenomenon](#25-the-empirical-phenomenon)
- [§3 What CLIFFGUARD Is (and Is Not)](#3-what-cliffguard-is-and-is-not)
  - [§3.1 What it is](#31-what-it-is)
  - [§3.2 What it is not](#32-what-it-is-not)
  - [§3.3 The 30-second pitch](#33-the-30-second-pitch)

### Part II — Threat Model
- [§4 The 9 Adversary Classes A1–A9](#4-the-9-adversary-classes-a1a9)
- [§5 Coverage Gaps](#5-coverage-gaps)
- [§6 Why Not Just Use a Content Classifier?](#6-why-not-just-use-a-content-classifier)

### Part III — The Defensive Architecture
- [§7 The 8 Components, Top-Down](#7-the-8-components-top-down)
- [§8 The 12 Primitives](#8-the-12-primitives)
  - [§8.1 VESTIBULE-LZ](#81-vestibule-lz)
  - [§8.2 VESTIBULE-PS](#82-vestibule-ps)
  - [§8.3 PROBE-RM](#83-probe-rm)
  - [§8.4 PROBE-MT](#84-probe-mt)
  - [§8.5 PROBE-HD](#85-probe-hd)
  - [§8.6 TRIPWIRE-H](#86-tripwire-h)
  - [§8.7 TRIPWIRE-R](#87-tripwire-r)
  - [§8.8 LOOKOUT-CT](#88-lookout-ct)
  - [§8.9 LOOKOUT-JG](#89-lookout-jg)
  - [§8.10 B-PROBE-LOGIT](#810-b-probe-logit)
  - [§8.11 B-PROBE-CONSISTENCY](#811-b-probe-consistency)
  - [§8.12 ATTEST-WH](#812-attest-wh)
- [§9 CONDUCTOR (the Orchestrator)](#9-conductor-the-orchestrator)
- [§10 LADDER (the Tier Router)](#10-ladder-the-tier-router)
- [§11 ATTEST (the Integrity Gate)](#11-attest-the-integrity-gate)

### Part IV — The Hardware Tiers
- [§12 The 4 Tiers in Detail](#12-the-4-tiers-in-detail)

### Part V — The Five Hypotheses
- [§13 H1: The Cliff Exists](#13-h1-the-cliff-exists)
- [§14 H2: FPR Decoupling — White-Box](#14-h2-fpr-decoupling--white-box)
- [§15 H3: FPR Decoupling — Black-Box](#15-h3-fpr-decoupling--black-box)
- [§16 H4: Composition Gain](#16-h4-composition-gain)
- [§17 H5: Tier C Structural Weakness](#17-h5-tier-c-structural-weakness)
- [§18 Theorem 14.1 — FPR-Decoupling Theorem](#18-theorem-141--fpr-decoupling-theorem)

### Part VI — The Five Folds (Corpus Discipline)
- [§19 Fold Isolation Discipline](#19-fold-isolation-discipline)
- [§20 Fold A: Calibration](#20-fold-a-calibration)
- [§21 Fold B: Cliff Measurement (H1)](#21-fold-b-cliff-measurement-h1)
- [§22 Fold C: Defense Composition (H2, H3, H4)](#22-fold-c-defense-composition-h2-h3-h4)
- [§23 Fold D: Bandit Drift](#23-fold-d-bandit-drift)
- [§24 Fold E: BCN-2 Construction](#24-fold-e-bcn-2-construction)
- [§25 The Fold A → Others Dependency](#25-the-fold-a--others-dependency)

### Part VII — Implementation Status
- [§26 What Is Done](#26-what-is-done)
- [§27 What Is Not Done](#27-what-is-not-done)
- [§28 Test Coverage Map](#28-test-coverage-map)

### Part VIII — How to Operate the System
- [§29 Smoke Test (No GPU)](#29-smoke-test-no-gpu)
- [§30 Real Evaluation on a GPU Host](#30-real-evaluation-on-a-gpu-host)
- [§31 Reproducibility](#31-reproducibility)

### Part IX — Reference Material
- [§32 Glossary](#32-glossary)
- [§33 Reading Order for the Rest of the Docs](#33-reading-order-for-the-rest-of-the-docs)
- [§34 Where to Ask Questions](#34-where-to-ask-questions)

---

## Part I — Orientation

---

### §1 TL;DR

*One page that tells you everything you need before going deeper.*

**The problem in one sentence.** A language model that reliably refuses harmful requests when running at full float16 precision may silently comply with those same requests when quantized to 3-bit — not because it has "forgotten" how to reason, but because the geometric structure in its residual stream that encodes the refusal decision has been compressed away.

**Why this matters.** The majority of open-weight LLM deployments on consumer hardware use GGUF Q4_K_M or Q3_K_M quantization. Egashira et al. (ICLR 2025) showed that an attacker who knows a deployment uses these formats can achieve attack-success-rate deltas up to 88.7% compared to float16 targets. The standard defenses (Llama Guard, PromptGuard, NeMo Guardrails) were not designed for this failure mode and do not observe the residual-stream signal that is collapsing.

**What CLIFFGUARD does.** CLIFFGUARD is a stateless defense system placed *in front of* a quantized edge LLM. It does not modify the model, does not retrain anything, and does not require a GPU for the defense layer itself. It runs 12 gates organized into 8 named components:

- **VESTIBULE** (2 gates): inspect the raw input for adversarial structure before any inference
- **PROBE** (3 gates): read the model's internal geometry — the refusal and harmfulness directions in the residual stream
- **B-PROBE** (2 gates): black-box fallback using only top-k logprobs (no hidden-state access needed)
- **TRIPWIRE** (2 gates): monitor per-token entropy and reference-ratio during generation
- **LOOKOUT** (2 gates): inspect the completed output for canary leakage and judge compliance
- **ATTEST** (1 gate): at boot, hash the model weights against a vendor manifest
- **CONDUCTOR**: a LinUCB contextual bandit that adapts gate weights online from incident feedback
- **LADDER**: a static tier router that selects which gates run on each hardware class

**The key insight.** The false-positive rate of each write-side gate is *decoupled* from the quantization scheme — after a per-scheme calibration pass on benign traffic, each gate's FPR stays within ε = 0.02 of the target across FP16, NF4, AWQ-INT4, Q4_K_M, and Q3_K_M. The defense is portable across quantization levels without retraining. (TPR is *not* decoupled — that is precisely what the cliff regime measures.)

**Hardware tiers.** CLIFFGUARD runs on four hardware classes. Tier A (RTX 5060 8 GB) runs all 12 gates including the full PROBE stack and the Llama Guard 3 judge. Tier B (Raspberry Pi 5 8 GB) runs 11 gates, dropping the slow LLM judge. Tier C (2 GB embedded) runs only 3 gates — VESTIBULE-LZ, VESTIBULE-PS, and ATTEST-WH — and is explicitly *not* defended against the quantization-cliff exploiter. Tier C+ adds PromptGuard-2-22M-INT4 as a fourth gate, which is expected to show meaningful cliff-exploiter detection.

**Current status.** Phase A (939-test scaffolding of all 12 primitives and 8 components) is complete. Phase B (real inference-engine adapters, live `load_model()` paths, corpus loaders, judge drivers) has its scaffolding complete and live-mode code wired behind `load_model()` / `live_mode=True` flags. Phase C (actual real-hardware evaluation runs, Folds A–E) has not been executed — no real corpus exists on disk, no real GPU runs have been performed. All five hypotheses (H1–H5) are pre-registered in `docs/preregistration.md` before any data collection.

**What you get in 5 minutes.** Clone the repo, run `uv sync`, then `uv run python scripts/dry_run.py --tier A --scheme FP16`. All 12 gates fire on synthetic data in under 1 second on any machine, with no GPU and no data download. See [§29](#29-smoke-test-no-gpu) for annotated output.

---

### §2 The Safety Cliff Problem

*The problem that CLIFFGUARD was built to solve: post-training quantization creates a non-linear alignment failure near 3-bit precision.*

#### §2.1 Quantization, in 10 lines

Post-training quantization (PTQ) maps each float32 or float16 model weight to a lower-bit representation — typically 4-bit (Q4_K_M, NF4, AWQ-INT4) or 3-bit (Q3_K_M, IQ3_XXS) — to reduce memory footprint. A 7B-parameter model at float16 occupies roughly 14 GB of VRAM; at Q4_K_M it drops to roughly 4 GB; at Q3_K_M to roughly 3 GB. This is what makes 7B+ models viable on consumer GPUs with 8 GB VRAM or on CPU-only boards with 8 GB RAM.

The quantization error is small on average: MMLU benchmark scores typically drop only 6–8 points from FP16 to Q3_K_M. The model can still reason, still produce coherent text, still follow instructions for benign tasks. General capability degrades gracefully. But safety alignment does not degrade gracefully. The RLHF fine-tuning that teaches a model to refuse harmful requests is encoded in a narrow geometric structure in the residual stream — a near-one-dimensional subspace identified by Arditi et al. (arXiv:2406.11717). When quantization compresses the weights, this narrow subspace is disproportionately damaged because small direction-specific errors accumulate more than the averaging-out that protects general capability.

The result: a model that confidently refuses "How do I make a bioweapon?" at FP16 may produce a detailed, helpful response to the same question at Q3_K_M — not because it has been jailbroken, but because the refusal geometry has been eroded below the threshold needed to trigger the refusal response.

#### §2.2 What "the cliff" actually means

The cliff is not a gradual degradation — it is a *jump discontinuity* in the safety alignment metric as a function of quantization bit-width. CLIFFGUARD measures this jump with three independent metrics, all defined formally in [docs/math.md](math.md):

**Geometric cliff (Δ_cliff).** The cosine distance between the refusal direction extracted from the FP16 model and the refusal direction extracted from the quantized model. When a model quantizes without a cliff, these directions remain close (cosine ≈ 1). When the cliff strikes, the direction shifts discontinuously — the quantized model's "internal compass toward refusing" has rotated away.

**Behavioral cliff (Δ_B-cliff).** The attack-success-rate (ASR) difference between FP16 and the quantized scheme on a fixed adversarial corpus. This is the "ground truth" of safety failure — how many more attacks actually succeed. The cliff is where this jump ≥ κ = 0.25.

**Wasserstein cliff (Δ_W-cliff).** The Wasserstein-1 distance between the distribution of refusal-margin scalars under the FP16 model and the distribution under the quantized model. This provides distributional evidence of the shift, independent of the single-sample cosine distance.

The cliff hypothesis (H1, §13) requires that *both* the geometric and behavioral metrics agree: a jump ≥ κ = 0.25 must appear at the *same* quantization boundary in at least 2 of 3 tested model families. This agreement requirement prevents the geometric metric from being validated circularly by the behavioral metric.

#### §2.3 Why this matters

Every major local-LLM inference framework — Ollama, LM Studio, GPT4All, koboldcpp, llama-cpp-python — uses GGUF as its primary format and defaults to Q4_K_M or Q3_K_M for 7B+ models on consumer hardware with 8 GB of VRAM or RAM. The deployment surface for the cliff is thus approximately: *every consumer-grade local LLM installation running a model that barely fits in 8 GB*.

This is not a hypothetical population. As of late 2025, Ollama has millions of monthly active users; llama-cpp-python is among the most-starred LLM-adjacent Python libraries on GitHub. The typical configuration — Llama-3-8B-Instruct Q4_K_M on an RTX 3060 or RTX 4070, or Qwen2.5-7B Q3_K_M on a Raspberry Pi 5 — is precisely the cliff-risk regime.

The attacker's advantage: because the cliff is near Q3_K_M and the attacker can observe the deployment format (e.g., from model metadata in Ollama's API response), they know which prompts need to be "cliff-tuned" — crafted to produce compliance only at 3-bit. These prompts look benign to any surface-text classifier trained on FP16 behavior, because the FP16 model would refuse them. Standard content classifiers (trained on FP16-refused examples) therefore have near-zero TPR against a cliff-tuned adversary.

#### §2.4 The paint-on-foundation analogy

Imagine a building whose structural foundation is a 7-billion-brick edifice optimized for general functionality. Safety alignment is a coat of paint applied to the foundation after construction — a thin but important layer that marks certain doors as "do not open." The paint is RLHF fine-tuning: it teaches the model which outputs to refuse, encoded as directional biases in the residual stream.

Quantization is aggressive sandblasting. For most of the building, the sandblasting removes only a thin layer of dust — general capability survives largely intact, like MMLU dropping only 8 points. But the paint is thin. In the regions where the paint is thinnest — the refusal subspace, which Arditi et al. showed is approximately one-dimensional — the sandblasting removes it entirely. The doors that were marked "do not open" become indistinguishable from the rest of the wall.

CLIFFGUARD is a set of motion sensors mounted on those specific doors. It does not repaint the building — that would require retraining. Instead, it watches whether those doors are being opened and triggers an alarm before the person asking can walk through.

#### §2.5 The empirical phenomenon

The cliff is empirically localised near Q3_K_M / IQ3_XXS for the Llama-3 and Mistral model families. Blueprint §11.3 documents the evidence: Hong et al. (arXiv:2403.15447) found a roughly 50-point toxicity-safety drop at GPTQ-3-bit for Llama-2-13B-Chat while MMLU dropped only ~8 points. Egashira et al. (ICLR 2025) measured attack-success-rate deltas of Δ = 88.7% (insecure code generation), Δ = 85.0% (content injection), and Δ = 30.1% (refusal bypass) between FP16 and quantized GGUF targets.

The CLIFFGUARD cliff threshold κ = 0.25 is calibrated against these empirical measurements. A Δ_cliff jump of 0.25 corresponds to a cosine similarity of 1 − 2·(0.25)² = 0.875 between the FP16 and quantized refusal directions — a substantial but not catastrophic shift that is nonetheless sufficient for practical cliff exploitation. The 0.25 threshold is pre-registered and must not be adjusted after data collection begins.

**Note on geometric cliff range (decisions_log C27):** The `geometric_cliff` metric computed in `cliffguard/eval/cliff_metrics.py` has a natural range of [0, √2], not [0, 1] — antipodal unit vectors give ‖r̂_FP16 − r̂_q̃‖ = 2.0, and dividing by √2 = 1.414 normalises to [0, √2]. The code does not clamp to [0, 1]. Blueprint §11.3 must be updated to state "normalised to [0, √2]" before submission; this is an open paper-revision item.

---

### §3 What CLIFFGUARD Is (and Is Not)

*Precise scoping prevents misapplication.*

#### §3.1 What it is

- **A stateless, per-request defense system** placed in front of a quantized edge LLM. Every request is evaluated independently; no user payload is persisted across requests. The only state that survives across requests is the CONDUCTOR's bandit arm matrices and CUSUM/EWMA sketch counters over *features*, not text.

- **A quantization-aware safety observer.** Each gate's decision threshold is calibrated per quantization scheme (FP16, NF4, AWQ-INT4, Q4_K_M, Q3_K_M) independently. The calibration map is what makes the FPR portable — the same gate logic produces the same false-positive rate regardless of which quantization scheme is in use.

- **A multi-layer defense-in-depth system** with 12 independent signals: input structure, residual-stream geometry, streaming entropy, output canaries, compliance judges, and weight-hash attestation. No single layer is the primary defense; the CONDUCTOR aggregates all layers via a learned weight vector.

- **An honest, pre-registered evaluation subject.** All five hypotheses that constitute the paper's empirical contribution are stated in `docs/preregistration.md` before any data collection. The system's own weaknesses — Tier C's inability to detect cliff exploiters, the TPR gap of B-PROBE — are explicitly pre-registered as expected outcomes, not buried.

#### §3.2 What it is not

CLIFFGUARD is **not** a jailbreak corpus or attack dataset. It measures the cliff and defends against it; it does not generate attack prompts. The BCN-2 dataset (Fold E) is a *cliff proximity* dataset — prompts near the FP16 refusal boundary — not a jailbreak recipe book.

CLIFFGUARD is **not** a content filter. It does not classify input text as "harmful" based on surface patterns. The VESTIBULE gates detect adversarial *structure* (GCG suffixes, injected delimiters), not harmful *content*. The LOOKOUT-JG gate uses an external judge (Llama Guard 3) as one gate among twelve, not as the primary decision mechanism.

CLIFFGUARD is **not** a server-side guardrail for cloud LLMs. It is explicitly designed for the edge quantized regime — locally-served open-weight models where the inference engine is accessible. For closed-weight API endpoints, only the B-PROBE branch applies (top-k logprobs only, no hidden-state access). See [docs/what_is_it.md](what_is_it.md) for the full FAQ.

CLIFFGUARD is **not** a retraining or fine-tuning approach. It does not modify the protected model's weights. It is placed in front of the model and operates entirely on the model's *outputs* (logprobs, hidden states, generated text). SecAlign and similar alignment-repair approaches that fine-tune the model do not compose with CLIFFGUARD — they target the same failure mode but through a different mechanism.

#### §3.3 The 30-second pitch

You have an edge LLM running at Q3_K_M. The RLHF alignment that made it safe at FP16 is degraded — some fraction of harmful requests that the full-precision model would refuse now get through. You cannot retrain the model (no data, no GPU budget, no time). You cannot run a GPU-scale safety classifier (it would be as quantization-vulnerable as the model itself). What you can do: place a lightweight, stateless observer in front of the model that watches the model's own internal geometry and output patterns, with per-scheme calibration so the false-positive rate stays constant regardless of which quantization is in use. That observer is CLIFFGUARD.

---

## Part II — Threat Model

---

### §4 The 9 Adversary Classes A1–A9

*CLIFFGUARD models nine adversary classes covering the full range from natural-language jailbreaks to supply-chain poisoning, each with distinct primary detection mechanisms.*

The adversary taxonomy follows Greshake et al. (arXiv:2302.12173) extended for the edge-quantized setting (blueprint §2.2). Each class has distinct capabilities, attack mechanisms, and the primitives specifically designed to catch it.

| ID | Name | Knowledge | Mechanism | Canonical Example | Primary Gate(s) |
|---|---|---|---|---|---|
| **A1** | Direct injector | Prompt text only | Natural-language hijacks, role-play, DAN | "Ignore previous instructions and…" | VESTIBULE-PS, LOOKOUT-JG |
| **A2** | Indirect / poisoned-weight | RAG/tool access | Instructions in retrieved docs; Egashira-style GGUF blob substitution | Malicious PDF causing tool-output injection; tampered GGUF | VESTIBULE-PS, ATTEST-WH |
| **A3** | Optimizer | Black-box query access | GCG / AutoDAN adversarial suffixes (high perplexity, low compressibility) | 50-token GCG suffix appended to benign query | VESTIBULE-LZ, PROBE-RM |
| **A4** | Iterator | Black-box query access + feedback | PAIR / TAP / Crescendo iterative refinement across turns | Multi-turn escalation from innocuous to harmful | TRIPWIRE-H, LOOKOUT-JG |
| **A5** | Scaler | Black-box query access | Best-of-N sampling, many-shot, randomised augmentation exploiting sampling variance | 64 random variations of a borderline prompt, any success counts | TRIPWIRE-H, LOOKOUT-CT |
| **A6** | Encoder | Prompt text only | ArtPrompt ASCII art, bijection learning, base64/cipher, low-resource languages | "Write the following in ASCII art: [harmful word]" | TRIPWIRE-R, VESTIBULE-LZ |
| **A7** | Quantization-cliff exploiter | Knows deployment uses Q3_K_M | Natural-language prompts whose refusal margin collapses only at low bit-width | Prompt that FP16 refuses but Q3_K_M complies with naturally | LADDER + ATTEST + cross-tier rollback |
| **A8** | Defender-aware adversary | White-box access to CLIFFGUARD's bandit weights and calibration tables | Exploits bandit arm preferences; shifts attack distribution toward weakest gate | Observes which gates are down-weighted; routes all prompts through that gate | CONDUCTOR safe-rollback |
| **A9** | Closed-weight black-box endpoint | Top-k logprobs only | Probes closed API for first-token logprob patterns | Systematic logprob-based probing of OpenAI/Anthropic endpoint | B-PROBE-LOGIT, B-PROBE-CONSISTENCY |

**A1 — Direct injector.** The most common class, and the one all existing defenses cover. The attacker sends natural-language instructions embedded in the user turn that attempt to override system-prompt constraints: "ignore previous instructions," DAN-style persona hijacks, role-play frames, or persuasive jailbreaks. CLIFFGUARD catches A1 primarily at VESTIBULE-PS (structural injection patterns: role-override phrases, separator injection, chat-template boundary tokens) and at LOOKOUT-JG (compliance judge applied to the generated output). A1 is well-covered at Tier A and B, and partially covered at Tier C via VESTIBULE-PS. A1 prompts that avoid structural markers (clean natural-language persuasion) may bypass VESTIBULE-PS but are caught by LOOKOUT-JG if they result in compliant responses.

**A2 — Indirect injector / poisoned-weight attacker.** Two distinct attack surfaces share this class. The first is indirect injection: the attacker controls content that is retrieved and injected into the model's context via RAG pipelines, tool outputs, emails, or web pages (BIPIA, AgentDojo, InjecAgent). This content can contain injected instructions invisible to the user. VESTIBULE-PS detects structural injection markers in retrieved content. The second is Egashira-style weight poisoning: the attacker produces a GGUF file that is behaviorally benign at FP16 but malicious at Q3_K_M (the poisoning is encoded in the quantization-specific representation). ATTEST-WH defends against this by verifying the SHA-256 hash of the deployed weights at boot against a vendor manifest. If the weights have been substituted, ATTEST returns BLOCK (Tier A/B) or DEGRADED (Tier C/C+).

**A3 — Optimizer.** GCG (Zou et al., arXiv:2307.15043) and AutoDAN (Liu et al., arXiv:2310.04451) generate adversarial suffixes by gradient descent over the model's loss. These suffixes are highly structured: they have high token entropy (the model has no preference for any particular vocabulary item), low compressibility (they look like random token sequences), and high perplexity under any natural-language model. VESTIBULE-LZ catches these via the compression-ratio gate: a GCG suffix has a compression ratio near 1.0 (incompressible), which is anomalous relative to natural text. PROBE-RM catches the residual-stream consequence: even before the suffix causes a harmful output, the refusal margin drops as the hidden state is steered toward the harmful subspace.

**A4 — Iterator.** PAIR (Chao et al., arXiv:2310.08419), TAP (Mehrotra et al., arXiv:2312.02119), and Crescendo (Russinovich et al., arXiv:2404.01833) use iterative black-box queries to refine injection attempts — starting from a failed attempt and progressively modifying the prompt until the model complies. TRIPWIRE-H is the primary counter: it monitors per-token entropy during generation. A successful injection typically produces a low-entropy, scripted-sounding response (the model is "locked on" to a compliant trajectory); TRIPWIRE-H's CUSUM statistic accumulates this signal across tokens and fires when the cumulative entropy drop exceeds the calibrated threshold. LOOKOUT-JG provides post-generation confirmation.

**A5 — Scaler.** Best-of-N sampling and randomised augmentation exploit sampling variance: send enough variations of a borderline prompt and at least one will produce a compliant response by chance. TRIPWIRE-H detects the entropy drop associated with compliance in the specific winning sample. LOOKOUT-CT provides canary-token protection: if any sample leaks a canary token into the output, the bloom filter detects it immediately, regardless of which sample triggers it.

**A6 — Encoder.** ArtPrompt (Jiang et al., arXiv:2402.11753), bijection-learning attacks (Paulus et al., arXiv:2410.01294), base64 encoding, and low-resource-language jailbreaks all attempt to encode harmful requests in representations that surface classifiers and perplexity gates fail to recognize. These representations have anomalous token distributions compared to natural text: base64 uses a restricted alphabet; ASCII art uses whitespace and punctuation in patterns unlike natural prose; low-resource languages produce token sequences that are far from the KenLM reference model trained on English/multilingual benign traffic. TRIPWIRE-R's log-likelihood ratio against the benign KenLM reference fires LOW when the input is improbable under the benign distribution. VESTIBULE-LZ catches compression anomalies in structured encodings.

**A7 — Quantization-cliff exploiter.** The adversary that CLIFFGUARD specifically targets. The attacker knows the deployment uses Q3_K_M (observable from deployment metadata) and crafts prompts that are *natural language* — not adversarial suffixes, not encoded content — that an FP16 model would refuse but the Q3_K_M model complies with. These prompts exploit the refusal-direction collapse: at Q3_K_M, the refusal margin for certain prompt types drops below the threshold needed to trigger a refusal response, even for prompts that look innocuous to surface classifiers. PROBE-RM directly measures this margin collapse. LADDER routes to the appropriate tier's gate set; cross-tier rollback is the structural mitigation when the tier cannot fully defend against A7 (see Tier C honest scope, §12). Tier C is explicitly *not defended* against A7.

**A8 — Defender-aware adversary.** The attacker knows CLIFFGUARD is deployed and has white-box access to the bandit's arm weights and calibration tables. They can therefore identify which gate is currently down-weighted (CONDUCTOR reduces weight on a gate when it has accumulated false positives in recent traffic) and route attacks to target that gate. The CONDUCTOR's safe-rollback rule limits this: the MIN_WEIGHT = 0.1 floor on TRIPWIRE-R and ATTEST-WH means those gates can never be fully down-weighted. EXP3.S provides the minimax-regret fallback under coordinated campaigns. Full Kerckhoffs-level resistance is not claimed.

**A9 — Closed-weight black-box endpoint adversary.** When the deployment is a closed-weight API (OpenAI, Anthropic, Gemini), neither the hidden states nor the raw logit tensor is accessible — only top-k log-probabilities (typically k ≤ 20 as of 2025–2026). B-PROBE-LOGIT and B-PROBE-CONSISTENCY operate exclusively on this observable. The FPR-decoupling theorem has a black-box corollary (§15) that guarantees the same FPR portability for these gates. TPR is honestly weaker — B-PROBE has strictly lower TPR than PROBE-RM at matched FPR. This is pre-registered in H3.

---

### §5 Coverage Gaps

*Three structural limitations the blueprint honestly admits — not engineering deficiencies but fundamental scope constraints.*

#### §5.1 Multi-turn social engineering and cross-session state

CLIFFGUARD is stateless per request. It has no memory of previous turns beyond the CUSUM running statistic (which is a scalar aggregate over features, not a payload store). An attacker executing a Crescendo-style multi-turn escalation — each turn individually safe, the combination harmful — can in principle stay below CLIFFGUARD's per-request detection thresholds throughout. TRIPWIRE-H accumulates signal within a single generation sequence, but inter-turn context is not tracked (blueprint §2.7: "persistent storage of user payloads is forbidden by deployment constraints"). The CONDUCTOR's bandit state updates on reward signals, but reward signals require LOOKOUT to fire, which requires a single turn to be detectable. Cross-session attacks are entirely outside scope.

This is a deliberate design constraint, not an oversight. The deployment context — phones, kiosks, embedded gateways — forbids payload persistence for privacy reasons. A session-aware defense would require storing user interactions, which conflicts with the core deployment requirement.

#### §5.2 Supply-chain attacks below the GGUF/safetensors layer

ATTEST-WH hashes the GGUF or safetensors blob and compares to a vendor manifest. This defends against weight substitution at the file level (Egashira-style) but not against compromised inference engines, malicious CUDA drivers, hardware fault injection, or GGML kernel-level modifications. A compromised `libllama.so` could produce altered hidden states while passing the weight-hash check. Blueprint §2.7 explicitly excludes this: "supply-chain compromise of the inference engine itself or below the GGUF / safetensors layer."

Mitigating this gap would require attestation at the operating-system and driver level, which is hardware-specific (AMD Secure Boot, ARM TrustZone) and far outside the scope of an application-layer defense.

#### §5.3 Multimodal injection and non-text attack surfaces

CLIFFGUARD operates on text-token streams. Multimodal injection — embedding harmful instructions in images, audio, or video that the vision/audio encoder translates into text-equivalent hidden states — is explicitly out of scope (blueprint §2.7: "multimodal image/audio injection… is future work"). The VESTIBULE gates operate on raw text; the PROBE gates operate on residual-stream activations that are token-level; TRIPWIRE operates on per-token entropy. All of these are undefined for pure image or audio inputs.

---

### §6 Why Not Just Use a Content Classifier?

*The structural reasons that a content classifier alone cannot solve the quantization-cliff problem.*

The natural question when faced with a prompt injection problem is: "Why not just train a classifier to recognize harmful prompts?" Llama Guard, PromptGuard 2, NeMo Guardrails, Constitutional Classifiers, and AEGIS all follow this pattern. CLIFFGUARD uses LOOKOUT-JG (Llama Guard 3) as one gate among twelve, but relies on it as the primary mechanism only at Tier C+. The reasons a classifier-only approach fails in the edge quantized setting are structural:

**Bypass by rephrasing (A3, A4, A6).** A classifier trained on surface text patterns is evaded by any adversary who encodes harmful intent in a representation the classifier has not seen: GCG adversarial suffixes, ASCII art, ciphers, bijection learning, low-resource languages, iteratively refined natural language. The attack surface for a text classifier is approximately infinite — the adversary controls the input encoding. CLIFFGUARD's non-surface signals (compression ratio, residual-stream geometry, token entropy) are harder to manipulate without also removing the attack's effectiveness.

**Cliff degradation of the classifier itself.** A classifier deployed on edge hardware faces the same quantization problem as the model it is protecting. A 7B Llama Guard model quantized to Q3_K_M to fit in 2 GB may have its own refusal geometry degraded — the same cliff attack that exploits the base model's alignment collapse could, in principle, also fool a quantized Llama Guard. CLIFFGUARD's primary signals (PROBE-RM's cosine projection, VESTIBULE-LZ's stdlib zlib compression, TRIPWIRE-H's Shannon entropy) are either model-free or based on summary statistics that are less vulnerable to quantization-induced collapse.

**No geometry signal.** A content classifier assigns a label from surface text without observing the model's internal state. It cannot see that the refusal-direction cosine has dropped to 0.1 (meaning the model's hidden state is far from the refusal subspace) or that the harmfulness-direction projection has spiked (meaning the input has been processed as harmful). These geometry signals — which are the mechanistic *cause* of the cliff — are invisible to any text classifier. PROBE-RM and PROBE-HD read these signals directly.

The analogy: a content classifier looks at the *words* in a prompt. CLIFFGUARD looks at whether the *model's internal state* has crossed the refusal boundary. The words can be changed; the residual-stream response to the semantics cannot.

---

## Part III — The Defensive Architecture

---

### §7 The 8 Components, Top-Down

*Eight named components compose into a layered defense — think of them as eight quality gates on an assembly line, each checking a different property of the request-response pair.*

The diagram below shows all eight components and their data flow. Gates that fire when their score exceeds a threshold (fires-HIGH) are marked with [H]; gates that fire when their score falls below a threshold (fires-LOW) are marked with [L].



**VESTIBULE** () runs two fast, model-free gates over the raw input string before any inference. Both operate on text alone — no GPU, no model call, no tokenization. VESTIBULE-LZ measures compression-ratio anomaly (GCG/AutoDAN suffixes are incompressible). VESTIBULE-PS detects structural injection patterns (role-override phrases, separator injection, chat-template boundary tokens). At Tier C and C+, these are two of only three or four active gates — the front line for narrow-scope embedded deployment.

**PROBE** () is the quantization-aware white-box observer. It reads the model residual stream at two token positions: the last user-instruction token (t_inst, for PROBE-HD) and the first post-instruction token (t_post-inst, for PROBE-RM and PROBE-MT). PROBE-RM measures distance from the refusal direction. PROBE-MT measures the rate of change of that margin across decoding layers. PROBE-HD measures how strongly the input activates the harmfulness direction at t_inst. All three require white-box access to the model hidden states, which Phase B wires via the  interface in .

**B-PROBE** () is the black-box fallback observer for closed-weight endpoints and NPU-frozen graphs without hidden-state access. B-PROBE-LOGIT applies a trained logistic head to the top-k log-probabilities of the first response token. B-PROBE-CONSISTENCY computes the Jensen-Shannon divergence across the first-token distributions of N paraphrases — a compliant response to a harmful instruction is highly consistent across paraphrases (low JSD), whereas a refusal is more variable. Both provide FPR portability via per-scheme calibration, with acknowledged TPR loss (pre-registered in H3).

**TRIPWIRE** () runs during generation, token by token, without requiring the full output to be buffered. TRIPWIRE-H applies a lower CUSUM to the per-token Shannon entropy: a sustained entropy drop signals a compliant trajectory. TRIPWIRE-R computes the log-likelihood ratio of each output token against a KenLM reference model trained on benign traffic: anomalously improbable sequences signal adversarial content. Both gates can stop generation mid-stream.

**LOOKOUT** () operates on the completed output. LOOKOUT-CT checks for canary tokens using a SHA-256-backed Bloom filter as a fast pre-filter before exact substring match. LOOKOUT-JG applies a paraphrase consistency judge (Llama Guard 3) and scores the compliance rate. Tier A uses Llama Guard 3; Tier B drops this gate due to CPU inference latency.

**CONDUCTOR** () aggregates all gate verdicts into a single BLOCK/ALLOW decision via a weighted vote. Weights are set online by a LinUCB contextual bandit learning from sparse incident feedback. The 14-dimensional context vector encodes gate scores, the ATTEST result, and the tier indicator. No user payload is stored; only aggregate scalars survive across requests.

**LADDER** () is static configuration. It takes a hardware tier (A, B, C, C+) and returns the ordered list of active gate names. Routes between white-box and black-box paths depending on observability. Queried once per deployment, not per request.

**ATTEST** () runs once at boot and caches its result. Hashes the GGUF or safetensors model file with SHA-256, compares to a vendor-signed manifest. The result — ALLOW, DEGRADED, or BLOCK — is placed at CONDUCTOR context vector index 12 for every subsequent request during the session.

### §8 The 12 Primitives

*Each primitive is a self-contained gate with a public evaluate() function, a calibrated threshold per quantization scheme, and a well-defined firing direction.*

The 12 primitives are organized into six component groups. Every `evaluate()` function returns a `GateVerdict` (and optionally a `Margin`), both defined in `cliffguard/types.py`. The `CalibrationTable` provides the per-scheme threshold `tau_q` via `calibration.tau(scheme)`.

---

#### §8.1 VESTIBULE-LZ

*Measures the compression ratio of the input text as a proxy for lexical entropy — high compression ratio (incompressible text) flags adversarial suffixes.*

**Analogy.** A randomness detector at the post office: legitimate letters compress well because words repeat and grammar patterns compact; a GCG adversarial suffix compresses like random bytes because no pattern repeats. The post office flags incompressible envelopes for inspection.

**What it actually does.** The gate computes `rho_LZ(x) = len(zlib.compress(x.encode())) / len(x.encode())`. For natural prose this ratio is roughly 0.3–0.6. For GCG adversarial suffixes generated by gradient descent, the ratio approaches 1.0 — the optimizer has found token sequences that are locally optimal for the attack loss but are structurally unlike any natural text pattern. The gate fires when `rho_LZ > tau_q`. Because zlib is a Python standard library function, VESTIBULE-LZ has zero setup cost and costs roughly 20 µs per 1 KB (blueprint §5.6).

**Formula (compact).**
```
rho_LZ(x) = len(zlib.compress(x.encode("utf-8"))) / len(x.encode("utf-8"))
fired = rho_LZ(x) > tau_q    [fires-HIGH]
```

**Firing direction.** Fires-HIGH — fires when `rho_LZ(x)` exceeds the threshold.

**Inputs / outputs.**
- Input: `text: str`, `calibration: CalibrationTable`, `scheme: QuantScheme`, `tier: Tier`
- Output: `GateVerdict(gate="VESTIBULE-LZ", fired: bool, score: float, threshold: float, tier: Tier, threat_model: None)`

**Where implemented.** `cliffguard/vestibule/lz.py:17-53`. Core functions: `compression_ratio(text)` (lines 17–26); `evaluate(...)` (lines 29–53).

**Where tested.** `tests/test_vestibule_lz.py` — 12 tests. Asserts: compression ratio of natural text < 1.0; ratio of random bytes close to 1.0; empty input returns 1.0; evaluate fires above threshold; GateVerdict fields correctly populated.

**Tier availability.** A, B, C, C+ — active on all tiers (`cliffguard/ladder/tier.py:TIER_GATES`).

**Failure modes.** Adversaries who craft GCG-style suffixes with artificially repeated token patterns can bring the compression ratio into the benign range. Very short inputs (< 20 characters) are noisy because the zlib header dominates. Natural-language cliff exploiters (A7) produce grammatical text that compresses normally — not detected.

---

#### §8.2 VESTIBULE-PS

*Detects structural injection patterns in the input text — role-override phrases, separator injection, and chat-template boundary tokens — using compiled regular expressions.*

**Analogy.** A document examiner checking for forged letterhead: legitimate users do not need to write "ignore previous instructions" or inject `[INST]` tokens mid-message. The examiner knows what legitimate documents look like and flags any with forged structure markers.

**What it actually does.** The gate implements the spotlighting defense concept of Hines et al. (arXiv:2403.14720). It computes a weighted signal score from three regex-based pattern classes:

1. **Role-override phrases** (weight 1.0): "ignore previous instructions," "act as," "you are now," "pretend you are," "forget your instructions." Pattern compiled with `re.IGNORECASE`.
2. **Separator injection** (weight 0.5): four or more repetitions of the same punctuation character (`----`, `====`, `####`, etc.), mimicking system-prompt delimiters.
3. **Boundary markers** (weight 1.0): `[INST]`, `[/INST]`, `<|im_start|>`, `<|im_end|>`, `### Instruction`, `### Response` — chat-template tokens appearing inside a user message.

The aggregate signal score is the weighted sum of match counts. The gate fires when `signal_score(x) > tau_q`.

**Formula (compact).**
```
signal_score(x) = 1.0 * count(role_override, x)
                + 0.5 * count(separator, x)
                + 1.0 * count(boundary, x)
fired = signal_score(x) > tau_q    [fires-HIGH]
```

**Firing direction.** Fires-HIGH — fires when `signal_score(x)` exceeds the threshold.

**Inputs / outputs.**
- Input: `text: str`, `calibration: CalibrationTable`, `scheme: QuantScheme`, `tier: Tier`
- Output: `GateVerdict(gate="VESTIBULE-PS", fired: bool, score: float, threshold: float, ...)`
- Intermediate: `count_signals(text)` returns `{"role_override": int, "separator": int, "boundary": int}`.

**Where implemented.** `cliffguard/vestibule/ps.py:1-99`. Three compiled regex patterns (lines 21–44); `count_signals(text)` (lines 52–58); `signal_score(text)` (lines 62–72); `evaluate(...)` (lines 75–99).

**Where tested.** `tests/test_vestibule_ps.py` — 19 tests. Asserts: clean benign text scores 0.0; role-override phrases detected case-insensitively; separator patterns require 4+ repetitions of the same character; boundary tokens detected; evaluate fires above threshold.

**Tier availability.** A, B, C, C+ — active on all tiers.

**Failure modes.** A4 (iterator) adversaries who gradually escalate without structural markers will not be flagged. A7 (cliff exploiter) uses clean natural language without injection markers. Subtle semantic persuasion without explicit role-override phrasing is not caught.

---

#### §8.3 PROBE-RM

*Measures the cosine similarity between the model's hidden state at the post-instruction token and the pre-computed refusal direction — a direct read of whether the model is "pointing toward refusing."*

**Analogy.** A compass that points toward "refuse." In a healthy FP16 model, the compass points consistently toward refusal when processing harmful requests. After cliff-level quantization, the compass has been demagnetised — it no longer points reliably in any direction. PROBE-RM reads the compass.

**What it actually does.** Following Arditi et al. (arXiv:2406.11717), the refusal direction `r_hat` is a unit vector in the model's residual-stream space computed during Fold A calibration. The refusal margin is the cosine projection of the hidden state `z_l(t_post-inst)` onto `r_hat`:

```
m_r = dot(r_hat, z_l(t_post-inst)) / (||r_hat|| * ||z||)
```

A high margin (≈ 1.0) means the model is in a refusing posture. A low margin (≈ 0.0 or negative) means it is not primed to refuse. The gate fires LOW: it fires when `m_r < tau_q`. In Phase A, the gate operates on synthetic numpy arrays. In Phase B, it requires `HiddenStateAdapter.get_hidden_states(prompt, layer)` from the engine adapter.

**Formula (compact).**
```
m_r = dot(r_hat, z_l) / (||r_hat|| * ||z||)
fired = m_r < tau_q    [fires-LOW]
```

**Firing direction.** Fires-LOW — a *low* margin is the risk signal.

**Inputs / outputs.**
- Input: `hidden_state: ndarray[float64, shape=(d,)]`, `refusal_direction: ndarray[float64, shape=(d,)]`, `calibration: CalibrationTable`, `scheme: QuantScheme`, `tier: Tier`
- Output: `tuple[Margin, GateVerdict]`
- `Margin(value=m_r, scheme=scheme, primitive="PROBE-RM", layer=None)`
- Zero-norm inputs raise `ValueError`.

**Where implemented.** `cliffguard/probe/rm.py:1-76`. `compute_margin(hidden_state, refusal_direction)` (lines 25–40); `evaluate(...)` (lines 43–76).

**Where tested.** `tests/test_probe_rm.py` — 17 tests. Asserts: zero-norm inputs raise `ValueError`; unit vectors in same direction give margin 1.0; orthogonal vectors give 0.0; evaluate fires when margin < threshold; Margin.primitive = "PROBE-RM".

**Tier availability.** A, B — requires white-box hidden-state access. Not active at Tier C or C+.

**Failure modes.** Requires white-box access — unavailable at closed-weight API deployments (A9) and Tier C/C+. Refusal direction must be calibrated per (model, scheme) pair. In the cliff regime, the direction itself may become unreliable — this is exactly what H1 tests.

---

#### §8.4 PROBE-MT

*Tracks the rate of change of the refusal margin across decoding steps — firing when the margin is actively falling, signalling a model drifting toward compliance mid-response.*

**Analogy.** Not just a position readout but a speedometer: PROBE-RM tells you where the compass points now; PROBE-MT tells you whether it is rotating away from "refuse" and how fast.

**What it actually does.** As the model generates response tokens, the refusal margin can be computed at each decoding step. PROBE-MT takes a sequence of margin values `[rho_1, rho_2, ..., rho_K_l]` (K_l = 3 layers per blueprint §5.2) and computes:

- `rho_dot` = mean of first differences: `mean(rho[1:] - rho[:-1])`
- `rho_ddot` = mean of second differences (acceleration of the drift)

A sustained negative `rho_dot` (margin consistently falling) is the primary firing signal. The gate fires LOW: `rho_dot < tau_q`. Note: `rho_ddot` is computed in `compute_trajectory()` but is surfaced separately to the CONDUCTOR context vector at index 4 via `build_context(probe_mt_rho_ddot=rho_ddot)` — a Phase A gate decision documented in `decisions_log.md` and `context.py:14`.

**Formula (compact).**
```
rho_dot  = mean(rho[1:] - rho[:-1])
rho_ddot = mean(diff(rho[1:] - rho[:-1]))
fired = rho_dot < tau_q    [fires-LOW]
```

**Firing direction.** Fires-LOW — fires when `rho_dot` is sufficiently negative.

**Inputs / outputs.**
- Input: `margins: ndarray[float64, shape=(N,)]` with N ≥ 3, `calibration: CalibrationTable`, `scheme: QuantScheme`, `tier: Tier`
- Output: `tuple[Margin, GateVerdict]`; N < 3 raises `ValueError`.
- CONDUCTOR context uses `rho_dot` at index 3 and `rho_ddot` at index 4.

**Where implemented.** `cliffguard/probe/mt.py:1-69`. `compute_trajectory(margins)` (lines 19–33); `evaluate(...)` (lines 36–69).

**Where tested.** `tests/test_probe_mt.py` — 19 tests. Asserts: N < 3 raises `ValueError`; flat sequence gives rho_dot ≈ 0.0; falling sequence gives negative rho_dot; evaluate fires when rho_dot < threshold.

**Tier availability.** A, B — requires white-box hidden-state access across multiple decoding steps. Not active at Tier C or C+.

**Failure modes.** Computationally expensive — requires multiple forward passes. If compliance is achieved via a single sudden step (not gradual drift), the averaged rho_dot may not fire. Requires same white-box access as PROBE-RM.

---

#### §8.5 PROBE-HD

*Measures the cosine similarity between the model's hidden state at the user-instruction token and the pre-computed harmfulness direction — indicating whether the input is being processed as harmful.*

**Analogy.** A toxicity sensor at the front door, distinct from the refusal compass inside. PROBE-RM reads the model's *response posture*; PROBE-HD reads the model's *recognition of harm* in the input itself.

**What it actually does.** Following Zhao et al. (arXiv:2507.11878, NeurIPS 2025), the harmfulness direction `h_hat` is a unit vector in the residual-stream space separating harmful from benign prompt representations at the *user-instruction token* `t_inst` — demonstrably distinct from the refusal direction `r_hat` at `t_post-inst`. The harmfulness margin is the cosine projection of `z_l(t_inst)` onto `h_hat`:

```
m_h = dot(h_hat, z_l(t_inst)) / (||h_hat|| * ||z||)
```

The gate fires HIGH: `m_h > tau_q`. The highest-risk configuration is high `m_h` (input processed as harmful) combined with low PROBE-RM margin (model not in refusing posture) — precisely the cliff regime.

**Formula (compact).**
```
m_h = dot(h_hat, z_l(t_inst)) / (||h_hat|| * ||z||)
fired = m_h > tau_q    [fires-HIGH]
```

**Firing direction.** Fires-HIGH — fires when harmfulness margin exceeds threshold. A *high* margin is the risk signal. Opposite to PROBE-RM.

**Inputs / outputs.**
- Input: `hidden_state: ndarray[float64, shape=(d,)]` at t_inst, `harmfulness_direction: ndarray[float64, shape=(d,)]`, `calibration`, `scheme`, `tier`
- Output: `tuple[Margin, GateVerdict]`; zero-norm inputs raise `ValueError`.

**Where implemented.** `cliffguard/probe/hd.py:1-78`. `compute_harmfulness_margin(hidden_state, harmfulness_direction)` (lines 23–39); `evaluate(...)` (lines 42–78).

**Where tested.** `tests/test_probe_hd.py` — 18 tests. Asserts: same-direction vectors give margin 1.0; evaluate fires when margin > threshold (HIGH direction); Margin.primitive = "PROBE-HD".

**Tier availability.** A, B — requires white-box hidden-state access. Not active at Tier C or C+.

**Failure modes.** Same white-box access requirement as PROBE-RM. The harmfulness direction must be calibrated separately using paired harmful/benign prompts at t_inst.

---

#### §8.6 TRIPWIRE-H

*Applies a one-sided lower CUSUM to the per-token Shannon entropy of the model's output distribution — firing when a sustained entropy drop signals a compliant, scripted response trajectory.*

**Analogy.** A wire stretched across a doorway that accumulates tension whenever the model's output becomes too "certain." A model generating refusal text is highly uncertain about which words to use; a model locked onto a compliant answer generates with low entropy — the output is scripted. The wire accumulates tension token by token; when tension exceeds the threshold, the alarm fires.

**What it actually does.** At each generation step, Shannon entropy `H_t = -sum_v p(v) * log2(p(v))` is computed from the token distribution. TRIPWIRE-H runs a one-sided lower CUSUM with reference parameter k (default k = 0.5 bits):

```
S_t = max(0, S_{t-1} - (H_t - k))
```

When H_t < k, the statistic increases; when H_t > k it decreases toward zero. The alarm fires when the final S_T > tau_q. Target ARL_0 ≈ 1000 on benign traffic (blueprint §5.4). The CUSUM was chosen over single-token thresholding because compliance requires *sustained* low entropy, not a brief dip.

**Formula (compact).**
```
S_t = max(0, S_{t-1} - (H_t - k))
fired = S_T > tau_q    [fires-HIGH]
```

**Firing direction.** Fires-HIGH — fires when the final CUSUM statistic exceeds threshold.

**Inputs / outputs.**
- Input: `entropies: ndarray[float64, shape=(T,)]`, `calibration`, `scheme`, `tier`, `k: float = 0.5`
- Output: `GateVerdict(gate="TRIPWIRE-H", ...)`. Empty sequence raises `ValueError`.

**Where implemented.** `cliffguard/tripwire/h.py:1-81`. `token_entropy(logprobs)` (lines 23–33); `cusum_statistic(entropies, k)` (lines 37–52); `evaluate(...)` (lines 55–81).

**Where tested.** `tests/test_tripwire_h.py` — 19 tests. Asserts: empty raises `ValueError`; constant entropy gives CUSUM ≈ 0; step-down causes CUSUM to accumulate; evaluate fires above threshold.

**Tier availability.** A, B — requires per-token logprob stream. Not active at Tier C or C+.

**Failure modes.** Does not fire if compliance is achieved in a single high-entropy token. A4 adversaries may observe this gate and craft prompts that produce high-entropy compliant responses (difficult in practice). Requires real per-token logprobs in Phase B.

---

#### §8.7 TRIPWIRE-R

*Computes the mean per-token log-likelihood ratio between the model's output and a KenLM reference model trained on benign traffic — firing when the output is anomalously improbable under the benign distribution.*

**Analogy.** A spell-checker for statistical normality. The KenLM model has memorized what benign traffic looks like token-by-token. An adversarially encoded input (A6: ASCII art, ciphers, low-resource languages) produces token sequences that the benign model has never seen. The log-likelihood ratio reports how surprised the benign model is.

**What it actually does.** For each input token, the gate computes:
```
LLR_t = log P_model(y_t | ctx) - log P_ref(y_t | ctx)
```
The mean LLR over the token sequence is the gate's score. A negative mean LLR means the input is less probable under the benign reference — signal of adversarial encoding. The gate fires LOW: `mean_LLR < tau_q`.

The KenLM reference model is trained in Fold A on the benign calibration corpus via `eval/kenlm_trainer.py`. KenLM order is 5 for Tier A/B, 3 for Tier C/C+ (lower memory budget — decisions_log C25). This gate is in the NEVER_DISABLE list in the CONDUCTOR alongside ATTEST-WH; its weight never falls below `MIN_WEIGHT = 0.1`.

**Formula (compact).**
```
LLR = mean(log_P_model(y_t) - log_P_ref(y_t))  for t in 1..T
fired = LLR < tau_q    [fires-LOW]
```

**Firing direction.** Fires-LOW — fires when the mean LLR falls below threshold.

**Inputs / outputs.**
- Input: `input_logprobs: ndarray[float64, shape=(T,)]`, `benign_logprobs: ndarray[float64, shape=(T,)]`, `calibration`, `scheme`, `tier`
- Output: `GateVerdict(gate="TRIPWIRE-R", ...)`. Empty or mismatched arrays raise `ValueError`.

**Where implemented.** `cliffguard/tripwire/r.py:1-70`. `log_likelihood_ratio(input_logprobs, benign_logprobs)` (lines 22–41); `evaluate(...)` (lines 44–70).

**Where tested.** `tests/test_tripwire_r.py` — 19 tests; `tests/test_tripwire_r_calibration.py` — 8 tests. Asserts: empty/mismatched arrays raise `ValueError`; equal logprob arrays give LLR ≈ 0; lower benign logprobs give negative LLR (fires); calibration tests verify per-scheme threshold application.

**Tier availability.** A, B — requires KenLM binary and trained ARPA file. Not active at Tier C or C+.

**Failure modes.** Requires KenLM binary (`lmplz`) installed; if absent, `kenlm_trainer.py` raises `NotImplementedError`. Natural-language cliff exploiters (A7) produce text that is highly probable under the benign KenLM — this gate does not detect them.

---

#### §8.8 LOOKOUT-CT

*Checks the model's output for leakage of per-session canary tokens — short random strings injected into the system prompt that should never appear in model output.*

**Analogy.** A bouncer with a memorized guest list. At session start, the bouncer memorizes a set of random codewords (canary tokens) placed in the system prompt. If the model shouts one of those codewords in its output, the bouncer immediately knows the model has been coerced into echoing injected content.

**What it actually does.** At session initialization, short random strings (canary tokens) are injected into the system prompt in a hidden position. After generation, LOOKOUT-CT scans the output using a SHA-256-backed Bloom filter (m=256, k=3) as a fast O(k) pre-filter, then confirms any hit with exact substring match. The gate fires HIGH when the count of matched canary tokens exceeds `tau_q`.

**Formula (compact).**
```
score = count(token in output for token in canary_tokens where bloom_filter confirms)
fired = score > tau_q    [fires-HIGH]
```

**Firing direction.** Fires-HIGH — fires when any canary token is found in the output.

**Inputs / outputs.**
- Input: `output_text: str`, `canary_tokens: list[str]`, `calibration`, `scheme`, `tier`
- Output: `GateVerdict(gate="LOOKOUT-CT", score=float, ...)`. Score = number of matched canary tokens.

**Where implemented.** `cliffguard/lookout/ct.py:1-93`. `BloomFilter` class (lines 17–40); `build_canary_filter(canary_tokens)` (lines 43–48); `check_output(output_text, canary_filter, canary_tokens)` (lines 51–64); `evaluate(...)` (lines 67–93).

**Where tested.** `tests/test_lookout_ct.py` — 29 tests. Asserts: Bloom filter has no false negatives on added items; exact-match confirmation prevents false positives; empty canary list fires nothing; evaluate score equals match count.

**Tier availability.** A, B — per `cliffguard/ladder/tier.py:TIER_GATES`. (Tier C/C+ do not include LOOKOUT-CT in their gate sets.)

**Failure modes.** A defender-aware attacker (A8) who knows the canary tokens can avoid echoing them while still complying. Canaries must be session-unique and kept secret. Gate cannot fire if canary injection at setup time is omitted.

---

#### §8.9 LOOKOUT-JG

*Applies a paraphrase-consistency compliance judge (Llama Guard 3) to the model's output — firing when multiple paraphrases of the output are all judged as compliant with a harmful instruction.*

**Analogy.** A second-opinion doctor who sends three copies of the report to three specialists and checks whether they all diagnose the same condition. A genuinely harmful compliant response is consistently identified across rephrasing; a borderline response is not.

**What it actually does.** Following SmoothLLM, LOOKOUT-JG paraphrases the model's output N times (N ∈ {2, 3}; default N=3), then applies the judge stack to each paraphrase. The compliance rate — the fraction judged compliant — is the gate's score. The gate fires HIGH when `compliance_rate > tau_q`. The judge stack uses StrongREJECT + Llama Guard 3 in Phase B. In Phase A, `Paraphraser` and `JudgeStack` are Protocol interfaces with stub implementations.

**Formula (compact).**
```
compliance_rate = count(judge.is_compliant(p) for p in paraphraser.paraphrase(response, n)) / n
fired = compliance_rate > tau_q    [fires-HIGH]
```

**Firing direction.** Fires-HIGH — fires when the compliance rate exceeds threshold.

**Inputs / outputs.**
- Input: `response: str`, `paraphraser: Paraphraser`, `judge: JudgeStack`, `calibration`, `scheme`, `tier`, `n: int = 3`
- Output: `GateVerdict(gate="LOOKOUT-JG", ...)`. n < 1 raises `ValueError`.

**Where implemented.** `cliffguard/lookout/jg.py:1-79`. `Paraphraser` Protocol (lines 20–26); `JudgeStack` Protocol (lines 29–34); `compliance_rate(...)` (lines 39–52); `evaluate(...)` (lines 55–79). Real implementations: `cliffguard/eval/judges.py:RealLlamaGuardJudge`, `RealStrongREJECTJudge`.

**Where tested.** `tests/test_lookout_jg.py` — 17 tests. Asserts: n < 1 raises `ValueError`; all-compliant judge gives rate 1.0; all-refusing gives 0.0; evaluate fires above threshold.

**Tier availability.** A only — Llama Guard 3 inference is too slow for CPU-only Tier B. Per `cliffguard/ladder/tier.py:TIER_GATES`.

**Failure modes.** Slow: requires N+1 LLM forward passes. Does not fire before the output is complete. Note: `judges.py` docstring cites §5.9 for this gate, but §5.9 in the blueprint is the runtime gate section, not the judge stack (§11.3, §12.6 are the relevant blueprint sections) — paper section numbering audit pending (decisions_log C26).

---

#### §8.10 B-PROBE-LOGIT

*Applies a trained logistic head to the top-k log-probability vector of the model's first response token — the black-box fallback for detecting compliance risk without hidden-state access.*

**Analogy.** An X-ray machine that cannot see inside the body but can detect suspicious density patterns from the outside. The first-token logprob vector is a coarse projection of the internal state — enough to distinguish some harmful trajectories.

**What it actually does.** The top-k log-probability vector `ell(q) ∈ R^k` is directly observable from any API endpoint providing logprob outputs. B-PROBE-LOGIT applies a logistic head:

```
score = sigmoid(dot(theta, ell(q)) + bias)
```

The weight vector `theta` is trained during Fold A calibration on paired (benign, harmful) examples. The gate fires HIGH when `score > tau_q`. Per-scheme calibration of `tau_q` provides FPR portability (Corollary 14.2). For Tier C+, B-PROBE-LOGIT uses PromptGuard-2-22M-INT4 (22M-parameter DeBERTa-xsmall, MIT-licensed) as the classifier rather than a simple linear head.

**Formula (compact).**
```
score = sigmoid(dot(theta, ell(q)) + bias)
fired = score > tau_q    [fires-HIGH]
```

**Firing direction.** Fires-HIGH — fires when the logistic score exceeds threshold.

**Inputs / outputs.**
- Input: `logprobs: ndarray[float64, shape=(k,)]`, `weights: ndarray[float64, shape=(k,)]`, `calibration`, `scheme`, `tier`, `bias: float = 0.0`
- Output: `tuple[Margin, GateVerdict]`
- `Margin(value=score, primitive="B-PROBE-LOGIT", layer=None)`. Shape mismatch raises `ValueError`.

**Where implemented.** `cliffguard/bprobe/logit.py:1-73`. `sigmoid(x)` (lines 24–28); `logistic_score(logprobs, weights, bias)` (lines 31–43); `evaluate(...)` (lines 46–73).

**Where tested.** `tests/test_bprobe_logit.py` — 26 tests. Asserts: shape mismatch raises `ValueError`; score is in (0.0, 1.0); evaluate fires above threshold; fires-HIGH direction confirmed.

**Tier availability.** A, B, C+ — active where B-PROBE is included. Note Tier C (basic) does NOT include B-PROBE-LOGIT; Tier C+ does.

**Failure modes.** The logistic head is the most fragile gate against A8 (defender-aware): once `theta` is known, prompts can minimize the score while eliciting compliance. TPR is honestly weaker than PROBE-RM (pre-registered in H3).

---

#### §8.11 B-PROBE-CONSISTENCY

*Measures the Jensen-Shannon divergence between the first-token log-probability distributions of N paraphrases of the input — firing when the distributions are anomalously consistent, signalling scripted compliance.*

**Analogy.** A variation detector: ask the same question N ways and see how varied the model's initial response probability is. A refusing model is highly uncertain about how to start; a compliant model starts the same way regardless of phrasing.

**What it actually does.** For N paraphrases of the input, the gate collects the top-k log-probability vector `ell_i(q)` for each. The Jensen-Shannon divergence across these N distributions is:

```
JSD = H(mean_i P_i) - (1/N) * sum_i H(P_i)
```

where `P_i = softmax(ell_i(q))`. JSD = 0 means all N distributions are identical; JSD = log(N) means maximally different. The gate fires LOW: low JSD = high consistency = scripted compliant responses. This is the black-box equivalent of PROBE-MT operating at the distributional level.

**Formula (compact).**
```
JSD = H(mean_i softmax(ell_i)) - mean_i H(softmax(ell_i))
fired = JSD < tau_q    [fires-LOW]
```

**Firing direction.** Fires-LOW — fires when JSD falls below threshold.

**Inputs / outputs.**
- Input: `logprob_matrix: ndarray[float64, shape=(N, V)]` where N ≥ 2, V ≥ 1, `calibration`, `scheme`, `tier`
- Output: `tuple[Margin, GateVerdict]`. N < 2 or V < 1 raises `ValueError`.
- JSD is in nats, bounded [0.0, log(N)].

**Where implemented.** `cliffguard/bprobe/consistency.py:1-95`. `_log_softmax(logprobs)` (lines 22–26); `js_divergence(logprob_matrix)` (lines 29–59); `evaluate(...)` (lines 62–95).

**Where tested.** `tests/test_bprobe_consistency.py` — 25 tests. Asserts: N < 2 raises `ValueError`; identical rows give JSD ≈ 0; evaluate fires when JSD < threshold; JSD is non-negative.

**Tier availability.** A, B — requires N API/model calls per prompt. Not active at Tier C; Tier C+ uses only B-PROBE-LOGIT.

**Failure modes.** Computationally expensive: N forward passes or API calls per request. An adversary who produces high-JSD compliant responses can evade this gate.

---

#### §8.12 ATTEST-WH

*Hashes the on-disk model weights at boot and compares against a vendor manifest — defending against Egashira-style supply-chain weight poisoning before any inference begins.*

**Analogy.** A tamper-evident seal on a shipping container: if the weights have been modified since the vendor computed the hash — even by a single byte — the seal is broken. If the manifest is missing, the system continues in degraded mode rather than refusing to operate.

**What it actually does.** At boot time (once, not per request), ATTEST-WH reads the GGUF or safetensors blob in 64 KB chunks and computes its SHA-256 hash. It loads the vendor manifest (JSON mapping filename → expected SHA-256 hex digest) and compares. Three outcomes:

- **ALLOW** (hash matches manifest): proceed normally.
- **DEGRADED** (no manifest entry for this filename): continue with heightened alert status; CONDUCTOR context index 12 = 0.5.
- **BLOCK** (hash mismatch, Tier A/B): hard block; CONDUCTOR context index 12 = 0.0. For Tier C/C+, mismatch returns DEGRADED rather than BLOCK (embedded boards may lack signed manifests).

The ATTEST result is cached for the session. ATTEST-WH is in the NEVER_DISABLE list alongside TRIPWIRE-R; its weight never falls below `MIN_WEIGHT = 0.1`.

**Firing direction.** N/A — ATTEST produces a categorical `AttestResult` enum (ALLOW/DEGRADED/BLOCK), not a continuous score.

**Inputs / outputs.**
- `hash_file(path: Path) -> str`: SHA-256 hex digest by streaming the file.
- `load_manifest(manifest_path: Path) -> dict[str, str]`: parses vendor manifest JSON.
- `attest(model_path: Path, manifest_path: Path, tier: Tier) -> AttestResult`: main entry point.
- `FileNotFoundError` if model or manifest file does not exist.

**Where implemented.** `cliffguard/attest/wh.py:1-91`. `AttestResult` enum (lines 23–26); `hash_file(path, chunk_size=65536)` (lines 29–42); `load_manifest(manifest_path)` (lines 45–61); `attest(model_path, manifest_path, tier)` (lines 67–91).

**Where tested.** `tests/test_attest_wh.py` — 16 tests. Asserts: ALLOW on matching hashes; BLOCK for Tier A/B on mismatch; DEGRADED for Tier C/C+ on mismatch; DEGRADED when manifest has no entry; `FileNotFoundError` for missing model or manifest.

**Tier availability.** A, B, C, C+ — active on all tiers.

**Failure modes.** Does not defend against compromised inference engines or CUDA drivers. Manifest must be distributed securely. Only hashes the weight file — not configuration files, tokenizer files, or tool plugins.

---


### §9 CONDUCTOR (the Orchestrator)

*The LinUCB contextual bandit that aggregates all gate verdicts into a single BLOCK/ALLOW decision and adapts gate weights online from sparse incident feedback.*

**Analogy.** Imagine an orchestra conductor who can hear which instruments are useful in different rooms — quiet libraries need different mics than concert halls. The CONDUCTOR does not change the instruments (the gates); it adjusts how loud each instrument's contribution is heard, based on which ones have been useful in recent requests.

#### §9.1 LinUCB Intuition

LinUCB (Li et al., WWW 2010, arXiv:1003.0146) is a contextual bandit algorithm that balances exploration (trying arms that haven't been well-tested in the current context) with exploitation (using arms that have historically performed well). For CLIFFGUARD, each "arm" is a gate, and the "context" is the 14-dimensional feature vector assembled from gate scores. The UCB score for arm a given context x is:

```
UCB(a, x) = x^T * A_a^{-1} * b_a   +   alpha * sqrt(x^T * A_a^{-1} * x)
```

where `A_a` is the d×d covariance matrix (initialized to identity) and `b_a` is the d-dimensional reward-weighted context sum (initialized to zero). The first term is the estimated expected reward; the second term is the exploration bonus. Alpha controls the exploration-exploitation tradeoff (default alpha = 1.0).

When an incident occurs (LOOKOUT fires, user complaints, canary trip), the CONDUCTOR updates the arm that was responsible:

```python
arm.A += outer(x, x)    # rank-1 update to covariance
arm.b += reward * x     # reward-weighted context update
```

The reward signal is: +1.0 for a correct block (attack caught), -1.0 for a miss (attack not caught), -0.2 for a false positive (benign request blocked). No raw user payload is stored; only the aggregate context vector x survives.

The aggregate verdict is a weighted vote: if `fired_score / total_weight > 0.5`, the request is blocked. Where `fired_score = sum(w_a for a in fired_gates)` and `total_weight = sum(w_a for all gates)`.

#### §9.2 The 14-Dimensional Context Vector

The CONDUCTOR context vector `x ∈ R^14` is assembled by `cliffguard/conductor/context.py:build_context(verdicts, tier, attest_result)`. All values are floats; missing gates (not active for the tier) are filled with 0.0.

```mermaid
flowchart LR
    V1["VESTIBULE-LZ score\n[index 0]"]
    V2["VESTIBULE-PS score\n[index 1]"]
    RM["PROBE-RM margin\n[index 2]"]
    MT1["PROBE-MT rho_dot\n[index 3]"]
    MT2["PROBE-MT rho_ddot\n[index 4]"]
    HD["PROBE-HD margin\n[index 5]"]
    TH["TRIPWIRE-H CUSUM\n[index 6]"]
    TR["TRIPWIRE-R LLR\n[index 7]"]
    CT["LOOKOUT-CT count\n[index 8]"]
    JG["LOOKOUT-JG rate\n[index 9]"]
    BL["B-PROBE-LOGIT score\n[index 10]"]
    BC["B-PROBE-CONSISTENCY JSD\n[index 11]"]
    AT["ATTEST-WH result\n[index 12]\n1.0=ALLOW / 0.5=DEGRADED / 0.0=BLOCK"]
    TI["Tier indicator\n[index 13]\n0.0=A / 0.33=B / 0.67=C / 1.0=C+"]

    V1 --> CTX["Context vector x in R^14"]
    V2 --> CTX
    RM --> CTX
    MT1 --> CTX
    MT2 --> CTX
    HD --> CTX
    TH --> CTX
    TR --> CTX
    CT --> CTX
    JG --> CTX
    BL --> CTX
    BC --> CTX
    AT --> CTX
    TI --> CTX
    CTX --> COND["Conductor.select_weights(x)\naggregate_verdict(verdicts, weights)"]
    COND --> BLOCK([BLOCK])
    COND --> ALLOW([ALLOW])
```

The index for each feature is fixed in `FEATURE_INDEX` in `cliffguard/conductor/context.py:38-53`. Index 12 (ATTEST-WH) is always driven by the `attest_result` enum, not by the ATTEST-WH gate's verdict score; index 13 (tier) is always set from the `TIER_INDICATOR` lookup regardless of gate verdicts.

#### §9.3 Never-Disable List and MIN_WEIGHT

Two gates are in the `NEVER_DISABLE` frozenset: `TRIPWIRE-R` and `ATTEST-WH`. Even if the bandit assigns a very low UCB score to these arms, `select_weights()` clamps their weight to at least `MIN_WEIGHT = 0.1`. This ensures that:

- TRIPWIRE-R always contributes its reference-ratio signal — an adversary cannot cause the bandit to fully ignore statistical anomalies
- ATTEST-WH always contributes the boot-time attestation result — a poisoned-weight deployment cannot be silently ignored even if all other gates are unreliable

Source: `cliffguard/conductor/bandit.py:36-37` (`NEVER_DISABLE`), `cliffguard/conductor/bandit.py:37` (`MIN_WEIGHT = 0.1`).

#### §9.4 ADWIN Drift Detection and Weight Reset

When the adversary's attack distribution shifts — e.g., the adversary learns the current arm weights and routes attacks through the weakest gate — the bandit's historical A matrices and b vectors reflect the old distribution and may cause poor weight selection.

The CONDUCTOR's `reset_weights()` method re-initialises all LinUCBArm states (sets A = I, b = 0 for every arm), effectively forgetting all learned weights. This is triggered by ADWIN drift detection.

**Implementation note (decisions_log C31).** Blueprint §6.4 specifies ADWIN (Bifet and Gavaldà 2007). The Phase B prompt specified a "Page-Hinkley variant," which is a different algorithm. Both are implemented in `cliffguard/eval/drift_sim.py`:
- `adwin_statistic(stream, warmup, delta)` — the Page-Hinkley proxy used during Phase A development
- `true_adwin_statistic(stream, delta)` — the Bifet-Gavaldà ADWIN algorithm added as an additive Phase B change

Before submission, the paper's §6.4 algorithm name and the implementation must be reconciled (decisions_log C31). The parameter `ADWIN_DELTA = 0.002` is defined in `cliffguard/eval/drift_sim.py` and corresponds to λ_thresh = −log(0.002) ≈ 6.21 for the Page-Hinkley variant.

#### §9.5 Where Implemented

- `cliffguard/conductor/bandit.py` — `LinUCBArm`, `Conductor` class with `select_weights()`, `update()`, `aggregate_verdict()`, `reset_weights()`.
- `cliffguard/conductor/context.py` — `build_context()`, `FEATURE_INDEX`, `TIER_INDICATOR`, `ATTEST_SCORE`, `CONTEXT_DIM = 14`.
- `tests/test_conductor.py` — 23 tests covering update mechanics, aggregate verdict, never-disable floor.
- `tests/test_context.py` — 28 tests covering build_context for all tier/gate combinations.

---

### §10 LADDER (the Tier Router)

*The static tier router that selects which gates run on each hardware class — configuration, not learning.*

**Analogy.** Airport security: an international hub (Tier A) runs 8 independent checkpoints — biometrics, X-ray, passport control, customs, explosives trace, body scanner, declaration, and agriculture. A regional airport (Tier C) runs 3 — ID check, X-ray, and emergency contact verification. The checkpoint list is determined by the facility's resources, not by learning from passenger history.

#### §10.1 Tier Gate Sets

The gate sets are defined in `cliffguard/ladder/tier.py:TIER_GATES` and are reproduced exactly below:

| Tier | Hardware Reference | Active Gates | Count |
|---|---|---|---|
| **A** | RTX 5060 8 GB | VESTIBULE-LZ, VESTIBULE-PS, PROBE-RM, PROBE-MT, PROBE-HD, TRIPWIRE-H, TRIPWIRE-R, LOOKOUT-CT, LOOKOUT-JG, B-PROBE-LOGIT, B-PROBE-CONSISTENCY, ATTEST-WH | 12 |
| **B** | Raspberry Pi 5 8 GB | VESTIBULE-LZ, VESTIBULE-PS, PROBE-RM, PROBE-MT, PROBE-HD, TRIPWIRE-H, TRIPWIRE-R, LOOKOUT-CT, B-PROBE-LOGIT, B-PROBE-CONSISTENCY, ATTEST-WH | 11 |
| **C** | 2 GB embedded (RK3588 / Jetson / Pi 4) | VESTIBULE-LZ, VESTIBULE-PS, ATTEST-WH | 3 |
| **C+** | 2 GB embedded + PromptGuard-2-22M | VESTIBULE-LZ, VESTIBULE-PS, B-PROBE-LOGIT, ATTEST-WH | 4 |

Tier B drops LOOKOUT-JG because Llama Guard 3 inference is too slow on CPU. Tier C drops all gates that require GPU/CUDA, multi-forward-pass, or KenLM binary installation. Tier C+ adds B-PROBE-LOGIT (powered by PromptGuard-2-22M-INT4) to the Tier C set.

Note (decisions_log C35): the blueprint §10 DOT diagram shows Tier C active gates as `VESTIBULE-LZ + PROBE-RM(1L) + PH + CT`, which differs from the code's `TIER_GATES[Tier.C]`. The code is the Phase A implementation authority; the blueprint diagram may reflect a planned Phase B configuration. Paper §10 must be reconciled with TIER_GATES before submission.

#### §10.2 White-Box vs Black-Box Routing

LADDER also handles observability routing. If `white_box=False` (closed-weight API, NPU-frozen graph without hidden-state output), the PROBE gates (PROBE-RM, PROBE-MT, PROBE-HD) are excluded from the returned gate list even if the tier nominally includes them. The B-PROBE gates (B-PROBE-LOGIT, B-PROBE-CONSISTENCY) replace them. This logic is in `cliffguard/ladder/router.py:route(tier, white_box)`.

```mermaid
flowchart TD
    Q1{Is tier A or B?}
    Q2{Is white_box=True?}
    Q3{Is tier C+?}

    Q1 -->|Yes| Q2
    Q1 -->|No: C or C+| Q3
    Q2 -->|Yes| FULL["All tier gates including PROBE"]
    Q2 -->|No: black-box mode| NOPROBE["Tier gates minus PROBE-RM/MT/HD\nB-PROBE-LOGIT and CONSISTENCY retained"]
    Q3 -->|Yes: C+| CPLUS["VESTIBULE-LZ, VESTIBULE-PS,\nB-PROBE-LOGIT, ATTEST-WH"]
    Q3 -->|No: C| CONLY["VESTIBULE-LZ, VESTIBULE-PS, ATTEST-WH"]
```

#### §10.3 CONDUCTOR Dimensionality per Tier

The LinUCB feature matrix dimension is 14 regardless of tier (CONTEXT_DIM = 14 is fixed). Missing gates fill their context vector indices with 0.0. The CONDUCTOR ARM set is always the full `ARMS` list. This design allows a single CONDUCTOR to be used across all tiers without retraining; the tier indicator at index 13 gives the bandit sufficient signal to learn tier-specific weight preferences.

The blueprint references `LinUCB |A|=16` for Tier A and `|A|=8` for Tier B (README.md). This refers to the number of active gates (arms) the CONDUCTOR weights — 12 gates + 4 metadata features would give |A|=16 context dimensions, but the CONDUCTOR code uses 14 dimensions (not 16). This is a documentation discrepancy; the code's 14 is authoritative.

#### §10.4 Where Implemented

- `cliffguard/ladder/tier.py` — `TIER_GATES` dict (lines 19–45); `gates_for_tier(tier)` (lines 48–50); `is_gate_active(gate_name, tier)` (lines 53–55).
- `cliffguard/ladder/router.py` — `route(tier, white_box)`, `gate_count(tier)`.
- `tests/test_integration_smoke.py` — 13 integration tests covering the full gate-selection and dry-run path.

---

### §11 ATTEST (the Integrity Gate)

*Boot-time SHA-256 weight attestation — the tamper-evident seal that defends against Egashira-style poisoned-weight supply-chain attacks.*

This section gives the operational summary of ATTEST-WH. The implementation details are covered in §8.12. Here we focus on the system-level role.

#### §11.1 Why Boot-Time Attestation

A poisoned-weight attack (A2, Egashira et al. ICLR 2025) produces a GGUF file where the weights are behaviorally benign at FP16 but malicious at Q3_K_M. The poisoning is encoded in the quantization-specific representation: the FP16 dequantized values are safe, but the Q3_K_M quantization buckets have been crafted to produce a specific harmful output for certain input prompts. An attacker who can substitute a model file in a model repository (or in a local Ollama model store) can deploy this attack at scale.

ATTEST-WH breaks this attack at the supply-chain layer: if the weights have been substituted between the time the vendor computed the manifest hash and the time of deployment, the SHA-256 mismatch is detected at boot before any inference begins.

#### §11.2 Three Verdict States

| Result | CONDUCTOR index 12 | Meaning | Action |
|---|---|---|---|
| `ALLOW` | 1.0 | Hash matches manifest | Proceed normally |
| `DEGRADED` | 0.5 | Manifest has no entry for this filename, or Tier C/C+ with mismatch | Continue with heightened alert status |
| `BLOCK` | 0.0 | Hash mismatch at Tier A or B | Hard block — no inference |

#### §11.3 Analogy — Tamper-Evident Seal

When a shipping container leaves the manufacturer, a tamper-evident seal is applied. Any tampering during transit breaks the seal visibly. ATTEST-WH works the same way: the vendor applies the "seal" (SHA-256 hash in the manifest) at production time. At boot, CLIFFGUARD checks the seal. A broken seal (hash mismatch) indicates the container has been tampered with.

The analogy has an important limit: ATTEST only checks the weight *file*. If the inference engine itself (`libllama.so`, the CUDA kernels) has been compromised, ATTEST does not detect it — the seal is on the container, not on the dock workers who unload it.

#### §11.4 Where Implemented

- `cliffguard/attest/wh.py:1-91` — `AttestResult` enum, `hash_file()`, `load_manifest()`, `attest()`.
- `tests/test_attest_wh.py` — 16 tests.
- Error handling: blueprint §Error Handling Contracts in `docs/engineering_reference.md`:
  - Hash mismatch at Tier A/B → `AttestResult.BLOCK` → CONDUCTOR hard-blocks
  - Manifest not found → `FileNotFoundError` → propagates to caller
  - Manifest exists but no entry for filename → `AttestResult.DEGRADED`

---

## Part IV — The Hardware Tiers

---

### §12 The 4 Tiers in Detail

*The four hardware tiers define the gate set, quantization schemes, and deployment scope — each tier is an honest statement of what CLIFFGUARD can and cannot do on that hardware.*

For full setup instructions, see [docs/setup.md](setup.md). For RTX 3050 users, see [docs/setup_3050.md](setup_3050.md).

---

#### §12.1 Tier A — RTX 5060 8 GB (Full Stack)

**Hardware reference.** RTX 5060 8 GB or any CUDA-capable GPU with ≥ 8 GB VRAM. Linux is recommended; Windows works for `torch` + `bitsandbytes` NF4 but `autoawq` and `vllm` are Linux-only (gated by `sys_platform == "linux"` in `pyproject.toml`, decisions_log 2026-05-03).

**In-scope quantization schemes.** FP16, NF4, AWQ-INT4, GGUF Q5_K_M, Q4_K_M, Q3_K_M. (AWQ-INT4 requires autoawq, Linux only.)

**Active gate set.** All 12 gates: VESTIBULE-LZ, VESTIBULE-PS, PROBE-RM, PROBE-MT, PROBE-HD, TRIPWIRE-H, TRIPWIRE-R, LOOKOUT-CT, LOOKOUT-JG, B-PROBE-LOGIT, B-PROBE-CONSISTENCY, ATTEST-WH.

**Model size.** 7–9 B parameters in NF4 or AWQ-INT4. Typical: Llama-3-8B-Instruct, Mistral-7B-v0.3, Qwen2.5-7B-Instruct.

**Budget envelope.** Full PROBE stack requires ~150 MB for refusal direction vectors and calibration tables. LOOKOUT-JG requires Llama Guard 3 (1B INT4, ~440 MB on-disk). Total defense overhead: < 1 GB VRAM (all gates combined), << inference cost of the protected model.

**Trade-off summary.** Full protection across all 9 adversary classes (A1–A9). Highest false-positive rate without calibration — but calibration eliminates this. The only tier where all five hypotheses (H1–H5) are fully tested. LOOKOUT-JG is expensive (N=3 LLM calls per request) but provides the highest-quality compliance judgment.

**Setup.**
```bash
git clone https://github.com/parnish007/CLIFFGUARD.git && cd CLIFFGUARD
uv sync --extra gpu
uv run python scripts/dry_run.py --tier A --scheme FP16
```

---

#### §12.2 Tier B — Raspberry Pi 5 8 GB (CPU Inference)

**Hardware reference.** Raspberry Pi 5 8 GB. Models run via `llama-cpp-python` on ARM64 CPU. Approximate throughput: Qwen-2.5-1.5B Q4_K_M at ~5–7 tok/s, 3B at ~3–5 tok/s (per Stratosphere Lab 2025 benchmarks, cited in docs/setup.md).

**In-scope quantization schemes.** GGUF Q4_K_M, GGUF Q3_K_M. (FP16 and NF4 require CUDA; AWQ-INT4 is Linux-CUDA-only.)

**Active gate set.** 11 gates: all Tier A gates except LOOKOUT-JG (Llama Guard 3 inference is too slow on CPU — O(seconds) per judgment, vs. O(ms) for all other gates).

**Model size.** 1.5B–3B GGUF models. Typical: Qwen2.5-1.5B-Instruct Q4_K_M, TinyLlama-1.1B Q4_K_M.

**Budget envelope.** `llama-cpp-python` builds from source on ARM64 (5–10 min). Model fits in RAM. PROBE stack requires CPU embedding extraction via `llama_get_embeddings_ith` callback (Phase B). No separate GPU overhead.

**Trade-off summary.** Near-full protection (11/12 gates). Loses the LLM-judge verdict (LOOKOUT-JG) but retains B-PROBE-CONSISTENCY as a distributional substitute. LinUCB context vector size 14 (same as Tier A; LOOKOUT-JG score defaults to 0.0 at index 9). The cliff exploiter (A7) is partially defended via PROBE-RM/HD — whether 1.5B models show the same cliff as 7B is an empirical question for Phase C.

**Setup.**
```bash
git clone https://github.com/parnish007/CLIFFGUARD.git && cd CLIFFGUARD
uv sync --extra gpu    # llama-cpp-python builds here; takes 5-10 min on ARM64
uv run python scripts/dry_run.py --tier B --scheme GGUF_Q4_K_M
```

---

#### §12.3 Tier C — 2 GB Embedded (Narrow Scope)

**Hardware reference.** RK3588 NPU W8A8 (10–15 tok/s on 1.1B per tinycomputers.io), Jetson Orin Nano 4 GB, Raspberry Pi 4 8 GB. Models: TinyLlama-1.1B, Qwen-2.5-0.5B/1.5B in Q3_K_M or Q4_K_M.

**In-scope quantization schemes.** GGUF Q3_K_M, IQ3_XXS, Q2_K, RKNN W8A8.

**Active gate set.** 3 gates: VESTIBULE-LZ, VESTIBULE-PS, ATTEST-WH.

**Budget envelope.** Q3_K_M base model ~1.4 GB + KV cache ~150 MB. Total model footprint leaves < 500 MB for the defense layer. All three active gates are essentially free: zlib compression (20 µs/KB), regex matching (< 100 µs), SHA-256 hashing (once at boot).

**Trade-off summary.** Narrow scope. No PROBE gates (no residual-stream access), no TRIPWIRE (no per-token logprob stream required by the defense), no LOOKOUT-JG (no LLM judge). ATTEST-WH mismatch returns DEGRADED rather than BLOCK (embedded boards often lack signed manifests). The bandit is NOT used — weights are fixed at deployment. CONDUCTOR falls back to a static expert-tuned policy with EWMA-based drift alarms only.

**Honest scope statement (blueprint §10, decisions_log C35).** Tier C is **not defended against A7 (quantization-cliff exploiter)**. The minimal gate set does not observe the refusal-margin signal. H5 pre-registers that Tier C will show no statistically significant ABR reduction against the cliff exploiter. Suitable *only* for fixed-grammar single-task assistants with no open-domain adversarial exposure.

**Setup.**
```bash
git clone https://github.com/parnish007/CLIFFGUARD.git && cd CLIFFGUARD
uv sync           # no --extra gpu needed
uv run python scripts/dry_run.py --tier C --scheme GGUF_Q3_K_M
```

---

#### §12.4 Tier C+ — 2 GB Embedded with PromptGuard-2-22M-INT4

**Hardware reference.** Same as Tier C. Adds Meta's PromptGuard-2-22M (DeBERTa-xsmall, 22 M parameters, MIT-licensed) as B-PROBE-LOGIT.

**In-scope quantization schemes.** Same as Tier C: GGUF Q3_K_M, IQ3_XXS, RKNN W8A8.

**Active gate set.** 4 gates: VESTIBULE-LZ, VESTIBULE-PS, B-PROBE-LOGIT (PromptGuard-2-22M-INT4), ATTEST-WH.

**Memory budget.** Q3_K_M base model ~1.4 GB + KV cache ~150 MB + PromptGuard-2-22M-INT4 ~30 MB + PROBE-RM final-layer projector ~50 MB ≈ 1.65 GB total. Fits under 1.8 GB.

**Trade-off summary.** Adds a dedicated surface classifier at minimal memory overhead relative to Tier C. PromptGuard-2-22M is a DeBERTa-xsmall model fine-tuned specifically on prompt injection detection. H5 pre-registers that Tier C+ will show a statistically significant ABR reduction against A7 (the cliff exploiter) while Tier C will not. Static weights (no bandit).

**Setup.**
```bash
# Same as Tier C install
uv run python scripts/dry_run.py --tier C_PLUS --scheme GGUF_Q3_K_M
```
PromptGuard-2-22M-INT4 weights: download from `meta-llama/Llama-Prompt-Guard-2-22M` on Hugging Face. Phase B wiring is in `cliffguard/engines/` (stub in Phase A).

---

## Part V — The Five Hypotheses

---

### §13 H1: The Cliff Exists

*The foundational hypothesis that a κ ≥ 0.25 jump in both the geometric and behavioral cliff metrics occurs at the same quantization boundary in at least 2 of 3 model families.*

#### §13.1 Formal Statement

> For at least 2 of 3 model families (Llama-3-8B, Mistral-7B, Qwen2.5-7B), both the geometric cliff metric Δ_cliff and the behavioral cliff metric Δ_B_cliff exhibit a jump of κ ≥ 0.25 at the same quantization boundary.

**Acceptance criterion:** `detect_cliff_boundary()` returns the same QuantScheme for ≥ 2 model families, with both Δ_cliff ≥ 0.25 and Δ_B_cliff ≥ 0.25 at that boundary.

**Null:** No consistent cliff boundary (< 2 families exceed κ = 0.25 at the same boundary).

Source: `docs/preregistration.md §H1`.

#### §13.2 Plain-English Intuition

H1 asks: is there really a cliff, or is the degradation gradual? A gradual degradation would mean "as you quantize further, attacks succeed a little more each step." The cliff hypothesis says "at one specific bit-width — empirically near Q3_K_M — there is a discontinuous jump where both the geometry of refusal in the residual stream and the actual attack success rate jump simultaneously."

The geometric–behavioral agreement requirement is the non-circularity discipline: if only the geometric metric jumps (but attacks do not succeed more), the residual-stream signal is not actually causal. If only the behavioral metric jumps (but geometry doesn't change), PROBE-RM wouldn't detect the attack. CLIFFGUARD's design center depends on both jumping together.

```mermaid
flowchart LR
    FP16[FP16 model] --> Q4["Q4_K_M model"]
    Q4 --> Q3["Q3_K_M model"]
    Q3 --> Q2["Q2_K model"]
    FP16 -->|"delta_cliff(FP16,Q4)"| M1["Jump < kappa?"]
    Q4 -->|"delta_cliff(Q4,Q3)"| M2["Jump >= kappa?\nCliff boundary!"]
    Q3 -->|"delta_cliff(Q3,Q2)"| M3["Post-cliff"]
    M1 -->|"No cliff"| CONT["Continue"]
    M2 -->|"H1 accepted"| CLIFF["Cliff at Q3_K_M"]
```

#### §13.3 How It's Tested

Fold B computes the three cliff metrics (`geometric_cliff`, `wasserstein_cliff`, `behavioral_cliff`) per (model, scheme) pair. `detect_cliff_boundary()` in `cliffguard/eval/cliff_metrics.py` identifies the boundary where all three agree. The test is run for each of the three model families independently; H1 accepts if ≥ 2 families agree on the same boundary.

**Implementation:** `cliffguard/eval/cliff_metrics.py` — 43 tests in `tests/test_cliff_metrics.py`.

**Geometric range note (decisions_log C27):** `geometric_cliff` has range [0, √2], not [0, 1]. The κ = 0.25 threshold is on this [0, √2] scale. Blueprint §11.3 must be updated before submission.

#### §13.4 Accepted vs Rejected

- **H1 accepted:** CLIFFGUARD's design premise is validated. PROBE-RM is measuring a geometrically real cliff. The evaluation proceeds to H2–H5.
- **H1 rejected:** The cliff is not consistently present across families or is not co-located geometrically and behaviorally. This would *not* mean CLIFFGUARD fails — the behavioral defense still works — but it would invalidate PROBE-RM's causal interpretation and require revising the paper's theoretical framing.

---

### §14 H2: FPR Decoupling — White-Box

*PROBE-RM's false-positive rate after per-quantization calibration varies less than ε = 0.02 across five quantization schemes.*

#### §14.1 Formal Statement

> PROBE-RM FPR after per-quantization calibration varies less than ε = 0.02 across {FP16, NF4, AWQ-INT4, Q4_K_M, Q3_K_M}.

**Metric:** `max(empirical_fpr) - min(empirical_fpr) < 0.02` where FPR is measured on Fold A benign prompts at `fpr_target = 0.05`.

**Acceptance criterion:** FPR range < 0.02.

**Null:** FPR range ≥ 0.02 (calibration fails to achieve portability).

Source: `docs/preregistration.md §H2`.

#### §14.2 Plain-English Intuition

H2 asks: once you have calibrated PROBE-RM's threshold for each quantization scheme separately, does the gate's false-positive rate stay constant across schemes? If yes, you can deploy the same gate logic across FP16, NF4, and Q3_K_M deployments without worrying that it will suddenly block 20% of benign traffic at one scheme versus 2% at another.

The FPR-decoupling theorem (§18) provides the theoretical justification: the benign refusal-margin distribution shifts under quantization, but the calibration map absorbs the shift — the FPR at the calibrated threshold stays within ε = 0.02.

```mermaid
flowchart LR
    FOLD_A["Fold A\n2000 benign prompts\nper scheme"] --> CAL["Calibrate tau_q\nfor each scheme"]
    CAL --> FPR_FP16["FPR at tau_FP16\non FP16 traffic"]
    CAL --> FPR_NF4["FPR at tau_NF4\non NF4 traffic"]
    CAL --> FPR_Q3["FPR at tau_Q3\non Q3_K_M traffic"]
    FPR_FP16 --> RANGE["max(FPR) - min(FPR)"]
    FPR_NF4 --> RANGE
    FPR_Q3 --> RANGE
    RANGE -->|"< 0.02"| ACCEPT["H2 accepted"]
    RANGE -->|">= 0.02"| REJECT["H2 rejected"]
```

#### §14.3 How It's Tested

Fold A calibrates the threshold `tau_q` for each scheme using the empirical (1 - fpr_target) quantile of the benign margin distribution. Fold C measures the actual FPR of PROBE-RM on held-out benign prompts for each scheme. The FPR range is `max(empirical_fpr across schemes) - min(empirical_fpr across schemes)`.

---

### §15 H3: FPR Decoupling — Black-Box

*B-PROBE-LOGIT's FPR after per-quantization calibration varies less than ε = 0.02 across the same schemes, and its TPR is strictly lower than PROBE-RM's (confirming the white-box advantage).*

#### §15.1 Formal Statement

> B-PROBE-LOGIT FPR after per-quantization calibration varies less than ε = 0.02 across the same schemes, with strictly lower TPR than PROBE-RM (white-box advantage confirmed).

**Acceptance criterion:** FPR range < 0.02 AND `TPR(B-PROBE-LOGIT) < TPR(PROBE-RM)` at matched FPR.

**Null:** FPR range ≥ 0.02 OR `TPR(B-PROBE-LOGIT) ≥ TPR(PROBE-RM)`.

Source: `docs/preregistration.md §H3`.

#### §15.2 Plain-English Intuition

H3 asks two things simultaneously: (1) does the FPR-decoupling hold for the black-box gate (using only top-k logprobs, no hidden states)? and (2) is the white-box gate strictly better in terms of attack detection (TPR)?

The second requirement is critical for honest scoping: the B-PROBE branch is a fallback, not a replacement. If B-PROBE-LOGIT had equal or better TPR than PROBE-RM, the white-box access requirement of PROBE could be questioned. By pre-registering that white-box has strictly higher TPR, the system honestly acknowledges the cost of the black-box mode.

```mermaid
flowchart LR
    H3["H3 requires BOTH:"]
    H3 --> FPR["B-PROBE-LOGIT FPR range < 0.02\n(same as H2 but for black-box)"]
    H3 --> TPR["TPR(B-PROBE-LOGIT) < TPR(PROBE-RM)\nat matched FPR"]
    FPR --> ACCEPT["H3 accepted\n(black-box FPR portable\nand white-box advantage confirmed)"]
    TPR --> ACCEPT
```

---

### §16 H4: Composition Gain

*The full 12-gate stack achieves strictly smaller attack success rate than any single primitive at matched false-positive rate.*

#### §16.1 Formal Statement

> The full primitive stack achieves strictly smaller ASR than any single primitive at matched FPR.

**Metric:** `ASR(full stack) < min(ASR(single primitive))` at `fpr_target = 0.05`.

**Acceptance criterion:** Wilcoxon signed-rank test p < 0.01 (Bonferroni-corrected α) on per-prompt block decisions (paired, full stack vs. best single primitive).

**Null:** No significant ASR reduction from composition (p ≥ 0.01).

Source: `docs/preregistration.md §H4`. Note: the preregistration document states p < 0.05 (uncorrected), but the statistical conventions section mandates Bonferroni correction (α = 0.01). The code's `stats.py` uses 0.01. The preregistration must be updated before submission (decisions_log C30).

#### §16.2 Plain-English Intuition

H4 asks: do the 12 gates together catch more attacks than any single gate alone? This is the "defense-in-depth" hypothesis. Each gate covers a different adversary class; an attacker who evades VESTIBULE-LZ (by not using an incompressible suffix) may be caught by PROBE-RM (if the refusal margin drops). The composition gain is statistically tested, not just asserted.

```mermaid
flowchart LR
    ATK["Attack corpus\nFold C"] --> SINGLE["Test each gate alone\nASR_1, ASR_2, ..., ASR_12"]
    ATK --> FULL["Test full 12-gate stack\nASR_full"]
    SINGLE --> BEST["min(ASR_1..12)"]
    BEST --> WILCOX["Wilcoxon signed-rank test\nFull stack vs. best single"]
    FULL --> WILCOX
    WILCOX -->|"p < 0.01"| ACCEPT["H4 accepted"]
    WILCOX -->|"p >= 0.01"| REJECT["H4 rejected"]
```

---

### §17 H5: Tier C Structural Weakness

*Tier C (3 gates) shows no significant ABR reduction against the cliff exploiter; Tier C+ (4 gates + PromptGuard-2-22M) does.*

#### §17.1 Formal Statement

> Tier C shows no statistically significant ABR reduction against A7 (cliff exploiter) relative to no-defense baseline. Tier C+ (with PromptGuard-2-22M-INT4) does show significant ABR reduction.

**Acceptance criterion:** `p(Tier C) ≥ 0.05 AND p(Tier C+) < 0.05` on per-prompt Wilcoxon signed-rank test vs. baseline.

**Null:** Tier C achieves significant ABR reduction (p < 0.05).

Source: `docs/preregistration.md §H5`. Note: same α = 0.01 discrepancy as H4; preregistration must be updated (decisions_log C30).

#### §17.2 Plain-English Intuition

H5 is the honest-scope hypothesis. It pre-registers that Tier C is *expected to fail* against the cliff exploiter — and expects Tier C+ to succeed. This is not a failure of the paper; it is an honest pre-registration of a known structural limitation. By pre-registering the expected failure, the paper cannot be accused of cherry-picking results.

```mermaid
flowchart LR
    CLIFF["A7 cliff exploit corpus"] --> TC["Test Tier C\n3 gates only"]
    CLIFF --> TCPLUS["Test Tier C+\n4 gates + PromptGuard-2-22M"]
    TC -->|"ABR vs baseline"| PCT["p(Tier C) >= 0.05\n(no sig reduction)"]
    TCPLUS -->|"ABR vs baseline"| PCTP["p(Tier C+) < 0.05\n(significant reduction)"]
    PCT --> ACCEPT["H5 accepted\n(both conditions met)"]
    PCTP --> ACCEPT
```

---

### §18 Theorem 14.1 — FPR-Decoupling Theorem

*The formal theoretical foundation for why FPR is portable across quantization schemes after calibration, and why TPR is not.*

#### §18.1 Statement

The FPR-decoupling theorem (blueprint §14.1, paraphrased) states: for any write-side gate g whose score function depends only on (a) the input string x and (b) summary statistics of model outputs (hidden-state projections, top-k logprobs, token entropy), there exists a calibration map P_q: R → R such that the false-positive rate of g composed with P_q is independent of the quantization scheme q̃ up to error ε = 0.02.

More precisely: let `tau_q̃` be the threshold calibrated on Fold A benign traffic for scheme q̃. Then for any two schemes q̃_1 and q̃_2:

```
|FPR(g, tau_{q̃_1}, benign under q̃_1) - FPR(g, tau_{q̃_2}, benign under q̃_2)| < ε = 0.02
```

**Why this holds:** The benign refusal-margin distribution shifts under quantization (the mean and variance of `m_r` on benign prompts change from FP16 to Q3_K_M). But the shape of the distribution (approximately Gaussian) is preserved. The calibrated threshold `tau_q̃` tracks the (1-α) quantile of this shifted distribution, so the FPR at `tau_q̃` remains ≈ α regardless of where the distribution has shifted.

**Why TPR is NOT decoupled:** TPR depends on the harmful-prompt margin distribution. In the cliff regime, the harmful-prompt distribution collapses toward the benign distribution (the residual-stream representation of harmful prompts becomes indistinguishable from benign prompts). The calibrated threshold `tau_q̃` is set against the benign distribution — it cannot move to accommodate a collapsing harmful distribution without raising FPR. TPR therefore falls in the cliff regime even as FPR stays constant. This is the mechanism that H1 and H2 measure.

#### §18.2 The G1 Geometric Assumption

The theorem rests on a geometric assumption (G1, blueprint §3.4): the safety subspace `S = span{r_hat, h_hat}` must be an approximately isometric projection of the full residual-stream space for the distribution of harmful vs. benign prompts. Formally:

```
||Pi_S z(t_inst, q) - Pi_S z(t_inst, q')|| >= xi * ||z(t_inst, q) - z(t_inst, q')||
```

for some ξ > 0 (the safety-subspace fraction). G1 is empirically supported by Arditi et al. and Zhao et al. for instruction-tuned chat LLMs at 7B–72B parameters. If G1 fails (e.g., the safety subspace is near-degenerate for a specific model family), PROBE-RM's margins become uninformative and H2 may be rejected even though CLIFFGUARD's input-side gates (VESTIBULE, B-PROBE) continue to function.

#### §18.3 Sensitivity Bound

The Sensitivity Corollary (blueprint §3.4) provides the formal link between quantization error and safety collapse:

```
||pi(q) - pi(q')|| <= L_{q̃} * (||delta_z||_2 + 2 * epsilon_{q̃})
```

where `pi(q) = Pi_S z(t_inst, q)` is the safety projection, `L_{q̃}` is the Lipschitz constant of the quantization operator on the residual stream, `delta_z` is the activation perturbation induced by the adversarial input, and `epsilon_{q̃}` is the per-quantization residual reconstruction error.

**The cliff is the regime where `epsilon_{q̃}` inflates enough that `pi(q)` flips sign on harmful prompts** — the safety projection of a harmful prompt crosses into the benign region, causing PROBE-RM to see a healthy-looking margin even when the model will comply.

#### §18.4 Worked Example

Suppose at FP16 calibration on Fold A benign traffic, the 95th percentile of the refusal-margin distribution is `tau_FP16 = 0.32` (PROBE-RM fires LOW when margin < 0.32; FPR at this threshold ≈ 0.05). At NF4, the benign margin distribution has shifted (quantization compresses the refusal direction, so benign prompts have slightly lower margins on average). The new 95th percentile is `tau_NF4 = 0.28`. After calibrating to `tau_NF4`, the FPR on NF4 benign traffic is again ≈ 0.05.

The attack in this example: a prompt that at FP16 produces margin `m_r = 0.25 < tau_FP16 = 0.32` → PROBE-RM fires → BLOCK. At NF4, the same prompt produces margin `m_r = 0.30 > tau_NF4 = 0.28` → PROBE-RM does NOT fire → ALLOW. This is a TPR loss due to the cliff — the gate misses the attack at NF4 because the harmful-prompt margin has moved into the benign range. FPR has not changed (benign prompts are still blocked at the same rate), but TPR has dropped. This is the cliff.

---

## Part VI — The Five Folds (Corpus Discipline)

---

### §19 Fold Isolation Discipline

*The five-fold structure is designed to prevent any calibration, training, or threshold data from leaking into the evaluation folds — every number in the paper comes from data that no gate has ever seen.*

#### §19.1 Why Folds Matter

In a typical ML evaluation, it is straightforward to inadvertently contaminate test data: calibrating thresholds on the same data you then test on inflates apparent performance. For CLIFFGUARD, this risk is acute because the calibration step (Fold A) directly determines the decision thresholds (tau_q) used in the evaluation folds (Folds B and C). If Fold A benign prompts appear in Fold B or C, the calibrated thresholds are overfitted to those prompts, and the FPR measured in Folds B and C is biased.

Additionally, the adversarial corpus (Folds B and C) must not be seen during calibration or direction extraction — otherwise the defender-aware adversary could craft prompts that look benign in the calibration data while being effective in the evaluation.

#### §19.2 SHA-256-Based Fold Isolation Check

Each fold entry in the corpus is identified by a SHA-256 hash of the prompt text. Before any fold is used, `cliffguard/eval/folds.py` computes the hash of every entry and verifies that no entry appears in both the calibration set and the evaluation set. If an intersection is found, the fold isolation check raises a `ValueError` and halts the run.

This check is implemented in `cliffguard/eval/folds.py` (tested in `tests/test_folds.py`, 13 tests). It is mandatory before any Fold B, C, D, or E operation — a design constraint enforced at the code level, not just by convention.

---

### §20 Fold A: Calibration

*The calibration fold — 2000 benign prompts per scheme — is the foundation for every threshold, direction vector, and reference model used in the evaluation.*

#### §20.1 Sources

Fold A uses two publicly available datasets:
- **Anthropic-HH-RLHF** (Apache 2.0): human preference data from Anthropic, containing benign multi-turn conversations.
- **OpenAssistant-OASST1** (Apache 2.0): the OpenAssistant dataset, containing benign human-AI conversations.

Both are downloaded automatically by `scripts/download_fold_a.py --download`. Data is written to `data/folds/fold_a/` and is gitignored. The minimum required corpus size is `MIN_CALIBRATION_SIZE = 2000` benign prompts (below this, the KS estimation error for the empirical quantile is too wide; documented in `cliffguard/eval/calibration.py`).

#### §20.2 What Gets Extracted

Running Fold A produces the following artifacts in `artifacts/runs/<run_id>/fold_a/`:

| Artifact | Function |
|---|---|
| `r_hat_<model>_<scheme>.npz` | Refusal direction vector per (model, scheme) — used by PROBE-RM |
| `h_hat_<model>_<scheme>.npz` | Harmfulness direction vector per (model, scheme) — used by PROBE-HD |
| `calibration_summary.json` | Calibrated threshold τ_q per gate per scheme — used by all gates |
| `arpa_<model>_<scheme>.bin` | Trained KenLM ARPA file — used by TRIPWIRE-R |
| `logistic_weights_<scheme>.npz` | Fitted logistic head weights — used by B-PROBE-LOGIT |
| `fp16_margins.npz` | FP16 margin distribution — used as reference for Fold E BCN-2 construction |

#### §20.3 Where Implemented

- `cliffguard/eval/refusal_direction.py` — `calibrate_refusal_direction()` — 33 tests in `test_refusal_direction.py`.
- `cliffguard/eval/harmfulness_direction.py` — `calibrate_harmfulness_direction()` — 32 tests in `test_harmfulness_direction.py`.
- `cliffguard/eval/threshold_calibrator.py` — `build_calibration_table()` per primitive — 22 tests in `test_threshold_calibrator.py`.
- `cliffguard/eval/kenlm_trainer.py` — `train_and_save()` — 19 tests in `test_kenlm_trainer.py`.
- `cliffguard/eval/five_fold_live.py` — `live_execute_fold_a()` — the real-mode orchestration function (added in Phase B, does not touch the test-locked `FiveFoldOrchestrator` stubs).

---

### §21 Fold B: Cliff Measurement (H1)

*Fold B measures the three cliff metrics across all (model, scheme) pairs — the direct test of H1.*

#### §21.1 Sources

Fold B uses the adversarial evaluation corpus assembled from:
- **AdvBench-50** (MIT): 50 adversarial behavior strings from Zou et al.
- **JailbreakBench** (MIT): curated jailbreak prompts across multiple categories.
- **Synthetic cliff-exploiters (A7)**: prompts generated per the BCN-2 protocol (Fold E) — natural-language prompts near the FP16 refusal boundary.

Phase B corpus assembly is currently manual for Folds B and C. Only Fold A has an automated downloader. The adversarial corpus schema is defined in `cliffguard/eval/attack_corpus.py` (20 tests in `test_attack_corpus.py`).

#### §21.2 Output

For each (model, scheme) pair, Fold B produces:
- `cliff_metrics.json` in `artifacts/runs/<run_id>/fold_b/`: contains `delta_cliff`, `delta_W_cliff`, `delta_B_cliff` per scheme.
- `gate_verdicts_fold_b.jsonl`: one record per prompt, per gate, per scheme.

`detect_cliff_boundary()` in `cliffguard/eval/cliff_metrics.py` identifies the scheme where all three metrics agree on a κ ≥ 0.25 jump.

---

### §22 Fold C: Defense Composition (H2, H3, H4)

*Fold C evaluates the full gate stack against the adversarial corpus — producing the composition-gain evidence for H4 and the FPR measurements for H2 and H3.*

Fold C runs on the same adversarial corpus as Fold B, but now with all gates active. For H2 and H3, it measures the empirical FPR of PROBE-RM and B-PROBE-LOGIT on the benign held-out set from Fold A (using the 200-prompt held-out split). For H4, it measures the ASR of each single gate vs. the full 12-gate stack on the adversarial corpus.

The judge agreement protocol (StrongREJECT + Llama Guard 3) is used to assign ground-truth labels to gate verdicts, resolving ambiguous cases. Implementation: `cliffguard/eval/five_fold_orchestrator.py` (stubs, 32 tests); real execution via `cliffguard/eval/five_fold_live.py`.

---

### §23 Fold D: Bandit Drift

*Fold D tests whether the CONDUCTOR's LinUCB bandit recovers from an adversarial distribution shift — the attacker learns current arm weights and routes attacks through the weakest gate.*

Fold D simulates three regimes:
1. **Warmup:** 100 benign + 50 attack prompts from the standard corpus; CONDUCTOR weights adapt.
2. **Drift:** Adversary learns current weights and shifts to attacking via the weakest arm; ADWIN/Page-Hinkley detects the drift and resets weights.
3. **Recovery:** Post-reset, CONDUCTOR re-learns weights from scratch.

The test measures whether the CONDUCTOR's weight reset + re-adaptation keeps ABR within a fixed band after the drift event. Implementation: `cliffguard/eval/drift_sim.py` (`adwin_statistic`, `true_adwin_statistic`) — 22 tests in `test_drift_sim.py`. ADWIN_DELTA = 0.002 is the drift detection sensitivity parameter.

---

### §24 Fold E: BCN-2 Construction

*Fold E constructs the Below-Cliff Naturals (BCN-2) dataset — paired prompts where Q3_K_M complies but FP16 refuses, using a different model family for paraphrase generation to avoid circularity.*

BCN-2 (Below-Cliff Naturals, N=2) is a paired dataset of prompts near the FP16 refusal boundary. For each prompt, the pair is: (a) the FP16 model refuses, (b) the Q3_K_M model complies. BCN-2 is the primary corpus for H1's cliff existence evidence.

**Non-circularity discipline:** the paraphraser used to generate near-boundary variants must come from a *different* model family than the one being cliff-tested. If you are measuring the cliff on Llama-3-8B, the BCN-2 paraphraser must use Mistral or Qwen — never Llama-3. This prevents the cliff metric from being validated against data generated by the model whose cliff you are measuring.

**Storage format:** JSONL per decisions_log C28. Blueprint §12.2 specifies LSH (Locality-Sensitive Hashing) storage for deduplication; this is deferred to the public corpus release phase. The current implementation uses plain JSONL with SHA-256 fold isolation.

Implementation: `cliffguard/eval/bcn2.py` (`BCN2Dataset`, `RealParaphraser`) — 30 tests in `test_bcn2.py`.

---

### §25 The Fold A → Others Dependency

*Fold A must complete before any other fold — it produces the calibration tables, direction vectors, and reference models that all other folds depend on.*

```mermaid
flowchart TD
    A["Fold A\n(calibration)\n2000 benign prompts per scheme"] --> B["Fold B\n(cliff measurement H1)\nadvbench + JailbreakBench"]
    A --> C["Fold C\n(defense composition H2-H4)\nsame adversarial corpus as B"]
    A --> D["Fold D\n(bandit drift)\nstandard corpus + drift injection"]
    A --> E["Fold E\n(BCN-2 construction)\ncross-family paraphraser"]
    B --> H1["H1 result\ndelta_cliff >= kappa?"]
    C --> H2["H2 result\nFPR range < epsilon?"]
    C --> H3["H3 result\nB-PROBE FPR + TPR gap?"]
    C --> H4["H4 result\ncomposition gain?"]
    C --> H5["H5 result\nTier C vs C+ vs baseline?"]
```

**Running only Fold A** (before committing to the full protocol):
```bash
uv run python scripts/run_evaluation_3050.py --model auto --folds A
```

Results are written to `artifacts/runs/<tier>_<hostname>_<timestamp>/fold_a/`.

Fold A must produce `calibration_summary.json` and all direction vectors before Folds B/C/D/E can start. A missing calibration table raises `KeyError` in `CalibrationTable.__getitem__` — a runtime guard that prevents accidentally running Fold B without Fold A having completed.

---

## Part VII — Implementation Status

---

### §26 What Is Done

*An accurate, honest map of the current repository state — what has been implemented, scaffolded, or stubbed.*

#### §26.1 Phase A — Scaffolding (939 tests at Phase A gate)

Phase A scaffolded all 12 primitives and 8 components with synthetic stubs, mypy strict types, and ruff-clean code. The 939-test Phase A gate count is from the `development.md` Phase A gate milestone.

| Task | Component | Implementation File(s) | Test File | Tests |
|---|---|---|---|---|
| T1 | Repo skeleton | `pyproject.toml`, `Makefile`, `.gitignore` | `test_smoke.py` | 1 |
| T2 | Package layout | `cliffguard/__init__.py`, all subpackages | — | — |
| T3 | Types (ThreatModel, Tier, QuantScheme) | `cliffguard/types.py` | `test_types.py` | 16 |
| T4 | Types (Margin, CalibrationTable, GateVerdict) | `cliffguard/types.py` | `test_types.py` | (included) |
| T5 | VESTIBULE-LZ | `cliffguard/vestibule/lz.py` | `test_vestibule_lz.py` | 12 |
| T6 | VESTIBULE-PS | `cliffguard/vestibule/ps.py` | `test_vestibule_ps.py` | 19 |
| T7 | PROBE-RM | `cliffguard/probe/rm.py` | `test_probe_rm.py` | 17 |
| T8 | PROBE-MT | `cliffguard/probe/mt.py` | `test_probe_mt.py` | 19 |
| T9 | PROBE-HD | `cliffguard/probe/hd.py` | `test_probe_hd.py` | 18 |
| T10 | TRIPWIRE-H | `cliffguard/tripwire/h.py` | `test_tripwire_h.py` | 19 |
| T11 | TRIPWIRE-R | `cliffguard/tripwire/r.py` | `test_tripwire_r.py` + `test_tripwire_r_calibration.py` | 19 + 8 = 27 |
| T12 | LOOKOUT-CT | `cliffguard/lookout/ct.py` | `test_lookout_ct.py` | 29 |
| T13 | LOOKOUT-JG | `cliffguard/lookout/jg.py` | `test_lookout_jg.py` | 17 |
| T14 | B-PROBE-LOGIT | `cliffguard/bprobe/logit.py` | `test_bprobe_logit.py` | 26 |
| T15 | B-PROBE-CONSISTENCY | `cliffguard/bprobe/consistency.py` | `test_bprobe_consistency.py` | 25 |
| T16 | ATTEST-WH | `cliffguard/attest/wh.py` | `test_attest_wh.py` | 16 |
| T17 | CONDUCTOR | `cliffguard/conductor/bandit.py` + `context.py` | `test_conductor.py` + `test_context.py` | 23 + 28 = 51 |
| T18 | LADDER + integration | `cliffguard/ladder/tier.py` + `router.py` | `test_integration_smoke.py` | 13 |

#### §26.2 Phase B — Harness (Scaffolding Complete; Live-Mode Wired)

All 17 Phase B tasks have scaffolding code on disk. Live-mode paths are gated behind `load_model()` / `live_mode=True` so the 939-test Phase A suite still passes unchanged. Real inference activates only on a GPU host with the appropriate binary installed.

| Task | What Was Done | Live Mode | Key File(s) | Tests |
|---|---|---|---|---|
| T19 | TransformersBnbAdapter: `load_model()`, hidden-state hook, `get_top_k_logprobs()` | LIVE (wired) | `cliffguard/engines/transformers_bnb.py` | `test_engines_transformers_bnb.py` (12) |
| T20 | AutoAWQAdapter, VLLMAdapter stubs (Linux-only) | STUB | `cliffguard/engines/autoawq.py`, `vllm.py` | `test_engines_autoawq.py` (6), `test_engines_vllm.py` (8) |
| T21 | LlamaCppAdapter: `load_model()`, embedding/logit callbacks | LIVE (wired) | `cliffguard/engines/llamacpp.py` | `test_engines_llamacpp.py` (14) |
| T22 | Corpus loaders, five-fold structure, SHA-256 fold isolation | STUB | `cliffguard/eval/folds.py` | `test_folds.py` (13) |
| T23 | Refusal-direction extractor (Arditi recipe) | STUB | `cliffguard/eval/refusal_direction.py` | `test_refusal_direction.py` (33) |
| T24 | Harmfulness-direction extractor (Zhao recipe) | STUB | `cliffguard/eval/harmfulness_direction.py` | `test_harmfulness_direction.py` (32) |
| T25 | KenLM trainer (`lmplz` subprocess; NotImplementedError if absent) | LIVE (if binary present) | `cliffguard/eval/kenlm_trainer.py` | `test_kenlm_trainer.py` (19), `test_kenlm_stub.py` (9) |
| T26 | Judge drivers: `RealStrongREJECTJudge`, `RealLlamaGuardJudge` | LIVE (if GPU present) | `cliffguard/eval/judges.py` | `test_judges.py` (37) |
| T27 | Cliff metrics: `geometric_cliff`, `wasserstein_cliff`, `behavioral_cliff`, `detect_cliff_boundary` | STUB | `cliffguard/eval/cliff_metrics.py` | `test_cliff_metrics.py` (43) |
| T28 | BCN-2 dataset constructor: `BCN2Dataset`, `RealParaphraser` | PARTIAL | `cliffguard/eval/bcn2.py` | `test_bcn2.py` (30) |
| T29 | Five-fold orchestrator: Fold A–E stubs + `live_execute_fold_a/b` | PARTIAL | `cliffguard/eval/five_fold_orchestrator.py`, `five_fold_live.py` | `test_five_fold_orchestrator.py` (32) |
| T30 | Statistical analysis: Wilcoxon, KS, Bonferroni (α=0.01), H1/H4/H5 | STUB | `cliffguard/eval/stats.py` | `test_stats.py` (53) |
| T31 | Bandit drift simulator: `adwin_statistic` + `true_adwin_statistic` | PARTIAL | `cliffguard/eval/drift_sim.py` | `test_drift_sim.py` (22) |
| T32 | Figure generation: 4 matplotlib figures | STUB | `cliffguard/eval/figures.py` | `test_figures.py` (24) |
| T33 | Reproducibility manifest builder: git hash + SHA-256, `verify_manifest` | STUB | `cliffguard/eval/repro.py`, `scripts/build_preregistration_manifest.py` | `test_repro.py` (33) |
| T34 | End-to-end dry run: `scripts/dry_run.py` (< 1s, no GPU) | LIVE | `scripts/dry_run.py` | `test_dry_run_e2e.py` (31) |
| T35 | README + runbook: `README.md`, `configs/example.yaml`, `docs/setup.md` | N/A | `README.md`, `docs/setup.md` | `test_readme.py` (22) |

Additional Phase B additions (from post-T35 commits):
- `scripts/run_evaluation_3050.py`: RTX 3050-specific evaluation driver (VRAM detection, model auto-selection)
- `cliffguard/eval/results_writer.py`: per-device per-run artifact directories — 50 tests in `test_results_writer.py`
- `cliffguard/eval/runner.py` — 19 tests in `test_runner.py`
- `cliffguard/eval/calibration.py` — 18 tests in `test_calibration.py`
- `cliffguard/eval/attack_corpus.py` — 20 tests in `test_attack_corpus.py`

---

### §27 What Is Not Done

*An honest accounting of what has not been run, built, or verified — with no fabricated numbers.*

#### §27.1 Phase C — Real Hardware Runs

No real corpus exists on disk. No real GPU inference has been performed for evaluation. All evaluation metrics (ASR, FPR, ABR, cliff metrics) are synthetic in the current codebase.

| Phase C Task | Status | Blocker |
|---|---|---|
| C.1 Fold A calibration on 3 model families × 5 schemes | Not started | Requires GPU host + corpus download |
| C.2 Fold B cliff measurement (H1) | Not started | Requires Fold A complete + adversarial corpus |
| C.3 Fold C defense composition (H2, H3, H4, H5) | Not started | Requires Fold A + B complete |
| C.4 Fold D bandit drift simulation | Not started | Requires Fold A + C complete |
| C.5 Fold E BCN-2 cross-family construction | Not started | Requires Fold A complete + paraphraser API |
| C.6 Statistical analysis, figure generation | Not started | Requires Folds B/C/D/E complete |
| C.7 Paper revision items | Not started | See §27.2 below |

#### §27.2 Paper-Revision Items (from `decisions_log.md`)

These are open items that must be resolved before submission. Each has an entry in `decisions_log.md` with a date and a code reference.

| Item | Blueprint Section | Issue | Code Status | Log Reference |
|---|---|---|---|---|
| `geometric_cliff` range [0, √2] not [0, 1] | §11.3 | Paper says normalised to [0,1]; code produces [0, √2] | Code is correct, paper needs update | C27 |
| KenLM order: 3-gram vs 5-gram | §5.5 | §5.5 is silent on Tier distinction; code uses order=3 for Tier C and order=5 for Tier A/B | Code correct, paper needs to distinguish | C25 |
| Section numbering audit | §12.5, §5.9, §11.2, §12.3 | `kenlm_trainer.py` cites §12.5 (= statistical power), `judges.py` cites §5.9 (= runtime gate), `bcn2.py` cites §11.2/§12.3 | Docstrings kept verbatim; paper audit needed | C25, C26, C28 |
| `preregistration.md` H4/H5 alpha | §H4, §H5 | Preregistration says p < 0.05; Bonferroni requires p < 0.01; `stats.py` uses 0.01 | Code is correct; preregistration needs update | C30 |
| ADWIN vs Page-Hinkley naming | §6.4 | Blueprint specifies ADWIN; Phase B implemented Page-Hinkley proxy + added true_adwin_statistic | Both now present; paper §6.4 must reconcile | C31 |
| Tier C gate set: blueprint DOT vs TIER_GATES | §10 | Blueprint DOT diagram shows VESTIBULE-LZ + PROBE-RM(1L) + PH + CT; code has VESTIBULE-LZ + VESTIBULE-PS + ATTEST-WH only | Code (TIER_GATES) is authoritative; paper needs update | C35 |
| Llama Guard 3 arXiv ID | verified_citations.md | `judges.py` cites 2312.06674 (LG1); correct ID for LG3 is 2411.17713-lineage | Docstring needs update | C26 |
| BCN-2 storage: LSH vs JSONL | §12.2 | Blueprint specifies LSH; code uses plain JSONL | JSONL is Phase A; add LSH before public release | C28 |

---

### §28 Test Coverage Map

*Every test file mapped to its target module, with coverage type and key assertions.*

The current total is **989 tests** (as reported by `uv run pytest --collect-only -q`). The Phase A gate milestone was 939; the additional 50 tests were added by `test_results_writer.py` in a subsequent session. All 989 pass with `mypy --strict` on 53 source files and `ruff check` clean.

| Module / Area | Implementation File | Test File | Tests | Coverage | Key Assertions |
|---|---|---|---|---|---|
| VESTIBULE-LZ | `vestibule/lz.py` | `test_vestibule_lz.py` | 12 | STUB | compression ratio, fires HIGH, degenerate empty input |
| VESTIBULE-PS | `vestibule/ps.py` | `test_vestibule_ps.py` | 19 | STUB | role-override detection, separator 4+ repeats, boundary tokens |
| PROBE-RM | `probe/rm.py` | `test_probe_rm.py` | 17 | STUB | cosine computation, fires LOW, zero-norm raises ValueError |
| PROBE-MT | `probe/mt.py` | `test_probe_mt.py` | 19 | STUB | rho_dot, rho_ddot, min-length ValueError |
| PROBE-HD | `probe/hd.py` | `test_probe_hd.py` | 18 | STUB | fires HIGH (opposite PROBE-RM), harmfulness margin |
| TRIPWIRE-H | `tripwire/h.py` | `test_tripwire_h.py` | 19 | STUB | CUSUM accumulates on entropy drop, fires HIGH |
| TRIPWIRE-R | `tripwire/r.py` | `test_tripwire_r.py` | 19 | STUB | LLR sign and firing direction |
| TRIPWIRE-R calibration | `tripwire/r.py` | `test_tripwire_r_calibration.py` | 8 | STUB | per-scheme threshold application |
| LOOKOUT-CT | `lookout/ct.py` | `test_lookout_ct.py` | 29 | STUB | Bloom filter, exact match, canary count |
| LOOKOUT-JG | `lookout/jg.py` | `test_lookout_jg.py` | 17 | STUB | compliance rate, n<1 ValueError |
| B-PROBE-LOGIT | `bprobe/logit.py` | `test_bprobe_logit.py` | 26 | STUB | logistic score, shape mismatch ValueError |
| B-PROBE-CONSISTENCY | `bprobe/consistency.py` | `test_bprobe_consistency.py` | 25 | STUB | JSD computation, N<2 ValueError, fires LOW |
| ATTEST-WH | `attest/wh.py` | `test_attest_wh.py` | 16 | LIVE | SHA-256 hash, ALLOW/BLOCK/DEGRADED per tier |
| CONDUCTOR bandit | `conductor/bandit.py` | `test_conductor.py` | 23 | STUB | UCB update, aggregate verdict, never-disable floor |
| CONDUCTOR context | `conductor/context.py` | `test_context.py` | 28 | STUB | 14-dim vector, tier indicator, ATTEST mapping |
| LADDER + integration | `ladder/tier.py` + `router.py` | `test_integration_smoke.py` | 13 | STUB | gate sets per tier, white-box/black-box routing |
| Core types | `types.py` | `test_types.py` | 16 | STUB | enum values, CalibrationTable.tau, GateVerdict fields |
| Transformers adapter | `engines/transformers_bnb.py` | `test_engines_transformers_bnb.py` | 12 | LIVE | stub mode raises NotImplementedError; live mode load gate |
| AutoAWQ adapter | `engines/autoawq.py` | `test_engines_autoawq.py` | 6 | STUB | NotImplementedError on all calls |
| vLLM adapter | `engines/vllm.py` | `test_engines_vllm.py` | 8 | STUB | NotImplementedError on all calls |
| llama.cpp adapter | `engines/llamacpp.py` | `test_engines_llamacpp.py` | 14 | LIVE | stub mode raises NotImplementedError; load gate |
| Calibration | `eval/calibration.py` | `test_calibration.py` | 18 | STUB | MIN_CALIBRATION_SIZE guard, per-scheme tau |
| Threshold calibrator | `eval/threshold_calibrator.py` | `test_threshold_calibrator.py` | 22 | STUB | empirical quantile computation |
| KenLM trainer | `eval/kenlm_trainer.py` | `test_kenlm_trainer.py` | 19 | LIVE | NotImplementedError if lmplz absent |
| KenLM stub | `eval/kenlm_stub.py` | `test_kenlm_stub.py` | 9 | STUB | stub interface contract |
| Attack corpus | `eval/attack_corpus.py` | `test_attack_corpus.py` | 20 | STUB | schema validation, SHA-256 entries |
| Runner | `eval/runner.py` | `test_runner.py` | 19 | STUB | fold runner contract |
| Refusal direction | `eval/refusal_direction.py` | `test_refusal_direction.py` | 33 | STUB | Arditi recipe, orthogonality |
| Harmfulness direction | `eval/harmfulness_direction.py` | `test_harmfulness_direction.py` | 32 | STUB | Zhao recipe, orthogonality vs r_hat |
| Cliff metrics | `eval/cliff_metrics.py` | `test_cliff_metrics.py` | 43 | STUB | geometric/behavioral/Wasserstein metrics, boundary detection |
| BCN-2 | `eval/bcn2.py` | `test_bcn2.py` | 30 | PARTIAL | non-circularity assertion, JSONL persistence |
| Five-fold orchestrator | `eval/five_fold_orchestrator.py` | `test_five_fold_orchestrator.py` | 32 | STUB | NotImplementedError stubs satisfied |
| Stats | `eval/stats.py` | `test_stats.py` | 53 | STUB | Wilcoxon, KS, Bonferroni α=0.01 |
| Drift simulator | `eval/drift_sim.py` | `test_drift_sim.py` | 22 | PARTIAL | Page-Hinkley + true ADWIN |
| Judges | `eval/judges.py` | `test_judges.py` | 37 | PARTIAL | stub + RealStrongREJECTJudge + RealLlamaGuardJudge |
| Figures | `eval/figures.py` | `test_figures.py` | 24 | STUB | matplotlib figure generation contract |
| Reproducibility | `eval/repro.py` | `test_repro.py` | 33 | STUB | manifest SHA-256, verify_manifest |
| Dry run E2E | `scripts/dry_run.py` | `test_dry_run_e2e.py` | 31 | LIVE | full pipeline < 1s on any machine |
| Results writer | `eval/results_writer.py` | `test_results_writer.py` | 50 | STUB | per-device per-run artifact dirs |
| Fold isolation | `eval/folds.py` | `test_folds.py` | 13 | STUB | SHA-256 isolation check |
| Orchestrator | `eval/orchestrator.py` | `test_orchestrator.py` | 14 | STUB | orchestrator flow contract |
| Cliff corpus gen | `eval/generate_cliff_corpus.py` | `test_generate_cliff_corpus.py` | 15 | STUB | corpus generation contract |
| Preregistration | `docs/preregistration.md` | `test_preregistration.py` | 11 | LIVE | document exists, hypotheses stated |
| Run evaluation | `scripts/run_evaluation_3050.py` | `test_run_evaluation.py` | 22 | LIVE | script entry point, arg parsing |
| README | `README.md` | `test_readme.py` | 22 | LIVE | README content, links, badges |
| Smoke | `tests/test_smoke.py` | — | 1 | LIVE | assert 1 + 1 == 2 |
| **TOTAL** | | | **989** | | |

**Coverage badge legend:**
- **LIVE**: Real code path tested or real file checked.
- **STUB**: Synthetic data exercises API contract; no real model inference.
- **PARTIAL**: Mix — stub tests + some live code paths (e.g., RealParaphraser in bcn2.py, true_adwin_statistic in drift_sim.py).

---

## Part VIII — How to Operate the System

---

### §29 Smoke Test (No GPU)

*The fastest way to verify CLIFFGUARD is correctly installed — all 12 gates, full CONDUCTOR, synthetic data, no GPU, no corpus, under 1 second.*

```bash
# After cloning and uv sync:
uv run python scripts/dry_run.py --tier A --scheme FP16
```

Expected output (from `docs/setup.md §What to Expect`):

```
Tier: A   Scheme: FP16
Gate                      Fired    Score     Margin
─────────────────────────────────────────────────────
VESTIBULE-LZ              False    0.312     +0.241
VESTIBULE-PS              False    0.000     -0.050
PROBE-RM                  False    0.721     +0.221
PROBE-MT                  False    0.683     +0.183
PROBE-HD                  False    0.412     -0.088
TRIPWIRE-H                False    0.023     -0.477
TRIPWIRE-R                False    0.891     +0.391
LOOKOUT-CT                False    0.000     -0.500
LOOKOUT-JG                False    0.100     -0.400
B-PROBE-LOGIT             False    0.201     -0.299
B-PROBE-CONSISTENCY       False    0.744     +0.244
ATTEST-WH                 False    1.000     +0.500

Context dim: 14   fired_score: 0.000   total_weight: 1.000
Decision: ALLOW
```

**Column annotations:**
- **Gate**: The primitive name.
- **Fired**: Whether this gate's verdict was BLOCK (True) or ALLOW (False).
- **Score**: The raw numeric output of the gate (compression ratio, margin, CUSUM value, etc.).
- **Margin**: `Score - Threshold` — positive means the score is safely away from the firing threshold; negative means the score is near or past the threshold in the non-firing direction.

For Tier C, only 3 gates run (VESTIBULE-LZ, VESTIBULE-PS, ATTEST-WH). For Tier C+, 4 gates run (adds B-PROBE-LOGIT). Test all tiers:

```bash
uv run python scripts/dry_run.py --tier B --scheme GGUF_Q4_K_M
uv run python scripts/dry_run.py --tier C --scheme GGUF_Q3_K_M
uv run python scripts/dry_run.py --tier C_PLUS --scheme GGUF_Q3_K_M
```

If the dry run exits with code 1, paste the full traceback in a GitHub issue. The dry run must complete in under 1 second on any machine.

---

### §30 Real Evaluation on a GPU Host

*The step-by-step procedure for running Fold A and Fold B on real hardware — produces the first real numbers.*

**Step 1: Prerequisites.** See [docs/setup.md](setup.md) for full setup instructions by tier. For RTX 3050 users, see [docs/setup_3050.md](setup_3050.md). Required: Python 3.11+, uv, git ≥ 2.40, CUDA-capable GPU (Tier A) or 8 GB RAM (Tier B).

```bash
git clone https://github.com/parnish007/CLIFFGUARD.git && cd CLIFFGUARD
uv sync --extra gpu
```

**Step 2: Verify smoke test.** Run `uv run python scripts/dry_run.py --tier A --scheme FP16` and confirm ALLOW decision with 12 gates.

**Step 3: Download Fold A corpus.** The Fold A calibration corpus (Anthropic-HH-RLHF + OASST1) is downloaded automatically:

```bash
uv run python scripts/download_fold_a.py --download
```

Data is written to `data/folds/fold_a/`. Follow any printed instructions for datasets requiring manual terms acceptance. Minimum required: 2000 benign prompts per scheme (`MIN_CALIBRATION_SIZE`).

**Step 4: Run Fold A calibration.** Fold A must complete before any other fold. For RTX 3050:

```bash
uv run python scripts/run_evaluation_3050.py --model auto --folds A
```

For Tier A general (RTX 5060 or better):

```bash
uv run python scripts/run_full_evaluation.py --config configs/my_run.yaml --tier A
```

Copy and edit `configs/example.yaml` first:

```bash
cp configs/example.yaml configs/my_run.yaml
# Edit: tier, schemes, data_dir, artifacts_dir, fpr_target, n_calibration
```

**Step 5: Verify Fold A artifacts.** After Fold A, the artifact directory contains:

```
artifacts/
└── runs/
    └── A_<hostname>_<YYYYMMDD_HHMMSS>/
        ├── run_metadata.json
        └── fold_a/
            ├── calibration_summary.json
            ├── gate_verdicts_fold_a.jsonl
            ├── r_hat_<model>_FP16.npz
            └── r_hat_<model>_NF4.npz
```

**Step 6: Assemble adversarial corpus (Fold B).** See `cliffguard/eval/attack_corpus.py` for the corpus schema. Sources: JailbreakBench, AdvBench-50. Assembly is currently manual (see `docs/preregistration.md §3` for required sizes: 500 attack prompts per adversary per scheme).

**Step 7: Run Fold B.** After the adversarial corpus is assembled:

```bash
uv run python scripts/run_full_evaluation.py --config configs/my_run.yaml --folds B
```

Results are written to `artifacts/runs/<run_id>/fold_b/cliff_metrics.json`.

---

### §31 Reproducibility

*Every evaluation run produces a cryptographic manifest that ties the results to a specific codebase state — ensuring the numbers in the paper can be independently verified.*

#### §31.1 The Manifest Builder

After any evaluation fold:

```bash
uv run python scripts/build_preregistration_manifest.py --tier A --schemes FP16 NF4
```

This produces `artifacts/runs/<run_id>/hypothesis_results.json` plus a top-level manifest containing:
- The git commit hash of the current codebase
- SHA-256 of `docs/preregistration.md` (the pre-registered protocol)
- SHA-256 of every artifact file in the run directory
- Tier, schemes, hostname, timestamp

The manifest ties the results to a specific codebase state and a specific pre-registered protocol — any reviewer can verify that the evaluation was run on the pre-registered code, not a modified version.

#### §31.2 The Compare Runs Script

To compare two runs (e.g., on different hardware or with different model versions):

```bash
uv run python scripts/compare_runs.py \
  artifacts/runs/A_host1_20260518_103014/ \
  artifacts/runs/A_host2_20260519_091022/
```

This produces a diff of the cliff metrics, FPR values, and gate verdicts — allowing independent replication to be verified quantitatively.

#### §31.3 SHA-256 Over Every Artifact

Every artifact file written by `results_writer.py` is included in the manifest hash. This means if any result file is modified after the fact, the manifest verification step (`verify_manifest()` in `cliffguard/eval/repro.py`) will detect the modification and raise an error. This provides tamper-evidence for the evaluation artifacts equivalent to ATTEST-WH's tamper-evidence for the model weights.

---

## Part IX — Reference Material

---

### §32 Glossary

*Every abbreviation and technical term used in the paper and codebase.*

| Term | Definition |
|---|---|
| **ABR** | Attack Block Rate — the fraction of attack prompts that CLIFFGUARD blocks. Complement of ASR. |
| **ADWIN** | Adaptive WINdowing — the Bifet-Gavaldà (2007) streaming change-point detection algorithm. In the codebase: `true_adwin_statistic` in `eval/drift_sim.py`; the Phase A proxy is Page-Hinkley (`adwin_statistic`). |
| **ASR** | Attack Success Rate — the fraction of attack prompts that bypass CLIFFGUARD and elicit a compliant response. |
| **ATTEST** | Boot-time SHA-256 weight attestation component — one of the 8 named CLIFFGUARD components. |
| **ATTEST-WH** | The specific primitive within ATTEST that hashes the GGUF/safetensors blob against a vendor manifest. |
| **AWQ-INT4** | Activation-aware Weight Quantization, 4-bit integer. Requires `autoawq`, Linux-only. |
| **BCN-2** | Below-Cliff Naturals, N=2 — a paired dataset of prompts where the Q3_K_M model complies but the FP16 model refuses, constructed with cross-family paraphrasers. |
| **B-PROBE** | Black-box fallback observer — uses top-k logprobs only, no hidden-state access. Includes B-PROBE-LOGIT and B-PROBE-CONSISTENCY. |
| **CONDUCTOR** | The LinUCB contextual bandit that orchestrates gate weights online. One of the 8 named components. |
| **CONTEXT_DIM** | The dimensionality of the CONDUCTOR feature vector: 14. Defined in `conductor/context.py`. |
| **CUSUM** | Cumulative Sum control chart — a sequential change-point detection statistic. Used in TRIPWIRE-H. |
| **EXP3.S** | Exponential-weight algorithm for Exploration and Exploitation, Switching variant — the minimax-regret fallback for CONDUCTOR under non-stationary attack distributions. |
| **FP16** | Float16 precision — the full-precision baseline against which quantized models are compared. |
| **FPR** | False Positive Rate — the rate at which benign prompts are incorrectly blocked by a gate. |
| **GateVerdict** | The return type of every `evaluate()` function: `{gate: str, fired: bool, score: float, threshold: float, tier: Tier, threat_model: Any}`. Defined in `cliffguard/types.py`. |
| **G1** | The geometric assumption (blueprint §3.4): the safety subspace is an approximately isometric projection of the full residual-stream space for harmful vs. benign prompts. |
| **GGUF** | GPT-Generated Unified Format — the on-disk format used by llama.cpp and Ollama for quantized models. |
| **GCG** | Greedy Coordinate Gradient — the Zou et al. (2023) method for generating adversarial suffixes by gradient descent. |
| **H1–H5** | The five pre-registered hypotheses. See §13–§17. |
| **IQ3_XXS** | A very aggressive 3-bit GGUF quantization format, even more compressed than Q3_K_M. |
| **JSD** | Jensen-Shannon Divergence — a symmetric, bounded measure of difference between probability distributions. Used in B-PROBE-CONSISTENCY. |
| **κ (kappa)** | The pre-registered cliff threshold: κ = 0.25. A jump in Δ_cliff ≥ 0.25 constitutes a cliff. |
| **KenLM** | An n-gram language model library used for TRIPWIRE-R's reference distribution. Order 5 for Tier A/B, order 3 for Tier C/C+ (decisions_log C25). |
| **LADDER** | The static tier router — selects which gates run on each hardware class. One of the 8 named components. |
| **LinUCB** | Linear Upper Confidence Bound — the contextual bandit algorithm used in CONDUCTOR. |
| **LOOKOUT** | The output-side monitor component — includes LOOKOUT-CT (canary tokens) and LOOKOUT-JG (compliance judge). |
| **Margin** | The return type for PROBE gates and B-PROBE: `{value: float, scheme: QuantScheme, primitive: str, layer: int | None}`. |
| **MIN_CALIBRATION_SIZE** | Minimum Fold A corpus size: 2000 benign prompts. Below this, KS estimation error is too wide. |
| **MIN_WEIGHT** | The floor weight for never-disable arms in the CONDUCTOR: 0.1. |
| **MVDS** | Multi-Valued Defense System — the four-tier hardware architecture (A/B/C/C+). |
| **NF4** | NormalFloat 4-bit — the bitsandbytes quantization format used by QLoRA and supported by transformers. |
| **PROBE** | White-box residual-stream observer component — includes PROBE-RM, PROBE-MT, PROBE-HD. |
| **Q3_K_M** | 3-bit GGUF quantization with K-means clustering (medium variant). The primary cliff-level format. |
| **Q4_K_M** | 4-bit GGUF quantization with K-means clustering (medium variant). Below the cliff for most models. |
| **RKNN W8A8** | Rockchip NPU format: 8-bit weights, 8-bit activations. Used on RK3588 boards. |
| **τ_q** (tau_q) | The per-scheme calibrated threshold for a gate. Computed at the empirical (1-α) or α quantile of the benign score distribution, depending on firing direction. |
| **t_inst** | The position of the last user-instruction token in the residual stream. Used by PROBE-HD and harmfulness direction. |
| **t_post-inst** | The position of the first post-instruction token in the residual stream. Used by PROBE-RM and refusal direction. |
| **TPR** | True Positive Rate — the rate at which attack prompts are correctly blocked. Explicitly NOT decoupled across schemes (the cliff is where TPR collapses). |
| **TRIPWIRE** | Streaming entropy and reference-ratio monitor component — includes TRIPWIRE-H (CUSUM) and TRIPWIRE-R (KenLM ratio). |
| **VESTIBULE** | Input-gate stack component — includes VESTIBULE-LZ (compression) and VESTIBULE-PS (provenance spotlighting). |
| **ε (epsilon)** | The FPR decoupling tolerance: ε = 0.02. The FPR of a calibrated gate must stay within this of the target FPR across all schemes. |
| **Δ_cliff** | The geometric cliff metric: cosine distance between the FP16 and quantized refusal directions. Range [0, √2] (not [0, 1] — decisions_log C27). |
| **Δ_B-cliff** | The behavioral cliff metric: attack-success-rate difference between FP16 and quantized scheme. |
| **Δ_W-cliff** | The Wasserstein cliff metric: Wasserstein-1 distance between the FP16 and quantized refusal-margin distributions. |

---

### §33 Reading Order for the Rest of the Docs

*Which documentation to read depending on your role.*

#### Quick path (1 hour)
For a first understanding of CLIFFGUARD with no prior context:
1. **§1 TL;DR** (this document) — the one-page summary.
2. **§2 The Safety Cliff Problem** (this document) — why this exists.
3. **§3 What CLIFFGUARD Is** (this document) — precise scope.
4. **§29 Smoke test** (this document) — run it and see it work.
5. **[docs/what_is_it.md](what_is_it.md)** — the FAQ for any remaining questions.

#### Researcher path (half day)
Understanding the threat model, architecture, and hypotheses well enough to assess the claims:
1. **Part II — Threat Model** (§4–§6, this document)
2. **Part III — Architecture** (§7–§11, this document, especially §8 primitives)
3. **Part V — Hypotheses** (§13–§18, this document)
4. **Part VI — Folds** (§19–§25, this document)
5. **[docs/math.md](math.md)** — formal definitions of all three cliff metrics and Theorem 14.1.
6. **[docs/preregistration.md](preregistration.md)** — the locked pre-registration.

#### Engineer path (half day)
Wiring real inference engines and running Phase C:
1. **§7 Components** and **§8 Primitives** (this document)
2. **[docs/engineering_reference.md](engineering_reference.md)** — Phase B wiring checklist, engine adapter APIs, data type contracts, error handling contracts.
3. **[docs/setup.md](setup.md)** — device-specific setup.
4. **Part VII — Implementation Status** (§26–§28, this document) — what to implement vs. what's already done.
5. **Part VIII — Operations** (§29–§31, this document) — running Fold A–E.

#### Paper reviewer path (2 hours)
Assessing the empirical claims and honesty of scope:
1. **§1 TL;DR** (this document) — status overview.
2. **Part V — Hypotheses** (§13–§18, this document) — all five hypotheses and Theorem 14.1.
3. **§27 What Is Not Done** (this document) — the honest paper-revision table.
4. **[docs/preregistration.md](preregistration.md)** — the locked protocol.
5. **[docs/math.md](math.md)** — formal cliff metric definitions.
6. **§28 Test Coverage Map** (this document) — which numbers are synthetic vs. real.

---

### §34 Where to Ask Questions

*Pointers to the right place for each type of question.*

**Technical implementation questions (how does X work, why does Y fire HIGH):**
- This document first (§8 has all 12 primitives with full detail).
- `cliffguard-unified-paper.md` (the 100-page blueprint) for anything not covered here — every claim traces to a section.
- `docs/engineering_reference.md` for Phase B wiring specifics.

**Deviation from the blueprint (code says X, paper says Y):**
- `decisions_log.md` is the tiebreaker. Every material divergence from the blueprint has an entry with date, reasoning, and `Affects:` note.

**What's planned vs. what's built:**
- `development.md` — the Phase A/B/C task tracker. Phase A tasks have `[x]`; Phase C has `[ ]`.

**Experimental results (I ran Fold X and got Y):**
- File a GitHub issue at `https://github.com/parnish007/CLIFFGUARD/issues` with the run artifact directory attached.
- Include `artifacts/runs/<run_id>/run_metadata.json` and the relevant `*_metrics.json` files.

**Pre-registration deviations (I need to change a threshold or metric):**
- Any deviation from `docs/preregistration.md` must be documented in `decisions_log.md` with a justification *before* the affected fold runs. No post-hoc threshold adjustment is permitted.

---

*Last updated: 2026-05-19; based on `cliffguard-unified-paper.md` at commit 3baf8f8. Run `git log --oneline -1` in the repo root for the current commit hash.*

