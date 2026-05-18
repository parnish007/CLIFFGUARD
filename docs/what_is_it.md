<div align="center">

[← README](../README.md) &nbsp;|&nbsp;
[What is it](what_is_it.md) &nbsp;|&nbsp;
[Architecture](architecture.md) &nbsp;|&nbsp;
[Math](math.md) &nbsp;|&nbsp;
[Setup](setup.md) &nbsp;|&nbsp;
[Engineering Ref](engineering_reference.md)

</div>

# What is CLIFFGUARD?

> **TL;DR:** Quantized edge LLMs have a safety cliff near Q3_K_M
> where RLHF alignment collapses. CLIFFGUARD is a stateless defense
> system placed in front of the model — 11 gates, 4 hardware tiers,
> online RL adaptation — that works without retraining the model or
> accessing model weights.

## The Problem

Large language models deployed at the edge are quantized to 4-bit or 3-bit precision to fit within 2–8 GB of RAM. Post-training quantization degrades RLHF safety alignment non-linearly: a model that reliably refuses harmful requests at FP16 may silently comply at Q3_K_M, not because general capability degrades proportionally (MMLU typically drops ~8 points), but because the refusal direction in the residual stream narrows and the margin between harmful and harmless prompt representations collapses. This boundary — empirically near Q3_K_M for Llama-3 and Mistral families — is the **safety cliff**. Egashira et al. (ICLR 2025) demonstrated attack-success-rate deltas of up to 88.7 % on GGUF-quantized models; Hong et al. (2403.15447) showed a ~50-point toxicity-safety drop at GPTQ-3-bit. A prompt injection exploiter who knows a deployment uses Q3_K_M can craft prompts that fall precisely into the cliff region — prompts that an FP16 model would refuse but the quantized deployment silently complies with.

## What CLIFFGUARD Does

CLIFFGUARD is a stateless, online-RL-adapted defense system placed in front of the quantized model. It does not modify model weights, retrain the model, or require any change to the model checkpoint. Instead it operates on input strings and summary statistics of model outputs (residual-stream projections, top-k logprobs, token entropy), and applies a per-quantization calibration map so that the false-positive rate of each gate is independent of the quantization scheme. Eight named components implement the full defense:

<table>
<tr>
<td width="50%" valign="top">

**VESTIBULE** — input gates, run before inference
- `VESTIBULE-LZ`: compression-ratio anomaly (GCG/AutoDAN suffixes)
- `VESTIBULE-PS`: provenance spotlight (injected-content structure)

**PROBE** — residual-stream gates, white-box only
- `PROBE-RM`: refusal-direction margin (Arditi et al.)
- `PROBE-MT`: margin trajectory derivative (drift toward compliance)
- `PROBE-HD`: harmfulness-direction margin (Zhao et al.)

**B-PROBE** — black-box fallback (top-k logprobs only)
- `B-PROBE-LOGIT`: logistic head on first-token logprob vector
- `B-PROBE-CONSISTENCY`: JSD across N paraphrases

</td>
<td width="50%" valign="top">

**TRIPWIRE** — streaming monitors during generation
- `TRIPWIRE-H`: entropy CUSUM (depression → spike pattern)
- `TRIPWIRE-R`: KenLM Neyman-Pearson LLR vs benign reference

**LOOKOUT** — output monitors, post-generation
- `LOOKOUT-CT`: canary token leakage check
- `LOOKOUT-JG`: compliance judge (Llama Guard 3)

**CONDUCTOR** — LinUCB bandit, adapts gate weights online.
ADWIN drift detection triggers weight resets.

**LADDER** — static tier router. Returns ordered gate list
per hardware tier and observability mode.

**ATTEST** — boot-time SHA-256 weight attestation.
Defends against poisoned-weight supply-chain attacks (A2).

</td>
</tr>
</table>

## What CLIFFGUARD Does NOT Do

CLIFFGUARD does not modify, retrain, or fine-tune the protected model. It does not claim to eliminate all prompt injection — it raises the cost for each adversary class and provides honest statements of where each tier's defenses become structurally weak (see H5 and Tier C scope). It does not require white-box access in all modes: B-PROBE provides a black-box fallback that extends the FPR-portability guarantee (with honestly acknowledged TPR loss) to closed-weight API endpoints. The system is not a content classifier that can be bypassed by rephrasing — the core signal (refusal-direction margin, token entropy trajectory) is derived from the model's own internal geometry, not from surface-level text patterns.

## The Nine Adversary Classes

CLIFFGUARD models nine adversary classes from the Greshake-style (arXiv:2302.12173) hierarchy, extended for edge-quantized deployments:

| ID | Name | Attack Mechanism | Primary Gate(s) |
|---|---|---|---|
| A1 | Direct injector | Natural-language hijacks ("ignore previous instructions"), DAN, role-play jailbreaks | VESTIBULE-PS, LOOKOUT-JG |
| A2 | Indirect / poisoned-weight | Instructions embedded in RAG/tool outputs; Egashira-style GGUF blob substitution | VESTIBULE-PS, ATTEST-WH |
| A3 | Optimizer | GCG / AutoDAN adversarial suffixes with high perplexity and low compressibility | VESTIBULE-LZ, PROBE-RM |
| A4 | Iterator | PAIR / TAP / Crescendo — black-box query iteration to refine injections | TRIPWIRE-H, LOOKOUT-JG |
| A5 | Scaler | Best-of-N sampling, many-shot, randomised augmentation exploiting sampling variance | TRIPWIRE-H, LOOKOUT-CT |
| A6 | Encoder | ArtPrompt ASCII art, bijection learning, base64/cipher, low-resource-language jailbreaks | TRIPWIRE-R, VESTIBULE-LZ |
| A7 | Quantization-cliff exploiter | Natural-language prompts whose refusal margin collapses only at Q3_K_M or below | LADDER + ATTEST + cross-tier rollback |
| A8 | Defender-aware adversary | Knows CLIFFGUARD is deployed; white-box access to bandit weights and calibration tables | CONDUCTOR safe-rollback |
| A9 | Closed-weight black-box endpoint | Targets API endpoints where only top-k logprobs are observable | B-PROBE-LOGIT, B-PROBE-CONSISTENCY |

**Tier C honest scope:** Tier C (2 GB embedded, 3 gates) is **not defended against A7**. H5 pre-registers that Tier C will show no statistically significant ABR reduction against the cliff exploiter. Tier C+ adds PromptGuard-2-22M-INT4 (B-PROBE-LOGIT) and is expected to show significant ABR reduction (H5 Tier C+ claim).

## How CLIFFGUARD Differs from Prior Work

None of the existing prompt-injection defenses were designed for the quantized edge setting:

| System | Approach | Gap vs. CLIFFGUARD |
|---|---|---|
| Llama Guard (Meta) | Safety classifier on prompt + response | Does not observe residual-stream geometry; degrades if quantized; no streaming detection |
| PromptGuard-2-22M (Meta) | DeBERTa-xsmall injection classifier | Surface-pattern only; no geometry signal; CLIFFGUARD uses it as one gate in C+ tier |
| NeMo Guardrails (NVIDIA) | Rule-based + LLM-as-judge | No cliff-awareness; no per-quantization calibration; LLM-as-judge is expensive on edge |
| Rebuff | Perplexity gate + canary injection | Perplexity alone must be re-tuned per quantization; CLIFFGUARD adds ten more signals |
| SecAlign (Stanford) | Fine-tunes model to prefer secure instructions | Modifies weights — incompatible with edge deployment constraints; cliff can re-emerge |
| Constitutional Classifiers (Anthropic) | Classifier trained on constitutional rules | Cloud-scale; no edge or quantization support |
| LlamaFirewall (Meta) | Composable pipeline of Llama Guard + PromptGuard + CodeShield | No residual-stream geometry; no RLHF-cliff awareness; no bandit adaptation |
| CaMeL (Google DeepMind) | Capability-based language for tool security | Addresses indirect injection, not the quantization cliff; requires model cooperation |
| AEGIS (ACL 2024) | Hedge-style combination of full LLM safety experts | Uses full LLMs as experts (expensive); no quantization-aware calibration |

**What CLIFFGUARD adds:** (1) per-quantization-scheme calibration so FPR is portable across bit-widths; (2) residual-stream geometry signals (PROBE-RM, PROBE-HD) that observe the refusal subspace directly rather than classifying surface text; (3) streaming CUSUM/EWMA (TRIPWIRE) that does not require buffering full outputs; (4) a contextual bandit (CONDUCTOR) that adapts gate weights online without storing user payloads; (5) honest pre-registration of where each tier's defenses structurally fail.

## FAQ

**Q: Does this require a GPU?**
**A (short):** No — Phase A runs on any laptop in under a second. Phase B inference requires tier-matched hardware.

Phase A scaffolding runs on any machine with Python 3.11+ and uv. `uv run python scripts/dry_run.py --tier A --scheme FP16` completes in under a second on any laptop. Phase B inference requires hardware matching the target tier — see [docs/setup.md](setup.md).

---

**Q: What models are supported?**
**A (short):** Tier A: 7–9B NF4/AWQ-INT4. Tier B: 1.5–3B GGUF. Tier C/C+: ≤ 1.5B Q3_K_M or RKNN W8A8. Closed-weight APIs via B-PROBE black-box path.

Tier A: 7–9B models in NF4 or AWQ-INT4, via `transformers` + `bitsandbytes` or `autoawq`. Tier B: 1.5B–3B GGUF models via `llama-cpp-python`. Tier C / C+: ≤ 1.5B GGUF Q3_K_M via `llama-cpp-python` or RKNN W8A8 via the board-specific runtime. Closed-weight APIs (OpenAI, Anthropic, Gemini) via the B-PROBE black-box path (top-k logprobs only, no hidden states).

---

**Q: What is the safety cliff?**
**A (short):** The quantization boundary near Q3_K_M where both Δ_cliff and Δ_B-cliff jump by κ ≥ 0.25 — a model that refused at FP16 now complies.

The quantization boundary — empirically near Q3_K_M for Llama-3 and Mistral — where both the geometric refusal-direction metric Δ_cliff and the behavioral attack-success-rate metric Δ_B-cliff shift discontinuously by κ ≥ 0.25. Below the cliff, a model that refused harmful requests at higher bit-width now complies with them. This is hypothesis H1 in the pre-registered evaluation.

---

**Q: What does FPR decoupling mean?**
**A (short):** Each gate's false-positive rate stays within ε = 0.02 of the target across all schemes after per-scheme calibration. TPR is **not** decoupled.

The false-positive rate of a write-side gate is independent of the quantization scheme up to a per-scheme calibration map. In practice: once we calibrate the gate threshold on a held-out benign corpus for each scheme, the empirical FPR stays within ε = 0.02 of the target across FP16, NF4, AWQ-INT4, Q4_K_M, and Q3_K_M. TPR is **not** decoupled — it may collapse in the cliff regime, because the gate's signal (the refusal-direction margin) collapses with the model's alignment. This is the core theorem underlying the system's design (H2 and H3).

---

**Q: Is the system pre-registered?**
**A (short):** Yes — all five hypotheses, thresholds, and the analysis plan are locked in `docs/preregistration.md` before any data collection.

All five hypotheses (H1–H5), thresholds (κ = 0.25, ε = 0.02, α_corrected = 0.01), acceptance criteria, and the statistical analysis plan are fixed in `docs/preregistration.md` before any data collection or model inference. The document is SHA-256 hashed and recorded in every reproducibility manifest produced by `scripts/build_preregistration_manifest.py`. Any deviation must be documented in `decisions_log.md` before the affected fold runs.

---

**Q: What is BCN-2?**
**A (short):** Below-Cliff Naturals, N=2 — paired prompts where the Q3_K_M model complies and the FP16 model refuses. Built with a cross-family paraphraser to avoid circularity.

Below-Cliff Naturals, N=2. A paired dataset of prompts near the FP16 refusal boundary that cross it at Q3_K_M: the Q3_K_M model complies, the FP16 model refuses. The dataset is constructed using a paraphraser from a **different** model family than the one being cliff-tested — a non-circularity discipline designed to prevent the cliff metric from being self-referentially validated. BCN-2 construction is Fold E of the evaluation.

---

**Q: What is the CONDUCTOR?**
**A (short):** A LinUCB bandit with a 14-dimension context vector that adapts gate weights online. No user payload stored. ADWIN drift detection resets weights when the adversary shifts strategy.

A LinUCB contextual bandit (Chu et al. 2011) that selects gate weights online from sparse incident feedback (reward +1 correct block, -1 miss, -0.2 false positive). The feature vector has 14 dimensions: 12 gate scores, 1 ATTEST result, 1 tier indicator. No user payload is ever stored — only aggregate scalars. ADWIN-based drift detection triggers partial weight resets when the adversary shifts strategy (e.g., learns the bandit's current arm weights and shifts to the weakest gate). EXP3.S provides a minimax-regret fallback under coordinated attack campaigns.

---

<div align="center">

[← Back to README](../README.md) &nbsp;·&nbsp;
[Open an issue](https://github.com/YOUR_USERNAME/CLIFFGUARD/issues) &nbsp;·&nbsp;
[preregistration.md](preregistration.md)

</div>
