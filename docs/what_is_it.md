# What is CLIFFGUARD?

## The Problem

Large language models deployed at the edge are quantized to 4-bit or 3-bit precision to fit within 2–8 GB of RAM. Post-training quantization degrades RLHF safety alignment non-linearly: a model that reliably refuses harmful requests at FP16 may silently comply at Q3_K_M, not because general capability degrades proportionally (MMLU typically drops ~8 points), but because the refusal direction in the residual stream narrows and the margin between harmful and harmless prompt representations collapses. This boundary — empirically near Q3_K_M for Llama-3 and Mistral families — is the **safety cliff**. Egashira et al. (ICLR 2025) demonstrated attack-success-rate deltas of up to 88.7 % on GGUF-quantized models; Hong et al. (2403.15447) showed a ~50-point toxicity-safety drop at GPTQ-3-bit. A prompt injection exploiter who knows a deployment uses Q3_K_M can craft prompts that fall precisely into the cliff region — prompts that an FP16 model would refuse but the quantized deployment silently complies with.

## What CLIFFGUARD Does

CLIFFGUARD is a stateless, online-RL-adapted defense system placed in front of the quantized model. It does not modify model weights, retrain the model, or require any change to the model checkpoint. Instead it operates on input strings and summary statistics of model outputs (residual-stream projections, top-k logprobs, token entropy), and applies a per-quantization calibration map so that the false-positive rate of each gate is independent of the quantization scheme. Eight named components implement the full defense:

- **VESTIBULE** runs before any model inference. Two gates: VESTIBULE-LZ (compression-ratio anomaly detection — adversarial suffixes from GCG/AutoDAN have characteristic low compressibility) and VESTIBULE-PS (provenance-aware spotlight — flags inputs with injected-content structure). These are pure input-string operations, quantization-agnostic.
- **PROBE** runs after the model's forward pass on the user prompt, before generation. Three gates: PROBE-RM projects the post-instruction hidden state onto the refusal direction (Arditi et al. 2406.11717) to measure the refusal margin; PROBE-MT tracks the trajectory of that margin across decoding steps (falling margin = model drifting toward compliance mid-generation); PROBE-HD projects the user-instruction hidden state onto the harmfulness direction (Zhao et al. 2507.11878) to detect request-stage harmfulness. These three gates require white-box residual-stream access.
- **B-PROBE** is the black-box fallback for closed-weight API endpoints or edge accelerators that expose only top-k logprobs. B-PROBE-LOGIT applies a logistic head to the first-token log-probability vector; B-PROBE-CONSISTENCY measures Jensen-Shannon divergence across N paraphrases of the input (low divergence = consistent compliance = suspicious).
- **TRIPWIRE** monitors the model's own token stream during generation. TRIPWIRE-H applies a one-sided CUSUM on per-token entropy (entropy depression followed by a spike is characteristic of in-context injection); TRIPWIRE-R computes a Neyman-Pearson log-likelihood ratio against a fixed KenLM benign reference (low ratio = input is unlikely under the benign distribution = adversarial encoding suspected).
- **LOOKOUT** monitors the model's output after generation. LOOKOUT-CT injects per-session canary tokens into the system prompt and checks whether they reappear in the output (leaked canary = successful prompt injection). LOOKOUT-JG runs a compliance judge (Llama Guard 3-1B-INT4 or DeBERTa-86M) to classify the final output.
- **CONDUCTOR** is a LinUCB contextual bandit that adapts gate weights online from sparse incident feedback, without storing any user payload. ADWIN drift detection triggers weight resets when the adversary shifts strategy. The feature vector has 14 dimensions drawn from all gate scores plus a tier indicator.
- **LADDER** is the static tier router. Given the hardware tier and observability mode (white-box or black-box), it returns the ordered list of gates to run. This is configuration, not learning.
- **ATTEST** is the boot-time weight-hash attestation. ATTEST-WH computes SHA-256 over the GGUF or safetensors blob and compares against a signed vendor manifest. This defends against Egashira-style poisoned-weight attacks (A2) at the supply-chain layer.

## What CLIFFGUARD Does NOT Do

CLIFFGUARD does not modify, retrain, or fine-tune the protected model. It does not claim to eliminate all prompt injection — it raises the cost for each adversary class and provides honest statements of where each tier's defenses become structurally weak (see H5 and Tier C scope). It does not require white-box access in all modes: B-PROBE provides a black-box fallback that extends the FPR-portability guarantee (with honestly acknowledged TPR loss) to closed-weight API endpoints. The system is not a content classifier that can be bypassed by rephrasing — the core signal (refusal-direction margin, token entropy trajectory) is derived from the model's own internal geometry, not from surface-level text patterns.

## FAQ

**Q: Does this require a GPU?**  
A: Phase A scaffolding runs on any machine with Python 3.11+ and uv. `uv run python scripts/dry_run.py --tier A --scheme FP16` completes in under a second on any laptop. Phase B inference requires hardware matching the target tier — see [docs/setup.md](setup.md).

**Q: What models are supported?**  
A: Tier A: 7–9B models in NF4 or AWQ-INT4, via `transformers` + `bitsandbytes` or `autoawq`. Tier B: 1.5B–3B GGUF models via `llama-cpp-python`. Tier C / C+: ≤ 1.5B GGUF Q3_K_M via `llama-cpp-python` or RKNN W8A8 via the board-specific runtime. Closed-weight APIs (OpenAI, Anthropic, Gemini) via the B-PROBE black-box path (top-k logprobs only, no hidden states).

**Q: What is the safety cliff?**  
A: The quantization boundary — empirically near Q3_K_M for Llama-3 and Mistral — where both the geometric refusal-direction metric Δ_cliff and the behavioral attack-success-rate metric Δ_B-cliff shift discontinuously by κ ≥ 0.25. Below the cliff, a model that refused harmful requests at higher bit-width now complies with them. This is hypothesis H1 in the pre-registered evaluation.

**Q: What does FPR decoupling mean?**  
A: The false-positive rate of a write-side gate is independent of the quantization scheme up to a per-scheme calibration map. In practice: once we calibrate the gate threshold on a held-out benign corpus for each scheme, the empirical FPR stays within ε = 0.02 of the target across FP16, NF4, AWQ-INT4, Q4_K_M, and Q3_K_M. TPR is **not** decoupled — it may collapse in the cliff regime, because the gate's signal (the refusal-direction margin) collapses with the model's alignment. This is the core theorem underlying the system's design (H2 and H3).

**Q: Is the system pre-registered?**  
A: Yes. All five hypotheses (H1–H5), thresholds (κ = 0.25, ε = 0.02, α_corrected = 0.01), acceptance criteria, and the statistical analysis plan are fixed in `docs/preregistration.md` before any data collection or model inference. The document is SHA-256 hashed and recorded in every reproducibility manifest produced by `scripts/build_preregistration_manifest.py`. Any deviation must be documented in `decisions_log.md` before the affected fold runs.

**Q: What is BCN-2?**  
A: Below-Cliff Naturals, N=2. A paired dataset of prompts near the FP16 refusal boundary that cross it at Q3_K_M: the Q3_K_M model complies, the FP16 model refuses. The dataset is constructed using a paraphraser from a **different** model family than the one being cliff-tested — a non-circularity discipline designed to prevent the cliff metric from being self-referentially validated. BCN-2 construction is Fold E of the evaluation.

**Q: What is the CONDUCTOR?**  
A: A LinUCB contextual bandit (Chu et al. 2011) that selects gate weights online from sparse incident feedback (reward +1 correct block, -1 miss, -0.2 false positive). The feature vector has 14 dimensions: 12 gate scores, 1 ATTEST result, 1 tier indicator. No user payload is ever stored — only aggregate scalars. ADWIN-based drift detection triggers partial weight resets when the adversary shifts strategy (e.g., learns the bandit's current arm weights and shifts to the weakest gate). EXP3.S provides a minimax-regret fallback under coordinated attack campaigns.
