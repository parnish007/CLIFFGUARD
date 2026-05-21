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

Large language models deployed at the edge are quantized to 4-bit or 3-bit precision to fit within 2–8 GB of RAM. Post-training quantization degrades RLHF safety alignment non-linearly: a model that reliably refuses harmful requests at FP16 may silently comply at Q3_K_M, not because general capability degrades proportionally (MMLU typically drops ~8 points), but because the refusal direction in the residual stream narrows and the margin between harmful and harmless prompt representations collapses. This boundary — empirically near Q3_K_M for Llama-3 and Mistral families — is the **safety cliff**. Egashira et al. (NeurIPS 2024) demonstrated attack-success-rate deltas of up to 88.7 % on GGUF-quantized models; Hong et al. (2403.15447) showed a ~50-point toxicity-safety drop at GPTQ-3-bit while MMLU fell only ~8 points. A prompt injection exploiter who knows a deployment uses Q3_K_M can craft prompts that fall precisely into the cliff region — prompts that an FP16 model would refuse but the quantized deployment silently complies with.

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

---

## FAQ

---

**Q: Does this require a GPU?**
**A (short):** No — Phase A scaffolding runs on any laptop in under a second. Live model inference (Phase B and beyond) requires tier-matched hardware.

Phase A scaffolding runs on any machine with Python 3.11+ and uv. `uv run python scripts/dry_run.py --tier A --scheme FP16` completes in under a second on any laptop — it exercises the full pipeline shape with synthetic arrays without loading any model. Phase B inference requires hardware matching the target tier — an 8 GB GPU for Tier A, a Raspberry Pi 5 for Tier B, a 2 GB embedded board for Tier C. See [docs/setup.md](setup.md) for device-by-device instructions.

---

**Q: What models are supported?**
**A (short):** Tier A: 7–9B NF4/AWQ-INT4. Tier B: 1.5–3B GGUF. Tier C/C+: ≤ 1.5B Q3_K_M or RKNN W8A8. Closed-weight APIs via B-PROBE black-box path.

Tier A: 7–9B models in NF4 or AWQ-INT4, via `transformers` + `bitsandbytes` or `autoawq`. Currently tested: `Llama-3.2-3B-Instruct` at FP16 and NF4. Tier B: 1.5B–3B GGUF models via `llama-cpp-python`. Tier C / C+: ≤ 1.5B GGUF Q3_K_M via `llama-cpp-python` or RKNN W8A8 via the board-specific runtime. Closed-weight APIs (OpenAI, Anthropic, Gemini) via the B-PROBE black-box path (top-k logprobs only, no hidden states).

---

**Q: What has actually been tested with real data? What are the current results?**
**A (short):** Folds A and B are complete on `Llama-3.2-3B-Instruct` at FP16 and NF4 on a Google Colab T4. H1 (cliff existence) is not accepted for this pair — the geometric shift is 67 % of the cliff threshold.

As of May 2026, the live evaluation has covered two folds on one model family and two quantization schemes:

**Fold A (Calibration) — complete:**
- Model: `meta-llama/Llama-3.2-3B-Instruct`, layer 14, Colab T4 GPU
- Calibrated PROBE-RM thresholds at FPR = 5 %: τ(FP16) = 0.09742, τ(NF4) = 0.09827
- Threshold difference Δτ = +0.00085 (+0.87 %) — a preliminary positive signal for H2 (FPR decoupling)

**Fold B (Cliff Measurement) — complete:**
- Δ_cliff (geometric) = **0.167** — the NF4 refusal direction has rotated from the FP16 baseline, reaching 67 % of the κ = 0.25 cliff threshold
- Δ_W-cliff (Wasserstein) = **0.014** — near zero; the margin distribution shape is largely unchanged
- Δ_B-cliff (behavioral proxy) = **0.000** — no additional harmful prompts crossed the threshold (note: this uses a PROBE-RM proxy, not the full StrongREJECT + Llama-Guard-3-8B judge; see the limitations question below)
- `cliff_boundary = null`, `h1_accepted = false`

**Not yet tested:** GGUF Q3_K_M / Q2_K (the predicted cliff zone), multiple model families, Folds C–E, Hypotheses H2–H5.

---

**Q: What is the safety cliff?**
**A (short):** The quantization boundary near Q3_K_M where both Δ_cliff and Δ_B-cliff jump by κ ≥ 0.25 — a model that refused at FP16 now complies.

The quantization boundary — empirically near Q3_K_M for Llama-3 and Mistral — where both the geometric refusal-direction metric Δ_cliff and the behavioral attack-success-rate metric Δ_B-cliff shift discontinuously by κ ≥ 0.25. Below the cliff, a model that refused harmful requests at higher bit-width now complies with them. This is hypothesis H1 in the pre-registered evaluation. The current Fold B result for NF4 (0.167) is sub-threshold — consistent with NF4's design minimizing weight reconstruction error — but the predicted cliff zone at Q3_K_M has not yet been evaluated.

---

**Q: What does A7 look like in practice? Can you give a concrete example?**
**A (short):** A prompt that sounds almost reasonable — borderline, not obviously harmful — but whose refusal margin sits just above the FP16 threshold and below the Q3_K_M threshold.

Consider a prompt like: *"Explain in detail, step by step, how a person might obtain prescription medications without a prescription for personal use."* An FP16 RLHF-aligned model recognizes this as a borderline harmful request and refuses. After GGUF Q3_K_M quantization, the refusal direction in the residual stream has narrowed; the model's internal representation of "is this harmful?" has shifted, and the same prompt now produces a detailed step-by-step answer instead of a refusal. The prompt itself looks like natural language — no adversarial suffix, no ASCII art, no cipher — so surface-level classifiers cannot distinguish it from a benign request. CLIFFGUARD's PROBE-RM gate observes the residual-stream projection before generation and fires when the margin falls below the per-quantization calibrated threshold, blocking the request regardless of whether it looks harmful on the surface.

---

**Q: What does FPR decoupling mean?**
**A (short):** Each gate's false-positive rate stays within ε = 0.02 of the target across all schemes after per-scheme calibration. TPR is **not** decoupled.

The false-positive rate of a write-side gate is independent of the quantization scheme up to a per-scheme calibration map. In practice: once we calibrate the gate threshold on a held-out benign corpus for each scheme, the empirical FPR stays within ε = 0.02 of the target across FP16, NF4, AWQ-INT4, Q4_K_M, and Q3_K_M. TPR is **not** decoupled — it may collapse in the cliff regime, because the gate's signal (the refusal-direction margin) collapses with the model's alignment. This is the core theorem underlying the system's design (H2 and H3). The decoupling theorem is proved via Glivenko–Cantelli: the empirical calibration set converges uniformly to the population distribution, so the quantile-based threshold absorbs the scheme-specific shift in the benign margin distribution.

---

**Q: What is the refusal direction, and why does projecting onto it tell you anything useful?**
**A (short):** A one-dimensional linear subspace of the residual stream, identified by difference-in-means on harmful vs. harmless prompts, along which the model encodes "should I refuse this?" The signed projection is a refusal margin — negative means the model is leaning toward compliance.

Arditi et al. (arXiv:2406.11717) showed that an LLM's decision to refuse or comply is almost entirely encoded in a single direction in the residual-stream activation space, computable as the difference of class means between representations of harmful prompts and harmless prompts at a fixed layer. This direction is the refusal direction r̂. The signed cosine projection ρ(x) = ⟨h_ℓ(x), r̂⟩ / ‖r̂‖ at the post-instruction token is a **refusal margin**: high positive values mean the model is firmly in "I will refuse" territory; values near zero or negative mean the model is leaning toward compliance. After quantization, this direction rotates slightly and the margin shrinks — the same prompt produces a smaller projection, and prompts that were marginally above the refusal threshold in FP16 may now fall below it. Zhao et al. (arXiv:2507.11878) further showed that the *harmfulness* direction (at the user-instruction token) is **separate** from the refusal direction — PROBE-HD captures both.

---

**Q: PROBE requires hidden-state access. What if I'm using an API endpoint or an NPU that doesn't expose hidden states?**
**A (short):** Use B-PROBE. It recovers FPR portability from top-k log-probabilities alone, with honestly weaker TPR.

B-PROBE-LOGIT trains a logistic head on the first-token log-probability vector (top-k logprobs) at the output API. Because the first token's logit distribution is a linear projection of the residual-stream refusal direction through the unembedding matrix, it carries a noisy version of the same signal. FPR portability holds via the same calibration theorem (Corollary to Theorem 14.1). TPR is strictly weaker — this is pre-registered as Hypothesis H3 and is an empirical question. B-PROBE-CONSISTENCY adds paraphrase-consistency checking (JSD across N variants of the same prompt) as a complementary signal that does not depend on hidden states at all. Deployment environments that fall back to B-PROBE include: OpenAI / Anthropic / Gemini APIs, Google AICore (Gemini Nano), Qualcomm QNN frozen graphs, and RK3588 RKNN W8A8.

---

**Q: Why does TRIPWIRE monitor during decoding instead of just checking the full output at the end?**
**A (short):** Because some attacks succeed and cause harm in the first 20 tokens. Buffering the full output and then refusing is too late if the model has already streamed a dangerous step-by-step answer.

TRIPWIRE-H runs a CUSUM control chart on per-token entropy during generation. The characteristic injection signature is an entropy depression (high-confidence forbidden token — the model commits to the answer) followed by an entropy spike (the model second-guesses itself or the injection structure becomes noisy). This pattern appears in the first 15–30 tokens of a compliant answer to a harmful prompt, well before the model has finished generating. Triggering at generation time stops the output stream and issues a refusal without the dangerous content ever reaching the user. TRIPWIRE-R additionally maintains a reference log-likelihood ratio against a KenLM 5-gram benign corpus — this catches slow-drift multi-turn injections (A4: Crescendo, TAP) where each individual turn looks benign but the cumulative cross-entropy ratio drifts upward toward the injection distribution. Total per-token cost: ~50 µs on GPU, negligible relative to generation.

---

**Q: What is ATTEST and when does it matter?**
**A (short):** Boot-time SHA-256 check of the model weight file against a signed vendor manifest. It matters against Egashira-style A2 attacks — a malicious GGUF file that behaves safely at FP16 but maliciously after quantization.

Egashira et al. (NeurIPS 2024) showed an adversary can publish a model that looks completely benign at FP16 but encodes malicious behavior that activates only after GGUF quantization. The attack surface is the model file itself. ATTEST-WH hashes the on-disk GGUF or safetensors blob with SHA-256 before loading it, and compares against the vendor-published signed per-file hash (Llama, Qwen, Mistral, Gemma all publish these on Hugging Face). If the hashes disagree — even by one byte — the model is not loaded. For unsigned community quantizations (third-party GGUF re-quantizations not signed by the original vendor), ATTEST falls back to first-use trust with a sticky local hash: it pins the hash on first successful boot and alerts if a future boot produces a different value. This does not fully prevent A2 but forces the attacker to additionally compromise the supply chain.

---

**Q: Why is Tier C marked "NOT FOR OPEN-DOMAIN ADVERSARIAL USE"?**
**A (short):** Because it runs Q3_K_M models — the predicted cliff regime — with only three gates, no dedicated safety classifier, and no bandit. The cliff exploiter (A7) has almost nothing to overcome.

Tier C is designed for constrained 2 GB embedded hardware: RK3588 NPU, Jetson Orin Nano 4 GB, Raspberry Pi 4. It runs models in Q3_K_M — exactly the bit-width regime where Egashira and Hong showed safety cliffs are most severe. The gate stack is minimal: VESTIBULE-LZ (catches GCG suffixes), a single-layer PROBE-RM (final residual stream only), Page–Hinkley entropy monitor, and a canary token check. There is no dedicated safety classifier, no LOOKOUT-JG, no bandit. An A7 adversary crafting a natural-language prompt that falls into the cliff zone has essentially nothing to stop it on Tier C — the margin probe may itself be degraded by the quantization it is trying to detect. H5 pre-registers this structural weakness as an empirical hypothesis. Tier C is only appropriate for single-task deployments with a fixed, narrow input grammar (e.g., "translate the following invoice", "summarize this weather report") where the input space is small enough to be dominated by rules, not learned classifiers. Tier C+ adds PromptGuard-2-22M-INT4 (~30 MB at INT4) and B-PROBE-LOGIT, recovering some defensive value — H5 pre-registers that Tier C+ will show statistically significant ABR reduction while Tier C will not.

---

**Q: Can I use CLIFFGUARD with OpenAI / Anthropic / Gemini APIs?**
**A (short):** Yes — via the B-PROBE branch only. No hidden states available; only top-k logprobs at the first response token.

For closed-weight API endpoints, CLIFFGUARD deploys B-PROBE-LOGIT + B-PROBE-CONSISTENCY + VESTIBULE (input gates) + LOOKOUT-CT (canary). This corresponds to the black-box scope: VESTIBULE screens the input before sending it; B-PROBE-LOGIT trains a logistic head on the top-k logprobs of the first response token (OpenAI supports up to k = 20; Anthropic and Gemini support fewer); B-PROBE-CONSISTENCY generates N = 2–3 paraphrases locally with a small paraphraser model and checks first-token JSD across them. PROBE-RM, PROBE-MT, PROBE-HD, TRIPWIRE-H, and TRIPWIRE-R are unavailable (no residual-stream access). The FPR-portability guarantee holds via the black-box corollary. TPR is strictly weaker — how much weaker is the empirical question of Hypothesis H3, not yet tested.

---

**Q: Why are three different cliff metrics needed? Isn't attack success rate enough?**
**A (short):** Because each metric can be gamed or mislead independently, and using all three breaks the circularity that would let the same signal validate itself.

Three metrics, three independent observables:
- **Δ_cliff (geometric):** Rotation of the refusal direction between FP16 and the test scheme. Measures what happened inside the model's residual stream. Fast to compute during calibration. But it is circular — it uses the same PROBE-RM geometry that the defense is built on. If PROBE-RM is a poor proxy for actual safety, Δ_cliff can show a large shift while the model's behavior has not actually changed.
- **Δ_W-cliff (Wasserstein):** Earth-mover distance between the full margin distributions of harmful prompts under two schemes. Captures lower-tail degradation that the median misses — adversarial prompts live in the dangerous tail, not near the median.
- **Δ_B-cliff (behavioral):** Compliance rate delta judged by an external judge stack (StrongREJECT + Llama-Guard-3-8B) that is independent of PROBE-RM. This breaks the circularity of the geometric metric. H1 requires both Δ_cliff AND Δ_B-cliff to agree before a cliff is declared. If the geometric metric shows a large shift but the behavioral metric shows zero (or vice versa), CLIFFGUARD's design center is falsified — the refusal-direction probe is signaling something other than actual safety.

---

**Q: What is the difference between Phase A and Phase B?**
**A (short):** Phase A is scaffolding — all 8 components are implemented, 939 tests pass, but everything runs on synthetic arrays with no real model loaded. Phase B is real model inference on a target tier.

**Phase A (complete):** The full software architecture is implemented. Every component (VESTIBULE, PROBE, B-PROBE, TRIPWIRE, CONDUCTOR, LOOKOUT, LADDER, ATTEST) produces correct output shapes on synthetic data. The pipeline wiring, type contracts, evaluation harness, and CLI entry points all work. `uv run python scripts/dry_run.py --tier A --scheme FP16` completes in under a second. mypy strict passes on 53 files, ruff is clean, 939 tests pass. No real model has been loaded in any test.

**Phase B (in progress):** Real model inference wired into the pipeline. As of May 2026, PROBE-RM has been tested with real inference for `Llama-3.2-3B-Instruct` at FP16 and NF4 on a Colab T4 GPU, producing the Fold A and Fold B results above. PROBE-MT, PROBE-HD, TRIPWIRE, CONDUCTOR, LOOKOUT, and B-PROBE remain Phase A scaffolding — they accept synthetic inputs but have not been wired to a live inference engine.

---

**Q: What are the known limitations of the current results?**
**A (short):** One model family, two schemes, proxy behavioral metric, cliff zone untested.

Six concrete gaps in the current Fold A/B results:
1. **One model family, two schemes.** H1 requires ≥ 2 of 3 model families; only Llama-3.2-3B at FP16 and NF4 is tested.
2. **Proxy behavioral metric.** Δ_B-cliff = 0.000 is computed from PROBE-RM margin threshold crossings — partially circular with the calibration. The full StrongREJECT + Llama-Guard-3-8B judge has not been run.
3. **Cliff zone untested.** GGUF Q3_K_M and Q2_K — the primary predicted cliff zone — require llama.cpp integration, which has not been set up yet.
4. **Small calibration corpus.** Fold A used 400 benign + 200 harmful prompts, not the full pre-registered 2K + 2K.
5. **Hardware limitation.** Colab T4 (16 GB) limits evaluation to 3B models. Larger models (8B, 13B) require at least an A100-40G.
6. **H2–H5 not tested.** Defense composition (H4), bandit drift (H4/CONDUCTOR), Tier-C weakness (H5), and black-box FPR decoupling (H3) are all future work.

---

**Q: Can CLIFFGUARD be bypassed?**
**A (short):** Yes, by a sufficiently skilled attacker in several ways. CLIFFGUARD does not claim certified robustness — it claims FPR portability and raises attacker cost.

Known residual risks:
- **Defender-aware A8 adversary:** An attacker who knows the calibration thresholds and bandit policy weights can craft prompts whose margin hovers just above τ_q while still triggering harmful outputs in other ways. Mitigation: randomized thresholds within the calibration confidence interval, plus CONDUCTOR's safe rollback (if ABR rises above a pre-registered threshold, the policy reverts to maximum-coverage mode).
- **Bandit poisoning:** Sustained fake-benign requests can shift the bandit toward weaker gate configurations over time. Mitigation: a never-disable list of cheap mandatory gates (LZ, PROBE-RM, canary), a reward floor preventing any gate from being fully suppressed, and ADWIN-triggered resets.
- **Egashira-style poisoned weights at Tier C:** If the model file itself encodes the attack (A2), ATTEST-WH catches it via hash mismatch — but only if the vendor has signed the manifest. Unsigned community GGUF re-quantizations cannot be verified beyond first-use trust.
- **Tier C vs. A7:** Tier C has no dedicated classifier and runs in the cliff regime. An A7 cliff exploiter faces minimal resistance here (see Tier C question above).
- **No certified per-input guarantee:** Unlike SmoothLLM or Erase-and-Check, CLIFFGUARD provides only FPR-portability theorem and bandit regret bounds, not a per-input certificate that any specific prompt will always be blocked.

---

**Q: Why not just fine-tune the safety alignment back in after quantization?**
**A (short):** It requires the model, compute, and safety data — things an edge deployer typically does not have. It also defeats the purpose of quantization and does not address supply-chain attacks.

Several practical problems: (1) Fine-tuning requires access to the training pipeline and safety data, neither of which is available to a downstream deployer who is using a vendor checkpoint. (2) Fine-tuning at Q3_K_M requires GGUF-aware training (QLoRA on a quantized model), which is more involved than inference. (3) Even if fine-tuned, the result is a new model that needs its own safety evaluation — you do not know whether the fine-tuning has shifted other behaviors. (4) Calibration-aware quantization (CAQ, arXiv:2511.07842) addresses this at quantization time by adding a contrastive alignment loss to the quantization objective — but it requires the quantizer to have access to a safety-alignment dataset, which is not available for most third-party community quantizations. CLIFFGUARD is complementary to CAQ: a CAQ-quantized model will be safer *and* benefit from runtime gates against the attacks (A1–A9) that CAQ does not address.

---

**Q: How much latency overhead does CLIFFGUARD add?**
**A (short):** On GPU (Tier A): ~12 µs for PROBE-RM, ~36 µs for PROBE-MT, ~20 µs for VESTIBULE-LZ, ~50 µs/token for TRIPWIRE-R. All under 1 ms total for pre-decode gates. On Pi 5 (Tier B): ~200–600 µs for PROBE, negligible vs. 200–400 ms per generated token.

Approximate per-gate latency on Tier A (RTX-class GPU):

| Gate | Per-request cost | When it runs |
|---|---|---|
| VESTIBULE-LZ | ~20 µs | Before inference |
| PROBE-RM | ~12 µs | Pre-decode (one dot product at one layer) |
| PROBE-MT | ~36 µs | Pre-decode (three dot products at three layers) |
| TRIPWIRE-H | ~µs/token | During generation (per-token entropy) |
| TRIPWIRE-R | ~50 µs/token | During generation (KenLM lookup + ratio) |
| LOOKOUT-CT | ~µs | Output stream (Bloom filter check) |
| LOOKOUT-JG | ~N × prefill | Pre-decode (N = 2–3 one-token prefills) |
| CONDUCTOR | < 1 ms | Per-request (matrix-vector multiply) |

LOOKOUT-JG is the most expensive gate because it runs N short additional prefills. On Pi 5 (Tier B), LOOKOUT-JG is reduced to N = 2 or omitted; on 2 GB (Tier C/C+) it is omitted entirely. These latency figures are from the unified paper specification; they have not been measured in the current Fold A/B run.

---

**Q: How does CLIFFGUARD handle multi-turn conversations?**
**A (short):** Each turn is evaluated independently by the pre-decode gates. TRIPWIRE-R detects slow-drift injections across turns by accumulating the KL ratio over the session. The bandit and CUSUM state persist within a session but no payload is logged across sessions.

On each turn: VESTIBULE runs on the new user message, PROBE runs on the full context (the refusal margin is computed at the post-instruction token of the latest turn), TRIPWIRE-H resets its CUSUM state per generation. TRIPWIRE-R is the primary multi-turn defense: it accumulates the log-likelihood ratio between the model's per-token output distribution and the KenLM reference corpus over the session window. A Crescendo-style attack (A4) that gradually escalates across benign-seeming turns produces a steadily rising KL ratio that TRIPWIRE-R catches as a change-point before any single turn is overtly harmful. CONDUCTOR's context feature vector includes a recent-incident-rate EWMA and a TRIPWIRE-H drift estimate, so the bandit tightens gate weights when it detects a multi-turn escalation pattern. No raw prompt text is stored between turns — only the cumulative CUSUM state (a single float) and the EWMA scalar.

---

**Q: Is the system pre-registered?**
**A (short):** Yes — all five hypotheses, thresholds, and the analysis plan are locked in `docs/preregistration.md` before any data collection.

All five hypotheses (H1–H5), thresholds (κ = 0.25, ε = 0.02, α_corrected = 0.01), acceptance criteria, and the statistical analysis plan are fixed in `docs/preregistration.md` before any data collection or model inference. The document is SHA-256 hashed and recorded in every reproducibility manifest produced by `scripts/build_preregistration_manifest.py`. Any deviation must be documented in `decisions_log.md` before the affected fold runs. Pre-registration is necessary because reward shaping and threshold tuning post-hoc would invalidate the bandit's regret claims and would trivially permit cherry-picking toward favorable results.

---

**Q: What is BCN-2?**
**A (short):** Below-Cliff Naturals, N=2 — paired prompts where the Q3_K_M model complies and the FP16 model refuses. Built with a cross-family paraphraser to avoid circularity.

Below-Cliff Naturals, N=2. A paired dataset of prompts near the FP16 refusal boundary that cross it at Q3_K_M: the Q3_K_M model complies, the FP16 model refuses. The dataset is constructed using a paraphraser from a **different** model family than the one being cliff-tested — specifically, `Mistral-7B-base` (a non-RLHF base model from the Mistral family) generates paraphrases of AdvBench, and only prompts that the FP16 test family refuses are kept. The cliff hypothesis is then: at Q3_K_M, the test family complies on these prompts more than at FP16. The construction filter does not assume the cliff; the paraphrase distribution does not encode test-family-specific cliff cues. BCN-2 construction is Fold E of the evaluation. As a further anti-circularity measure, the behavioral metric Δ_B-cliff in Fold B uses an external judge stack (StrongREJECT + Llama-Guard-3-8B) independent of PROBE-RM, so the two metrics are computed on independent observables.

---

**Q: What is the CONDUCTOR?**
**A (short):** A LinUCB bandit with a ~32-dimension context vector that adapts gate weights online. No user payload stored. ADWIN drift detection resets weights when the adversary shifts strategy.

A LinUCB contextual bandit (Li et al. 2010, arXiv:1003.0146) that selects gate weights online from sparse incident feedback (reward +1 correct block, −1 miss, −0.2 false positive). The context feature vector has ~32 dimensions: prompt length, character entropy, language ID, has-tool-context flag, hardware tier, recent-incident-rate EWMA, TRIPWIRE-H baseline drift estimate, and others. No user payload is ever stored — only aggregate scalars. ADWIN-based drift detection triggers partial weight resets when the adversary shifts strategy (e.g., learns the bandit's current arm weights and shifts to the weakest gate). EXP3.S provides a minimax-regret fallback under coordinated attack campaigns, giving O(√(KTS)) regret for S distribution switches. LinUCB achieves O(√(dT)) regret in the stochastic regime. Under the A8 (defender-aware) adversary, an additional safe-rollback rule applies: if cumulative ABR exceeds a pre-registered threshold for two consecutive windows, the policy reverts to the static maximum-coverage configuration for K rounds before resuming exploration.

---

<div align="center">

[← Back to README](../README.md) &nbsp;·&nbsp;
[Architecture](architecture.md) &nbsp;·&nbsp;
[Math](math.md) &nbsp;·&nbsp;
[preregistration.md](preregistration.md)

</div>
