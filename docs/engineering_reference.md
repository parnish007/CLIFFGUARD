<div align="center">

[← README](../README.md) &nbsp;|&nbsp;
[What is it](what_is_it.md) &nbsp;|&nbsp;
[Architecture](architecture.md) &nbsp;|&nbsp;
[Math](math.md) &nbsp;|&nbsp;
[Setup](setup.md) &nbsp;|&nbsp;
[Engineering Ref](engineering_reference.md)

</div>

# Engineering Reference

Reference for Phase B implementers wiring real inference engines to the CLIFFGUARD scaffolding. All modules listed below are fully implemented in Phase A with synthetic stubs; Phase B replaces the stubs with real engine adapters and real calibration data.

> **Who this document is for:** Phase B implementers wiring real
> inference engines to the scaffolding. If you are running the dry
> run or trying the evaluation for the first time, start with
> [docs/setup.md](setup.md) instead.

## Module Map

| Module | Blueprint § | Public API | Fires | Phase B Action Needed |
|---|---|---|---|---|
| `vestibule/lz.py` | §5.6 | `compression_ratio`, `evaluate` | HIGH | None — stdlib `zlib` only |
| `vestibule/ps.py` | §5.7 | `count_signals`, `signal_score`, `evaluate` | HIGH | None — regex + heuristics |
| `probe/rm.py` | §5.1 | `compute_margin`, `evaluate` | LOW | Wire `HiddenStateAdapter.get_hidden_states` |
| `probe/mt.py` | §5.2 | `compute_trajectory`, `evaluate` | LOW | Wire `HiddenStateAdapter.get_hidden_states` |
| `probe/hd.py` | §5.3 | `compute_harmfulness_margin`, `evaluate` | HIGH | Wire `HiddenStateAdapter.get_hidden_states` |
| `tripwire/h.py` | §5.4 | `token_entropy`, `cusum_statistic`, `evaluate` | HIGH | Wire real per-token logprobs from engine |
| `tripwire/r.py` | §5.5 | `log_likelihood_ratio`, `evaluate` | LOW | Wire KenLM via `eval/kenlm_trainer.py` |
| `lookout/ct.py` | §5.8 | `BloomFilter`, `check_output`, `evaluate` | HIGH | None — stdlib only |
| `lookout/jg.py` | §5.9 | `compliance_rate`, `evaluate` | HIGH | Wire real Paraphraser + Judge (Llama Guard 3) |
| `bprobe/logit.py` | §5.10 | `logistic_score`, `evaluate` | HIGH | Fit logistic weights on Fold A |
| `bprobe/consistency.py` | §5.11 | `js_divergence`, `evaluate` | LOW | Wire real paraphraser for N-sample logprobs |
| `attest/wh.py` | §5.12 | `hash_file`, `attest` | — | Wire real manifest path + signed vendor hash |
| `conductor/bandit.py` | §6 | `Conductor.select_weights`, `update`, `aggregate_verdict` | — | None — Phase A is complete |
| `ladder/router.py` | §10 | `route`, `gate_count` | — | None — Phase A is complete |
| `ladder/tier.py` | §10 | `gates_for_tier`, `is_gate_active` | — | None — Phase A is complete |

## Data Type Contracts

All public API functions operate on types defined in `cliffguard/types.py`. Phase B implementers must not create ad-hoc return types.

| Type | Module | Key Fields | Notes |
|---|---|---|---|
| `GateVerdict` | `types.py` | `gate_name: GateName`, `fired: bool`, `score: float`, `margin: Margin \| None` | Returned by every `evaluate()` function |
| `Margin` | `types.py` | `value: float`, `threshold: float`, `scheme: QuantScheme` | Carries the calibrated threshold alongside the score |
| `CalibrationTable` | `eval/calibration.py` | `{QuantScheme: float}` mapping scheme → τ_q | Built by `build_calibration_table()` per primitive |
| `FoldResult` | `eval/fold_runner.py` | `fold_name`, `tier`, `scheme`, `n_prompts`, `n_blocked`, `n_passed`, `fpr`, `asr`, `notes` | Input to `results_writer.write_fold_result()` |
| `Tier` | `types.py` | Enum: `A`, `B`, `C`, `C_PLUS` | Used by LADDER router; `Tier.value` is the string form |
| `QuantScheme` | `types.py` | Enum: `FP16`, `NF4`, `AWQ_INT4`, `GGUF_Q4_K_M`, `GGUF_Q3_K_M`, … | `QuantScheme.value` is the string form |
| `HiddenStateAdapter` | `engines/base.py` | Abstract base; `get_hidden_states(prompt, layer)`, `get_top_k_logprobs(prompt, k)` | Phase B must subclass this |
| `GateName` | `types.py` | Literal string union of all 12 gate names | Used as dict keys in CONDUCTOR context builder |

**Serialisation rule:** enum fields must be serialised via `.value` before writing to JSON (see `results_writer._serialize_fold_result`). `json.load()` returns `Any`; use `# type: ignore[no-any-return]` and narrow with `isinstance()` before accessing fields.

## Error Handling Contracts

| Condition | Module | Behaviour | Action for Phase B |
|---|---|---|---|
| Hidden states unavailable (NPU frozen graph, API endpoint) | `probe/*.py` | `HiddenStateAdapter.get_hidden_states` raises `NotImplementedError` | LADDER routes to black-box path; PROBE gates return `GateVerdict(fired=False, score=0.0, margin=None)` |
| KenLM binary not found | `tripwire/r.py` | `FileNotFoundError` on ARPA path at load time | `evaluate()` returns `GateVerdict(fired=False, …)` with a log warning; TRIPWIRE-H continues alone |
| ATTEST hash mismatch | `attest/wh.py` | `attest()` returns `AttestResult.BLOCK` | CONDUCTOR sets context vector index 12 to 0.0; final verdict is BLOCK regardless of other gates |
| ATTEST manifest not found | `attest/wh.py` | `attest()` returns `AttestResult.DEGRADED` | CONDUCTOR sets context vector index 12 to 0.5; system continues in degraded mode |
| Calibration table missing for scheme | `eval/calibration.py` | `KeyError` on `CalibrationTable[scheme]` | Build calibration tables for all deployed schemes before production use |
| Fold A corpus below `MIN_CALIBRATION_SIZE` | `eval/calibration.py` | Raises `ValueError` | Collect ≥ 2000 benign prompts; minimum required for KS estimation error to stay within ε = 0.02 |

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `UV_CACHE_DIR` | system default | Redirect uv package cache (useful when C: is full on Windows) |
| `UV_PROJECT_ENVIRONMENT` | `.venv/` | Redirect virtual environment directory |
| `HF_HOME` | `~/.cache/huggingface` | Redirect HuggingFace model cache |
| `CLIFFGUARD_ARTIFACTS_DIR` | `artifacts/` | Override the artifacts directory at runtime (takes precedence over `configs/*.yaml`) |
| `CLIFFGUARD_DATA_DIR` | `data/` | Override the data directory at runtime |
| `CLIFFGUARD_LOG_LEVEL` | `WARNING` | Set logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

None of these are required for Phase A dry runs. `UV_CACHE_DIR` and `UV_PROJECT_ENVIRONMENT` are most commonly needed on Windows — see the Troubleshooting section in [docs/setup.md](setup.md).

## Engine Adapters

Phase B inference hooks live in `cliffguard/engines/`. Each adapter exposes two methods: `get_hidden_states(prompt, layers)` returning a dict of layer → hidden-state tensor, and `get_top_k_logprobs(prompt, k)` returning a top-k log-probability array.

| Adapter | Tier | Mode | File | Phase B Status |
|---|---|---|---|---|
| `TransformersBnbAdapter` | A | white-box | `engines/transformers_bnb.py` | Stub — implement `get_hidden_states` via `output_hidden_states=True` hook |
| `AutoAWQAdapter` | A | white-box | `engines/autoawq.py` | Stub — identical hook API to transformers |
| `VLLMAdapter` | A/B | black-box | `engines/vllm.py` | Stub — implement `get_top_k_logprobs` via `SamplingParams(logprobs=k)` |
| `LlamaCppAdapter` | B/C | both | `engines/llamacpp.py` | Stub — implement both via `llama_get_embeddings_ith` (final layer) and `llama_get_logits_ith` |

**Key reference:** blueprint §18.1–§18.4 gives concrete API paths for each engine, including known caveats (e.g. `device_map="auto"` multi-GPU hidden-state zero-out bug, llama.cpp eval-callback enabling).

## Phase B Wiring Checklist

Follow these steps in order. Each step depends on the previous.

- [ ] **Step 1** — Implement `get_hidden_states` in the engine
  adapter matching your hardware tier.
- [ ] **Step 2** — Calibrate refusal direction:
  `eval/refusal_direction.calibrate_refusal_direction()`
- [ ] **Step 3** — Calibrate harmfulness direction:
  `eval/harmfulness_direction.calibrate_harmfulness_direction()`
- [ ] **Step 4** — Train KenLM reference model:
  `eval/kenlm_trainer.train_and_save()` (order=5 Tier A/B,
  order=3 Tier C/C+)
- [ ] **Step 5** — Build calibration tables:
  `eval/threshold_calibrator.build_calibration_table()` per primitive
- [ ] **Step 6** — Implement `execute_fold_b` and `execute_fold_c`
  in `eval/five_fold_orchestrator.py`
- [ ] **Step 7** — Run full evaluation:
  `scripts/run_full_evaluation.py --config configs/my_run.yaml`
- [ ] **Step 8** — Build reproducibility manifest:
  `scripts/build_preregistration_manifest.py`

## Key Constants

| Constant | Value | Location | Meaning |
|---|---|---|---|
| `KAPPA` | `0.25` | `eval/cliff_metrics.py` | Pre-registered cliff threshold — the minimum Δ_cliff jump that constitutes a cliff |
| `ALPHA_CORRECTED` | `0.01` | `eval/stats.py` | Bonferroni-corrected significance level (0.05 / 5 hypotheses) |
| `MIN_CALIBRATION_SIZE` | `2000` | `eval/calibration.py` | Minimum Fold A corpus size — below this the KS estimation error is too wide |
| `CONTEXT_DIM` | `14` | `conductor/context.py` | CONDUCTOR feature vector dimension |
| `MIN_WEIGHT` | `0.1` | `conductor/bandit.py` | Floor weight for never-disable arms (TRIPWIRE-R, ATTEST-WH) |
| `ADWIN_DELTA` | `0.002` | `eval/drift_sim.py` | Page-Hinkley ADWIN sensitivity parameter (λ_thresh = −log(δ) ≈ 6.21) |

## Firing Direction Reference

Gate firing direction matters for the CONDUCTOR feature vector and for threshold calibration. Fires-HIGH gates fire when the score **exceeds** τ_q; Fires-LOW gates fire when the score **falls below** τ_q.

| Fires-HIGH (score > τ_q) | Fires-LOW (score < τ_q) |
|---|---|
| VESTIBULE-LZ | PROBE-RM |
| VESTIBULE-PS | PROBE-MT |
| PROBE-HD | TRIPWIRE-R |
| TRIPWIRE-H | B-PROBE-CONSISTENCY |
| LOOKOUT-CT | |
| LOOKOUT-JG | |
| B-PROBE-LOGIT | |

> **Calibration implication:** for Fires-HIGH gates, τ_q is the
> empirical (1 − α) quantile of the benign score distribution.
> For Fires-LOW gates, τ_q is the empirical α quantile.
> Both target FPR = α = 0.05. See `eval/threshold_calibrator.py`.

---

<div align="center">

[← Back to README](../README.md) &nbsp;·&nbsp;
[Open an issue](https://github.com/YOUR_USERNAME/CLIFFGUARD/issues) &nbsp;·&nbsp;
[preregistration.md](preregistration.md)

</div>
