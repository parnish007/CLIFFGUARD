# Engineering Reference

Reference for Phase B implementers wiring real inference engines to the CLIFFGUARD scaffolding. All modules listed below are fully implemented in Phase A with synthetic stubs; Phase B replaces the stubs with real engine adapters and real calibration data.

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

1. **Implement `get_hidden_states`** in the engine adapter matching your target hardware (see table above). Verify by running `scripts/dry_run.py` in Phase B mode — it should produce real margin values, not synthetic floats.

2. **Calibrate the refusal direction:** run `eval/refusal_direction.py` on the Fold A corpus. This produces `r_hat_{model}_{scheme}.npy` in `artifacts/directions/`. One direction vector per (model family, quantization scheme) pair.

3. **Calibrate the harmfulness direction:** run `eval/harmfulness_direction.py` on Fold A. Produces `h_hat_{model}_{scheme}.npy`. Uses Zhao et al.'s difference-in-means recipe on the user-instruction token position.

4. **Train KenLM reference model:** run `eval/kenlm_trainer.train_and_save` on the Fold A benign corpus. Order 5 for Tier A/B (`kenlm.order_tier_ab`), order 3 for Tier C/C+ (`kenlm.order_tier_c`). Produces `artifacts/kenlm/benign_{scheme}.arpa`.

5. **Build calibration tables:** run `eval/threshold_calibrator.build_calibration_table` for each primitive on the Fold A benign corpus. This sets per-scheme thresholds τ_q targeting `fpr_target = 0.05`. Produces `artifacts/calibration/{primitive}_{scheme}.json`.

6. **Wire FiveFoldOrchestrator:** implement `execute_fold_b` and `execute_fold_c` in `eval/five_fold_orchestrator.py` (currently raise `NotImplementedError`). These methods require the calibration tables and direction vectors produced in steps 2–5.

7. **Run full evaluation:**
   ```bash
   uv run python scripts/run_full_evaluation.py --config configs/my_run.yaml
   ```

8. **Capture reproducibility manifest:**
   ```bash
   uv run python scripts/build_preregistration_manifest.py \
     --tier A --schemes FP16 NF4 GGUF_Q4_K_M GGUF_Q3_K_M
   ```
   The manifest SHA-256-hashes all data files, artifact files, and `docs/preregistration.md`. A manifest is valid if the git hash is present, the preregistration hash matches, and all data files exist on disk.

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

**Calibration implication:** for Fires-HIGH gates, τ_q is the empirical (1 − α) quantile of the benign score distribution. For Fires-LOW gates, τ_q is the empirical α quantile. Both targeting the same FPR = α = 0.05. The `threshold_calibrator` in `eval/threshold_calibrator.py` handles this automatically via the `fires_high` parameter.
