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
