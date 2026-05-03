# `decisions_log.md` — Running record of decisions made during CLIFFGUARD development

This file accumulates non-obvious decisions that future tasks (and
future-you) need to remember. Desktop appends entries here when
something material is decided.

---

## 2026-MM-DD — Task 0 — Project initialization

Context: Project setup phase before Task 1.
Decision: Repo at C:\Users\AB\Desktop\Projects\CLIFFGUARD. Python 3.11,
uv for packaging, pytest / ruff / mypy --strict for QA. No Docker in
Phase A. GPU dependencies behind [gpu] extra in Phase B.
Reasoning: User specified the path; uv chosen over pip for
reproducibility and lockfile support.
Affects: All tasks.

---

## 2026-05-03 — Phase A Gate — Firing-direction groups and rho_ddot

Context: Phase A deep validation pass.
Decision 1: Two firing-direction groups documented across primitives.
  Fires-HIGH: VESTIBULE-LZ, VESTIBULE-PS, PROBE-HD, TRIPWIRE-H,
              LOOKOUT-CT, LOOKOUT-JG, B-PROBE-LOGIT
  Fires-LOW:  PROBE-RM, PROBE-MT, TRIPWIRE-R, B-PROBE-CONSISTENCY
CONDUCTOR aggregate_verdict handles both via the fired boolean.
Affects: Task 17 — already correct.

Decision 2: PROBE-MT computes rho_ddot but did not surface it to
CONDUCTOR feature vector in Phase A. RESOLVED: surfaced in
cliffguard/conductor/context.py at FEATURE_INDEX["PROBE-MT-rho_ddot"]
index 4. CLOSED.

---

## 2026-05-03 — Task 20 — sys_platform markers on Linux-only GPU deps

Context: autoawq and vllm call torch at build time; uv lockfile fails
on Windows without the marker.
Decision: autoawq and vllm use "; sys_platform == 'linux'" in gpu extra.
Reasoning: Technically correct; preserves downstream pip install semantics.
Affects: All future gpu-extra packages — check Windows buildability first.

---

## 2026-05-03 — Task 20 — pydantic>=2.0 floor added

Context: uv re-resolved lockfile and silently downgraded pydantic v2
to v1, breaking mypy.
Decision: pydantic pinned to >=2.0 in core dependencies.
Affects: All tasks — pydantic v2 now guaranteed.

---

## 2026-05-03 — Task 27 — geometric_cliff range is [0, sqrt(2)] not [0, 1]

Context: Antipodal unit vectors give ||r - r_FP16|| = 2.0; divided by
sqrt(2) = 1.414, not 1.0.
Decision: No clamping. Range [0, sqrt(2)] documented in code and tests.
Affects: Paper Section 11.3 must say "normalised to [0, sqrt(2)]" before
submission — flag at revision time.

---

## 2026-05-03 — PHASE B DIVERGENCE — Orchestrator generated own prompts

Context: The orchestrator (Claude Desktop) was issuing its own prompt
text for Phase B tasks instead of reading verbatim PROMPT FOR CLAUDE
CODE blocks from development.md. This caused 12 of 17 Phase B tasks
to diverge from the plan.

WHAT WAS BUILT vs WHAT development.md PLANNED:
  Tasks 19-21: Match development.md (engine adapter stubs). DONE.
  Task 22 built: eval/calibration.py (JSONL loader + synthetic corpus).
    dev.md Task 22 requires: eval/folds.py (five-fold structure,
    FoldEntry dataclass, fold isolation check, download_fold_a.py).
    STATUS: SUPPLEMENTARY — must still build dev.md Task 22.
  Task 23 built: eval/refusal_direction.py (difference-in-means).
    dev.md Task 23 requires same but with HiddenStateAdapter integration
    and .npz save/load. STATUS: PARTIAL — needs completion.
  Task 24 built: eval/threshold_calibrator.py (FPR calibrator).
    dev.md Task 24 requires: eval/harmfulness_direction.py (Zhao recipe,
    paired harmless construction, orthogonality validator).
    STATUS: SUPPLEMENTARY — must still build dev.md Task 24.
  Task 25 built: eval/kenlm_stub.py + eval/tripwire_r_calibration.py.
    dev.md Task 25 requires: eval/kenlm_trainer.py (real KenLM trainer
    with lmplz subprocess, Tier C+ memory budget, ArpaSize estimator).
    STATUS: PARTIAL — must still build dev.md Task 25.
  Task 26 built: eval/attack_corpus.py (AttackPrompt dataclass).
    dev.md Task 26 requires: eval/judges.py (StrongREJECT + LG3
    judge stack with agreement-required protocol).
    STATUS: SUPPLEMENTARY — must still build dev.md Task 26.
  Task 27 built: eval/cliff_metrics.py (geometric + behavioral only).
    dev.md Task 27 requires all three: geometric + Wasserstein-2 +
    behavioral. Wasserstein-2 (delta_w_cliff) is missing.
    STATUS: PARTIAL — must add delta_w_cliff.
  Tasks 28-33 built: runner.py, conductor/context.py,
    eval/orchestrator.py (request-cycle), run_full_evaluation.py,
    generate_cliff_corpus.py, docs/preregistration.md.
    dev.md Tasks 28-33 require: bcn2.py, FiveFoldOrchestrator,
    stats.py, drift_sim.py, figures.py, repro.py + manifest builder.
    STATUS: SUPPLEMENTARY — all dev.md tasks still unbuilt.

WHAT IS STILL MISSING from dev.md:
  C22: eval/folds.py + scripts/download_fold_a.py
  C23: Completion of eval/refusal_direction.py (HiddenStateAdapter
       integration + .npz save/load)
  C24: eval/harmfulness_direction.py (Zhao et al. recipe)
  C25: eval/kenlm_trainer.py (real KenLM trainer)
  C26: eval/judges.py (StrongREJECT + LG3 stack)
  C27: delta_w_cliff added to eval/cliff_metrics.py
  C28: eval/bcn2.py (BCN-2 cross-family constructor)
  C29: eval/five_fold_orchestrator.py (FiveFoldOrchestrator)
  C30: eval/stats.py (power calc, Mann-Whitney)
  C31: eval/drift_sim.py (bandit drift simulator)
  C32: eval/figures.py (matplotlib figures)
  C33: eval/repro.py + scripts/build_preregistration_manifest.py
  C34: scripts/dry_run.py + tests/test_dry_run_e2e.py
  C35: README.md replacement + configs/example.yaml

Resolution: From this point all prompts are read verbatim from
development.md. Missing tasks are labeled C22-C35 (Continuation)
in commit messages to distinguish from already-committed Tasks 1-33.
Claude Code is informed of this context in every prompt header.
Affects: All remaining tasks.
