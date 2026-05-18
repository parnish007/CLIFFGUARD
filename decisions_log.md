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

CONTINUATION STATUS: C22–C35 ALL DONE.

---

## 2026-05-03 — C25 — KenLM order: 3-gram vs 5-gram

Context: Prompt specified order=3; blueprint §5.5 says 5-gram for full
deployment.
Decision: order=3 retained as default for Tier C/C+ budget. Callers
pass order=5 for Tier A/B. Comment added in code.
Paper action: §5.5 should distinguish Tier C+ (order=3) from Tier A/B
(order=5).

---

## 2026-05-03 — C25 — §12.5 section reference misaligned in kenlm_trainer.py

Context: Prompt cited §12.5; blueprint §12.5 is statistical power.
Decision: Docstring verbatim; paper §12 numbering audit needed.

---

## 2026-05-03 — C26 — Blueprint section mismatches in judges.py

1. §5.9 cited — blueprint §5.9 is LOOKOUT-JG runtime gate, not judge
   stack. Real sections are §11.3 and §12.6. Fix docstring before submission.
2. arXiv:2312.06674 (LG1) cited for LG3 — blueprint cites
   arXiv:2411.17713-lineage. verified_citations.md needs LG3 ID update.
3. attack_success_rate returns 0.0 when all disputed — safe default.

---

## 2026-05-03 — C28 — BCN-2 storage format conflict

Context: Blueprint §12.2 says LSH storage; prompt specifies plain JSONL.
Decision: JSONL followed — LSH is a release concern, not Phase A.
Also: fp16_refused hardcoded True on load (filter guarantees refused-only).
Affects: Phase B — add LSH layer before public BCN-2 corpus release.

---

## 2026-05-03 — C28 — Section reference mismatches in bcn2.py

Prompt cited §11.2 (= Wasserstein) and §12.3 (= Models) for BCN-2.
Docstrings kept verbatim; paper section numbering audit needed.

---

## 2026-05-03 — C30 — preregistration.md alpha conflict with stats.py

Context: preregistration.md H4/H5 cite p < 0.05 (uncorrected); §4
mandates Bonferroni → alpha = 0.01. stats.py uses 0.01 (correct).
Decision: preregistration.md H4/H5 acceptance criteria must be updated
to p < 0.01 before submission.

---

## 2026-05-03 — C31 — ADWIN vs Page-Hinkley algorithm conflict

Context: Blueprint §6.4 specifies ADWIN (Bifet & Gavaldà 2007).
Prompt specified "Page-Hinkley variant". These are distinct algorithms.
Followed prompt (Page-Hinkley). Combined warmup+drift stream used for
detection — drift-only stream with ADWIN_DELTA=0.002 requires ~3100
steps; combined stream fires within T_drift=50.
Affects: Phase B — replace with true ADWIN if §6.4 is authoritative.
Paper action: §6.4 algorithm name vs implementation must be reconciled.

---

## 2026-05-03 — C35 — Tier C gate list: blueprint DOT diagram vs TIER_GATES

Context: Blueprint §10 DOT diagram shows Tier C active gates as
VESTIBULE-LZ + PROBE-RM(1L) + PH + CT. cliffguard/ladder/tier.py
TIER_GATES defines Tier C as VESTIBULE-LZ, VESTIBULE-PS, ATTEST-WH only.
Decision: README reflects TIER_GATES (the code), not the blueprint diagram.
The code is the Phase A implementation authority; the blueprint diagram
may reflect a planned Phase B configuration.
Affects: Paper §10 — reconcile DOT diagram with actual Phase A gate set
before submission. Flag at paper revision.
