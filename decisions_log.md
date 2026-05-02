# `decisions_log.md` — Running record of decisions made during CLIFFGUARD development

This file accumulates non-obvious decisions that future tasks (and future-you) need to remember. Desktop appends entries here when something material is decided. Each entry is dated and references the task that triggered it.

Format:

```
## YYYY-MM-DD — Task N — One-line summary

Context: what was being decided.
Decision: what was chosen.
Reasoning: why this and not the alternative.
Affects: which later tasks depend on this.
```

---

## 2026-MM-DD — Task 0 — Project initialization

Context: Project setup phase before Task 1.
Decision: Repo at `C:\Users\AB\Desktop\Projects\CLIFFGUARD`. Python 3.11, `uv` for packaging, `pytest` / `ruff` / `mypy --strict` for QA. No Docker in Phase A. GPU dependencies behind `[gpu]` extra in Phase B.
Reasoning: User specified the path; `uv` chosen over `pip` for reproducibility and lockfile support.
Affects: All tasks.

---

(Append new entries below as the project progresses.)
