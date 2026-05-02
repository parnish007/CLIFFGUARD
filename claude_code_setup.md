# `claude_code_setup.md` — One-time bootstrap

You read this once. It exists so future-you can verify the environment is sane without re-deriving it from scratch.

---

## 1. What this document covers

A single bootstrap path that gets you from "Claude Code is installed somewhere on this machine" to "I can run a Claude Code prompt from `development.md` and it will write files into the right place." Nothing more. No experiments, no calibration, no real models — that's `development.md`'s job.

---

## 2. Prerequisites (verify before bootstrapping)

Run these in PowerShell from anywhere. Each must succeed.

```powershell
claude --version              # Claude Code CLI is installed
node --version                # Node 18+ — Claude Code dependency
git --version                 # Git is installed
python --version              # Python 3.11.x (3.12 is fine; 3.10 will fail later)
```

If `claude` is not found, install Claude Code first via the official installer. The other three are standard developer tooling. Stop and fix any that fail before continuing.

---

## 3. Create the repo

```powershell
cd C:\Users\AB\Desktop\Projects
mkdir CLIFFGUARD
cd CLIFFGUARD
git init
```

Confirm with `git status` — should report "On branch main" or "On branch master" with no commits.

---

## 4. Install `uv` (Python package manager)

We use `uv` instead of `pip` because it has a real lockfile, is ~10× faster, and is becoming the standard for reproducible Python research code. `pip` is fine but you'd be fighting it later.

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell, then verify:

```powershell
uv --version
```

---

## 5. Configure Claude Code for this repo

Claude Code needs filesystem access scoped to `C:\Users\AB\Desktop\Projects\CLIFFGUARD`. There are two ways this can be configured: per-invocation (`--add-dir`) or persistent (allowed directories list). We use persistent.

From inside the repo:

```powershell
cd C:\Users\AB\Desktop\Projects\CLIFFGUARD
claude config add allowedDirectories "C:\Users\AB\Desktop\Projects\CLIFFGUARD"
```

Verify:

```powershell
claude config list
```

You should see the path in the allowed directories list. If Claude Code uses a different config schema on your installed version, run `claude config --help` and check what the equivalent flag is. The principle is: **Claude Code is allowed to read and write inside the repo and nowhere else.**

---

## 6. Smoke test

This confirms the loop you'll use for the rest of the project.

From inside the repo, start Claude Code:

```powershell
claude
```

Paste exactly this prompt:

```
Create a file at the repo root called HELLO.md containing exactly the
single line:

CLIFFGUARD bootstrap successful.

Then list all files in the repo root and confirm HELLO.md exists.
```

Claude Code should write the file and confirm it. Exit Claude Code (Ctrl+C or `/exit`).

Verify from PowerShell:

```powershell
Get-Content HELLO.md
```

Output should be exactly: `CLIFFGUARD bootstrap successful.`

If this works, the loop is wired correctly. Delete `HELLO.md` (`Remove-Item HELLO.md`) and you're done with bootstrap.

---

## 7. The loop you'll use for every `development.md` task

Every task follows this pattern. Internalize it once.

1. **Desktop:** open the project, ask for the next task. Desktop reads `development.md`, gives you the task's PROMPT FOR CLAUDE CODE block.
2. **You:** copy the prompt block.
3. **PowerShell:** `cd C:\Users\AB\Desktop\Projects\CLIFFGUARD && claude`
4. **Claude Code:** paste the prompt. Let it work. It will write files, possibly run scripts, possibly ask clarifying questions.
5. **Claude Code:** when done, exit with `/exit` or Ctrl+C.
6. **You:** report back to Desktop — paste the list of files Claude Code created or modified, and any console output that mattered.
7. **Desktop:** runs the per-task acceptance check from `development.md`. If pass, marks the task done in the state tracker. If fail, hands you a follow-up prompt for Claude Code.
8. **Repeat.**

At phase boundaries (end of Phase A, end of Phase B), Desktop does a deep validation pass — actually reads the generated files, cross-references against the unified blueprint, flags inconsistencies. Per-task checks are lightweight; phase-end checks are thorough.

---

## 8. What to commit, when

After each task that produces files, commit them. Don't accumulate uncommitted work — if a later task corrupts something, you want to be able to roll back precisely.

```powershell
git add .
git commit -m "Task N: <one-line description from development.md>"
```

Use the task number from `development.md` as the commit prefix. This makes the history readable later.

We do **not** push to GitHub yet. The repo stays local until Phase A is complete and Desktop has validated it. Public release is governed by §19 of the unified blueprint and §20 (responsible disclosure).

---

## 9. When something goes sideways

If Claude Code produces wrong output, three options in order of preference:

1. **Follow-up prompt.** Desktop drafts a corrective prompt; you paste it into Claude Code. Cheaper than starting over.
2. **Revert and retry.** `git checkout .` to discard uncommitted changes, or `git reset --hard HEAD~1` if you already committed bad work. Re-run the task with a sharpened prompt.
3. **Escalate to Desktop.** Sometimes the prompt itself is wrong. Tell Desktop what happened, let it revise the task in `development.md`, then re-run.

Do not let bad output stay in the repo "to fix later." It compounds into a mess.

---

## 10. Done

You're set up. Open Claude Desktop, load the CLIFFGUARD project, ask for "Task 1 from `development.md`."
