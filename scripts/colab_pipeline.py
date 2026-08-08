"""A resumable step runner, so an exceeded Colab runtime costs minutes and not hours.

Colab disconnects. Free runtimes are reclaimed after a few hours, Pro runtimes
after twelve, a browser tab closing takes the session with it, and a 3B model at
2 bits will occasionally take the host's RAM with it too. The measurements this
project runs are eight to twenty hours of GPU time, so "the session ended" is not
an exceptional case to handle at the end; it is the normal case to design around.

There are two layers of resumption and they do different jobs.

**Inside a step**, the ladder and the graders cache per scheme, keyed by model,
prompt count, token budget and decode settings, and those caches are written to
Drive the moment each scheme finishes. A run killed during the 3-bit rung comes
back and finds 8.5 through 3.5 bits already on disk. That layer already existed;
what this module adds is the guarantee that a cache is written atomically, so a
kill during the write cannot leave a truncated file that the next attempt reads
as real (see `write_json_atomic`).

**Between steps**, this module keeps a journal on Drive recording what has
finished. A step that completed with the same arguments is skipped. A step whose
arguments changed is re-run, and says why, because silently reusing the result of
a different command is how a pipeline reports numbers nobody computed.

The journal is deliberately dumb: one JSON file, human-readable, safe to delete.
If it disagrees with reality, reality wins -- a step is only skipped if its
declared output still exists.

Typical use, from a notebook cell::

    from scripts.colab_pipeline import Pipeline, Step

    pipe = Pipeline(journal=DRIVE / "cliffguard_journal.json",
                    log_dir=DRIVE / "logs",
                    deadline_hours=11.0)
    pipe.add(Step("suites", [sys.executable, "scripts/download_eval_suites.py",
                             "--download"], produces=Path("data/eval_suites/MANIFEST.json")))
    ...
    pipe.run()

Nothing here imports torch, so the module can be exercised on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Sequence

# rc == -9 is a SIGKILL, which on Colab means the host OOM killer, not a bug in
# the step. Retrying is the right response because each attempt starts from the
# caches the previous one left behind, so it makes progress rather than looping.
OOM_RETURNCODE = -9

# The statuses a step can be left in. `ok` is the only one that permits a skip.
OK, FAILED, TIMEOUT, KILLED, SKIPPED, PENDING = (
    "ok", "failed", "timeout", "killed", "skipped", "pending")


def _atomic_write_text(path: Path, text: str) -> None:
    """Rename-into-place, so a kill during the write cannot truncate the journal.

    The journal is the one file whose corruption would lose the whole session's
    bookkeeping, and it is rewritten after every step -- which is exactly the
    file most likely to be caught mid-write. A sibling temporary keeps the
    rename on one filesystem; Drive and /tmp are not the same one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                tmp.replace(path)
                return
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))
    finally:
        tmp.unlink(missing_ok=True)


def _new_process_group() -> dict[str, Any]:
    """Popen kwargs that make the child the leader of its own process group."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _kill_tree(proc: "subprocess.Popen[str]") -> None:
    """Kill the step and everything it started, then wait for the pipe to close.

    Falls back to killing the single process when the group call is unavailable
    or the group is already gone, because a partial kill is still better than
    none -- and the caller is already on the timeout path.
    """
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), 9)
        else:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, check=False)
    except (OSError, ProcessLookupError, PermissionError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _is_interpreter(arg: str) -> bool:
    """Whether argv[0] names a Python interpreter rather than the work itself."""
    stem = Path(arg).name.lower()
    if stem.endswith(".exe"):
        stem = stem[:-4]
    return stem.startswith("python") or stem in {"py", "pypy", "pypy3"}


def argv_fingerprint(argv: Sequence[str], extra: Any = None) -> str:
    """A stable hash of a command, with the interpreter path excluded.

    `sys.executable` differs between a local venv and Colab, and between two
    Colab runtimes after a Python upgrade. Including it would invalidate every
    step on a machine change, which is the opposite of what a resumable journal
    is for. What identifies the work is the script and its arguments.

    Only an argument that actually *is* an interpreter is dropped. Dropping
    argv[0] unconditionally would make `["tool-a"]` and `["tool-b"]` the same
    step, so switching between two single-word commands would reuse the wrong
    result -- a false skip with no symptom.

    `extra` carries whatever else changes the meaning of the step without
    appearing in its arguments: the declared output path and the environment
    overrides. A step that succeeded while writing somewhere else is not this
    step.
    """
    body = list(argv)
    if body and _is_interpreter(body[0]):
        body = body[1:]
    payload = json.dumps({"argv": body, "extra": extra}, sort_keys=True,
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Step:
    """One subprocess, with what it needs to be skipped safely.

    `produces` is the honesty check. Without it the journal is the only witness
    that a step ran, and a journal restored from a stale Drive copy would skip
    work whose output no longer exists. With it, a skip requires both the record
    and the artifact.

    `argv` and `produces` may each be a callable, resolved immediately before the
    step is considered. That exists for one specific case: a run directory is
    named with the timestamp at which it was created, so the grading steps cannot
    know their own argument until the ladder step ahead of them has run. Building
    the pipeline in two halves would work but would split the journal, which is
    the one thing that must stay whole across sessions.
    """

    name: str
    argv: Sequence[str] | Callable[[], Sequence[str]]
    produces: Path | Callable[[], Path | None] | None = None
    timeout_s: float = 3 * 3600.0
    retries_on_oom: int = 2
    # An estimate, used only to decide whether to start a step near the end of a
    # session. Wrong estimates cost a wasted check, never a wrong result.
    estimated_minutes: float = 30.0
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError(
                f"step name {self.name!r} must be non-empty and usable as a "
                "filename: it names this step's log file and its journal entry")
        if not callable(self.argv) and not self.argv:
            raise ValueError(f"step {self.name!r} has no command to run")

    def resolve_argv(self) -> list[str]:
        argv = self.argv() if callable(self.argv) else self.argv
        argv = [str(a) for a in argv]
        if not argv:
            raise ValueError(f"step {self.name!r} resolved to an empty command")
        return argv

    def resolve_produces(self) -> Path | None:
        target = self.produces() if callable(self.produces) else self.produces
        return Path(target) if target is not None else None


class Pipeline:
    """Steps in order, with a journal that survives the runtime that ran them."""

    def __init__(
        self,
        journal: Path,
        log_dir: Path | None = None,
        cwd: Path | None = None,
        deadline_hours: float | None = None,
        keep_going: bool = False,
    ) -> None:
        self.journal_path = Path(journal)
        self.log_dir = Path(log_dir) if log_dir else self.journal_path.parent / "logs"
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.keep_going = keep_going
        self.steps: list[Step] = []
        self.started = time.time()
        # A Colab session has a wall-clock limit and no warning before it ends.
        # Knowing it lets the pipeline stop cleanly between steps rather than be
        # killed inside one -- the difference between resuming at the next step
        # and resuming inside a half-finished grading pass.
        self.deadline = (self.started + deadline_hours * 3600.0
                         if deadline_hours else None)
        self.state: dict[str, Any] = self._load()

    # -- journal -----------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        if not self.journal_path.exists():
            return {"steps": {}, "sessions": []}
        try:
            loaded = json.loads(self.journal_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            # A journal that cannot be read is treated as absent: the cost is
            # re-checking steps whose caches will hit anyway, which is seconds.
            # Dying here would strand a session that is otherwise fine.
            spoiled = self.journal_path.with_suffix(".corrupt.json")
            try:
                self.journal_path.replace(spoiled)
                print(f"[journal] unreadable ({exc}); moved to {spoiled.name} "
                      "and starting a fresh one", flush=True)
            except OSError:
                print(f"[journal] unreadable ({exc}); starting fresh", flush=True)
            return {"steps": {}, "sessions": []}
        loaded.setdefault("steps", {})
        loaded.setdefault("sessions", [])
        return loaded

    def _save(self) -> None:
        _atomic_write_text(self.journal_path,
                           json.dumps(self.state, indent=2, default=str))

    # -- construction ------------------------------------------------------
    def add(self, step: Step) -> "Pipeline":
        if any(s.name == step.name for s in self.steps):
            raise ValueError(
                f"two steps are called {step.name!r}. The name keys this step's "
                "journal entry and its log file, so a duplicate would make one "
                "step's result stand in for the other's.")
        self.steps.append(step)
        return self

    def extend(self, steps: Iterable[Step]) -> "Pipeline":
        for step in steps:
            self.add(step)
        return self

    # -- decisions ---------------------------------------------------------
    def _fingerprint(self, step: Step, argv: Sequence[str]) -> str:
        """This step's identity: its command, its declared output, its environment.

        `CLIFFGUARD_ARTIFACTS` is singled out because it silently relocates every
        run directory the step writes. A step that succeeded with runs going to
        Drive is not the same step as one writing into an ephemeral clone, even
        though the command is character-for-character identical.
        """
        return argv_fingerprint(argv, {
            "produces": str(step.resolve_produces() or ""),
            "env": dict(sorted(step.env.items())),
            "artifacts": os.environ.get("CLIFFGUARD_ARTIFACTS", ""),
            "cwd": str(self.cwd),
        })

    def _skip_reason(self, step: Step, argv: Sequence[str]) -> str | None:
        """Why this step needs no work, or None if it does."""
        record = self.state["steps"].get(step.name)
        if not record or record.get("status") != OK:
            return None
        if record.get("argv") != self._fingerprint(step, argv):
            print(f"[{step.name}] command, output path or environment changed "
                  "since it last succeeded; re-running rather than reusing a "
                  "different step's result", flush=True)
            return None
        produces = step.resolve_produces()
        if produces is not None:
            target = produces if produces.is_absolute() else self.cwd / produces
            if not target.exists():
                print(f"[{step.name}] journal says done but {target} is gone; "
                      "re-running", flush=True)
                return None
        return f"done at {record.get('finished', 'an earlier session')}"

    def _budget_minutes(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, (self.deadline - time.time()) / 60.0)

    # -- execution ---------------------------------------------------------
    def _run_once(self, step: Step, argv: Sequence[str], log: Path) -> tuple[int, str]:
        """One attempt. Streams to stdout and to a log file at the same time.

        The log is what survives a closed browser tab. Colab's cell output lives
        in the front end, so a disconnect loses it entirely; a file on Drive is
        the only record of why a step failed at 3am.
        """
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **step.env}
        deadline = time.time() + step.timeout_s
        timed_out = threading.Event()

        # The step is launched as the leader of its own process group, so the
        # timeout can kill the whole tree. Killing only the immediate child is
        # not enough: a torch dataloader worker, or anything else the step
        # spawned, inherits the stdout pipe and holds its write end open, so
        # `for line in proc.stdout` below keeps blocking after the child is
        # gone. The timeout would then not be a timeout at all -- the session
        # would hang past its deadline and be reclaimed mid-step, which is the
        # exact outcome this class exists to avoid.
        proc = subprocess.Popen(
            list(argv), cwd=str(self.cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors="replace", **_new_process_group())

        def watchdog() -> None:
            while proc.poll() is None:
                if time.time() > deadline:
                    timed_out.set()
                    _kill_tree(proc)
                    return
                time.sleep(5.0)

        timer = threading.Thread(target=watchdog, daemon=True)
        timer.start()

        tail: list[str] = []
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(f"\n==== {step.name} :: {' '.join(argv)} :: "
                         f"{time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                handle.write(line)
                tail.append(line)
                if len(tail) > 40:
                    tail.pop(0)
            handle.flush()
        returncode = proc.wait()
        timer.join(timeout=1.0)
        if timed_out.is_set():
            return returncode, TIMEOUT
        if returncode == 0:
            return returncode, OK
        return returncode, KILLED if returncode == OOM_RETURNCODE else FAILED

    def run(self, only: Sequence[str] | None = None,
            force: Sequence[str] | None = None) -> int:
        """Every step that still needs doing. Returns the count that failed.

        `only` restricts to named steps; `force` re-runs named steps even when
        the journal says they are done. Both take step names, not globs, because
        a typo in a glob silently runs nothing.
        """
        chosen = [s for s in self.steps if only is None or s.name in only]
        if only:
            unknown = set(only) - {s.name for s in self.steps}
            if unknown:
                raise SystemExit(
                    f"no such step: {sorted(unknown)}. "
                    f"This pipeline has {[s.name for s in self.steps]}")
        forced = set(force or ())
        self.log_dir.mkdir(parents=True, exist_ok=True)
        session = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "steps": [s.name for s in chosen]}
        self.state["sessions"].append(session)
        self._save()

        failures = 0
        for step in chosen:
            try:
                argv = step.resolve_argv()
            except Exception as exc:                     # noqa: BLE001
                # A lazily-built command usually depends on an earlier step's
                # output. Failing to resolve is a real failure of this step, not
                # a crash of the pipeline: the remaining steps may still be
                # worth running, and the reason belongs in the journal.
                failures += 1
                self.state["steps"][step.name] = {
                    "status": FAILED, "returncode": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
                self._save()
                print(f"[{step.name}] could not build its command: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                if not self.keep_going:
                    break
                continue

            if step.name not in forced:
                reason = self._skip_reason(step, argv)
                if reason:
                    print(f"[{step.name}] skipped -- {reason}", flush=True)
                    continue

            budget = self._budget_minutes()
            if budget is not None and budget < step.estimated_minutes:
                print(f"\n[{step.name}] NOT STARTED: about {step.estimated_minutes:.0f} "
                      f"minutes of work with {budget:.0f} minutes left in this "
                      "session. Stopping cleanly here so the next session resumes "
                      "at this step instead of inside it.", flush=True)
                self.state["steps"][step.name] = {
                    "status": PENDING, "reason": "insufficient session budget",
                    "argv": self._fingerprint(step, argv)}
                self._save()
                break

            log = self.log_dir / f"{step.name}.log"
            attempts = 0
            while True:
                attempts += 1
                print(f"\n{'=' * 72}\n[{step.name}] attempt {attempts}: "
                      f"{' '.join(argv)}\n{'=' * 72}", flush=True)
                began = time.time()
                returncode, status = self._run_once(step, argv, log)
                elapsed = time.time() - began
                self.state["steps"][step.name] = {
                    "status": status, "returncode": returncode,
                    "argv": self._fingerprint(step, argv),
                    "command": list(argv),
                    "attempts": attempts,
                    "minutes": round(elapsed / 60.0, 2),
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "log": str(log),
                }
                self._save()
                if status == OK:
                    print(f"[{step.name}] ok in {elapsed / 60:.1f} min", flush=True)
                    break
                if status == KILLED and attempts <= step.retries_on_oom:
                    # Worth retrying precisely because the caches advanced: the
                    # next attempt skips every scheme this one finished. A step
                    # that is genuinely too big for the machine will exhaust the
                    # retries and stop, rather than loop.
                    print(f"[{step.name}] killed by the OOM killer (rc={returncode}) "
                          f"after {elapsed / 60:.1f} min. Retrying -- the caches "
                          "written so far mean the retry starts further along.",
                          flush=True)
                    continue
                failures += 1
                print(f"[{step.name}] {status} (rc={returncode}) after "
                      f"{elapsed / 60:.1f} min; log at {log}", flush=True)
                for line in ("last lines:", *_tail(log, 15)):
                    print(f"    {line.rstrip()}", flush=True)
                break
            if failures and not self.keep_going:
                print("\nstopping: a step failed and keep_going is off. Fix it and "
                      "re-run this cell; the finished steps will be skipped.",
                      flush=True)
                break
        session["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        self.report()
        return failures

    # -- reporting ---------------------------------------------------------
    def report(self) -> None:
        print(f"\n{'=' * 72}\npipeline state ({self.journal_path})\n{'=' * 72}")
        print(f"{'step':28s} {'status':9s} {'rc':>5s} {'min':>7s}  attempts")
        for step in self.steps:
            record = self.state["steps"].get(step.name, {})
            print(f"{step.name:28s} {record.get('status', '-'):9s} "
                  f"{str(record.get('returncode', '-')):>5s} "
                  f"{str(record.get('minutes', '-')):>7s}  "
                  f"{record.get('attempts', '-')}")
        pending = [s.name for s in self.steps
                   if self.state["steps"].get(s.name, {}).get("status") != OK]
        if pending:
            print(f"\nstill to do: {pending}")
            print("Re-run the pipeline cell in a fresh runtime; finished steps are "
                  "skipped and partly-finished ones resume from their caches.")
        else:
            print("\nevery step is done.")


def latest_run(runs_root: Path, label: str) -> Path:
    """The most recent run directory carrying `label`, for a step that grades it.

    Run directories are named `<utc>_<git-sha>_<label>`, so the label is knowable
    when the pipeline is built and the directory is not. Resolving by label at
    the moment the grading step starts is what lets a resumed session find the
    ladder output a previous session produced.

    Newest wins, and that is a deliberate choice with a sharp edge: re-running a
    ladder step with different settings under the same label creates a second
    directory, and every grading step after it will address the new one. That is
    the behaviour a re-run wants. It is also why the label carries the model and
    the suite -- two different measurements must never share one.
    """
    runs_root = Path(runs_root)
    matches = sorted((p for p in runs_root.glob(f"*_{label}") if p.is_dir()),
                     key=lambda p: p.name)
    if not matches:
        existing = sorted(p.name for p in runs_root.glob("*") if p.is_dir())
        raise FileNotFoundError(
            f"no run directory ending in _{label} under {runs_root}. "
            f"Present: {existing[-8:] if existing else 'none'}. The step that "
            "produces it either has not run or wrote somewhere else -- check "
            "CLIFFGUARD_ARTIFACTS.")
    return matches[-1]


def _tail(path: Path, n: int) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return ["(log unreadable)"]


def ensure_repo(root: Path, url: str, branch: str = "main",
                run: Callable[..., Any] = subprocess.run) -> Path:
    """Clone the repository, or bring an existing clone up to date.

    Colab notebooks usually re-clone into a fresh /content on every session,
    which is fine and is what makes the code in this repository the single
    source of truth for what runs. When the clone lives on Drive instead --
    slower, but it keeps local edits -- a plain clone would fail on the second
    session, so this handles both.
    """
    root = Path(root)
    if (root / ".git").exists():
        run(["git", "-C", str(root), "fetch", "--depth", "1", "origin", branch],
            check=True)
        run(["git", "-C", str(root), "checkout", branch], check=True)
        run(["git", "-C", str(root), "reset", "--hard", f"origin/{branch}"],
            check=True)
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--branch", branch, url, str(root)],
            check=True)
    return root
