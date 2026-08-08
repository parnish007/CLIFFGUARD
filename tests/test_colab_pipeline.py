"""The resume guarantee, exercised rather than assumed.

The whole value of the journal is that a session which dies part-way through can
be restarted and will not redo finished work, and will not skip work that only
looks finished. Those are two separate failure modes and both are cheap to get
wrong:

  * skipping too little wastes GPU hours, which is the visible failure;
  * skipping too much reports numbers nobody computed, which is the invisible one.

So every test here is about which steps actually execute on a second run. The
steps themselves are tiny Python one-liners, because the property under test is
the bookkeeping, not the science.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pytest

from scripts.colab_pipeline import (
    OK,
    _is_interpreter,
    Pipeline,
    Step,
    argv_fingerprint,
    ensure_repo,
    latest_run,
)


def touch_step(name: str, target: Path, **kwargs) -> Step:
    """A step that appends one line to a file, so executions can be counted."""
    code = (f"open(r'{target}', 'a').write('ran\\n')")
    return Step(name=name, argv=[sys.executable, "-c", code], **kwargs)


def runs(target: Path) -> int:
    return len(target.read_text().splitlines()) if target.exists() else 0


def make(tmp_path: Path, **kwargs) -> Pipeline:
    return Pipeline(journal=tmp_path / "journal.json",
                    log_dir=tmp_path / "logs", cwd=tmp_path, **kwargs)


def test_finished_step_is_not_rerun(tmp_path: Path) -> None:
    counter = tmp_path / "count.txt"
    for _ in range(3):
        pipe = make(tmp_path)
        pipe.add(touch_step("a", counter))
        assert pipe.run() == 0
    assert runs(counter) == 1, "a completed step ran again on a later session"


def test_only_the_unfinished_step_runs_after_a_failure(tmp_path: Path) -> None:
    first, third = tmp_path / "a.txt", tmp_path / "c.txt"
    failing = Step("b", [sys.executable, "-c", "import sys; sys.exit(3)"])

    pipe = make(tmp_path)
    pipe.add(touch_step("a", first)).add(failing).add(touch_step("c", third))
    assert pipe.run() == 1
    assert runs(first) == 1
    assert runs(third) == 0, "a later step ran even though an earlier one failed"

    # Second session: the failing step is replaced by one that works, which is
    # what a human fixing the bug and re-running the cell actually does.
    pipe = make(tmp_path)
    pipe.add(touch_step("a", first))
    pipe.add(Step("b", [sys.executable, "-c", "pass"]))
    pipe.add(touch_step("c", third))
    assert pipe.run() == 0
    assert runs(first) == 1, "the finished step was redone"
    assert runs(third) == 1


def test_changed_arguments_force_a_rerun(tmp_path: Path) -> None:
    counter = tmp_path / "count.txt"
    pipe = make(tmp_path)
    pipe.add(touch_step("a", counter))
    pipe.run()

    # Same step name, different command. Reusing the old result here would
    # attribute one command's output to another -- the invisible failure.
    other = tmp_path / "other.txt"
    pipe = make(tmp_path)
    pipe.add(touch_step("a", other))
    pipe.run()
    assert runs(other) == 1, "a step whose arguments changed was skipped"


def test_missing_output_forces_a_rerun(tmp_path: Path) -> None:
    """The journal is not trusted on its own when the step declares an output."""
    produced = tmp_path / "artifact.txt"
    step = lambda: Step(                                       # noqa: E731
        "make", [sys.executable, "-c",
                 f"open(r'{produced}', 'w').write('x')"],
        produces=produced)

    pipe = make(tmp_path)
    pipe.add(step())
    pipe.run()
    assert produced.exists()

    produced.unlink()
    pipe = make(tmp_path)
    pipe.add(step())
    pipe.run()
    assert produced.exists(), "journal said done, output was gone, step was skipped"


def test_timeout_is_recorded_and_not_retried(tmp_path: Path) -> None:
    """A hang repeats; retrying it burns the session for nothing.

    OOM kills are retried because the caches advance between attempts. A timeout
    carries no such promise, so it stops and says so.
    """
    pipe = make(tmp_path)
    pipe.add(Step("slow", [sys.executable, "-c", "import time; time.sleep(30)"],
                  timeout_s=1.0, retries_on_oom=2))
    assert pipe.run() == 1
    record = json.loads((tmp_path / "journal.json").read_text())["steps"]["slow"]
    assert record["status"] == "timeout"
    assert record["attempts"] == 1


def test_session_budget_stops_before_starting_a_long_step(tmp_path: Path) -> None:
    """Better to stop between steps than be killed inside one."""
    early, late = tmp_path / "early.txt", tmp_path / "late.txt"
    pipe = make(tmp_path, deadline_hours=0.05)               # three minutes
    pipe.add(touch_step("early", early, estimated_minutes=0.1))
    pipe.add(touch_step("late", late, estimated_minutes=600.0))
    pipe.run()
    assert runs(early) == 1
    assert runs(late) == 0
    record = json.loads((tmp_path / "journal.json").read_text())["steps"]["late"]
    assert record["status"] == "pending"

    # And with a fresh session the deferred step runs.
    pipe = make(tmp_path)
    pipe.add(touch_step("early", early, estimated_minutes=0.1))
    pipe.add(touch_step("late", late, estimated_minutes=600.0))
    pipe.run()
    assert runs(early) == 1
    assert runs(late) == 1


def test_a_corrupt_journal_does_not_strand_the_session(tmp_path: Path) -> None:
    """Drive can truncate a file; that must cost a redo, not the run."""
    counter = tmp_path / "count.txt"
    (tmp_path / "journal.json").write_text('{"steps": {"a": {"status": "o')
    pipe = make(tmp_path)
    pipe.add(touch_step("a", counter))
    assert pipe.run() == 0
    assert runs(counter) == 1
    assert (tmp_path / "journal.corrupt.json").exists(), "the bad file was not kept"


def test_output_is_streamed_to_a_log_file(tmp_path: Path) -> None:
    """A closed browser tab loses the cell output; the log is what remains."""
    pipe = make(tmp_path)
    pipe.add(Step("talk", [sys.executable, "-c", "print('a distinctive line')"]))
    pipe.run()
    assert "a distinctive line" in (tmp_path / "logs" / "talk.log").read_text()


def test_force_reruns_a_finished_step(tmp_path: Path) -> None:
    counter = tmp_path / "count.txt"
    for force in (None, ["a"]):
        pipe = make(tmp_path)
        pipe.add(touch_step("a", counter))
        pipe.run(force=force)
    assert runs(counter) == 2


def test_only_restricts_and_rejects_unknown_names(tmp_path: Path) -> None:
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    pipe = make(tmp_path)
    pipe.add(touch_step("a", a)).add(touch_step("b", b))
    pipe.run(only=["b"])
    assert runs(a) == 0 and runs(b) == 1

    with pytest.raises(SystemExit) as excinfo:
        make(tmp_path).add(touch_step("a", a)).run(only=["typo"])
    assert "typo" in str(excinfo.value)


def test_duplicate_step_names_are_refused(tmp_path: Path) -> None:
    pipe = make(tmp_path)
    pipe.add(touch_step("a", tmp_path / "a.txt"))
    with pytest.raises(ValueError, match="two steps"):
        pipe.add(touch_step("a", tmp_path / "b.txt"))


def test_step_name_must_be_usable_as_a_filename() -> None:
    with pytest.raises(ValueError, match="filename"):
        Step("a/b", [sys.executable, "-c", "pass"])
    with pytest.raises(ValueError, match="no command"):
        Step("a", [])


def test_fingerprint_ignores_the_interpreter_path() -> None:
    """A venv path change must not invalidate every step in the journal."""
    assert (argv_fingerprint(["/usr/bin/python", "s.py", "--n", "8"])
            == argv_fingerprint([r"C:\py\python.exe", "s.py", "--n", "8"]))
    assert (argv_fingerprint(["python", "s.py", "--n", "8"])
            != argv_fingerprint(["python", "s.py", "--n", "9"]))


def test_fingerprint_drops_only_an_actual_interpreter() -> None:
    """Dropping argv[0] unconditionally makes two different tools one step."""
    assert argv_fingerprint(["tool-a"]) != argv_fingerprint(["tool-b"])
    assert argv_fingerprint(["gzip", "x"]) != argv_fingerprint(["bzip2", "x"])
    assert (argv_fingerprint(["python3.11", "s.py"])
            == argv_fingerprint(["/opt/python", "s.py"]))


@pytest.mark.parametrize("name", ["python", "python3", "python3.11", "py",
                                  "pypy", "pypy3.10", "PYTHON.EXE",
                                  r"C:\py\python.exe", "/usr/bin/python3"])
def test_interpreter_names_are_recognised(name: str) -> None:
    assert _is_interpreter(name)


@pytest.mark.parametrize("name", ["pythontool", "mypython", "python-wrapper",
                                  "pypydoc", "pyright", "pytest"])
def test_lookalike_names_are_not_treated_as_interpreters(name: str) -> None:
    """`startswith("python")` swallows `pythontool`, which would make it and any
    other single-word command the same step in the journal."""
    assert not _is_interpreter(name)
    assert argv_fingerprint([name, "x"]) != argv_fingerprint(["other", "x"])


def test_fingerprint_separates_steps_that_differ_only_in_where_they_write() -> None:
    same = ["python", "s.py"]
    assert (argv_fingerprint(same, {"produces": "/a/out.json"})
            != argv_fingerprint(same, {"produces": "/b/out.json"}))


def test_relocating_the_artifacts_root_forces_a_rerun(
        tmp_path: Path, monkeypatch) -> None:
    """A run written into an ephemeral clone is not the run written to Drive,
    even though the command is character-for-character the same."""
    counter = tmp_path / "count.txt"
    monkeypatch.setenv("CLIFFGUARD_ARTIFACTS", str(tmp_path / "a"))
    make(tmp_path).add(touch_step("s", counter)).run()
    monkeypatch.setenv("CLIFFGUARD_ARTIFACTS", str(tmp_path / "b"))
    make(tmp_path).add(touch_step("s", counter)).run()
    assert runs(counter) == 2


def test_changing_the_declared_output_forces_a_rerun(tmp_path: Path) -> None:
    counter = tmp_path / "count.txt"
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text("{}")
    second.write_text("{}")
    make(tmp_path).add(touch_step("s", counter, produces=first)).run()
    make(tmp_path).add(touch_step("s", counter, produces=second)).run()
    assert runs(counter) == 2, "a step now writing elsewhere reused the old record"


def test_the_timeout_kills_grandchildren_and_does_not_hang(tmp_path: Path) -> None:
    """A worker holding the stdout pipe open would defeat the timeout entirely.

    Killing only the immediate child leaves the pipe's write end open in the
    grandchild, so the reader blocks past the deadline and the session is
    reclaimed inside the step instead of between steps -- the one outcome this
    class exists to prevent.
    """
    child = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "time.sleep(120)\n"
    )
    pipe = make(tmp_path)
    pipe.add(Step("spawner", [sys.executable, "-c", child], timeout_s=2.0))
    began = time.time()
    assert pipe.run() == 1
    elapsed = time.time() - began
    assert elapsed < 40, f"the step ran {elapsed:.0f}s past a 2s timeout"
    record = json.loads((tmp_path / "journal.json").read_text())["steps"]["spawner"]
    assert record["status"] == "timeout"


def test_ensure_repo_clones_then_updates(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake(argv, **kwargs):
        calls.append(list(argv))
        return None

    target = tmp_path / "clone"
    ensure_repo(target, "https://example.invalid/r.git", run=fake)
    assert calls[0][:2] == ["git", "clone"]

    calls.clear()
    (target / ".git").mkdir(parents=True)
    ensure_repo(target, "https://example.invalid/r.git", run=fake)
    assert [c[3] for c in calls] == ["fetch", "checkout", "reset"], (
        "an existing clone must be updated in place, not re-cloned over")


def test_lazy_argv_is_resolved_from_an_earlier_step(tmp_path: Path) -> None:
    """The grading steps cannot know the run directory until the ladder made it."""
    runs = tmp_path / "runs"
    seen = tmp_path / "seen.txt"

    def make_run_dir() -> None:
        (runs / "20260101-000000_abc1234_lbl").mkdir(parents=True)

    pipe = make(tmp_path)
    pipe.add(Step("ladder", [sys.executable, "-c",
                             f"import pathlib; pathlib.Path(r'{runs}/"
                             f"20260101-000000_abc1234_lbl').mkdir(parents=True)"]))
    pipe.add(Step("grade", lambda: [
        sys.executable, "-c",
        f"open(r'{seen}', 'w').write(r'{latest_run(runs, 'lbl')}')"]))
    assert pipe.run() == 0
    assert seen.read_text().endswith("20260101-000000_abc1234_lbl")


def test_a_lazy_step_that_cannot_resolve_fails_rather_than_crashes(
        tmp_path: Path) -> None:
    def boom():
        return [sys.executable, "-c", str(latest_run(tmp_path / "nope", "lbl"))]

    pipe = make(tmp_path)
    pipe.add(Step("grade", boom))
    assert pipe.run() == 1
    record = json.loads((tmp_path / "journal.json").read_text())["steps"]["grade"]
    assert record["status"] == "failed"
    assert "FileNotFoundError" in record["error"]


def test_latest_run_picks_the_newest_and_never_a_prefix_match(tmp_path: Path) -> None:
    for name in ("20260101-000000_a_qwen3b-xstest",
                 "20260102-000000_a_qwen3b-xstest",
                 "20260103-000000_a_qwen3b-xstest-extra"):
        (tmp_path / name).mkdir()
    # A label that is a prefix of another label must not match it: the two are
    # different measurements, and grading one as the other is silent and wrong.
    assert latest_run(tmp_path, "qwen3b-xstest").name.startswith("20260102")


def test_journal_records_every_step_the_report_prints(tmp_path: Path) -> None:
    pipe = make(tmp_path)
    pipe.add(touch_step("a", tmp_path / "a.txt"))
    pipe.add(touch_step("b", tmp_path / "b.txt"))
    pipe.run()
    state = json.loads((tmp_path / "journal.json").read_text())
    assert {"a", "b"} <= set(state["steps"])
    assert all(state["steps"][k]["status"] == OK for k in ("a", "b"))
    assert state["sessions"], "a session with no record cannot be audited later"
