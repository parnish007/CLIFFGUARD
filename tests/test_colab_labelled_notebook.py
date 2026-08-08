"""The notebook and the repository must not drift apart.

This project has already paid for that failure once: a notebook and the modules
it called diverged silently, and the notebook could not have produced a result on
any clone -- it named a file that did not exist, under a gitignored directory.
Nobody found out until the GPU time had been spent.

So the pipeline-construction cell of `colab_labelled.ipynb` is executed here, for
real, against stubs. What is checked is the shape of the plan: that every script
it names exists, that the step graph is ordered so a grader never precedes the
ladder that feeds it, that the lazily-built commands resolve against a run
directory produced by an earlier step, and that two suites cannot collide on one
cache. None of that needs a GPU, and all of it is invisible to a syntax check.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "colab_labelled.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_every_code_cell_parses(notebook: dict) -> None:
    for i, src in enumerate(_cells(notebook)):
        try:
            ast.parse(src)
        except SyntaxError as exc:                      # pragma: no cover
            pytest.fail(f"code cell {i} does not parse: {exc}")


def test_no_cell_uses_a_name_a_later_cell_defines(notebook: dict) -> None:
    """Run-all order is the only order a Colab user runs.

    A cell that reads a constant the config cell has not set yet raises
    NameError on the first execution -- and this notebook already had that,
    with the preflight checking JUDGE_MODEL two cells before it was assigned.
    That is the cheapest possible bug to catch and the most annoying to hit,
    because it fails after the pip install and the Drive mount.

    Deliberately approximate: it walks names rather than executing, so it
    over-approximates what a cell defines (both branches of an `if`, every
    loop target). Over-approximating means it cannot produce a false alarm,
    only miss one, which is the right direction for a guard nobody will debug.
    """
    # The `builtins` module, not `__builtins__`: inside an imported module the
    # latter is the namespace dict rather than the module, so dir() on it
    # returns dict methods and every `print` looks unbound.
    import builtins as _builtins

    bound: set[str] = set(dir(_builtins)) | {"__builtins__", "__name__"}
    problems: list[str] = []
    for i, src in enumerate(_cells(notebook)):
        tree = ast.parse(src)
        loaded: set[str] = set()
        stored: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                (loaded if isinstance(node.ctx, ast.Load) else stored).add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    stored.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
                stored.add(node.name)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                stored.add(node.name)
            elif isinstance(node, ast.arg):
                stored.add(node.arg)
            elif isinstance(node, ast.Global):
                stored.update(node.names)
        unbound = sorted(loaded - stored - bound)
        if unbound:
            problems.append(f"cell {i} uses {unbound} before any cell binds them")
        bound |= stored
    assert not problems, "\n".join(problems)


def test_the_notebook_only_names_scripts_that_exist(notebook: dict) -> None:
    """A renamed script is a two-second failure here and an hour's failure there."""
    named = set()
    for src in _cells(notebook):
        for token in src.replace('"', " ").replace("'", " ").split():
            if token.startswith("scripts/") and token.endswith(".py"):
                named.add(token)
    assert named, "the notebook names no repository script at all"
    missing = sorted(n for n in named if not (REPO / n).exists())
    assert not missing, f"notebook calls scripts that do not exist: {missing}"


def _pipeline_cell(notebook: dict) -> str:
    matches = [src for src in _cells(notebook) if "pipe = Pipeline(" in src]
    assert len(matches) == 1, f"expected one pipeline cell, found {len(matches)}"
    return matches[0]


@pytest.fixture
def built(notebook: dict, tmp_path: Path):
    """Execute the pipeline cell with a Pipeline that records instead of running."""
    from scripts import colab_pipeline

    recorded: list = []

    class Recording(colab_pipeline.Pipeline):
        def add(self, step):                            # type: ignore[override]
            recorded.append(step)
            return super().add(step)

        def run(self, only=None, force=None):           # type: ignore[override]
            return 0

    runs_root = tmp_path / "artifacts" / "runs"
    runs_root.mkdir(parents=True)

    namespace = {
        "__builtins__": __builtins__,
        "sys": sys, "json": json, "Path": Path,
        "Pipeline": Recording, "Step": colab_pipeline.Step,
        "latest_run": colab_pipeline.latest_run,
        "REPO_DIR": REPO, "DRIVE_ROOT": tmp_path, "ARTIFACTS": tmp_path / "artifacts",
        "CACHE_ROOT": tmp_path / "cache", "RUNS_ROOT": runs_root,
        "JOURNAL": tmp_path / "journal.json",
        "MODELS": [("qwen3b", "Qwen/Qwen2.5-3B-Instruct"),
                   ("phi35", "microsoft/Phi-3.5-mini-instruct"),
                   ("smol17", "HuggingFaceTB/SmolLM2-1.7B-Instruct")],
        "CORPUS": "xstest",
        "JUDGE_MODEL": "Qwen/Qwen2.5-7B-Instruct", "JUDGE_4BIT": True,
        "N_PER_CLASS": 150, "BITS": [8, 4], "MAX_NEW": 48,
        "BATCH": 8, "JUDGE_BATCH": 4, "DEADLINE_HOURS": 3.5,
        "gen_min": 30.0, "load_min": 12.0, "nll_min": 2.0, "judge_min": 15.0,
    }
    # The cell's import line would shadow the recording subclass.
    source = _pipeline_cell(notebook).replace(
        "from scripts.colab_pipeline import Pipeline, Step, latest_run", "")
    exec(compile(source, "<pipeline-cell>", "exec"), namespace)   # noqa: S102
    return recorded, runs_root, namespace


def test_every_model_gets_a_ladder_and_exactly_one_grading_pass(built) -> None:
    """Two steps per model, not three.

    The five-way grader also emits the three-way labels, because the three-way
    template defines REFUSE as exactly REFUSE+DEFLECT+DISCLAIM. A second grading
    step would be the same 7B sweep twice, which on a free T4 is the difference
    between fitting in a session and not.
    """
    steps, _, ns = built
    names = [s.name for s in steps]
    for model_key, _ in ns["MODELS"]:
        label = f"lab-{model_key}-{ns['CORPUS']}"
        assert f"ladder-{label}" in names
        assert f"grade-{label}" in names
    grading = [n for n in names if n.startswith(("grade-", "judge3-", "taxonomy-"))]
    assert len(grading) == len(ns["MODELS"]), (
        f"expected one grading step per model, got {grading}")
    assert names[0] == "suites"
    assert names[-2:] == ["analyse-labelled", "analyse-matrix"]


def test_a_grader_never_precedes_the_ladder_it_grades(built) -> None:
    """Ordering is the whole basis of lazy resolution: a grader that ran first
    would find no run directory and fail, or worse, find a stale one."""
    steps, _, ns = built
    names = [s.name for s in steps]
    for model_key, _ in ns["MODELS"]:
        label = f"lab-{model_key}-{ns['CORPUS']}"
        assert names.index(f"ladder-{label}") < names.index(f"grade-{label}")


def test_two_models_never_share_a_cache_directory(built) -> None:
    """Same corpus, same n, same token budget: the cache filenames would collide
    and the second model would silently read the first model's completions."""
    steps, _, _ = built
    caches = []
    for step in steps:
        if not step.name.startswith("ladder-"):
            continue
        argv = step.resolve_argv()
        caches.append(argv[argv.index("--cache") + 1])
    assert len(caches) == len(set(caches)), f"cache directories collide: {caches}"


def test_each_ladder_gets_its_own_prompt_file_and_label(built) -> None:
    steps, _, ns = built
    seen = set()
    for step in steps:
        if not step.name.startswith("ladder-"):
            continue
        argv = step.resolve_argv()
        prompts = argv[argv.index("--prompts") + 1]
        label = argv[argv.index("--label") + 1]
        assert Path(prompts).name.replace(".jsonl", "") == ns["CORPUS"]
        assert label not in seen, f"two ladders share the label {label}"
        seen.add(label)
        assert "--no-activations" in argv, (
            "the residual stream is a second full forward over every prompt and "
            "nothing downstream reads it")


def test_grader_commands_resolve_against_a_run_directory_made_earlier(built) -> None:
    steps, runs_root, ns = built
    label = f"lab-{ns['MODELS'][0][0]}-{ns['CORPUS']}"
    (runs_root / f"20260101-000000_abc1234_{label}").mkdir()

    for step in steps:
        if step.name == f"grade-{label}":
            argv = step.resolve_argv()
            assert argv[1].startswith("scripts/")
            assert label in argv[2], f"{step.name} points at {argv[2]}"
            assert "--judge-4bit" in argv, "JUDGE_4BIT was set and did not reach argv"
            produced = step.resolve_produces()
            assert produced is not None and label in str(produced)


def test_a_grader_whose_ladder_never_ran_fails_loudly(built) -> None:
    """Silently grading nothing is the failure this project has already had."""
    steps, _, ns = built
    label = f"lab-{ns['MODELS'][0][0]}-{ns['CORPUS']}"
    step = next(s for s in steps if s.name == f"grade-{label}")
    with pytest.raises(FileNotFoundError, match=label):
        step.resolve_argv()


def test_the_analyses_read_only_this_notebook_s_runs(built) -> None:
    """`--include lab-*` is what keeps these runs from being merged with the
    paper's, which measure different prompts and would silently pool."""
    steps, _, _ = built
    for name in ("analyse-labelled", "analyse-matrix"):
        argv = next(s for s in steps if s.name == name).resolve_argv()
        assert "--include" in argv and argv[argv.index("--include") + 1] == "*lab-*"


def test_run_directories_are_configured_onto_drive(notebook: dict) -> None:
    """Written relative to the clone they die with the session, after the GPU
    time that produced them has been spent."""
    joined = "\n".join(_cells(notebook))
    assert "CLIFFGUARD_ARTIFACTS" in joined
    assert 'os.environ["CLIFFGUARD_ARTIFACTS"]' in joined


def test_the_preflight_checks_the_label_tokens_before_any_gpu_time(
        notebook: dict) -> None:
    """Labels sharing a first token give verdicts that look fine and mean
    nothing; discovering that after an hour of grading is the expensive path."""
    cells = _cells(notebook)
    preflight = next(src for src in cells if "PREFLIGHT OK" in src)
    assert "label_first_token_ids" in preflight
    assert cells.index(preflight) < cells.index(_pipeline_cell(notebook))
