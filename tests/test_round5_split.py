"""The derived Round 5 notebooks must stay in step with the master.

Round 5 is too long for one free Colab session, so it ships as two notebooks
generated from one master. The failure mode that matters is silent: the master
gets edited, a guard line moves, and the splitter emits a notebook where a step
it was supposed to gate runs anyway -- burning an hour of GPU on work the other
part already did, or worse, running it twice on different data.

So these check the contract rather than the output prose: every patch point
still resolves, the parts partition the steps, every emitted cell is valid
Python, and each step is gated in exactly the notebook that should not run it.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "notebooks" / "colab_round5.ipynb"


def splitter():
    path = REPO / "scripts" / "split_round5.py"
    spec = importlib.util.spec_from_file_location("_split_round5", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(
    not MASTER.is_file(), reason="round 5 master notebook not in this checkout")


def test_every_patch_point_resolves_against_the_master() -> None:
    """A moved guard must fail loudly, not produce an ungated notebook."""
    m = splitter()
    import os
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        m.check(m.load())          # raises SystemExit if anything moved
    finally:
        os.chdir(cwd)


def test_the_parts_partition_the_steps() -> None:
    m = splitter()
    a, b = m.PARTS["a"][0], m.PARTS["b"][0]
    assert not (a & b), "a step in both parts would be run twice"
    assert a | b == set(m.STEP_CELLS), "every step must land in some part"


@pytest.mark.parametrize("part", ["a", "b"])
def test_emitted_notebook_is_valid_python_and_gates_correctly(part: str) -> None:
    m = splitter()
    import os
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        nb = m.build_part(m.load(), part)
    finally:
        os.chdir(cwd)

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            ast.parse(src)
        except SyntaxError as exc:              # pragma: no cover - failure path
            pytest.fail(f"cell {i} of round 5{part} is not valid Python: {exc}")

    runs = m.PARTS[part][0]
    for step, (idx, _) in m.STEP_CELLS.items():
        src = "".join(nb["cells"][idx]["source"])
        gated = f"want({step!r})" in src
        assert gated == (step not in runs), (
            f"round 5{part}: {step} is {'gated' if gated else 'active'} but "
            f"this part runs {sorted(runs)}")

    config = "".join(nb["cells"][m.CONFIG_CELL]["source"])
    assert f"RUN_STEPS = {sorted(runs)!r}" in config
    assert "def want(" in config, "the gate helper must be defined with RUN_STEPS"


@pytest.mark.parametrize("part", ["a", "b"])
def test_helper_is_defined_before_any_step_uses_it(part: str) -> None:
    """`want` lives in the config cell, which must precede every step cell."""
    m = splitter()
    assert all(m.CONFIG_CELL < idx for idx, _ in m.STEP_CELLS.values()), (
        "a step cell precedes the configuration cell, so want() would be "
        "undefined when that step runs")
