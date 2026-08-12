"""What the generation cache key must separate, and what it must not.

The key decides whether text is regenerated. Two runs whose text can differ must
land on different keys, or the second silently inherits the first's output and
any experiment comparing them measures nothing.

That is not hypothetical here. The batch size was absent from the key while a
comment directly above claimed it was present -- the comment sat over the
ACTIVATIONS key, which does carry it. Round 5's batch-isolation step generates
the same prompts at batch 8 and batch 16 to test whether greedy nondeterminism
is a batching effect; under the old key the second run hit the first's cache,
the texts were identical by construction, and the step would have reported 0.0%
divergence and concluded that batch size does not explain a 9--12% effect.

These tests read the key's SHAPE out of the source rather than calling `main`,
which needs torch and a GPU. That is a weaker check than executing it and it is
the one that can run everywhere; `test_behavioural_ladder.py` covers behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SOURCE = (Path(__file__).resolve().parents[1] / "scripts"
          / "run_behavioural_ladder.py").read_text(encoding="utf-8")


def key_expression(name: str) -> str:
    """The f-string pieces of one cache path, joined."""
    match = re.search(rf"{name} = \(args\.cache /\n(.*?)\)\n", SOURCE, re.S)
    assert match, f"{name} is not built the way this test expects"
    return match.group(1)


@pytest.mark.parametrize("component,why", [
    ("_n{len(prompts)}", "a 250-prompt run is not a prefix of a 500-prompt one"),
    ("_t{args.max_new_tokens}", "48 tokens and 256 tokens are different text"),
    ("{corpus_key}", "different prompts, obviously"),
    ("{decode_key}", "greedy and sampled are different text, and so are two seeds"),
    ("{batch_key}", "fp16 kernels reduce in a batch-dependent order"),
])
def test_the_completions_key_separates(component: str, why: str) -> None:
    assert component in key_expression("cache_text"), why


def test_the_batch_key_is_the_generation_batch_not_the_activation_batch() -> None:
    """`--act-batch-size` is a different flag and would not separate the runs.

    Using it here would look right, pass the test above, and leave the defect
    exactly where it was: the batch-isolation step varies `--batch-size`.
    """
    assert re.search(r"batch_key = f\"_b\{args\.batch_size\}\"", SOURCE)


def test_batch_key_is_bound_where_the_nll_cache_reads_it() -> None:
    """Present in the source is not the same as in scope at the use site.

    `batch_key` was assigned inside the nested `run_scheme` and read again when
    the NLL cache path is built, which is in `main` and outside it. Every test
    above searches the source text, so all of them passed while a real run would
    raise NameError -- after generating every completion for every rung and
    immediately before scoring them, which is the most expensive point in the
    script to discover it.

    Compiling the module and walking the symbol table catches the scope, which
    reading the text cannot. This is a static check because executing `main`
    needs a GPU and model weights.
    """
    import symtable

    table = symtable.symtable(SOURCE, "run_behavioural_ladder.py", "exec")
    main_fn = table.lookup("main").get_namespace()
    assert "batch_key" in [s.get_name() for s in main_fn.get_symbols()], (
        "batch_key is not bound in main(), where the NLL cache path reads it")

    binding = main_fn.lookup("batch_key")
    assert binding.is_assigned(), (
        "batch_key is referenced in main() but never assigned there; it is "
        "being read out of a nested scope and will raise NameError at runtime")

    nested = {child.get_name() for child in main_fn.get_children()}
    assert "run_scheme" in nested, (
        "this test is anchored to run_scheme; if it was renamed, re-check that "
        "batch_key did not move back inside it")


def test_the_token_id_key_follows_the_completions_key() -> None:
    """Token ids must not outlive the text they came from.

    `make_prefix_run.py` cuts the short window from these ids, so a stale
    tokens_ file beside fresh completions would produce a prefix of text that
    was never generated -- and the whole point of that derivation is that both
    budgets describe one act of decoding.
    """
    assert re.search(
        r"cache_ids = cache_text\.with_name\(\s*cache_text\.name\.replace\(",
        SOURCE)


def test_the_activation_key_keeps_its_own_batch_size() -> None:
    """It is collected in a separate pass, at its own batch size."""
    assert "_b{args.act_batch_size}" in key_expression("cache_acts")


def test_the_comment_about_batching_sits_with_the_key_it_describes() -> None:
    """The defect was a true comment attached to the wrong key.

    Nothing enforces where a comment lives, so this checks the one property
    that mattered: the paragraph explaining why batching is in the key appears
    before `batch_key` is defined, not orphaned above an unrelated line.
    """
    explanation = SOURCE.index("fp16 kernels reduce in a batch-dependent order")
    definition = SOURCE.index('batch_key = f"_b{args.batch_size}"')
    completions = SOURCE.index("cache_text = (args.cache /")
    assert explanation < definition < completions
