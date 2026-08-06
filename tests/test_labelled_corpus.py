"""The labelled-corpus path, and the two silent failures it was built around.

Both were found by running the thing rather than by reading it, and neither
raised on its own:

  * these suites are written class-ordered, so truncating one to `n` gave a
    single-class corpus -- every over-refusal count computed on prompts of which
    none was harmful;
  * `--n` counted per class through Fold A and in total through a labelled
    suite, so the same flag produced 500 prompts one way and 250 the other while
    both manifests recorded n=250.
"""

from __future__ import annotations

import json

import pytest

from scripts.run_local_ladder import load_labelled_prompts


def _corpus(tmp_path, rows, name="suite.jsonl"):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _class_ordered(n_benign=250, n_harmful=200):
    """The shape XSTest ships in: every safe prompt before every unsafe one."""
    return (
        [{"prompt": f"benign {i}", "harm_label": "benign", "source": "s"}
         for i in range(n_benign)]
        + [{"prompt": f"harmful {i}", "harm_label": "harmful", "source": "s"}
           for i in range(n_harmful)]
    )


# ---------------------------------------------------------------------------
# balance
# ---------------------------------------------------------------------------


def test_truncating_a_class_ordered_suite_keeps_both_classes(tmp_path) -> None:
    """Reading the file in order would give 16 benign prompts and no harmful
    ones, which is not an error but is not an experiment either."""
    path = _corpus(tmp_path, _class_ordered())
    prompts, _, labels = load_labelled_prompts(path, 8)
    assert len(prompts) == 16
    assert labels.count("harmful") == 8
    assert labels.count("benign") == 8


def test_n_counts_per_class_as_it_does_for_fold_a(tmp_path) -> None:
    """`load_prompts(250)` yields 250 harmful and 250 benign. The labelled path
    has to mean the same thing by the same flag."""
    path = _corpus(tmp_path, _class_ordered())
    prompts, _, labels = load_labelled_prompts(path, 100)
    assert len(prompts) == 200
    assert labels.count("harmful") == 100 and labels.count("benign") == 100


def test_prefix_property_survives_interleaving(tmp_path) -> None:
    """A run at one n must be a prefix of a run at a larger n, or two runs
    cannot be compared."""
    path = _corpus(tmp_path, _class_ordered())
    small, _, _ = load_labelled_prompts(path, 10)
    large, _, _ = load_labelled_prompts(path, 25)
    assert large[: len(small)] == small


def test_single_class_suite_is_not_reordered(tmp_path) -> None:
    """AdvBench is all harmful. There is nothing to balance, and the file's own
    order must survive."""
    rows = [{"prompt": f"harmful {i}", "harm_label": "harmful", "source": "adv"}
            for i in range(20)]
    path = _corpus(tmp_path, rows)
    prompts, _, labels = load_labelled_prompts(path, 5)
    assert prompts == [f"harmful {i}" for i in range(5)]
    assert set(labels) == {"harmful"}


def test_unbalanced_suite_exhausts_the_smaller_class_and_continues(tmp_path) -> None:
    rows = _class_ordered(n_benign=3, n_harmful=10)
    path = _corpus(tmp_path, rows)
    _, _, labels = load_labelled_prompts(path, None)
    assert labels.count("benign") == 3 and labels.count("harmful") == 10


# ---------------------------------------------------------------------------
# refusing bad input rather than guessing
# ---------------------------------------------------------------------------


def test_missing_corpus_names_the_builder(tmp_path) -> None:
    with pytest.raises(SystemExit, match="download_eval_suites"):
        load_labelled_prompts(tmp_path / "absent.jsonl", 10)


def test_unlabelled_row_is_refused(tmp_path) -> None:
    """An unlabelled corpus is the thing this path exists to replace; defaulting
    such a row to either class would silently fabricate the annotation."""
    path = _corpus(tmp_path, [
        {"prompt": "a", "harm_label": "harmful"},
        {"prompt": "b", "harm_label": "unknown"},
    ])
    with pytest.raises(SystemExit, match="harm_label"):
        load_labelled_prompts(path, 10)


def test_missing_harm_label_is_refused(tmp_path) -> None:
    path = _corpus(tmp_path, [{"prompt": "a"}, {"prompt": "b"}])
    with pytest.raises(SystemExit, match="harm_label"):
        load_labelled_prompts(path, 10)


def test_duplicate_prompts_are_dropped(tmp_path) -> None:
    """A repeat is the same prompt counted twice: it narrows every interval
    without adding information."""
    rows = [{"prompt": "same", "harm_label": "harmful"}] * 3 + [
        {"prompt": f"b{i}", "harm_label": "benign"} for i in range(4)]
    path = _corpus(tmp_path, rows)
    prompts, _, _ = load_labelled_prompts(path, None)
    assert len(prompts) == len(set(prompts)) == 5


def test_too_small_a_corpus_is_refused(tmp_path) -> None:
    path = _corpus(tmp_path, [{"prompt": "a", "harm_label": "harmful"}])
    with pytest.raises(SystemExit, match="usable prompts"):
        load_labelled_prompts(path, 10)


def test_blank_lines_and_whitespace_prompts_are_skipped(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"prompt": "real one", "harm_label": "harmful"}) + "\n"
        + "\n"
        + json.dumps({"prompt": "   ", "harm_label": "benign"}) + "\n"
        + "\n".join(json.dumps({"prompt": f"b{i}", "harm_label": "benign"})
                    for i in range(4)) + "\n",
        encoding="utf-8")
    prompts, _, _ = load_labelled_prompts(path, None)
    assert "   " not in prompts and len(prompts) == 5


def test_source_falls_back_to_the_file_stem(tmp_path) -> None:
    rows = [{"prompt": f"p{i}", "harm_label": "harmful"} for i in range(5)]
    path = _corpus(tmp_path, rows, name="advbench.jsonl")
    _, sources, _ = load_labelled_prompts(path, None)
    assert set(sources) == {"advbench"}
