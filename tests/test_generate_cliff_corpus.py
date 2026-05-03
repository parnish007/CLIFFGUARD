import json
import random

import pytest

from scripts.generate_cliff_corpus import main, make_cliff_prompt

_REQUIRED_FIELDS = {
    "text",
    "adversary",
    "scheme",
    "expected_blocked",
    "source",
    "cliff_margin_fp16",
    "cliff_margin_q3km",
}


# ---------------------------------------------------------------------------
# make_cliff_prompt
# ---------------------------------------------------------------------------


def test_make_cliff_prompt_has_all_required_fields() -> None:
    rng = random.Random(0)
    record = make_cliff_prompt(0, rng)
    assert _REQUIRED_FIELDS <= set(record.keys())


def test_make_cliff_prompt_adversary_is_a7() -> None:
    rng = random.Random(0)
    record = make_cliff_prompt(0, rng)
    assert record["adversary"] == "A7"


def test_make_cliff_prompt_scheme_is_gguf_q3km() -> None:
    rng = random.Random(0)
    record = make_cliff_prompt(0, rng)
    assert record["scheme"] == "GGUF_Q3_K_M"


def test_make_cliff_prompt_expected_blocked_is_true_string() -> None:
    rng = random.Random(0)
    record = make_cliff_prompt(0, rng)
    assert record["expected_blocked"] == "true"


def test_make_cliff_prompt_source_is_synthetic_phase_a() -> None:
    rng = random.Random(0)
    record = make_cliff_prompt(0, rng)
    assert record["source"] == "synthetic-phase-a"


def test_make_cliff_prompt_cliff_margin_fp16_in_range() -> None:
    rng = random.Random(7)
    for i in range(50):
        record = make_cliff_prompt(i, rng)
        val = float(record["cliff_margin_fp16"])
        assert 0.05 <= val <= 0.25, f"cliff_margin_fp16={val} out of [0.05, 0.25]"


def test_make_cliff_prompt_cliff_margin_q3km_in_range() -> None:
    rng = random.Random(7)
    for i in range(50):
        record = make_cliff_prompt(i, rng)
        val = float(record["cliff_margin_q3km"])
        assert -0.15 <= val <= 0.04, f"cliff_margin_q3km={val} out of [-0.15, 0.04]"


def test_make_cliff_prompt_returns_dict_of_strings() -> None:
    rng = random.Random(0)
    record = make_cliff_prompt(0, rng)
    assert all(isinstance(v, str) for v in record.values())


# ---------------------------------------------------------------------------
# main — file output
# ---------------------------------------------------------------------------


def test_main_writes_correct_line_count(tmp_path: pytest.TempPathFactory) -> None:
    out = tmp_path / "test_corpus.jsonl"  # type: ignore[operator]
    code = main(["--n", "10", "--seed", "1", "--output", str(out)])
    assert code == 0
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]  # type: ignore[union-attr]
    assert len(lines) == 10


def test_main_each_line_is_valid_json(tmp_path: pytest.TempPathFactory) -> None:
    out = tmp_path / "corpus.jsonl"  # type: ignore[operator]
    main(["--n", "5", "--seed", "2", "--output", str(out)])
    for line in out.read_text(encoding="utf-8").splitlines():  # type: ignore[union-attr]
        if line.strip():
            obj = json.loads(line)
            assert _REQUIRED_FIELDS <= set(obj.keys())


def test_main_each_line_has_correct_adversary(tmp_path: pytest.TempPathFactory) -> None:
    out = tmp_path / "corpus.jsonl"  # type: ignore[operator]
    main(["--n", "5", "--seed", "3", "--output", str(out)])
    for line in out.read_text(encoding="utf-8").splitlines():  # type: ignore[union-attr]
        if line.strip():
            obj = json.loads(line)
            assert obj["adversary"] == "A7"


def test_main_creates_parent_dirs(tmp_path: pytest.TempPathFactory) -> None:
    out = tmp_path / "nested" / "deep" / "corpus.jsonl"  # type: ignore[operator]
    code = main(["--n", "3", "--output", str(out)])
    assert code == 0
    assert out.exists()  # type: ignore[union-attr]


def test_main_deterministic_same_seed(tmp_path: pytest.TempPathFactory) -> None:
    out1 = tmp_path / "run1.jsonl"  # type: ignore[operator]
    out2 = tmp_path / "run2.jsonl"  # type: ignore[operator]
    main(["--n", "5", "--seed", "42", "--output", str(out1)])
    main(["--n", "5", "--seed", "42", "--output", str(out2)])
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")  # type: ignore[union-attr]


def test_main_different_seeds_produce_different_output(
    tmp_path: pytest.TempPathFactory,
) -> None:
    out1 = tmp_path / "seed1.jsonl"  # type: ignore[operator]
    out2 = tmp_path / "seed2.jsonl"  # type: ignore[operator]
    main(["--n", "5", "--seed", "1", "--output", str(out1)])
    main(["--n", "5", "--seed", "2", "--output", str(out2)])
    first1 = json.loads(out1.read_text(encoding="utf-8").splitlines()[0])  # type: ignore[union-attr]
    first2 = json.loads(out2.read_text(encoding="utf-8").splitlines()[0])  # type: ignore[union-attr]
    assert first1["cliff_margin_fp16"] != first2["cliff_margin_fp16"]


def test_main_first_record_deterministic_with_seed(
    tmp_path: pytest.TempPathFactory,
) -> None:
    out = tmp_path / "corpus.jsonl"  # type: ignore[operator]
    main(["--n", "3", "--seed", "99", "--output", str(out)])
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])  # type: ignore[union-attr]
    # Re-generate independently and compare
    rng = random.Random(99)
    expected = make_cliff_prompt(0, rng)
    assert first["cliff_margin_fp16"] == expected["cliff_margin_fp16"]
    assert first["cliff_margin_q3km"] == expected["cliff_margin_q3km"]
