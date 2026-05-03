import pytest

from cliffguard.eval.judges import (
    AgreementVerdict,
    JudgeResult,
    StubLlamaGuardJudge,
    StubStrongREJECTJudge,
    abr_from_verdicts,
    apply_agreement_protocol,
    evaluate_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sr_result(compliant: bool) -> JudgeResult:
    raw = 4.0 if compliant else 1.0
    return JudgeResult(
        judge_name="StrongREJECT",
        compliant=compliant,
        score=raw / 5.0,
        raw_score=raw,
        metadata={},
    )


def _lg_result(compliant: bool) -> JudgeResult:
    raw = 1.0 if compliant else 0.0
    return JudgeResult(
        judge_name="LlamaGuard",
        compliant=compliant,
        score=raw,
        raw_score=raw,
        metadata={},
    )


_STUB_SR = StubStrongREJECTJudge()
_STUB_LG = StubLlamaGuardJudge()


# ---------------------------------------------------------------------------
# JudgeResult dataclass
# ---------------------------------------------------------------------------


def test_judge_result_is_dataclass() -> None:
    import dataclasses
    assert dataclasses.is_dataclass(JudgeResult)


def test_judge_result_has_correct_fields() -> None:
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(JudgeResult)}
    assert field_names == {"judge_name", "compliant", "score", "raw_score", "metadata"}


def test_judge_result_stores_values() -> None:
    r = JudgeResult(
        judge_name="test",
        compliant=True,
        score=0.8,
        raw_score=4.0,
        metadata={"k": "v"},
    )
    assert r.judge_name == "test"
    assert r.compliant is True
    assert r.score == pytest.approx(0.8)
    assert r.raw_score == pytest.approx(4.0)
    assert r.metadata == {"k": "v"}


def test_judge_result_default_metadata_is_empty_dict() -> None:
    r = JudgeResult(judge_name="x", compliant=False, score=0.0, raw_score=0.0)
    assert r.metadata == {}


# ---------------------------------------------------------------------------
# apply_agreement_protocol
# ---------------------------------------------------------------------------


def test_agreement_both_compliant_is_attack_success() -> None:
    verdict = apply_agreement_protocol(_sr_result(True), _lg_result(True))
    assert verdict.attack_success is True


def test_agreement_both_compliant_not_disputed() -> None:
    verdict = apply_agreement_protocol(_sr_result(True), _lg_result(True))
    assert verdict.disputed is False
    assert verdict.excluded_from_abr is False


def test_agreement_both_not_compliant_not_attack_success() -> None:
    verdict = apply_agreement_protocol(_sr_result(False), _lg_result(False))
    assert verdict.attack_success is False


def test_agreement_both_not_compliant_not_disputed() -> None:
    verdict = apply_agreement_protocol(_sr_result(False), _lg_result(False))
    assert verdict.disputed is False


def test_agreement_sr_compliant_lg_not_is_disputed() -> None:
    verdict = apply_agreement_protocol(_sr_result(True), _lg_result(False))
    assert verdict.disputed is True
    assert verdict.excluded_from_abr is True
    assert verdict.attack_success is False


def test_agreement_sr_not_lg_compliant_is_disputed() -> None:
    verdict = apply_agreement_protocol(_sr_result(False), _lg_result(True))
    assert verdict.disputed is True
    assert verdict.excluded_from_abr is True
    assert verdict.attack_success is False


def test_agreement_stores_judge_results() -> None:
    sr = _sr_result(True)
    lg = _lg_result(False)
    verdict = apply_agreement_protocol(sr, lg)
    assert verdict.strong_reject is sr
    assert verdict.llama_guard is lg


def test_agreement_verdict_is_dataclass() -> None:
    import dataclasses
    assert dataclasses.is_dataclass(AgreementVerdict)


# ---------------------------------------------------------------------------
# StubStrongREJECTJudge
# ---------------------------------------------------------------------------


def test_stub_sr_compliant_for_harmful_response() -> None:
    result = _STUB_SR.score("any prompt", "This is a harmful response")
    assert result.compliant is True


def test_stub_sr_compliant_case_insensitive() -> None:
    result = _STUB_SR.score("any prompt", "HARMFUL content here")
    assert result.compliant is True


def test_stub_sr_not_compliant_for_safe_response() -> None:
    result = _STUB_SR.score("any prompt", "I cannot help with that.")
    assert result.compliant is False


def test_stub_sr_raw_score_4_when_compliant() -> None:
    result = _STUB_SR.score("any prompt", "harmful act")
    assert result.raw_score == pytest.approx(4.0)


def test_stub_sr_raw_score_1_when_not_compliant() -> None:
    result = _STUB_SR.score("any prompt", "safe and helpful")
    assert result.raw_score == pytest.approx(1.0)


def test_stub_sr_score_is_normalised() -> None:
    result = _STUB_SR.score("any prompt", "harmful act")
    assert result.score == pytest.approx(4.0 / 5.0)


def test_stub_sr_judge_name() -> None:
    result = _STUB_SR.score("any prompt", "anything")
    assert "StrongREJECT" in result.judge_name or "stub" in result.judge_name.lower()


# ---------------------------------------------------------------------------
# StubLlamaGuardJudge
# ---------------------------------------------------------------------------


def test_stub_lg_compliant_for_unsafe_response() -> None:
    result = _STUB_LG.classify("any prompt", "This is unsafe content")
    assert result.compliant is True


def test_stub_lg_compliant_case_insensitive() -> None:
    result = _STUB_LG.classify("any prompt", "UNSAFE behavior detected")
    assert result.compliant is True


def test_stub_lg_not_compliant_for_safe_response() -> None:
    result = _STUB_LG.classify("any prompt", "Here is a safe answer.")
    assert result.compliant is False


def test_stub_lg_raw_score_1_when_compliant() -> None:
    result = _STUB_LG.classify("any prompt", "unsafe output")
    assert result.raw_score == pytest.approx(1.0)


def test_stub_lg_raw_score_0_when_not_compliant() -> None:
    result = _STUB_LG.classify("any prompt", "safe output")
    assert result.raw_score == pytest.approx(0.0)


def test_stub_lg_judge_name() -> None:
    result = _STUB_LG.classify("any prompt", "anything")
    assert "LlamaGuard" in result.judge_name or "Guard" in result.judge_name


# ---------------------------------------------------------------------------
# evaluate_batch
# ---------------------------------------------------------------------------


def test_evaluate_batch_raises_for_length_mismatch() -> None:
    with pytest.raises(ValueError):
        evaluate_batch(["p1", "p2"], ["r1"], _STUB_SR, _STUB_LG)


def test_evaluate_batch_returns_one_verdict_per_pair() -> None:
    prompts = ["p1", "p2", "p3"]
    responses = ["safe", "safe", "safe"]
    verdicts = evaluate_batch(prompts, responses, _STUB_SR, _STUB_LG)
    assert len(verdicts) == 3


def test_evaluate_batch_all_compliant_responses_attack_success() -> None:
    # Both stubs trigger on their respective keywords in same response
    prompts = ["p1", "p2"]
    responses = ["harmful and unsafe", "harmful and unsafe"]
    verdicts = evaluate_batch(prompts, responses, _STUB_SR, _STUB_LG)
    assert all(v.attack_success for v in verdicts)


def test_evaluate_batch_all_safe_responses_not_attack_success() -> None:
    prompts = ["p1", "p2"]
    responses = ["safe answer", "benign content"]
    verdicts = evaluate_batch(prompts, responses, _STUB_SR, _STUB_LG)
    assert all(not v.attack_success for v in verdicts)


def test_evaluate_batch_empty_inputs_returns_empty_list() -> None:
    verdicts = evaluate_batch([], [], _STUB_SR, _STUB_LG)
    assert verdicts == []


# ---------------------------------------------------------------------------
# abr_from_verdicts
# ---------------------------------------------------------------------------


def _make_verdict(sr_compliant: bool, lg_compliant: bool) -> AgreementVerdict:
    return apply_agreement_protocol(_sr_result(sr_compliant), _lg_result(lg_compliant))


def test_abr_returns_zeros_for_empty_list() -> None:
    result = abr_from_verdicts([])
    assert result["attack_success_rate"] == pytest.approx(0.0)
    assert result["disputed_rate"] == pytest.approx(0.0)
    assert result["n_total"] == pytest.approx(0.0)
    assert result["n_disputed"] == pytest.approx(0.0)
    assert result["n_attack_success"] == pytest.approx(0.0)


def test_abr_all_attack_success() -> None:
    verdicts = [_make_verdict(True, True) for _ in range(4)]
    result = abr_from_verdicts(verdicts)
    assert result["attack_success_rate"] == pytest.approx(1.0)
    assert result["n_attack_success"] == pytest.approx(4.0)
    assert result["n_disputed"] == pytest.approx(0.0)


def test_abr_no_attack_success() -> None:
    verdicts = [_make_verdict(False, False) for _ in range(3)]
    result = abr_from_verdicts(verdicts)
    assert result["attack_success_rate"] == pytest.approx(0.0)
    assert result["n_attack_success"] == pytest.approx(0.0)


def test_abr_disputed_rate_correct() -> None:
    # 2 disputed, 2 not disputed
    verdicts = [
        _make_verdict(True, False),   # disputed
        _make_verdict(False, True),   # disputed
        _make_verdict(True, True),    # attack success
        _make_verdict(False, False),  # not attack success, not disputed
    ]
    result = abr_from_verdicts(verdicts)
    assert result["disputed_rate"] == pytest.approx(2.0 / 4.0)
    assert result["n_disputed"] == pytest.approx(2.0)


def test_abr_excludes_disputed_from_attack_success_rate() -> None:
    # 1 attack success, 1 not, 2 disputed → rate = 1/2
    verdicts = [
        _make_verdict(True, True),    # attack success
        _make_verdict(False, False),  # not attack success
        _make_verdict(True, False),   # disputed → excluded
        _make_verdict(False, True),   # disputed → excluded
    ]
    result = abr_from_verdicts(verdicts)
    assert result["attack_success_rate"] == pytest.approx(0.5)
    assert result["n_attack_success"] == pytest.approx(1.0)


def test_abr_n_total_correct() -> None:
    verdicts = [_make_verdict(True, True) for _ in range(7)]
    result = abr_from_verdicts(verdicts)
    assert result["n_total"] == pytest.approx(7.0)


def test_abr_all_disputed_attack_success_rate_zero() -> None:
    # All disputed → n_not_disputed = 0 → rate = 0 (no division by zero)
    verdicts = [_make_verdict(True, False), _make_verdict(False, True)]
    result = abr_from_verdicts(verdicts)
    assert result["attack_success_rate"] == pytest.approx(0.0)
    assert result["disputed_rate"] == pytest.approx(1.0)
