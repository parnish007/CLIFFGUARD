import sys
from unittest.mock import MagicMock, patch

import pytest

from cliffguard.engines.vllm import VLLMAdapter


def _make_adapter(**kwargs: object) -> VLLMAdapter:
    mock_vllm = MagicMock()
    with patch.dict(sys.modules, {"vllm": mock_vllm}):
        return VLLMAdapter(
            model_name_or_path=str(kwargs.get("model_name_or_path", "meta-llama/Llama-2-7b-hf")),
            max_logprobs=int(kwargs.get("max_logprobs", 20)),
        )


def test_adapter_raises_import_error_when_vllm_missing() -> None:
    with patch.dict(sys.modules, {"vllm": None}):  # type: ignore[dict-item]
        with pytest.raises(ImportError, match="uv sync --extra gpu"):
            VLLMAdapter("some/model")


def test_adapter_import_error_message_contains_install_hint() -> None:
    with patch.dict(sys.modules, {"vllm": None}):  # type: ignore[dict-item]
        with pytest.raises(ImportError) as exc_info:
            VLLMAdapter("some/model")
    assert "uv sync --extra gpu" in str(exc_info.value)


def test_adapter_stores_model_name_or_path() -> None:
    adapter = _make_adapter(model_name_or_path="mistralai/Mistral-7B-v0.1")
    assert adapter.model_name_or_path == "mistralai/Mistral-7B-v0.1"


def test_adapter_stores_max_logprobs_default() -> None:
    adapter = _make_adapter()
    assert adapter.max_logprobs == 20


def test_adapter_stores_max_logprobs_custom() -> None:
    adapter = _make_adapter(max_logprobs=10)
    assert adapter.max_logprobs == 10


def test_get_hidden_states_raises_not_implemented() -> None:
    adapter = _make_adapter()
    with pytest.raises(NotImplementedError):
        adapter.get_hidden_states("hello", layer=0)


def test_get_hidden_states_message_contains_black_box() -> None:
    adapter = _make_adapter()
    with pytest.raises(NotImplementedError) as exc_info:
        adapter.get_hidden_states("hello", layer=0)
    assert "black-box" in str(exc_info.value)


def test_get_top_k_logprobs_raises_not_implemented() -> None:
    adapter = _make_adapter()
    with pytest.raises(NotImplementedError, match="GPU hardware"):
        adapter.get_top_k_logprobs("hello", k=20)
