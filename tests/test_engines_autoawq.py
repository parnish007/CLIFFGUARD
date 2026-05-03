import sys
from unittest.mock import MagicMock, patch

import pytest

from cliffguard.engines.autoawq import AutoAWQAdapter


def _make_adapter(**kwargs: object) -> AutoAWQAdapter:
    mock_awq = MagicMock()
    with patch.dict(sys.modules, {"awq": mock_awq}):
        return AutoAWQAdapter(
            model_name_or_path=str(kwargs.get("model_name_or_path", "TheBloke/Llama-2-7B-AWQ")),
            layer=int(kwargs.get("layer", 16)),
        )


def test_adapter_raises_import_error_when_awq_missing() -> None:
    with patch.dict(sys.modules, {"awq": None}):  # type: ignore[dict-item]
        with pytest.raises(ImportError, match="uv sync --extra gpu"):
            AutoAWQAdapter("some/model", layer=0)


def test_adapter_import_error_message_contains_install_hint() -> None:
    with patch.dict(sys.modules, {"awq": None}):  # type: ignore[dict-item]
        with pytest.raises(ImportError) as exc_info:
            AutoAWQAdapter("some/model", layer=0)
    assert "uv sync --extra gpu" in str(exc_info.value)


def test_adapter_stores_model_name_or_path() -> None:
    adapter = _make_adapter(model_name_or_path="TheBloke/Mistral-7B-AWQ")
    assert adapter.model_name_or_path == "TheBloke/Mistral-7B-AWQ"


def test_adapter_stores_layer() -> None:
    adapter = _make_adapter(layer=12)
    assert adapter.layer == 12


def test_get_hidden_states_raises_not_implemented() -> None:
    adapter = _make_adapter()
    with pytest.raises(NotImplementedError, match="GPU hardware"):
        adapter.get_hidden_states("hello", layer=0)


def test_get_top_k_logprobs_raises_not_implemented() -> None:
    adapter = _make_adapter()
    with pytest.raises(NotImplementedError, match="GPU hardware"):
        adapter.get_top_k_logprobs("hello", k=20)
