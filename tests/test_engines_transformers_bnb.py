import sys
from unittest.mock import MagicMock, patch

import pytest

from cliffguard.engines.transformers_bnb import HiddenStateAdapter, TransformersBnbAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter_with_mock_torch(**kwargs: object) -> TransformersBnbAdapter:
    """Instantiate TransformersBnbAdapter with torch patched to a mock so the
    ImportError guard passes on machines without GPU dependencies."""
    mock_torch = MagicMock()
    with patch.dict(sys.modules, {"torch": mock_torch}):
        return TransformersBnbAdapter(
            model_name_or_path=str(kwargs.get("model_name_or_path", "meta-llama/Llama-2-7b-hf")),
            layer=int(kwargs.get("layer", 16)),
            quantization=str(kwargs.get("quantization", "nf4")),
        )


# ---------------------------------------------------------------------------
# Protocol import
# ---------------------------------------------------------------------------


def test_hidden_state_adapter_protocol_importable() -> None:
    assert HiddenStateAdapter is not None


def test_transformers_bnb_adapter_importable() -> None:
    assert TransformersBnbAdapter is not None


# ---------------------------------------------------------------------------
# ImportError guard
# ---------------------------------------------------------------------------


def test_adapter_raises_import_error_when_torch_missing() -> None:
    with patch.dict(sys.modules, {"torch": None}):  # type: ignore[dict-item]
        with pytest.raises(ImportError, match="uv sync --extra gpu"):
            TransformersBnbAdapter("some/model", layer=0)


def test_adapter_import_error_message_contains_install_hint() -> None:
    with patch.dict(sys.modules, {"torch": None}):  # type: ignore[dict-item]
        with pytest.raises(ImportError) as exc_info:
            TransformersBnbAdapter("some/model", layer=0)
    assert "uv sync --extra gpu" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Successful construction (torch mocked to succeed)
# ---------------------------------------------------------------------------


def test_adapter_stores_model_name_or_path() -> None:
    adapter = _make_adapter_with_mock_torch(model_name_or_path="mistralai/Mistral-7B-v0.1")
    assert adapter.model_name_or_path == "mistralai/Mistral-7B-v0.1"


def test_adapter_stores_layer() -> None:
    adapter = _make_adapter_with_mock_torch(layer=8)
    assert adapter.layer == 8


def test_adapter_stores_quantization_default() -> None:
    adapter = _make_adapter_with_mock_torch()
    assert adapter.quantization == "nf4"


def test_adapter_stores_quantization_int8() -> None:
    adapter = _make_adapter_with_mock_torch(quantization="int8")
    assert adapter.quantization == "int8"


# ---------------------------------------------------------------------------
# Stub methods raise NotImplementedError
# ---------------------------------------------------------------------------


def test_get_hidden_states_raises_not_implemented() -> None:
    adapter = _make_adapter_with_mock_torch()
    with pytest.raises(NotImplementedError, match="GPU hardware"):
        adapter.get_hidden_states("hello", layer=0)


def test_get_top_k_logprobs_raises_not_implemented() -> None:
    adapter = _make_adapter_with_mock_torch()
    with pytest.raises(NotImplementedError, match="GPU hardware"):
        adapter.get_top_k_logprobs("hello", k=20)


def test_get_hidden_states_error_message_mentions_script() -> None:
    adapter = _make_adapter_with_mock_torch()
    with pytest.raises(NotImplementedError) as exc_info:
        adapter.get_hidden_states("hello", layer=0)
    assert "run_full_evaluation.py" in str(exc_info.value)


def test_get_top_k_logprobs_error_message_mentions_script() -> None:
    adapter = _make_adapter_with_mock_torch()
    with pytest.raises(NotImplementedError) as exc_info:
        adapter.get_top_k_logprobs("hello")
    assert "run_full_evaluation.py" in str(exc_info.value)
