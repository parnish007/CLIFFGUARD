import sys
from unittest.mock import MagicMock, patch

import pytest

from cliffguard.types import Tier
from cliffguard.engines.llamacpp import LlamaCppAdapter


def _make_adapter(**kwargs: object) -> LlamaCppAdapter:
    mock_llama_cpp = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
        return LlamaCppAdapter(
            model_path=str(kwargs.get("model_path", "/models/llama-2-7b.Q4_K_M.gguf")),
            tier=kwargs.get("tier", Tier.B),  # type: ignore[arg-type]
            n_ctx=int(kwargs.get("n_ctx", 2048)),
            white_box=bool(kwargs.get("white_box", True)),
        )


# ---------------------------------------------------------------------------
# ImportError guard
# ---------------------------------------------------------------------------


def test_adapter_raises_import_error_when_llama_cpp_missing() -> None:
    with patch.dict(sys.modules, {"llama_cpp": None}):  # type: ignore[dict-item]
        with pytest.raises(ImportError, match="uv sync --extra gpu"):
            LlamaCppAdapter("/models/test.gguf", tier=Tier.B)


def test_adapter_import_error_message_contains_install_hint() -> None:
    with patch.dict(sys.modules, {"llama_cpp": None}):  # type: ignore[dict-item]
        with pytest.raises(ImportError) as exc_info:
            LlamaCppAdapter("/models/test.gguf", tier=Tier.B)
    assert "uv sync --extra gpu" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Constructor stores args on self
# ---------------------------------------------------------------------------


def test_adapter_stores_model_path() -> None:
    adapter = _make_adapter(model_path="/data/mistral.Q3_K_M.gguf")
    assert adapter.model_path == "/data/mistral.Q3_K_M.gguf"


def test_adapter_stores_tier() -> None:
    adapter = _make_adapter(tier=Tier.C)
    assert adapter.tier == Tier.C


def test_adapter_stores_n_ctx() -> None:
    adapter = _make_adapter(n_ctx=4096)
    assert adapter.n_ctx == 4096


def test_adapter_stores_white_box() -> None:
    adapter = _make_adapter(white_box=False)
    assert adapter.white_box is False


def test_adapter_default_n_ctx_is_2048() -> None:
    adapter = _make_adapter()
    assert adapter.n_ctx == 2048


def test_adapter_default_white_box_is_true() -> None:
    adapter = _make_adapter()
    assert adapter.white_box is True


# ---------------------------------------------------------------------------
# get_hidden_states
# ---------------------------------------------------------------------------


def test_get_hidden_states_raises_not_implemented_white_box_true() -> None:
    adapter = _make_adapter(white_box=True)
    with pytest.raises(NotImplementedError):
        adapter.get_hidden_states("hello", layer=0)


def test_get_hidden_states_white_box_true_message_mentions_hardware() -> None:
    adapter = _make_adapter(white_box=True)
    with pytest.raises(NotImplementedError) as exc_info:
        adapter.get_hidden_states("hello", layer=0)
    assert "hardware" in str(exc_info.value).lower()


def test_get_hidden_states_raises_not_implemented_white_box_false() -> None:
    adapter = _make_adapter(white_box=False)
    with pytest.raises(NotImplementedError):
        adapter.get_hidden_states("hello", layer=0)


def test_get_hidden_states_white_box_false_message_contains_black_box() -> None:
    adapter = _make_adapter(white_box=False)
    with pytest.raises(NotImplementedError) as exc_info:
        adapter.get_hidden_states("hello", layer=0)
    assert "black-box" in str(exc_info.value)


# ---------------------------------------------------------------------------
# get_top_k_logprobs
# ---------------------------------------------------------------------------


def test_get_top_k_logprobs_raises_not_implemented_white_box_true() -> None:
    adapter = _make_adapter(white_box=True)
    with pytest.raises(NotImplementedError):
        adapter.get_top_k_logprobs("hello", k=20)


def test_get_top_k_logprobs_raises_not_implemented_white_box_false() -> None:
    adapter = _make_adapter(white_box=False)
    with pytest.raises(NotImplementedError):
        adapter.get_top_k_logprobs("hello", k=20)
