"""Transformers + bitsandbytes hidden-state adapter — see blueprint §18.1.

Provides the HiddenStateAdapter Protocol and the TransformersBnbAdapter
concrete implementation for extracting residual-stream activations and
top-k logprobs from quantized HuggingFace models.

In Phase A this module contains typed stubs only. Real execution requires
GPU hardware and the [gpu] optional-dependencies group:
  uv sync --extra gpu
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt


class HiddenStateAdapter(Protocol):
    """Adapter interface: given a prompt, return per-layer hidden
    states at t_post-inst and t_inst plus top-k logprobs of the
    first response token.

    All methods are documented as stubs — real implementation
    requires GPU and is called by Phase B scripts only.
    """

    def get_hidden_states(
        self,
        prompt: str,
        layer: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (z_t_inst, z_t_post_inst) — hidden states at the
        given layer for t_inst and t_post-inst positions.
        Shape: (hidden_dim,) each."""
        ...

    def get_top_k_logprobs(
        self,
        prompt: str,
        k: int = 20,
    ) -> npt.NDArray[np.float64]:
        """Return the top-k log-probabilities of the first response
        token. Shape: (k,)."""
        ...


class TransformersBnbAdapter:
    """Concrete HiddenStateAdapter backed by transformers +
    bitsandbytes NF4 / INT8 quantization.

    Implements forward hooks to extract residual stream activations
    at a chosen layer per blueprint §18.1.

    In Phase A this class is a typed stub only — __init__ raises
    ImportError if torch is not installed, preventing accidental
    use on machines without GPU dependencies.
    """

    def __init__(
        self,
        model_name_or_path: str,
        layer: int,
        quantization: str = "nf4",
    ) -> None:
        """Raise ImportError immediately if torch is not importable.
        Document parameters:
          model_name_or_path: HuggingFace model id or local path.
          layer: residual-stream layer to hook (0-indexed).
          quantization: "nf4" or "int8" (bitsandbytes modes).
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TransformersBnbAdapter requires torch. "
                "Install with: uv sync --extra gpu"
            ) from exc
        self.model_name_or_path = model_name_or_path
        self.layer = layer
        self.quantization = quantization

    def get_hidden_states(
        self,
        prompt: str,
        layer: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Stub — requires GPU. See blueprint §18.1 for hook
        implementation details."""
        raise NotImplementedError(
            "get_hidden_states requires GPU hardware. "
            "Run via scripts/run_full_evaluation.py on a GPU host."
        )

    def get_top_k_logprobs(
        self,
        prompt: str,
        k: int = 20,
    ) -> npt.NDArray[np.float64]:
        """Stub — requires GPU. See blueprint §18.1."""
        raise NotImplementedError(
            "get_top_k_logprobs requires GPU hardware. "
            "Run via scripts/run_full_evaluation.py on a GPU host."
        )
