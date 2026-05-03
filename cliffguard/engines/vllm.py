"""vLLM inference-engine adapter — see blueprint §18.3.

Adapter for vLLM, which provides PagedAttention-based high-throughput
inference. vLLM exposes logprobs via its sampling parameters but does
not expose per-layer hidden states by default — hidden-state access
requires a custom vLLM fork or the vLLM OpenAI-compatible API with
logprobs=True. This adapter targets the logprobs-only (black-box)
path for vLLM deployments.

In scaffolding mode (Phase A), this is a typed stub only.
Requires: uv sync --extra gpu (adds vllm to the gpu extra).
"""

import numpy as np
import numpy.typing as npt


class VLLMAdapter:
    def __init__(
        self,
        model_name_or_path: str,
        max_logprobs: int = 20,
    ) -> None:
        """Raise ImportError if vllm is not importable.
        max_logprobs: how many top-k logprobs to request (≤ 20
        per blueprint §2.3 black-box constraint)."""
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "VLLMAdapter requires vllm. "
                "Install with: uv sync --extra gpu"
            ) from exc
        self.model_name_or_path = model_name_or_path
        self.max_logprobs = max_logprobs

    def get_hidden_states(
        self,
        prompt: str,
        layer: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Not available for vLLM without custom fork.
        Always raises NotImplementedError.
        Use get_top_k_logprobs for the black-box path instead."""
        raise NotImplementedError(
            "vLLM does not expose per-layer hidden states by default. "
            "Use get_top_k_logprobs for the black-box path."
        )

    def get_top_k_logprobs(
        self,
        prompt: str,
        k: int = 20,
    ) -> npt.NDArray[np.float64]:
        """Stub — requires GPU. See blueprint §18.3."""
        raise NotImplementedError(
            "VLLMAdapter requires GPU hardware."
        )
