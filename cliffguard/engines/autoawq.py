"""AutoAWQ inference-engine adapter — see blueprint §18.2.

Adapter for AWQ-INT4 quantized models via the autoawq library.
AWQ (Activation-aware Weight Quantization) preserves salient weights
and achieves better perplexity than GPTQ at 4-bit on most benchmarks.

In scaffolding mode (Phase A), this is a typed stub only.
Requires: uv sync --extra gpu (adds autoawq to the gpu extra).
"""

import numpy as np
import numpy.typing as npt


class AutoAWQAdapter:
    def __init__(
        self,
        model_name_or_path: str,
        layer: int,
    ) -> None:
        """Raise ImportError if awq is not importable."""
        try:
            import awq  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "AutoAWQAdapter requires autoawq. "
                "Install with: uv sync --extra gpu"
            ) from exc
        self.model_name_or_path = model_name_or_path
        self.layer = layer

    def get_hidden_states(
        self,
        prompt: str,
        layer: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Stub — requires GPU. See blueprint §18.2."""
        raise NotImplementedError(
            "AutoAWQAdapter requires GPU hardware."
        )

    def get_top_k_logprobs(
        self,
        prompt: str,
        k: int = 20,
    ) -> npt.NDArray[np.float64]:
        """Stub — requires GPU. See blueprint §18.2."""
        raise NotImplementedError(
            "AutoAWQAdapter requires GPU hardware."
        )
