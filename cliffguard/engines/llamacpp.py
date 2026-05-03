"""llama.cpp / GGUF inference-engine adapter — see blueprint §18.4.

Adapter for llama.cpp via the llama-cpp-python binding. llama.cpp is
the primary inference engine for Tier B (Pi 5, Q4_K_M / Q3_K_M) and
Tier C / C+ (2 GB embedded, Q3_K_M ≤1.5B). GGUF format is universal
across all quantization schemes in scope: Q6_K, Q5_K_M, Q4_K_M,
Q3_K_M, IQ3_XXS, Q2_K, IQ2_XXS.

Hidden-state access via llama.cpp requires llama_get_embeddings() or
the embeddings=True flag on the Llama constructor (final-layer only).
Per-layer access requires libllama with LLAMA_API_VERBOSE and custom
patching — this is documented in blueprint §18.4 as a known
limitation. For Tier B the adapter uses final-layer embeddings only;
for Tier C it falls back to black-box top-k logprobs via the
logprobs parameter on Llama.__call__.

In scaffolding mode (Phase A), this is a typed stub only.
llama-cpp-python builds on all platforms (Windows, Linux, macOS)
without CUDA — no sys_platform marker needed.
Requires: uv sync --extra gpu
"""

import numpy as np
import numpy.typing as npt

from cliffguard.types import Tier


class LlamaCppAdapter:
    """Adapter for llama.cpp / GGUF models via llama-cpp-python.

    Supports two observability modes per blueprint §18.4:
      white_box=True  → final-layer embeddings via llama_get_embeddings
      white_box=False → top-k logprobs via Llama.__call__ logprobs param
    """

    def __init__(
        self,
        model_path: str,
        tier: Tier,
        n_ctx: int = 2048,
        white_box: bool = True,
    ) -> None:
        """Raise ImportError if llama_cpp is not importable.
        Parameters:
          model_path: path to the .gguf file.
          tier: deployment tier (affects observability mode).
          n_ctx: context window size.
          white_box: if True, use final-layer embeddings;
                     if False, use logprobs only (black-box path).
        """
        try:
            import llama_cpp  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LlamaCppAdapter requires llama-cpp-python. "
                "Install with: uv sync --extra gpu"
            ) from exc
        self.model_path = model_path
        self.tier = tier
        self.n_ctx = n_ctx
        self.white_box = white_box

    def get_hidden_states(
        self,
        prompt: str,
        layer: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return final-layer embeddings as a proxy for hidden states.
        Only available when white_box=True.
        Raises NotImplementedError if white_box=False.
        Note: llama.cpp exposes final-layer only; per-layer access
        requires custom libllama build (blueprint §18.4 limitation).
        """
        if not self.white_box:
            raise NotImplementedError(
                "LlamaCppAdapter in black-box mode does not expose "
                "hidden states. Use get_top_k_logprobs instead."
            )
        raise NotImplementedError(
            "LlamaCppAdapter requires GPU/edge hardware. "
            "Run via scripts/run_full_evaluation.py on a Tier B/C host."
        )

    def get_top_k_logprobs(
        self,
        prompt: str,
        k: int = 20,
    ) -> npt.NDArray[np.float64]:
        """Return top-k logprobs via Llama.__call__ logprobs parameter.
        Available in both white_box and black_box modes.
        """
        raise NotImplementedError(
            "LlamaCppAdapter requires GPU/edge hardware. "
            "Run via scripts/run_full_evaluation.py on a Tier B/C host."
        )
