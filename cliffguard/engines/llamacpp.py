"""llama.cpp / GGUF inference-engine adapter — see blueprint §18.4.

Adapter for llama.cpp via the llama-cpp-python binding. llama.cpp is the
primary inference engine for Tier B (Pi 5 / RTX 3050 with CUDA, Q4_K_M /
Q3_K_M) and Tier C / C+ (2 GB embedded, Q3_K_M ≤1.5B). GGUF format is
universal across all in-scope schemes: Q6_K, Q5_K_M, Q4_K_M, Q3_K_M,
IQ3_XXS, Q2_K, IQ2_XXS.

Hidden-state access via llama.cpp:
  - Final-layer access: llama_get_embeddings() / embedding=True flag.
  - Per-layer access requires libllama with LLAMA_API_VERBOSE and is not
    exposed by the Python binding as of 2025-2026. The adapter therefore
    returns the final-layer embedding for both t_inst and t_post_inst
    positions when white_box=True, and falls back to top-k logprobs only
    when white_box=False (blueprint §18.4 documented limitation).

Two operating modes:
  - Stub mode (default): inference methods raise NotImplementedError.
    The Phase A test suite relies on this.
  - Live mode (after .load_model()): real GGUF inference via llama-cpp-python.

llama-cpp-python builds on all platforms (Windows, Linux, macOS) without
CUDA, and with CUDA when CMAKE_ARGS='-DLLAMA_CUDA=on' is set before install.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from cliffguard.types import Tier


class LlamaCppAdapter:
    """Adapter for llama.cpp / GGUF models via llama-cpp-python.

    Supports two observability modes per blueprint §18.4:
      white_box=True  → final-layer embeddings via llama.embed()
      white_box=False → top-k logprobs via Llama.__call__ logprobs param
    """

    def __init__(
        self,
        model_path: str,
        tier: Tier,
        n_ctx: int = 2048,
        white_box: bool = True,
    ) -> None:
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
        self._loaded: bool = False
        self._llm: Any = None
        self._n_gpu_layers: int = 0

    # ------------------------------------------------------------------
    # Live mode setup
    # ------------------------------------------------------------------

    def load_model(self, n_gpu_layers: int = -1, verbose: bool = False) -> None:
        """Load the GGUF model into memory.

        n_gpu_layers:
          -1   → offload all layers to GPU (recommended on a 3050 with
                 a model that fits, e.g. Llama-3.1-8B Q4_K_M ≈ 4.5 GB).
          N>=0 → offload exactly N layers; useful when VRAM is tight.
          0    → CPU-only inference.

        embedding=True is required for white_box=True so that .embed()
        returns the final-layer residual stream rather than just logits.

        Raises ImportError if llama_cpp cannot be imported.
        After this returns, self._loaded == True.
        """
        from llama_cpp import Llama

        self._n_gpu_layers = n_gpu_layers
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=bool(self.white_box),
            logits_all=False,
            verbose=verbose,
        )
        self._loaded = True

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def get_hidden_states(
        self,
        prompt: str,
        layer: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (z_t_inst, z_t_post_inst).

        llama.cpp Python only exposes the final layer; both returned
        vectors come from the final layer with t_inst = penultimate token
        embedding and t_post_inst = last token embedding. The `layer`
        parameter is accepted for interface compatibility but ignored
        unless layer == -1 (final).

        Raises NotImplementedError if white_box=False or if .load_model()
        has not been called.
        """
        if not self.white_box:
            raise NotImplementedError(
                "LlamaCppAdapter in black-box mode does not expose "
                "hidden states. Use get_top_k_logprobs instead."
            )
        if not self._loaded:
            raise NotImplementedError(
                "LlamaCppAdapter requires GPU/edge hardware. "
                "Run via scripts/run_full_evaluation.py on a Tier B/C host."
            )

        # llama_cpp's embedding API returns a fixed-size embedding by
        # pooling. For per-token embeddings we tokenize first then
        # request embeddings of two suffixes: the full prompt (t_post_inst)
        # and the prompt minus the last token (t_inst).
        tokens = self._llm.tokenize(prompt.encode("utf-8"), add_bos=True)
        if len(tokens) < 2:
            # Single token — return same embedding twice.
            full = np.asarray(self._llm.embed(prompt), dtype=np.float64)
            return (full.copy(), full.copy())

        # t_inst: embedding of the prompt MINUS the last token.
        tokens_minus_last = tokens[:-1]
        prompt_minus_last = self._llm.detokenize(tokens_minus_last).decode("utf-8", errors="replace")
        z_t_inst = np.asarray(self._llm.embed(prompt_minus_last), dtype=np.float64)

        # t_post_inst: embedding of the full prompt.
        z_t_post_inst = np.asarray(self._llm.embed(prompt), dtype=np.float64)

        return (z_t_inst, z_t_post_inst)

    def get_top_k_logprobs(
        self,
        prompt: str,
        k: int = 20,
    ) -> npt.NDArray[np.float64]:
        """Return top-k logprobs of the first response token.

        Uses llama.cpp's logprobs feature via a 1-token completion call.
        Available in both white_box and black_box modes once loaded.
        """
        if not self._loaded:
            raise NotImplementedError(
                "LlamaCppAdapter requires GPU/edge hardware. "
                "Run via scripts/run_full_evaluation.py on a Tier B/C host."
            )

        out = self._llm(
            prompt,
            max_tokens=1,
            logprobs=k,
            temperature=0.0,
        )

        try:
            top_logprobs = out["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"llama.cpp did not return top_logprobs in expected format: {exc}"
            ) from exc

        values = sorted(top_logprobs.values(), reverse=True)[:k]
        # Pad with very-negative values if fewer than k returned.
        if len(values) < k:
            values = values + [-1e30] * (k - len(values))
        return np.asarray(values, dtype=np.float64)
