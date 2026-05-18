"""Transformers + bitsandbytes hidden-state adapter — see blueprint §18.1.

Provides the HiddenStateAdapter Protocol and the TransformersBnbAdapter
concrete implementation for extracting residual-stream activations and
top-k logprobs from quantized HuggingFace models.

Two operating modes:
  - Stub mode (default after construction): every inference method raises
    NotImplementedError. This is what Phase A tests assert; do not change it.
  - Live mode (after .load_model() is called on a real GPU host): real
    forward passes via transformers + bitsandbytes return residual stream
    states and top-k logprobs.

Live mode requires the [gpu] extras to be installed:
  uv sync --extra gpu
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class HiddenStateAdapter(Protocol):
    """Adapter interface: given a prompt, return per-layer hidden
    states at t_post-inst and t_inst plus top-k logprobs of the
    first response token."""

    def get_hidden_states(
        self,
        prompt: str,
        layer: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (z_t_inst, z_t_post_inst). Shape: (hidden_dim,) each."""
        ...

    def get_top_k_logprobs(
        self,
        prompt: str,
        k: int = 20,
    ) -> npt.NDArray[np.float64]:
        """Return top-k log-probabilities of the first response token. Shape: (k,)."""
        ...


class TransformersBnbAdapter:
    """Concrete HiddenStateAdapter backed by transformers + bitsandbytes
    NF4 / INT8 quantization. See blueprint §18.1.

    Default construction only validates that torch is importable and stores
    config. Call .load_model() to actually load weights and switch to
    live mode. This two-step pattern keeps the Phase A test suite green
    (which mocks torch and verifies NotImplementedError on inference calls)
    while still permitting real inference on a GPU host.
    """

    def __init__(
        self,
        model_name_or_path: str,
        layer: int,
        quantization: str = "nf4",
    ) -> None:
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
        # Live-mode handles populated by load_model().
        self._loaded: bool = False
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._device: Any = None

    # ------------------------------------------------------------------
    # Live mode setup
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the model weights and tokenizer for live inference.

        Quantization values accepted:
          fp16  — no quantization (full half-precision)
          nf4   — bitsandbytes 4-bit NormalFloat with double quant
          int8  — bitsandbytes 8-bit
          awq   — pre-quantized AWQ checkpoint (requires AutoAWQ-converted weights)

        Raises ImportError if any required dependency is missing.
        Raises ValueError if self.quantization is unrecognised.
        After this returns, self._loaded == True and inference methods
        will execute real forward passes instead of raising NotImplementedError.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs: dict[str, Any] = {
            "output_hidden_states": True,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }

        if self.quantization == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif self.quantization == "nf4":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "int8":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "awq":
            # AutoAWQ pre-quantized; transformers loads natively.
            kwargs["torch_dtype"] = torch.float16
        else:
            raise ValueError(
                f"Unknown quantization {self.quantization!r}. "
                "Use one of: fp16, nf4, int8, awq."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, **{k: v for k, v in kwargs.items() if v is not None}
        )
        self._model.eval()
        self._loaded = True

    def _format_prompt(self, prompt: str) -> str:
        """Apply chat template if the tokenizer has one; otherwise return as-is."""
        if self._tokenizer is None:
            return prompt
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            try:
                return self._tokenizer.apply_chat_template(  # type: ignore[no-any-return]
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return prompt
        return prompt

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def get_hidden_states(
        self,
        prompt: str,
        layer: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (z_t_inst, z_t_post_inst) at the given residual-stream layer.

        Conventions (per Arditi 2024 / Zhao 2025):
          z_t_inst       = hidden state at the penultimate input token
                           (the last token of the user's instruction itself).
          z_t_post_inst  = hidden state at the final input position
                           (the position from which the model predicts the
                           first response token).

        In Phase A (stub mode) this raises NotImplementedError. After
        .load_model() it returns real activations.
        """
        if not self._loaded:
            raise NotImplementedError(
                "get_hidden_states requires GPU hardware. "
                "Run via scripts/run_full_evaluation.py on a GPU host."
            )

        torch = self._torch
        formatted = self._format_prompt(prompt)
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states)
        if layer < 0 or layer >= n_layers:
            raise ValueError(
                f"layer {layer} out of range [0, {n_layers - 1}] for this model"
            )

        layer_states = hidden_states[layer][0]  # (seq_len, hidden_dim)
        seq_len = layer_states.shape[0]

        if seq_len < 2:
            # Degenerate single-token input — t_inst == t_post_inst.
            z_t_inst = layer_states[-1].to(torch.float32).cpu().numpy().astype(np.float64)
            z_t_post_inst = z_t_inst.copy()
        else:
            z_t_inst = layer_states[-2].to(torch.float32).cpu().numpy().astype(np.float64)
            z_t_post_inst = layer_states[-1].to(torch.float32).cpu().numpy().astype(np.float64)

        return (z_t_inst, z_t_post_inst)

    def get_top_k_logprobs(
        self,
        prompt: str,
        k: int = 20,
    ) -> npt.NDArray[np.float64]:
        """Return the top-k log-probabilities of the first response token."""
        if not self._loaded:
            raise NotImplementedError(
                "get_top_k_logprobs requires GPU hardware. "
                "Run via scripts/run_full_evaluation.py on a GPU host."
            )

        torch = self._torch
        formatted = self._format_prompt(prompt)
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            last_logits = outputs.logits[0, -1, :]
            log_probs = torch.log_softmax(last_logits.float(), dim=-1)
            top_k = torch.topk(log_probs, k=min(k, log_probs.shape[0]))

        return top_k.values.cpu().numpy().astype(np.float64)  # type: ignore[no-any-return]
