"""Harmfulness-direction extractor — see blueprint §5.3, §12.3.

Implements the Zhao et al. recipe for extracting the harmfulness direction ĥ
from a model's residual stream at token position t_inst:

  1. Collect paired hidden states z_l(t_inst) for N matched pairs of harmful
     and harmless prompts (matched on surface form to control confounds).
  2. Compute the paired-difference mean:
       d = mean(z_harmful_i - z_harmless_i)   for i in 1..N
  3. Normalise: ĥ = d / ||d||

The result ĥ is used by PROBE-RM alongside the refusal direction r̂.

Key distinction from refusal_direction.py (Arditi):
  - Arditi uses unpaired difference-in-means at t_post_inst
  - Zhao uses PAIRED differences at t_inst to control surface-form confounds

Per blueprint §12.4: orthogonality of ĥ and r̂ is checked after extraction;
|ĥ · r̂| > ORTHOGONALITY_WARN_THRESHOLD triggers a UserWarning.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from cliffguard.engines.transformers_bnb import HiddenStateAdapter

ORTHOGONALITY_WARN_THRESHOLD: float = 0.3


def paired_difference_mean(
    harmful_states: npt.NDArray[np.float64],
    harmless_states: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the paired-difference mean harmfulness direction.

    harmful_states:  shape (N, hidden_dim)
    harmless_states: shape (N, hidden_dim) — must be same N and hidden_dim

    Returns the unnormalised direction d of shape (hidden_dim,):
      d = mean(z_harmful_i - z_harmless_i)

    Raises ValueError if arrays are empty, N differs, or hidden_dim differs.
    """
    if harmful_states.shape[0] == 0:
        raise ValueError("harmful_states must be non-empty")
    if harmless_states.shape[0] == 0:
        raise ValueError("harmless_states must be non-empty")
    if harmful_states.ndim != 2 or harmless_states.ndim != 2:
        raise ValueError(
            f"Expected 2-D arrays, got shapes {harmful_states.shape} and {harmless_states.shape}"
        )
    if harmful_states.shape[0] != harmless_states.shape[0]:
        raise ValueError(
            f"N mismatch: harmful has {harmful_states.shape[0]} rows, "
            f"harmless has {harmless_states.shape[0]} rows. "
            "Zhao et al. recipe requires paired prompts (equal N)."
        )
    if harmful_states.shape[1] != harmless_states.shape[1]:
        raise ValueError(
            f"hidden_dim mismatch: harmful has {harmful_states.shape[1]}, "
            f"harmless has {harmless_states.shape[1]}"
        )
    result: npt.NDArray[np.float64] = np.mean(
        harmful_states - harmless_states, axis=0
    )
    return result


def extract_harmfulness_direction(
    harmful_states: npt.NDArray[np.float64],
    harmless_states: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute and normalise the harmfulness direction ĥ.

    Returns ĥ of shape (hidden_dim,) with unit norm.
    Raises ValueError (via paired_difference_mean) for shape/count mismatches
    or empty arrays.
    Raises ValueError if the resulting direction has zero norm
    (degenerate case: all paired differences cancel out).
    """
    d = paired_difference_mean(harmful_states, harmless_states)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise ValueError(
            "Harmfulness direction has zero norm: mean of paired differences is zero. "
            "The harmful/harmless pairs produce identical mean hidden states."
        )
    result: npt.NDArray[np.float64] = d / norm
    return result


def orthogonality_check(
    h_hat: npt.NDArray[np.float64],
    r_hat: npt.NDArray[np.float64],
) -> float:
    """Check orthogonality of ĥ and r̂; warn if |ĥ · r̂| > threshold.

    Returns |ĥ · r̂| (the absolute cosine similarity).
    Issues UserWarning if the value exceeds ORTHOGONALITY_WARN_THRESHOLD.

    Per blueprint §12.4: high overlap between the harmfulness and refusal
    directions suggests the model conflates the two concepts, which may
    degrade PROBE-RM's ability to distinguish them independently.
    """
    overlap = float(abs(float(np.dot(h_hat, r_hat))))
    if overlap > ORTHOGONALITY_WARN_THRESHOLD:
        warnings.warn(
            f"Harmfulness direction ĥ and refusal direction r̂ are not orthogonal: "
            f"|ĥ · r̂| = {overlap:.4f} > {ORTHOGONALITY_WARN_THRESHOLD}. "
            "PROBE-RM may conflate harmfulness and refusal signals.",
            UserWarning,
            stacklevel=2,
        )
    return overlap


def collect_hidden_states_t_inst(
    adapter: HiddenStateAdapter,
    prompts: list[str],
    layer: int,
) -> npt.NDArray[np.float64]:
    """Collect residual-stream hidden states at position t_inst.

    Uses the FIRST element of adapter.get_hidden_states(prompt, layer),
    i.e. z_t_inst, per the Zhao et al. recipe.

    Returns array of shape (len(prompts), hidden_dim).

    In Phase A, adapter.get_hidden_states raises NotImplementedError.
    This function propagates that error — it is intended to be called
    only on a GPU host with real adapters.
    """
    states: list[npt.NDArray[np.float64]] = []
    for prompt in prompts:
        pair = adapter.get_hidden_states(prompt, layer)
        states.append(pair[0])
    result: npt.NDArray[np.float64] = np.stack(states, axis=0)
    return result


def calibrate_harmfulness_direction(
    adapter: HiddenStateAdapter,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    layer: int,
    r_hat: npt.NDArray[np.float64] | None = None,
    save_path: Path | None = None,
) -> npt.NDArray[np.float64]:
    """Full calibration pipeline per blueprint §12.4:
      1. collect_hidden_states_t_inst for harmful_prompts
      2. collect_hidden_states_t_inst for harmless_prompts
      3. extract_harmfulness_direction(harmful_states, harmless_states)
      4. If r_hat is provided, run orthogonality_check(h_hat, r_hat)
      5. If save_path is not None, save_direction(h_hat, save_path)
      6. Return h_hat

    harmful_prompts and harmless_prompts must be the same length (paired).

    This is the function a GPU-side runner calls to produce ĥ for
    a given (model, layer, scheme) triple.
    """
    harmful_states = collect_hidden_states_t_inst(adapter, harmful_prompts, layer)
    harmless_states = collect_hidden_states_t_inst(adapter, harmless_prompts, layer)
    h_hat = extract_harmfulness_direction(harmful_states, harmless_states)
    if r_hat is not None:
        orthogonality_check(h_hat, r_hat)
    if save_path is not None:
        from cliffguard.eval.refusal_direction import save_direction
        save_direction(h_hat, save_path)
    return h_hat
