"""Refusal-direction extractor — see blueprint §5.1, §12.3.

Implements the Arditi et al. (arXiv:2406.11717) recipe for extracting
the refusal direction r̂ from a model's residual stream:

  1. Collect hidden states z_l(t_post-inst) at layer l for a set of
     harmful prompts (which the model refuses) and a set of harmless
     prompts (which the model complies with).
  2. Compute the difference-in-means direction:
       r = mean(z_harmful) - mean(z_harmless)
  3. Normalise: r̂ = r / ||r||

The result r̂ is used by PROBE-RM as the refusal direction.

In scaffolding mode (Phase A), operates on synthetic hidden-state
arrays. Phase B provides real hidden states from the engine adapters.

Per blueprint §12.3 (Fold A): the refusal direction is extracted on
the calibration corpus before any evaluation fold is unblinded.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt


def difference_in_means(
    harmful_states: npt.NDArray[np.float64],
    harmless_states: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the difference-in-means refusal direction.
    harmful_states:  shape (N_harm, hidden_dim)
    harmless_states: shape (N_safe, hidden_dim)
    Returns the unnormalised direction r of shape (hidden_dim,).
    Raises ValueError if arrays have different hidden_dim,
    or if either array is empty."""
    if harmful_states.shape[0] == 0:
        raise ValueError("harmful_states must be non-empty")
    if harmless_states.shape[0] == 0:
        raise ValueError("harmless_states must be non-empty")
    if harmful_states.shape[1] != harmless_states.shape[1]:
        raise ValueError(
            f"hidden_dim mismatch: harmful has {harmful_states.shape[1]}, "
            f"harmless has {harmless_states.shape[1]}"
        )
    result: npt.NDArray[np.float64] = (
        np.mean(harmful_states, axis=0) - np.mean(harmless_states, axis=0)
    )
    return result


def extract_refusal_direction(
    harmful_states: npt.NDArray[np.float64],
    harmless_states: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute and normalise the refusal direction r̂.
    Returns r̂ of shape (hidden_dim,) with unit norm.
    Raises ValueError (via difference_in_means) for shape mismatch
    or empty arrays.
    Raises ValueError if the resulting direction has zero norm
    (degenerate case: mean harmful == mean harmless)."""
    r = difference_in_means(harmful_states, harmless_states)
    norm = float(np.linalg.norm(r))
    if norm == 0.0:
        raise ValueError(
            "Refusal direction has zero norm: mean(harmful) == mean(harmless). "
            "The two prompt sets produce identical mean hidden states."
        )
    result: npt.NDArray[np.float64] = r / norm
    return result


def save_direction(
    direction: npt.NDArray[np.float64],
    path: Path,
) -> None:
    """Save the refusal direction to a .npy file at path.
    Creates parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, direction)


def load_direction(path: Path) -> npt.NDArray[np.float64]:
    """Load a refusal direction from a .npy file.
    Raises FileNotFoundError if path does not exist.
    Raises ValueError if the loaded array is not 1-D."""
    if not path.exists():
        raise FileNotFoundError(f"Refusal direction file not found: {path}")
    arr: npt.NDArray[np.float64] = np.load(path).astype(np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"Expected a 1-D refusal direction array, got shape {arr.shape}"
        )
    return arr
