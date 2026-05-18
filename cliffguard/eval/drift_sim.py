"""Bandit drift simulator — see blueprint §6.4, Fold D.

Simulates concept drift in the CONDUCTOR's reward signal to test
ADWIN-triggered weight reset behaviour. In a real deployment, drift
occurs when an adversary learns the bandit's current gate weights and
shifts attack strategy to exploit the weakest gate. The simulator
injects synthetic drift by:

  1. Running the CONDUCTOR for T_warm warmup steps with stable rewards.
  2. At step T_drift, reversing the reward signal for a target arm
     (simulating the adversary learning and exploiting that gate).
  3. Monitoring the ADWIN statistic on the reward stream.
  4. Measuring latency (steps) from T_drift until ADWIN fires and
     reset_weights() is called.
  5. Measuring ABR recovery: steps from reset until ABR returns to
     within epsilon of pre-drift level.

Per blueprint §6.4: ADWIN window uses Page-Hinkley variant with
delta=0.002. The simulator uses a simplified threshold-based ADWIN
proxy (no external ADWIN library required in Phase A).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from cliffguard.conductor.bandit import ARMS, Conductor
from cliffguard.types import Tier  # noqa: F401

ADWIN_DELTA: float = 0.002  # blueprint §6.4


def true_adwin_statistic(
    reward_stream: "npt.NDArray[np.float64]",
    delta: float = ADWIN_DELTA,
    max_buckets: int = 5,
) -> tuple[bool, int]:
    """True ADWIN (ADaptive WINdowing) per Bifet & Gavaldà (2007).

    Maintains a window W partitioned into exponential-histogram buckets.
    At each new element, scans every possible split W = W_0 · W_1 and
    declares change if |mean(W_0) - mean(W_1)| > epsilon_cut, where
      m       = 1 / (1/|W_0| + 1/|W_1|)        (harmonic mean of sub-sizes)
      delta'  = delta / |W|                    (Bonferroni over splits)
      epsilon_cut = sqrt( (1 / (2*m)) * ln(4 / delta') )

    On detection, drops the older partition W_0 and returns the index in
    the input stream where the change was localised. Otherwise returns
    (False, len(reward_stream)).

    This implementation uses a simplified exponential-histogram with at
    most max_buckets buckets per power-of-two size class. For research-grade
    use, replace with the `river.drift.ADWIN` implementation from the
    river streaming-ML library (drop-in compatible signature).

    Distinct from the simpler Page-Hinkley `adwin_statistic` above:
    Page-Hinkley detects sustained mean shifts via cumulative-sum monitoring;
    true ADWIN explicitly adapts the comparison window size to the data and
    has tighter (rigorous) false-positive guarantees.

    Raises ValueError if reward_stream is empty or delta not in (0, 1).
    """
    import math

    if reward_stream.size == 0:
        raise ValueError("reward_stream must not be empty")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    n = len(reward_stream)
    # window: list of (sum, count) buckets, oldest first.
    window: list[tuple[float, int]] = []

    def total() -> tuple[float, int]:
        s = sum(b[0] for b in window)
        c = sum(b[1] for b in window)
        return s, c

    for i in range(n):
        x = float(reward_stream[i])
        window.append((x, 1))

        # Compress: merge adjacent buckets of equal count when exceeding max_buckets.
        # Keep this simple and approximate (real ADWIN uses exponential histograms).
        while True:
            merged = False
            counts = [b[1] for b in window]
            for j in range(len(window) - 1):
                if counts[j] == counts[j + 1]:
                    # Tentatively merge.
                    same_size_run = [b for b in window if b[1] == counts[j]]
                    if len(same_size_run) > max_buckets:
                        s_merge = window[j][0] + window[j + 1][0]
                        c_merge = window[j][1] + window[j + 1][1]
                        window = window[:j] + [(s_merge, c_merge)] + window[j + 2:]
                        merged = True
                        break
            if not merged:
                break

        s_total, c_total = total()
        if c_total < 4:
            continue

        # Scan all splits W = W_0 (older) || W_1 (newer).
        prefix_sum = 0.0
        prefix_count = 0
        delta_prime = delta / c_total
        detected_split = -1
        for j in range(len(window) - 1):
            prefix_sum += window[j][0]
            prefix_count += window[j][1]
            suffix_count = c_total - prefix_count
            if prefix_count < 1 or suffix_count < 1:
                continue
            mean_0 = prefix_sum / prefix_count
            mean_1 = (s_total - prefix_sum) / suffix_count
            m_harm = 1.0 / (1.0 / prefix_count + 1.0 / suffix_count)
            epsilon_cut = math.sqrt((1.0 / (2.0 * m_harm)) * math.log(4.0 / delta_prime))
            if abs(mean_0 - mean_1) > epsilon_cut:
                detected_split = j
                break

        if detected_split >= 0:
            # Localise the change to the boundary between W_0 and W_1.
            # Convert bucket-split back to stream index: count of older buckets
            # = number of samples seen up to and including the boundary.
            older_samples = sum(b[1] for b in window[: detected_split + 1])
            change_idx = (i + 1) - (c_total - older_samples)
            # Drop older partition for future steps.
            window = window[detected_split + 1:]
            return (True, max(0, change_idx))

    return (False, n)


def adwin_statistic(
    reward_stream: npt.NDArray[np.float64],
    delta: float = ADWIN_DELTA,
) -> tuple[bool, int]:
    """Simplified Page-Hinkley ADWIN proxy.
    Scans reward_stream for a mean shift exceeding delta.
    Returns (drift_detected: bool, drift_index: int).
    drift_index is the index where drift is first detected,
    or len(reward_stream) if no drift detected.
    Raises ValueError if reward_stream is empty."""
    if reward_stream.size == 0:
        raise ValueError("reward_stream must not be empty")

    n = len(reward_stream)
    lambda_thresh = -math.log(delta)

    # Page-Hinkley with running-mean correction (detects downward mean shift).
    # cumsum accumulates (x_i - running_mean - delta) at each step.
    # When cumsum drops far below its historical maximum (ph = max - cumsum),
    # a sustained downward shift has been detected.
    sum_x = 0.0
    cumsum = 0.0
    max_cumsum = 0.0

    for i in range(n):
        x = float(reward_stream[i])
        sum_x += x
        running_mean = sum_x / (i + 1)
        cumsum += x - running_mean - delta
        if cumsum > max_cumsum:
            max_cumsum = cumsum
        ph = max_cumsum - cumsum
        if ph > lambda_thresh:
            return (True, i)

    return (False, n)


def run_drift_simulation(
    conductor: Conductor,
    context_dim: int,
    target_arm: str,
    T_warm: int = 100,
    T_drift: int = 100,
    T_recover: int = 100,
    seed: int = 42,
) -> dict[str, object]:
    """Run the full drift simulation.

    Phase 1 (warmup, T_warm steps):
      - Generate random context vectors of shape (context_dim,).
      - Reward target_arm with +1.0, all others with 0.0.
      - Update conductor for target_arm each step.

    Phase 2 (drift, T_drift steps):
      - Reverse: reward target_arm with -1.0 (adversary exploiting it).
      - Collect reward stream for ADWIN monitoring.
      - Call adwin_statistic on the reward stream.
      - If drift detected: call conductor.reset_weights() and record
        reset_step (step within Phase 2 when reset occurred).

    Phase 3 (recovery, T_recover steps):
      - Resume positive rewards for target_arm.
      - Measure how many steps until mean reward over a 10-step window
        returns to >= 0.5 (recovery threshold).

    Returns dict with keys:
      drift_detected: bool
      reset_step: int | None  (step in phase 2 when reset fired)
      recovery_steps: int | None  (steps in phase 3 to recover)
      warmup_mean_reward: float
      drift_mean_reward: float
      recovery_mean_reward: float

    Raises ValueError if target_arm not in ARMS.
    Raises ValueError if T_warm, T_drift, or T_recover < 1."""
    if target_arm not in ARMS:
        raise ValueError(
            f"Unknown target_arm: {target_arm!r}. Valid arms: {ARMS}"
        )
    if T_warm < 1:
        raise ValueError(f"T_warm must be >= 1, got {T_warm}")
    if T_drift < 1:
        raise ValueError(f"T_drift must be >= 1, got {T_drift}")
    if T_recover < 1:
        raise ValueError(f"T_recover must be >= 1, got {T_recover}")

    rng = np.random.default_rng(seed)

    # Phase 1: Warmup — stable positive rewards for target_arm.
    warmup_rewards: list[float] = []
    for _ in range(T_warm):
        ctx = rng.standard_normal(context_dim).astype(np.float64)
        conductor.update(target_arm, ctx, 1.0)
        warmup_rewards.append(1.0)

    # Phase 2: Drift — reversed reward (adversary exploiting target arm).
    drift_rewards: list[float] = []
    for _ in range(T_drift):
        ctx = rng.standard_normal(context_dim).astype(np.float64)
        conductor.update(target_arm, ctx, -1.0)
        drift_rewards.append(-1.0)

    # ADWIN monitors the continuous stream (warmup + drift).
    # A stable warmup followed by a sudden reversal gives a clear signal.
    combined = np.array(warmup_rewards + drift_rewards, dtype=np.float64)
    drift_detected, adwin_index = adwin_statistic(combined)

    reset_step: int | None = None
    if drift_detected:
        # Convert combined-stream index to a step index within Phase 2.
        reset_step = max(0, adwin_index - T_warm)
        conductor.reset_weights()

    # Phase 3: Recovery — resume positive rewards; measure steps to recover.
    recovery_rewards: list[float] = []
    recovery_steps_val: int | None = None
    for step in range(T_recover):
        ctx = rng.standard_normal(context_dim).astype(np.float64)
        conductor.update(target_arm, ctx, 1.0)
        recovery_rewards.append(1.0)
        if recovery_steps_val is None and len(recovery_rewards) >= 10:
            window_mean = float(np.mean(recovery_rewards[-10:]))
            if window_mean >= 0.5:
                recovery_steps_val = step + 1

    # Recovery is only meaningful when a reset was triggered.
    if not drift_detected:
        recovery_steps_val = None

    return {
        "drift_detected": drift_detected,
        "reset_step": reset_step,
        "recovery_steps": recovery_steps_val,
        "warmup_mean_reward": float(np.mean(warmup_rewards)),
        "drift_mean_reward": float(np.mean(drift_rewards)),
        "recovery_mean_reward": float(np.mean(recovery_rewards)),
    }
