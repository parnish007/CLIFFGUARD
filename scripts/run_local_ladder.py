"""Stages 1-4 on local hardware: a matched quantization ladder, eta, and prediction.

Runs the full behavioural-rate-distortion measurement on a 6 GB laptop GPU.

Two ladder sources, selected with --ladder-kind:

  rtn   (default) Round-to-nearest, per-group asymmetric, applied in-process to
        the cached FP16 checkpoint at bit-widths 8..2. No downloads. This is the
        CLEANEST possible realization of assumption A3: one checkpoint, one
        quantizer family, one fixed group size, and bit-width as the only thing
        that varies. It is a genuine ordinal dose axis, which is exactly what
        NF4/AWQ/GPTQ are not. Both W_fp16 and W_q are in memory, so eta is
        measured from the true weight perturbation rather than inferred from a
        file format's payload accounting.

  gguf  llama.cpp k-quants from a prebuilt matched ladder. Deployment-realistic
        but the rungs differ in block structure as well as bit-width, and eta's
        bit regressor comes from tensor payload bytes. Requires the GGUF files.

Both paths load each rung as a dense fp16 torch model, so peak VRAM is one copy
of the model (~3.1 GB for Qwen2.5-1.5B) regardless of ladder length. That is what
makes Stage 4 possible at all: without dense loading there is no measured d' at
the low-precision rungs to compare a prediction against.

What is measured, in order:

  S0  rotation replication per scheme                        -> claim C1
  S1  isotropy of the direction perturbation per scheme      -> claim C5
  S2  eta from WEIGHTS along the ladder, exponent FITTED     -> A3, C2
  S3  held-out d' per scheme; eta(weights) vs eta(d' decay)  -> C2, falsifier F5
  S4  fit eta_4 on the high-precision rungs, PREDICT d' on
      the low-precision rungs with no refitting              -> claim C3

Everything lands in an immutable artifacts/runs/<stamp>_<sha>_<label>/ directory.

Usage:
  python scripts/run_local_ladder.py                        # RTN ladder, n=250
  python scripts/run_local_ladder.py --n 24 --smoke         # fast wiring check
  python scripts/run_local_ladder.py --ladder-kind gguf     # k-quant ladder
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliffguard.eval.composition import collapse_bits_threshold_closed_form, d_prime_at_bits
from cliffguard.eval.discriminability import (
    d_prime_with_ci,
    gaussianity_gap,
    held_out_d_prime,
    implied_eta,
)
from cliffguard.eval.isotropy import isotropy_test
from cliffguard.eval.noise_floor import (
    angle_between,
    difference_in_means,
    rotation_replication,
)
from cliffguard.eval.noise_spectrum import (
    EtaMeasurement,
    fit_eta_vs_bits_report,
    measure_gguf_pair,
)
from cliffguard.eval.storage import new_run, record_corpus, record_environment

# Mutable so the same code path serves any checkpoint; main() sets them from argv.
# Keeping them module-level (rather than threading a config object through every
# loader) keeps the loader signatures small enough to read at a glance.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_STEM = "qwen2.5-1.5b-instruct"
GGUF_LADDER = ["fp16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]
RTN_BITS = [8, 7, 6, 5, 4, 3, 2]

# How the checkpoint is placed. None means one device, via .to("cuda"), which is
# what every result so far used and what fits models up to about 32B on an 80 GB
# card. "auto" shards across visible GPUs and is required above that: a 70B
# checkpoint is 141 GB of fp16 weights before the quantization transients, so it
# does not fit any single device sold today. Set by --device-map on the runners.
DEVICE_MAP: Any = None

FloatArray = Any


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def load_prompts(n: int) -> tuple[list[str], list[str]]:
    """Load the real Fold A corpus. Refuses to invent placeholder prompts.

    Two dedupe rules, both load-bearing. Repeats within a class fake precision:
    the same activation counted twice narrows every interval without adding
    information. A prompt present in BOTH classes is a direct label
    contradiction -- hh-rlhf reuses human turns across rows, so it happens --
    and it deflates d' for reasons unrelated to quantization.
    """
    base = Path("data/folds/fold_a")
    harmful_f, benign_f = base / "anthropic_hh_refused.jsonl", base / "anthropic_hh_benign.jsonl"
    if not (harmful_f.exists() and benign_f.exists()):
        raise SystemExit(
            f"Fold A corpus not found under {base}/.\n"
            "Run: python scripts/download_fold_a.py --download\n"
            "Refusing to substitute synthetic prompts -- the result would be meaningless."
        )

    def read(path: Path, want: int, exclude: frozenset[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prompt = json.loads(line)["prompt"].strip()
            if prompt and prompt not in seen and prompt not in exclude:
                seen.add(prompt)
                out.append(prompt)
            if len(out) >= want:
                break
        return out

    harmful = read(harmful_f, n, frozenset())
    benign = read(benign_f, n, frozenset(harmful))
    if len(harmful) < 4 or len(benign) < 4:
        raise SystemExit(f"corpus too small: {len(harmful)} refused / {len(benign)} benign")
    if len(harmful) < n or len(benign) < n:
        print(f"WARNING: only {len(harmful)} refused / {len(benign)} benign unique prompts "
              f"available (asked for {n} each)")
    return harmful, benign


def load_labelled_prompts(
    path: Path, n_per_class: int | None = None
) -> tuple[list[str], list[str], list[str]]:
    """A prompt corpus that knows which of its prompts are harmful.

    Returns (prompts, sources, harm_labels), aligned. `harm_labels` is what the
    Fold A corpus cannot supply and what the 2x2 needs: on a harmful prompt a
    lost refusal is a safety regression, and on a benign prompt a gained refusal
    is an over-refusal. Without the label those two are the same number.

    `n_per_class` counts PER HARM CLASS, matching what `--n` means on the Fold A
    path, where `load_prompts(250)` yields 250 harmful and 250 benign. The two
    routes previously disagreed: the same `--n 250` produced 500 prompts through
    Fold A and 250 through a labelled suite, so a run described in a manifest as
    n=250 could be either size depending on a flag elsewhere. Nothing crashed,
    which is what made it worth fixing.

    The two classes are INTERLEAVED after each is capped, and that is not
    cosmetic either. These suites are written class-ordered -- XSTest lists its
    250 safe prompts before its 200 unsafe ones -- so taking the first n of the
    file gives a single-class corpus for any n below 250: not a crash, but every
    over-refusal count computed on prompts of which none was harmful. Round-robin
    keeps each class's own order, keeps the result deterministic, and keeps the
    prefix property that makes two runs at different n comparable.

    A single-class suite (AdvBench is all harmful) interleaves to itself, so
    nothing is reordered where there is nothing to balance, and `n_per_class`
    caps the one class it has.

    Prompts are deduplicated because a repeat inside one run is the same prompt
    counted twice, which narrows every interval without adding information.

    Built by scripts/download_eval_suites.py.
    """
    import itertools
    if not path.exists():
        raise SystemExit(
            f"no labelled corpus at {path}.\n"
            "Run: python scripts/download_eval_suites.py --download")

    by_class: dict[str, list[tuple[str, str, str]]] = {"harmful": [], "benign": []}
    seen: set[str] = set()
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = str(row.get("prompt", "")).strip()
        label = row.get("harm_label")
        if not prompt or prompt in seen:
            continue
        if label not in ("harmful", "benign"):
            raise SystemExit(
                f"{path}:{line_number} has harm_label={label!r}. Every prompt "
                "must be labelled harmful or benign; an unlabelled corpus is the "
                "one this path exists to replace.")
        seen.add(prompt)
        by_class[label].append((prompt, str(row.get("source", path.stem)), label))

    if n_per_class is not None:
        for label in by_class:
            by_class[label] = by_class[label][:n_per_class]
    interleaved = [
        row
        for pair in itertools.zip_longest(by_class["harmful"], by_class["benign"])
        for row in pair
        if row is not None
    ]

    if len(interleaved) < 4:
        raise SystemExit(f"{path}: only {len(interleaved)} usable prompts")
    prompts = [r[0] for r in interleaved]
    sources = [r[1] for r in interleaved]
    labels = [r[2] for r in interleaved]
    return prompts, sources, labels


# ---------------------------------------------------------------------------
# RTN quantization
# ---------------------------------------------------------------------------


def rtn_quantize_dequantize(weight: Any, bits: int, group: int) -> Any:
    """Per-group asymmetric round-to-nearest, applied along the INPUT axis.

    This is the standard PTQ baseline the AWQ and GPTQ papers call RTN: split
    each row into contiguous groups of `group` input channels, fit an affine
    (scale, zero) per group from that group's min/max, round to `bits` levels,
    and map back. Nothing is salience-aware, which is precisely the property
    claim C5 predicts produces an isotropic perturbation.

    Returns a tensor of the same shape and dtype, holding the dequantized values.
    """
    import torch

    out_features, in_features = weight.shape
    original_dtype = weight.dtype
    work = weight.to(torch.float32)

    pad = (-in_features) % group
    if pad:
        work = torch.nn.functional.pad(work, (0, pad))
    grouped = work.reshape(out_features, -1, group)

    # The padded positions must NOT participate in the group's min/max. Zero
    # padding widens the final group's range whenever the real values do not
    # straddle zero, which coarsens the step for every real weight in that group.
    # Measured on a tail whose true range was [5, 6]: mean absolute error 0.0156
    # when quantized on its own range, 0.0999 when zero-padded into a full group
    # -- a 6.4x inflation applied silently to the last group of every matrix.
    if pad:
        valid = torch.ones_like(work, dtype=torch.bool)
        valid[:, in_features:] = False
        valid_groups = valid.reshape(out_features, -1, group)
        lo = torch.where(valid_groups, grouped, torch.inf).amin(dim=-1, keepdim=True)
        hi = torch.where(valid_groups, grouped, -torch.inf).amax(dim=-1, keepdim=True)
    else:
        lo = grouped.amin(dim=-1, keepdim=True)
        hi = grouped.amax(dim=-1, keepdim=True)
    levels = float(2**bits - 1)
    # Degenerate (constant) groups get scale 1 so that q rounds to 0 and the
    # group reconstructs exactly as `lo`, rather than dividing by zero.
    scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
    codes = torch.clamp(torch.round((grouped - lo) / scale), 0.0, levels)
    restored = codes * scale + lo

    restored = restored.reshape(out_features, -1)
    if pad:
        restored = restored[:, :in_features]
    return restored.to(original_dtype)


def rtn_bits_per_parameter(bits: int, group: int) -> float:
    """Exact stored bits per weight: the code plus its share of a group's
    fp16 scale and fp16 zero point. No file-format estimation involved."""
    return float(bits) + 32.0 / float(group)


def salience_scaled_quantize_dequantize(
    weight: Any, bits: int, group: int, channel_scale: Any, alpha: float
) -> Any:
    """RTN preceded by AWQ-style activation-aware per-input-channel scaling.

    This exists to make claim C5 a CONTROLLED experiment rather than a comparison
    across published checkpoints. Setting our own RTN against someone else's AWQ
    build confounds salience-awareness with group size, calibration set, kernel,
    and release. Here the checkpoint, the group size, the bit budget, the corpus,
    and the code path are all identical, and the ONLY difference is whether the
    quantizer protects high-activation input channels.

    The transform is AWQ's: scale each input channel j by s_j = mean|x_j|^alpha,
    quantize the scaled weight, then divide the scale back out. Because the
    inverse fold is exact, this is a pure weight transform -- no runtime change,
    no custom kernel -- and the stored bit budget is unchanged, since s is
    per-channel metadata of the same order as the existing per-group scales.

    alpha=0 reduces exactly to plain RTN, which is the control.
    """
    import torch

    if alpha == 0.0:
        return rtn_quantize_dequantize(weight, bits, group)
    scale = torch.clamp(channel_scale.to(weight.device, torch.float32), min=1e-5) ** alpha
    scale = scale / scale.mean()          # keep the weight magnitude comparable to the control
    scaled = weight.to(torch.float32) * scale.unsqueeze(0)
    restored = rtn_quantize_dequantize(scaled.to(weight.dtype), bits, group)
    return (restored.to(torch.float32) / scale.unsqueeze(0)).to(weight.dtype)


def collect_channel_scales(model: Any, prompts: list[str], tokenizer: Any) -> dict[str, Any]:
    """mean |x_j| per input channel for every block Linear, over benign prompts.

    Benign prompts only: this is calibration data, and using harmful prompts here
    would leak behavioural supervision into what is meant to be a weights-only
    quantizer.
    """
    import torch

    sums: dict[str, Any] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str) -> Any:
        def hook(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            x = inputs[0].detach()
            flat = x.reshape(-1, x.shape[-1]).abs().sum(dim=0).float()
            if name in sums:
                sums[name] += flat
                counts[name] += x.reshape(-1, x.shape[-1]).shape[0]
            else:
                sums[name] = flat
                counts[name] = x.reshape(-1, x.shape[-1]).shape[0]

        return hook

    for name, module in target_linears(model):
        handles.append(module.register_forward_hook(make_hook(f"{name}.weight")))

    with torch.no_grad():
        for prompt in prompts:
            out = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            batch = (
                {k: v.to("cuda") for k, v in out.items() if k in ("input_ids", "attention_mask")}
                if hasattr(out, "keys")
                else {"input_ids": out.to("cuda")}
            )
            model(**batch)

    for handle in handles:
        handle.remove()
    return {name: sums[name] / max(counts[name], 1) for name in sums}


def target_linears(model: Any) -> list[tuple[str, Any]]:
    """Transformer-block Linear layers only.

    Embeddings and the output head are left at FP16, matching what every
    practical scheme does (GGUF keeps token_embd and output at high precision;
    AWQ and GPTQ skip lm_head). Quantizing them would confound the ladder with
    a vocabulary effect that has nothing to do with the residual-stream claim.
    """
    import torch

    picked: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "lm_head" in name or "embed" in name:
            continue
        picked.append((name, module))
    return picked


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------


def host_memory() -> tuple[float, float]:
    """(resident GB for this process, available GB on the machine).

    Read from /proc, so it is Linux-only and returns (nan, nan) elsewhere. That
    is the platform the long ladders run on; on a laptop the numbers would only
    be decoration.
    """
    def _read(path: str, key: str) -> float:
        # Everything here is caught, including a malformed line. This function
        # only reports; a kernel that lays /proc out differently must not be
        # able to end a run that is otherwise working.
        try:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.startswith(key):
                    return float(line.split()[1]) / 1e6      # kB -> GB
        except (OSError, ValueError, IndexError, UnicodeDecodeError):
            pass
        return float("nan")

    return _read("/proc/self/status", "VmRSS:"), _read("/proc/meminfo", "MemAvailable:")


def release_host_memory(tag: str = "") -> None:
    """Give freed heap back to the OS between model loads, and say what is left.

    A ladder loads and discards one model per rung. `del model` followed by
    `torch.cuda.empty_cache()` returns the GPU allocation, but the host side has
    a different allocator: glibc keeps freed arenas mapped for reuse rather than
    handing them back, so resident memory ratchets upward across rungs even
    though nothing is leaked in the Python sense. On a hosted runtime with a
    fixed memory ceiling, one of the later loads is the one that crosses it, and
    the process is killed with SIGKILL -- no traceback, no exception, exit code
    -9. That is how the 1.5B GSM8K ladder died after six of eight schemes.

    `malloc_trim(0)` asks glibc for the arenas back. It is guarded rather than
    assumed because the symbol does not exist on Windows or under musl, and a
    missing allocator detail must not end a run.

    The memory figures are printed because the next failure should be
    diagnosable from the log alone: a run that climbs half a gigabyte per rung
    is a different problem from one that sits flat and then dies.
    """
    import ctypes

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                  # noqa: BLE001
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass                                           # not glibc; nothing to trim
    rss, available = host_memory()
    if rss == rss:                                     # NaN check without numpy
        print(f"[mem{' ' + tag if tag else ''}] host RSS {rss:.2f} GB, "
              f"available {available:.2f} GB", flush=True)


def _quarantine(path: Path, reason: str) -> None:
    """Move an unusable cache file aside so the next read misses instead of dying."""
    spoiled = path.with_suffix(path.suffix + ".corrupt")
    try:
        path.replace(spoiled)
        print(f"[cache] {path.name} is unreadable ({reason}); moved to "
              f"{spoiled.name} and will be regenerated", flush=True)
    except OSError as exc:                             # noqa: BLE001
        print(f"[cache] {path.name} is unreadable ({reason}) and could not be "
              f"moved aside ({exc}); regenerating anyway", flush=True)


def read_json_cache(path: Path) -> Any | None:
    """Cached JSON, or None when the file cannot be trusted.

    Caches are written to Drive as each scheme finishes, which is what makes an
    interrupted session resumable -- and it is also what makes a half-written
    file possible, since a disconnect can land in the middle of a write.

    `json.loads` on a truncated file raises, and the retry logic re-reads the
    same cache on every attempt, so one bad file would turn a resumable run into
    three identical failures followed by a give-up. Treating an unreadable cache
    as a missing one costs the time to regenerate that one scheme and nothing
    else, which is the cheaper mistake by a wide margin.

    The file is moved aside rather than deleted, so what went wrong is still
    there to look at.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        _quarantine(path, f"{type(exc).__name__}: {exc}")
        return None


def read_npy_cache(path: Path) -> FloatArray | None:
    """Cached array, or None when the file cannot be trusted. See read_json_cache."""
    if not path.exists():
        return None
    try:
        return np.load(path)
    except (ValueError, OSError, EOFError) as exc:
        _quarantine(path, f"{type(exc).__name__}: {exc}")
        return None


def gguf_filename(quant: str) -> str:
    return f"{GGUF_STEM}-{quant}.gguf"


def load_fp16_model() -> Any:
    """from_pretrained with the transformers 4.x/5.x dtype kwarg difference absorbed.

    With DEVICE_MAP set the model is sharded at load time and NOT moved
    afterwards: calling .to() on a sharded model either fails or silently
    collapses it onto one device, which is the opposite of what was asked for.
    """
    import torch
    from transformers import AutoModelForCausalLM

    extra = {} if DEVICE_MAP is None else {"device_map": DEVICE_MAP}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True, **extra
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True, **extra
        )
    return model if DEVICE_MAP is not None else model.to("cuda")


def load_deployed_model(repo: str) -> Any:
    """Load a checkpoint that is ALREADY quantized by a deployed method.

    AWQ and GPTQ checkpoints carry their quantization in the weights, so they are
    loaded rather than constructed. transformers dispatches to the right kernel
    from the repo's quantization_config; the only thing this wrapper adds is the
    dtype-kwarg difference between transformers 4.x and 5.x, and a device map,
    because these kernels expect their weights already on the GPU.
    """
    import torch
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.float16, device_map={"": 0},
            low_cpu_mem_usage=True
        )


def deployed_quantization_config(repo: str) -> dict[str, Any]:
    """What a pre-quantized checkpoint says about its own quantization.

    Recorded so a deployed scheme is described by the checkpoint rather than by
    its label. "AWQ_4B" is a name we chose; `bits=4, group_size=128` is what the
    file asserts, and the two can disagree.

    The stored bits per parameter is reported on the same accounting the RTN
    rungs use -- code bits plus the per-group fp16 scale and zero -- so the two
    families are at least measured the same way. It is NOT a position on the RTN
    ordinal axis: that axis holds one quantizer family with bit-width as the
    only thing varying, and AWQ changes the algorithm as well. Analysis keeps
    these schemes off the ladder fit; this figure exists to be quoted in a
    table, not regressed on.
    """
    from transformers import AutoConfig

    try:
        raw = getattr(AutoConfig.from_pretrained(repo), "quantization_config", None)
    except Exception as exc:                           # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    if raw is None:
        return {"error": "checkpoint declares no quantization_config"}
    cfg = dict(raw) if isinstance(raw, dict) else dict(getattr(raw, "__dict__", {}) or {})
    out: dict[str, Any] = {
        k: v for k, v in cfg.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }
    bits = out.get("bits") or out.get("w_bit")
    group = out.get("group_size") or out.get("q_group_size")
    if isinstance(bits, int) and isinstance(group, int) and group > 0:
        out["bits_per_param_stored"] = float(bits) + 32.0 / float(group)
    return out


def load_nf4_model() -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # T4/Ampere-laptop safe; matches earlier Stage 0
        bnb_4bit_use_double_quant=False,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float16, quantization_config=cfg, device_map={"": 0}
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, quantization_config=cfg, device_map={"": 0}
        )


def load_gguf_model(quant: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            GGUF_REPO, gguf_file=gguf_filename(quant), dtype=torch.float16, low_cpu_mem_usage=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            GGUF_REPO,
            gguf_file=gguf_filename(quant),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    return model.to("cuda")


def load_rtn_model(
    bits: int,
    group: int,
    eta_weight_names: list[str],
    channel_scales: dict[str, Any] | None = None,
    alpha: float = 0.0,
    direction: FloatArray | None = None,
) -> tuple[Any, dict[str, float]]:
    """FP16 checkpoint with every block Linear replaced by its RTN reconstruction.

    With `alpha > 0` and `channel_scales` supplied, the reconstruction is
    salience-aware (AWQ-style) at the same bit budget -- the controlled contrast
    for claim C5.

    Returns the model and, when `direction` is supplied, one scalar per named
    weight: the projected perturbation variance sigma^2 that eta is built from.

    The scalar matters. An earlier version returned the (fp16, quantized) matrix
    PAIR as float64 and held every pair until activation collection finished. For
    a 3B checkpoint at mid-depth that is roughly 6-7 GB of host RAM on top of the
    interpreter, tokenizer, model load, and downloads -- enough to have the kernel
    killed on a standard hosted runtime even when GPU memory is fine. Each
    variance is now computed while the two tensors are still in scope and only the
    scalar survives.
    """
    import torch

    from cliffguard.eval.noise_spectrum import projected_perturbation_variance

    model = load_fp16_model()
    sigma: dict[str, float] = {}
    wanted = set(eta_weight_names)
    original = quantized = None      # bound even if the model has no target Linears
    with torch.no_grad():
        for name, module in target_linears(model):
            weight_name = f"{name}.weight"
            original = module.weight.data
            scale = None if channel_scales is None else channel_scales.get(weight_name)
            if alpha > 0.0 and scale is not None:
                quantized = salience_scaled_quantize_dequantize(
                    original, bits, group, scale, alpha
                )
            else:
                quantized = rtn_quantize_dequantize(original, bits, group)
            if weight_name in wanted and direction is not None:
                fp = original.detach().to(torch.float32).cpu().numpy().astype(np.float64)
                qz = quantized.detach().to(torch.float32).cpu().numpy().astype(np.float64)
                sigma[weight_name] = projected_perturbation_variance(fp, qz, direction)
                del fp, qz
            module.weight.data = quantized
        # Both names still hold the last matrix once the loop ends, and
        # `original` holds a tensor nothing else references now that the module
        # points at `quantized`. One matrix is not the OOM, but it is free to
        # drop and it keeps the post-quantization memory reading honest.
        del original, quantized
    return model, sigma


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------


def collect(
    scheme: str,
    loader: Callable[[], Any],
    harmful: list[str],
    benign: list[str],
    layer: int,
    cache_dir: Path,
) -> tuple[FloatArray, FloatArray]:
    """Residual stream at `layer`, last token, one forward per prompt.

    The cache key MUST include the class. Without it the benign pass silently
    reuses the harmful activations and difference-in-means comes out as the
    zero vector.
    """
    import torch
    from transformers import AutoTokenizer

    keys = {
        cls: cache_dir / f"acts_{scheme}_{cls}_L{layer}_{len(harmful)}.npy" for cls in ("h", "l")
    }
    if all(p.exists() for p in keys.values()):
        print(f"[{scheme}] cache hit", flush=True)
        return np.load(keys["h"]), np.load(keys["l"])

    # One tokenizer for every scheme. Identical tokenization is a precondition of
    # the paired design; a GGUF-embedded tokenizer differing by one token would
    # break row alignment invisibly.
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    started = time.time()
    print(f"[{scheme}] loading ...", flush=True)
    model = loader()
    model.eval()
    on_cuda = sum(p.numel() for p in model.parameters() if p.is_cuda)
    print(
        f"[{scheme}] loaded in {time.time() - started:.0f}s | {on_cuda / 1e9:.3f} B params on cuda"
        f" | {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated",
        flush=True,
    )

    def encode(prompt: str) -> dict[str, Any]:
        out = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(out, "keys"):        # BatchEncoding on newer transformers
            return {k: v.to("cuda") for k, v in out.items() if k in ("input_ids", "attention_mask")}
        return {"input_ids": out.to("cuda")}

    result: dict[str, FloatArray] = {}
    with torch.no_grad():
        for cls, prompts in (("h", harmful), ("l", benign)):
            rows, tick = [], time.time()
            for i, prompt in enumerate(prompts):
                hidden = model(**encode(prompt), output_hidden_states=True).hidden_states
                rows.append(hidden[layer][0, -1, :].float().cpu().numpy())
                if (i + 1) % 100 == 0:
                    print(
                        f"   [{scheme}/{cls}] {i + 1}/{len(prompts)}"
                        f"  {(time.time() - tick) / (i + 1) * 1000:.0f} ms/prompt",
                        flush=True,
                    )
            arr = np.stack(rows).astype(np.float64)
            np.save(keys[cls], arr)
            result[cls] = arr

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{scheme}] done in {time.time() - started:.0f}s  shape={result['h'].shape}", flush=True)
    return result["h"], result["l"]


def collect_rtn(
    bits: int,
    group: int,
    eta_weight_names: list[str],
    harmful: list[str],
    benign: list[str],
    layer: int,
    cache_dir: Path,
    channel_scales: dict[str, Any] | None = None,
    alpha: float = 0.0,
    scheme_prefix: str = "RTN",
    direction: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, dict[str, float]]:
    """Activations plus per-matrix sigma^2 for one RTN rung, in one model load.

    sigma^2 is captured here rather than in a second pass because it does not
    depend on s^2 (eta = sigma^2 / s^2, and s^2 is only known once the FP16
    direction exists). Re-quantizing a second time purely to measure weights
    would double the run time for no gain.
    """
    scheme = f"{scheme_prefix}_{bits}B"
    # n is part of the key because sigma^2 is measured against the FP16 direction,
    # and that direction is estimated from these n prompts. A cache hit across
    # different n would silently pair one ladder's weights with another's direction.
    sigma_path = cache_dir / f"sigma2_{scheme}_g{group}_L{layer}_n{len(harmful)}.json"
    keys = {
        cls: cache_dir / f"acts_{scheme}_{cls}_L{layer}_{len(harmful)}.npy" for cls in ("h", "l")
    }
    if all(p.exists() for p in keys.values()) and sigma_path.exists():
        print(f"[{scheme}] cache hit", flush=True)
        return (
            np.load(keys["h"]),
            np.load(keys["l"]),
            json.loads(sigma_path.read_text(encoding="utf-8")),
        )

    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    started = time.time()
    salience = f", salience alpha={alpha}" if alpha > 0.0 else ""
    print(f"[{scheme}] loading + quantizing (group={group}{salience}) ...", flush=True)
    model, sigma = load_rtn_model(
        bits, group, eta_weight_names, channel_scales, alpha, direction
    )
    model.eval()
    print(
        f"[{scheme}] ready in {time.time() - started:.0f}s | "
        f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated | "
        f"sigma^2 for {len(sigma)}/{len(eta_weight_names)} eta matrices",
        flush=True,
    )
    if sigma:
        sigma_path.write_text(json.dumps(sigma, indent=2), encoding="utf-8")

    def encode(prompt: str) -> dict[str, Any]:
        out = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(out, "keys"):
            return {k: v.to("cuda") for k, v in out.items() if k in ("input_ids", "attention_mask")}
        return {"input_ids": out.to("cuda")}

    result: dict[str, FloatArray] = {}
    with torch.no_grad():
        for cls, prompts in (("h", harmful), ("l", benign)):
            rows, tick = [], time.time()
            for i, prompt in enumerate(prompts):
                hidden = model(**encode(prompt), output_hidden_states=True).hidden_states
                rows.append(hidden[layer][0, -1, :].float().cpu().numpy())
                if (i + 1) % 100 == 0:
                    print(
                        f"   [{scheme}/{cls}] {i + 1}/{len(prompts)}"
                        f"  {(time.time() - tick) / (i + 1) * 1000:.0f} ms/prompt",
                        flush=True,
                    )
            arr = np.stack(rows).astype(np.float64)
            np.save(keys[cls], arr)
            result[cls] = arr

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[{scheme}] done in {time.time() - started:.0f}s  shape={result['h'].shape}", flush=True)
    return result["h"], result["l"], sigma


# ---------------------------------------------------------------------------
# eta helpers
# ---------------------------------------------------------------------------


def measure_eta_chunked(
    fp16_path: Path,
    quant_path: Path,
    directions: dict[str, FloatArray],
    s_squared: float,
    chunk: int,
) -> tuple[float, list[dict[str, Any]], float | None, float | None, list[str]]:
    """measure_gguf_pair over disjoint tensor subsets, summing sigma^2.

    eta_proxy is by construction sum(sigma^2)/s^2, so chunking is exactly
    equivalent to one call (verified to 1e-9 relative). It exists only to cap
    peak RAM: all 14 dequantized ffn_down matrices from both files at float64 is
    ~3.6 GB, chunks of four is ~1 GB.
    """
    names = list(directions)
    total_sigma = 0.0
    per_matrix: list[dict[str, Any]] = []
    skipped: list[str] = []
    wholefile: float | None = None
    payload: float | None = None
    for start in range(0, len(names), chunk):
        subset = {n: directions[n] for n in names[start : start + chunk]}
        report = measure_gguf_pair(str(fp16_path), str(quant_path), subset, s_squared)
        if report.bits_per_param_wholefile is not None:
            wholefile, payload = report.bits_per_param_wholefile, report.bits_per_param_payload
        skipped.extend(report.skipped)
        if not report.available:
            continue
        total_sigma += float(report.sigma_squared or 0.0)
        per_matrix.extend(
            {"weight": m.weight_name, "sigma_squared": m.sigma_squared, "eta": m.eta}
            for m in report.measurements
        )
        gc.collect()
    return total_sigma, per_matrix, wholefile, payload, skipped


def head_sha256(path: Path, n_bytes: int = 1 << 20) -> str:
    """Hash of the first 1 MiB -- enough to detect a swapped or truncated file
    without reading 10 GB. Named so it is never mistaken for a whole-file digest."""
    with path.open("rb") as fh:
        return hashlib.sha256(fh.read(n_bytes)).hexdigest()


def margins(matrix: FloatArray, direction: FloatArray) -> FloatArray:
    unit = direction / np.linalg.norm(direction)
    return (matrix @ unit) / np.linalg.norm(matrix, axis=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:  # noqa: C901 - a linear experiment script, split would obscure it
    global MODEL_ID, GGUF_REPO, GGUF_STEM

    # A run takes twenty minutes and is usually watched through a redirect or a
    # notebook pipe, where Python block-buffers stdout at 8 KB. Without this the
    # stage results sit invisible in the buffer until the process exits, and a
    # stalled run is indistinguishable from a slow one.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID, help="HF checkpoint for every scheme")
    ap.add_argument("--gguf-repo", default=None, help="matched GGUF ladder repo (--ladder-kind gguf)")
    ap.add_argument("--gguf-stem", default=None, help="filename stem inside --gguf-repo")
    ap.add_argument("--n", type=int, default=250, help="prompts per class (pre-registered MDE 250)")
    ap.add_argument("--layer", type=int, default=-1,
                    help="residual-stream read point; -1 selects mid-depth from the model "
                         "config, matching the 14/28 point used on the 1.5B model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--ladder-kind", choices=["rtn", "gguf"], default="rtn")
    ap.add_argument("--bits", type=int, nargs="*", default=RTN_BITS, help="RTN bit-widths")
    ap.add_argument("--group", type=int, default=64, help="RTN quantization group size")
    ap.add_argument(
        "--salience-alpha", type=float, default=0.0,
        help="AWQ-style activation-aware scaling exponent. 0 = plain RTN (the control). "
             "With alpha>0 the ladder is salience-aware at the SAME bit budget, which is the "
             "controlled test of C5.",
    )
    ap.add_argument("--salience-calib", type=int, default=64,
                    help="benign prompts used to estimate per-channel activation scales")
    ap.add_argument("--gguf-ladder", nargs="*", default=GGUF_LADDER)
    ap.add_argument("--eta-chunk", type=int, default=4)
    ap.add_argument("--skip-nf4", action="store_true")
    ap.add_argument("--cache", type=Path, default=Path("artifacts/local_ladder_cache"))
    ap.add_argument("--label", default=None)
    ap.add_argument("--smoke", action="store_true", help="fewer bootstrap and null draws")
    args = ap.parse_args()

    import torch

    MODEL_ID = args.model
    if args.gguf_repo:
        GGUF_REPO = args.gguf_repo
    if args.gguf_stem:
        GGUF_STEM = args.gguf_stem

    # Namespace the cache by checkpoint: activation files are keyed by scheme,
    # class, layer and n, none of which distinguish a 1.5B run from a 3B one.
    model_slug = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in MODEL_ID)
    args.cache = args.cache / model_slug
    args.cache.mkdir(parents=True, exist_ok=True)
    n_null = 50 if args.smoke else 400
    n_boot = 200 if args.smoke else 2000
    is_rtn = args.ladder_kind == "rtn"
    label = args.label or f"local-ladder-{args.ladder_kind}"

    # One layer rule for every runner. A literal index is not portable: 14 is
    # mid-depth at 28 layers and near the output at 16, so the same number means
    # a different measurement on a different checkpoint. Validated against the
    # config BEFORE any weights are downloaded, because layer 0 breaks the eta
    # writer-block range and an over-deep layer only fails after a model load.
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_ID)
    n_layers = int(getattr(config, "num_hidden_layers", None)
                   or config.text_config.num_hidden_layers)
    if args.layer < 0:
        args.layer = n_layers // 2
        print(f"layer: auto-selected {args.layer} (mid-depth of {n_layers})")
    if not (1 <= args.layer <= n_layers):
        raise SystemExit(
            f"--layer {args.layer} is outside 1..{n_layers} for {MODEL_ID}. "
            "hidden_states[k] is the output of block k-1, so layer 0 is the embedding "
            "and has no writer blocks for the eta measurement."
        )

    print(f"model: {MODEL_ID}   layer {args.layer}/{n_layers}")
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    harmful, benign = load_prompts(args.n)
    print(f"corpus: {len(harmful)} refused / {len(benign)} benign")

    # eta is measured on residual-stream WRITERS at blocks feeding hidden_states[layer].
    # down_proj is [n_embd, n_ff]: non-square, so the direction unambiguously matches
    # axis 0. o_proj is square and its axis order cannot be recovered from shape, which
    # is why it is excluded -- guessing wrong projects along the wrong axis silently.
    writer_blocks = list(range(args.layer))
    if is_rtn:
        eta_names = [f"model.layers.{b}.mlp.down_proj.weight" for b in writer_blocks]
    else:
        eta_names = [f"blk.{b}.ffn_down.weight" for b in writer_blocks]

    gguf_paths: dict[str, Path] = {}
    if not is_rtn:
        from huggingface_hub import hf_hub_download

        print("\n--- ladder files ---")
        for quant in args.gguf_ladder:
            try:
                path = Path(hf_hub_download(GGUF_REPO, gguf_filename(quant)))
            except Exception as exc:                      # noqa: BLE001 - report and continue
                print(f"  {quant:8s} UNAVAILABLE: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            gguf_paths[quant] = path
            print(f"  {quant:8s} {path.stat().st_size / 1e9:5.2f} GB  {path.name}")
        if "fp16" not in gguf_paths:
            raise SystemExit("no F16 GGUF reference; eta needs a matched full-precision member")

    # ---- activations ----------------------------------------------------
    print(f"\n--- collecting activations ({args.ladder_kind} ladder) ---")
    acts: dict[tuple[str, str], FloatArray] = {}
    sigma_by_scheme: dict[str, dict[str, float]] = {}
    scheme_bits: dict[str, float] = {}

    acts[("FP16", "h")], acts[("FP16", "l")] = collect(
        "FP16", load_fp16_model, harmful, benign, args.layer, args.cache
    )
    direction_fp16 = difference_in_means(acts[("FP16", "h")], acts[("FP16", "l")])
    r_fp16 = direction_fp16 / np.linalg.norm(direction_fp16)

    schemes: list[str] = ["FP16"]
    if not args.skip_nf4:
        acts[("NF4", "h")], acts[("NF4", "l")] = collect(
            "NF4", load_nf4_model, harmful, benign, args.layer, args.cache
        )
        schemes.append("NF4")

    channel_scales: dict[str, Any] | None = None
    scheme_prefix = "RTN"
    if is_rtn and args.salience_alpha > 0.0:
        scheme_prefix = "SAL"
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        calib = benign[: args.salience_calib]
        print(f"estimating per-channel activation scales on {len(calib)} BENIGN prompts "
              f"(no harmful prompts, no labels) ...")
        scale_model = load_fp16_model()
        scale_model.eval()
        channel_scales = collect_channel_scales(scale_model, calib, tok)
        del scale_model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  scales for {len(channel_scales)} weight matrices; "
              f"alpha={args.salience_alpha}")

    if is_rtn:
        for bits in args.bits:
            scheme = f"{scheme_prefix}_{bits}B"
            h, low, sigma = collect_rtn(
                bits, args.group, eta_names, harmful, benign, args.layer, args.cache,
                channel_scales, args.salience_alpha, scheme_prefix, r_fp16,
            )
            acts[(scheme, "h")], acts[(scheme, "l")] = h, low
            sigma_by_scheme[scheme] = sigma
            scheme_bits[scheme] = rtn_bits_per_parameter(bits, args.group)
            schemes.append(scheme)
    else:
        for quant in gguf_paths:
            scheme = f"GGUF_{quant.upper()}"
            acts[(scheme, "h")], acts[(scheme, "l")] = collect(
                scheme, (lambda q: lambda: load_gguf_model(q))(quant),
                harmful, benign, args.layer, args.cache,
            )
            schemes.append(scheme)

    for scheme in schemes:
        assert acts[(scheme, "h")].shape == acts[("FP16", "h")].shape, f"{scheme} misaligned"
        assert acts[(scheme, "l")].shape == acts[("FP16", "l")].shape, f"{scheme} misaligned"
    dim = int(acts[("FP16", "h")].shape[1])
    print(f"row alignment verified across {len(schemes)} schemes; D={dim}")

    directions = {s: difference_in_means(acts[(s, "h")], acts[(s, "l")]) for s in schemes}

    # ---- S0 -------------------------------------------------------------
    print("\n=== STAGE 0: does the rotation replicate? (C1) ===")
    stage0: dict[str, Any] = {}
    for scheme in schemes:
        if scheme == "FP16":
            continue
        rep = rotation_replication(
            acts[("FP16", "h")], acts[("FP16", "l")],
            acts[(scheme, "h")], acts[(scheme, "l")],
            n_splits=args.splits, seed=args.seed,
        )
        rotation = angle_between(directions["FP16"], directions[scheme])
        stage0[scheme] = {
            "passes": rep.passes(), "z_score": rep.z_score, "median_cosine": rep.median_cosine,
            "null_sd": rep.null_sd, "n_splits": rep.n_splits, "dimension": rep.dimension,
            "rotation_deg": rotation,
        }
        print(f"{scheme:10s} rotation {rotation:6.2f} deg | {rep.summary()}")
    n_pass = sum(v["passes"] for v in stage0.values())
    print(f"{n_pass}/{len(stage0)} schemes PASS")

    # ---- S1 -------------------------------------------------------------
    print("\n=== STAGE 1: is the perturbation isotropic? (C5) ===")
    stage1: dict[str, Any] = {}
    for scheme in schemes:
        if scheme == "FP16":
            continue
        iso = isotropy_test(directions["FP16"], directions[scheme], n_null=n_null, seed=args.seed)
        stage1[scheme] = {
            "angle_deg": iso.angle_deg, "cosine": iso.cosine, "max_abs_z": iso.max_abs_z,
            "concentration_z": {str(f): iso.concentration_z(f) for f in iso.concentration_observed},
            "excess_kurtosis_delta": iso.excess_kurtosis_delta,
            "excess_kurtosis_reference": iso.excess_kurtosis_reference,
            "irrecoverable_fraction": iso.irrecoverable_fraction,
            "concentration_null_not_rejected": iso.is_isotropic(),
        }
        verdict = "not rejected" if iso.is_isotropic() else "REJECTED"
        print(f"{scheme:10s} concentration null: {verdict:12s} max|z|={iso.max_abs_z:6.2f}  "
              f"irrecoverable={iso.irrecoverable_fraction:.3f}")

    # ---- S2 -------------------------------------------------------------
    print("\n=== STAGE 2: eta from weights along the ladder (A3, C2) ===")
    margins_benign_fp16 = margins(acts[("FP16", "l")], direction_fp16)
    s_squared = float(np.var(margins_benign_fp16, ddof=0))
    if s_squared <= 0.0:
        raise SystemExit("FP16 benign margin variance is zero; eta is undefined")
    print(f"s^2 (FP16 benign margin variance, benign only -- no labels) = {s_squared:.6e}")
    print(f"measuring {len(eta_names)} matrices per rung (blocks 0..{writer_blocks[-1]})")

    eta_measurements: dict[str, EtaMeasurement] = {}
    stage2_detail: dict[str, Any] = {}

    if is_rtn:
        for bits in args.bits:
            scheme = f"{scheme_prefix}_{bits}B"
            sigma = sigma_by_scheme.get(scheme, {})
            if not sigma:
                print(f"  {scheme:10s} SKIPPED -- no captured weight matrices")
                continue
            total_sigma = float(sum(sigma.values()))
            eta = total_sigma / s_squared
            per_param = scheme_bits[scheme]
            eta_measurements[scheme] = EtaMeasurement(
                bits_per_param_wholefile=per_param, bits_per_param_payload=per_param, eta=eta
            )
            stage2_detail[scheme] = {
                "eta": eta, "sigma_squared": total_sigma, "bits_per_param_payload": per_param,
                "bits_per_param_wholefile": per_param, "code_bits": bits, "group": args.group,
                "n_matrices": len(sigma),
                "per_matrix": [{"weight": k, "sigma_squared": v, "eta": v / s_squared}
                               for k, v in sigma.items()],
            }
            print(f"  {scheme:10s} eta={eta:12.6g}  bits/param={per_param:6.3f}  "
                  f"({len(sigma)} matrices)")
    else:
        import gguf as gguf_pkg

        # GGUFReader.shape is ne order, the REVERSE of the numpy array the
        # projection sees, so the axis probe must dequantize rather than read .shape.
        reader = gguf_pkg.GGUFReader(str(gguf_paths["fp16"]))
        probe = next((t for t in reader.tensors if str(t.name) == eta_names[0]), None)
        if probe is None:
            print(f"WARNING: {eta_names[0]} absent from the F16 GGUF; check blk.* naming")
        else:
            np_shape = gguf_pkg.dequantize(probe.data, probe.tensor_type).shape
            ok = "axis 0 (correct)" if np_shape[0] == r_fp16.size else "AXIS MISMATCH"
            print(f"axis check: {eta_names[0]} ne {tuple(int(x) for x in probe.shape)} "
                  f"numpy {np_shape} direction {r_fp16.size} -> {ok}")
        del reader, probe
        gc.collect()

        eta_directions = {name: r_fp16 for name in eta_names}
        for quant in gguf_paths:
            if quant == "fp16":
                continue
            started = time.time()
            total_sigma, per_matrix, wholefile, payload, skipped = measure_eta_chunked(
                gguf_paths["fp16"], gguf_paths[quant], eta_directions, s_squared, args.eta_chunk
            )
            if not per_matrix or payload is None or wholefile is None:
                print(f"  {quant:8s} SKIPPED -- no matched matrices or bit accounting. "
                      f"reasons: {skipped[:2]}")
                continue
            scheme = f"GGUF_{quant.upper()}"
            eta = total_sigma / s_squared
            eta_measurements[scheme] = EtaMeasurement(
                bits_per_param_wholefile=wholefile, bits_per_param_payload=payload, eta=eta
            )
            stage2_detail[scheme] = {
                "eta": eta, "sigma_squared": total_sigma, "bits_per_param_payload": payload,
                "bits_per_param_wholefile": wholefile, "n_matrices": len(per_matrix),
                "per_matrix": per_matrix, "skipped": skipped[:8],
                "head_sha256_1mib": head_sha256(gguf_paths[quant]),
            }
            print(f"  {quant:8s} eta={eta:12.6g}  payload bits/param={payload:6.3f}  "
                  f"({len(per_matrix)} matrices, {time.time() - started:.0f}s)")

    eta_fit = None
    stage2_fit: dict[str, Any] | None = None
    if len(eta_measurements) >= 3:
        eta_fit = fit_eta_vs_bits_report(eta_measurements)
        ci = (
            f"[{eta_fit.exponent_ci_low:.4f}, {eta_fit.exponent_ci_high:.4f}]"
            if eta_fit.exponent_ci_low is not None
            else "unavailable"
        )
        print(f"\n  eta_4={eta_fit.eta_4:.6g}  exponent={eta_fit.exponent:.4f}  95% CI {ci}")
        print(f"  R^2={eta_fit.r_squared:.4f}  rmse(log)={eta_fit.rmse_log:.4f}  "
              f"n={eta_fit.n_points}")
        print(f"  model-conditional CI excludes 4: "
              f"{eta_fit.model_conditional_exponent_ci_excludes_four}")
        stage2_fit = {
            "eta_4": eta_fit.eta_4, "exponent": eta_fit.exponent,
            "exponent_ci": [eta_fit.exponent_ci_low, eta_fit.exponent_ci_high],
            "r_squared": eta_fit.r_squared, "rmse_log": eta_fit.rmse_log,
            "n_points": eta_fit.n_points,
            "model_conditional_exponent_ci_excludes_four":
                eta_fit.model_conditional_exponent_ci_excludes_four,
            "skipped_zero_schemes": list(eta_fit.skipped_zero_schemes),
        }
    else:
        print(f"\n  only {len(eta_measurements)} usable rungs -- need >= 3 to fit the exponent")

    # ---- S3 -------------------------------------------------------------
    print("\n=== STAGE 3: d' per scheme (C2) ===")
    stage3: dict[str, Any] = {}
    for scheme in schemes:
        mh = margins(acts[(scheme, "h")], directions[scheme])
        ml = margins(acts[(scheme, "l")], directions[scheme])
        # r = mean(harmful) - mean(harmless) makes harmful score HIGH, so every
        # readout is fires_high=True. Using False is what once reported d' = -0.642.
        held_mean, held_std = held_out_d_prime(
            acts[(scheme, "h")], acts[(scheme, "l")],
            n_splits=args.splits, fires_high=True, seed=args.seed,
        )
        in_sample = d_prime_with_ci(mh, ml, fires_high=True, n_bootstrap=n_boot, seed=args.seed)
        gap = gaussianity_gap(mh, ml, fires_high=True)
        stage3[scheme] = {
            "d_prime_held_out": held_mean, "d_prime_held_out_std": held_std,
            "d_prime_in_sample": in_sample.d_prime,
            "ci": [in_sample.ci_low, in_sample.ci_high], "auc": in_sample.auc,
            "gaussianity_gap": gap,
        }
        print(f"{scheme:10s} held-out d'={held_mean:+.4f} +/-{held_std:.4f}   "
              f"in-sample {in_sample.d_prime:+.4f} [{in_sample.ci_low:+.4f}, "
              f"{in_sample.ci_high:+.4f}]   gap {gap:.3f}"
              f"{'' if gap <= 0.05 else '  (A2 FAILS)'}")

    d_prime_0 = stage3["FP16"]["d_prime_held_out"]
    print(f"\nd'_0 (FP16, held out) = {d_prime_0:.4f}")

    print("\n--- falsifier F5: eta(weights) vs eta(d' decay) ---")
    f5: dict[str, Any] = {}
    for scheme, measurement in eta_measurements.items():
        if scheme not in stage3:
            continue
        d_q = stage3[scheme]["d_prime_held_out"]
        eta_w = measurement.eta
        if d_q <= 0.0 or d_q >= d_prime_0:
            print(f"  {scheme:10s} d'={d_q:+.4f} does not decay below d'_0; implied eta undefined")
            f5[scheme] = {"d_prime": d_q, "eta_behavioural": None, "eta_weights": eta_w,
                          "ratio": None}
            continue
        eta_b = implied_eta(d_prime_0, d_q)
        ratio = eta_b / eta_w if eta_w > 0 else None
        f5[scheme] = {"d_prime": d_q, "eta_behavioural": eta_b, "eta_weights": eta_w,
                      "ratio": ratio}
        tail = f"ratio={ratio:.4g}" if ratio is not None else "ratio undefined (eta_weights=0)"
        print(f"  {scheme:10s} d'={d_q:+.4f}  eta(d')={eta_b:.6g}  eta(w)={eta_w:.6g}  {tail}")

    ratios = [v["ratio"] for v in f5.values() if v["ratio"] and np.isfinite(v["ratio"])]
    ratio_spread = float(max(ratios) / min(ratios)) if len(ratios) >= 3 else None
    if ratio_spread is not None:
        # The absolute ratio is not interpretable: eta carries an arbitrary scale
        # from s^2 and from which matrices were measured. Its CONSTANCY across
        # rungs is scale-free, and that is the pre-registered form of F5.
        print(f"\n  F5 (scale-free): ratio spread across rungs = {ratio_spread:.3f}x")
        print("  Near 1 => weight-space and behavioural accounts of eta move together.")
        print("  Far from 1 => F5 fires and C2's MECHANISM is refuted.")

    # ---- S4 -------------------------------------------------------------
    print("\n=== STAGE 4: out-of-sample prediction (C3) ===")
    usable = sorted(
        ((s, m.bits_per_param_payload) for s, m in eta_measurements.items() if s in stage3),
        key=lambda kv: -kv[1],
    )
    stage4: dict[str, Any] | None = None
    if len(usable) >= 4:
        split = len(usable) // 2
        train, test = usable[:split], usable[split:]

        # BOTH parameters must come from the training rungs only. An earlier
        # version took `base` from `eta_fit`, which is fitted over every rung
        # including the held-out half, so the "out-of-sample" prediction was
        # conditioned on the answers it was predicting.
        train_measurements = {s: eta_measurements[s] for s, _ in train}
        if len(train_measurements) >= 3:
            train_fit = fit_eta_vs_bits_report(train_measurements)
            base = train_fit.exponent
            base_source = f"fitted on TRAIN rungs only ({len(train_measurements)} points)"
        else:
            base = 4.0
            base_source = "assumed 4.0 (fewer than 3 training rungs)"
        eta_4 = float(np.median([eta_measurements[s].eta * (base ** (b - 4.0)) for s, b in train]))
        print(f"train (high precision): {[s for s, _ in train]}")
        print(f"test  (low precision) : {[s for s, _ in test]}")
        print(f"base={base:.4f} ({base_source})   eta_4={eta_4:.6g}  [TRAIN rungs only]")
        try:
            b_star = collapse_bits_threshold_closed_form(d_prime_0, args.alpha, eta_4, base)
            print(f"predicted collapse b* = {b_star:.2f} bits at alpha={args.alpha}")
        except ValueError as exc:
            b_star = None
            print(f"collapse b* undefined: {exc}")

        rows, errors = [], []
        for scheme, bits in test:
            predicted = d_prime_at_bits(bits, d_prime_0, eta_4, base)
            actual = stage3[scheme]["d_prime_held_out"]
            sd = stage3[scheme]["d_prime_held_out_std"]
            error = predicted - actual
            errors.append(error)
            rows.append({"scheme": scheme, "bits": bits, "predicted": predicted,
                         "actual": actual, "actual_std": sd, "error": error,
                         "within_1_sd": bool(abs(error) <= sd)})
            print(f"  {scheme:10s} bits={bits:6.2f}  predicted={predicted:+.4f}  "
                  f"measured={actual:+.4f} +/-{sd:.4f}  err={error:+.4f}  "
                  f"{'within 1 sd' if abs(error) <= sd else 'OUTSIDE 1 sd'}")
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        hits = sum(r["within_1_sd"] for r in rows)
        print(f"\nout-of-sample RMSE = {rmse:.4f} d' units; {hits}/{len(rows)} within 1 sd")
        stage4 = {"base": base, "base_source": base_source, "eta_4_train": eta_4,
                  "d_prime_0": d_prime_0, "b_star": b_star, "alpha": args.alpha,
                  "train": [s for s, _ in train], "test": [s for s, _ in test],
                  "rows": rows, "rmse": rmse, "within_1_sd": hits, "n_test": len(rows)}
    else:
        print(f"only {len(usable)} rungs have both eta and d'; need >= 4 for a train/test split")

    # ---- persist --------------------------------------------------------
    extra: dict[str, Any] = {
        "stage": "local_stages_1_4", "ladder_kind": args.ladder_kind,
        "layer": args.layer, "n_per_class": args.n, "seed": args.seed,
        "n_splits": args.splits, "alpha": args.alpha, "s_squared": s_squared,
        "eta_weight_names": eta_names, "schemes": schemes, "smoke": args.smoke,
        "n_null": n_null, "n_bootstrap": n_boot, "gpu": torch.cuda.get_device_name(0),
        "label_ceiling_note": (
            "'refused' describes the hh-rlhf REJECTED response, not this model's behaviour. "
            "d' is bounded by label quality independently of quantization."
        ),
        "eta_note": (
            "eta sums per-matrix projected weight-perturbation variances over residual-stream "
            "writers, which assumes independent layer contributions and isotropic unit-variance "
            "activations. Both are false. Per-matrix values are stored so any other aggregation "
            "is recomputable without re-running."
        ),
    }
    if is_rtn:
        extra.update({
            "rtn_bits": args.bits, "rtn_group": args.group,
            "salience_alpha": args.salience_alpha,
            "salience_calib_prompts": args.salience_calib if args.salience_alpha > 0 else 0,
            "salience_note": (
                "alpha>0 applies AWQ-style activation-aware per-input-channel scaling before "
                "RTN and folds the inverse back out. Bit budget, group size, checkpoint, corpus "
                "and code path are identical to the alpha=0 control, so the ONLY difference is "
                "salience-awareness. This is the controlled form of the C5 comparison; setting "
                "our RTN against a published AWQ build would confound the algorithm with its "
                "group size, calibration set, and release."
            ),
            "rtn_scheme": "per-group asymmetric round-to-nearest along the input axis",
            "rtn_scope": "transformer-block nn.Linear only; embeddings and lm_head stay FP16",
            "bits_per_param_formula": "code_bits + 32/group (fp16 scale + fp16 zero per group)",
        })
    else:
        extra.update({
            "gguf_repo": GGUF_REPO, "gguf_ladder": list(gguf_paths),
            "eta_chunk": args.eta_chunk,
            "gguf_head_sha256_1mib": {q: head_sha256(p) for q, p in gguf_paths.items()},
        })

    run = new_run(label, model_id=MODEL_ID, extra=extra)
    record_environment(run)
    record_corpus(run, "harmful_refused", harmful)
    record_corpus(run, "benign", benign)
    for scheme in schemes:
        run.save_array("activations", f"{scheme}_harmful", acts[(scheme, "h")])
        run.save_array("activations", f"{scheme}_benign", acts[(scheme, "l")])
        run.save_array("directions", f"r_hat_{scheme}", directions[scheme])
        run.save_array("margins", f"{scheme}_harmful",
                       margins(acts[(scheme, "h")], directions[scheme]))
        run.save_array("margins", f"{scheme}_benign",
                       margins(acts[(scheme, "l")], directions[scheme]))
    run.save_json("stage0_replication", stage0)
    run.save_json("stage1_isotropy", stage1)
    run.save_json("stage2_eta", {"s_squared": s_squared, "schemes": stage2_detail})
    if stage2_fit is not None:
        run.save_json("stage2_fit", stage2_fit)
    run.save_json("stage3_dprime", stage3)
    run.save_json("stage3_f5", {"d_prime_0": d_prime_0, "per_scheme": f5,
                                "ratio_spread": ratio_spread})
    if stage4 is not None:
        run.save_json("stage4_prediction", stage4)
    run.write_manifest()

    summary = (
        f"{args.ladder_kind} ladder: {n_pass}/{len(stage0)} replicate; d'_0={d_prime_0:.3f}"
        + (f"; exponent={eta_fit.exponent:.2f}" if eta_fit is not None else "")
        + (f"; OOS RMSE={stage4['rmse']:.3f}" if stage4 else "; no OOS split")
    )
    run.append_to_index(summary)
    print(f"\nrun: {run.path}\n{summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
