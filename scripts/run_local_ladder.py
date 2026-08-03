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


def gguf_filename(quant: str) -> str:
    return f"{GGUF_STEM}-{quant}.gguf"


def load_fp16_model() -> Any:
    """from_pretrained with the transformers 4.x/5.x dtype kwarg difference absorbed."""
    import torch
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    return model.to("cuda")


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
) -> tuple[Any, dict[str, Any]]:
    """FP16 checkpoint with every block Linear replaced by its RTN reconstruction.

    With `alpha > 0` and `channel_scales` supplied, the reconstruction is
    salience-aware (AWQ-style) at the same bit budget -- the controlled contrast
    for claim C5.

    Returns the model and, for the weights named in `eta_weight_names`, the
    (fp16, quantized) numpy pair needed for the eta measurement. Only those few
    matrices are retained: keeping all of them as float64 would cost several GB.
    """
    import torch

    model = load_fp16_model()
    captured: dict[str, Any] = {}
    wanted = set(eta_weight_names)
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
            if weight_name in wanted:
                captured[weight_name] = (
                    original.detach().to(torch.float32).cpu().numpy().astype(np.float64),
                    quantized.detach().to(torch.float32).cpu().numpy().astype(np.float64),
                )
            module.weight.data = quantized
    return model, captured


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

    from cliffguard.eval.noise_spectrum import projected_perturbation_variance

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    started = time.time()
    salience = f", salience alpha={alpha}" if alpha > 0.0 else ""
    print(f"[{scheme}] loading + quantizing (group={group}{salience}) ...", flush=True)
    model, captured = load_rtn_model(bits, group, eta_weight_names, channel_scales, alpha)
    model.eval()
    print(
        f"[{scheme}] ready in {time.time() - started:.0f}s | "
        f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated | "
        f"captured {len(captured)}/{len(eta_weight_names)} eta matrices",
        flush=True,
    )

    # sigma^2 needs the direction, which is not known yet on the first rung.
    # It is computed after collection using the FP16 direction passed in via
    # the module-level slot below, so capture is deferred to the caller.
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

    # The direction used for sigma^2 must be the FP16 one, which the caller owns.
    # It is stashed on the function object so the signature stays simple.
    direction = getattr(collect_rtn, "direction", None)
    sigma: dict[str, float] = {}
    if direction is not None:
        for name, (w_fp16, w_q) in captured.items():
            sigma[name] = projected_perturbation_variance(w_fp16, w_q, direction)
        sigma_path.write_text(json.dumps(sigma, indent=2), encoding="utf-8")
    del captured
    gc.collect()

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

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID, help="HF checkpoint for every scheme")
    ap.add_argument("--gguf-repo", default=None, help="matched GGUF ladder repo (--ladder-kind gguf)")
    ap.add_argument("--gguf-stem", default=None, help="filename stem inside --gguf-repo")
    ap.add_argument("--n", type=int, default=250, help="prompts per class (pre-registered MDE 250)")
    ap.add_argument("--layer", type=int, default=14)
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

    print(f"model: {MODEL_ID}")
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
    # collect_rtn needs the FP16 direction to compute sigma^2 in the same pass.
    collect_rtn.direction = r_fp16                              # type: ignore[attr-defined]

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
                channel_scales, args.salience_alpha, scheme_prefix,
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
        base = eta_fit.exponent if eta_fit is not None else 4.0
        base_source = "fitted" if eta_fit is not None else "assumed 4.0 (no fit available)"
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
