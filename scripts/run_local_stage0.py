"""Stage 0 on local hardware — the gate that conditions the whole project.

Runs the real experiment on a 6 GB laptop GPU using a small instruct model:
extract residual-stream activations at a fixed layer for FP16 and NF4 over the
SAME prompts, then test whether the FP16 -> NF4 rotation of the refusal
direction REPLICATES across disjoint prompt halves.

Why replication and not an angle threshold: at fixed prompts the sampling noise
is common to both schemes and largely cancels, so the observed rotation is not
explained by prompt-sampling noise in the first place. The real question is
whether the rotation is systematic (a property of the quantizer) or
idiosyncratic to these prompts. Two earlier gate designs were wrong; see
docs/build_log.md Entry 5.

Requires a torch environment. On this machine that lives OUTSIDE the project
venv because C: is nearly full:

    D:/AI/cliffguard-gpu/.venv/Scripts/python.exe scripts/run_local_stage0.py

Model choice is deliberately small so FP16 fits in 6 GB alongside activations.
Llama-3.2-3B FP16 is ~6.4 GB and does NOT fit; use Colab for that.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cliffguard.eval.discriminability import (  # noqa: E402
    d_prime_with_ci,
    gaussianity_gap,
    held_out_d_prime,
    implied_eta,
)
from cliffguard.eval.isotropy import isotropy_test  # noqa: E402
from cliffguard.eval.noise_floor import (  # noqa: E402
    angle_between,
    difference_in_means,
    paired_direction_shift,
    rotation_replication,
    split_half_noise_floor,
)
from cliffguard.eval.storage import (  # noqa: E402
    new_run,
    record_corpus,
    record_environment,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUT = REPO / "artifacts" / "local_stage0"


def load_prompts(n: int) -> tuple[list[str], list[str]]:
    """Load the real Fold A corpus. Refuses to invent placeholder prompts."""
    base = REPO / "data" / "folds" / "fold_a"
    harmful_f = base / "anthropic_hh_refused.jsonl"
    benign_f = base / "anthropic_hh_benign.jsonl"
    if not (harmful_f.exists() and benign_f.exists()):
        raise SystemExit(
            f"Fold A corpus not found under {base}.\n"
            "Run: python scripts/download_fold_a.py --download\n"
            "Refusing to substitute synthetic prompts — the result would be meaningless."
        )

    def read(p: Path) -> list[str]:
        return [
            json.loads(line)["prompt"]
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    harmful, benign = read(harmful_f)[:n], read(benign_f)[:n]
    if len(harmful) < 4 or len(benign) < 4:
        raise SystemExit("corpus too small for a split-half control")
    return harmful, benign


def collect_activations(
    model_id: str, scheme: str, cls: str, prompts: list[str], layer: int, out_dir: Path
) -> np.ndarray:
    """One forward pass per prompt; residual stream at `layer`, last token.

    Cached to .npy so the analysis can be re-run without a GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tag = model_id.split("/")[-1]
    # NOTE: the class MUST be in the cache key. Without it the benign call
    # silently reuses the harmful activations and the difference-in-means
    # direction collapses to the zero vector.
    cache = out_dir / f"acts_{tag}_{scheme}_{cls}_L{layer}_{len(prompts)}.npy"
    if cache.exists():
        print(f"  [cache] {cache.name}")
        return np.load(cache)

    # transformers 5.x: `dtype` replaced `torch_dtype`, and output_hidden_states
    # belongs on the forward call, not the loader.
    kwargs: dict[str, object] = {"device_map": "auto"}
    if scheme == "FP16":
        kwargs["dtype"] = torch.float16
    elif scheme == "NF4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        raise ValueError(f"unsupported scheme {scheme!r}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  loading {model_id} as {scheme} ...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    rows = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                truncation=True,
                max_length=512,
            )
            # transformers 5.x returns a BatchEncoding; older versions a Tensor.
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            ids = ids.to(model.device)
            out = model(ids, output_hidden_states=True)
            hs = out.hidden_states[layer][0, -1, :]
            rows.append(hs.float().cpu().numpy())
            if (i + 1) % 50 == 0:
                print(f"    {scheme}: {i + 1}/{len(prompts)}")

    arr = np.stack(rows).astype(np.float64)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    del model
    torch.cuda.empty_cache()
    print(f"  saved {cache.name} shape={arr.shape}")
    return arr


def margins(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    d = direction / np.linalg.norm(direction)
    return (acts @ d) / np.linalg.norm(acts, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n", type=int, default=200, help="prompts per class")
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    harmful, benign = load_prompts(args.n)

    run = new_run(
        label=f"stage0-{args.model.split('/')[-1]}-fp16-nf4",
        model_id=args.model,
        extra={"layer": args.layer, "n_per_class": args.n,
               "n_splits": args.splits, "seed": args.seed},
    )
    record_corpus(run, "harmful_refused", harmful)
    record_corpus(run, "benign", benign)
    record_environment(run)
    print(f"run dir: {run.path}\n")
    print(f"corpus: {len(harmful)} refused / {len(benign)} benign")
    print(f"model : {args.model}  layer={args.layer}\n")

    acts: dict[tuple[str, str], np.ndarray] = {}
    for scheme in ("FP16", "NF4"):
        print(f"[{scheme}]")
        acts[(scheme, "h")] = collect_activations(
            args.model, scheme, "harmful", harmful, args.layer, args.out
        )
        acts[(scheme, "l")] = collect_activations(
            args.model, scheme, "benign", benign, args.layer, args.out
        )
        print()

    r_fp16 = difference_in_means(acts[("FP16", "h")], acts[("FP16", "l")])
    r_nf4 = difference_in_means(acts[("NF4", "h")], acts[("NF4", "l")])
    observed = angle_between(r_fp16, r_nf4)
    dim = int(r_fp16.size)

    print("=" * 72)
    print("STAGE 0 — does the FP16 -> NF4 rotation replicate?")
    print("=" * 72)
    rep = rotation_replication(
        acts[("FP16", "h")], acts[("FP16", "l")],
        acts[("NF4", "h")], acts[("NF4", "l")],
        n_splits=args.splits, seed=args.seed,
    )
    print(f"observed rotation : {observed:.3f} deg (D={dim})")
    print(rep.summary())
    print()

    print("--- diagnostics (NOT the gate) ---")
    floor = split_half_noise_floor(
        acts[("FP16", "h")], acts[("FP16", "l")], n_splits=args.splits, seed=args.seed
    )
    print(floor.summary())
    paired = paired_direction_shift(
        acts[("FP16", "h")], acts[("FP16", "l")],
        acts[("NF4", "h")], acts[("NF4", "l")],
        n_bootstrap=500, seed=args.seed,
    )
    print(paired.summary())
    print()

    print("=" * 72)
    print("ISOTROPY (interpretable only if Stage 0 PASSED)")
    print("=" * 72)
    iso = isotropy_test(r_fp16, r_nf4, n_null=400, seed=args.seed)
    print(iso.summary())
    print()

    print("=" * 72)
    print("DISCRIMINABILITY")
    print("=" * 72)
    dprime = {}
    for scheme, direction in (("FP16", r_fp16), ("NF4", r_nf4)):
        mh = margins(acts[(scheme, "h")], direction)
        ml = margins(acts[(scheme, "l")], direction)
        run.save_array("margins", f"{scheme}_harmful", mh)
        run.save_array("margins", f"{scheme}_benign", ml)
        # r_hat = mean(harmful) - mean(harmless), so harmful scores HIGH on it:
        # this is a HARMFULNESS direction (fires HIGH), not a refusal margin.
        d = d_prime_with_ci(mh, ml, fires_high=True, n_bootstrap=2000, seed=args.seed)
        gap = gaussianity_gap(mh, ml, fires_high=True)
        ho_mean, ho_std = held_out_d_prime(
            acts[(scheme, "h")], acts[(scheme, "l")], n_splits=20,
            fires_high=True, seed=args.seed,
        )
        dprime[scheme] = {
            "d_prime_in_sample": d.d_prime, "ci": [d.ci_low, d.ci_high],
            "d_prime_held_out": ho_mean, "d_prime_held_out_std": ho_std,
            "gap": gap,
        }
        print(f"{scheme:5s} {d.summary()}")
        print(f"      HELD-OUT d' = {ho_mean:.3f} +/- {ho_std:.3f}  <-- use this one")
        print(f"      gaussianity gap = {gap:.3f} "
              f"{'(A2 OK)' if gap <= 0.05 else '(A2 FAILS — closed-form TPR void)'}")

    eta_behavioural = None
    if dprime["FP16"]["d_prime_held_out"] > 0 and dprime["NF4"]["d_prime_held_out"] > 0:
        # Held-out, not in-sample: the direction estimator overfits.
        eta_behavioural = implied_eta(
            dprime["FP16"]["d_prime_held_out"], dprime["NF4"]["d_prime_held_out"]
        )
        print(f"\nimplied eta(NF4) from d' decay = {eta_behavioural:.4f}")
        print("Compare against eta measured independently from the weights")
        print("(cliffguard/eval/noise_spectrum.py). Disagreement refutes C2 — falsifier F5.")

    result = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": args.model, "layer": args.layer,
        "n_per_class": len(harmful), "dimension": dim,
        "stage0": {
            "observed_rotation_deg": observed,
            "median_cosine": rep.median_cosine,
            "z_score": rep.z_score,
            "null_sd": rep.null_sd,
            "n_splits": rep.n_splits,
            "PASSES": rep.passes(),
        },
        "diagnostics": {
            "split_half_median_deg": floor.median_deg,
            "paired_observed_deg": paired.observed_angle_deg,
        },
        "isotropy": {
            "angle_deg": iso.angle_deg,
            "excess_kurtosis_delta": iso.excess_kurtosis_delta,
            "excess_kurtosis_reference": iso.excess_kurtosis_reference,
            "max_abs_z": iso.max_abs_z,
            "is_isotropic": iso.is_isotropic(),
            "irrecoverable_fraction": iso.irrecoverable_fraction,
        },
        "d_prime": dprime,
        "implied_eta_nf4": eta_behavioural,
    }
    # Persist raw arrays alongside the summary so the analysis can be redone
    # without a GPU, and so the numbers are auditable (defect D3).
    for (scheme, cls), arr in acts.items():
        run.save_array("activations", f"{scheme}_{cls}", arr)
    run.save_array("directions", "r_fp16", r_fp16)
    run.save_array("directions", "r_nf4", r_nf4)
    out_file = run.save_json("stage0_result", result)
    run.write_manifest()
    run.append_to_index(
        f"Stage 0 {'PASS' if rep.passes() else 'NULL'} "
        f"(rot {observed:.1f} deg, replication z={rep.z_score:+.1f}); "
        f"held-out d' FP16={dprime['FP16']['d_prime_held_out']:.3f} "
        f"NF4={dprime['NF4']['d_prime_held_out']:.3f}; "
        f"concentration null "
        f"{'REJECTED' if not iso.is_isotropic() else 'not rejected'}"
    )

    print("\n" + "=" * 72)
    if rep.passes():
        print("VERDICT: PASS — the rotation replicates. Premise P1 survives.")
        print("         Isotropy and d' results above are interpretable.")
    else:
        print("VERDICT: NULL — the rotation does NOT replicate across prompt halves.")
        print("         It is idiosyncratic to these prompts, not a property of the")
        print("         quantizer. Do NOT claim a geometric quantization effect.")
        print("         The theorem PREDICTS direction retention, so this is consistent")
        print("         with it — but claim C1 is unproven and everything downstream")
        print("         must be reported as conditional.")
    print("=" * 72)
    print(f"written: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
