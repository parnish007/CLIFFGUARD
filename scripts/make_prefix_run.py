"""Derive a short-window run from one long generation pass.

Comparing separate greedy generations at 48 and 256 tokens quietly assumes
that their first 48 tokens are identical.  That is usually plausible, but it
is a floating-point determinism assumption rather than an experimental result.
This script makes the short run from exact token prefixes of the long run, so
the two grading windows are paired on the same generated text.  An optional
comparison with an independently generated short run reports whether the
assumption held in practice.

Usage:
  python scripts/make_prefix_run.py <source-run-dir> --tokens 48 \
      [--compare <48-token-run-dir>] [--label <suffix>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliffguard.eval.storage import new_run


def _positive_int(value: str) -> int:
    """Reject a zero-token prefix before loading a model tokenizer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{run_dir}: no manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    return payload


def _load_completion_files(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load every scheme without guessing from a manifest that may be stale."""
    paths = sorted((run_dir / "results").glob("completions_*.json"))
    if not paths:
        raise ValueError(f"{run_dir}: no results/completions_<scheme>.json files")

    payloads: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        completions = payload.get("completions") if isinstance(payload, dict) else None
        if not isinstance(completions, list) or not all(
                isinstance(text, str) for text in completions):
            raise ValueError(f"{path}: expected a string list under 'completions'")
        payloads[path.name] = payload
    return payloads


def _excerpt(text: str, limit: int = 96) -> str:
    """Keep disagreement reports readable without hiding line-break differences."""
    quoted = repr(text)
    return quoted if len(quoted) <= limit else quoted[:limit - 3] + "..."


def _agreement_report(
    prefixes: dict[str, list[str]],
    comparison: dict[str, dict[str, Any]],
    compare_run: Path,
) -> dict[str, Any]:
    """Measure decoding drift while retaining enough examples to inspect it.

    Read a disagreement carefully: it has two possible causes and this
    comparison cannot separate them.

    The interesting one is that batched greedy decoding did not in fact produce
    the same first 48 tokens at the two budgets, which is exactly the
    assumption the prefix run exists to avoid relying on.

    The dull one is that tokenization is not guaranteed idempotent. The prefix
    side is `decode(encode(long_text)[:n])`, so a completion whose text does not
    re-encode to the tokens that produced it can differ here by a space or a
    merge boundary while the generation was identical. Byte equality is
    therefore a strict test, and `different` is an upper bound on genuine drift.

    That is why the first few differing pairs are kept rather than only counted:
    a handful of whitespace-boundary differences reads very differently from a
    completion that diverges mid-sentence, and only looking at them tells you
    which you have.
    """
    schemes: dict[str, Any] = {}
    for filename, prefix_texts in prefixes.items():
        compare_payload = comparison.get(filename)
        if compare_payload is None:
            raise ValueError(
                f"{compare_run}: missing results/{filename} needed to compare "
                "the corresponding scheme")
        compare_texts = compare_payload["completions"]
        if len(compare_texts) != len(prefix_texts):
            raise ValueError(
                f"{compare_run}: results/{filename} has {len(compare_texts)} "
                f"completions, but the source has {len(prefix_texts)}")

        differences = [
            index for index, (prefix, generated) in enumerate(
                zip(prefix_texts, compare_texts))
            if prefix.encode("utf-8") != generated.encode("utf-8")
        ]
        schemes[filename.removeprefix("completions_").removesuffix(".json")] = {
            "identical": len(prefix_texts) - len(differences),
            "different": len(differences),
            "n_completions": len(prefix_texts),
            "first_differences": [
                {
                    "index": index,
                    "prefix": _excerpt(prefix_texts[index]),
                    "comparison": _excerpt(compare_texts[index]),
                }
                for index in differences[:3]
            ],
        }
    return {"compare_run_id": _load_manifest(compare_run).get(
        "run_id", compare_run.name), "schemes": schemes}


def _print_agreement(report: dict[str, Any]) -> None:
    print(f"prefix agreement with {report['compare_run_id']}:")
    for scheme, result in report["schemes"].items():
        print(f"  {scheme}: {result['identical']}/{result['n_completions']} "
              f"byte-identical; {result['different']} differ")
        for difference in result["first_differences"]:
            print(f"    index {difference['index']}: prefix {difference['prefix']}")
            print(f"             compare {difference['comparison']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_run", type=Path,
                    help="completed 256-token run directory")
    ap.add_argument("--tokens", type=_positive_int, default=48,
                    help="number of tokenizer tokens to retain (default: 48)")
    ap.add_argument("--compare", type=Path,
                    help="separate short-window run to check for generation drift")
    ap.add_argument("--label", metavar="SUFFIX",
                    help="append SUFFIX to the derived run label")
    args = ap.parse_args()

    try:
        source = args.source_run.resolve()
        manifest = _load_manifest(source)
        model_id = manifest.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"{source}: manifest has no model_id")
        prompts = source / "results" / "prompts.json"
        if not prompts.is_file():
            raise ValueError(f"{source}: no results/prompts.json to copy")
        completion_payloads = _load_completion_files(source)

        comparison_payloads: dict[str, dict[str, Any]] | None = None
        if args.compare is not None:
            comparison = args.compare.resolve()
            _load_manifest(comparison)
            comparison_payloads = _load_completion_files(comparison)

        # Importing transformers above would make even --help depend on a GPU
        # environment.  All path and manifest failures should remain actionable
        # on the CPU-only machine used to inspect completed runs.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prefixes: dict[str, list[str]] = {}
        derived_payloads: dict[str, dict[str, Any]] = {}
        for filename, payload in completion_payloads.items():
            truncated = [
                tokenizer.decode(
                    tokenizer.encode(text, add_special_tokens=False)[:args.tokens],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for text in payload["completions"]
            ]
            prefixes[filename] = truncated
            # Preserve any per-file metadata: a prefix changes the completion,
            # not the identity of the scheme that produced it.
            derived_payloads[filename] = {**payload, "completions": truncated}

        agreement = None
        if comparison_payloads is not None:
            agreement = _agreement_report(prefixes, comparison_payloads,
                                          args.compare.resolve())

        source_label = manifest.get("label", source.name)
        label = f"{source_label}-prefix{args.tokens}"
        if args.label:
            label += f"-{args.label}"
        carried = {
            key: manifest[key]
            for key in ("corpora", "model_id", "seed", "schemes", "n_prompts")
            if key in manifest
        }
        carried.update({
            "max_new_tokens": args.tokens,
            "derived_from": manifest.get("run_id", source.name),
            "derivation": "token-prefix",
        })
        if agreement is not None:
            carried["prefix_agreement"] = agreement

        run = new_run(label, model_id=model_id, extra=carried)
        # new_run normally creates a timestamped path, but it intentionally uses
        # exist_ok for resumable writers.  A collision must not turn this derived
        # run into an in-place mutation of a completed experiment.
        if (run.path / "manifest.json").exists():
            raise ValueError(f"target directory already exists: {run.path}")
        shutil.copyfile(prompts, run.path / "results" / "prompts.json")
        for filename, payload in derived_payloads.items():
            run.save_json(filename.removesuffix(".json"), payload)
        run.write_manifest()
        run.append_to_index(
            f"{args.tokens}-token prefixes derived from "
            f"{carried['derived_from']}")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except ImportError:
        print("error: transformers is required to make token prefixes", file=sys.stderr)
        return 2

    print(f"prefix run: {run.path}")
    if agreement is not None:
        _print_agreement(agreement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
