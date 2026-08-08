"""Assemble a prompt corpus that carries per-prompt harmfulness labels.

The manuscript's sharpest limitation is that its prompts have none. HH-RLHF
first-turn utterances span harmful, benign and ambiguous requests without
annotation, so a transition from baseline refusal to quantized compliance is a
change in the model's own decision and nothing more. We cannot call it a safety
event, and -- more awkwardly, since the headline is that refusal *rises* -- we
cannot say whether that rise is a safety improvement or a usability regression.

With labelled prompts both questions become answerable, and they are different
questions:

    harmful prompt,  baseline refused, rung complied  ->  safety regression
    benign  prompt,  baseline complied, rung refused  ->  over-refusal regression

The present design lumps these together. This script is what separates them.

Suites, and why each is here:

  or-bench-hard    Prompts that LOOK harmful and are not. This is the
                   over-refusal probe, and it is the one that prices the
                   conservative shift the paper reports. Its authors built it by
                   rewriting harmful prompts until they were benign and then
                   checking the result with moderators, so a refusal here is a
                   false positive by construction.
  or-bench-toxic   The same authors' genuinely toxic control, built alongside
                   the benign set, so the two share construction and differ in
                   the thing that matters.
  xstest           250 safe and 200 unsafe prompts in matched pairs, an
                   independent take on the same contrast. Noted in the
                   literature as easy for current models; kept as a control
                   rather than a headline.
  strongreject     A graded-harm set built specifically to resist the
                   non-substantive-output inflation this project characterises,
                   which makes it the most directly comparable prior instrument.
  harmbench        Standard behaviours only. Contextual behaviours need a
                   ContextString this pipeline does not thread through, and
                   copyright behaviours are a different task; including either
                   would put prompts in the corpus the runner cannot present
                   faithfully.
  advbench         Widely used, so it is what makes our numbers comparable with
                   the jailbreak literature, whatever its known weaknesses.

Provenance note. HarmBench, AdvBench and StrongREJECT are read from the
repositories that publish them rather than from HuggingFace mirrors. The mirrors
are gated behind an access agreement, and a corpus builder that fails with an
authentication error on a fresh clone is a corpus builder that does not work.
The upstream repositories are also the authoritative copies.

Usage:
  python scripts/download_eval_suites.py --download
  python scripts/download_eval_suites.py --download --suites or-bench-hard xstest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

_OUT_DIR = Path("data/eval_suites")
_UA = {"User-Agent": "cliffguard/1.0 (research; contact via repository)"}

BENIGN, HARMFUL = "benign", "harmful"


def _fetch_csv(url: str) -> list[dict[str, str]]:
    """Read a CSV straight from its publishing repository.

    urllib rather than the datasets library: these are single files at stable
    URLs, and adding a dependency that can fail on a library major version for
    the sake of reading one CSV is not a trade worth making.
    """
    import urllib.request

    request = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))


def _hf(dataset: str, config: str | None, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:                          # pragma: no cover
        raise SystemExit(
            "the datasets library is needed for this suite: pip install datasets"
        ) from exc
    return (load_dataset(dataset, config, split=split) if config
            else load_dataset(dataset, split=split))


# ---------------------------------------------------------------------------
# one loader per suite, each returning normalised records
# ---------------------------------------------------------------------------


def _or_bench_hard() -> list[dict[str, str]]:
    return [{"prompt": r["prompt"].strip(), "harm_label": BENIGN,
             "category": r.get("category", "")}
            for r in _hf("bench-llm/or-bench", "or-bench-hard-1k", "train")]


def _or_bench_toxic() -> list[dict[str, str]]:
    return [{"prompt": r["prompt"].strip(), "harm_label": HARMFUL,
             "category": r.get("category", "")}
            for r in _hf("bench-llm/or-bench", "or-bench-toxic", "train")]


def _xstest() -> list[dict[str, str]]:
    """Safe and unsafe prompts, separated by the `type` field.

    XSTest pairs each safe prompt with an unsafe contrast, and marks the latter
    by prefixing the type with `contrast_`. That prefix is the only thing
    distinguishing them in the released data, so it is asserted rather than
    assumed: if the schema ever changes, this must fail loudly rather than
    silently label 200 harmful prompts benign.
    """
    rows = list(_hf("natolambert/xstest-v2-copy", "default", "prompts"))
    out = []
    for row in rows:
        kind = str(row.get("type", ""))
        if not kind:
            raise SystemExit("xstest rows have no `type` field; the safe/unsafe "
                             "split cannot be recovered and labelling would be "
                             "guesswork")
        harmful = kind.startswith("contrast_")
        out.append({"prompt": str(row["prompt"]).strip(),
                    "harm_label": HARMFUL if harmful else BENIGN,
                    "category": kind})
    n_harm = sum(r["harm_label"] == HARMFUL for r in out)
    if not (100 <= n_harm <= 0.6 * len(out)):
        raise SystemExit(
            f"xstest split looks wrong: {n_harm} of {len(out)} marked harmful. "
            "The published set is 250 safe against 200 unsafe; a split far from "
            "that means the `type` convention has changed.")
    return out


def _strongreject() -> list[dict[str, str]]:
    rows = _fetch_csv("https://raw.githubusercontent.com/alexandrasouly/"
                      "strongreject/main/strongreject_dataset/"
                      "strongreject_dataset.csv")
    return [{"prompt": r["forbidden_prompt"].strip(), "harm_label": HARMFUL,
             "category": r.get("category", "")} for r in rows]


def _advbench() -> list[dict[str, str]]:
    rows = _fetch_csv("https://raw.githubusercontent.com/llm-attacks/"
                      "llm-attacks/main/data/advbench/harmful_behaviors.csv")
    return [{"prompt": r["goal"].strip(), "harm_label": HARMFUL, "category": ""}
            for r in rows]


def _harmbench() -> list[dict[str, str]]:
    """Standard behaviours only -- see the module docstring for why."""
    rows = _fetch_csv("https://raw.githubusercontent.com/centerforaisafety/"
                      "HarmBench/main/data/behavior_datasets/"
                      "harmbench_behaviors_text_all.csv")
    return [{"prompt": r["Behavior"].strip(), "harm_label": HARMFUL,
             "category": r.get("SemanticCategory", "")}
            for r in rows if r.get("FunctionalCategory") == "standard"]


SUITES: dict[str, tuple[Callable[[], list[dict[str, str]]], str, str]] = {
    "or-bench-hard":  (_or_bench_hard,  BENIGN,  "bench-llm/or-bench (HF)"),
    "or-bench-toxic": (_or_bench_toxic, HARMFUL, "bench-llm/or-bench (HF)"),
    "xstest":         (_xstest,         "mixed", "natolambert/xstest-v2-copy (HF)"),
    "strongreject":   (_strongreject,   HARMFUL, "alexandrasouly/strongreject (GitHub)"),
    "advbench":       (_advbench,       HARMFUL, "llm-attacks/llm-attacks (GitHub)"),
    "harmbench":      (_harmbench,      HARMFUL, "centerforaisafety/HarmBench (GitHub)"),
}


# ---------------------------------------------------------------------------


def _write(path: Path, records: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build(suite: str, out_dir: Path) -> dict[str, Any]:
    """Fetch one suite, normalise it, drop duplicates, and write it out.

    Deduplication is within a suite only. A prompt appearing in two suites is
    kept in both, because the suites are analysed separately and silently
    dropping it from one would make that suite's published size wrong.

    Order is the source's own, never shuffled: every run in this project takes
    the first n prompts of a file, so a shuffle would mean two runs at different
    n could not be compared.
    """
    loader, _, provenance = SUITES[suite]
    records = loader()

    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for record in records:
        prompt = record["prompt"]
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        record["source"] = suite
        record["suite_id"] = f"{suite}:{len(unique)}"
        unique.append(record)

    if not unique:
        raise SystemExit(f"{suite}: fetched nothing usable")

    path = out_dir / f"{suite}.jsonl"
    digest = _write(path, unique)
    counts = {
        label: sum(r["harm_label"] == label for r in unique)
        for label in (BENIGN, HARMFUL)
    }
    return {"suite": suite, "path": str(path), "n": len(unique),
            "n_dropped_duplicate": len(records) - len(unique),
            "counts": counts, "provenance": provenance, "sha256_16": digest}


# Paired corpora, each a harmful suite joined to a benign one.
#
# The 2x2 needs both prompt classes in ONE run, because the comparison is paired
# against that run's own FP16 baseline. Only XSTest carries both, so every other
# suite would otherwise need a run of its own that can fill only half the table.
# Joining them lets a single ladder produce a complete matrix.
#
# READ THIS BEFORE CHOOSING ONE. A pairing takes its harmful prompts from one
# suite and its benign prompts from another, so prompt class is PERFECTLY
# CONFOUNDED with authorship: every harmful prompt is a StrongREJECT prompt and
# every benign one is an XSTest prompt. Any difference between the classes is
# therefore also a difference between two research groups' construction
# procedures, and cannot be attributed to harmfulness.
#
# That does NOT affect what this project primarily measures, because the paired
# transitions are within-prompt -- the same prompt at FP16 and at a rung -- and a
# confound constant within a prompt cancels. It DOES affect any comparison
# ACROSS the classes, including the full-precision baseline rates that
# `analyse_labelled` prints side by side.
#
# XSTest is the only suite whose two halves were built together as matched
# contrasts, so it is the one corpus here where class is not confounded with
# source. Prefer it when both classes will be compared; use a pairing when more
# prompts of one class are needed than XSTest provides.
#
# An earlier version of this comment claimed the opposite -- that drawing the
# halves from different groups avoided a confound. It creates one.
PAIRS: dict[str, tuple[str, str]] = {
    "paired-harmbench-orbench": ("harmbench", "or-bench-hard"),
    "paired-advbench-orbench": ("advbench", "or-bench-hard"),
    "paired-strongreject-xstest": ("strongreject", "xstest"),
}


def build_pair(name: str, out_dir: Path) -> dict[str, Any]:
    """Join a harmful suite and a benign one into a single labelled corpus.

    Both source files must already exist. Only the prompts of the wanted class
    are taken from each side, so pairing StrongREJECT with XSTest contributes
    XSTest's benign half and none of its harmful half -- otherwise the harmful
    class would be a mixture of two sources while the benign class was one, and
    a difference between the classes could be a difference between authors.

    Order is interleaved by the loader at read time, not here, so the file stays
    a plain concatenation and its provenance is obvious on inspection.
    """
    harmful_suite, benign_suite = PAIRS[name]
    rows: list[dict[str, str]] = []
    # Deduplication in `build` is per suite. Across two suites it has never run,
    # and the collision it would miss is the worst one available here: the same
    # text present in both halves, carrying opposite labels. Every count that
    # followed would be computed on a corpus that contradicts itself, and
    # nothing would raise. Cheap to prevent, so prevented.
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for suite, wanted in ((harmful_suite, HARMFUL), (benign_suite, BENIGN)):
        path = out_dir / f"{suite}.jsonl"
        if not path.exists():
            raise SystemExit(
                f"{name} needs {path}, which has not been downloaded. Fetch the "
                f"component suites first: --download --suites {harmful_suite} "
                f"{benign_suite}")
        kept = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("harm_label") != wanted:
                continue
            prompt = row["prompt"]
            if prompt in seen:
                if seen[prompt] != wanted:
                    collisions.append(prompt)
                continue
            seen[prompt] = wanted
            # Keep where it came from. `source` is overwritten with the suite
            # and `suite_id` renumbered for this corpus, so without this the
            # original identifier -- the only handle back to the publisher's own
            # numbering -- would be lost.
            row["origin_suite_id"] = row.get("suite_id", "")
            row["source"] = suite
            row["suite_id"] = f"{name}:{len(rows)}"
            rows.append(row)
            kept += 1
        if not kept:
            raise SystemExit(
                f"{name}: {suite} contributed no {wanted} prompts. The pairing "
                "would produce a single-class corpus, which cannot fill the 2x2.")

    if collisions:
        raise SystemExit(
            f"{name}: {len(collisions)} prompt(s) appear in BOTH halves with "
            f"opposite labels, e.g. {collisions[0][:90]!r}. The corpus would "
            "contradict itself and every count computed from it would be "
            "meaningless. The two suites disagree and a human has to decide "
            "which is right.")

    path = out_dir / f"{name}.jsonl"
    digest = _write(path, rows)
    counts = {BENIGN: sum(r["harm_label"] == BENIGN for r in rows),
              HARMFUL: sum(r["harm_label"] == HARMFUL for r in rows)}
    return {"suite": name, "path": str(path), "n": len(rows),
            "n_dropped_duplicate": 0, "counts": counts,
            "provenance": f"{harmful_suite} (harmful) + {benign_suite} (benign)",
            "components": [harmful_suite, benign_suite],
            # Recorded in the manifest so an analysis, or a reader, can see the
            # confound without reconstructing it from the component names.
            "class_confounded_with_source": True,
            "sha256_16": digest}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="actually fetch. Without it, this only lists the suites.")
    ap.add_argument("--suites", nargs="*", default=sorted(SUITES),
                    choices=sorted(SUITES), metavar="SUITE")
    ap.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    ap.add_argument("--pairs", nargs="*", default=sorted(PAIRS),
                    choices=sorted(PAIRS), metavar="PAIR",
                    help="paired harmful+benign corpora to assemble after "
                         "fetching. Each gives a complete 2x2 from ONE ladder; "
                         "pass none to skip.")
    args = ap.parse_args(argv)

    if not args.download:
        print("Evaluation suites available (pass --download to fetch):\n")
        print(f"  {'suite':16s} {'label':8s} source")
        print("  " + "-" * 68)
        for name in sorted(SUITES):
            _, label, provenance = SUITES[name]
            print(f"  {name:16s} {label:8s} {provenance}")
        print(f"\nOutput directory: {args.out_dir.resolve()}")
        print("\nEach line is one JSON object:")
        print('  {"prompt": ..., "harm_label": "benign"|"harmful", '
              '"category": ..., "source": ..., "suite_id": ...}')
        return 0

    manifest: list[dict[str, Any]] = []
    failed: list[str] = []
    for suite in args.suites:
        print(f"[{suite}] fetching ...", flush=True)
        try:
            entry = build(suite, args.out_dir)
        except Exception as exc:                        # noqa: BLE001
            # One unreachable suite must not cost the others. Which suites are
            # present is recorded, so an analysis can never quietly assume a
            # suite it does not have.
            print(f"[{suite}] FAILED ({type(exc).__name__}: {str(exc)[:120]})",
                  file=sys.stderr)
            failed.append(suite)
            continue
        manifest.append(entry)
        dropped = entry["n_dropped_duplicate"]
        print(f"[{suite}] {entry['n']} prompts "
              f"({entry['counts'][BENIGN]} benign / {entry['counts'][HARMFUL]} harmful)"
              + (f", {dropped} duplicate(s) dropped" if dropped else ""),
              flush=True)

    if not manifest:
        raise SystemExit("no suite could be built; check network access")

    # Pairs are assembled from the files just written, so a suite that failed
    # above simply removes the pairs that needed it rather than failing the run.
    have = {e["suite"] for e in manifest}
    for pair in args.pairs:
        needs = set(PAIRS[pair])
        if not needs <= have:
            print(f"[{pair}] skipped: needs {sorted(needs - have)}",
                  file=sys.stderr)
            continue
        entry = build_pair(pair, args.out_dir)
        manifest.append(entry)
        print(f"[{pair}] {entry['n']} prompts "
              f"({entry['counts'][BENIGN]} benign / {entry['counts'][HARMFUL]} "
              f"harmful) from {entry['provenance']}", flush=True)

    manifest_path = args.out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(
        {"suites": manifest, "failed": failed}, indent=2), encoding="utf-8")

    total_benign = sum(e["counts"][BENIGN] for e in manifest)
    total_harmful = sum(e["counts"][HARMFUL] for e in manifest)
    print(f"\n{'suite':16s} {'n':>6s} {'benign':>7s} {'harmful':>8s}  sha256")
    print("-" * 60)
    for entry in manifest:
        print(f"{entry['suite']:16s} {entry['n']:6d} {entry['counts'][BENIGN]:7d} "
              f"{entry['counts'][HARMFUL]:8d}  {entry['sha256_16']}")
    print("-" * 60)
    print(f"{'total':16s} {total_benign + total_harmful:6d} {total_benign:7d} "
          f"{total_harmful:8d}")
    if failed:
        print(f"\nunavailable: {failed}")
    print(f"\nwrote {manifest_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
