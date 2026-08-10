"""Work out which per-prompt judge cache came from which label scorer.

A graded run stores one file per scheme, named `judge_<fingerprint>_<scheme>`,
and the fingerprint is a hash of the grading policy. That makes two gradings of
the same text distinguishable, which is the point -- but the files themselves
are bare verdict lists with no self-description, so once a run carries three of
them nothing on disk says which is which.

The fingerprint is recoverable rather than lost. It is a hash of a small dict
whose fields are all either fixed constants of the grader or recorded in the
run, so the mapping can be recomputed on a CPU in milliseconds and checked
against the filenames actually present. That is what this does.

Three scorers appear in this project:

  first-token-legacy   the published grading. Its identity predates the
                       `scoring` key, so the key is absent from the hash.
  first-token          the same measurement, re-run in round 3 under the
                       identity that names the scorer. Byte-identical verdicts
                       to the legacy cache wherever both exist, which is how we
                       know the judge itself is deterministic.
  letter               the corrected scorer. Under Qwen2.5, REFUSE/COMPLY/
                       UNCLEAR tokenize to three-character prefixes while
                       DISCLAIM is a whole word, so first-token logits over
                       those strings are not comparable quantities. The letter
                       template scores five verified single-token options.

Resolution is by construction, not by guessing: a fingerprint is only reported
when the recomputed hash equals a filename on disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

# Kept in step with scripts/classify_completions_judge.py. A drift between the
# two would make this resolver quietly stop matching anything, so the test
# suite imports both and asserts they agree rather than trusting the copy.
LABELS = ("REFUSE", "COMPLY", "UNCLEAR")
DEFAULT_JUDGE = "Qwen/Qwen2.5-7B-Instruct"

MODES = ("first-token-legacy", "first-token", "letter")


def _identity(judge: str, four_bit: bool, template: str, scoring: str | None,
              n_prompts: int, completion_chars: int | None) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "judge": judge,
        "four_bit": four_bit,
        "labels": list(LABELS),
        "template": template,
        "n_prompts": n_prompts,
    }
    # Absent, not None: a sentinel is still a change to the hashed object and
    # would fail to reproduce the hashes already on disk.
    if scoring is not None:
        identity["scoring"] = scoring
    if completion_chars is not None:
        identity["completion_chars"] = completion_chars
    return identity


def fingerprint(identity: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _judge_module():
    """The three-way grader, imported from its own file.

    Loaded by path rather than by package import because scripts/ is not a
    package. Copying the template text here instead would create exactly the
    silent-drift failure this module is trying to prevent.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "classify_completions_judge.py"
    spec = importlib.util.spec_from_file_location("_cliffguard_judge_templates", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise RuntimeError(f"cannot load grader templates from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _templates(letter_order: Sequence[str] | None = None) -> tuple[str, str]:
    """The two prompt templates. `letter_order` permutes the multiple-choice one.

    Passing an order here is how a grading made under a permuted assignment is
    located: the order changes the template, the template changes the
    fingerprint, and the fingerprint is the filename. Without it such a file is
    simply unresolvable, which is safe but not useful.
    """
    module = _judge_module()
    letters = (module.letter_template(letter_order) if letter_order
               else module.LETTER_TEMPLATE)
    return module.JUDGE_TEMPLATE, letters


def present_fingerprints(run: Path) -> dict[str, set[str]]:
    """Fingerprints on disk, per scheme, from the per-prompt cache filenames.

    Includes the five-way grader's collapsed caches, whose names carry a
    `collapsed` prefix on the fingerprint; the prefix is kept, because a
    collapsed three-way view of a five-way grader is a different instrument
    from the three-way grader and must not resolve to the same key.
    """
    out: dict[str, set[str]] = {}
    for path in sorted(run.glob("results/judge_*_*.json")):
        if path.name == "judge_classification.json":
            continue
        parts = path.stem.split("_", 2)
        if len(parts) != 3:
            continue
        _, token, scheme = parts
        out.setdefault(scheme, set()).add(token)
    return out


def resolve(run: Path, judge: str | None = None, four_bit: bool = True,
            completion_chars: int | None = None,
            letter_order: Sequence[str] | None = None) -> dict[str, str]:
    """Map scorer mode -> fingerprint, for the modes actually present.

    `completion_chars` is included in the hash only when the window bound; the
    caller passes it when it did. Both possibilities are tried, so a run graded
    under a binding window still resolves.

    `letter_order` selects WHICH letter grading is being looked for. The
    default is the published assignment; an option-order replicate is found by
    passing its own order, and the two never resolve to the same digest.
    """
    judge_template, letter_template = _templates(letter_order)
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_prompts = manifest.get("n_prompts")
    if not isinstance(n_prompts, int):
        prompts = run / "results" / "prompts.json"
        if not prompts.is_file():
            return {}
        payload = json.loads(prompts.read_text(encoding="utf-8"))
        n_prompts = len(payload["prompts"] if isinstance(payload, dict) else payload)

    on_disk: set[str] = set()
    for tokens in present_fingerprints(run).values():
        on_disk |= tokens

    windows = [None] if completion_chars is None else [None, completion_chars]
    candidates = {
        "first-token-legacy": (judge_template, None),
        "first-token": (judge_template, "first-token"),
        "letter": (letter_template, "letter"),
    }

    found: dict[str, str] = {}
    for mode, (template, scoring) in candidates.items():
        for window in windows:
            digest = fingerprint(_identity(
                judge or DEFAULT_JUDGE, four_bit, template, scoring,
                n_prompts, window))
            if digest in on_disk:
                found[mode] = digest
                break
    return found


def cache_files(run: Path, digest: str) -> dict[str, Path]:
    """The per-scheme cache files carrying one fingerprint."""
    out: dict[str, Path] = {}
    for path in sorted(run.glob(f"results/judge_{digest}_*.json")):
        out[path.stem.split("_", 2)[-1]] = path
    return out


# --------------------------------------------------------------------------
# The five-way taxonomy grader
#
# Its fingerprint hashes the graded text itself rather than a prompt count, so
# it cannot be recomputed from metadata alone -- but it can be recomputed
# exactly from the run, which is stronger. Everything below reconstructs the
# hash from the stored prompts and completions and matches it against the
# filenames, so a mode is only ever reported when the arithmetic agrees.


def _taxonomy_module():
    import importlib.util

    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "classify_completion_taxonomy.py")
    spec = importlib.util.spec_from_file_location("_cliffguard_taxonomy", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise RuntimeError(f"cannot load taxonomy grader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def taxonomy_content_hash(prompts: list[str],
                          completions: dict[str, list[str]],
                          schemes: list[str]) -> str:
    """Length-prefixed digest of exactly what the judge was asked about.

    Length-prefixed rather than delimiter-separated for the reason the grader
    gives: no byte is guaranteed absent from model output, so a separator can
    be forged by the data and two different (prompt, completion) splits could
    hash alike.
    """
    digest = hashlib.sha256()

    def absorb(text: str) -> None:
        blob = text.encode("utf-8", "replace")
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)

    for scheme in schemes:
        # zip() stops at the shorter side, so a completion list with extra or
        # missing rows would hash as though those rows did not exist and could
        # collide with a correct run. The fingerprint is a truncated 64 bits
        # and cannot be made collision-proof, but it must at least not be
        # handed a silently truncated input.
        if len(completions[scheme]) != len(prompts):
            raise ValueError(
                f"{scheme}: {len(completions[scheme])} completions against "
                f"{len(prompts)} prompts; hashing them would describe a "
                "different run than the one on disk")
        absorb(scheme)
        for prompt, completion in zip(prompts, completions[scheme]):
            absorb(prompt)
            absorb(completion)
    return digest.hexdigest()


def resolve_taxonomy(run: Path, judge: str | None = None, four_bit: bool = True,
                     completion_chars: int = 2000, max_length: int = 2560,
                     batch_size: int = 4,
                     letter_order: Sequence[str] | None = None) -> dict[str, str]:
    """Map 'letter'/'first-token' -> taxonomy fingerprint, by recomputation.

    A run may also carry a legacy cache written before the fingerprint gained
    its `policy` block. Those cannot be reproduced under the current identity
    and are reported separately by `unresolved_taxonomy`, which is honest: the
    resolver will not claim a mapping it cannot derive.
    """
    module = _taxonomy_module()
    results = run / "results"
    prompts_file = results / "prompts.json"
    if not prompts_file.is_file():
        return {}
    prompts = json.loads(prompts_file.read_text(encoding="utf-8"))["prompts"]

    on_disk = {p.stem.split("_", 2)[1] for p in results.glob("taxonomy_*_*.json")}
    schemes = sorted({p.stem.split("_", 2)[-1]
                      for p in results.glob("taxonomy_*_*.json")})
    if not schemes:
        return {}

    found: dict[str, str] = {}
    letters = (module.letter_template(letter_order) if letter_order
               else module.LETTER_TEMPLATE)
    for scoring, template in (("letter", letters),
                              ("first-token", module.TAXONOMY_TEMPLATE)):
        # A grading covers some subset of schemes; the hash is over exactly
        # that subset, so each candidate subset is tried rather than assuming
        # the full ladder.
        for subset in _scheme_subsets(results, on_disk):
            try:
                completions = {
                    s: json.loads((results / f"completions_{s}.json")
                                  .read_text(encoding="utf-8"))["completions"]
                    for s in subset
                }
            except FileNotFoundError:
                continue
            try:
                content = taxonomy_content_hash(prompts, completions, subset)
            except ValueError:
                # A subset whose completions do not line up with the prompts
                # cannot be the one that was graded. Skip it rather than
                # letting a malformed run abort resolution of the others.
                continue
            digest = fingerprint({
                "judge": judge or DEFAULT_JUDGE,
                "four_bit": four_bit,
                "labels": list(module.LABELS),
                "template": template,
                "n_prompts": len(prompts),
                "content": content,
                "policy": {"prompt_chars": module.PROMPT_CHARS,
                           "completion_chars": completion_chars,
                           "max_length": max_length, "padding_side": "left",
                           "batch_size": int(batch_size), "scoring": scoring},
            })
            if digest in on_disk:
                found[scoring] = digest
                break
    return found


def _scheme_subsets(results: Path, on_disk: set[str]) -> list[list[str]]:
    """Candidate scheme sets a grading could have covered, most likely first."""
    by_digest: dict[str, list[str]] = {}
    for path in sorted(results.glob("taxonomy_*_*.json")):
        _, digest, scheme = path.stem.split("_", 2)
        by_digest.setdefault(digest, []).append(scheme)
    # Deduplicate while keeping the smallest sets first: an FP16-only re-grade
    # is the common round-3 case and matching it early avoids hashing the whole
    # eight-rung ladder for nothing.
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for schemes in sorted(by_digest.values(), key=len):
        key = tuple(sorted(schemes))
        if key not in seen:
            seen.add(key)
            out.append(sorted(schemes))
    return out


def unresolved_taxonomy(run: Path, resolved: dict[str, str]) -> set[str]:
    """Taxonomy fingerprints on disk that no current identity reproduces."""
    results = run / "results"
    on_disk = {p.stem.split("_", 2)[1] for p in results.glob("taxonomy_*_*.json")}
    return on_disk - set(resolved.values())
