"""Re-grade stored completions with a judge from a different model family.

The paper's dependent variable currently rests on one grader: Qwen2.5-7B loaded
in 4-bit, which shares a family with two of the three models it judges, has no
validation against any external label, and is observably wrong on some
completions -- it scored "Stalin was the leader of the Soviet Union from 1922 to
1953" as a refusal. Every headline number inherits that.

A second judge from an unrelated family is the cheapest thing that can move
this. It needs no GPU, because the completions are already stored: grading is a
pure function of text we have on disk, so the whole comparison is an API sweep
over saved artifacts.

Design choices that matter for the comparison to mean anything:

  same template   The judge prompt is imported from the 7B pipeline rather than
                  rewritten, so a disagreement is a grader difference and not a
                  prompt difference.

  same gate       Degeneracy is decided exactly as before and outside this
                  script. We ask the API only about completions the composite
                  gate admits, which is also what keeps the call count down.

  stratified      Grading all 8,000 completions is unnecessary and does not fit
                  a free tier. The claim rests on the coherent band, so the
                  default sweep covers FP16 and the rungs carrying the result.
                  --schemes widens it when quota allows.

  resumable       Every verdict is cached per (provider, model, run, scheme,
                  index). Interrupting the sweep costs nothing, which matters
                  on a 15-requests-per-minute free tier.

Keys are read from the environment or a gitignored .env. They are never logged,
never written to an artifact, and never included in any output file.

Usage:
  python scripts/judge_via_api.py --provider groq
  python scripts/judge_via_api.py --provider gemini --schemes FP16 RTN_4B
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_paper_data import MODEL_LABELS
from scripts.classify_completions_judge import JUDGE_TEMPLATE, LABELS
from scripts.reanalyse_runs import analyse, load_run
from scripts.run_behavioural_ladder import is_degenerate

# The rungs the paper's claims actually rest on. FP16 is mandatory -- every
# result is a paired comparison against it, so grading a rung without it buys
# nothing.
DEFAULT_SCHEMES = ("FP16", "RTN_4B")


class Provider:
    """One hosted judge, reduced to: name a model, send text, get a label."""

    def __init__(self, name: str, env_var: str, url: str, model: str,
                 build: Callable[[str, str], dict[str, Any]],
                 extract: Callable[[dict[str, Any]], str],
                 min_interval: float) -> None:
        self.name = name
        self.env_var = env_var
        self.url = url
        self.model = model
        self.build = build
        self.extract = extract
        # Free tiers are rate-limited far below what the network allows, and
        # being throttled costs more time than pacing does.
        self.min_interval = min_interval


def _openai_style(model: str) -> Callable[[str, str], dict[str, Any]]:
    def build(prompt: str, _model: str = model) -> dict[str, Any]:
        return {
            "model": _model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 4,
        }
    return build


def _openai_extract(payload: dict[str, Any]) -> str:
    return str(payload["choices"][0]["message"]["content"])


def _gemini_build(prompt: str, _model: str = "") -> dict[str, Any]:
    return {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 4},
    }


def _gemini_extract(payload: dict[str, Any]) -> str:
    return str(payload["candidates"][0]["content"]["parts"][0]["text"])


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        "groq", "GROQ_API_KEY",
        "https://api.groq.com/openai/v1/chat/completions",
        "llama-3.3-70b-versatile",
        _openai_style("llama-3.3-70b-versatile"), _openai_extract, 2.1),
    "nvidia": Provider(
        "nvidia", "NVIDIA_API_KEY",
        "https://integrate.api.nvidia.com/v1/chat/completions",
        "meta/llama-3.1-70b-instruct",
        _openai_style("meta/llama-3.1-70b-instruct"), _openai_extract, 1.55),
    "gemini": Provider(
        "gemini", "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-flash-latest:generateContent",
        "gemini-flash-latest",
        _gemini_build, _gemini_extract, 4.1),
    "openrouter": Provider(
        "openrouter", "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1/chat/completions",
        "mistralai/mistral-large",
        _openai_style("mistralai/mistral-large"), _openai_extract, 1.1),
}


def load_env(path: Path = Path(".env")) -> None:
    """Read a gitignored .env without adding a dependency, and without
    overwriting anything already set in the real environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def normalise(raw: str) -> str:
    """Map free text onto the three labels, or UNPARSED.

    A hosted model will not always answer with one bare word, and silently
    folding an unparseable reply into COMPLY or REFUSE would invent agreement.
    UNPARSED is kept distinct and reported.
    """
    upper = raw.strip().upper()
    for label in LABELS:
        if upper.startswith(label) or label in upper:
            return label
    return "UNPARSED"


def call(provider: Provider, key: str, prompt: str, retries: int = 5) -> str:
    body = json.dumps(provider.build(prompt, provider.model)).encode("utf-8")
    url = provider.url
    # urllib announces itself as "Python-urllib/3.x", which Groq's edge blocks
    # with a Cloudflare 1010 before the request reaches the API. The same
    # request from curl succeeds, so the block is on the agent string alone.
    headers = {"Content-Type": "application/json",
               "User-Agent": "cliffguard-judge/1.0",
               "Accept": "application/json"}
    if provider.name == "gemini":
        # Google moved from ?key= to a header; the query form still works but
        # the header is what the current quickstart documents.
        headers["X-goog-api-key"] = key
    else:
        headers["Authorization"] = f"Bearer {key}"

    delay = provider.min_interval
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return normalise(provider.extract(json.loads(response.read())))
        except urllib.error.HTTPError as exc:
            # 429 and 5xx are worth waiting out; a 4xx is a bug in the request
            # and retrying it just burns quota.
            if exc.code != 429 and exc.code < 500:
                detail = exc.read().decode("utf-8", "replace")[:200]
                raise SystemExit(
                    f"{provider.name} returned {exc.code}: {detail}") from exc
            time.sleep(delay)
            delay *= 2
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError):
            time.sleep(delay)
            delay *= 2
    return "UNPARSED"


def gradable_indices(run: dict[str, Any], scheme: str) -> list[int]:
    """Completions the composite gate admits. Degenerate text is not sent: the
    gate's verdict does not depend on the grader, so paying an API call for it
    would only add cost and a chance of the judge inventing a label."""
    threshold = analyse(run)["threshold"]
    texts, values = run["completions"][scheme], run["nll"][scheme]
    return [i for i, (t, v) in enumerate(zip(texts, values))
            if not is_degenerate(t, float(v), threshold)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    ap.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    ap.add_argument("--schemes", nargs="*", default=list(DEFAULT_SCHEMES))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap completions per scheme, for a first smoke sweep")
    ap.add_argument("--model", default=None, help="override the provider's model")
    args = ap.parse_args()

    load_env()
    provider = PROVIDERS[args.provider]
    if args.model:
        provider.model = args.model
        provider.build = _openai_style(args.model) if args.provider != "gemini" \
            else _gemini_build
    key = os.environ.get(provider.env_var, "").strip()
    if not key:
        raise SystemExit(
            f"{provider.env_var} is not set. Put it in .env (gitignored) or "
            f"export it. See .env.example.")

    tag = f"{provider.name}_{provider.model}".replace("/", "-").replace(".", "-")
    total_calls = 0

    for run_dir in sorted(args.runs.iterdir()):
        if not run_dir.is_dir():
            continue
        run = load_run(run_dir)
        if run is None or not run["judge_raw"]:
            continue
        model = MODEL_LABELS.get(run["manifest"].get("model_id", "?"))
        if model is None or len(run["completions"]["FP16"]) < 150:
            continue

        prompts = json.loads(
            (run_dir / "results" / "prompts.json").read_text(encoding="utf-8"))
        prompts = prompts.get("prompts", prompts) if isinstance(prompts, dict) \
            else prompts

        # Prompt-major, not scheme-major. The order matters more than it looks.
        #
        # Every claim this sweep can support is PAIRED: a prompt the baseline
        # declined and the rung answered, or the reverse. A prompt graded at one
        # scheme and not the other contributes nothing. Grading scheme by scheme
        # means a sweep that dies -- on quota, on a rate limit, on a session
        # ending -- has spent its entire budget on one side of every pair.
        #
        # That is not hypothetical. It is what the published external sweeps
        # look like: one grader returned 482 full-precision verdicts against 26
        # at the rung, so 508 calls bought a comparison on 26 prompts. Another
        # bought 5. Interleaving makes any prefix of the sweep a complete
        # paired sample of the prompts it reached, so an interrupted sweep is a
        # smaller experiment rather than a broken one.
        schemes = [s for s in args.schemes if s in run["completions"]]
        if not schemes:
            continue
        files = {s: run_dir / "results" / f"judge_api_{tag}_{s}.json"
                 for s in schemes}
        verdicts: dict[str, dict[str, str]] = {
            s: (json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})
            for s, p in files.items()}

        # A prompt the gate rejects at either end cannot be paired at all, so
        # spending a call on its other end buys nothing either.
        gradable = set.intersection(
            *(set(gradable_indices(run, s)) for s in schemes))
        todo = [i for i in sorted(gradable)
                if any(str(i) not in verdicts[s] for s in schemes)]
        if args.limit:
            todo = todo[: args.limit]
        cached = min(len(verdicts[s]) for s in schemes)
        print(f"[{model}] {len(schemes)} schemes, {cached} prompts fully "
              f"cached, {len(todo)} to grade", flush=True)

        def flush() -> None:
            for scheme in schemes:
                files[scheme].write_text(
                    json.dumps(verdicts[scheme], sort_keys=True),
                    encoding="utf-8")

        for n, i in enumerate(todo, 1):
            for scheme in schemes:
                if str(i) in verdicts[scheme]:
                    continue
                prompt = JUDGE_TEMPLATE.format(
                    prompt=prompts[i][:600],
                    completion=run["completions"][scheme][i][:600] or "(empty)")
                verdicts[scheme][str(i)] = call(provider, key, prompt)
                total_calls += 1
                time.sleep(provider.min_interval)
            # Every prompt, not every 25. A crash between writes loses a whole
            # batch, and losing it asymmetrically is the failure this ordering
            # exists to prevent; the cost is one small write per prompt against
            # several seconds of rate-limited waiting.
            flush()
            if n % 25 == 0 or n == len(todo):
                print(f"   {n}/{len(todo)} prompts", flush=True)

    print(f"\n{total_calls} API calls. Verdicts written beside the 7B judge's.")
    print("Next: python scripts/compare_judges.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
