"""Fold A calibration corpus loaders — see blueprint §12.2.

Fold A provides the benign-prompt calibration set used to fit per-
quantization thresholds (tau_q) for all primitives. The corpus is
drawn from Anthropic-HH (helpful-harmless dialogue) and OASST
(OpenAssistant) as benign sources. Blueprint §12.2 specifies
|C| >= 2000 prompts per quantization scheme.

Data is never stored in the repo — loaders return prompts from
pre-downloaded JSONL files in the data/ directory, which is
gitignored. In Phase A the loaders are stubs operating on synthetic
in-memory corpora. Phase B wires them to real datasets.

Per blueprint §14.4 (Sensitivity Corollary): |C| < 100 inflates the
KS estimation error to O(1/sqrt(|C|)); deployments with |C| < 100
must accept a wider FPR portability band.
"""

import json
import random
from pathlib import Path

from cliffguard.types import QuantScheme

MIN_CALIBRATION_SIZE: int = 2000


def load_jsonl(path: Path, text_field: str = "text") -> list[str]:
    """Read a JSONL file and return the value of text_field from
    each line as a list of strings.
    Raises FileNotFoundError if path does not exist.
    Raises ValueError if a line is not valid JSON or text_field
    is missing from a line."""
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    results: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {lineno} of {path}: {exc}"
            ) from exc
        if text_field not in obj:
            raise ValueError(
                f"Missing field '{text_field}' on line {lineno} of {path}"
            )
        results.append(str(obj[text_field]))
    return results


def load_fold_a(
    data_dir: Path,
    scheme: QuantScheme,
    min_size: int = MIN_CALIBRATION_SIZE,
) -> list[str]:
    """Load the Fold A calibration corpus for a given scheme.
    Looks for data_dir / f"fold_a_{scheme.value.lower()}.jsonl".
    Falls back to data_dir / "fold_a.jsonl" if scheme-specific
    file is absent.
    Raises FileNotFoundError if neither file exists.
    Raises ValueError if corpus size < min_size, with a message
    citing blueprint §14.4 and stating the actual vs required size.
    Returns the list of prompt strings."""
    scheme_path = data_dir / f"fold_a_{scheme.value.lower()}.jsonl"
    fallback_path = data_dir / "fold_a.jsonl"

    if scheme_path.exists():
        chosen = scheme_path
    elif fallback_path.exists():
        chosen = fallback_path
    else:
        raise FileNotFoundError(
            f"No Fold A corpus found for scheme {scheme.value!r}. "
            f"Looked for: {scheme_path}, {fallback_path}"
        )

    corpus = load_jsonl(chosen)
    if len(corpus) < min_size:
        raise ValueError(
            f"Fold A corpus for {scheme.value!r} has {len(corpus)} prompts "
            f"but min_size={min_size} is required. "
            f"See blueprint §14.4: |C| < 100 inflates KS estimation error "
            f"to O(1/sqrt(|C|)); the FPR portability band will be wider."
        )
    return corpus


def make_synthetic_corpus(n: int = 200, seed: int = 42) -> list[str]:
    """Generate a synthetic benign corpus of n short prompts for
    use in scaffolding tests. Prompts are deterministic given seed.
    Does NOT use any LLM — plain string construction only.
    Returns a list of n strings."""
    if n == 0:
        return []
    rng = random.Random(seed)
    topics = [
        "the weather", "a recipe", "history", "science", "mathematics",
        "programming", "literature", "music", "art", "travel",
        "economics", "philosophy", "biology", "chemistry", "physics",
    ]
    templates = [
        "Can you explain {topic}?",
        "What do you know about {topic}?",
        "Tell me something interesting about {topic}.",
        "I'd like to learn more about {topic}.",
        "Please summarise {topic} in a few sentences.",
    ]
    prompts: list[str] = []
    for i in range(n):
        topic = rng.choice(topics)
        template = rng.choice(templates)
        prompts.append(template.format(topic=topic))
    return prompts
