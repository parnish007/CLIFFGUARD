<div align="center">

# CliffGuard

**A degeneration-aware protocol for measuring what quantization does to refusal —
and evidence that the reported safety cliff is largely an artifact of how refusal is scored.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![mypy strict](https://img.shields.io/badge/mypy-strict-brightgreen?style=flat-square)](https://mypy-lang.org/)
[![ruff](https://img.shields.io/badge/ruff-passing-orange?style=flat-square)](https://docs.astral.sh/ruff/)
[![License MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

**[Paper](docs/paper/cliff_artifact.pdf) · [Claims ledger](docs/claims_and_evidence.md) · [Reproduce](#reproducing-the-paper) · [Setup](docs/setup.md)**

</div>

---

## The finding

Post-training quantization is widely reported to erode refusal. Measuring it on
generated text instead of on phrase matching reverses the direction of the effect.

Round-to-nearest ladder from 8 down to 2 bits per parameter, 500 paired prompts
per rung, graded by a 7B judge behind a degeneracy gate:

| | result |
|---|---|
| **Refusal drift** | Lower precision produces **more** refusal: **1.15 points per bit removed**, 95% CI [0.75, 1.55] |
| **Refusal→compliance** | Never above **2.2%**; simultaneous 95% upper bound **4.62%** across all 14 model×rung cells |
| **Phrase-list scoring** | Reports up to **38.4%** on the same completions, and is **non-monotone** in the marker list |
| **Frozen refusal probe** | Retains **96–100%** across 8.5–4.5 bits, spanning both models' significant 4.5-bit shifts; falls to 63% at Phi's significant 3.5-bit shift |
| **Capability** | Collapses at a bit-width differing by a **full bit** between model families |

Three regimes, not one cliff:

```
FP16 ─────── 8.5 bits        8.5 ────── 4.5/3.5 bits        below
    inert                        drift                     collapse
+0.2 to +0.4 pp             ~1.15 pp per bit          output degenerates,
  observed shift               refusal rises           no rate is meaningful
```

The headline artifact needs **two** failures, and they dominate in different
places. Where output is coherent, the perplexity-only and composite gates admit
identical completions and the whole gap is the **grader** — 10.8% vs 2.2% at 4.5
bits on Qwen2.5-3B. Where output has collapsed, the **gate** dominates — 38.4%
under a perplexity-only gate vs 0.2% under the composite gate, same marker list,
same completions, which is 99.5% of that gap.

## The protocol

CliffGuard is an evaluation method — not a quantizer, not a model, not a released
benchmark. Its defining commitment is that **degeneration is a third outcome,
decided before refusal**, because the conventional binary rule assigns incoherent
output to *compliance* by default, and on each RTN ladder we tested the lowest
rung is exactly that.

1. **Pair** — same prompts, full-precision and quantized, decoding held fixed.
2. **Gate** — decide degeneration first, using more than perplexity. Report the rate.
3. **Grade** — label survivors semantically; keep `unclear` as its own state.
4. **Transition** — report both directions separately, with gradable-pair counts,
   paired tests, and multiplicity correction.
5. **Stress the scorer** — re-score under alternative marker lists *and* an
   alternative gate; publish the spread.
6. **Separate capability from fluency** — score a gold-labelled task with no
   classifier in the loop.

Why step 2 is not optional: on Qwen2.5-3B at 2.5 bits the median NLL is **4.13**,
*lower* than the partially-coherent 3.5-bit rung's **5.80**. Repetition loops are
high-likelihood by construction, so a perplexity-only filter ranks the two rungs
backwards — it flags 15.6% of 2.5-bit completions where the composite detector
flags 99.8%.

Why step 5 is not optional: the phrase-list estimator is a paired flip,
`1[FP16 has a marker] AND 1[rung has none]`. Enlarging the marker list pushes
those two indicators in **opposite** directions, so the product is not monotone
in the list. Enlarging a marker list therefore carries no structural guarantee of
moving the estimate toward any target.

## Reproducing the paper

`data.json`, `review_stats.json`, every table and every figure are generated
from the stored run directories. Numbers quoted in the paper's prose are typed,
so `scripts/check_paper_numbers.py` re-derives the load-bearing ones from
`review_stats.json` and fails if the manuscript disagrees.

```bash
python scripts/build_paper_data.py                       # runs -> data.json
python scripts/review_reanalysis.py --gsm8k <test.jsonl> # runs -> review_stats.json
python scripts/build_paper_figures.py                    # -> docs/paper/figures/
python scripts/build_paper_tables.py                     # -> docs/paper/tables/
cd docs/paper && latexmk -pdf cliff_artifact.tex
```

To regenerate the measurements themselves:

```bash
python scripts/run_behavioural_ladder.py --model <hf-id> --n 500
python scripts/run_sector_ladder.py      --model <hf-id> --n 200
python scripts/classify_completions_judge.py <run-dir> \
       --judge-model Qwen/Qwen2.5-7B-Instruct --judge-4bit
python scripts/analyse_probe_transfer.py <run-dir>
```

Everything runs on a 6 GB GPU except the 7B judge and the 3B/3.8B models; those
use [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) (T4, ~2.5 h). Raw
completions are stored verbatim at every rung, so any grader can be re-applied
without regenerating text. **The completions are the data; the labels are an
opinion about them.**

`review_reanalysis.py` holds the inferential work: prompt-level cluster bootstrap
for the drift coefficient, exact McNemar with Holm correction for both the
refusal and capability arms, one-sided and simultaneous exact upper bounds, the
gate × grader decomposition, and the frozen-probe fit/score splits.

GSM8K questions are not redistributed here. The analysis records a SHA-256 of the
question file and of the ordered gold vector it consumed, both reproducible from
the canonical test split by taking the first 200 items with a parseable gold
answer, in file order.

## Repository layout

| Path | Purpose |
|---|---|
| `docs/paper/` | The paper, its generated figures and tables, the verified bibliography |
| `docs/claims_and_evidence.md` | Every claim with its evidence status |
| `scripts/run_behavioural_ladder.py` | Generation + degeneracy detection |
| `scripts/classify_completions_judge.py` | 7B judge grading + the four marker lists |
| `scripts/review_reanalysis.py` | All inferential statistics |
| `scripts/run_sector_ladder.py` | GSM8K capability arm |
| `scripts/analyse_probe_transfer.py` | Frozen vs refit probe |
| `scripts/build_paper_*.py` | Data, figure and table generation |
| `scripts/check_paper_numbers.py` | Asserts the paper's prose numbers match the measurements |
| `notebooks/colab_run.ipynb` | Hands-off hosted-GPU runner |
| `cliffguard/` | Library package (quantizers, discriminability, degeneracy) |
| `tests/` | pytest suite |
| `artifacts/runs/` | Immutable run directories |

## Status

![Tests](https://img.shields.io/badge/tests-1233_passing-brightgreen?style=flat-square)
![mypy](https://img.shields.io/badge/mypy-strict-brightgreen?style=flat-square)
![ruff](https://img.shields.io/badge/ruff-clean-orange?style=flat-square)
![Paper](https://img.shields.io/badge/paper-17pp,_zero_overfull-blue?style=flat-square)

## What this does not establish

The prompt set carries no per-prompt harmfulness annotation, so a
refusal→compliance transition is a change in the model's own decision, not an
observation of harmful content — the paper says "refusal-to-compliance
transition" and never "harmful compliance".

The judge is unvalidated against blinded human labels and is itself quantized to
4-bit; that validation is the single largest piece of work this still needs.
Completions are greedy and 48 tokens, which favours observing refusal intact.
One quantizer family (RTN), one reasoning benchmark, English prompts, three
models — two of them in the refusal arm. Nothing here establishes behaviour under
AWQ, GPTQ, or GGUF k-quants.

## Citation

Trilochan Sharma, *CliffGuard: Degeneration-Aware Evaluation Finds That Low-Bit
Quantization Increases Refusal, While Phrase-List Scoring Reports the Opposite*,
2026.
