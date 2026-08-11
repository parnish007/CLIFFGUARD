<div align="center">

[← README](../README.md) &nbsp;|&nbsp;
**Docs index** &nbsp;|&nbsp;
[Claims ledger](claims_and_evidence.md) &nbsp;|&nbsp;
[Setup](setup.md)

</div>

# Documentation

| document | what it is for |
|---|---|
| [`methodology.md`](methodology.md) | **The single reference.** What is measured and why, the protocol constants, decisions we reversed (so they are not re-litigated), invariants, and debugging by symptom. |
| [`claims_and_evidence.md`](claims_and_evidence.md) | Every claim, its evidence, and — as importantly — the claims that are **not** made. Start here if you want to know what this project asserts. |
| [`data_provenance.md`](data_provenance.md) | Which run produced which number, across three experiments run months apart. Read before comparing any two figures — the rounds differ in corpus, grader, class count and bit labelling, and each difference moves numbers. |
| [`round3_runbook.md`](round3_runbook.md) | The operator's page for a hosted round: where every input comes from, what to check before starting, what to do with the archive. Written for round 3; the same shape applies to rounds 4 and 5. |
| [`preregistration_round5.md`](preregistration_round5.md) | The round-5 protocol, **frozen before the round was run**. Two confirmatory hypotheses with thresholds a null result can fail, two exploratory ones labelled as such, and the measurement decisions inherited unchanged from earlier rounds. `scripts/analyse_deployed.py` scores it by arithmetic. |
| [`setup.md`](setup.md) | Which setup guide you want, and how to re-analyse stored runs with no GPU. |
| [`setup_local_gpu.md`](setup_local_gpu.md) | What actually stops a local run, measured: host RAM, not VRAM. Includes the working torch/transformers pin. |
| [`setup_colab.md`](setup_colab.md) | Hosted T4, for the 3B/3.8B models and the 7B judge, which do not fit 6 GB. |
| [`second_judge.md`](second_judge.md) | Adding an independent grader, and why the paper's headline needed one. |

## What is not here

**The manuscript.** `docs/paper/` holds the paper, its figures, its tables and
the consolidated measurement files (`data.json`, `review_stats.json`,
`judge_agreement.json`), and none of it is tracked. Results are released
deliberately, not as a side effect of a commit.

That is not a gap in the reproduction path. Everything in `docs/paper/` is
*generated*, and the code that generates it is published:

```bash
python scripts/build_paper_data.py                       # runs -> data.json
python scripts/review_reanalysis.py --gsm8k <test.jsonl> # runs -> review_stats.json
python scripts/build_paper_figures.py                    # -> docs/paper/figures/
python scripts/build_paper_tables.py                     # -> docs/paper/tables/
```

Run those against your own `artifacts/runs` and you get the same files. The one
thing you cannot regenerate without the run directories is the measurements
themselves, which is what `scripts/run_behavioural_ladder.py` and
`scripts/run_sector_ladder.py` produce.

**Earlier design documents.** This project began as a prompt-injection defence
with a gate-stack architecture, and the theory, architecture and preregistration
documents describing it are kept locally but not published: no result here
depends on that design, and 54 of its 60 library modules were unreachable from
the pipeline the paper actually uses. Publishing them would misrepresent what
this code is.
