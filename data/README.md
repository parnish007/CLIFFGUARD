# data/

Four files are tracked here. Everything else under `data/` is ignored and is
rebuilt by `scripts/download_fold_a.py` and `scripts/download_eval_suites.py`.

## Why these four are in the repository

A hosted session with no Drive state to restore would otherwise begin by
fetching `Anthropic/hh-rlhf` at an **unpinned** revision and pulling XSTest
through the `datasets` library. Three things outside this repository would then
decide whether the session starts: two Hub endpoints and one wheel. None of them
can produce a wrong result — preflight hashes both corpora before any GPU time
is spent, so a changed upstream is a caught error rather than a quiet
substitution — but each of them can end a session before it begins.

295 KB removes that. The rebuild path is unchanged and still works; it is now
the fallback rather than the only route.

## What they are

| file | rows | derivation |
|---|---|---|
| `folds/fold_a/anthropic_hh_refused.jsonl` | 800 | first human turn of `Anthropic/hh-rlhf` conversations whose assistant reply declines |
| `folds/fold_a/anthropic_hh_benign.jsonl` | 1200 | first human turn of conversations whose assistant reply does not |
| `eval_suites/xstest.jsonl` | 450 | XSTest v2, 250 safe and 200 unsafe, split on the `type` field's `contrast_` prefix |
| `eval_suites/MANIFEST.json` | — | the build record for the above, written by the downloader |

The behavioural runs read the **first 250 of each** Fold A file, interleaved to
500 prompts. The labelled runs read **300 of the 450** XSTest prompts. Those
subsets, not the files, are what the published runs were built from, and they
are what preflight checks.

## Hashes

Whole-file SHA-256, which is what tells you the files below are the ones this
commit shipped:

```
ec622966f00086b406dd735e2e4a2405a51ac52cb2a35f248f973212f11e29bf  folds/fold_a/anthropic_hh_benign.jsonl
31521086bf6eeb2cbb8340a2b8bad23d543a2ec2551019beb1452e66f8de76ac  folds/fold_a/anthropic_hh_refused.jsonl
fc7d21ae9c00371237467c2fa4a47c18b8945917720ea8a0e30b789678d29a0d  eval_suites/xstest.jsonl
```

Preflight checks something different and stronger: the SHA-256 of the **ordered
prompt sequence each run actually consumed**, against the hash recorded in that
run's manifest. Those are
`7da25bf88ee0409ce4900a12052e15849a2898ed01cfdcdfe6409bbfc11bd9b5` for the
500-prompt Fold A pairing and
`33874ac77bd574a74283cd024466f442e69da870fa1195fcde8a9107433f9ce4` for the
300-prompt XSTest slice, both in `scripts/preflight_round2.py`. A file that
hashes correctly but is read in a different order would pass the table above and
fail preflight, which is the right way round.

## Licences

Redistributed with attribution, both permissively licensed.

- **Anthropic HH-RLHF** — MIT. <https://huggingface.co/datasets/Anthropic/hh-rlhf>
- **XSTest** — CC-BY-4.0, obtained via `natolambert/xstest-v2-copy`. Röttger et
  al., *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in
  Large Language Models*, NAACL 2024.

Both are prompt sets only: no model outputs, no annotations of ours, and no
results. Results are published deliberately and are not in this repository.
