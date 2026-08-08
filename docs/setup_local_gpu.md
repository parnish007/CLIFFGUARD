<div align="center">

[← Docs index](README.md) &nbsp;|&nbsp;
[Setup](setup.md) &nbsp;|&nbsp;
[Colab](setup_colab.md)

</div>

# Running models locally, and what actually stops you

Measured on an RTX 3050 6 GB laptop, 2026-08-08. The conclusion is short:
**the GPU is not the constraint — host RAM is.**

## What was measured

| check | result |
|---|---|
| CUDA available, fp16 matmul | ✅ works |
| VRAM free | **5.35 GB of 6.44 GB** |
| Host RAM free | **1.8 GB of 16.8 GB** (89% load) |
| SmolLM2-135M load + generate | ✅ 0.27 GB VRAM |
| Qwen2.5-0.5B load | ❌ **segfault** |
| Full ladder, FP16 rung, 24 prompts | ✅ 8 s, 0.24 s/prompt |
| Full ladder, second rung | ❌ segfault on the second `from_pretrained` |

The segfault reproduces with **plain `transformers.from_pretrained` and no code
from this repository**, so it is not a defect here. `from_pretrained`
materialises the checkpoint in host RAM before it reaches the GPU, and a 0.5B
fp16 checkpoint plus the loader's own overhead does not fit in 1.8 GB.

Loading a *second* model in the same process is the harder case, and it is what
a ladder does at every rung. Windows has no `malloc_trim`, so freeing the first
model returns nothing to the OS; the next `from_pretrained` starts from a heap
that is still occupied. That is why the FP16 rung succeeded and the 8-bit rung
did not.

## To make it work

Free host RAM. On the machine measured, the top consumers were Chrome (1.2 GB),
an editor (0.7 GB) and agent processes (1.6 GB). Closing browsers and editors
recovers 2–3 GB, which is the difference between 135M and roughly 1.5B.

Rough guide, assuming ~2× the fp16 checkpoint size free in host RAM plus about
1 GB of slack:

| model | fp16 | host RAM you want free |
|---|---|---|
| SmolLM2-135M | 0.27 GB | ~1.5 GB |
| Qwen2.5-0.5B | 1.0 GB | ~3 GB |
| Qwen2.5-1.5B | 3.1 GB | ~7 GB |
| Qwen2.5-3B | 6.2 GB | ~13 GB — not this machine |

VRAM is the *second* limit and a looser one here: 5.35 GB free fits anything up
to about 1.5B in fp16 with generation transients.

## The environment

C: was full (540 MB), so the environment lives on D:.

```bash
python -m venv D:/cliffguard-venv
D:/cliffguard-venv/Scripts/python.exe -m pip install \
    torch --index-url https://download.pytorch.org/whl/cu121
D:/cliffguard-venv/Scripts/python.exe -m pip install \
    "transformers>=4.44,<4.47" accelerate bitsandbytes scipy pydantic \
    sentencepiece protobuf
```

**Pin transformers below 4.47.** transformers 5.x installs cleanly against
torch 2.5.1 and then segfaults on every `from_pretrained`. Verified pair:
`torch 2.5.1+cu121` with `transformers 4.46.3`.

Two environment variables keep the large files off a full C: drive:

```bash
export HF_HOME=D:/hf                                  # model weights
export CLIFFGUARD_ARTIFACTS=D:/cliffguard-local/artifacts   # run directories
```

## Running

```bash
HF_HOME=/d/hf CLIFFGUARD_ARTIFACTS=/d/cliffguard-local/artifacts \
D:/cliffguard-venv/Scripts/python.exe scripts/run_behavioural_ladder.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompts data/eval_suites/xstest.jsonl \
  --n 12 --bits 8 4 --max-new-tokens 24 --batch-size 4 \
  --no-activations \
  --cache /d/cliffguard-local/cache/e2e --label local-e2e
```

`--no-activations` matters locally for the same reason it matters on a free T4:
the residual stream is a second full forward over every prompt, and only the
probe arm reads it.

## What local work is actually worth doing

Not the headline measurement — these models are too small for the refusal
result to mean anything, and a 135M model has barely any refusal behaviour to
lose. What it *is* good for:

- **Validating the pipeline before spending Colab time.** A ladder at `--n 12`
  exercises corpus loading, the corpus fingerprint, generation, the degeneracy
  gate, caching and the run-directory layout in under a minute.
- **The family axis at small scale.** SmolLM2, Qwen and TinyLlama at 135M–1.5B
  are three pretraining pipelines, which is the cheapest available evidence on
  whether anything measured is a property of quantization or of one checkpoint.

The real measurement belongs on the T4. See
[`setup_colab.md`](setup_colab.md) and `notebooks/colab_labelled.ipynb`.
