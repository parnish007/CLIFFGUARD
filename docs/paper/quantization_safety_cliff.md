# The Quantization Safety Cliff Is a Measurement Artifact

**Post-training quantization degrades capability, not alignment. What the
literature measures as a safety cliff is produced by refusal-phrase classifiers
scoring degraded text as harmful compliance.**

---

## Abstract

Post-training quantization is widely reported to break safety alignment
disproportionately: models that refuse harmful requests at full precision are
said to comply at 3–4 bits. We re-examine this claim with an instrument designed
to fail loudly rather than quietly.

Across three instruction-tuned models (Qwen2.5-1.5B, Qwen2.5-3B,
Phi-3.5-mini-3.8B) and a round-to-nearest ladder from 8 down to 2 bits, we
generate real completions at every rung, classify them with a 7B judge rather
than a phrase list, and gate the classification on a degeneracy detector that
combines perplexity with surface repetition statistics.

**Harmful compliance never exceeds 2.2 % at any bit-width on any model.** The
same completions scored with a conventional refusal-marker list yield apparent
unsafe-flip rates up to **38.4 %** — an artifact of counting two things as
compliance: degenerate output containing no refusal phrase, and refusals worded
outside the marker list, including refusals in other languages.

What quantization does produce is a **capability cliff**, and its location is
model-specific rather than universal: GSM8K accuracy is statistically flat down
to 5.5 bits/param on Qwen2.5-3B and to 3.5 on Phi-3.5-mini, then collapses to
zero. On both models the reasoning collapse arrives *before* the model stops
producing language.

We further show that a linear refusal probe — the standard geometric proxy —
retains 96–100 % of its full-precision discriminability across the entire range
where behaviour is unchanged, and only fails where the model has already
degenerated. The probe is not measuring safety; it is measuring fluency.

---

## 1. The claim under test

The safety-cliff hypothesis holds that quantization damages RLHF alignment
non-linearly and disproportionately: a model that refuses at FP16 complies at
Q3_K_M, while general capability degrades only gradually. If true, it is a
serious deployment problem, because edge quantization is routine and safety
evaluation is not typically repeated per bit-width.

The hypothesis rests almost entirely on **automatic refusal classification**. A
model's completion is scored as a refusal if it contains one of a list of
phrases — *"I cannot"*, *"I'm sorry"*, *"as an AI"* — and as compliance
otherwise. Every downstream number inherits whatever that list does.

We began this work assuming the cliff was real and set out to explain its
mechanism. The measurement did not survive its own instrumentation.

---

## 2. Method

### 2.1 The ladder

Per-group asymmetric round-to-nearest, group size 64, applied in-process to every
transformer-block `nn.Linear`; embeddings and the output head stay at FP16, as
every practical scheme does. Bit-widths 8, 7, 6, 5, 4, 3, 2, giving stored costs
of `b + 32/64` bits per parameter exactly.

RTN rather than GGUF k-quants is deliberate. RTN varies **only** bit-width — one
checkpoint, one quantizer family, one group size — which is the ordinal dose axis
the hypothesis is stated over. k-quants vary block structure and per-tensor type
assignment simultaneously; their payload widths are not even integers (Q4_K_M
stores 5.003 bits/param).

### 2.2 Behaviour, not proxies

At every rung we generate greedy completions (48 new tokens, left-padded
batching) for 500 prompts, and ask the question directly: **does this model
refuse the same prompts the FP16 model refused?**

Labels come from the model's own completions. They are not corpus labels — see
§5.1 for why that distinction destroyed our first three months of results.

### 2.3 A three-way classifier, degeneracy first

Completions are labelled **refusal**, **compliance**, or **degenerate**, with
degeneracy decided first. This ordering is the single most important design
decision in the paper.

A 2-bit model emits text like:

```
on \n:.@@\n \ \ \ \ \ \ \ \4\n ""lllv．:\noo@@\no@@\n6o@@
```

A two-way classifier finds no refusal phrase in that and scores it
**compliance**. Our own first implementation reported a **93.8 % unsafe-flip
rate at 2 bits** on this basis. The model had not agreed to do something harmful;
it had stopped producing language.

Degeneracy is detected by combining:

- mean per-token NLL under the FP16 reference, above 3× that reference's own
  median;
- distinct-trigram ratio below 0.60;
- largest single-token share above 0.35;
- alphabetic-character fraction below 0.70.

**Perplexity alone is insufficient, and fails in the direction that matters.**
Repetition loops are *highly predictable*: the 2-bit rung's median NLL was 4.13,
**lower** than the partially-coherent 3-bit rung's 5.80. A perplexity-only gate
flagged 64 % of 3-bit completions and 16 % of 2-bit ones — exactly backwards. The
surface statistics separate the regimes without ambiguity:

| | distinct-3 | max-token share | alphabetic |
|---|---|---|---|
| coherent | 0.999 | 0.06 | 97 % |
| 2-bit | 0.525 | 0.64 | 54 % |

### 2.4 A judge, not a phrase list

Non-degenerate completions are graded by **Qwen2.5-7B-Instruct** (NF4), scoring
REFUSE / COMPLY / UNCLEAR by first-token logit so the decision is deterministic
and needs no output parsing.

We report the judge **alongside four phrase-list variants** and publish every
disagreement, because we had already watched two instruments fail and did not
intend to adopt a third silently.

---

## 3. Results

### 3.1 There is no safety cliff

Coherent harmful compliance, judged, as a fraction of all prompts:

| bits/param | Qwen2.5-3B | Phi-3.5-mini |
|---|---|---|
| 16 (FP16) | — | — |
| 8.5 | 0.2 % | 0.6 % |
| 7.5 | 1.6 % | 1.0 % |
| 6.5 | 1.0 % | 0.8 % |
| 5.5 | 1.0 % | 2.0 % |
| 4.5 | **2.2 %** | 0.8 % |
| 3.5 | 0.0 % † | 0.2 % |
| 2.5 | 0.0 % † | 0.0 % † |

† the model is degenerate at this rung and cannot comply with anything.

The maximum is **2.2 %**, at 4.5 bits on one model. There is no rung, on either
model, at which a quantized model coherently answers a meaningful fraction of the
requests its full-precision version refused.

### 3.2 The artifact, quantified

The identical completions, scored with a refusal-marker list:

| bits/param | judge | phrase list | inflation |
|---|---|---|---|
| 4.5 | 2.2 % | 10.8 % | **4.9×** |
| 3.5 | 0.0 % | 15.4 % | ∞ |
| 2.5 | 0.0 % | 38.4 % | ∞ |

The phrase list reports a monotone rising "safety cliff" reaching 38.4 % at the
rung where the model produces nothing but token salad.

Two mechanisms produce the phantom, and they pull in the same direction:

1. **Degenerate output contains no refusal phrase**, so absence of a marker is
   read as compliance.
2. **Degraded models refuse in a different register.** Of 240 flips our list
   called compliance on a 1.5B model at 3 bits, 143 began with *"Sorry"* and 9
   were in a non-Latin script — including a Chinese-language refusal to a
   suicide-method request, scored as harmful compliance.

The second is the harder problem, because it survives any degeneracy gate. It is
also self-inflicted: tightening the marker list to remove false positives at FP16
(*"As an AI language model, I don't have the ability to…"* is a deflection, not a
refusal) **creates** false negatives at the quantized rungs. Both directions of
error come from the same unfit instrument. Across four marker variants the same
3-bit number ranges from **10.4 % to 48.0 %** — a 4.6× swing on unvalidated
string choices.

### 3.3 What actually breaks: capability

GSM8K, gold arithmetic answers, exact binomial test against the FP16 rate:

**Qwen2.5-3B** (FP16 accuracy 0.185)

| bits/param | accuracy | 95 % CI | p | degenerate |
|---|---|---|---|---|
| 8.5 | 0.195 | [0.142, 0.257] | 0.72 | 0 % |
| 6.5 | 0.175 | [0.125, 0.235] | 0.78 | 0 % |
| 5.5 | 0.180 | [0.129, 0.240] | 0.93 | 0 % |
| **4.5** | **0.115** | [0.074, 0.168] | **0.010** | 0 % |
| 3.5 | 0.000 | [0.000, 0.018] | 2×10⁻¹⁸ | 48 % |

**Phi-3.5-mini** (FP16 accuracy 0.265)

| bits/param | accuracy | 95 % CI | p | degenerate |
|---|---|---|---|---|
| 8.5 | 0.250 | [0.192, 0.316] | 0.69 | 0 % |
| 5.5 | 0.250 | [0.192, 0.316] | 0.69 | 0 % |
| 4.5 | 0.265 | [0.205, 0.332] | 1.00 | 0 % |
| **3.5** | 0.250 | [0.192, 0.316] | 0.69 | 1.5 % |
| 2.5 | 0.000 | [0.000, 0.018] | 3×10⁻²⁷ | 98 % |

**The collapse point is model-specific.** Qwen2.5-3B degrades significantly at
4.5 bits and is destroyed by 3.5. Phi-3.5-mini is statistically indistinguishable
from full precision at 3.5 and collapses only at 2.5 — a full bit lower. There is
no universal threshold.

At Qwen-3B's 4.5-bit rung, accuracy falls by 38 % relative while **degeneracy is
zero**: the model writes fluent, plausible arithmetic and gets it wrong.
Capability failure precedes fluency failure by a full rung. **A deployment check
that screens only for incoherent output passes a model that has lost a third of
its reasoning.**

### 3.4 The geometric proxy tracks fluency, not safety

A linear refusal direction fitted on FP16 activations and frozen, then applied to
every rung:

| bits/param | Qwen2.5-3B | Phi-3.5-mini |
|---|---|---|
| 8.5 | 100.3 % | 100.3 % |
| 5.5 | 96.3 % | 100.3 % |
| 4.5 | 97.6 % | 100.0 % |
| 3.5 | 34.8 % | 64.4 % |
| 2.5 | 11.8 % | −1.3 % |

Retention is essentially perfect exactly where behaviour is unchanged, and
collapses exactly where the model degenerates. It carries no information about
the 4.5-bit reasoning loss on Qwen-3B, which is the one real capability change in
the coherent range. **The probe is a fluency detector wearing a safety label.**

---

## 4. What this means

**For practitioners.** Do not screen quantized checkpoints with refusal-phrase
classifiers or activation probes. Both will report a model safe when it has lost
a third of its reasoning, and unsafe when it has merely become incoherent. Screen
for capability on a gold-labelled task, and screen for degeneracy directly.

**For the literature.** Published safety-cliff magnitudes should be treated as
upper bounds contaminated by degeneracy and by refusal-phrasing drift, unless the
work reports a degeneracy rate and validates its classifier against differently
worded and non-English refusals. We do not claim prior results are wrong; we
claim the instrument that produced them cannot distinguish the finding from the
artifact, and that when we control for both, the effect largely disappears.

**For alignment robustness.** The interesting result is the negative one. RLHF
refusal behaviour is *more* robust to weight quantization than arithmetic
reasoning is. Refusal survives to the point of linguistic collapse; multi-step
arithmetic does not. That ordering is the opposite of the safety-cliff intuition
and it is worth explaining.

---

## 5. Limitations, and errors we made

We list our own failures because each one produced a plausible, publishable, and
false number, and because the correction path is the methodological contribution.

### 5.1 The corpus labels were near-random

Our first labels came from a standard construction: a prompt is "refused" if the
hh-rlhf *rejected response* contains a refusal marker. That is a property of the
response, not the prompt. The resulting classes contained *"How much time do you
spend with your family?"* as harmful and *"Can you provide me the home address
for actor Ryan Reynolds?"* as benign. Agreement with the model's own behaviour
was **0.524** — chance. Every result computed against that partition was void,
including a headline finding that d′ does not degrade under quantization.

### 5.2 We measured the wrong estimand

Every d′ was computed by refitting the probe direction on each scheme's own
activations, which answers *"can a fresh probe be trained here?"*, not *"does the
original readout survive?"*. The two diverge sharply at low precision. §3.4 uses
the frozen probe.

### 5.3 The degeneracy gate ran backwards

Described in §2.3. Perplexity-only gating inverted the 2-bit and 3-bit rows.

### 5.4 A fitted constant was our own arithmetic

We reported a fitted noise-growth base of 4.355 as evidence against the
theoretical value of 4. Our quantizer uses `levels = 2^b − 1`, so error variance
scales as `1/(2^b − 1)²`; fitting that theoretical sequence through the same
model gives **4.3429**. We had rediscovered our own step-size formula.

### 5.5 Standing limits

- Greedy decoding only. One completion estimates a deterministic decision, not
  behaviour under sampling; a low-temperature safety failure could be missed.
- No independent harmfulness annotation. Labels describe *what the model does*,
  not whether the request was harmful. We measure **behavioural change under
  quantization**, not absolute safety.
- The judge is unvalidated against blinded human labels. It is one instrument
  reported beside four others, not ground truth.
- The judge runs in NF4 on a T4.
- Three models, one quantizer family, one language for the safety arm, one
  reasoning benchmark.
- Absolute GSM8K accuracy is low (0.185–0.265) because generation is zero-shot
  with a 192-token budget. Only the relative cliff is measured.

---

## 6. Reproduction

Everything runs on a 6 GB laptop GPU except the 7B judge and the 3B/3.8B models.

```bash
python scripts/run_behavioural_ladder.py --model <hf-id> --n 250
python scripts/run_sector_ladder.py      --model <hf-id> --n 200
python scripts/classify_completions_judge.py <run-dir> \
       --judge-model Qwen/Qwen2.5-7B-Instruct --judge-4bit
python scripts/reanalyse_runs.py artifacts/runs
```

Raw completions are saved verbatim for every rung, so any classifier can be
re-applied without regenerating. That property is what made §5.3 recoverable at
zero GPU cost, and we recommend it as standard practice: **the completions are
the data; the labels are an opinion about them.**
