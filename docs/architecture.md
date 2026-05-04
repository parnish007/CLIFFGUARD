<div align="center">

[← README](../README.md) &nbsp;|&nbsp;
[What is it](what_is_it.md) &nbsp;|&nbsp;
[Architecture](architecture.md) &nbsp;|&nbsp;
[Math](math.md) &nbsp;|&nbsp;
[Setup](setup.md) &nbsp;|&nbsp;
[Engineering Ref](engineering_reference.md)

</div>

# System Architecture

## 1. Request Cycle

> The request cycle is stateless per-request. ATTEST-WH runs once
> at boot and its result is cached. All other gates run on every
> request in the order shown. The CONDUCTOR aggregates all verdicts
> into a single block/allow decision.

```mermaid
flowchart TD
    Boot([Boot]) --> ATTEST["ATTEST-WH\n(boot-time, cached)\nSHA-256 vs vendor manifest"]
    ATTEST -->|ALLOW / DEGRADED| START

    START([Request arrives]) --> V["VESTIBULE\nLZ + PS\n(input string gates)"]
    V -->|risk_in > tau_hard| BLOCK_V([BLOCK — refuse])
    V -->|pass| ENGINE["Model Inference\n(forward pass)"]

    ENGINE -->|white-box path| PROBE["PROBE\nRM + MT + HD\n(residual-stream gates)"]
    ENGINE -->|black-box path| BPROBE["B-PROBE\nLOGIT + CONSISTENCY\n(top-k logprobs)"]
    ENGINE --> TRIPWIRE["TRIPWIRE\nH + R\n(streaming entropy / ratio)"]

    PROBE --> CONDUCTOR
    BPROBE --> CONDUCTOR
    TRIPWIRE -->|alarm| STOP_GEN([Stop generation])
    TRIPWIRE -->|continue| ENGINE2["Generation continues"]

    ENGINE2 --> LOOKOUT["LOOKOUT\nCT + JG\n(output monitors)"]
    LOOKOUT --> CONDUCTOR

    CONDUCTOR["CONDUCTOR\nLinUCB bandit\nbuild_context → select_weights → aggregate_verdict"]
    CONDUCTOR -->|fired_score / total_weight > 0.5| BLOCK([BLOCK])
    CONDUCTOR -->|else| ALLOW([ALLOW])

    ALLOW -->|outcome feedback| CONDUCTOR
    BLOCK -->|incident logged| CONDUCTOR
```

## 2. Eight Components

> Firing directions are shown inline (fires HIGH or fires LOW).
> Gates in the left column (VESTIBULE, PROBE, LOOKOUT) run before
> and after generation. Gates in the right column (TRIPWIRE) run
> during generation.

```mermaid
flowchart LR
    subgraph VESTIBULE["VESTIBULE (input gates)"]
        LZ[VESTIBULE-LZ\ncompression ratio\nfires HIGH]
        PS[VESTIBULE-PS\nprovenance spotlight\nfires HIGH]
    end

    subgraph PROBE["PROBE (white-box, pre-decode)"]
        RM[PROBE-RM\nrefusal margin\nfires LOW]
        MT[PROBE-MT\nmargin trajectory\nfires LOW]
        HD[PROBE-HD\nharmfulness direction\nfires HIGH]
    end

    subgraph BPROBE["B-PROBE (black-box)"]
        LOGIT[B-PROBE-LOGIT\nlogistic head on logprobs\nfires HIGH]
        CONS[B-PROBE-CONSISTENCY\nJSD across N paraphrases\nfires LOW]
    end

    subgraph TRIPWIRE["TRIPWIRE (streaming)"]
        TH[TRIPWIRE-H\nentropy CUSUM\nfires HIGH]
        TR[TRIPWIRE-R\nKenLM reference ratio\nfires LOW]
    end

    subgraph LOOKOUT["LOOKOUT (output)"]
        CT[LOOKOUT-CT\ncanary token check\nfires HIGH]
        JG[LOOKOUT-JG\ncompliance judge\nfires HIGH]
    end

    subgraph CONDUCTOR["CONDUCTOR"]
        LINUCB[LinUCB bandit\nper-arm A matrix + b vector]
        ADWIN[ADWIN drift detector\nweight reset on drift]
    end

    subgraph LADDER["LADDER"]
        ROUTER[tier router\nroute + gate_count]
    end

    subgraph ATTEST["ATTEST"]
        WH[ATTEST-WH\nSHA-256 weight attestation\nboot-time only]
    end

    VESTIBULE --> CONDUCTOR
    PROBE --> CONDUCTOR
    BPROBE --> CONDUCTOR
    TRIPWIRE --> CONDUCTOR
    LOOKOUT --> CONDUCTOR
    LADDER -->|gate list| VESTIBULE
    LADDER -->|gate list| PROBE
    LADDER -->|gate list| BPROBE
    ATTEST -.->|result| CONDUCTOR
```

## 3. Tier Gate Selection

```mermaid
stateDiagram-v2
    [*] --> TierA : RTX 5060 8GB
    [*] --> TierB : Pi 5 8GB CPU
    [*] --> TierC : 2GB embedded
    [*] --> TierCPlus : 2GB + PG2-22M

    TierA : Tier A
    TierA : All 12 gates active
    TierA : VESTIBULE-LZ, VESTIBULE-PS
    TierA : PROBE-RM, PROBE-MT, PROBE-HD
    TierA : TRIPWIRE-H, TRIPWIRE-R
    TierA : LOOKOUT-CT, LOOKOUT-JG
    TierA : B-PROBE-LOGIT, B-PROBE-CONSISTENCY
    TierA : ATTEST-WH
    TierA : LinUCB |A|=16

    TierB : Tier B
    TierB : 11 gates (no LOOKOUT-JG)
    TierB : VESTIBULE-LZ, VESTIBULE-PS
    TierB : PROBE-RM, PROBE-MT, PROBE-HD
    TierB : TRIPWIRE-H, TRIPWIRE-R
    TierB : LOOKOUT-CT
    TierB : B-PROBE-LOGIT, B-PROBE-CONSISTENCY
    TierB : ATTEST-WH
    TierB : LinUCB |A|=8

    TierC : Tier C (NARROW SCOPE)
    TierC : 3 gates only
    TierC : VESTIBULE-LZ, VESTIBULE-PS
    TierC : ATTEST-WH
    TierC : No bandit — static weights
    TierC : NOT defended against A7

    TierCPlus : Tier C+
    TierCPlus : 4 gates
    TierCPlus : VESTIBULE-LZ, VESTIBULE-PS
    TierCPlus : B-PROBE-LOGIT (PromptGuard-2-22M)
    TierCPlus : ATTEST-WH
    TierCPlus : Static weights
```

## 4. CONDUCTOR Feature Vector

> The 14-dimensional context vector is assembled by
> `conductor/context.py`. Missing gates (not active for the tier)
> are filled with 0.0. The tier indicator at index 13 is always
> present regardless of tier.

The CONDUCTOR context vector has 14 dimensions. All values are floats in [0, 1] or [-1, 1]. Missing gates (not active for the tier) are filled with 0.0.

| Index | Feature | Source Gate | Notes |
|---|---|---|---|
| 0 | VESTIBULE-LZ score | VESTIBULE-LZ | Compression ratio (higher = more compressible = less suspicious) |
| 1 | VESTIBULE-PS score | VESTIBULE-PS | Signal score (higher = more provenance signals detected) |
| 2 | PROBE-RM margin | PROBE-RM | Refusal-direction cosine (lower = further from refusal subspace = risk) |
| 3 | PROBE-MT rho_dot | PROBE-MT | First derivative of margin trajectory (negative = margin falling) |
| 4 | PROBE-MT rho_ddot | PROBE-MT | Second derivative — computed in MT, deferred to context builder |
| 5 | PROBE-HD margin | PROBE-HD | Harmfulness-direction cosine at user-instruction token |
| 6 | TRIPWIRE-H CUSUM | TRIPWIRE-H | Final CUSUM statistic (higher = more cumulative entropy deviation) |
| 7 | TRIPWIRE-R LLR | TRIPWIRE-R | Log-likelihood ratio vs benign reference (lower = more adversarial) |
| 8 | LOOKOUT-CT count | LOOKOUT-CT | Number of canary tokens found in output (normally 0) |
| 9 | LOOKOUT-JG rate | LOOKOUT-JG | Compliance rate across N paraphrases (higher = suspicious) |
| 10 | B-PROBE-LOGIT score | B-PROBE-LOGIT | Logistic head output on top-k logprobs |
| 11 | B-PROBE-CONSISTENCY JSD | B-PROBE-CONSISTENCY | Jensen-Shannon divergence across N paraphrases |
| — | *Metadata dimensions* | — | — |
| 12 | ATTEST-WH result | ATTEST-WH | 1.0=ALLOW, 0.5=DEGRADED, 0.0=BLOCK |
| 13 | Tier indicator | LADDER | 0.0=A, 0.33=B, 0.67=C, 1.0=C_PLUS |

## 5. Component Interaction (Sequence)

> The sequence shows the white-box path (PROBE gets hidden states).
> On the black-box path, PROBE is skipped and B-PROBE receives
> top-k logprobs instead. TRIPWIRE runs concurrently with generation
> via a streaming callback.

```mermaid
sequenceDiagram
    participant User
    participant VESTIBULE
    participant ENGINE as Model Engine
    participant PROBE
    participant BPROBE as B-PROBE
    participant TRIPWIRE
    participant LOOKOUT
    participant CONDUCTOR
    participant Decision

    User->>VESTIBULE: raw prompt + tool context
    VESTIBULE->>ENGINE: passed prompt (risk_in < tau_hard)
    ENGINE-->>PROBE: hidden states at layers L (white-box)
    ENGINE-->>BPROBE: top-k logprobs at first token (black-box)
    ENGINE-->>TRIPWIRE: streaming token entropies + logprobs
    PROBE->>CONDUCTOR: Margin + GateVerdict (RM, MT, HD)
    BPROBE->>CONDUCTOR: Margin + GateVerdict (LOGIT, CONSISTENCY)
    TRIPWIRE->>CONDUCTOR: GateVerdict (H, R)
    ENGINE-->>LOOKOUT: completed output tokens
    LOOKOUT->>CONDUCTOR: GateVerdict (CT, JG)
    CONDUCTOR->>CONDUCTOR: build_context(verdicts, tier, attest_result)
    CONDUCTOR->>CONDUCTOR: select_weights(context_vector)
    CONDUCTOR->>Decision: aggregate_verdict(verdicts, weights)
    Decision-->>User: ALLOW (serve output) or BLOCK (refuse)
    Decision-->>CONDUCTOR: reward signal for bandit update
```

---

<div align="center">

[← Back to README](../README.md) &nbsp;·&nbsp;
[Open an issue](https://github.com/YOUR_USERNAME/CLIFFGUARD/issues) &nbsp;·&nbsp;
[preregistration.md](preregistration.md)

</div>
