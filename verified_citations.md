# `verified_citations.md` — Ground-truth citations for the CLIFFGUARD project

Every arXiv citation used in the unified blueprint or codebase should appear here, marked as **verified** (you have personally confirmed the paper exists, the arXiv ID resolves, and the title/authors match) or **unverified** (cited from training data only, awaiting verification).

Desktop must consult this file before introducing any new citation. If a citation is not here, Desktop must either web-search to verify and add it, or mark the citation `[unverified — cite needed]` in the document until you can verify it manually.

---

## How to verify a citation

1. Open `https://arxiv.org/abs/<arxiv_id>` in a browser.
2. Confirm the title matches what you have. Confirm the first author and year match. Confirm the venue (if cited).
3. Update this file: change status to **verified**, add the date.
4. If the title or authors don't match what was claimed in the blueprint, fix the blueprint.

---

## Load-bearing citations (verify these first)

| arXiv ID | Title (verbatim) | First author | Year | Venue | Status | Verified date |
|---|---|---|---|---|---|---|
| 2406.11717 | Refusal in Language Models Is Mediated by a Single Direction | Andy Arditi | 2024 | NeurIPS 2024 | unverified | — |
| 2507.11878 | LLMs Encode Harmfulness and Refusal Separately | (Zhao et al.) | 2025 | NeurIPS 2025 | unverified | — |
| 2403.15447 | (cited as Hong et al., 3-bit safety cliff) | Hong | 2024 | — | unverified | — |
| 2405.18137 | (cited as Egashira et al., GGUF poisoning) | Egashira | 2024 | NeurIPS 2024 | unverified | — |
| 2505.23786 | Mind the Gap | — | 2025 | ICLR 2025 (TV17MLZGuA) | unverified | — |
| 2501.18837 | Constitutional Classifiers | Sharma | 2025 | — | unverified | — |
| 2505.03574 | LlamaFirewall | Meta | 2025 | — | unverified | — |
| 2403.14720 | (Hines et al. — spotlighting) | Hines | 2024 | — | unverified | — |
| 2411.17713 | (Llama-Guard-3 lineage) | — | 2024 | — | unverified | — |
| 2402.10260 | StrongREJECT | — | 2024 | — | unverified | — |
| 2511.07842 | (AAQ/CAQ — Wee et al.) | Wee | 2025 | — | unverified | — |
| 2312.17673 | Jatmo | Piet | 2023 | — | unverified | — |
| 2404.05993 | AEGIS | Ghosh | 2024 | — | unverified | — |
| 2310.03684 | SmoothLLM | — | 2023 | — | unverified | — |
| 2307.15043 | GCG | Zou | 2023 | — | unverified | — |
| 2310.04451 | AutoDAN | Liu | 2023 | — | unverified | — |
| 2310.08419 | PAIR | — | 2023 | — | unverified | — |
| 2312.02119 | TAP | — | 2023 | — | unverified | — |
| 2404.01833 | Crescendo | — | 2024 | — | unverified | — |
| 2402.11753 | ArtPrompt | — | 2024 | — | unverified | — |
| 2412.03556 | Best-of-N | — | 2024 | — | unverified | — |
| 2402.04249 | HarmBench | — | 2024 | — | unverified | — |
| 2404.01318 | JailbreakBench | — | 2024 | — | unverified | — |
| 2406.13352 | AgentDojo | — | 2024 | — | unverified | — |
| 2403.02691 | InjecAgent | — | 2024 | — | unverified | — |
| 2311.01011 | TensorTrust | — | 2023 | — | unverified | — |
| 2305.14314 | NF4 (QLoRA / Dettmers) | Dettmers | 2023 | — | unverified | — |
| 2208.07339 | LLM.int8() | Dettmers | 2022 | — | unverified | — |
| 2402.05162 | (Wei et al. — <1% safety weights) | Wei | 2024 | — | unverified | — |
| 2309.00614 | (Jain et al. — perplexity gate) | Jain | 2023 | — | unverified | — |
| 2308.14132 | (Alon & Kamfonas — perplexity) | Alon | 2023 | — | unverified | — |
| 2302.12173 | (Greshake — indirect injection) | Greshake | 2023 | — | unverified | — |
| 2401.06373 | (Zeng et al. — persuasion) | Zeng | 2024 | — | unverified | — |
| 2402.01359 | TESSERACT | — | 2024 | — | unverified | — |
| 1003.0146 | LinUCB | Li | 2010 | WWW 2010 | unverified | — |
| 1612.00410 | VIB | — | 2016 | — | unverified | — |
| 2211.15070 | KCUSUM | Wei | 2022 | — | unverified | — |
| 2509.09708 | (Beyond I'm Sorry — SAE refusal) | — | 2025 | — | unverified | — |
| 2505.17306 | (cross-lingual refusal universality) | — | 2025 | — | unverified | — |

---

## Citations marked unverified in the blueprint itself

These are already flagged `[unverified — cite needed]` in the unified blueprint. Do NOT use them in new content until they are independently verified.

- **Wollschläger et al. — Geometry of Refusal.** Venue unverified; arXiv ID not yet located.
- **Constitutional Classifiers++ (arXiv 2601.04603).** arXiv ID format is suspicious (2026 prefix). Verify before citing.
- **NASB — 73.7 % NF4-Qwen-2.5-7B figure.** Per blueprint §17, this exact paper could not be verified; use Egashira and Hong et al. instead.

---

## Adding a new citation

When a new paper is needed:

1. Web-search the title to find the arXiv abstract page.
2. Copy the arXiv ID, full title, first author, and year into a new row above.
3. Mark **verified** and dated.
4. Only then cite it in blueprint or code.

If you cannot find the paper, do **not** invent an arXiv ID. Mark `[unverified — cite needed]` in the document instead.
