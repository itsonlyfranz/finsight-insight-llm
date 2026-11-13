# SFT vs SFT + DPO

Greedy decoding, same prompts, same scoring code. Differences are paired by example; the interval is a 2,000-resample paired bootstrap (seed 0).

## Held-out test split (n = 246)

| Metric | SFT | SFT + DPO | Δ (DPO − SFT) | 95% CI (paired bootstrap) |
|---|---|---|---|---|
| Number fidelity | 100.0% | 100.0% | +0.0 pp | [+0.0, +0.0] pp |
| Format valid | 100.0% | 100.0% | +0.0 pp | [+0.0, +0.0] pp |
| App-accepted | 100.0% | 100.0% | +0.0 pp | [+0.0, +0.0] pp |
| ROUGE-L | 0.454 | 0.458 | +0.003 | [-0.028, +0.037] |
| Fact coverage | 0.881 | 0.847 | −0.035 | [-0.058, -0.011] |

**ROUGE-L by task**

| Task | n | SFT | SFT + DPO |
|---|---|---|---|
| insight | 117 | 0.517 | 0.499 |
| simulation | 78 | 0.322 | 0.372 |
| goal | 51 | 0.513 | 0.495 |

**Fact coverage by task**

| Task | n | SFT | SFT + DPO |
|---|---|---|---|
| insight | 117 | 0.957 | 0.957 |
| simulation | 78 | 0.919 | 0.774 |
| goal | 51 | 0.650 | 0.706 |

## Adversarial stress set (n = 234)

Each prompt carries 8 near-duplicate decoy figures (digit transpositions, off-by-a-little amounts). *Decoy rate* is the share of outputs that state any decoy: grounded (the guard passes it) but wrong.

| Metric | SFT | SFT + DPO | Δ (DPO − SFT) | 95% CI (paired bootstrap) |
|---|---|---|---|---|
| Number fidelity | 100.0% | 100.0% | +0.0 pp | [+0.0, +0.0] pp |
| Format valid | 100.0% | 100.0% | +0.0 pp | [+0.0, +0.0] pp |
| App-accepted | 100.0% | 100.0% | +0.0 pp | [+0.0, +0.0] pp |
| ROUGE-L | 0.464 | 0.435 | −0.029 | [-0.064, +0.006] |
| Fact coverage | 0.842 | 0.788 | −0.054 | [-0.083, -0.024] |
| Decoy rate (lower is better) | 0.9% | 1.7% | +0.9 pp | |

**Number fidelity by task**

| Task | n | SFT | SFT + DPO |
|---|---|---|---|
| insight | 108 | 100.0% | 100.0% |
| simulation | 77 | 100.0% | 100.0% |
| goal | 49 | 100.0% | 100.0% |

**ROUGE-L by task**

| Task | n | SFT | SFT + DPO |
|---|---|---|---|
| insight | 108 | 0.533 | 0.500 |
| simulation | 77 | 0.374 | 0.337 |
| goal | 49 | 0.456 | 0.447 |

**Fact coverage by task**

| Task | n | SFT | SFT + DPO |
|---|---|---|---|
| insight | 108 | 0.926 | 0.931 |
| simulation | 77 | 0.957 | 0.760 |
| goal | 49 | 0.476 | 0.517 |

## GGUF Q5_K_M (test split)

| Model | Number fidelity | Format valid | ROUGE-L | Fact coverage | Decode tok/s | Mean latency | Size |
|---|---|---|---|---|---|---|---|
| SFT, GGUF Q5_K_M | 100.0% | 100.0% | 0.456 | 0.894 | 70.9 | 0.92 s | 1,125 MB |
| SFT + DPO, GGUF Q5_K_M | 100.0% | 100.0% | 0.453 | 0.867 | 71.5 | 0.88 s | 1,125 MB |
