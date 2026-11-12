---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
language:
- en
library_name: gguf
pipeline_tag: text-generation
tags:
- finance
- lora
- gguf
- fine-tuned
datasets:
- finsight-insight-synthetic
---

# FinSight Insight 1.5B (draft model card)

> Draft for review. Nothing has been uploaded.

A LoRA fine-tune of **Qwen2.5-1.5B-Instruct** that rewrites structured personal-finance facts (spending insights, what-if savings simulations, savings-goal milestones) into short, warm, plain-text messages, using only the numbers it is given. It was built for the FinSight Android app, where a deterministic engine computes the numbers and the model only phrases them.

## Files (not committed to git; produced by `make export`)

| File | Size | Notes |
|---|---|---|
| `finsight-insight-qwen2.5-1.5b-Q5_K_M.gguf` | 1,125,050,144 B (1.1 GB) | **recommended** |
| `finsight-insight-qwen2.5-1.5b-Q8_0.gguf` | 1,646,572,832 B (1.6 GB) | no measurable gain over Q5_K_M |
| `finsight-insight-qwen2.5-1.5b-Q4_K_M.gguf` | 986,048,288 B (0.99 GB) | smallest/fastest; occasionally breaks the format |
| `finsight-insight-qwen2.5-1.5b-f16.gguf` | 3,093,669,152 B (3.1 GB) | reference |
| LoRA adapter (`adapters.safetensors`) | 73,901,662 B (74 MB) | for use with the base model in MLX |

## Intended use and prompt format

Use the same prompts as the app: a system message ("You are FinSight, a concise personal-finance coach ... Use only the numbers provided ...") and a user message made of the facts block followed by one instruction. For insights:

```
Monthly take-home income: ₱35,000.00
Insight type: risk nudge
Headline: High Cafe & Dining Velocity
Findings: You've recorded 10 cafe & dining outings totaling ₱2,940.02 (~8.4% of your monthly income). ...
Suggested action: Try a 2-day homebrew routine ... ~₱735.00/mo ...
Amount at stake: ₱2,940.02

Rewrite the findings and the suggested action in a warm, specific voice for the user. Reply in exactly two lines:
INSIGHT: <at most two sentences>
ACTION: <one concrete sentence>
```

Output is `INSIGHT: ...` / `ACTION: ...`. The exact system prompt and the other two instructions are in `src/finsight_llm/prompts.py`. **Always validate the output** with a number guard (every money figure must appear in the facts) and fall back to deterministic text on failure.

```bash
llama-cli -m finsight-insight-qwen2.5-1.5b-Q5_K_M.gguf --jinja -sys "<system prompt>" -p "<facts + instruction>" --temp 0
```

Out of scope: financial advice, tax or investment guidance, any input that is not in the facts format, other currencies or languages.

## Training data

1,618 training / 114 validation examples of fully synthetic facts with templated reference texts; see the dataset card. No real finances.

## Training

LoRA rank 16, scale 20, dropout 0.05 on all attention and MLP projections (28 layers; 18.5M trainable parameters), AdamW, lr 2e-5 with warmup and cosine decay, batch 4, 600 iterations, loss on the reply only, `mlx-lm` on an Apple M2 Pro (32 GB). Training took 1,790 s including validation; peak memory footprint 14.55 GB. Final validation loss 0.034.

## Evaluation

246 held-out synthetic test examples, greedy decoding:

| Model | Number fidelity | Format valid | ROUGE-L | Fact coverage | Decode tok/s | Size |
|---|---|---|---|---|---|---|
| Base, zero-shot | 97.6% | 58.9% | 0.218 | 0.631 | 47.9 | 3,099 MB |
| Base + 2-shot | 100.0% | 95.1% | 0.335 | 0.894 | 39.3 | 3,099 MB |
| Fine-tuned (MLX bf16) | 100.0% | 100.0% | 0.429 | 0.881 | 30.5 | 3,099 MB + 74 MB |
| Fine-tuned GGUF Q8_0 | 100.0% | 100.0% | 0.425 | 0.913 | 71.8 | 1,647 MB |
| **Fine-tuned GGUF Q5_K_M** | 100.0% | 100.0% | 0.428 | 0.894 | 70.9 | 1,125 MB |
| Fine-tuned GGUF Q4_K_M | 100.0% | 98.0% | 0.434 | 0.846 | 83.5 | 986 MB |

Number fidelity is the pass rate of a port of the app's money-figure guard. Speeds are llama.cpp (GGUF rows) and mlx-lm (MLX rows) on one M2 Pro; see the repository README for definitions and caveats.

## Limitations and risks

- Trained and tested only on synthetic data from one generator; real-world robustness is unmeasured.
- Reference texts are templated, so reference-based metrics reward matching a house style. No human or LLM-judge evaluation was done.
- Number fidelity of 100% on 246 prompts is not a guarantee; keep the guard.
- Single run, single seed, no confidence intervals.
- Output wording is repetitive by design (few phrasings per case).
- Not financial advice. May state generic recommendations that came from the facts (for example savings-account yield text) without verifying them.

## License

Apache-2.0, inherited from the base model Qwen2.5-1.5B-Instruct (https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
