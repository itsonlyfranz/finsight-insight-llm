# Results (held-out test split)

n = 246 test examples per model, greedy decoding.

| Model | Number fidelity | Format valid | App-accepted | ROUGE-L | Fact coverage | Decode tok/s | Mean latency | Size on disk |
|---|---|---|---|---|---|---|---|---|
| Base model, zero-shot (MLX bf16) | 97.6% | 58.9% | 97.6% | 0.252 | 0.631 | 47.9 | 1.91 s | 3,099 MB |
| Base model + 2-shot prompt (MLX bf16) | 100.0% | 95.1% | 100.0% | 0.373 | 0.894 | 39.3 | 1.94 s | 3,099 MB |
| Fine-tuned (LoRA), zero-shot (MLX bf16) | 100.0% | 100.0% | 100.0% | 0.454 | 0.881 | 30.5 | 2.17 s | 3,173 MB |
| Fine-tuned, GGUF f16 | 100.0% | 100.0% | 100.0% | 0.448 | 0.920 | 47.1 | 1.45 s | 3,094 MB |
| Fine-tuned, GGUF Q8_0 | 100.0% | 100.0% | 100.0% | 0.447 | 0.913 | 71.8 | 0.95 s | 1,647 MB |
| Fine-tuned, GGUF Q5_K_M | 100.0% | 100.0% | 100.0% | 0.456 | 0.894 | 70.9 | 0.92 s | 1,125 MB |
| Fine-tuned, GGUF Q4_K_M | 100.0% | 98.0% | 100.0% | 0.458 | 0.846 | 83.5 | 0.76 s | 986 MB |

**Number fidelity by task**

| Model | insight | simulation | goal |
|---|---|---|---|
| Base model, zero-shot (MLX bf16) | 99.1% | 93.6% | 100.0% |
| Base model + 2-shot prompt (MLX bf16) | 100.0% | 100.0% | 100.0% |
| Fine-tuned (LoRA), zero-shot (MLX bf16) | 100.0% | 100.0% | 100.0% |
| Fine-tuned, GGUF f16 | 100.0% | 100.0% | 100.0% |
| Fine-tuned, GGUF Q8_0 | 100.0% | 100.0% | 100.0% |
| Fine-tuned, GGUF Q5_K_M | 100.0% | 100.0% | 100.0% |
| Fine-tuned, GGUF Q4_K_M | 100.0% | 100.0% | 100.0% |

**Format validity by task**

| Model | insight | simulation | goal |
|---|---|---|---|
| Base model, zero-shot (MLX bf16) | 48.7% | 88.5% | 37.3% |
| Base model + 2-shot prompt (MLX bf16) | 100.0% | 100.0% | 76.5% |
| Fine-tuned (LoRA), zero-shot (MLX bf16) | 100.0% | 100.0% | 100.0% |
| Fine-tuned, GGUF f16 | 100.0% | 100.0% | 100.0% |
| Fine-tuned, GGUF Q8_0 | 100.0% | 100.0% | 100.0% |
| Fine-tuned, GGUF Q5_K_M | 100.0% | 100.0% | 100.0% |
| Fine-tuned, GGUF Q4_K_M | 95.7% | 100.0% | 100.0% |

**ROUGE-L by task**

| Model | insight | simulation | goal |
|---|---|---|---|
| Base model, zero-shot (MLX bf16) | 0.278 | 0.229 | 0.229 |
| Base model + 2-shot prompt (MLX bf16) | 0.433 | 0.311 | 0.331 |
| Fine-tuned (LoRA), zero-shot (MLX bf16) | 0.517 | 0.322 | 0.513 |
| Fine-tuned, GGUF f16 | 0.507 | 0.325 | 0.501 |
| Fine-tuned, GGUF Q8_0 | 0.507 | 0.331 | 0.489 |
| Fine-tuned, GGUF Q5_K_M | 0.507 | 0.342 | 0.514 |
| Fine-tuned, GGUF Q4_K_M | 0.501 | 0.401 | 0.446 |
