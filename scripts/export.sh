#!/usr/bin/env bash
# Fuse the LoRA adapter into the base model, convert to GGUF (llama.cpp) and quantize.
# Needs: llama.cpp binaries on PATH (brew install llama.cpp) and the llama.cpp source tree for
# convert_hf_to_gguf.py (set LLAMA_CPP_SRC; use the tag matching your binaries, e.g. v0.9.0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"
MODEL="${MODEL:-models/Qwen2.5-1.5B-Instruct}"
ADAPTER="${ADAPTER:-artifacts/adapters}"
SRC="${LLAMA_CPP_SRC:?set LLAMA_CPP_SRC to a llama.cpp checkout}"
OUT=artifacts/gguf
mkdir -p "$OUT"

# 1. merge adapter weights into the base weights (HF safetensors, bf16)
"$PY" -m mlx_lm fuse --model "$MODEL" --adapter-path "$ADAPTER" --save-path artifacts/fused-hf
# 2. HF -> GGUF (f16)
"$PY" "$SRC/convert_hf_to_gguf.py" artifacts/fused-hf --outtype f16 --outfile "$OUT/finsight-insight-qwen2.5-1.5b-f16.gguf"
# 3. quantize
for Q in Q8_0 Q5_K_M Q4_K_M; do
  llama-quantize "$OUT/finsight-insight-qwen2.5-1.5b-f16.gguf" "$OUT/finsight-insight-qwen2.5-1.5b-$Q.gguf" "$Q"
done
ls -l "$OUT"/*.gguf
