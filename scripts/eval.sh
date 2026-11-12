#!/usr/bin/env bash
# Evaluate base, few-shot and fine-tuned models, plus every GGUF in artifacts/gguf, on data/test.jsonl.
# Run on an otherwise idle machine: it measures latency and tokens/s.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"
export PYTHONPATH=src
mkdir -p results/raw
for v in base fewshot finetuned; do
  "$PY" -m finsight_llm.evaluate --variant "$v" --out "results/raw/$v.jsonl"
done
for f in artifacts/gguf/*.gguf; do
  [ -e "$f" ] || continue
  q=$(basename "$f" .gguf); q=${q##*-}           # f16, Q8_0, Q5_K_M, Q4_K_M
  "$PY" -m finsight_llm.evaluate --variant gguf --gguf "$f" --name "gguf-$q" --out "results/raw/gguf-$q.jsonl"
done
"$PY" -m finsight_llm.report
