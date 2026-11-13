#!/usr/bin/env bash
# Preference tuning (DPO) on top of the SFT adapter: mine hard negatives, build pairs, train, evaluate, export.
# Needs artifacts/adapters (run scripts/train.sh first). Run on an otherwise idle machine with the display idle.
#   scripts/dpo.sh pairs     sample the SFT model on train prompts + build data/preferences
#   scripts/dpo.sh train     DPO training -> artifacts/adapters_dpo (time and peak memory in results/dpo/train.log)
#   scripts/dpo.sh eval      SFT and SFT+DPO on the stress set, SFT+DPO on the test set, then results/dpo/RESULTS.md
#   scripts/dpo.sh export    fuse + GGUF Q5_K_M of the DPO model, evaluate it on the test set, refresh the report
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"
export PYTHONPATH=src
BETA="${BETA:-0.1}"; LR="${LR:-5e-6}"; ITERS="${ITERS:-700}"
mkdir -p results/dpo/raw
pairs() {
  "$PY" -m finsight_llm.sample_sft --out data/preferences/sft_samples.jsonl --k 4 --temp 1.0
  "$PY" -m finsight_llm.preferences --samples data/preferences/sft_samples.jsonl --out data/preferences
  "$PY" -m finsight_llm.stress --out data/stress.jsonl
}
train() {
  /usr/bin/time -l "$PY" -m finsight_llm.dpo --beta "$BETA" --lr "$LR" --iters "$ITERS" --out artifacts/adapters_dpo "$@" 2>&1 | tee results/dpo/train.log
  cp artifacts/adapters_dpo/dpo_stats.json results/dpo/dpo_stats.json
}
evaluate() {
  "$PY" -m finsight_llm.evaluate --variant finetuned --adapter artifacts/adapters --test data/stress.jsonl --name sft-stress --out results/dpo/raw/sft-stress.jsonl
  "$PY" -m finsight_llm.evaluate --variant finetuned --adapter artifacts/adapters_dpo --name dpo --out results/dpo/raw/dpo.jsonl
  "$PY" -m finsight_llm.evaluate --variant finetuned --adapter artifacts/adapters_dpo --test data/stress.jsonl --name dpo-stress --out results/dpo/raw/dpo-stress.jsonl
  "$PY" -m finsight_llm.dpo_report
}
export_gguf() {
  MODEL="${MODEL:-models/Qwen2.5-1.5B-Instruct}"
  SRC="${LLAMA_CPP_SRC:?set LLAMA_CPP_SRC to a llama.cpp checkout}"
  OUT=artifacts/gguf-dpo; mkdir -p "$OUT"
  "$PY" -m mlx_lm fuse --model "$MODEL" --adapter-path artifacts/adapters_dpo --save-path artifacts/fused-hf-dpo
  "$PY" "$SRC/convert_hf_to_gguf.py" artifacts/fused-hf-dpo --outtype f16 --outfile "$OUT/finsight-insight-dpo-qwen2.5-1.5b-f16.gguf"
  llama-quantize "$OUT/finsight-insight-dpo-qwen2.5-1.5b-f16.gguf" "$OUT/finsight-insight-dpo-qwen2.5-1.5b-Q5_K_M.gguf" Q5_K_M
  "$PY" -m finsight_llm.evaluate --variant gguf --gguf "$OUT/finsight-insight-dpo-qwen2.5-1.5b-Q5_K_M.gguf" --name dpo-gguf-Q5_K_M --out results/dpo/raw/dpo-gguf-Q5_K_M.jsonl
  "$PY" -m finsight_llm.dpo_report
}
step="${1:-all}"; shift || true
case "$step" in
  pairs) pairs ;;
  train) train "$@" ;;
  eval) evaluate ;;
  export) export_gguf ;;
  all) pairs; train; evaluate; export_gguf ;;
  *) echo "usage: $0 [pairs|train|eval|export|all]" >&2; exit 2 ;;
esac
