#!/usr/bin/env bash
# Fine-tune with LoRA. Wall-clock time and peak memory are recorded in results/train.log.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"
mkdir -p results artifacts
start=$(date +%s)
/usr/bin/time -l "$PY" -m mlx_lm lora -c configs/lora.yaml "$@" 2>&1 | tee results/train.log
end=$(date +%s)
echo "wall_clock_seconds: $((end - start))" | tee -a results/train.log
