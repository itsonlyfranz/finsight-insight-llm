"""Evaluation harness.

    python -m finsight_llm.evaluate --variant base       --out results/raw/base.jsonl
    python -m finsight_llm.evaluate --variant fewshot    --out results/raw/fewshot.jsonl
    python -m finsight_llm.evaluate --variant finetuned  --out results/raw/finetuned.jsonl
    python -m finsight_llm.evaluate --variant gguf --gguf artifacts/gguf/x.gguf --out results/raw/x.jsonl

Variants
  base       the unmodified instruct model, prompted exactly as the app prompts it (zero-shot)
  fewshot    the same base model with two worked examples per task prepended (prompt-only baseline)
  finetuned  the base model with the LoRA adapter, zero-shot, same prompt as `base`
  gguf       any GGUF served by llama.cpp, zero-shot

All variants use greedy decoding and the held-out test split. Every generation is written to a jsonl
file (raw output, token counts, timings, per-row scores) together with the exact command and settings.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from . import dataset, metrics, prompts

FEWSHOT_KINDS = {
    "insight": ["insight_cafe_lifestyle_inflation", "insight_emergency_fund_status"],
    "simulation": ["simulation", "simulation"],
    "goal": ["goal", "goal"],
}


def load_test(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def fewshot_examples() -> dict[str, list[dict]]:
    """Two fixed training examples per task (the first ones of the wanted kinds, deterministic)."""
    train = dataset.generate()["train"]
    shots: dict[str, list[dict]] = {}
    for task, kinds in FEWSHOT_KINDS.items():
        chosen, used = [], set()
        for kind in kinds:
            for r in train:
                if r["task"] == task and r["kind"] == kind and r["id"] not in used:
                    chosen.append(r)
                    used.add(r["id"])
                    break
        shots[task] = chosen
    return shots


def build_messages(row: dict, shots: dict[str, list[dict]] | None) -> list[dict]:
    msgs = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]
    for ex in (shots or {}).get(row["task"], []):
        msgs.append({"role": "user", "content": prompts.user_prompt(ex["task"], ex["facts"])})
        msgs.append({"role": "assistant", "content": ex["reference"]})
    msgs.append({"role": "user", "content": prompts.user_prompt(row["task"], row["facts"])})
    return msgs


def run(backend, rows: list[dict], shots, out_path: Path, meta: dict) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # warm-up so first-call compilation does not pollute latency
    backend.generate(build_messages(rows[0], shots), 16)
    scored = []
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for i, row in enumerate(rows):
            gen = backend.generate(build_messages(row, shots), prompts.MAX_TOKENS[row["task"]])
            score = metrics.score_output(row["task"], gen["text"], row["facts"], row["reference"])
            rec = {"id": row["id"], "task": row["task"], "kind": row["kind"], "output": gen["text"],
                   "tokens": gen["tokens"], "prompt_tokens": gen["prompt_tokens"], "seconds": gen["seconds"],
                   "decode_tps": gen["decode_tps"], "peak_mem_gb": gen["peak_mem_gb"], "score": score}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            scored.append(rec)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(rows)}", flush=True)
    summary = metrics.summarise(scored)
    summary["by_task"] = {t: metrics.summarise([r for r in scored if r["task"] == t])
                          for t in sorted({r["task"] for r in scored})}
    summary["peak_mem_gb"] = max((r["peak_mem_gb"] for r in scored), default=0.0) or None
    return summary


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=["base", "fewshot", "finetuned", "gguf"])
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="artifacts/adapters")
    ap.add_argument("--gguf")
    ap.add_argument("--test", default="data/test.jsonl", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--name")
    args = ap.parse_args(argv)

    rows = load_test(args.test, args.limit)
    shots = fewshot_examples() if args.variant == "fewshot" else None
    if args.variant == "gguf":
        from .backends import LlamaServerBackend
        backend = LlamaServerBackend(args.gguf)
        size_path = Path(args.gguf)
    else:
        from .backends import MLXBackend
        backend = MLXBackend(args.model, args.adapter if args.variant == "finetuned" else None)
        size_path = Path(args.model)
    size_bytes = dir_size(size_path)
    if args.variant == "finetuned":
        size_bytes_adapter = dir_size(Path(args.adapter) / "adapters.safetensors")
    else:
        size_bytes_adapter = 0
    meta = {"name": args.name or args.variant, "variant": args.variant, "command": " ".join(sys.argv),
            "model": args.model if args.variant != "gguf" else args.gguf,
            "adapter": args.adapter if args.variant == "finetuned" else None,
            "decoding": "greedy (temperature 0)", "n_test": len(rows),
            "machine": platform.platform(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        summary = run(backend, rows, shots, args.out, meta)
    finally:
        backend.close()
    summary["model_size_bytes"] = size_bytes
    summary["adapter_size_bytes"] = size_bytes_adapter
    summary["meta"] = meta
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_task", "meta")}, indent=2))


if __name__ == "__main__":
    main()
