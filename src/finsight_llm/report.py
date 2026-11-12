"""Collect results/raw/*.summary.json into results/RESULTS.md and results/results.json."""
from __future__ import annotations

import json
from pathlib import Path

ORDER = ["base", "fewshot", "finetuned", "gguf-f16", "gguf-Q8_0", "gguf-Q5_K_M", "gguf-Q4_K_M"]
LABEL = {
    "base": "Base model, zero-shot (MLX bf16)",
    "fewshot": "Base model + 2-shot prompt (MLX bf16)",
    "finetuned": "Fine-tuned (LoRA), zero-shot (MLX bf16)",
    "gguf-f16": "Fine-tuned, GGUF f16",
    "gguf-Q8_0": "Fine-tuned, GGUF Q8_0",
    "gguf-Q5_K_M": "Fine-tuned, GGUF Q5_K_M",
    "gguf-Q4_K_M": "Fine-tuned, GGUF Q4_K_M",
}


def _f(x, nd=1, suffix=""):
    return "n/a" if x is None else f"{x:.{nd}f}{suffix}"


def _mb(b):
    return f"{b / 1e6:,.0f} MB" if b else "n/a"


def load(raw_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(raw_dir.glob("*.summary.json")):
        s = json.loads(p.read_text())
        out[s["meta"]["name"]] = s
    return out


def table(results: dict[str, dict], names: list[str]) -> str:
    head = ("| Model | Number fidelity | Format valid | App-accepted | ROUGE-L | Fact coverage | "
            "Decode tok/s | Mean latency | Size on disk |\n|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for n in names:
        s = results.get(n)
        if not s:
            continue
        size = s["model_size_bytes"] + s.get("adapter_size_bytes", 0)
        rows.append(f"| {LABEL.get(n, n)} | {_f(s['number_fidelity_pct'], 1, '%')} | {_f(s['format_valid_pct'], 1, '%')} | "
                    f"{_f(s['accepted_pct'], 1, '%')} | {_f(s['rouge_l'], 3)} | {_f(s['fact_coverage'], 3)} | "
                    f"{_f(s['decode_tok_per_s'], 1)} | {_f(s['mean_latency_s'], 2, ' s')} | {_mb(size)} |")
    return head + "\n".join(rows) + "\n"


def per_task(results: dict[str, dict], names: list[str], metric: str, title: str) -> str:
    tasks = ["insight", "simulation", "goal"]
    head = f"**{title}**\n\n| Model | " + " | ".join(tasks) + " |\n|---|" + "---|" * len(tasks) + "\n"
    rows = []
    for n in names:
        s = results.get(n)
        if not s:
            continue
        cells = [_f(s["by_task"].get(t, {}).get(metric), 1 if "pct" in metric else 3,
                    "%" if "pct" in metric else "") for t in tasks]
        rows.append(f"| {LABEL.get(n, n)} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def main(raw_dir: str = "results/raw", out_dir: str = "results") -> None:
    results = load(Path(raw_dir))
    names = [n for n in ORDER if n in results] + [n for n in results if n not in ORDER]
    md = ["# Results (held-out test split)\n", f"n = {next(iter(results.values()))['n']} test examples per model, greedy decoding.\n",
          table(results, names)]
    md.append(per_task(results, names, "number_fidelity_pct", "Number fidelity by task"))
    md.append(per_task(results, names, "format_valid_pct", "Format validity by task"))
    md.append(per_task(results, names, "rouge_l", "ROUGE-L by task"))
    Path(out_dir, "RESULTS.md").write_text("\n".join(md))
    Path(out_dir, "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
