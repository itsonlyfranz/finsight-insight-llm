"""SFT vs SFT+DPO report: `results/dpo/RESULTS.md` and `results/dpo/results.json`.

    python -m finsight_llm.dpo_report

Inputs (all produced by `scripts/dpo.sh`):
  results/raw/finetuned.jsonl              SFT on the 246-row test set (the committed baseline run)
  results/dpo/raw/dpo.jsonl                SFT+DPO on the same test set
  results/dpo/raw/sft-stress.jsonl         SFT on the adversarial stress set
  results/dpo/raw/dpo-stress.jsonl         SFT+DPO on the stress set
  results/dpo/raw/dpo-gguf-Q5_K_M.jsonl    SFT+DPO exported to GGUF Q5_K_M, on the test set
Per-row scores are paired by example id, so each difference comes with a paired-bootstrap 95% interval.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

from . import stress as stress_mod

METRICS = [("grounded", "Number fidelity", 100.0, "%"), ("format_valid", "Format valid", 100.0, "%"),
           ("accepted", "App-accepted", 100.0, "%"), ("rouge_l", "ROUGE-L", 1.0, ""),
           ("fact_coverage", "Fact coverage", 1.0, "")]
RUNS = {
    "sft_test": "results/raw/finetuned.jsonl",
    "dpo_test": "results/dpo/raw/dpo.jsonl",
    "sft_stress": "results/dpo/raw/sft-stress.jsonl",
    "dpo_stress": "results/dpo/raw/dpo-stress.jsonl",
    "dpo_gguf_q5": "results/dpo/raw/dpo-gguf-Q5_K_M.jsonl",
    "sft_gguf_q5": "results/raw/gguf-Q5_K_M.jsonl",
}


def read_run(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()[1:] if l.strip())}


def values(run: dict[str, dict], key: str, scale: float) -> dict[str, float]:
    return {i: float(r["score"][key]) * scale for i, r in run.items()}


def paired_ci(a: dict[str, float], b: dict[str, float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (b - a) over shared ids with a paired-bootstrap 95% interval."""
    ids = sorted(set(a) & set(b))
    d = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return mean(d), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def fmt_val(x: float, unit: str) -> str:
    return f"{x:.1f}%" if unit == "%" else f"{x:.3f}"


def compare_table(sft: dict, dpo: dict, extra: dict | None = None) -> str:
    head = "| Metric | SFT | SFT + DPO | Δ (DPO − SFT) | 95% CI (paired bootstrap) |\n|---|---|---|---|---|\n"
    rows = []
    for key, label, scale, unit in METRICS:
        a, b = values(sft, key, scale), values(dpo, key, scale)
        d, lo, hi = paired_ci(a, b)
        sign = "+" if d >= 0 else "−"
        dtxt = f"{sign}{abs(d):.1f} pp" if unit == "%" else f"{sign}{abs(d):.3f}"
        ci = (f"[{lo:+.1f}, {hi:+.1f}] pp" if unit == "%" else f"[{lo:+.3f}, {hi:+.3f}]")
        rows.append(f"| {label} | {fmt_val(mean(a.values()), unit)} | {fmt_val(mean(b.values()), unit)} | {dtxt} | {ci} |")
    if extra:
        for label, (x, y) in extra.items():
            rows.append(f"| {label} | {x:.1f}% | {y:.1f}% | {y - x:+.1f} pp | |")
    return head + "\n".join(rows) + "\n"


def by_task(sft: dict, dpo: dict, key: str, scale: float, unit: str) -> str:
    out = ["| Task | n | SFT | SFT + DPO |", "|---|---|---|---|"]
    for t in ("insight", "simulation", "goal"):
        ids = [i for i, r in sft.items() if r["task"] == t and i in dpo]
        out.append(f"| {t} | {len(ids)} | {fmt_val(mean(float(sft[i]['score'][key]) * scale for i in ids), unit)} | "
                   f"{fmt_val(mean(float(dpo[i]['score'][key]) * scale for i in ids), unit)} |")
    return "\n".join(out) + "\n"


def decoys(run: dict[str, dict], stress_rows: dict[str, dict]) -> float:
    hits = sum(stress_mod.decoy_rate(r["output"], stress_rows[i]["decoys"]) for i, r in run.items())
    return 100.0 * hits / len(run)


def speed(path: Path) -> dict:
    s = json.loads(path.with_suffix(".summary.json").read_text())
    return {k: s.get(k) for k in ("decode_tok_per_s", "mean_latency_s", "model_size_bytes", "adapter_size_bytes")}


def main() -> None:
    runs = {k: read_run(Path(p)) for k, p in RUNS.items() if Path(p).exists()}
    stress_rows = {r["id"]: r for r in (json.loads(l) for l in Path("data/stress.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    md = ["# SFT vs SFT + DPO\n",
          "Greedy decoding, same prompts, same scoring code. Differences are paired by example; the interval is a "
          "2,000-resample paired bootstrap (seed 0).\n"]
    summary: dict = {}
    if {"sft_test", "dpo_test"} <= runs.keys():
        n = len(runs["dpo_test"])
        md.append(f"## Held-out test split (n = {n})\n")
        md.append(compare_table(runs["sft_test"], runs["dpo_test"]))
        for key, label, scale, unit in METRICS[3:]:
            md.append(f"**{label} by task**\n\n" + by_task(runs["sft_test"], runs["dpo_test"], key, scale, unit))
        summary["test"] = {m[0]: [mean(values(runs[r], m[0], m[2]).values()) for r in ("sft_test", "dpo_test")] for m in METRICS}
    if {"sft_stress", "dpo_stress"} <= runs.keys():
        n = len(runs["dpo_stress"])
        d = (decoys(runs["sft_stress"], stress_rows), decoys(runs["dpo_stress"], stress_rows))
        md.append(f"## Adversarial stress set (n = {n})\n")
        md.append("Each prompt carries 8 near-duplicate decoy figures (digit transpositions, off-by-a-little amounts). "
                  "*Decoy rate* is the share of outputs that state any decoy: grounded (the guard passes it) but wrong.\n")
        md.append(compare_table(runs["sft_stress"], runs["dpo_stress"], {"Decoy rate (lower is better)": d}))
        for key, label, scale, unit in METRICS[:1] + METRICS[3:]:
            md.append(f"**{label} by task**\n\n" + by_task(runs["sft_stress"], runs["dpo_stress"], key, scale, unit))
        summary["stress"] = {m[0]: [mean(values(runs[r], m[0], m[2]).values()) for r in ("sft_stress", "dpo_stress")] for m in METRICS}
        summary["stress"]["decoy_rate_pct"] = list(d)
    if "dpo_gguf_q5" in runs:
        md.append("## GGUF Q5_K_M (test split)\n")
        head = "| Model | Number fidelity | Format valid | ROUGE-L | Fact coverage | Decode tok/s | Mean latency | Size |\n|---|---|---|---|---|---|---|---|\n"
        rows = []
        for key, label in (("sft_gguf_q5", "SFT, GGUF Q5_K_M"), ("dpo_gguf_q5", "SFT + DPO, GGUF Q5_K_M")):
            if key not in runs:
                continue
            r, sp = runs[key], speed(Path(RUNS[key]))
            m = lambda k, s=1.0: mean(values(r, k, s).values())  # noqa: E731
            rows.append(f"| {label} | {m('grounded', 100):.1f}% | {m('format_valid', 100):.1f}% | {m('rouge_l'):.3f} | "
                        f"{m('fact_coverage'):.3f} | {sp['decode_tok_per_s']:.1f} | {sp['mean_latency_s']:.2f} s | "
                        f"{sp['model_size_bytes'] / 1e6:,.0f} MB |")
            summary.setdefault("gguf_q5", {})[key] = {"rouge_l": m("rouge_l"), "fact_coverage": m("fact_coverage"), **sp}
        md.append(head + "\n".join(rows) + "\n")
    Path("results/dpo").mkdir(parents=True, exist_ok=True)
    Path("results/dpo/RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    Path("results/dpo/results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
