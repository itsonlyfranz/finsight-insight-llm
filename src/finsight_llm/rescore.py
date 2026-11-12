"""Recompute per-row scores and summaries from stored raw outputs (no model is run).

    python -m finsight_llm.rescore

Generation outputs, token counts, timings, sizes and metadata are kept as measured; only the scores
derived from the stored text are recomputed against data/test.jsonl, then the report is regenerated.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import metrics, report
from .evaluate import load_test


def rescore_file(raw: Path, test_rows: dict[str, dict]) -> None:
    lines = raw.read_text(encoding="utf-8").splitlines()
    meta, recs = lines[0], [json.loads(ln) for ln in lines[1:] if ln.strip()]
    for rec in recs:
        t = test_rows[rec["id"]]
        rec["score"] = metrics.score_output(rec["task"], rec["output"], t["facts"], t["reference"])
    raw.write_text("\n".join([meta] + [json.dumps(r, ensure_ascii=False) for r in recs]) + "\n", encoding="utf-8")

    summ_path = raw.with_suffix(".summary.json")
    old = json.loads(summ_path.read_text())
    new = metrics.summarise(recs)
    new["by_task"] = {t: metrics.summarise([r for r in recs if r["task"] == t]) for t in sorted({r["task"] for r in recs})}
    for k in ("peak_mem_gb", "model_size_bytes", "adapter_size_bytes", "meta"):
        new[k] = old[k]
    summ_path.write_text(json.dumps(new, indent=2) + "\n")


def main(raw_dir: str = "results/raw", test: str = "data/test.jsonl") -> None:
    test_rows = {r["id"]: r for r in load_test(Path(test))}
    for raw in sorted(Path(raw_dir).glob("*.jsonl")):
        rescore_file(raw, test_rows)
    report.main(raw_dir)


if __name__ == "__main__":
    main()
