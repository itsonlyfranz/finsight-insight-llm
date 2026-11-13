"""Sample the SFT model on *training* prompts to mine hard negatives for DPO.

    python -m finsight_llm.sample_sft --out data/preferences/sft_samples.jsonl --k 4 --temp 1.0

Only `data/train.jsonl` is read; the test split is never touched. Output: one line per prompt with the
`k` sampled completions (the builder in `preferences.py` decides which of them lose to the reference).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import preferences, prompts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="artifacts/adapters")
    ap.add_argument("--train", default="data/train.jsonl", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import mlx.core as mx
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(args.seed)
    model, tok = load(args.model, adapter_path=args.adapter)
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)
    rows = preferences.load_rows(args.train)[: args.limit]
    # the prompt list repeats each row k times, so one batch holds several samples of the same prompt
    jobs = [r for r in rows for _ in range(args.k)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    got: dict[str, list[str]] = {r["id"]: [] for r in rows}
    t0 = time.time()
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i: i + args.batch]
        ids = [tok.apply_chat_template(prompts.messages(r["task"], r["facts"]), add_generation_prompt=True)
               for r in chunk]
        res = batch_generate(model, tok, ids, max_tokens=[prompts.MAX_TOKENS[r["task"]] for r in chunk],
                             sampler=sampler)
        for r, text in zip(chunk, res.texts):
            got[r["id"]].append(text)
        print(f"{min(i + args.batch, len(jobs))}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "samples": got[r["id"]]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
