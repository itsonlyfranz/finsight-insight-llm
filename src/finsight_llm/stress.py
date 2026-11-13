"""Adversarial stress set: facts with many figures, several of them near-duplicates of the ones that matter.

The ordinary generator gives the model only the figures it needs. Here each prompt additionally carries a block
of "prior-month" figures that are **decoys**: digit transpositions, off-by-a-little amounts and rounded copies of the
real figures. A model that copies the wrong one produces text that still passes the number guard (the decoy is in the
facts) but states the wrong number, which the fact-coverage metric and `decoy_rate` below expose. A model that
invents a figure fails the guard.

The references are the ordinary templated texts for the *real* figures, so every existing metric applies unchanged.
The stress set is generated from its own seed, is never used for training, and shares no facts with train/valid/test.

    python -m finsight_llm.stress --out data/stress.jsonl
    python -m finsight_llm.stress --score results/raw/dpo-stress.jsonl      # adds decoy_rate to a stress run
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from . import dataset, guard, prompts
from .money import fmt

STRESS_SEED = 20260614
STRESS_PROFILES = 165  # about 230 rows, comparable in size to the 246-row test set
HEADER = "Prior-month figures (for comparison only, do not report them as current):\n"
_FIG = re.compile(r"₱[0-9][0-9,]*(?:\.[0-9]+)?")


def _transpose(value: float, rng: random.Random) -> float:
    s = str(int(value))
    idx = [i for i in range(len(s) - 1) if s[i] != s[i + 1] and not (i == 0 and s[1] == "0")]
    if not idx:
        return value + 45
    i = rng.choice(idx)
    return float(s[:i] + s[i + 1] + s[i] + s[i + 2:]) + (value - int(value))


def decoys_for(facts: str, rng: random.Random, n: int = 8) -> list[float]:
    """Near-duplicates of the figures in `facts`, none of which collides with a real figure (tolerance 0.51)."""
    real = sorted(set(guard.money_figures(facts)))
    real = [v for v in real if v >= 100] or real
    out: list[float] = []
    makers = [lambda v: _transpose(v, rng), lambda v: v + rng.choice([-1, 1]) * rng.choice([1.0, 5.0, 10.0, 50.0]),
              lambda v: round(v * rng.choice([0.98, 1.02, 1.05])), lambda v: v + rng.choice([-0.9, 0.9, 1.1])]
    tries = 0
    while len(out) < n and tries < 200 and real:
        tries += 1
        v = round(rng.choice(makers)(rng.choice(real)), 2)
        allowed = real + out
        if v > 0 and all(abs(v - a) >= 0.51 for a in allowed):
            out.append(v)
    return out


def add_decoys(row: dict, rng: random.Random, n: int = 8) -> dict:
    ds = decoys_for(row["facts"], rng, n)
    labels = ["Last month", "Two months ago", "Previous cycle", "Same month last year", "Earlier estimate"]
    lines = "".join(f"- {labels[i % len(labels)]}: {fmt(v)}\n" for i, v in enumerate(ds))
    new = dict(row)
    new["facts"] = row["facts"] + HEADER + lines
    new["decoys"] = ds
    return new


def generate() -> list[dict]:
    rng = random.Random(STRESS_SEED)
    taken = {r["facts"] for rows in dataset.generate().values() for r in rows}  # never reuse a train/valid/test input
    rows = []
    for r in dataset.build_split(rng, STRESS_PROFILES):
        if r["facts"] in taken:
            continue
        taken.add(r["facts"])
        rows.append(add_decoys(r, rng))
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            rec = dataset.to_chat(r)
            rec.update({"id": f"stress-{i:05d}", "task": r["task"], "kind": r["kind"], "facts": r["facts"],
                        "reference": r["reference"], "decoys": r["decoys"]})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def decoy_rate(output: str, decoys: list[float]) -> bool:
    """True when the output states any decoy figure (a grounded but wrong number)."""
    return any(abs(g - d) < 0.51 for g in guard.money_figures(output) for d in decoys)


def score_decoys(raw_path: Path, stress_path: Path) -> dict:
    by_id = {json.loads(l)["id"]: json.loads(l) for l in stress_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    hits = n = 0
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if "_meta" in rec:
            continue
        n += 1
        hits += decoy_rate(rec["output"], by_id[rec["id"]]["decoys"])
    return {"n": n, "decoy_outputs": hits, "decoy_rate_pct": 100.0 * hits / n if n else None}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/stress.jsonl"))
    ap.add_argument("--score", type=Path, help="raw jsonl from an evaluation on the stress set: print decoy rate")
    args = ap.parse_args(argv)
    if args.score:
        print(json.dumps(score_decoys(args.score, args.out), indent=2))
        return
    rows = generate()
    write(args.out, rows)
    print(f"wrote {len(rows)} stress rows to {args.out}")


if __name__ == "__main__":
    main()
