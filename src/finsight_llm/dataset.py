"""Synthetic dataset generator.

Everything is sampled from seeded random distributions; no real transactions, people, or accounts are
used. Each *profile* is one imaginary early-career earner (income + monthly category totals). A profile is
fed through the ported FinSight engines to obtain the facts the app would send to its model, and a
reference text is composed from the templates in `templates.py`. Splits are made **by profile**, so no
profile's numbers appear in more than one split.

    python -m finsight_llm.dataset --out data
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import date
from pathlib import Path

from . import engine, guard, prompts, templates
from .money import fmt

SEED = 20260512
SPLIT_PROFILES = {"train": 1000, "valid": 80, "test": 160}
SIM_CATEGORIES = ["CAFES_DINING", "TECH_GADGETS", "SUBSCRIPTIONS", "LIFESTYLE", "TRANSPORT", "GROCERIES"]
GOAL_TITLES = ["Emergency Fund", "New Laptop", "Japan Trip", "Motorcycle Down Payment",
               "Pag-IBIG MP2 Top-up", "Home Office Setup", "Wedding Fund", "Board Exam Review"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def _r2(x: float) -> float:
    return round(x, 2)


def sample_profile(rng: random.Random) -> dict:
    """One imaginary earner: monthly take-home income (PHP) and per-category monthly totals."""
    income = float(rng.choice(range(18_000, 150_001, 500)))
    # skew toward early-career incomes
    if rng.random() < 0.6:
        income = float(rng.choice(range(22_000, 70_001, 500)))
    cafe_count = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 22])
    avg_ticket = rng.uniform(120, 480)
    cafe = _r2(cafe_count * avg_ticket)
    totals = {
        "CAFES_DINING": cafe,
        "TECH_GADGETS": _r2(rng.choice([0, 0, 0, rng.uniform(300, 6000), rng.uniform(2500, 18000)])),
        "HOUSING_RENT": _r2(income * rng.uniform(0.12, 0.38)) if rng.random() < 0.9 else 0.0,
        "TRANSPORT": _r2(income * rng.uniform(0.02, 0.10)),
        "SUBSCRIPTIONS": _r2(rng.choice([0, rng.uniform(150, 1800), rng.uniform(1200, 4800), rng.uniform(2600, 5200)])),
        "GROCERIES": _r2(income * rng.uniform(0.05, 0.15)),
        "LIFESTYLE": _r2(rng.choice([0, rng.uniform(300, 3000), rng.uniform(2000, 9000)])),
    }
    if rng.random() < 0.08:  # a few profiles with no essentials recorded (engine falls back to a default)
        totals["HOUSING_RENT"] = totals["GROCERIES"] = totals["TRANSPORT"] = 0.0
    return {"income": income, "totals": totals, "cafe_count": cafe_count}


def insight_facts(ins: engine.Insight, income: float) -> str:
    kind = ins.type.lower().replace("_", " ")
    s = f"Monthly take-home income: {fmt(income)}\nInsight type: {kind}\nHeadline: {ins.title}\nFindings: {ins.description}\n"
    if ins.action_suggestion:
        s += f"Suggested action: {ins.action_suggestion}\n"
    if ins.impact_amount is not None:
        s += f"Amount at stake: {fmt(ins.impact_amount)}\n"
    return s


def insight_example(rng: random.Random, profile: dict) -> dict | None:
    insights = engine.generate_insights(profile["totals"], profile["cafe_count"], profile["income"])
    if not insights:
        return None
    ins = insights[rng.randrange(len(insights))]
    facts = insight_facts(ins, profile["income"])
    text = templates.pick(rng, templates.INSIGHT_TEXT[ins.id])(ins.values)
    action = templates.pick(rng, templates.INSIGHT_ACTION[ins.id])(ins.values)
    return {"task": "insight", "kind": ins.id, "facts": facts, "reference": f"INSIGHT: {text}\nACTION: {action}"}


def simulation_example(rng: random.Random, profile: dict) -> dict | None:
    cats = rng.sample(SIM_CATEGORIES, rng.choice([1, 2, 2, 3]))
    cuts = {c: rng.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]) for c in cats}
    sim = engine.simulate(profile["totals"], cuts)
    if sim.monthly <= 0.5:
        return None
    cut_text = "; ".join(f"{engine.CATEGORIES[c][0]} cut {round(f * 100)}%" for c, f in cuts.items())
    facts = (f"Planned spending cuts: {cut_text}\nFreed cashflow per month: {fmt(sim.monthly)}\n"
             f"Savings after 3 months: {fmt(sim.p3)}\nSavings after 6 months: {fmt(sim.p6)}\n"
             f"Savings after 12 months: {fmt(sim.p12)} (4.5% yearly yield assumed)\n")
    v = {"monthly": sim.monthly, "p3": sim.p3, "p6": sim.p6, "p12": sim.p12, "cuts": cut_text}
    return {"task": "simulation", "kind": "simulation", "facts": facts,
            "reference": templates.pick(rng, templates.SIM_TEXT)(v)}


def goal_example(rng: random.Random, profile: dict) -> dict:
    target = float(rng.choice(range(15_000, 400_001, 5_000)))
    saved = float(rng.choice([0, rng.randrange(0, int(target), 500), rng.randrange(0, int(target), 500), target]))
    freed = float(rng.choice([0, 0, rng.randrange(500, 9000, 100)]))
    months = engine.months_to_goal(target, saved, freed)
    ref = date(2026, rng.randrange(1, 13), 15)
    total_months = ref.year * 12 + (ref.month - 1) + months
    when = f"{MONTHS[total_months % 12]} {total_months // 12}"
    remaining = max(target - saved, 0.0)
    progress = int(min(saved / target, 1.0) * 100)
    title = rng.choice(GOAL_TITLES)
    facts = (f"Savings goal: {title}\nTarget: {fmt(target)}; saved so far: {fmt(saved)}\n"
             f"Remaining: {fmt(remaining)}\nProgress: {progress}%\n"
             f"Extra freed cashflow per month from the simulator: {fmt(freed)}\n"
             f"Projected completion: {when} ({months} months)\n")
    v = {"title": title, "target": target, "saved": saved, "remaining": remaining, "progress": progress,
         "freed": freed, "months": months, "date": when}
    pool = templates.GOAL_TEXT_DONE if months == 0 else templates.GOAL_TEXT
    return {"task": "goal", "kind": "goal", "facts": facts, "reference": templates.pick(rng, pool)(v)}


def build_split(rng: random.Random, n_profiles: int) -> list[dict]:
    rows: list[dict] = []
    for _ in range(n_profiles):
        p = sample_profile(rng)
        cands = [insight_example(rng, p)]
        if rng.random() < 0.5:
            cands.append(simulation_example(rng, p))
        if rng.random() < 0.3:
            cands.append(goal_example(rng, p))
        rows.extend(c for c in cands if c)
    return rows


def generate(seed: int = SEED, sizes: dict[str, int] | None = None) -> dict[str, list[dict]]:
    sizes = sizes or SPLIT_PROFILES
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for split, n in sizes.items():
        rows = []
        for row in build_split(rng, n):
            key = row["facts"]
            if key in seen:  # never let an identical input appear in two places
                continue
            seen.add(key)
            row["id"] = f"{split}-{len(rows):05d}"
            row["split"] = split
            if not guard.is_grounded(row["reference"], row["facts"]):
                raise AssertionError(f"reference not grounded: {row['id']}")
            rows.append(row)
        out[split] = rows
    return out


def to_chat(row: dict) -> dict:
    msgs = prompts.messages(row["task"], row["facts"]) + [{"role": "assistant", "content": row["reference"]}]
    return {"messages": msgs}


def write(out_dir: Path, splits: dict[str, list[dict]]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, rows in splits.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                rec = to_chat(r)
                if split == "test":  # test keeps its metadata for evaluation; mlx-lm ignores extra keys
                    rec.update({k: r[k] for k in ("id", "task", "kind", "facts", "reference")})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[split] = {"examples": len(rows), "by_task": dict(Counter(r["task"] for r in rows)),
                        "by_kind": dict(Counter(r["kind"] for r in rows))}
    (out_dir / "stats.json").write_text(json.dumps({"seed": SEED, "splits": stats}, indent=2) + "\n")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument("--seed", default=SEED, type=int)
    args = ap.parse_args()
    stats = write(args.out, generate(args.seed))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
