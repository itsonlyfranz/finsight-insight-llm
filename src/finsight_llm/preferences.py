"""Preference-pair builder for DPO.

Every pair is `(prompt, chosen, rejected)` built from the *training* facts only:

* chosen   = the dataset's reference text (it passes the number guard and the format contract by construction)
* rejected = one of
    - synthetic corruptions of the reference, each a realistic failure that the existing guard / format
      checker / fact-coverage metric catches (`CORRUPTIONS` below), or
    - a sampled output of the SFT model that loses to the reference under the existing metrics
      (`from_sft_samples`).

Nothing here needs MLX; the module is pure Python so the builder is unit-tested in CI.

    python -m finsight_llm.preferences --samples data/preferences/sft_samples.jsonl --out data/preferences
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable

from . import guard, metrics, prompts
from .money import fmt

SEED = 20260613
VALID_FRACTION = 0.10
_FIG = re.compile(r"(?:₱|\$|\bPHP)\s?[0-9][0-9,]*(?:\.[0-9]+)?", re.IGNORECASE)


# ---- reading the existing SFT files --------------------------------------------------------------

def parse_chat_row(row: dict) -> dict:
    """Recover task / facts / reference from a chat-format row of data/{train,valid}.jsonl."""
    msgs = row["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    reference = next(m["content"] for m in msgs if m["role"] == "assistant")
    for task, instr in prompts.INSTRUCTIONS.items():
        suffix = "\n" + instr
        if user.endswith(suffix):
            return {"task": task, "facts": user[: -len(suffix)], "reference": reference}
    raise ValueError("user prompt does not end with a known instruction")


def load_rows(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            r = parse_chat_row(json.loads(line))
            r["id"] = f"{path.stem}-{i:05d}"
            rows.append(r)
    return rows


# ---- helpers ----------------------------------------------------------------------------------------

def _figures(text: str) -> list[re.Match]:
    return list(_FIG.finditer(text))


def _value(token: str) -> float:
    return guard.money_figures(token)[0]


def _fmt_like(token: str, value: float) -> str:
    """Format `value` in the style of `token` (decimals kept when the original had them)."""
    decimals = len(token.split(".")[1]) if "." in token else 0
    prefix = "₱"
    return f"{prefix}{value:,.{decimals}f}"


def _swap_digits(value: float, rng: random.Random) -> float:
    """Transpose two adjacent digits of the integer part (a near-duplicate amount)."""
    s = str(int(value))
    if len(s) < 2:
        return value + 10
    for _ in range(8):
        i = rng.randrange(len(s) - 1)
        if s[i] != s[i + 1]:
            t = s[:i] + s[i + 1] + s[i] + s[i + 2:]
            if t[0] != "0":
                return float(t) + (value - int(value))
    return value * 1.1


def _unused(value: float, facts: str, rng: random.Random) -> float:
    """A perturbation of `value` that is guaranteed not to be in the facts (so the guard must catch it)."""
    allowed = guard.money_figures(facts)
    cands = [_swap_digits(value, rng), value * 1.1, value * 0.9, value + 1000, value * 2, value / 2 + 7]
    rng.shuffle(cands)
    for c in cands:
        c = round(c, 2)
        if c > 0 and all(abs(c - a) >= 0.51 for a in allowed):
            return c
    return round(value * 3 + 17, 2)


# ---- synthetic corruptions: (task, reference, facts, rng) -> rejected text or None ----------------------

def changed_figure(task, ref, facts, rng):
    """One money figure replaced by a near-duplicate / perturbed amount that is not in the facts."""
    ms = _figures(ref)
    if not ms:
        return None
    m = rng.choice(ms)
    new = _fmt_like(m.group(0), _unused(_value(m.group(0)), facts, rng))
    return ref[: m.start()] + new + ref[m.end():]


def invented_figure(task, ref, facts, rng):
    """A plausible-sounding extra figure (annualised / rounded-up) that the facts never gave."""
    ms = _figures(ref)
    if not ms:
        return None
    base = _value(rng.choice(ms).group(0))
    inv = fmt(_unused(base * rng.choice([12, 6, 1.5, 0.25, 3]), facts, rng))
    clause = rng.choice([
        f" That could add up to about {inv} over time.",
        f" Experts say people like you can save {inv} more.",
        f" You could realistically reach {inv} by year end.",
    ])
    if task == "insight":
        head, _, tail = ref.partition("\nACTION:")
        return head.rstrip() + clause + "\nACTION:" + tail
    return ref.rstrip() + clause


def misattributed_figure(task, ref, facts, rng):
    """A figure swapped for a *different* figure from the facts: grounded, but wrong. Fact coverage drops."""
    ms = _figures(ref)
    pool = sorted({m.group(0) for m in _figures(facts)} - {m.group(0) for m in ms}, key=str)
    if not ms or not pool:
        return None
    m = rng.choice(ms)
    return ref[: m.start()] + rng.choice(pool) + ref[m.end():]


def _first_sentence(text: str) -> str:
    return metrics.sentences(text)[0] if metrics.sentences(text) else text


def broken_format(task, ref, facts, rng):
    """Output that breaks the contract the app asks for."""
    if task == "insight":
        ins, _, act = ref.partition("\nACTION:")
        body = ins.replace("INSIGHT:", "").strip()
        act = act.strip()
        return rng.choice([
            f"INSIGHT: {body}",                                      # ACTION line missing
            f"{body} {act}",                                          # contract ignored, one blob
            f"INSIGHT: **{body}**\nACTION: {act}",                   # markdown
            f"Here is your insight:\nINSIGHT: {body}\nACTION: {act}",  # preamble line
            f"INSIGHT: {body}\nACTION: {act} Also, review your budget weekly. Track every purchase.",  # 3 sentences
            f"INSIGHT: {body}\n\nACTION: {act}\nNOTE: figures are estimates.",  # extra line
        ])
    if task == "simulation":
        sents = metrics.sentences(ref)
        return rng.choice([
            _first_sentence(ref),                                     # too short
            "\n".join(f"- {s}" for s in sents),                       # bullet list
            " ".join(sents) + " Keep going. Stay consistent. You can do it.",  # too long
            "**" + ref + "**",                                        # markdown
        ])
    sents = metrics.sentences(ref)
    return rng.choice([
        ref + " Keep it up. You are doing great.",                    # two+ sentences
        ref.rstrip(".!") + ", and remember that consistent saving habits over many months will help you reach "
        "every other milestone you set later on.",                    # > 30 words
        "**" + ref + "**",
        "\n".join(sents) + "\nStay motivated!",
    ])


_VAGUE = {
    "insight": [
        "INSIGHT: Your spending patterns could use some attention, so it is worth taking a closer look.\n"
        "ACTION: Try to spend a little less and save a little more.",
        "INSIGHT: Money habits matter a lot over time and small changes can make a difference.\n"
        "ACTION: Review your budget regularly and stay disciplined.",
        "INSIGHT: There is room to improve how you manage your money this month.\n"
        "ACTION: Consider setting aside more for the future.",
    ],
    "simulation": [
        "Cutting back on spending can help you save more over time. Every little bit counts. "
        "Staying consistent is the key to building savings.",
        "Making changes to your budget is generally a good idea. It can free up some money for other things. "
        "Over time this tends to help.",
    ],
    "goal": [
        "Keep working toward your goal and you will get there eventually.",
        "Saving is a journey, so stay positive and keep going.",
        "Great job so far, and remember that every peso you set aside brings you closer.",
    ],
}
_OFF_TOPIC = {
    "insight": [
        "INSIGHT: Cryptocurrency markets have been volatile lately, so diversifying across several coins may pay off.\n"
        "ACTION: Look into a high-yield investment product that guarantees strong monthly returns.",
        "INSIGHT: The weather has been unpredictable, which affects how people plan their weekends.\n"
        "ACTION: Pick a hobby that fits your schedule and stick with it.",
    ],
    "simulation": [
        "Stock picking requires patience and research. Many investors prefer index funds for the long run. "
        "A guaranteed return is rarely available, but some products come close.",
        "Travel and entertainment are great ways to relax. Plan something fun for the weekend. "
        "You deserve a treat after a long week.",
    ],
    "goal": [
        "Consider moving your savings into a high-return crypto fund to reach your target much faster.",
        "Remember to drink enough water and get plenty of sleep while you work toward the things you want.",
    ],
}


def vague_text(task, ref, facts, rng):
    """In-format but generic: no figures from the facts (fact coverage 0)."""
    return rng.choice(_VAGUE[task])


def off_topic_text(task, ref, facts, rng):
    """In-format but off-topic, with the kind of unsupported claims the system prompt forbids."""
    return rng.choice(_OFF_TOPIC[task])


CORRUPTIONS: dict[str, Callable] = {
    "changed_figure": changed_figure,
    "invented_figure": invented_figure,
    "misattributed_figure": misattributed_figure,
    "broken_format": broken_format,
    "vague_text": vague_text,
    "off_topic_text": off_topic_text,
}
# sampling weights: the failures the product guard rejects (figures, format) dominate
WEIGHTS = {"changed_figure": 3, "invented_figure": 2, "misattributed_figure": 1,
           "broken_format": 3, "vague_text": 1, "off_topic_text": 1}


def make_rejected(row: dict, rng: random.Random, kind: str | None = None,
                  exclude: set[str] | frozenset[str] = frozenset()) -> tuple[str, str] | None:
    """Return `(corruption_name, rejected_text)` for a row, or None. The result is verified to be a real failure."""
    names = [n for n in CORRUPTIONS if n not in exclude]
    if not names:
        return None
    order = [kind] if kind else rng.choices(names, weights=[WEIGHTS[n] for n in names], k=len(names) * 3)
    order += [n for n in names if n not in order]  # deterministic fallback so every type is tried
    for name in dict.fromkeys(order):
        text = CORRUPTIONS[name](row["task"], row["reference"], row["facts"], rng)
        if text and text.strip() != row["reference"].strip() and is_failure(row, text):
            return name, text
    return None


def is_failure(row: dict, text: str) -> bool:
    """Does `text` lose to the reference under the existing checks? (not clean, or misses reference figures)."""
    s = metrics.score_output(row["task"], text, row["facts"], row["reference"])
    return (not s["clean_pass"]) or s["fact_coverage"] < 1.0


# ---- sampled SFT outputs ---------------------------------------------------------------------------------

ROUGE_FLOOR = 0.8  # a clean, fully-covering sample still loses if it drifts this far from the reference


def sft_sample_loses(row: dict, text: str) -> tuple[bool, dict]:
    s = metrics.score_output(row["task"], text, row["facts"], row["reference"])
    loses = (not s["clean_pass"]) or s["fact_coverage"] < 1.0 or s["rouge_l"] < ROUGE_FLOOR
    return loses and text.strip() != row["reference"].strip(), s


def from_sft_samples(row: dict, samples: list[str]) -> tuple[str, dict] | None:
    """Worst losing sample for a row (fails the guard/format first, then lowest fact coverage, then ROUGE-L)."""
    best = None
    for text in samples:
        loses, s = sft_sample_loses(row, text)
        if not loses:
            continue
        key = (s["clean_pass"], s["fact_coverage"], s["rouge_l"])
        if best is None or key < best[0]:
            best = (key, text, s)
    return (best[1], best[2]) if best else None


# ---- assembling the dataset ------------------------------------------------------------------------------

def make_pair(row: dict, rejected: str, rtype: str, source: str) -> dict:
    return {"id": row["id"], "task": row["task"], "source": source, "rejected_type": rtype,
            "prompt": prompts.messages(row["task"], row["facts"]), "chosen": row["reference"], "rejected": rejected}


def split_ids(ids: list[str], seed: int = SEED, frac: float = VALID_FRACTION) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = sorted(ids)
    rng.shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * frac))
    return set(shuffled[n_valid:]), set(shuffled[:n_valid])


def build(rows: list[dict], samples: dict[str, list[str]] | None = None, seed: int = SEED,
          synthetic_per_row: int = 2) -> dict[str, list[dict]]:
    """Pairs for `rows` (training facts), split into train/valid by prompt id.

    Per row: up to `synthetic_per_row` distinct synthetic corruptions, plus the worst losing SFT sample when
    `samples` has one. All pairs of a row land in the same split.
    """
    rng = random.Random(seed)
    train_ids, valid_ids = split_ids([r["id"] for r in rows], seed)
    out: dict[str, list[dict]] = {"train": [], "valid": []}
    for row in rows:
        pairs: list[dict] = []
        used: set[str] = set()
        for _ in range(synthetic_per_row):
            res = make_rejected(row, rng, exclude=used)
            if res:
                used.add(res[0])
                pairs.append(make_pair(row, res[1], res[0], "synthetic"))
        if samples and row["id"] in samples:
            hit = from_sft_samples(row, samples[row["id"]])
            if hit:
                pairs.append(make_pair(row, hit[0], "sft_sample", "sft_sample"))
        out["train" if row["id"] in train_ids else "valid"].extend(pairs)
    return out


def summarise(splits: dict[str, list[dict]]) -> dict:
    return {s: {"pairs": len(p), "prompts": len({x["id"] for x in p}),
                "by_task": dict(Counter(x["task"] for x in p)),
                "by_source": dict(Counter(x["source"] for x in p)),
                "by_rejected_type": dict(Counter(x["rejected_type"] for x in p))}
            for s, p in splits.items()}


def load_samples(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.setdefault(r["id"], []).extend(r["samples"])
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/train.jsonl", type=Path, help="training facts (never the test split)")
    ap.add_argument("--samples", type=Path, help="sampled SFT outputs (jsonl: id, samples[])")
    ap.add_argument("--out", default="data/preferences", type=Path)
    ap.add_argument("--seed", default=SEED, type=int)
    args = ap.parse_args(argv)
    rows = load_rows(args.train)
    samples = load_samples(args.samples) if args.samples and args.samples.exists() else None
    splits = build(rows, samples, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, pairs in splits.items():
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    stats = {"seed": args.seed, "source_rows": len(rows), "used_sft_samples": bool(samples),
             "splits": summarise(splits)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
