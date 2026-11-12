"""Per-output metrics: format validity, number fidelity (guard), reference similarity, fact coverage."""
from __future__ import annotations

import re
from statistics import mean

from . import guard

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9₱.,%]+")


def sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text.strip()) if s]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1 over lowercase word tokens."""
    c, r = _tokens(candidate), _tokens(reference)
    lcs = _lcs(c, r)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(c), lcs / len(r)
    return 2 * p * rec / (p + rec)


def fact_coverage(candidate: str, reference: str) -> float:
    """Share of the reference's money figures that the candidate also states (1.0 if reference has none)."""
    ref = set(guard.money_figures(reference))
    if not ref:
        return 1.0
    got = guard.money_figures(candidate)
    return sum(any(abs(g - r) < 0.51 for g in got) for r in ref) / len(ref)


def format_valid(task: str, raw: str) -> bool:
    """Does the *raw* model output follow the contract the app asks for?

    insight:    exactly two lines, `INSIGHT:` (<= 2 sentences) then `ACTION:` (1 sentence), plain text
    simulation: 2-3 sentences of plain text
    goal:       1 sentence, <= 30 words
    """
    text = raw.strip()
    if not text or "<think>" in text.lower() or re.search(r"[*_`#]", text):
        return False
    if task == "insight":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) != 2 or not lines[0].upper().startswith("INSIGHT:") or not lines[1].upper().startswith("ACTION:"):
            return False
        ins, act = lines[0].split(":", 1)[1].strip(), lines[1].split(":", 1)[1].strip()
        return bool(ins) and bool(act) and len(sentences(ins)) <= 2 and len(sentences(act)) == 1
    if "\n" in text:
        return False
    n = len(sentences(text))
    if task == "simulation":
        return 2 <= n <= 3
    if task == "goal":
        return n == 1 and len(text.split()) <= 30
    raise ValueError(task)


def score_output(task: str, raw: str, facts: str, reference: str) -> dict:
    cleaned = guard.clean(raw)
    insight, action = guard.parse_two_lines(cleaned) if task == "insight" else (cleaned, None)
    parsed = (insight or "") + (" " + action if action else "")
    if task == "insight" and insight is None:
        parsed_ok = False
    else:
        parsed_ok = bool(parsed.strip())
    grounded = guard.is_grounded(parsed or cleaned, facts)
    fmt_ok = format_valid(task, raw)
    ref_text = reference.replace("INSIGHT:", "").replace("ACTION:", "") if task == "insight" else reference
    cand = parsed or cleaned
    return {
        "format_valid": fmt_ok,
        "parsed": parsed_ok,
        "grounded": grounded,
        # what the app would actually show: parsed, non-empty and passing the guard (otherwise: deterministic fallback)
        "accepted": parsed_ok and grounded,
        "clean_pass": fmt_ok and grounded,
        "ungrounded_figures": guard.ungrounded_figures(parsed or cleaned, facts),
        "rouge_l": rouge_l(cand, ref_text),
        "fact_coverage": fact_coverage(cand, reference),
    }


def summarise(rows: list[dict]) -> dict:
    """Aggregate scored rows (each with 'score', 'tokens', 'seconds', 'tps')."""
    n = len(rows)
    if n == 0:
        return {}
    pct = lambda key: 100.0 * sum(r["score"][key] for r in rows) / n  # noqa: E731
    gen = [r for r in rows if r.get("seconds")]
    total_tokens = sum(r["tokens"] for r in gen)
    total_seconds = sum(r["seconds"] for r in gen)
    lat = sorted(r["seconds"] for r in gen)
    return {
        "n": n,
        "number_fidelity_pct": pct("grounded"),
        "format_valid_pct": pct("format_valid"),
        "accepted_pct": pct("accepted"),
        "clean_pass_pct": pct("clean_pass"),
        "rouge_l": mean(r["score"]["rouge_l"] for r in rows),
        "fact_coverage": mean(r["score"]["fact_coverage"] for r in rows),
        "decode_tok_per_s": (total_tokens / total_seconds) if total_seconds else None,
        "mean_latency_s": mean(lat) if lat else None,
        "p95_latency_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
        "mean_output_tokens": (total_tokens / len(gen)) if gen else None,
    }
