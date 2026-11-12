"""Python port of FinSight's output handling: `NumberGuard`, `OutputCleaner` and the two-line parser.

The Android app (`AiEnhancer.kt`) discards any generated text that mentions a money figure that was not in
the facts it sent to the model. This module reproduces that rule so the evaluation measures exactly the
property the product enforces.
"""
from __future__ import annotations

import re

# Peso is the app currency ("₱" or "PHP"); a model that still writes "$" is checked the same way.
_MONEY = re.compile(r"(?:₱|\$|\bPHP)\s?([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_DANGLING_THINK = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_MARKDOWN = re.compile(r"[*_`#>]+")
MAX_CHARS = 600


def money_figures(text: str) -> list[float]:
    """Every money figure in `text` as a float (thousands separators removed)."""
    out: list[float] = []
    for m in _MONEY.finditer(text):
        raw = m.group(1).replace(",", "").rstrip(".")
        try:
            out.append(float(raw))
        except ValueError:
            continue  # the app treats an unparseable figure as harmless
    return out


def is_grounded(generated: str, facts: str) -> bool:
    """True when every money figure in `generated` matches (within 0.51) a figure in `facts`."""
    allowed = money_figures(facts)
    return all(any(abs(a - v) < 0.51 for a in allowed) for v in money_figures(generated))


def ungrounded_figures(generated: str, facts: str) -> list[float]:
    allowed = money_figures(facts)
    return [v for v in money_figures(generated) if not any(abs(a - v) < 0.51 for a in allowed)]


def clean(raw: str) -> str:
    """Remove reasoning blocks / markdown / quoting and bound the length (OutputCleaner.clean)."""
    text = _THINK_BLOCK.sub("", raw)
    if "</think>" in text.lower():
        text = _DANGLING_THINK.sub("", text)
    text = _MARKDOWN.sub("", text).strip().strip('"').strip()
    if len(text) > MAX_CHARS:
        cut = text[:MAX_CHARS]
        head = cut.rsplit(".", 1)[0] if "." in cut else cut
        text = cut if len(head) < 40 else head + "."
    return text


def parse_two_lines(text: str) -> tuple[str | None, str | None]:
    """Parse `INSIGHT: ...` / `ACTION: ...`; plain text is accepted as the insight (like the app)."""
    insight = action = None
    for line in text.splitlines():
        t = line.strip()
        low = t.lower()
        if low.startswith("insight:"):
            insight = t.split(":", 1)[1].strip()
        elif low.startswith("action:"):
            action = t.split(":", 1)[1].strip()
    if insight is None and action is None and text.strip():
        insight = " ".join(line.strip() for line in text.splitlines()).strip()
    return insight, action
