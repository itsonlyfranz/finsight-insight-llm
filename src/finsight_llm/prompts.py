"""Prompts exactly as the FinSight app builds them (AiEnhancer.kt), so train/eval inputs match production."""
from __future__ import annotations

from .money import SYMBOL

SYSTEM_PROMPT = (
    "You are FinSight, a concise personal-finance coach for early-career earners. "
    "Use only the numbers provided; never invent amounts, rates, products or guarantees. "
    f"All amounts are in Philippine pesos ({SYMBOL}); write them with the {SYMBOL} sign, never with a dollar sign. "
    "Plain text only, no markdown, no emojis."
)

INSIGHT_INSTRUCTION = (
    "Rewrite the findings and the suggested action in a warm, specific voice for the user. "
    "Reply in exactly two lines:\nINSIGHT: <at most two sentences>\nACTION: <one concrete sentence>"
)
SIMULATION_INSTRUCTION = "Explain in two or three sentences what these cuts mean for the user, in plain language."
GOAL_INSTRUCTION = "Write one encouraging sentence (at most 30 words) about the next milestone for this goal."

INSTRUCTIONS = {"insight": INSIGHT_INSTRUCTION, "simulation": SIMULATION_INSTRUCTION, "goal": GOAL_INSTRUCTION}
# maxTokens the app uses for each feature
MAX_TOKENS = {"insight": 160, "simulation": 140, "goal": 70}


def user_prompt(task: str, facts: str) -> str:
    return f"{facts}\n{INSTRUCTIONS[task]}"


def messages(task: str, facts: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(task, facts)},
    ]
