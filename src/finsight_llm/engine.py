"""Python port of FinSight's deterministic engines, used to produce realistic *facts*.

`generate_insights` mirrors `AiInsightsEngine.generateInsights` (finsight-android) and `simulate` mirrors
`ForwardSimulatorEngine.simulate`. The Android app feeds the *output* of these engines to the language
model as "facts"; the model's job is only to re-phrase them. Working from the same engines keeps the
synthetic facts in exactly the shape (and with the same arithmetic) the product produces.

Inputs are category totals rather than individual transactions: the engines only ever use per-category
sums and the cafe visit count, so this is equivalent for fact generation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .money import fmt

# (display name, default monthly budget) as in TransactionCategory.kt
CATEGORIES: dict[str, tuple[str, float]] = {
    "CAFES_DINING": ("Cafes & Dining", 5000.0),
    "TECH_GADGETS": ("Tech & Gadgets", 3500.0),
    "HOUSING_RENT": ("Rent & Housing", 14000.0),
    "TRANSPORT": ("Transport & Commute", 2100.0),
    "SUBSCRIPTIONS": ("Subscriptions", 2500.0),
    "GROCERIES": ("Groceries & Essentials", 4200.0),
    "LIFESTYLE": ("Lifestyle & Leisure", 2800.0),
}


@dataclass
class Insight:
    id: str
    title: str
    description: str
    type: str  # RISK_NUDGE | GROWTH_FOCUS | TIP
    impact_amount: float | None
    action_suggestion: str | None
    severity: str
    # numbers the description/action were built from; used to write grounded reference texts
    values: dict[str, float] = field(default_factory=dict)


def _pct(x: float) -> str:
    return f"{x:.1f}"


def generate_insights(
    totals: dict[str, float],
    cafe_count: int,
    monthly_income: float,
    custom_budgets: dict[str, float] | None = None,
) -> list[Insight]:
    custom_budgets = custom_budgets or {}
    if not any(v > 0 for v in totals.values()):
        return []  # app: nothing recorded -> say nothing
    out: list[Insight] = []

    def budget(cat: str) -> float:
        return custom_budgets.get(cat, CATEGORIES[cat][1])

    # 1. Cafe hopping
    cafe_spend = totals.get("CAFES_DINING", 0.0)
    cafe_pct = cafe_spend / monthly_income * 100.0 if monthly_income > 0 else 0.0
    cafe_budget = budget("CAFES_DINING")
    if cafe_count > 8 or cafe_pct > 15.0 or cafe_spend > cafe_budget:
        if cafe_pct > 20.0 or cafe_spend > cafe_budget * 1.5:
            sev = "HIGH"
        elif cafe_pct > 15.0 or cafe_count > 10:
            sev = "MEDIUM"
        else:
            sev = "LOW"
        desc = f"You've recorded {cafe_count} cafe & dining outings totaling {fmt(cafe_spend)}"
        if monthly_income > 0:
            desc += f" (~{_pct(cafe_pct)}% of your monthly income)"
        desc += ". Small frequent swipes easily slip under the radar for junior engineers."
        savings = max(cafe_spend * 0.25, 400.0)
        out.append(Insight(
            "insight_cafe_lifestyle_inflation", "High Cafe & Dining Velocity", desc, "RISK_NUDGE", cafe_spend,
            f"Try a 2-day homebrew routine during weekdays to reclaim ~{fmt(savings)}/mo while keeping social weekend brunches intact.",
            sev, {"count": cafe_count, "spend": cafe_spend, "pct": cafe_pct, "savings": savings},
        ))

    # 2. Tech gadget spikes
    tech = totals.get("TECH_GADGETS", 0.0)
    tech_budget = budget("TECH_GADGETS")
    tech_pct = tech / monthly_income * 100.0 if monthly_income > 0 else 0.0
    if tech > tech_budget or tech_pct > 10.0:
        sev = "HIGH" if tech > tech_budget * 1.5 else "MEDIUM"
        over = tech - tech_budget
        desc = f"Tech purchases reached {fmt(tech)} this cycle"
        desc += f" ({fmt(over)} over your {fmt(tech_budget)} target)." if over > 0 else "."
        out.append(Insight(
            "insight_tech_gadget_spike", "Tech & Gadget Spend Spike", desc, "RISK_NUDGE", tech,
            "Adopt a 14-day cooling-off rule on new peripherals and mechanical keyboards to avoid dopamine impulse buys.",
            sev, {"spend": tech, "over": max(over, 0.0), "budget": tech_budget},
        ))

    # 3. Subscription drift
    subs = totals.get("SUBSCRIPTIONS", 0.0)
    if subs > budget("SUBSCRIPTIONS") or subs > 2500.0:
        out.append(Insight(
            "insight_subscription_drift", "Recurring Subscription Drift",
            f"Monthly recurring subscriptions currently sit at {fmt(subs)}. Overlapping media or AI coding copilot tools add up fast.",
            "RISK_NUDGE", subs,
            "Audit active tools and downgrade duplicate streaming or SaaS subscriptions to free up ~₱500-₱1,100/mo.",
            "LOW", {"spend": subs, "free_low": 500.0, "free_high": 1100.0},
        ))

    # 4. Emergency buffer + discretionary pacing
    total_expense = sum(v for v in totals.values())
    net_savings = max(monthly_income - total_expense, 0.0)
    essentials = totals.get("HOUSING_RENT", 0.0) + totals.get("GROCERIES", 0.0) + totals.get("TRANSPORT", 0.0)
    baseline = essentials if essentials > 0 else 19600.0
    reserve = monthly_income * 0.9 + net_savings
    months = reserve / baseline
    out.append(Insight(
        "insight_emergency_fund_status", f"Emergency Fund on Track: {months:.1f} Months",
        f"Your essential overhead (rent, groceries, commute) is ~{fmt(baseline)}/mo. You currently retain {fmt(net_savings)} in monthly free cashflow.",
        "GROWTH_FOCUS", net_savings,
        "Maintain 3-6 months of liquid reserves in a 4.5%+ APY High-Yield Savings Account before scaling aggressive investments.",
        "LOW", {"essentials": baseline, "free": net_savings, "months": months},
    ))
    disc = totals.get("CAFES_DINING", 0.0) + totals.get("TECH_GADGETS", 0.0) + totals.get("LIFESTYLE", 0.0)
    rate = net_savings / monthly_income * 100.0 if monthly_income > 0 else 0.0
    out.append(Insight(
        "insight_discretionary_growth", "Discretionary Spending Pacing",
        f"Discretionary spend is {fmt(disc)}. You are successfully routing {_pct(rate)}% of your earnings to wealth accumulation.",
        "GROWTH_FOCUS", disc,
        f"Automate an auto-transfer of {fmt(net_savings * 0.7)} into your savings on every direct deposit date.",
        "LOW", {"spend": disc, "rate": rate, "transfer": net_savings * 0.7},
    ))

    # 5. Contextual tips
    out.append(Insight(
        "insight_tip_13th_month_pay", "Plan Your 13th-Month Pay Before It Lands",
        "If you are a regular employee, your 13th-month pay arrives as a one-time lump sum, and lump sums are the easiest money to spend without noticing. Deciding its job in advance turns it into a head start on your emergency fund.",
        "TIP", monthly_income * 0.5,
        "Set aside at least half of it for your emergency fund or a Pag-IBIG MP2 savings account, and make sure your SSS, PhilHealth and Pag-IBIG contributions are posting under your employer.",
        "LOW", {"stake": monthly_income * 0.5},
    ))
    out.append(Insight(
        "insight_tip_employer_stipend", "Leverage WFH & Learning Stipends",
        "Before paying out-of-pocket for keyboards, monitors, or tech books, check your company's fringe benefits for equipment and education budgets.",
        "TIP", 3500.0,
        "Submit monitor and desk setup receipts through your corporate expense management system.",
        "LOW", {"stake": 3500.0},
    ))
    return out


# ---- forward simulator ----------------------------------------------------------------------------

def compound_savings(monthly_deposit: float, months: int, annual_yield: float = 0.045) -> float:
    if monthly_deposit <= 0.0 or months <= 0:
        return 0.0
    r = annual_yield / 12.0
    bal = 0.0
    for _ in range(months):
        bal = (bal + monthly_deposit) * (1.0 + r)
    return bal


def _round2(x: float) -> float:
    return math.floor(x * 100.0 + 0.5) / 100.0  # Kotlin roundToInt() rounds half up


@dataclass
class Simulation:
    cuts: dict[str, float]  # category -> fraction
    monthly: float
    p3: float
    p6: float
    p12: float


def simulate(baseline: dict[str, float], cuts: dict[str, float], annual_yield: float = 0.045) -> Simulation:
    total = 0.0
    for cat, frac in cuts.items():
        total += baseline.get(cat, 0.0) * min(max(frac, 0.0), 1.0)
    m = _round2(total)
    return Simulation(
        cuts, m,
        _round2(compound_savings(m, 3, annual_yield)),
        _round2(compound_savings(m, 6, annual_yield)),
        _round2(compound_savings(m, 12, annual_yield)),
    )


def months_to_goal(target: float, saved: float, freed: float, base_savings: float = 4200.0) -> int:
    remaining = max(target - saved, 0.0)
    contribution = base_savings + freed
    if remaining <= 0.0:
        return 0
    return math.ceil(remaining / contribution)
