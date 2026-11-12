"""Reference-text templates.

The target texts of the dataset are composed from hand-written phrasings (several per insight kind) filled
with numbers taken from the facts. Every target is checked with the number guard at generation time. The
variation is in wording and sentence structure; the numbers are always copied from the facts.
"""
from __future__ import annotations

import random
from typing import Callable

from .money import fmt, fmt_whole

Tpl = Callable[[dict[str, float]], str]


def _pct(x: float) -> str:
    return f"{x:.1f}"


# ---- insight texts, keyed by AiInsightsEngine insight id ------------------------------------------

INSIGHT_TEXT: dict[str, list[Tpl]] = {
    "insight_cafe_lifestyle_inflation": [
        lambda v: f"{int(v['count'])} cafe and dining outings added up to {fmt(v['spend'])} this month, about {_pct(v['pct'])}% of your income. Those small swipes are easy to overlook, but together they matter.",
        lambda v: f"Eating out is taking a real bite: {int(v['count'])} visits cost you {fmt(v['spend'])}, roughly {_pct(v['pct'])}% of what you earn each month.",
        lambda v: f"Your cafe and dining tab reached {fmt(v['spend'])} across {int(v['count'])} outings, close to {_pct(v['pct'])}% of your monthly income. It's the kind of spending that hides in plain sight.",
        lambda v: f"Heads up: {int(v['count'])} dining outings put you at {fmt(v['spend'])}, or {_pct(v['pct'])}% of your income. Frequent small purchases are the easiest ones to miss.",
        lambda v: f"You've spent {fmt(v['spend'])} on {int(v['count'])} cafe and dining trips, which is about {_pct(v['pct'])}% of your monthly income.",
    ],
    "insight_tech_gadget_spike": [
        lambda v: (f"Tech purchases reached {fmt(v['spend'])} this cycle, {fmt(v['over'])} above your {fmt(v['budget'])} target. A few big buys can quietly undo a good month."
                   if v["over"] > 0 else f"Tech purchases reached {fmt(v['spend'])} this cycle. It's worth checking that each buy was planned rather than impulsive."),
        lambda v: (f"Your gadget spending climbed to {fmt(v['spend'])}, which is {fmt(v['over'])} over the {fmt(v['budget'])} you aimed for."
                   if v["over"] > 0 else f"Your gadget spending came to {fmt(v['spend'])} this cycle, a noticeable share of your income."),
        lambda v: (f"This cycle you put {fmt(v['spend'])} into tech and gadgets, {fmt(v['over'])} past your {fmt(v['budget'])} target. Upgrades feel small one at a time but add up."
                   if v["over"] > 0 else f"You put {fmt(v['spend'])} into tech and gadgets this cycle. Upgrades feel small one at a time but add up."),
        lambda v: (f"Tech and gadgets cost {fmt(v['spend'])} so far, overshooting your {fmt(v['budget'])} target by {fmt(v['over'])}."
                   if v["over"] > 0 else f"Tech and gadgets cost {fmt(v['spend'])} so far this cycle, which is a big slice of your income."),
    ],
    "insight_subscription_drift": [
        lambda v: f"Your recurring subscriptions now total {fmt(v['spend'])} a month. Overlapping streaming or AI tools are the usual culprits.",
        lambda v: f"Subscriptions are costing you {fmt(v['spend'])} every month, and it's easy to pay twice for the same thing without noticing.",
        lambda v: f"Monthly subscriptions have crept up to {fmt(v['spend'])}. Media and coding-assistant plans tend to overlap more than people expect.",
        lambda v: f"You're paying {fmt(v['spend'])} a month for recurring subscriptions, and a few of them may be doing the same job.",
    ],
    "insight_emergency_fund_status": [
        lambda v: f"Your essentials (rent, groceries and commute) run about {fmt(v['essentials'])} a month, and you keep {fmt(v['free'])} in free cashflow. At this pace your emergency fund is on track at about {v['months']:.1f} months.",
        lambda v: f"Covering rent, groceries and commute takes roughly {fmt(v['essentials'])} each month, leaving {fmt(v['free'])} free. That puts your safety net around {v['months']:.1f} months.",
        lambda v: f"Good news: after about {fmt(v['essentials'])} of monthly essentials you still have {fmt(v['free'])} of free cashflow, which keeps your emergency buffer near {v['months']:.1f} months.",
        lambda v: f"Your basics cost around {fmt(v['essentials'])} a month and {fmt(v['free'])} is left over. That supports an emergency cushion of roughly {v['months']:.1f} months.",
    ],
    "insight_discretionary_growth": [
        lambda v: f"Discretionary spending is {fmt(v['spend'])} this month, and you're still saving {_pct(v['rate'])}% of your earnings. That's a healthy balance.",
        lambda v: f"You spent {fmt(v['spend'])} on wants like dining, tech and leisure while routing {_pct(v['rate'])}% of your income into savings.",
        lambda v: f"Nice pacing: {fmt(v['spend'])} of discretionary spend and {_pct(v['rate'])}% of your earnings going toward building wealth.",
        lambda v: f"Wants came to {fmt(v['spend'])} this month, and {_pct(v['rate'])}% of your pay is still heading to savings.",
    ],
    "insight_tip_13th_month_pay": [
        lambda v: "Your 13th-month pay will land as one lump sum, and lump sums are the easiest money to spend without noticing. Giving it a job before it arrives gives your emergency fund a head start.",
        lambda v: f"When your 13th-month pay arrives, about {fmt(v['stake'])} could be at stake if it goes unplanned. Deciding its purpose now makes it far more likely to build your safety net.",
        lambda v: "Lump sums like a 13th-month bonus disappear fast when they have no plan. Decide now where it goes and it becomes a real boost to your emergency fund.",
        lambda v: f"A 13th-month bonus is a chance to move the needle, with roughly {fmt(v['stake'])} on the table. Plan it ahead of time so it doesn't vanish into everyday spending.",
    ],
    "insight_tip_employer_stipend": [
        lambda v: "Before paying out of pocket for a keyboard, monitor or tech book, see what your employer already covers. Equipment and learning budgets often go unused.",
        lambda v: f"Your company may cover equipment and learning costs, which could keep up to {fmt(v['stake'])} in your pocket. It's worth checking the fringe benefits before you buy.",
        lambda v: "Many employers offer WFH and learning stipends that people forget to claim. Check yours before spending your own money on desk gear or books.",
        lambda v: f"Up to {fmt(v['stake'])} of setup and learning costs might be reimbursable through your employer. Look at your benefits before you pay yourself.",
    ],
}

INSIGHT_ACTION: dict[str, list[Tpl]] = {
    "insight_cafe_lifestyle_inflation": [
        lambda v: f"Brew at home on two weekdays and you could free up about {fmt(v['savings'])} a month while keeping your weekend brunches.",
        lambda v: f"Try a two-day homebrew routine on weekdays to reclaim around {fmt(v['savings'])} a month, and keep the social brunches.",
        lambda v: f"Swap two weekday cafe stops for coffee at home and you could win back roughly {fmt(v['savings'])} a month.",
        lambda v: f"Start with two homebrew weekdays; it could return about {fmt(v['savings'])} a month without touching your weekend plans.",
    ],
    "insight_tech_gadget_spike": [
        lambda v: "Put new peripherals and keyboards through a 14-day cooling-off period before you buy.",
        lambda v: "Wait 14 days before any new gadget purchase; if you still want it afterwards, it's a real need.",
        lambda v: "Adopt a 14-day cooling-off rule for peripherals and keyboards so impulse buys fade before checkout.",
        lambda v: "Add a 14-day pause before buying new tech to keep impulse purchases in check.",
    ],
    "insight_subscription_drift": [
        lambda v: f"Audit your active tools and drop duplicate streaming or SaaS plans to free up roughly {fmt(v['free_low'])} to {fmt(v['free_high'])} a month.",
        lambda v: f"List every subscription, cancel the overlapping ones, and you could free up about {fmt(v['free_low'])} to {fmt(v['free_high'])} a month.",
        lambda v: f"Review your subscriptions this week and downgrade the duplicates to recover around {fmt(v['free_low'])} to {fmt(v['free_high'])} monthly.",
        lambda v: f"Cancel or downgrade overlapping media and SaaS plans to reclaim roughly {fmt(v['free_low'])} to {fmt(v['free_high'])} a month.",
    ],
    "insight_emergency_fund_status": [
        lambda v: "Keep building 3 to 6 months of reserves in a high-yield savings account (4.5%+ APY) before taking on riskier investments.",
        lambda v: "Aim for 3 to 6 months of liquid reserves in a 4.5%+ APY high-yield savings account before investing aggressively.",
        lambda v: "Hold 3 to 6 months of expenses in a high-yield savings account paying 4.5%+ APY before you scale up investing.",
        lambda v: "Top up a 4.5%+ APY high-yield savings account until it covers 3 to 6 months, then think about bolder investments.",
    ],
    "insight_discretionary_growth": [
        lambda v: f"Set an automatic transfer of {fmt(v['transfer'])} to your savings on every payday.",
        lambda v: f"Automate {fmt(v['transfer'])} into savings on each direct deposit date so it happens before you can spend it.",
        lambda v: f"Schedule an auto-transfer of {fmt(v['transfer'])} to your savings every time your salary lands.",
        lambda v: f"Let {fmt(v['transfer'])} move into savings automatically on every deposit date.",
    ],
    "insight_tip_13th_month_pay": [
        lambda v: "Put at least half of it into your emergency fund or a Pag-IBIG MP2 account, and confirm your SSS, PhilHealth and Pag-IBIG contributions are posting under your employer.",
        lambda v: "Set aside at least half for your emergency fund or Pag-IBIG MP2, and check that your SSS, PhilHealth and Pag-IBIG contributions are being posted by your employer.",
        lambda v: "Commit at least half to your emergency fund or a Pag-IBIG MP2 savings account, and make sure your employer is posting your SSS, PhilHealth and Pag-IBIG contributions.",
    ],
    "insight_tip_employer_stipend": [
        lambda v: "Submit your monitor and desk setup receipts through your company's expense system.",
        lambda v: "File your monitor and desk setup receipts through your corporate expense management system.",
        lambda v: "Send your desk and monitor receipts to your company's expense portal for reimbursement.",
    ],
}


# ---- simulator explanations -------------------------------------------------------------------------

SIM_TEXT: list[Callable[[dict[str, float | str]], str]] = [
    lambda v: f"Making these cuts frees up {fmt(v['monthly'])} every month. Kept up, that grows to {fmt(v['p12'])} after 12 months at the assumed 4.5% yearly yield, with {fmt(v['p6'])} already in hand at 6 months.",
    lambda v: f"If you hold to this plan you'll free {fmt(v['monthly'])} a month. That adds up to {fmt(v['p3'])} in 3 months, {fmt(v['p6'])} in 6 months and {fmt(v['p12'])} in a year.",
    lambda v: f"These cuts return {fmt(v['monthly'])} to your budget each month. After a year of saving it, with the assumed 4.5% yield, you'd have {fmt(v['p12'])}.",
    lambda v: f"Sticking to this plan means {fmt(v['monthly'])} extra per month. You'd see {fmt(v['p3'])} after 3 months and {fmt(v['p12'])} after 12, so the habit pays off steadily.",
    lambda v: f"Your planned cuts ({v['cuts']}) free {fmt(v['monthly'])} a month. Over 6 months that is {fmt(v['p6'])}, and over 12 months it reaches {fmt(v['p12'])}.",
    lambda v: f"A small change goes a long way. {fmt(v['monthly'])} freed per month becomes {fmt(v['p12'])} in 12 months, assuming a 4.5% yearly yield.",
]


# ---- goal milestones -------------------------------------------------------------------------------

GOAL_TEXT: list[Callable[[dict[str, float | str]], str]] = [
    lambda v: f"You're {v['progress']}% of the way to {v['title']}; with {fmt(v['freed'])} extra each month you could be done by {v['date']}.",
    lambda v: f"Only {fmt(v['remaining'])} left for {v['title']}, and at this pace you'll reach it in {v['months']} months, around {v['date']}.",
    lambda v: f"Keep going: {fmt(v['saved'])} saved toward {fmt(v['target'])}, and your extra {fmt(v['freed'])} a month brings {v['title']} into reach by {v['date']}.",
    lambda v: f"{v['progress']}% done on {v['title']}, and about {v['months']} more months at this rhythm will land it in {v['date']}.",
    lambda v: f"Your next milestone is within sight: {fmt(v['remaining'])} to go for {v['title']}, finishing around {v['date']}.",
]
GOAL_TEXT_DONE = [
    lambda v: f"Congratulations, {v['title']} is fully funded at {fmt(v['target'])}, so it's time to pick your next goal.",
    lambda v: f"You did it: {fmt(v['saved'])} saved covers your {fmt(v['target'])} target for {v['title']}.",
]


def pick(rng: random.Random, options: list):
    return options[rng.randrange(len(options))]


def whole(x: float) -> str:  # re-export for callers
    return fmt_whole(x)
