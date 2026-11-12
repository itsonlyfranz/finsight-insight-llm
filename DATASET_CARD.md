---
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- finance
- synthetic
- instruction-tuning
pretty_name: FinSight Insight (synthetic)
size_categories:
- 1K<n<10K
---

# Dataset card: FinSight Insight (synthetic)

> Draft for review. Nothing has been uploaded.

Fully synthetic training data for teaching a small language model to turn a personal-finance **facts block** into the short, warm, number-faithful text that the FinSight Android app shows to users. **No real people, transactions, accounts or finances are used anywhere.**

## Contents

| Split | Examples | insight | simulation | goal |
|---|---|---|---|---|
| `train` | 1,618 | 842 | 472 | 304 |
| `valid` | 114 | 53 | 37 | 24 |
| `test` (held out) | 246 | 117 | 78 | 51 |

Files are JSON Lines in the chat format used by `mlx-lm` (`{"messages": [system, user, assistant]}`). `test.jsonl` additionally carries `id`, `task`, `kind`, `facts` and `reference` so evaluation can score outputs; the train and valid files contain only `messages`.

### Tasks (mirror the three AI features of the app)

| Task | Input ("facts") | Target |
|---|---|---|
| `insight` | income, insight type, headline, findings, suggested action and amount at stake, as produced by the app's insights engine | exactly two lines: `INSIGHT:` (at most two sentences) and `ACTION:` (one sentence) |
| `simulation` | planned spending cuts, freed cashflow per month and projected savings at 3, 6 and 12 months | two to three plain sentences |
| `goal` | goal title, target, saved so far, remaining, progress, extra freed cashflow and projected completion | one encouraging sentence of at most 30 words |

The system prompt and the per-task instruction are copied verbatim from the app, so a model trained on this data sees the same input it would see in production. All amounts are Philippine pesos written with the peso sign.

## How it was generated

`python -m finsight_llm.dataset --out data` (seed `20260512`, fully deterministic).

1. **Profiles.** Each example starts from an imaginary earner: a monthly income (₱18,000 to ₱150,000, skewed toward early-career levels) and monthly totals for seven spending categories (cafes & dining with a visit count, tech & gadgets, rent, transport, subscriptions, groceries, lifestyle), all sampled from seeded distributions.
2. **Facts.** The profile is run through a Python port of the app's deterministic engines (`AiInsightsEngine`, `ForwardSimulatorEngine`) to produce the same numbers and wording the app would hand to its model. For `insight`, one of the profile's insights is chosen at random; `simulation` and `goal` examples are added for a random subset of profiles.
3. **Targets.** The reference text is composed from hand-written phrasings (4 to 6 per insight kind, 6 for simulations, 5 for goals) filled with numbers copied from the facts. It is not model-generated.
4. **Checks.** Every reference is verified with a port of the app's money-figure guard (no money figure outside the facts) while generating; generation aborts otherwise. No input block appears twice across the dataset.
5. **Splits** are made per profile before any example is drawn for the next split, so no profile contributes to more than one split.

## Intended use

Supervised fine-tuning and evaluation of small models on number-faithful rewriting of finance facts. It is a portfolio/research dataset, not a source of financial advice.

## Limitations

- **Templated targets.** Wording variety is bounded by the hand-written phrasings; a model can reach very high overlap with the references without being "creative". Reference-based scores should be read with that in mind.
- **Narrow distribution.** Seven categories, one currency (PHP), the app's fixed rules and thresholds. Static tips (13th-month pay, employer stipends) repeat the same advice with different wording.
- **Synthetic spending patterns** are plausible but not calibrated to real-world statistics.
- The facts reproduce rules from a particular app; the text is not general financial advice and should not be relied on as such.

## License

Apache-2.0 (the generator code, templates and generated data).
