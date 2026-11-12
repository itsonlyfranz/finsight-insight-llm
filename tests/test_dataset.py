import json
import random

from finsight_llm import dataset, engine, guard, prompts

SMALL = {"train": 40, "valid": 8, "test": 12}


def test_generation_is_deterministic():
    a = dataset.generate(seed=1, sizes=SMALL)
    b = dataset.generate(seed=1, sizes=SMALL)
    assert a == b
    assert a != dataset.generate(seed=2, sizes=SMALL)


def test_every_reference_is_grounded_and_well_formed():
    splits = dataset.generate(seed=3, sizes={"train": 150, "valid": 10, "test": 10})
    for rows in splits.values():
        for r in rows:
            assert guard.is_grounded(r["reference"], r["facts"]), r["id"]
            if r["task"] == "insight":
                insight, action = guard.parse_two_lines(r["reference"])
                assert insight and action
                assert r["reference"].count("\n") == 1
            else:
                assert "INSIGHT:" not in r["reference"]


def test_no_input_shared_across_splits():
    s = dataset.generate(seed=4, sizes=SMALL)
    sets = {k: {r["facts"] for r in v} for k, v in s.items()}
    assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["valid"]) and not (sets["valid"] & sets["test"])


def test_chat_format_matches_app_prompt():
    row = dataset.generate(seed=5, sizes={"train": 5, "valid": 1, "test": 1})["train"][0]
    msgs = dataset.to_chat(row)["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == prompts.SYSTEM_PROMPT
    assert msgs[1]["content"].endswith(prompts.INSTRUCTIONS[row["task"]])
    assert "₱" in prompts.SYSTEM_PROMPT


def test_write_outputs_jsonl(tmp_path):
    dataset.write(tmp_path, dataset.generate(seed=6, sizes=SMALL))
    test_rows = [json.loads(line) for line in (tmp_path / "test.jsonl").read_text().splitlines()]
    assert {"messages", "id", "task", "facts", "reference"} <= set(test_rows[0])
    train_rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert set(train_rows[0]) == {"messages"}  # no metadata leaks into training files


def test_engine_port_matches_kotlin_arithmetic():
    # hand-computed from AiInsightsEngine.kt for: income 45000, cafes 12 visits totalling 6000
    ins = {i.id: i for i in engine.generate_insights(
        {"CAFES_DINING": 6000.0, "HOUSING_RENT": 14000.0, "GROCERIES": 4200.0, "TRANSPORT": 2100.0}, 12, 45000.0)}
    cafe = ins["insight_cafe_lifestyle_inflation"]
    assert cafe.severity == "MEDIUM"  # count > 10, pct 13.3 (<15), spend < 1.5x budget
    assert "~13.3% of your monthly income" in cafe.description
    assert "₱1,500.00/mo" in cafe.action_suggestion  # 25% of 6000
    emerg = ins["insight_emergency_fund_status"]
    # essentials 20300; net = 45000-26300 = 18700; reserve = 40500+18700 = 59200; months = 2.9
    assert emerg.title == "Emergency Fund on Track: 2.9 Months"


def test_empty_profile_yields_no_insights():
    assert engine.generate_insights({}, 0, 45000.0) == []


def test_simulator_matches_kotlin_compounding():
    sim = engine.simulate({"CAFES_DINING": 5000.0}, {"CAFES_DINING": 0.2})
    assert sim.monthly == 1000.0
    # (balance + deposit) * (1 + 0.045/12), 12 times
    bal = 0.0
    for _ in range(12):
        bal = (bal + 1000.0) * (1 + 0.045 / 12)
    assert sim.p12 == round(bal, 2)
    assert engine.simulate({"CAFES_DINING": 5000.0}, {}).monthly == 0.0


def test_months_to_goal():
    assert engine.months_to_goal(100000, 0, 0) == 24  # ceil(100000/4200)
    assert engine.months_to_goal(100, 100, 0) == 0


def test_sampler_covers_every_insight_kind():
    rng = random.Random(0)
    kinds = set()
    for _ in range(600):
        p = dataset.sample_profile(rng)
        kinds |= {i.id for i in engine.generate_insights(p["totals"], p["cafe_count"], p["income"])}
    assert len(kinds) == 7


def test_every_reference_satisfies_the_output_contract():
    from finsight_llm import metrics
    splits = dataset.generate(seed=7, sizes={"train": 300, "valid": 10, "test": 10})
    for rows in splits.values():
        for r in rows:
            assert metrics.format_valid(r["task"], r["reference"]), (r["task"], r["reference"])
