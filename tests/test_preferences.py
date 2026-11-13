import json
import random
from pathlib import Path

import pytest

from finsight_llm import dataset, guard, metrics, preferences, prompts

ROWS = None


def rows():
    """A small, deterministic slice of the real training facts, parsed back from chat format."""
    global ROWS
    if ROWS is None:
        data = dataset.generate(sizes={"train": 60, "valid": 1, "test": 1})["train"]
        ROWS = []
        for i, r in enumerate(data):
            chat = dataset.to_chat(r)
            parsed = preferences.parse_chat_row(chat)
            parsed["id"] = f"train-{i:05d}"
            ROWS.append(parsed)
    return ROWS


def test_parse_chat_row_roundtrips_facts_and_task():
    src = dataset.generate(sizes={"train": 15, "valid": 1, "test": 1})["train"]
    for r in src:
        p = preferences.parse_chat_row(dataset.to_chat(r))
        assert (p["task"], p["facts"], p["reference"]) == (r["task"], r["facts"], r["reference"])


def test_chosen_is_a_clean_grounded_reference_and_rejected_is_a_real_failure():
    built = preferences.build(rows())
    pairs = built["train"] + built["valid"]
    assert pairs
    by_id = {r["id"]: r for r in rows()}
    for p in pairs:
        row = by_id[p["id"]]
        assert p["chosen"] == row["reference"] != p["rejected"]
        ok = metrics.score_output(row["task"], p["chosen"], row["facts"], row["reference"])
        assert ok["clean_pass"] and ok["fact_coverage"] == 1.0
        assert preferences.is_failure(row, p["rejected"]), p["rejected_type"]
        assert p["prompt"] == prompts.messages(row["task"], row["facts"])


def test_each_corruption_catches_what_it_claims():
    rng = random.Random(1)
    checked = set()
    for row in rows():
        for name, fn in preferences.CORRUPTIONS.items():
            text = fn(row["task"], row["reference"], row["facts"], rng)
            if not text:
                continue
            s = metrics.score_output(row["task"], text, row["facts"], row["reference"])
            if name in ("changed_figure", "invented_figure"):
                assert not s["grounded"], (name, text)
            elif name == "broken_format":
                assert not s["format_valid"], (name, text)
            elif name in ("vague_text", "off_topic_text"):
                assert s["format_valid"], (name, text)
                if guard.money_figures(row["reference"]):
                    assert s["fact_coverage"] < 1.0, (name, text)
            elif name == "misattributed_figure":
                assert s["grounded"], (name, text)  # wrong but in the facts; build() keeps it only if coverage drops
            checked.add(name)
    assert checked == set(preferences.CORRUPTIONS)


def test_changed_figure_is_a_near_duplicate_not_in_facts():
    rng = random.Random(3)
    row = next(r for r in rows() if r["task"] == "simulation")
    text = preferences.changed_figure(row["task"], row["reference"], row["facts"], rng)
    bad = guard.ungrounded_figures(text, row["facts"])
    assert len(bad) == 1


def test_build_is_deterministic_and_splits_by_prompt():
    a = preferences.build(rows())
    b = preferences.build(rows())
    assert a == b
    tr, va = {p["id"] for p in a["train"]}, {p["id"] for p in a["valid"]}
    assert tr and va and not tr & va
    assert len(va) / (len(tr) + len(va)) == pytest.approx(preferences.VALID_FRACTION, abs=0.05)


def test_up_to_two_distinct_synthetic_types_per_prompt():
    built = preferences.build(rows())
    per: dict[str, list[str]] = {}
    for p in built["train"] + built["valid"]:
        per.setdefault(p["id"], []).append(p["rejected_type"])
    assert all(len(v) <= 2 and len(set(v)) == len(v) for v in per.values())
    assert sum(len(v) == 2 for v in per.values()) > len(per) // 2


def test_sft_samples_only_count_when_they_lose():
    row = rows()[0]
    good = row["reference"]
    # identical to the reference: never a negative
    assert preferences.from_sft_samples(row, [good]) is None
    bad = preferences.changed_figure(row["task"], good, row["facts"], random.Random(0))
    vague = preferences.vague_text(row["task"], good, row["facts"], random.Random(0))
    hit = preferences.from_sft_samples(row, [good, vague, bad])
    assert hit and hit[0] == bad  # the guard failure is worse than the vague text
    built = preferences.build([row], {row["id"]: [good, bad]})
    srcs = [p["source"] for p in built["train"] + built["valid"]]
    assert "sft_sample" in srcs


def test_test_split_is_never_read():
    src = Path(preferences.__file__).read_text()
    assert "test.jsonl" not in src.replace('"test.jsonl" not in', "")


def test_pairs_are_json_serialisable(tmp_path):
    out = tmp_path / "p"
    train = tmp_path / "train.jsonl"
    data = dataset.generate(sizes={"train": 20, "valid": 1, "test": 1})["train"]
    train.write_text("\n".join(json.dumps(dataset.to_chat(r), ensure_ascii=False) for r in data) + "\n", encoding="utf-8")
    preferences.main(["--train", str(train), "--out", str(out)])
    stats = json.loads((out / "stats.json").read_text())
    assert stats["source_rows"] == len(data)
    assert (out / "train.jsonl").exists() and (out / "valid.jsonl").exists()
