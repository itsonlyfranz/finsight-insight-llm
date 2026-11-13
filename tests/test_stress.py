import json
import random

from finsight_llm import guard, stress


def test_stress_rows_keep_reference_grounded_and_add_near_duplicate_decoys():
    rows = stress.generate()[:40]
    assert rows
    for r in rows:
        assert guard.is_grounded(r["reference"], r["facts"])  # decoys are in the facts, references still pass
        assert r["decoys"] and stress.HEADER in r["facts"]
        real = guard.money_figures(r["facts"].split(stress.HEADER)[0])
        assert all(all(abs(d - a) >= 0.51 for a in real) for d in r["decoys"])  # never collide with a real figure
        # decoys really are close to some real figure (within 10% or a digit transposition)
        assert all(any(abs(d - a) <= max(60, 0.12 * a) or sorted(str(int(d))) == sorted(str(int(a))) for a in real if a)
                   for d in r["decoys"])


def test_stress_is_deterministic_and_disjoint_from_other_splits():
    a, b = stress.generate(), stress.generate()
    assert [r["facts"] for r in a] == [r["facts"] for r in b]
    from finsight_llm import dataset
    others = {r["facts"] for rows in dataset.generate().values() for r in rows}
    assert not any(r["facts"].split(stress.HEADER)[0] in others for r in a)


def test_decoy_rate_detects_wrong_but_grounded_figure():
    assert stress.decoy_rate("You save ₱1,234.00 a month", [1234.0, 55.0])
    assert not stress.decoy_rate("You save ₱1,000.00 a month", [1234.0])


def test_score_decoys(tmp_path):
    sp, rp = tmp_path / "s.jsonl", tmp_path / "r.jsonl"
    sp.write_text(json.dumps({"id": "a", "decoys": [10.0]}) + "\n" + json.dumps({"id": "b", "decoys": [20.0]}) + "\n")
    rp.write_text(json.dumps({"_meta": {}}) + "\n" + json.dumps({"id": "a", "output": "₱10"}) + "\n"
                  + json.dumps({"id": "b", "output": "₱99"}) + "\n")
    assert stress.score_decoys(rp, sp)["decoy_rate_pct"] == 50.0
