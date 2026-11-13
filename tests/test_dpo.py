import math

from finsight_llm import dpo


def test_loss_is_ln2_when_policy_equals_reference():
    assert dpo.dpo_loss_value(-10.0, -10.0, 0.1) == math.log(2)
    assert dpo.dpo_loss_value(-3.0 + 7.0, -3.0 + 7.0, 0.3) == math.log(2)


def test_loss_falls_as_chosen_margin_grows_and_rises_when_it_shrinks():
    ref = 5.0
    better = dpo.dpo_loss_value(ref + 20, ref, 0.1)
    worse = dpo.dpo_loss_value(ref - 20, ref, 0.1)
    assert better < math.log(2) < worse


def test_loss_matches_closed_form():
    z = 0.1 * (12.0 - 4.0)
    assert math.isclose(dpo.dpo_loss_value(12.0, 4.0, 0.1), -math.log(1 / (1 + math.exp(-z))))


def test_extreme_margin_does_not_overflow():
    assert math.isfinite(dpo.dpo_loss_value(-1e6, 1e6, 0.5))
