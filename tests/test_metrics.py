from finsight_llm import metrics


def test_rouge_l_ignores_trailing_punctuation():
    # tokens: [save, per, month] vs [save, per, month]; "month." / "month," must equal "month"
    assert metrics.rouge_l("Save per month.", "save per month") == 1.0
    assert metrics.rouge_l("Save, per month,", "save per month") == 1.0


def test_rouge_l_keeps_money_figures_whole():
    # candidate [you, save, ₱1,928.41, a, month]; reference [save, ₱1,928.41, monthly]
    # LCS = 2 (save, ₱1,928.41); P = 2/5, R = 2/3; F1 = 2*(0.4*2/3)/(0.4+2/3) = 0.5
    assert abs(metrics.rouge_l("You save ₱1,928.41 a month.", "Save ₱1,928.41 monthly.") - 0.5) < 1e-9
    # a different figure must not match
    assert metrics.rouge_l("₱1,928.41", "₱1,928.42") == 0.0
