from finsight_llm import guard
from finsight_llm.money import fmt, fmt_whole

FACTS = "Freed cashflow per month: ₱1,928.41\nSavings after 12 months: ₱23,712.81\nsuggest ~₱500-₱1,100/mo"


def test_money_format():
    assert fmt(1234.5) == "₱1,234.50"
    assert fmt(-1234.5) == "-₱1,234.50"
    assert fmt_whole(1234.5) == "₱1,234" or fmt_whole(1234.5) == "₱1,235"  # tie handling differs by runtime
    assert fmt_whole(1234.6) == "₱1,235"


def test_grounded_exact_and_rounded():
    assert guard.is_grounded("You free ₱1,928.41 a month", FACTS)
    assert guard.is_grounded("about ₱1,928 a month", FACTS)  # within 0.51
    assert guard.is_grounded("save ₱500 to ₱1,100", FACTS)


def test_ungrounded_detected():
    assert not guard.is_grounded("You free ₱2,000 a month", FACTS)
    assert guard.ungrounded_figures("₱1,928.41 and ₱99,999", FACTS) == [99999.0]


def test_dollar_and_php_are_checked_too():
    assert not guard.is_grounded("save $750 monthly", FACTS)
    assert not guard.is_grounded("save PHP 750 monthly", FACTS)
    assert guard.is_grounded("save PHP 1,928.41 monthly", FACTS)


def test_text_without_money_is_grounded():
    assert guard.is_grounded("Keep going, 4.5% yield, 12 months.", FACTS)


def test_clean_strips_markdown_and_think():
    assert guard.clean("<think>hmm</think>**Hi** there") == "Hi there"
    assert guard.clean("x</think> answer") == "answer"
    assert guard.clean('"quoted"') == "quoted"


def test_clean_truncates_at_sentence():
    long = ("This is a sentence that is long enough to count. " * 20).strip()
    out = guard.clean(long)
    assert len(out) <= guard.MAX_CHARS + 1 and out.endswith(".")


def test_parse_two_lines():
    assert guard.parse_two_lines("INSIGHT: a b\nACTION: c d") == ("a b", "c d")
    assert guard.parse_two_lines("insight: only") == ("only", None)
    assert guard.parse_two_lines("free text\nmore") == ("free text more", None)
    assert guard.parse_two_lines("") == (None, None)
