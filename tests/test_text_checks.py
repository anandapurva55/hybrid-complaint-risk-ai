import numpy as np

from hybrid_risk import text_checks as tc
from hybrid_risk.decision import DEFAULT_CONFIG, assess


def fake_tokenize(word):
    """Pretend long unknown words split into 3-letter pieces; short words are one piece."""
    return [word] if len(word) <= 5 else [word[i:i + 3] for i in range(0, len(word), 3)]


def test_very_short_text_has_no_score():
    assert tc.pieces_per_word("ok fine", fake_tokenize) is None
    assert tc.english_share("ok fine") is None
    assert not tc.is_noisy("ok", fake_tokenize, 1.5)
    assert not tc.is_non_english("ok", 0.1)


def test_clean_text_is_not_noisy():
    assert not tc.is_noisy("the cat sat on the mat and it was nice", fake_tokenize, 1.5)


def test_fragmenting_text_is_noisy():
    assert tc.is_noisy("terriblequality zippermalfunction worstpurchase everbought", fake_tokenize, 1.5)


def test_english_share_separates_english_from_spanish():
    assert tc.english_share("The zipper broke and it was not what I expected") > 0.3
    assert tc.is_non_english("Muy mala calidad, se rompió el primer día y nadie responde", 0.10)
    assert not tc.is_non_english("The zipper broke and it was not what I expected", 0.10)


def test_noisy_or_non_english_text_is_never_auto_replied():
    confident_low = np.array([[0.97, 0.02, 0.01]])
    for flag in ("noisy_text", "non_english"):
        res = assess(confident_low, confident_low, dict(DEFAULT_CONFIG), extra_flags={flag: np.array([True])})
        assert res["level"][0] == "low"
        assert res["action"][0] == "manual_review" and res["review"][0]


def test_default_config_carries_the_thresholds():
    assert DEFAULT_CONFIG["noisy_pieces_per_word"] > 1.44   # above anything seen in real validation reviews
    assert 0 < DEFAULT_CONFIG["min_english_share"] < 0.2
