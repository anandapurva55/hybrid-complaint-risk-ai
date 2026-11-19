"""
Behavioural tests on the real models (CheckList-style). They assert floors for what the system is
designed to handle, and record sarcasm as a KNOWN limitation (xfail) rather than hiding it.
Skipped when the model artifacts or torch are not available.
"""
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from evaluation import behaviour_cases as bc  # noqa: E402
from evaluation import evaluate_behaviour as eb  # noqa: E402
from hybrid_risk.paths import MODELS  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODELS / "transformer_assets" / "weights_int8.npz"), reason="model artifacts not present")

FLOOR = 0.9


@pytest.fixture(scope="module")
def results():
    typos = lambda texts, seeds, fn: [fn(t, s) for t in texts for s in seeds]
    return {
        "neg": eb.summarise(bc.NEGATIVE, "caught"),
        "neg_mild_typos": eb.summarise(typos(bc.NEGATIVE, (1, 2, 3), bc.add_typos), "caught"),
        "neg_heavy_typos": eb.summarise(typos(bc.NEGATIVE, (1, 2, 3), bc.heavy_typos), "caught"),
        "neg_caps": eb.summarise([t.upper() for t in bc.NEGATIVE], "caught"),
        "neg_foreign": eb.summarise(bc.NON_ENGLISH_NEGATIVE, "caught"),
        "mixed": eb.summarise(bc.MIXED, "caught"),
        "sarcasm": eb.summarise(bc.SARCASTIC_NEGATIVE, "caught"),
        "pos": eb.summarise(bc.POSITIVE, "auto_reply"),
        "pos_caps": eb.summarise([t.upper() for t in bc.POSITIVE], "auto_reply"),
    }


@pytest.mark.parametrize("key", ["neg", "neg_mild_typos", "neg_heavy_typos", "neg_caps", "neg_foreign", "mixed"])
def test_negative_feedback_is_never_silently_auto_replied(results, key):
    assert results[key]["caught_with_review_rules"] >= FLOOR, results[key]["examples_that_missed"]


@pytest.mark.parametrize("key", ["pos", "pos_caps"])
def test_clearly_positive_feedback_is_still_answered_automatically(results, key):
    assert results[key]["auto_reply_rate"] >= FLOOR


def test_review_rules_add_value_for_non_english_text(results):
    assert results["neg_foreign"]["caught_with_review_rules"] > results["neg_foreign"]["caught_by_model_tier_alone"]


def test_negation_changes_the_direction_of_the_score():
    p = lambda ts: [eb.run(t)["probabilities"]["high"] for t in ts]
    neg, pos = p(bc.NEGATED_NEGATIVE), p(bc.NEGATED_POSITIVE)
    assert min(neg) > max(pos)


def test_shouting_does_not_change_the_tier():
    same = [eb.run(t)["risk_level"] == eb.run(t.upper())["risk_level"] for t in bc.NEGATIVE + bc.POSITIVE]
    assert sum(same) / len(same) >= FLOOR


@pytest.mark.xfail(reason="KNOWN LIMITATION: sarcastic complaints read as positive to both text models", strict=False)
def test_sarcastic_complaints_are_caught(results):
    assert results["sarcasm"]["caught_with_review_rules"] >= FLOOR
