"""
Deterministic input-quality checks that feed the review triggers.

The text models were trained on clean English product reviews, so they are unreliable on text that
looks nothing like that. Two cheap signals catch most such input, and both are calibrated on real
validation reviews so they almost never fire on ordinary feedback:

  * noisy_text   - the tokenizer splits words into many pieces. Misspellings fragment into word-
                   pieces; across real validation reviews the mean is 1.08 pieces per word and the
                   maximum 1.44, while typo-heavy and non-English text scores 1.6 to 2.4.
  * non_english  - almost no common English function words ("the", "and", "it", "not", ...). Real
                   reviews have a median share of about 0.42; fewer than 0.2% fall below 0.10.

Both only apply once a text has at least MIN_WORDS words (shorter text has its own trigger).
"""
import re

WORD = re.compile(r"[A-Za-z']+")
MIN_WORDS = 4
ENGLISH_FUNCTION_WORDS = frozenset(
    "the a an and is it to of i this was for not but with my in on that are be have they so as at very too me you its it's".split()
)


def words(text):
    return WORD.findall(str(text))


def pieces_per_word(text, tokenize):
    """Average number of tokenizer pieces per alphabetic word, or None for very short text."""
    w = words(text)
    if len(w) < MIN_WORDS:
        return None
    return sum(len(tokenize(x)) for x in w) / len(w)


def english_share(text):
    """Share of words that are common English function words, or None for very short text."""
    w = [x.lower() for x in words(text)]
    if len(w) < MIN_WORDS:
        return None
    return sum(x in ENGLISH_FUNCTION_WORDS for x in w) / len(w)


def is_noisy(text, tokenize, max_pieces_per_word):
    v = pieces_per_word(text, tokenize)
    return v is not None and v > max_pieces_per_word


def is_non_english(text, min_share):
    v = english_share(text)
    return v is not None and v < min_share
