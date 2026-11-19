"""
Behavioural test cases in the spirit of CheckList: small, readable, template-style probes of how the
whole decision (models + review triggers) behaves on input the training data barely covers.

Each category states the property we WANT. evaluation/evaluate_behaviour.py measures how often it
holds and writes results/behaviour_results.json; tests/test_behaviour.py asserts floors for the
categories the system is designed to handle, and reports sarcasm as a known limitation.
"""
import random
import re

NEGATIVE = [
    "The zipper broke after one wear and customer service never replied to my emails.",
    "Terrible quality, the fabric ripped the first time I washed it.",
    "Very disappointed, the dress looks cheap and nothing like the photos.",
    "The seam tore open on day two and the color bled all over my other clothes.",
    "Awful fit, way too small and the material is scratchy. I want a refund.",
    "Poor stitching, the hem came undone immediately and I am returning it.",
    "This sweater pilled after one wash and looks ancient. Complete waste of money.",
    "Worst purchase ever, the buttons fell off and nobody answers my messages.",
    "The jacket arrived with a huge stain and a broken zipper, unacceptable.",
    "I am furious, third order in a row that arrived damaged and late.",
    "Cheap material, uncomfortable, and the sizing chart is totally wrong.",
    "Fell apart in the wash, do not buy this, I will never order again.",
]
POSITIVE = [
    "Love this sweater, the color is gorgeous and it fits perfectly.",
    "Beautiful dress, great quality fabric, I got so many compliments.",
    "Super comfortable and true to size, I will definitely buy another.",
    "Perfect fit, soft material and it arrived quickly. Very happy.",
    "Great jacket, warm and stylish, exactly as described.",
    "Fantastic quality for the price, I wear these leggings every day.",
    "Lovely blouse, the fabric drapes beautifully and it is so flattering.",
    "Absolutely love it, perfect for the summer and washes really well.",
    "Excellent purchase, the stitching is solid and the colors are vibrant.",
    "Wonderful skirt, comfortable all day and it looks expensive.",
]
MIXED = [
    "Gorgeous colour and the fabric feels lovely, but the zipper broke after two wears and support never answered.",
    "Love the style but it runs two sizes small, so sadly I am returning it.",
    "The dress is beautiful, however the hem was uneven and one seam is already coming apart.",
    "Great fabric, terrible fit, and the return process was a nightmare.",
    "I wanted to love this top but it shrank badly and the print faded after one wash.",
    "Cute and comfortable, although the stitching on the sleeve is already loose.",
]
NEGATED_POSITIVE = ["This is not bad at all, actually quite good.", "I can't complain, it is not cheap-looking at all.",
                    "Not disappointed in the slightest, it fits wonderfully."]
NEGATED_NEGATIVE = ["Not good. Not worth the money at all.", "Not happy, the quality is not what was promised.",
                    "This is not comfortable and not well made."]
NON_ENGLISH_NEGATIVE = [
    "Muy mala calidad, se rompió el primer día y nadie responde mis correos.",
    "Très mauvaise qualité, la fermeture éclair s'est cassée dès le premier jour.",
    "Schlechte Qualität, der Reißverschluss ist sofort kaputt gegangen und niemand antwortet.",
    "Qualità pessima, la cucitura si è strappata al primo lavaggio e non rispondono.",
    "Qualidade horrível, o tecido rasgou no primeiro dia e ninguém responde.",
]
# Sarcasm is a KNOWN LIMITATION: it is measured and reported, not asserted.
SARCASTIC_NEGATIVE = [
    "Oh great, another top that fell apart after one wash. Just what I wanted.",
    "Wow, I love paying full price for fabric that feels like sandpaper.",
    "Fantastic, the zipper lasted a whole two minutes. Really impressive engineering.",
    "Sure, because everybody dreams of a dress that shrinks three sizes in the wash.",
    "Thanks so much for the lovely hole in the sleeve, it really completes the look.",
]


def add_typos(text, seed=0, rate=0.35):
    """Deterministic misspellings: drop, swap or double a letter in about `rate` of the longer words."""
    rng = random.Random(seed)
    def mangle(m):
        w = m.group(0)
        if len(w) < 4 or rng.random() > rate:
            return w
        i = rng.randrange(1, len(w) - 1)
        kind = rng.choice(("drop", "swap", "double"))
        if kind == "drop":
            return w[:i] + w[i + 1:]
        if kind == "swap":
            return w[:i] + w[i + 1] + w[i] + w[i + 2:] if i + 2 <= len(w) else w
        return w[:i] + w[i] + w[i:]
    return re.sub(r"[A-Za-z]+", mangle, text)


def heavy_typos(text, seed=0):
    return add_typos(text, seed=seed, rate=0.8)
