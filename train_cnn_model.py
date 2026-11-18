"""
CNN component: OPTIONAL DEMO, NOT PART OF THE DECISION.

hybrid_inference.py no longer uses this model. It classifies 28x28 Fashion-MNIST
clothing categories, so it cannot judge a real complaint photo, and its
category->risk table below is an illustrative assumption. Feeding it into the
score (30% weight, with a constant when no image exists) added noise and hid
the "high" class, so it was removed from the decision; see README, "Why the CNN
is not in the decision". The script is kept to document the architecture and
how a real image model would plug in once outcome-labelled photos exist.

Trains a small CNN image classifier on a subset of Fashion-MNIST and saves it
to cnn_model.keras, then documents how its output is turned into the
`cnn_score` consumed by the hybrid system.

--------------------------------------------------------------------------
Why the CNN's contribution to the hybrid score is "simulated"
--------------------------------------------------------------------------
The business scenario calls for an image-risk signal derived from photos
customers attach to complaints (for example, a photo of a damaged product).
No such labelled dataset exists publicly, and the chosen primary dataset
(the Women's E-Commerce Clothing Reviews dataset) has no images at all.

So this script trains a genuine CNN, with a real architecture, real training,
and real accuracy and F1 on held-out data, on Fashion-MNIST, to demonstrate
the CNN end to end. What is necessarily simulated is the last step: mapping
the CNN's 10-class clothing-category prediction onto an escalation-risk
score, since Fashion-MNIST carries no risk labels. That mapping is the
CATEGORY_RISK_WEIGHT table below: illustrative numbers, not learned from
data, and clearly documented as an assumption.

--------------------------------------------------------------------------
How the CNN score would be produced in a real system
--------------------------------------------------------------------------
In production you would train this same architecture on customer-submitted
product photos labelled with the outcome of the complaint (escalated or
not), so the CNN learns directly which visual patterns, such as visible
damage, the wrong item, or poor packaging, correlate with escalation. Its
softmax output would then be converted straight into a risk score between
0 and 1 (for example, the predicted probability of the "damaged or incorrect
item" class), with no hand-written category-weight table needed.

Run:
    python train_cnn_model.py
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras import layers, models

from risk_common import RANDOM_STATE

MODEL_PATH = "cnn_model.keras"

# Subset sizes chosen to keep runtime manageable; Fashion-MNIST ships with
# 60k training images and 10k test images in total.
TRAIN_SUBSET = 8000
TEST_SUBSET = 2000
EPOCHS = 5
BATCH_SIZE = 64

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# SIMULATED mapping from predicted clothing category -> "image risk weight"
# in [0, 1]. These are illustrative assumptions (e.g. footwear/bags are
# structurally complex and more often photographed as damaged/misshipped
# than basic tops), NOT learned from labelled complaint images. See the
# module docstring for how this would be replaced in a real system.
CATEGORY_RISK_WEIGHT = {
    "T-shirt/top": 0.35,
    "Trouser": 0.30,
    "Pullover": 0.40,
    "Dress": 0.45,
    "Coat": 0.45,
    "Sandal": 0.55,
    "Shirt": 0.35,
    "Sneaker": 0.60,
    "Bag": 0.65,
    "Ankle boot": 0.60,
}
_RISK_WEIGHT_VECTOR = np.array([CATEGORY_RISK_WEIGHT[c] for c in FASHION_MNIST_CLASSES])


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    rng = np.random.default_rng(RANDOM_STATE)
    train_idx = rng.choice(len(x_train), size=TRAIN_SUBSET, replace=False)
    test_idx = rng.choice(len(x_test), size=TEST_SUBSET, replace=False)
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    x_test, y_test = x_test[test_idx], y_test[test_idx]

    # Normalise to [0, 1] and add the channel dimension -> (N, 28, 28, 1)
    x_train = (x_train / 255.0).astype("float32")[..., None]
    x_test = (x_test / 255.0).astype("float32")[..., None]
    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(FASHION_MNIST_CLASSES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def cnn_score_from_proba(proba):
    """Convert softmax class probabilities into a single [0, 1] `cnn_score`.

    SIMULATED step (see module docstring): weights the predicted category
    distribution by the illustrative CATEGORY_RISK_WEIGHT table rather than
    a learned risk association.
    """
    proba = np.asarray(proba)
    return proba @ _RISK_WEIGHT_VECTOR


def main():
    tf.random.set_seed(RANDOM_STATE)
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"Train subset: {x_train.shape[0]} images | Test subset: {x_test.shape[0]} images")

    model = build_model()
    model.summary()

    model.fit(
        x_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    proba_test = model.predict(x_test, verbose=0)
    y_pred = proba_test.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    print(f"\n--- Test ({len(x_test)} images) ---")
    print(f"Accuracy: {acc:.3f} | Macro-F1: {f1_macro:.3f}")
    print(classification_report(y_test, y_pred, target_names=FASHION_MNIST_CLASSES, digits=3))

    print("\nSample cnn_score values (simulated category-risk weighting, 0=low .. 1=high):")
    scores = cnn_score_from_proba(proba_test[:5])
    for i, score in enumerate(scores):
        true_cls = FASHION_MNIST_CLASSES[y_test[i]]
        pred_cls = FASHION_MNIST_CLASSES[y_pred[i]]
        print(f"  true={true_cls:12s} predicted={pred_cls:12s} cnn_score={score:.3f}")

    model.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
