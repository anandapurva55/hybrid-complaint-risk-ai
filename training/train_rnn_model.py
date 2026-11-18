"""
RNN / LSTM component.

Trains a small LSTM text classifier directly on customer review text to
predict risk_level, using the same train, validation, and test splits
produced in data_exploration.ipynb as the classical baseline, so every
component is evaluated on identical rows (no leakage, comparable metrics).

Trains on the FULL training split (earlier versions used a 6,000-row subset),
with early stopping on validation loss. No class weights: the decision layer
works on probabilities, and class imbalance is handled by threshold tuning
(see decision.py / tune_decision.py).

Run (after `python -m training.make_splits`):
    python -m training.train_rnn_model
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from hybrid_risk.paths import DATA_PROCESSED, MODELS
from hybrid_risk.risk_common import CLASSES, RANDOM_STATE, TARGET, TEXT_FEATURE

MODEL_PATH = str(MODELS / "rnn_model.keras")

MAX_TOKENS = 10000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 64
EPOCHS = 12  # upper bound; early stopping picks the best epoch
BATCH_SIZE = 64
TRAIN_SUBSET = None  # None = use the full train split

# low=0, medium=1, high=2, matching risk_common.ORDINAL_WEIGHTS / CLASSES order,
# so the softmax output columns line up directly with proba_to_score's expectation.
LABEL_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def load_split(name):
    df = pd.read_csv(DATA_PROCESSED / f"{name}.csv")
    texts = df[TEXT_FEATURE].astype(str).to_numpy()
    labels = df[TARGET].map(LABEL_TO_ID).to_numpy()
    return texts, labels


def stratified_subsample(texts, labels, size, seed):
    if size is None or size >= len(texts):
        return texts, labels
    idx_subset, _ = train_test_split(
        np.arange(len(texts)), train_size=size, stratify=labels, random_state=seed
    )
    return texts[idx_subset], labels[idx_subset]


def build_model(vectorizer):
    model = models.Sequential([
        layers.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        layers.Embedding(input_dim=MAX_TOKENS, output_dim=EMBEDDING_DIM, mask_zero=True),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def evaluate(model, texts, labels, split_name):
    proba = model.predict(texts, verbose=0)
    y_pred = proba.argmax(axis=1)

    acc = accuracy_score(labels, y_pred)
    f1_macro = f1_score(labels, y_pred, average="macro")
    print(f"\n--- {split_name} ({len(texts)} rows) ---")
    print(f"Accuracy: {acc:.3f} | Macro-F1: {f1_macro:.3f}")
    print(classification_report(labels, y_pred, target_names=CLASSES, digits=3))
    return proba


def main():
    tf.random.set_seed(RANDOM_STATE)

    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")
    test_texts, test_labels = load_split("test")

    train_texts, train_labels = stratified_subsample(train_texts, train_labels, TRAIN_SUBSET, RANDOM_STATE)
    print(f"Train subset: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # Fit the vectorizer ONLY on training text, to avoid leakage (see data_exploration.ipynb)
    vectorizer = layers.TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=SEQUENCE_LENGTH)
    vectorizer.adapt(train_texts)

    model = build_model(vectorizer)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    model.fit(
        train_texts, train_labels,
        validation_data=(val_texts, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    evaluate(model, val_texts, val_labels, "Validation")
    evaluate(model, test_texts, test_labels, "Test")

    model.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
