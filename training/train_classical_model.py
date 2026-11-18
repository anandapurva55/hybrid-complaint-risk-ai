"""
Classical ML baseline for complaint escalation risk.

Trains a single scikit-learn pipeline (TF-IDF over review text plus scaled and
encoded tabular features, feeding a Logistic Regression model) on train.csv,
reports accuracy and macro-F1 on val.csv and test.csv, and saves the fitted
pipeline to classical_model.pkl for use by hybrid_inference.py.

"Recommended IND" is intentionally excluded (it is a near-proxy of the rating
the label is derived from); see risk_common.py and evaluate_hybrid.py's ablation.

Run (after `python -m training.make_splits`):
    python -m training.train_classical_model
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hybrid_risk.paths import DATA_PROCESSED, MODELS
from hybrid_risk.risk_common import (
    CATEGORICAL_FEATURES,
    CLASSES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEXT_FEATURE,
    reorder_proba,
)

MODEL_PATH = MODELS / "classical_model.pkl"


def load_split(name):
    return pd.read_csv(DATA_PROCESSED / f"{name}.csv")


def build_pipeline():
    """TF-IDF for text, scaling for numeric, one-hot for categorical -> Logistic Regression.

    No class_weight: reweighting distorts the predicted probabilities, and the
    decision layer (decision.py) works on probabilities. The ~78/12/10 class
    imbalance is handled afterwards by tuning decision thresholds on the
    validation set instead of by skewing the model.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2), TEXT_FEATURE),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def evaluate(pipeline, df, split_name):
    X = df[FEATURE_COLUMNS]
    y_true = df[TARGET]
    y_pred = pipeline.predict(X)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n--- {split_name} ({len(df)} rows) ---")
    print(f"Accuracy: {acc:.3f} | Macro-F1: {f1_macro:.3f}")
    print(classification_report(y_true, y_pred, labels=CLASSES, digits=3))
    return {"accuracy": acc, "f1_macro": f1_macro}


def main():
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    pipeline = build_pipeline()

    print(f"Training Logistic Regression on {len(train_df)} rows ...")
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET])

    evaluate(pipeline, val_df, "Validation")
    evaluate(pipeline, test_df, "Test")

    # Sanity-check the probability ordering that hybrid_inference.py relies on.
    sample = test_df.iloc[:5]
    proba = reorder_proba(pipeline.predict_proba(sample[FEATURE_COLUMNS]), pipeline.classes_)
    print("\nSample probabilities (columns: low, medium, high):")
    for (_, row), p in zip(sample.iterrows(), proba):
        print(f"  true={row[TARGET]:6s} p={np.round(p, 3)}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
