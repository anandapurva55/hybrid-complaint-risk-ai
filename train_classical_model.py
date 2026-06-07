"""
Classical ML baseline for complaint escalation risk.

Trains a single scikit-learn pipeline (TF-IDF over review text plus scaled and
encoded tabular customer-profile features, feeding a Logistic Regression model)
on train.csv, reports accuracy and macro-F1 on val.csv and test.csv, and saves
the fitted pipeline to classical_model.pkl for use by hybrid_inference.py.

Run:
    python train_classical_model.py
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from risk_common import (
    CATEGORICAL_FEATURES,
    CLASSES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEXT_FEATURE,
    proba_to_score,
)

MODEL_PATH = "classical_model.pkl"


def load_split(name):
    return pd.read_csv(f"{name}.csv")


def build_pipeline():
    """TF-IDF for text, scaling for numeric, one-hot for categorical -> Logistic Regression.

    class_weight="balanced" compensates for the ~78/12/10 class imbalance found
    during data exploration (otherwise the model could just predict "low" and
    still look accurate).
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
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
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

    # Sanity-check the ordinal score conversion that hybrid_inference.py relies on.
    sample = test_df.iloc[:5]
    proba = pipeline.predict_proba(sample[FEATURE_COLUMNS])
    scores = proba_to_score(proba, pipeline.classes_)
    print("\nSample classical_score values (ordinal expected value, 0=low .. 1=high):")
    for (_, row), score in zip(sample.iterrows(), scores):
        print(f"  true={row[TARGET]:6s} predicted_label={pipeline.predict(row[FEATURE_COLUMNS].to_frame().T)[0]:6s} classical_score={score:.3f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
