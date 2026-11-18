"""
Create the train/val/test splits used by every trained component.

This is the script equivalent of the split cells in data_exploration.ipynb
(same cleaning, same label derivation, same stratified 70/15/15 split with
RANDOM_STATE = 42), so a fresh clone can reproduce train.csv / val.csv /
test.csv without opening the notebook:

    python make_splits.py

The label is a PROXY derived from the star rating (see risk_common.rating_to_risk
and the README's "What this is, and is not" section).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from risk_common import RANDOM_STATE, TARGET, TEXT_FEATURE, rating_to_risk

RAW_CSV = "Womens Clothing E-Commerce Reviews.csv"


def build_splits(raw_csv=RAW_CSV):
    df = pd.read_csv(raw_csv).drop(columns=["Unnamed: 0"])
    df[TARGET] = df["Rating"].apply(rating_to_risk)
    df = df.dropna(subset=[TEXT_FEATURE]).copy()
    for col in ["Division Name", "Department Name", "Class Name"]:
        df[col] = df[col].fillna("Unknown")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[TARGET], random_state=RANDOM_STATE
    )
    return train_df, val_df, test_df


def main():
    train_df, val_df, test_df = build_splits()
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split.to_csv(f"{name}.csv", index=False)
        print(f"{name:5s}: {len(split):5d} rows  {split[TARGET].value_counts(normalize=True).round(3).to_dict()}")
    print("Saved train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
