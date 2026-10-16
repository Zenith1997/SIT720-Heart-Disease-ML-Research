"""Audit exact row duplicates before evaluating a heart-disease classifier.

This is a lightweight verification tool, not a clinical prediction system.
Original classifier experiments are preserved in the published notebook.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def audit(frame: pd.DataFrame, seed: int = 42) -> dict:
    """Remove exact duplicates first, then perform a stratified hold-out split.

    This matches the independent sample definition in the coursework notebook.
    It never tunes classifiers or reports inflated predictive metrics.
    """
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    original = frame.loc[:, EXPECTED_COLUMNS].copy()
    if original.isna().any().any():
        raise ValueError("Missing values need handling before this duplicate audit")
    unique = original.drop_duplicates().reset_index(drop=True)
    counts = unique["target"].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Two target classes with at least two unique rows each are required")
    train, test = train_test_split(
        unique, test_size=0.2, random_state=seed, stratify=unique["target"]
    )
    # The original notebook's key finding was duplicate leakage: none of the
    # complete records in the test set should also appear in the training set.
    overlap = pd.concat([train, test]).duplicated().sum()
    if overlap:
        raise AssertionError("Train and test contain identical complete rows")
    return {
        "original_rows": int(len(original)),
        "duplicate_rows": int(original.duplicated().sum()),
        "unique_rows": int(len(unique)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "overlapping_complete_rows": 0,
        "random_seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a duplicate-aware train/test audit")
    parser.add_argument("--data", type=Path, default=Path("heart.csv"))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"{args.data} not found. See DATA.md for sources.")
    print(json.dumps(audit(pd.read_csv(args.data)), indent=2))


if __name__ == "__main__":
    main()
