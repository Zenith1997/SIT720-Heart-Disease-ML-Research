import pandas as pd
import pytest

from audit_duplicates import EXPECTED_COLUMNS, audit


def sample():
    # Artificial rows to test duplicate removal only; not clinical data.
    frame = pd.DataFrame([
        [40 + i, i % 2, i % 4, 100 + i, 180 + i, i % 2, 0,
         150 + i, 0, float(i % 4), 0, i % 2, i % 3, i % 2]
        for i in range(30)
    ], columns=EXPECTED_COLUMNS)
    return pd.concat([frame, frame.iloc[:10]], ignore_index=True)


def test_duplicate_audit_before_split():
    summary = audit(sample())
    assert summary["original_rows"] == 40
    assert summary["duplicate_rows"] == 10
    assert summary["unique_rows"] == 30
    assert summary["train_rows"] + summary["test_rows"] == 30
    assert summary["overlapping_complete_rows"] == 0


def test_missing_target_is_rejected():
    with pytest.raises(ValueError, match="Missing columns"):
        audit(sample().drop(columns=["target"]))


def test_missing_values_are_not_silently_imputed():
    frame = sample()
    frame.loc[0, "age"] = None
    with pytest.raises(ValueError, match="Missing values"):
        audit(frame)


def test_audit_does_not_modify_input():
    frame = sample()
    before = frame.copy(deep=True)
    audit(frame)
    pd.testing.assert_frame_equal(frame, before)
