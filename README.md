# When 98.54% accuracy isn't the whole story

**A reproducibility study of heart-disease classification, stacking and duplicate-data leakage.**

At first, a stacking classifier looked nearly perfect: **98.54% test accuracy**. But the source dataset contained repeated records. I checked for exact duplicates, removed them *before* dividing the data into training and testing sets, and compared the models under a more credible evaluation protocol.

## What changed

| From the original saved experiments | Recorded result |
|---|---:|
| Original rows | 1,025 |
| Exact duplicate rows | **723** |
| Distinct rows | 302 |
| Stacking on the duplicate-rich random split | 98.54% accuracy |
| Stacking on a deduplicated 20% held-out split | **80.33% accuracy** |
| Calibrated soft-voting on the same deduplicated split | 80.33% accuracy |

The two headline accuracies are **not** a controlled before-and-after comparison of otherwise identical modelling conditions. The deduplicated test set contains just 61 records, so uncertainty matters. The soft-voting approach did **not** improve accuracy compared with stacking on that split. This is an educational methods study, not a clinical diagnostic model.

## Why it's useful

An apparently great score can result from the way data is partitioned rather than from an especially effective model. This notebook reproduces a published stacking approach, examines leakage, then studies calibration, F1, ROC-AUC and uncertainty under a duplicate-aware protocol.

## Explore the actual work

- [Executed research notebook](SIT720_11.1HD_heart_stacking_research_FINAL_EXECUTED.ipynb)
- [Research report](SIT720_11.1HD_Report_FINAL.pdf)
- [Standalone duplicate audit](audit_duplicates.py)
- [Data source and rerun notes](DATA.md)
- [Automated audit tests](tests/test_data_audit.py)

## Check the data audit

```bash
python -m pip install -r requirements.txt
# Obtain the original heart.csv data (see DATA.md)
python audit_duplicates.py --data heart.csv
```

For the full notebook: `python -m pip install -r requirements-notebook.txt` followed by `jupyter notebook`. The original notebook and report are **preserved unchanged**. Its stored results were not independently rerun for this repository update.

To run the tests without any clinical records:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

**Author:** Zenith Anthony Dabarera | [AI/ML repositories](https://github.com/Zenith1997)
