# Heart Disease ML: Stacking Reproduction and Data Leakage Audit

A reproduction study showing how duplicate records can inflate model evaluation, followed by a comparison on unique records.

**Python · scikit-learn · XGBoost · model evaluation · calibration**

## Why this project matters

A high score is useful only when the evaluation is credible. This project investigates the difference between a paper-style random split and an evaluation that removes exact duplicate records before splitting.

## Evidence from the saved experiment

| Finding | Recorded result |
|---|---:|
| Original dataset | 1,025 rows |
| Exact duplicate rows | 723 |
| Unique records | 302 |
| Stacking accuracy on the duplicate-rich split | 98.54% |
| Stacking accuracy on the unique-record split | 80.33% |
| Proposed calibrated soft-voting accuracy on unique records | 80.33% |

These are different experimental protocols, not a like-for-like estimate of a deployment improvement. The proposed approach did **not** improve accuracy over stacking on the unique-record evaluation. Its recorded F1 was 0.8182 versus 0.8125 for stacking; the test set contained only 61 records.

## Technical work

- Reproduce and compare classical classifiers and a stacking ensemble.
- Audit duplicate rows before interpreting headline metrics.
- Tune candidate models using training-data cross-validation.
- Compare a leakage-aware calibrated soft-voting approach with stacking.
- Report discrimination, calibration and bootstrap uncertainty.

## Explore the project

- [Executed notebook](SIT720_11.1HD_heart_stacking_research_FINAL_EXECUTED.ipynb)
- [Research report](SIT720_11.1HD_Report_FINAL.pdf)
- [Evidence notes](EVIDENCE.md)

## Run locally

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install jupyter numpy pandas matplotlib scikit-learn xgboost
jupyter notebook
```

Open the notebook and follow its data-loading cell for `heart.csv` or the configured fallback source. Dataset access requires network connectivity if no local copy is supplied. These dependencies reflect notebook imports; this publication is not a fresh environment validation.

## Status and limitations

The notebook contains saved execution outputs. Results above come from those outputs, not a new run. This is educational research on a small public dataset, with no external clinical validation.

**Author:** Zenith Anthony Dabarera · SIT720 Machine Learning  
[AI/ML portfolio](https://github.com/Zenith1997)
