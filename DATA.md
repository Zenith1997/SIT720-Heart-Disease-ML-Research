# Dataset and reproducibility

The coursework notebook uses the **1,025-row heart.csv dataset** published as `johnsmith88/heart-disease-dataset` on Kaggle, with a public fallback mirror in the notebook. The CSV is not checked into this repository. Refer to the original source and comply with its licence and terms before sharing a copy.

Place `heart.csv` beside the notebook or use the notebook's source-loading cell to download it. Expected columns: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`.

After installing `requirements.txt`, run:

```bash
python audit_duplicates.py --data heart.csv
```

With the same source data as the submitted notebook, the previously **recorded** counts were 1,025 rows, 723 exact duplicate rows and 302 unique complete rows. The new audit script checks deduplication before splitting and does not train a classifier or claim to recreate historical model scores.

For the complete research experiments, install `requirements-notebook.txt` and open the [original executed notebook](SIT720_11.1HD_heart_stacking_research_FINAL_EXECUTED.ipynb). Version differences or data-source changes may affect rerun results.
