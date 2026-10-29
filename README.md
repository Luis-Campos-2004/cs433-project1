# CS-433 Project 1: predicting heart disease (MICHD) from BRFSS 2015

We predict whether a respondent of the 2015 Behavioral Risk Factor Surveillance
System survey has coronary heart disease (MICHD) from 321 survey answers, using
only NumPy. Final model: BRFSS-aware preprocessing, one-hot encoding of the
categorical answers, L2-regularized logistic regression trained by gradient
descent, and a decision threshold tuned for F1 (5-fold cross-validated F1 = 0.426).

## Setup

- Python 3 with `numpy` (enough for `run.py`), plus `matplotlib` for the
  figures and `pytest` for the tests.
- The data is **not** in the repository. Download the project dataset from
  AIcrowd and put the three CSV files in a `data/` folder at the root:

```
data/x_train.csv
data/y_train.csv
data/x_test.csv
```

## Reproduce our submission

From the root of the repository:

```bash
python run.py
```

It takes about 3 minutes and writes `submissions/submission.csv`, the file of
our final AIcrowd submission. There is no randomness in the pipeline, so it
always writes exactly the same file.

## Reproduce the report

```bash
python experiments.py     # about 1 hour: model comparison + ablation (5-fold CV)
python run_mlp.py         # one-hidden-layer MLP, appended to the results
python run_blend.py       # logistic + MLP blend, appended to the results
python make_figures.py    # figures/models/*.pdf from the results
```

`experiments.py` writes `results/cv_results.csv` (the experiment log) and
`results/cv_results.md` (the same table, readable), plus
`figures/models/convergence.pdf`. All experiments use the same 5 stratified
folds (seed 42), and every step (preprocessing, features, model, threshold) is
fitted on the training folds only.

## Tests

```bash
pytest -q
```

## Files

| File | Content |
|---|---|
| `implementations.py` | The 6 required methods (GD, SGD, least squares, ridge, logistic, regularized logistic) |
| `helpers.py` | Course helpers to load the data and write the submission |
| `run.py` | Trains the final model on all training rows and writes the submission |
| `experiments.py` | Cross-validation experiments behind the report |
| `run_mlp.py`, `run_blend.py` | Cross-validation of the MLP and of the logistic + MLP blend |
| `run_blend_submission.py` | Submission file of the blend (not our final submission) |
| `make_figures.py` | Figures of the report from the experiment log |
| `model.py` | First baseline: mean imputation + regularized logistic regression |
| `src/preprocessing.py` | BRFSS-aware preprocessing (special codes, column selection, imputation, scaling) |
| `src/features.py` | One-hot encoding and squared features |
| `src/training.py` | Model wrappers and the gradient descent step size |
| `src/cross_validation.py` | Stratified k-fold, F1, threshold tuning |
| `src/mlp.py` | One-hidden-layer MLP in NumPy |
| `notebooks/preprocessing.ipynb` | Exploratory data analysis and preprocessing figures |
| `tests/` | Unit tests |
| `figures/`, `results/` | Figures and cross-validation results used in the report |
