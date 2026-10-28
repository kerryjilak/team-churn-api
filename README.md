# team-churn-api

A churn prediction model for a telecom customer dataset, with a FastAPI service that will serve it. The model is a scikit-learn logistic regression trained on the UCI churn dataset (`ucimlrepo` id 563).

## Repository layout

```
team-churn-api/
├── api/                     # FastAPI app; deployment contract: api.main:app
├── production_app/          # production logic and tests (services/, tests/)
├── ml_platform/
│   ├── training/churn/      # train.py and saved model artifacts
│   ├── serving/
│   ├── mlflow/
│   ├── monitoring/
│   └── features/
├── data_platform/
├── distributed/
├── genai/
├── client_delivery/
├── pyproject.toml           # team-owned dependencies
└── uv.lock                  # team-owned exact versions
```

Empty folders contain a `.gitkeep` so Git tracks them.

## Setup

Requires Python 3.13 (see `.python-version`) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
```

`--locked` installs the exact versions in `uv.lock` and fails if the lock file doesn't match `pyproject.toml`. Do not regenerate `uv.lock` to get past that error. Dependency changes go through a peer-reviewed PR.

## Train the model

From the repository root:

```bash
uv run --locked python ml_platform/training/churn/train.py
```

Training uses a logistic regression pipeline (imputation, then standard scaling) with an 80/20 train/test split, `random_state=42`, stratified on the target.

## Dataset

- **Source:** UCI churn dataset, fetched with `ucimlrepo` (id 563). It is downloaded at runtime, not stored in this repo.
- **Size:** 3,150 customers, 13 input features.
- **Target:** `Churn` (1 = churned, 0 = did not churn).

| Class | Share | Approx. count |
|-------|-------|---------------|
| 0 (did not churn) | 84.29% | 2,655 |
| 1 (churned) | 15.71% | 495 |

The classes are imbalanced, so accuracy alone is misleading. Judge the model on the churn class (precision, recall, F1) and ROC AUC.

## Feature names

The 13 input features, as they appear in the dataset:

1. Call Failure
2. Complains
3. Subscription Length
4. Charge Amount
5. Seconds of Use
6. Frequency of use
7. Frequency of SMS
8. Distinct Called Numbers
9. Age Group
10. Tariff Plan
11. Status
12. Age
13. Customer Value

## Metrics

Test set: 630 customers (20% of 3,150).

| Metric | Value |
|--------|-------|
| ROC AUC | 0.9243 |
| Accuracy | 0.84 |
| Precision, churn class (1) | 0.50 |
| Recall, churn class (1) | 0.86 |
| F1, churn class (1) | 0.63 |

Full classification report:

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (did not churn) | 0.97 | 0.84 | 0.90 | 531 |
| 1 (churned) | 0.50 | 0.86 | 0.63 | 99 |
| Macro avg | 0.73 | 0.85 | 0.76 | 630 |
| Weighted avg | 0.90 | 0.84 | 0.86 | 630 |

The model catches most churners (recall 0.86), but about half of the customers it flags as churners don't churn (precision 0.50). Record the run date, the git commit, and the scikit-learn version with the saved artifacts.

## Feature coefficients

Coefficients from the reference run in the lab. Re-check them against your run once `train.py` is complete.

| Feature | Coefficient | Odds ratio |
|---------|-------------|------------|
| Frequency of SMS | -1.812163 | 0.163301 |
| Frequency of use | -1.769819 | 0.170364 |
| Complains | 1.112679 | 3.042497 |
| Call Failure | 0.938941 | 2.557273 |
| Status | 0.698531 | 2.010797 |
| Charge Amount | -0.696633 | 0.498260 |
| Seconds of Use | 0.580483 | 1.786902 |
| Distinct Called Numbers | -0.507615 | 0.601930 |
| Customer Value | 0.472329 | 1.603725 |
| Subscription Length | -0.361796 | 0.696424 |
| Age | -0.068191 | 0.934082 |
| Tariff Plan | 0.028535 | 1.028946 |
| Age Group | 0.020847 | 1.021066 |

**Top 3 by absolute coefficient:** Frequency of SMS, Frequency of use, Complains.

Positive coefficients push the prediction toward churn. Negative coefficients push it away.

## Team workflow

Training is split into two roles:

1. **Trainer:** runs the training script, records the metrics above, and commits the artifacts.
2. **Checker:** confirms the feature names and the saved files match this README before the PR is approved.

## Artifacts

Commit only the small trusted outputs, so CI and fresh clones can reproduce results without retraining:

- the trained model file
- a metadata file with the metrics, feature names, and library versions

Do not commit the downloaded dataset.

## Interpreting errors

A **false negative** is a customer who will churn but is predicted to stay. That customer is missed, and the business can lose them because nobody acted to keep them. Recall on the churn class is the metric that tracks this risk most directly.
