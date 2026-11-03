import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=563)

#calling the churn dataset from the ucimlrepo package and printing the shape of the features and the distribution of the target variable
X = dataset.data.features
y = dataset.data.targets.iloc[:, 0]

print("Shape:", X.shape)
print("Target distribution:")
print(y.value_counts(normalize=True))
print(dataset.variables)


# Training the model itself with the data using Logistic Regression and a pipeline that includes imputation and scaling. The data is split into training and testing sets, with stratification to maintain the target variable distribution.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )),
])

pipeline.fit(X_train, y_train)


# Get the coefficients from the fitted logistic regression model
model = pipeline.named_steps["model"]

coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0],
})

# Calculate odds ratios to help interpret the coefficients
coefficients["Odds Ratio"] = np.exp(coefficients["Coefficient"])

# Rank features by the absolute size of their coefficients
coefficients["Absolute Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Absolute Coefficient",
    ascending=False
)

print("Feature coefficients:")
print(coefficients.to_string(index=False))

print("\nTop 3 features by absolute coefficient:")
print(coefficients.head(3).to_string(index=False))


# The evaluation of the model is done using classification report and ROC-AUC score to assess the performance of the logistic regression model on the test set.
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))


# Saving the trained model and the feature names to disk for future use. The model is saved using joblib, and the feature names are saved in a JSON file.
artifact_dir = Path("ml_platform/training/churn/models")
artifact_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    pipeline,
    artifact_dir / "churn_pipeline.joblib",
)

(artifact_dir / "feature_names.json").write_text(
    json.dumps(X.columns.tolist(), indent=2)
)