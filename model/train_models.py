"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates each with 6 metrics, saves every trained model to
model/*.pkl, and writes test_data.csv (used later by the Streamlit app).

Dataset: Breast Cancer Wisconsin (Diagnostic)
Source : scikit-learn built-in (originally UCI ML Repository)
         https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Samples: 569  |  Features: 30  |  Classes: 2 (malignant=0, benign=1)
"""

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

os.makedirs("model", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")   # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution:\n{y.value_counts()}")

# ---------------------------------------------------------------------------
# 2. Train / test split (80/20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features (helps Logistic Regression / KNN converge & perform well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save the RAW (unscaled) test data + true labels -> this is what gets
# uploaded to the Streamlit app and submitted as test_data.csv
test_data = X_test.copy()
test_data["target"] = y_test.values
test_data.to_csv("test_data.csv", index=False)
print(f"\nSaved test_data.csv with {len(test_data)} rows")

# ---------------------------------------------------------------------------
# 3. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbor Classifier": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes Classifier": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

# ---------------------------------------------------------------------------
# 4. Train, evaluate, save each model
# ---------------------------------------------------------------------------
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(metrics)

    # Save trained model
    fname = "model/" + name.lower().replace(" ", "_") + ".pkl"
    with open(fname, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {fname}")

# ---------------------------------------------------------------------------
# 5. Comparison table
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results).set_index("ML Model Name").round(4)
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(results_df.to_string())

results_df.to_csv("model_comparison_results.csv")
print("\nSaved model_comparison_results.csv")
