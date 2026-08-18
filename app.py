"""
app.py
------
Streamlit app to demonstrate 5 trained classification models on the
Breast Cancer Wisconsin (Diagnostic) dataset.

Features (per assignment requirements):
  a. Dataset upload option (CSV)
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix + classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer- Classifier Comparison", layout="wide")

st.title("🔬 Breast Cancer - Classification Model Comparison App")
st.markdown(
    "Upload your test CSV, pick a model, and view its evaluation metrics, "
    "confusion matrix, and classification report."
)

# ---------------------------------------------------------------------------
# Load available trained models
# ---------------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree Classifier": "model/decision_tree.pkl",
    "K-Nearest Neighbor Classifier": "model/knn.pkl",
    "Naive Bayes Classifier": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
}


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    with open("model/scaler.pkl", "rb") as f:
        return pickle.load(f)


scaler = load_scaler()

# ---------------------------------------------------------------------------
# (a) Dataset upload
# ---------------------------------------------------------------------------
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv (must include a 'target' column)", type=["csv"]
)

# ---------------------------------------------------------------------------
# (b) Model selection dropdown
# ---------------------------------------------------------------------------
st.sidebar.header("2. Choose a Model")
selected_model_name = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column with true labels.")
    else:
        X = df.drop(columns=["target"])
        y_true = df["target"]

        X_scaled = scaler.transform(X)

        model = load_model(MODEL_FILES[selected_model_name])
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

        # -------------------------------------------------------------
        # (c) Evaluation metrics
        # -------------------------------------------------------------
        st.subheader(f"📊 Evaluation Metrics — {selected_model_name}")

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }

        cols = st.columns(6)
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.4f}")

        # -------------------------------------------------------------
        # (d) Confusion matrix + classification report
        # -------------------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df)

        # -------------------------------------------------------------
        # All-models comparison on the uploaded data
        # -------------------------------------------------------------
        st.subheader("📈 All Models on Your Uploaded Data")
        rows = []
        for name, path in MODEL_FILES.items():
            m = load_model(path)
            pred = m.predict(X_scaled)
            proba = m.predict_proba(X_scaled)[:, 1]
            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_true, pred),
                "AUC": roc_auc_score(y_true, proba),
                "Precision": precision_score(y_true, pred),
                "Recall": recall_score(y_true, pred),
                "F1": f1_score(y_true, pred),
                "MCC": matthews_corrcoef(y_true, pred),
            })
        comparison_df = pd.DataFrame(rows).set_index("Model").round(4)
        st.dataframe(comparison_df.style.highlight_max(axis=0, color="lightgreen"))

else:
    st.info("👈 Upload `test_data.csv` from the sidebar to get started.")
    st.markdown(
        "Don't have the file? It's included in the GitHub repo as "
        "`test_data.csv` — download it from there and upload it here."
    )
