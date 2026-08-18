# Breast Cancer Classification — Model Comparison App

## a. Problem Statement

This project implements and compares five classification algorithms to predict
whether a breast tumor is **malignant** or **benign** based on features computed
from digitized images of a fine needle aspirate (FNA) of a breast mass. This is
a **binary classification** problem with real clinical relevance: correctly
identifying malignant cases (minimizing false negatives) is critical for early
diagnosis.

## b. Dataset Description

- **Dataset**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: UCI Machine Learning Repository / scikit-learn built-in loader
  (`sklearn.datasets.load_breast_cancer`) —
  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Instances**: 569 (exceeds the 500-instance minimum)
- **Features**: 30 numeric features (exceeds the 12-feature minimum) — e.g.
  `mean radius`, `mean texture`, `mean perimeter`, `mean area`,
  `mean smoothness`, `worst concavity`, etc. — computed from cell nuclei
  present in the image.
- **Target classes**: `0 = malignant` (212 instances), `1 = benign`
  (357 instances)
- **Train/Test split**: 80/20 stratified split (455 train / 114 test)
- Features were standardized (`StandardScaler`) before training
  distance-/gradient-based models (Logistic Regression, kNN).

## c. GitHub Repository Link

> `<PASTE YOUR GITHUB REPO LINK HERE AFTER PUSHING>`

## d. Models Used

All 5 models were trained on the same 80/20 train/test split of the dataset
above, then evaluated on the held-out 114-row test set.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9932 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer across every single metric. After standardizing the 30 features, the two classes become close to linearly separable, which plays directly to Logistic Regression's strength. Its high MCC (0.9623) confirms the result isn't a fluke of class imbalance. |
| Decision Tree | Weakest model on every metric. A single unpruned tree overfits the training data and has high variance — it memorizes noise in the training split rather than learning generalizable decision boundaries, which shows up as the lowest AUC (0.9157) and MCC (0.8174). |
| kNN | Strong, well-balanced performance (Accuracy 0.9561, AUC 0.9788). Scaling the features before training was essential here, since kNN relies directly on Euclidean distance and unscaled features (e.g. `area` vs `smoothness`) would otherwise dominate the distance calculation. |
| Naive Bayes | Decent but not top-tier (Accuracy 0.9298). Gaussian Naive Bayes assumes features are conditionally independent given the class, which is a poor assumption here since many cell-measurement features (e.g. radius, perimeter, area) are highly correlated. Interestingly its AUC (0.9868) is still high, meaning its predicted probabilities rank patients well even though the default 0.5 threshold produces more misclassifications. |
| Random Forest (Ensemble) | Ties kNN on accuracy/precision/recall/F1 but has a noticeably higher AUC (0.9932) and is far more robust than the single Decision Tree it's built from — averaging many trees cancels out the overfitting/high-variance problem seen above, at the cost of being less interpretable than a single tree. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it achieved the top score on all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC), making it the clear best choice for this dataset. |

---

## Project Structure

```
project-folder/
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── test_data.csv                 # Held-out test set (features + true labels)
├── model_comparison_results.csv  # Raw metric output from train_models.py
└── model/
    ├── train_models.py           # Source code that trains & saves all 5 models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl                # StandardScaler fitted on training data
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py     # (optional) re-trains all models from scratch
streamlit run app.py
```

## Live App

> `<PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE AFTER DEPLOYING>`

## App Features

- **Dataset upload**: Upload `test_data.csv` (or any CSV with the same 30
  feature columns + a `target` column) via the sidebar.
- **Model selection dropdown**: Choose any of the 5 trained models to
  evaluate.
- **Evaluation metrics display**: Accuracy, AUC, Precision, Recall, F1,
  MCC shown instantly for the selected model.
- **Confusion matrix & classification report**: Visual confusion matrix
  heatmap plus a full per-class classification report table.
- **All-models comparison**: A bonus table showing how all 5 models perform
  side-by-side on whatever data you uploaded.
