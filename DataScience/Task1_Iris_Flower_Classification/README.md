# CodeAlpha Data Science Internship — Task 1: Iris Flower Classification

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Passed](https://img.shields.io/badge/tests-25%2F25%20passed-brightgreen.svg)](tests/)
[![Internship: CodeAlpha](https://img.shields.io/badge/Internship-CodeAlpha%20(Sept%202026)-orange.svg)](https://codealpha.tech/)

---

## 1. Project Overview
This repository hosts the official, verified implementation for **Task 1: Iris Flower Classification** as part of the **CodeAlpha Data Science Internship (September 2026)**.

The objective is to train a supervised machine learning classification pipeline to identify Iris flower species based on morphological measurements of their sepals and petals, applying sound Data Science principles such as zero-leakage cross-validation, proper pipeline feature scaling, and multi-metric test set evaluation.

---

## 2. Official Task Objective
- **Input Data:** Morphological measurements of Iris flower specimens:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target Species:**
  - *Iris setosa* (Class `0`)
  - *Iris versicolor* (Class `1`)
  - *Iris virginica* (Class `2`)
- **Deliverables:** Reusable preprocessing, candidate model benchmarking, out-of-sample test evaluation, prediction engine, exploratory visualizations, and test suite.

---

## 3. Dataset & Data Quality
* **Provenance:** Acquired programmatically via Scikit-learn's official benchmark loader (`sklearn.datasets.load_iris`) and saved as [`data/raw/iris.csv`](data/raw/iris.csv) in accordance with CodeAlpha task specifications.
* **Dimensions:** 150 instances, 4 numeric continuous predictors, 1 categorical target.
* **Missing Values:** Exactly 0 missing values across all columns (100% complete).
* **Duplicate Sample:** Exactly 1 duplicate row detected (canonical sample at index 142). Retained in accordance with Fisher/Anderson benchmark literature as a valid biological specimen with identical rounded measurements.

| Feature Name | Description | Range (Min – Max) | Mean ± Std Dev |
| :--- | :--- | :---: | :---: |
| `sepal_length` | Length of sepal in centimeters | 4.30 – 7.90 cm | 5.84 ± 0.83 cm |
| `sepal_width` | Width of sepal in centimeters | 2.00 – 4.40 cm | 3.06 ± 0.44 cm |
| `petal_length` | Length of petal in centimeters | 1.00 – 6.90 cm | 3.76 ± 1.76 cm |
| `petal_width` | Width of petal in centimeters | 0.10 – 2.50 cm | 1.20 ± 0.76 cm |

---

## 4. Methodology & Data Leakage Prevention
1. **Stratified Partitioning:** An 80/20 Stratified Shuffle Split (`random_state=42`) was performed once, creating:
   - **Training Set (120 samples):** Exactly 40 samples per species.
   - **Test Set (30 samples):** Exactly 10 samples per species.
2. **Zero Data Leakage:** The 30-sample test set remained strictly unseen during candidate model selection and hyperparameter assessment.
3. **Pipeline Encapsulation:** For scale-sensitive estimators (Logistic Regression, KNN), `StandardScaler` was packaged inside Scikit-learn `Pipeline` objects, ensuring mean and variance statistics were calculated strictly on training folds.

---

## 5. Candidate Model Benchmarking (Training Set Only)
Candidate models were compared using **5-Fold Stratified Cross-Validation (`shuffle=True`, `random_state=42`)** on the 120 training samples:

| Candidate Algorithm | Preprocessing Pipeline | Mean 5-Fold CV Accuracy | Fold Std Dev | Selection Status |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | `StandardScaler` + `LogisticRegression(max_iter=200)` | **95.83%** | $\pm 0.0264$ | **Selected** |
| **K-Nearest Neighbors** | `StandardScaler` + `KNeighborsClassifier(n_neighbors=5)` | **95.83%** | $\pm 0.0264$ | Candidate |
| **Decision Tree** | Single Decision Tree (Raw Measurements) | **95.00%** | $\pm 0.0167$ | Candidate |
| **Random Forest** | 100 Decision Trees (Raw Measurements) | **95.00%** | $\pm 0.0312$ | Candidate |

*Selection Rationale:* **Logistic Regression** tied for the highest CV accuracy (95.83%) and was selected for its high interpretability, linear decision stability, and well-calibrated class probability estimates.

---

## 6. Final Model Test Evaluation (Unseen 30 Samples)
The selected model was fitted once on the entire 120-sample training dataset and evaluated once on the isolated 30-sample test set:

- **Test Accuracy:** **93.33%** (28/30 correctly classified)
- **Macro Precision:** **0.9333**
- **Macro Recall:** **0.9333**
- **Macro F1-Score:** **0.9333**
- **Weighted F1-Score:** **0.9333**

### Per-Class Test Breakdown
| Species Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| *Iris setosa* | 1.0000 | 1.0000 | 1.0000 | 10 |
| *Iris versicolor* | 0.9000 | 0.9000 | 0.9000 | 10 |
| *Iris virginica* | 0.9000 | 0.9000 | 0.9000 | 10 |

### Confusion Matrix
```text
                  Predicted Setosa   Predicted Versicolor   Predicted Virginica
True Setosa               10                   0                      0
True Versicolor            0                   9                      1
True Virginica             0                   1                      9
```
*Notice: Setosa is linearly separable and 100% correct. Versicolor and Virginica have natural physical overlap in petal dimensions, resulting in 1 mutual boundary misclassification.*

---

## 7. Generated Visualizations
All high-resolution figures are located in [`reports/figures/`](reports/figures/):
1. **Class Distribution:** [`reports/figures/class_distribution.png`](reports/figures/class_distribution.png)
2. **Feature Distributions (Boxplots):** [`reports/figures/feature_distributions.png`](reports/figures/feature_distributions.png)
3. **Pairwise Feature Relationships:** [`reports/figures/pairwise_feature_relationships.png`](reports/figures/pairwise_feature_relationships.png)
4. **Model Comparison:** [`reports/figures/model_comparison.png`](reports/figures/model_comparison.png)
5. **Confusion Matrix:** [`reports/figures/confusion_matrix.png`](reports/figures/confusion_matrix.png)

---

## 8. Prediction Example & Demonstration
Inference can be performed programmatically via `src/predict.py`:

```python
from src.predict import predict_species

# Predict species for a flower with:
# sepal_length=5.1 cm, sepal_width=3.5 cm, petal_length=1.4 cm, petal_width=0.2 cm
result = predict_species(5.1, 3.5, 1.4, 0.2)

print("Predicted Species:", result["predicted_species"])
print("Class Probabilities:", result["probabilities"])
```

**Output:**
```text
Predicted Species: setosa
Class Probabilities: {'setosa': 0.9808, 'versicolor': 0.0192, 'virginica': 0.0}
```

A demonstration script with representative archetype samples is runnable via:
```powershell
python src/prediction_demo.py
```

---

## 9. Project Structure
```text
CodeAlpha_Iris_Flower_Classification/
│
├── .venv/                         # Python 3.13 virtual environment
│
├── data/
│   └── raw/
│       └── iris.csv               # Raw Iris dataset (150 samples)
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_inspection.py         # Dataset acquisition and structural verification
│   ├── preprocessing.py           # Preprocessing, label encoding, and stratified split
│   ├── model_training.py          # 5-fold CV candidate benchmarking, training, and evaluation
│   ├── predict.py                 # Production prediction engine with validation and probabilities
│   ├── prediction_demo.py         # Clean representative sample demonstration script
│   └── visualization.py           # Exploratory and distribution plotting module
│
├── models/
│   ├── iris_classifier.joblib     # Serialized trained Logistic Regression pipeline
│   └── model_metadata.json        # Schema, label mappings, CV scores, and test metrics
│
├── reports/
│   ├── figures/                   # Generated PNG plots (distributions, pairplot, CM, CV comparison)
│   ├── dataset_inspection_report.md
│   ├── preprocessing_report.md
│   ├── model_evaluation_report.md
│   └── final_project_report.md    # Comprehensive internship submission report
│
├── tests/
│   ├── test_preprocessing.py      # 9 unit tests for dataset loading, splits, and leakage checks
│   ├── test_model_training.py     # 6 unit tests for candidate models, training, metrics, and persistence
│   └── test_prediction.py          # 10 unit tests for input validation, inference, and plotting
│
├── .gitignore                     # Data Science Git ignore configuration
├── requirements.txt               # Pinned project dependencies
├── README.md                      # Complete project documentation
└── main.py                        # Entrypoint orchestration script
```

---

## 10. Installation & Execution Guide

### 1. Clone & Set Up Environment
```powershell
# Navigate to project directory
cd CodeAlpha_Iris_Flower_Classification

# Create virtual environment (if not already existing)
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Inspection
```powershell
python src/data_inspection.py
```

### 3. Run Preprocessing Pipeline
```powershell
python src/preprocessing.py
```

### 4. Run Model Training & Evaluation
```powershell
python src/model_training.py
```

### 5. Generate Exploratory Visualizations
```powershell
python src/visualization.py
```

### 6. Run Prediction Demonstration
```powershell
python src/prediction_demo.py
```

### 7. Run Complete Unit Test Suite
```powershell
python -m unittest discover tests
```

---

## 11. Limitations & Future Improvements
* **Dataset Scale:** The benchmark contains 150 instances from 1936. While ideal for methodology demonstration, it does not represent modern wild population variance.
* **Geographical Boundary:** Measurements originate from Gaspé Peninsula specimens. Different environmental climates, soils, or hybridization would require fine-tuning or transfer learning.
* **Distinction of Metrics:** The 5-fold cross-validation accuracy of **95.83%** reflects training stability; out-of-sample generalization was confirmed at **93.33%** on the isolated test set.
* **Future Work:** Integration with modern interactive dashboards (e.g., Streamlit) and continuous integration (CI) workflows for automated regression testing.
