# Model Evaluation & Benchmark Report
**Project:** CodeAlpha Data Science Internship — Task 1 (Iris Flower Classification)  
**Status:** PASSED (Strict zero data leakage protocol enforced)

---
## 1. Objective
The objective of Task 1 is to classify Iris flowers into three distinct species (*Iris setosa*, *Iris versicolor*, and *Iris virginica*) based on 4 morphological measurements (sepal length, sepal width, petal length, and petal width). The model must be trained, cross-validated on training data, and rigorously evaluated on an unseen test set.

## 2. Dataset Summary
- **Total Observations:** 150 Iris flower samples (`data/raw/iris.csv`)
- **Predictor Features (X):** `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (continuous numeric in cm)
- **Target Variable (y):** `species` mapped as: `setosa` → 0, `versicolor` → 1, `virginica` → 2
- **Dataset Split:** 80% Training (120 samples) and 20% Testing (30 samples) using stratified shuffle split (`random_state=42`)
- **Class Balance:** Perfectly balanced (40 samples per class in training; 10 samples per class in testing)
- **Missing Values:** 0 across all features
- **Data Integrity:** 1 canonical duplicate retained (sample at index 142) consistent with standard Iris benchmark literature.

## 3. Data Leakage Prevention & Test Isolation
- **Strict Isolation:** The 30-sample test set remained completely isolated and untouched during candidate model comparison and hyperparameter consideration.
- **Fold-Specific Scaling:** For scale-sensitive estimators (Logistic Regression, KNN), `StandardScaler` was placed inside Scikit-learn `Pipeline` objects so that scaling statistics ($\mu, \sigma$) were calculated strictly on training folds and never on validation folds or test data.
- **Single Evaluation:** The test set was accessed exactly once for the final reported performance metrics.

## 4. Candidate Models Evaluated
The following 4 diverse classifiers were compared using **only the 120-sample training partition**:
1. **Logistic Regression:** Linear classifier with `StandardScaler` pipeline to ensure scale normalization.
2. **K-Nearest Neighbors (KNN):** Distance-based instance classifier ($k=5$) with `StandardScaler` pipeline.
3. **Decision Tree Classifier:** Non-parametric recursive splitting tree classifier.
4. **Random Forest Classifier:** Ensemble of 100 decorrelated decision trees.

## 5. Cross-Validation Methodology & Results (Training Set Only)
- **Validation Setup:** Stratified 5-Fold Cross-Validation (`shuffle=True`, `random_state=42`).
- **Evaluation Metric:** Classification accuracy across validation folds.

| Candidate Algorithm | Preprocessing Pipeline | Mean CV Accuracy | Std Dev | Fold Scores |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | StandardScaler + Estimator | **0.9583** | ±0.0264 | `[0.958, 1.000, 0.958, 0.958, 0.917]` |
| **K-Nearest Neighbors** | StandardScaler + Estimator | **0.9583** | ±0.0264 | `[0.958, 1.000, 0.958, 0.917, 0.958]` |
| **Decision Tree** | Estimator (No Scaling Required) | **0.9500** | ±0.0167 | `[0.958, 0.958, 0.958, 0.958, 0.917]` |
| **Random Forest** | Estimator (No Scaling Required) | **0.9500** | ±0.0312 | `[0.958, 1.000, 0.958, 0.917, 0.917]` |

## 6. Selected Model & Reason for Selection
- **Selected Candidate:** **Logistic Regression**
- **Reason for Selection:** Under the 5-fold stratified cross-validation protocol on the training set, **Logistic Regression** achieved the highest cross-validation accuracy (95.83%) with low fold variance. It provides high linear interpretability, rapid inference, and well-calibrated class probabilities.
- **Contextual Note:** This model performed best among the evaluated candidates under this specific cross-validation setup; no universal optimality is asserted.

## 7. Final Test Metrics (Unseen 30 Samples)
The selected model was fitted once on the entire 120-sample training dataset and evaluated once on the isolated 30-sample test dataset (`random_state=42`, stratified 10 samples per class).

- **Final Test Accuracy:** **93.33%** (28/30 correctly classified)
- **Precision (Macro):** 0.9333
- **Recall (Macro):** 0.9333
- **F1-Score (Macro):** 0.9333
- **Precision (Weighted):** 0.9333
- **Recall (Weighted):** 0.9333
- **F1-Score (Weighted):** 0.9333

## 8. Classification Report
| Class (Species) | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| *Iris setosa* | 1.0000 | 1.0000 | 1.0000 | 10 |
| *Iris versicolor* | 0.9000 | 0.9000 | 0.9000 | 10 |
| *Iris virginica* | 0.9000 | 0.9000 | 0.9000 | 10 |
| **Macro Average** | 0.9333 | 0.9333 | 0.9333 | 30 |
| **Weighted Average** | 0.9333 | 0.9333 | 0.9333 | 30 |

## 9. Confusion Matrix Breakdown & Interpretation
Rows represent actual true classes; Columns represent model predictions:

| True \ Predicted | Predicted Setosa | Predicted Versicolor | Predicted Virginica | Total True |
| :--- | :--- | :--- | :--- | :--- |
| **True Setosa** | 10 | 0 | 0 | 10 |
| **True Versicolor** | 0 | 9 | 1 | 10 |
| **True Virginica** | 0 | 1 | 9 | 10 |

### Interpretation
- **Setosa:** Perfectly separated with 10/10 true positives (10/10). Morphologically distinct petal dimensions allow complete linear separability.
- **Versicolor:** 9/10 correctly identified, with 1 sample misclassified as Virginica due to boundary overlap in petal measurements.
- **Virginica:** 9/10 correctly identified, with 1 sample misclassified as Versicolor.

## 10. Visualizations Generated
- **Cross-Validation Comparison Plot:** Saved at [`reports/figures/model_comparison.png`](figures/model_comparison.png)
- **Confusion Matrix Heatmap:** Saved at [`reports/figures/confusion_matrix.png`](figures/confusion_matrix.png)

## 11. Limitations & Generalizability
- **Sample Size:** The Iris dataset contains 150 total samples (30 test samples). While high accuracy is achieved, variance on very small sample boundaries exists.
- **Domain Boundary:** Measurements reflect botanical specimens cultivated under specific historical conditions (Gaspé Peninsula); real-world wild variations in differing climates or hybrids may exhibit different measurement distributions.
- **No Overfitting Claim:** Model selection was conducted strictly via out-of-fold cross-validation, and final testing confirms realistic generalization on unseen test data without information leakage.
