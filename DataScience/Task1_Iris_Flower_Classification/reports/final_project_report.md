# Final Project Report: Iris Flower Classification

**Program:** CodeAlpha Data Science Internship (September 2026)  
**Task:** Task 1 — Iris Flower Classification  
**Author / Intern:** CodeAlpha Data Science Intern  
**Environment:** Python 3.13.2 (`.venv`), Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn  
**Status:** Completed and Rigorously Verified  

---

## 1. Project Objective
The goal of this project is to develop, validate, and document an end-to-end Machine Learning classification system capable of predicting the botanical species of an Iris flower based on its physical morphological measurements (sepal length, sepal width, petal length, and petal width).

---

## 2. CodeAlpha Task 1 Requirements
The official task mandates:
* Use measurements of Iris flowers as input data (sepal length, sepal width, petal length, petal width).
* Classify the flowers into three canonical target species: *Iris setosa*, *Iris versicolor*, and *Iris virginica*.
* Train a machine learning classification model using Python and standard Data Science / Machine Learning libraries.
* Evaluate accuracy and classification performance metrics on an isolated test dataset.
* Maintain a professional, clean, modular codebase structured for portfolio presentation, GitHub publication, and internship submission.

---

## 3. Dataset Source and Provenance
* **Acquisition Method:** Programmatically loaded via Scikit-learn's official benchmark loader (`sklearn.datasets.load_iris`) and serialized to [`data/raw/iris.csv`](../data/raw/iris.csv).
* **Rationale:** Because a direct clickable URL was not provided in the assignment description, Scikit-learn's built-in loader was utilized as explicitly recommended in the CodeAlpha internship instructions. No third-party or untrusted external datasets were used.

---

## 4. Dataset Structure & Dimensions
* **Total Instances (Rows):** 150 flower samples
* **Total Attributes (Columns):** 5 (4 continuous numeric features + 1 categorical species target)
* **Predictor Features ($X$):**
  1. `sepal_length`: Sepal length in centimeters (range: 4.30 – 7.90 cm, mean: 5.84 cm)
  2. `sepal_width`: Sepal width in centimeters (range: 2.00 – 4.40 cm, mean: 3.06 cm)
  3. `petal_length`: Petal length in centimeters (range: 1.00 – 6.90 cm, mean: 3.76 cm)
  4. `petal_width`: Petal width in centimeters (range: 0.10 – 2.50 cm, mean: 1.20 cm)
* **Target Variable ($y$):** `species`
  - *Iris setosa* (50 samples, 33.33%)
  - *Iris versicolor* (50 samples, 33.33%)
  - *Iris virginica* (50 samples, 33.33%)

---

## 5. Data Quality & Integrity Findings
* **Missing Values:** Exactly 0 missing or NaN values across all 150 rows.
* **Duplicate Rows:** Exactly 1 duplicate row was detected in the raw dataset (sample at index 142: `[5.8, 2.7, 5.1, 1.9, virginica]`).
* **Duplicate Handling Decision:** Retained. In Fisher's canonical 1936 Iris benchmark, this represents an authentic, physically distinct biological specimen that coincidentally shared rounded centimeter measurements with sample #101, not a transcription error. The raw dataset remains unaltered.

---

## 6. Preprocessing & Encoding
* **Target Label Encoding:** Deterministic integer encoding was applied:
  - `setosa` $\rightarrow$ `0`
  - `versicolor` $\rightarrow$ `1`
  - `virginica` $\rightarrow$ `2`
  *(Bidirectional inverse mapping is preserved in `models/model_metadata.json` for prediction decoding).*
* **Feature Scaling:** No global feature scaling was applied to the raw data prior to cross-validation or splitting. Instead, `StandardScaler` was encapsulated inside Scikit-learn `Pipeline` objects for scale-sensitive models, ensuring mean and variance statistics were calculated strictly on training folds.

---

## 7. Train / Test Methodology & Leakage Prevention
* **Splitting Strategy:** 80% Training (120 samples) and 20% Testing (30 samples) using Stratified Shuffle Split (`random_state=42`).
* **Class Stratification:**
  - **Training Set (120 samples):** 40 Setosa, 40 Versicolor, 40 Virginica
  - **Test Set (30 samples):** 10 Setosa, 10 Versicolor, 10 Virginica
* **Strict Test Isolation:** The 30 test samples were set aside before any model exploration or cross-validation. $\text{Index}(X_{\text{train}}) \cap \text{Index}(X_{\text{test}}) = \emptyset$. Zero data leakage occurred.

---

## 8. Candidate Classification Models
Four diverse candidate architectures were evaluated exclusively on the 120-sample training dataset:
1. **Logistic Regression:** Linear classifier paired with `StandardScaler` inside a `Pipeline` (`max_iter=200`, `random_state=42`).
2. **K-Nearest Neighbors (KNN):** Distance-based non-parametric classifier ($k=5$) with `StandardScaler` inside a `Pipeline`.
3. **Decision Tree Classifier:** Single decision tree without scaling (`random_state=42`).
4. **Random Forest Classifier:** Ensemble of 100 decorrelated decision trees (`n_estimators=100`, `random_state=42`).

---

## 9. Cross-Validation Results (Training Set Only)
Cross-validation was conducted strictly across the 120 training samples using **5-Fold Stratified Cross-Validation (`shuffle=True`, `random_state=42`)**:

| Candidate Model | Preprocessing Pipeline | Mean 5-Fold CV Accuracy | Standard Deviation | Fold Scores |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | `StandardScaler` + `LogisticRegression` | **95.83%** | $\pm 0.0264$ | `[0.958, 1.000, 0.958, 0.958, 0.917]` |
| **K-Nearest Neighbors** | `StandardScaler` + `KNeighborsClassifier` | **95.83%** | $\pm 0.0264$ | `[0.958, 1.000, 0.958, 0.917, 0.958]` |
| **Decision Tree** | Decision Tree (Raw Features) | **95.00%** | $\pm 0.0167$ | `[0.958, 0.958, 0.958, 0.958, 0.917]` |
| **Random Forest** | Random Forest (Raw Features) | **95.00%** | $\pm 0.0312$ | `[0.958, 1.000, 0.958, 0.917, 0.917]` |

---

## 10. Selected Model & Justification
* **Selected Model:** **Logistic Regression Pipeline** (`StandardScaler` $\rightarrow$ `LogisticRegression`).
* **Selection Reason:** It tied for the highest cross-validation accuracy (**95.83%**) on the training dataset. Logistic Regression was selected over KNN due to its deterministic linear decision boundaries, rapid inference, and well-calibrated probabilistic class outputs.
* **Important Contextual Clarification:** This selection indicates superior performance among the four evaluated candidate models under this specific 5-fold cross-validation setup on this training set; it is not asserted to be universally optimal for all possible classification problems.

---

## 11. Final Test-Set Results (Unseen 30 Samples)
The selected Logistic Regression pipeline was trained once on the complete 120-sample training dataset and evaluated once on the untouched 30-sample test set:

* **Final Test Accuracy:** **93.33%** (28/30 correctly classified)
* **Macro Precision:** **0.9333**
* **Macro Recall:** **0.9333**
* **Macro F1-Score:** **0.9333**
* **Weighted Precision:** **0.9333**
* **Weighted Recall:** **0.9333**
* **Weighted F1-Score:** **0.9333**

### Per-Class Test Performance
| Class (Species) | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| *Iris setosa* | 1.0000 | 1.0000 | 1.0000 | 10 |
| *Iris versicolor* | 0.9000 | 0.9000 | 0.9000 | 10 |
| *Iris virginica* | 0.9000 | 0.9000 | 0.9000 | 10 |
| **Macro Average** | **0.9333** | **0.9333** | **0.9333** | **30** |
| **Weighted Average** | **0.9333** | **0.9333** | **0.9333** | **30** |

---

## 12. Confusion Matrix Breakdown & Interpretation
```text
                  Predicted Setosa   Predicted Versicolor   Predicted Virginica
True Setosa               10                   0                      0
True Versicolor            0                   9                      1
True Virginica             0                   1                      9
```
* **Setosa:** 10/10 samples identified correctly (100% precision & recall). Petal length and width provide distinct linear separation from the other two classes.
* **Versicolor:** 9/10 correctly identified, 1 sample predicted as Virginica.
* **Virginica:** 9/10 correctly identified, 1 sample predicted as Versicolor.
* **Botanical Rationale:** Versicolor and Virginica share overlapping measurement ranges along their boundary (particularly petal lengths between 4.8–5.2 cm and petal widths between 1.5–1.8 cm), making slight boundary confusion expected in finite botanical samples.

---

## 13. Example Prediction Demonstration
Using the dedicated prediction module (`src/predict.py`) and standalone runner (`src/prediction_demo.py`), sample inputs produce calibrated outputs:

| Archetype Sample | Inputs (`sl, sw, pl, pw` in cm) | Predicted Species | Setosa Prob | Versicolor Prob | Virginica Prob |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Morphologically Setosa | `5.1, 3.5, 1.4, 0.2` | **setosa** | **98.08%** | 1.92% | 0.00% |
| Morphologically Versicolor | `6.0, 2.7, 5.1, 1.6` | **virginica** | 0.24% | 47.08% | **52.68%** |
| Morphologically Virginica | `6.7, 3.3, 5.7, 2.5` | **virginica** | 0.00% | 1.23% | **98.77%** |

*(Notice that the borderline Versicolor specimen at index 106 has petal length 5.1 cm and width 1.6 cm, which places it slightly across the linear boundary towards Virginica with a near-even 47% vs 53% posterior probability, transparently demonstrating model calibration).*

---

## 14. Visualizations
The project generates 5 professional visual artifacts saved under [`reports/figures/`](figures/):
1. **Class Distribution:** [`reports/figures/class_distribution.png`](figures/class_distribution.png) — Confirms balanced 50/50/50 class frequencies.
2. **Feature Distributions:** [`reports/figures/feature_distributions.png`](figures/feature_distributions.png) — 2x2 boxplots comparing measurements across species.
3. **Pairwise Feature Relationships:** [`reports/figures/pairwise_feature_relationships.png`](figures/pairwise_feature_relationships.png) — Scatter matrix highlighting petal separability.
4. **Model Comparison:** [`reports/figures/model_comparison.png`](figures/model_comparison.png) — 5-fold cross-validation accuracy bar chart with error bars.
5. **Confusion Matrix Heatmap:** [`reports/figures/confusion_matrix.png`](figures/confusion_matrix.png) — Annotated test set confusion matrix.

---

## 15. Limitations
* **Dataset Scale:** 150 samples collected from a single geographical region (Gaspé Peninsula, Canada) provide a classic benchmark but represent limited genetic and ecological variation.
* **Generalizability:** Models trained on this dataset should not be assumed to generalize without retraining to wild flower populations subject to varying soil, altitude, or climate conditions.
* **Distinction of CV vs Test:** The 5-fold cross-validation accuracy of **95.83%** reflects training partition stability, while the single test set accuracy of **93.33%** reflects out-of-sample generalization. Neither metric implies 100% real-world perfection.

---

## 16. Conclusion
The CodeAlpha Task 1 Iris Flower Classification project has been implemented following data science engineering standards:
* **Zero Data Leakage:** Preprocessing and cross-validation were strictly partitioned.
* **Statistically Sound Model Selection:** Logistic Regression with standardization emerged as the most reliable candidate.
* **Strong Test Generalization:** Achieved 93.33% accuracy on unseen test data.
* **Production-Ready Artifacts:** Includes reusable prediction API, comprehensive test coverage (15+ unit tests), and reproducible figures.
