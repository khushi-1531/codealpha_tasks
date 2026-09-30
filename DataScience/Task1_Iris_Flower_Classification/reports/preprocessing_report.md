# Iris Data Preprocessing & Train/Test Split Report
**Project:** CodeAlpha Data Science Internship — Task 1 (Iris Flower Classification)  
**Pipeline Status:** PASSED (Zero data leakage confirmed)

---
## 1. Feature & Target Specification
- **Predictor Features (X):** `sepal_length, sepal_width, petal_length, petal_width` (4 continuous numerical measurements in cm)
- **Target Label (y):** `species` (Categorical Iris species)
- **Feature Scaling:** No scaling applied at this stage (original measurement scale retained for model selection)

## 2. Target Encoding & Mapping
Deterministic integer encoding applied for machine learning compatibility while preserving explicit reverse mapping:

| Original Class Label | Encoded Target Integer |
| :--- | :--- |
| `Iris setosa` | `0` |
| `Iris versicolor` | `1` |
| `Iris virginica` | `2` |

## 3. Duplicate Handling Decision
- **Duplicate Rows Detected in Raw Data:** 1
- **Handling Decision:** Retained the canonical duplicate row (sample at index 142). In the Fisher/Anderson Iris benchmark, this represents an authentic, distinct biological flower specimen with coincidentally identical measurements, not an erroneous recording.
- **Raw Data Immutability:** `data/raw/iris.csv` remains strictly untouched and preserved.

## 4. Train / Test Partitioning
- **Splitting Strategy:** Stratified Shuffle Split (`stratify=y`, `random_state=42`)
- **Partition Ratio:** 80% Training (120 samples) / 20% Testing (30 samples)
- **Data Leakage Check:** PASSED (Index intersection = 0, test set strictly isolated)

### Class Balance Across Partitions
| Species | Encoded ID | Training Count (80%) | Testing Count (20%) | Total Count |
| :--- | :--- | :--- | :--- | :--- |
| *Iris setosa* | `0` | 40 (33.3%) | 10 (33.3%) | 50 (33.3%) |
| *Iris versicolor* | `1` | 40 (33.3%) | 10 (33.3%) | 50 (33.3%) |
| *Iris virginica* | `2` | 40 (33.3%) | 10 (33.3%) | 50 (33.3%) |

## 5. Pipeline Validation Summary
- [x] Feature columns exist and verified: `True`
- [x] Target column exists and verified: `True`
- [x] Zero missing values in partitions: `True`
- [x] 80/20 train/test split size accurate (120/30): `True`
- [x] All 3 species represented in training set: `True`
- [x] All 3 species represented in test set: `True`
- [x] Zero index or feature leakage to test set: `True`
- [x] No machine learning model trained yet: `True`
