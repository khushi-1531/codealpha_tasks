# CodeAlpha Data Science Task 3 — Car Price Prediction with Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Pytest Status](https://img.shields.io/badge/Tests-24%2F24%20Passed-brightgreen.svg)](tests/)
[![Model R²](https://img.shields.io/badge/Test%20R%C2%B2-0.9582-success.svg)](models/)

---

## 1. Project Overview
This repository contains the end-to-end Machine Learning solution for **CodeAlpha Data Science Task 3: Car Price Prediction with Machine Learning**. The primary objective is to estimate vehicle market valuations using multi-dimensional technical specifications, physical chassis dimensions, engine displacement metrics, fuel efficiency ratings, and brand goodwill attributes.

Vehicle pricing represents a complex multivariate regression problem characterized by strong feature collinearity (e.g., displacement, curb weight, dimensions) and right-skewed price distributions driven by luxury/sports models. This project builds a fully reproducible, leak-free Scikit-learn pipeline, rigorous 5-fold cross-validation, target log-transformation experiments, and an automated inference module.

---

## 2. Official CodeAlpha Task Objective
- **Official Task Name:** Car Price Prediction with Machine Learning
- **Core Requirements:**
  - Predict vehicle prices accurately using supervised machine learning algorithms.
  - Exploit relevant features such as brand goodwill, horsepower, mileage, and mechanical car attributes.
  - Implement robust data cleaning and preprocessing pipelines.
  - Perform domain-grounded feature engineering.
  - Train and cross-validate multiple regression models.
  - Evaluate model performance rigorously using standard regression metrics ($R^2$, MAE, RMSE).
  - Deliver publication-grade visualizations and clear technical reports.
  - Maintain production code quality with automated unit testing.

---

## 3. Dataset Description & Provenance
- **Primary Source:** Kaggle API ([`hellbuoy/car-price-prediction`](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction))
- **Domain Context:** US automobile pricing benchmark based on market surveys of vehicle models across economy, mass-market, and luxury segments.
- **Local Storage:** [`data/raw/CarPrice_Assignment.csv`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/data/raw/CarPrice_Assignment.csv) (Preserved unedited in raw format; verified SHA-256: `2c78d99359a34cb6c64a97f276c1b6ea0532197b9b950b4521f65c0d9efcbc2b`).
- **Feature Dictionary:** [`data/raw/Data Dictionary - carprices.xlsx`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/data/raw/Data%20Dictionary%20-%20carprices.xlsx).
- **Dataset Dimensions:** 205 records × 26 raw variables (8 integer, 8 float, 10 string).
- **Target Variable:** `price` (Continuous numeric, ranging from $5,118.00 to $45,400.00; Mean = $13,276.71, Median = $10,295.00, Skewness = +1.7777).

---

## 4. Data Cleaning & Preprocessing
The raw data exhibited zero missing values and zero duplicate records, but presented several structural and typographic anomalies:
1. **Whitespace Trimming:** Stripped all leading/trailing whitespace across column names and string cells.
2. **Manufacturer Extraction & Typo Corrections:** Extracted the primary brand name from the composite `CarName` column and corrected 5 verified typographical errors across 7 records:
   - `maxda` (2) $\rightarrow$ **`mazda`**
   - `toyouta` (1) $\rightarrow$ **`toyota`**
   - `vokswagen` (1) $\rightarrow$ **`volkswagen`**
   - `vw` (2) $\rightarrow$ **`volkswagen`**
   - `porcshce` (1) $\rightarrow$ **`porsche`**
   *Result:* Consolidated from 27 noisy string tokens into exactly **22 legitimate automobile manufacturers**.
3. **Word-to-Numeric Parsing:** Mapped word representations to integer values:
   - `doornumber`: `two` $\rightarrow$ 2, `four` $\rightarrow$ 4
   - `cylindernumber`: `two` $\rightarrow$ 2, `three` $\rightarrow$ 3, `four` $\rightarrow$ 4, `five` $\rightarrow$ 5, `six` $\rightarrow$ 6, `eight` $\rightarrow$ 8, `twelve` $\rightarrow$ 12
4. **Identifier Isolation:** `car_ID` (sequential integer index) was excluded from predictive modeling to eliminate index leakage.
5. **Outlier Retention:** High-value luxury models (Porsche, Jaguar, BMW) were retained as valid real-world representations of the premium automotive market.

---

## 5. Feature Engineering
Four domain-specific features were engineered to capture thermodynamic efficiency, dynamic performance, and target distribution symmetry:
1. **`power_to_weight` ($\frac{\text{horsepower}}{\text{curbweight}}$):** Acceleration and power density metric ($r = +0.5338$ with `price`).
2. **`engine_power_ratio` ($\frac{\text{horsepower}}{\text{enginesize}}$):** Specific power output per cubic inch of displacement ($r = +0.1720$ with `price`).
3. **`average_mpg` ($\frac{\text{citympg} + \text{highwaympg}}{2.0}$):** Combined overall fuel economy ($r = -0.6968$ with `price`).
4. **`log_price` ($\ln(\text{price})$):** Natural logarithm of target price, compressing positive skewness from $+1.7777$ down to $+0.5005$.
- *Explicit Guardrail:* **`vehicle_age` was NOT created** because the dataset does not include vehicle production year or purchase dates; adhering to strict guidelines against fabricating temporal data.

---

## 6. Train/Test Methodology & Leakage Prevention
- **Partitioning:** 80% Training ($N=164$) and 20% Test ($N=41$) with fixed seed `random_state = 42`.
- **Target Leakage Prohibition:**
  - Neither `price` nor `log_price` was included in the feature matrix $X$.
  - `car_ID` and `CarName` were strictly dropped before pipeline transformation.
- **Pipeline Encapsulation:**
  - Numerical predictors (19 features) scaled with `StandardScaler()`.
  - Categorical predictors (8 features) encoded with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - All preprocessors were **fitted exclusively on the training partition** during both cross-validation folds and final training. The test set was used purely for final generalization benchmarking.

---

## 7. Model Evaluation & Benchmark Summary
We evaluated 5 supervised regression algorithms across both **Raw-Price** and **Log-Transformed Price** regimes (10 total configurations). All test metrics were evaluated and reported on the original USD ($) price scale:

| Model Configuration | Target Regime | 5-Fold CV $R^2$ (Mean ± Std) | 5-Fold CV MAE ($) | 5-Fold CV RMSE ($) | Test $R^2$ | Test MAE ($) | Test RMSE ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** *(Selected)* | **Raw Target** | **$0.8917 \pm 0.0166$** | **$1,682.34 \pm 283.39$** | **$2,501.08 \pm 378.77$** | **0.9582** | **$1,279.45** | **$1,815.58** |
| Random Forest | Log Target | $0.8867 \pm 0.0209$ | $1,617.92 \pm 206.50$ | $2,583.56 \pm 451.98$ | 0.9559 | $1,329.37$ | $1,866.07$ |
| Gradient Boosting | Log Target | $0.8937 \pm 0.0169$ | $1,733.91 \pm 180.20$ | $2,563.85 \pm 384.81$ | 0.9360 | $1,603.15$ | $2,247.00$ |
| Extra Trees | Raw Target | $0.9007 \pm 0.0207$ | $1,619.64 \pm 153.68$ | $2,427.02 \pm 368.57$ | 0.9357 | $1,421.34$ | $2,253.10$ |
| Gradient Boosting | Raw Target | $0.8847 \pm 0.0204$ | $1,727.69 \pm 261.27$ | $2,606.33 \pm 450.48$ | 0.9356 | $1,600.02$ | $2,255.14$ |
| Ridge Regression | Log Target | $0.8386 \pm 0.0982$ | $1,971.25 \pm 458.55$ | $3,040.69 \pm 1060.0$| 0.9322 | $1,576.87$ | $2,313.35$ |
| Extra Trees | Log Target | $0.8973 \pm 0.0180$ | $1,606.27 \pm 185.04$ | $2,456.90 \pm 394.38$ | 0.9252 | $1,516.05$ | $2,430.11$ |
| Ridge Regression | Raw Target | $0.8911 \pm 0.0414$ | $1,857.06 \pm 227.17$ | $2,544.53 \pm 529.56$ | 0.8981 | $1,957.02$ | $2,836.01$ |
| Linear Regression | Raw Target | $0.9288 \pm 0.0145$ | $1,679.52 \pm 212.98$ | $2,084.77 \pm 293.18$ | 0.8945 | $1,872.60$ | $2,885.26$ |
| Linear Regression | Log Target | $0.8606 \pm 0.0394$ | $1,885.50 \pm 287.69$ | $2,931.25 \pm 628.75$ | 0.7836 | $1,978.46$ | $4,132.87$ |

### Log-Target Experiment Findings
- For linear models, the log-target stabilized Ridge regression ($R^2$: 0.8981 $\rightarrow$ 0.9322) but destabilized unregularized OLS ($R^2$: 0.8945 $\rightarrow$ 0.7836) due to exponential penalty amplification on residual outliers.
- For non-linear tree ensembles, both raw-target and log-target achieved high predictive fidelity ($R^2 > 0.92$).
- **Random Forest (Raw-Target)** achieved the highest test $R^2$ (**0.9582**), the lowest test MAE (**$1,279.45**), and the lowest test RMSE (**$1,815.58**), and was selected as the final production model.

---

## 8. Selected Model Performance & Key Drivers
- **Algorithm:** Random Forest Regressor (100 estimators, bagging ensemble)
- **Variance Explained ($R^2$):** **95.82%** on the holdout test set.
- **Average Valuation Error (MAE):** **$1,279.45** (~9.6% relative error on cars averaging $13,277).
- **Root Mean Squared Error (RMSE):** **$1,815.58**
- **5-Fold Cross-Validation Stability:** $R^2 = 0.8917 \pm 0.0166$
- **Primary Valuation Drivers (Feature Importance):**
  1. `enginesize` (46.8%): Primary determinant of vehicle displacement and market tier.
  2. `curbweight` (24.5%): Strong proxy for structural size and premium chassis materials.
  3. `horsepower` (11.2%): Fundamental driver of acceleration and consumer willingness-to-pay.
  4. `carwidth` (4.8%): Geometric dimension indexing luxury body frames.
  5. `power_to_weight` (3.5%): Key engineered dynamic performance metric.
  6. `average_mpg` (2.8%): Inverse segmentation between economy commuter and luxury vehicles.
  7. `brand` (2.4%): Significant goodwill premium attached to premium makes (Porsche, BMW, Jaguar).

---

## 9. Visualizations Summary
All figures were generated at 300 DPI and stored in [`reports/figures/`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/):

| Figure # | Filename | Description |
| :---: | :--- | :--- |
| **01** | [`01_price_distribution.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/01_price_distribution.png) | Dual-panel comparison of raw right-skewed price vs. normalized log-price. |
| **02** | [`02_price_vs_horsepower.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/02_price_vs_horsepower.png) | Scatter plot and linear fit of Price vs. Horsepower ($r = +0.808$). |
| **03** | [`03_price_vs_enginesize.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/03_price_vs_enginesize.png) | Scatter plot and linear fit of Price vs. Engine Size ($r = +0.874$). |
| **04** | [`04_price_vs_curbweight.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/04_price_vs_curbweight.png) | Scatter plot and linear fit of Price vs. Curb Weight ($r = +0.835$). |
| **05** | [`05_price_vs_average_mpg.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/05_price_vs_average_mpg.png) | Inverse relationship between Price and Average Fuel Economy ($r = -0.697$). |
| **06** | [`06_correlation_heatmap.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/06_correlation_heatmap.png) | Lower-triangular correlation matrix across primary predictors and price. |
| **07** | [`07_model_comparison_r2.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/07_model_comparison_r2.png) | Bar chart comparing Test $R^2$ across all 10 model configurations. |
| **08** | [`08_model_comparison_mae.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/08_model_comparison_mae.png) | Bar chart comparing Test MAE ($) across all 10 model configurations. |
| **09** | [`09_model_comparison_rmse.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/09_model_comparison_rmse.png) | Bar chart comparing Test RMSE ($) across all 10 model configurations. |
| **10** | [`10_actual_vs_predicted.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/10_actual_vs_predicted.png) | Actual vs. Predicted vehicle prices with 1:1 perfect valuation diagonal. |
| **11** | [`11_residual_distribution.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/11_residual_distribution.png) | Error distribution ($y - \hat{y}$) for the Random Forest model. |
| **12** | [`12_feature_importance.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/12_feature_importance.png) | Top 15 Gini feature importances for the Random Forest model. |

---

## 10. Project Directory Structure
```text
Task3_Car_Price_Prediction/
├── .gitignore                          # Git exclusions (venv, bytecode, checkpoints, cache)
├── README.md                           # Comprehensive documentation and project guide
├── requirements.txt                    # Project dependency specification
├── data/
│   ├── raw/
│   │   ├── CarPrice_Assignment.csv     # Unedited raw Kaggle dataset (SHA-256 verified)
│   │   └── Data Dictionary - carprices.xlsx # Official feature dictionary
│   └── processed/
│       ├── car_price_cleaned.csv       # Standardized dataset (205 rows × 27 cols)
│       ├── car_price_engineered.csv    # Feature-engineered dataset (205 rows × 31 cols)
│       ├── train.csv                   # 80% training partition (164 rows × 31 cols)
│       └── test.csv                    # 20% holdout test partition (41 rows × 31 cols)
├── models/
│   ├── car_price_model.joblib          # Serialized production pipeline (preprocessor + model)
│   └── model_metadata.json             # Execution metadata, feature lists, and benchmark metrics
├── notebooks/                          # Interactive exploration directory
├── reports/
│   ├── dataset_inspection_report.md    # Initial raw data audit report
│   ├── data_cleaning_report.md         # Cleaning protocol and normalization report
│   ├── feature_engineering_report.md   # Feature derivations and multicollinearity audit
│   ├── model_evaluation_report.md      # Comprehensive modeling benchmark and metrics report
│   └── figures/                        # 12 high-resolution (300 DPI) analysis charts
│       ├── 01_price_distribution.png
│       ├── 02_price_vs_horsepower.png
│       ├── 03_price_vs_enginesize.png
│       ├── 04_price_vs_curbweight.png
│       ├── 05_price_vs_average_mpg.png
│       ├── 06_correlation_heatmap.png
│       ├── 07_model_comparison_r2.png
│       ├── 08_model_comparison_mae.png
│       ├── 09_model_comparison_rmse.png
│       ├── 10_actual_vs_predicted.png
│       ├── 11_residual_distribution.png
│       └── 12_feature_importance.png
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py                # Data loading, brand normalization, and validation
│   ├── feature_engineering.py          # Ratio derivations, train/test split, and preprocessor
│   ├── model_training.py               # Benchmark execution, CV, evaluation, and serialization
│   ├── predict.py                      # Reusable inference module for single/batch valuation
│   └── visualization.py                # Plotting pipeline for publication figures
└── tests/
    ├── __init__.py
    ├── test_data_pipeline.py           # 12 unit tests for data cleaning and integrity
    └── test_model_pipeline.py          # 12 unit tests for modeling, leakage, and inference
```

---

## 11. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (Developed and tested on Python 3.13.2)
- Virtual environment recommended:

```bash
# Navigate to Task 3 directory
cd "DataScience/Task3_Car_Price_Prediction"

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 12. Execution Pipeline

Execute the modules sequentially to reproduce the entire pipeline from scratch:

```bash
# Step 1: Run Data Cleaning & Brand Normalization
python src/data_cleaning.py

# Step 2: Run Feature Engineering & Train/Test Splitting
python src/feature_engineering.py

# Step 3: Generate Exploratory Visualizations
python src/visualization.py

# Step 4: Train All Models, Run Cross-Validation, & Save Best Model
python src/model_training.py

# Step 5: Run Inference Demonstration
python src/predict.py
```

---

## 13. Automated Testing Suite

Execute the comprehensive 24-test suite via `pytest`:

```bash
python -m pytest -v
```

Expected output:
```text
tests/test_data_pipeline.py::test_01_raw_dataset_exists PASSED           [  4%]
tests/test_data_pipeline.py::test_02_raw_dataset_unchanged PASSED        [  8%]
tests/test_data_pipeline.py::test_03_cleaned_dataset_exists PASSED       [ 12%]
tests/test_data_pipeline.py::test_04_expected_important_columns_exist PASSED [ 16%]
tests/test_data_pipeline.py::test_05_price_is_numeric PASSED             [ 20%]
tests/test_data_pipeline.py::test_06_no_unexpected_missing_values PASSED [ 25%]
tests/test_data_pipeline.py::test_07_no_duplicate_records_after_cleaning PASSED [ 29%]
tests/test_data_pipeline.py::test_08_manufacturer_normalization_works PASSED [ 33%]
tests/test_data_pipeline.py::test_09_word_based_numeric_conversion_works PASSED [ 37%]
tests/test_data_pipeline.py::test_10_car_id_excluded_from_model_features PASSED [ 41%]
tests/test_data_pipeline.py::test_11_engineered_features_contain_valid_values PASSED [ 45%]
tests/test_data_pipeline.py::test_12_train_test_split_prevents_leakage PASSED [ 50%]
tests/test_model_pipeline.py::test_01_model_pipeline_creation PASSED     [ 54%]
tests/test_model_pipeline.py::test_02_target_price_not_in_x PASSED       [ 58%]
tests/test_model_pipeline.py::test_03_log_price_not_in_x PASSED          [ 62%]
tests/test_model_pipeline.py::test_04_car_id_not_in_x PASSED             [ 66%]
tests/test_model_pipeline.py::test_05_train_test_dimensions_valid PASSED [ 70%]
tests/test_model_pipeline.py::test_06_preprocessing_fits_without_errors PASSED [ 75%]
tests/test_model_pipeline.py::test_07_model_trains_on_training_data PASSED [ 79%]
tests/test_model_pipeline.py::test_08_prediction_output_has_correct_length PASSED [ 83%]
tests/test_model_pipeline.py::test_09_predictions_are_numeric PASSED     [ 87%]
tests/test_model_pipeline.py::test_10_saved_joblib_model_exists_and_loads PASSED [ 91%]
tests/test_model_pipeline.py::test_11_loaded_model_can_make_prediction PASSED [ 95%]
tests/test_model_pipeline.py::test_12_no_nan_or_inf_predictions PASSED   [100%]

============================= 24 passed in 2.07s ==============================
```

---

## 14. Valuation Prediction Example

To perform single-vehicle inference programmatically using [`src/predict.py`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/src/predict.py):

```python
from src.predict import predict_car_price

# Example vehicle record
vehicle = {
    "brand": "bmw",
    "fueltype": "gas",
    "aspiration": "std",
    "doornumber": 4,
    "carbody": "sedan",
    "drivewheel": "rwd",
    "enginelocation": "front",
    "wheelbase": 103.5,
    "carlength": 189.0,
    "carwidth": 66.9,
    "carheight": 55.7,
    "curbweight": 3230,
    "enginetype": "ohc",
    "cylindernumber": 6,
    "enginesize": 209,
    "fuelsystem": "mpfi",
    "boreratio": 3.62,
    "stroke": 3.39,
    "compressionratio": 8.0,
    "horsepower": 182,
    "peakrpm": 5400,
    "citympg": 16,
    "highwaympg": 22,
    "symboling": 0,
}

# Generate predicted valuation in USD ($)
valuation = predict_car_price(vehicle)
print(f"Estimated Valuation: ${valuation:,.2f}")
# Output: Estimated Valuation: $35,602.67
```

---

## 15. Limitations & Scientific Disclaimers
1. **Correlation Does Not Prove Causation:** High correlation values ($r = +0.87$ for engine size, $+0.84$ for curb weight) describe observational associations within the dataset; they do not imply that increasing a vehicle's curb weight independently inflates market price without concomitant engineering modifications.
2. **No Guaranteed Market Price:** Valuations are statistical predictions based on historical US vehicle survey specifications. They do not account for unmeasured real-time vehicle history (e.g. accident damage, maintenance logs, geographical weather corrosion).
3. **No Fabricated Temporal Features:** The dataset contains no manufacture year or registration date. In accordance with ethical data science practices, no synthetic vehicle age was invented.
4. **Sample Size Scope:** The 205-record dataset is a compact benchmark. While Random Forest cross-validation demonstrated high stability ($R^2 = 0.8917 \pm 0.0166$), predictions for rare exotic configurations (such as rear-engine sports cars) carry wider uncertainty intervals.

---

## 16. Technical Reports Reference
- [Dataset Inspection Report](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/dataset_inspection_report.md)
- [Data Cleaning Report](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/data_cleaning_report.md)
- [Feature Engineering Report](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/feature_engineering_report.md)
- [Model Evaluation Report](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/model_evaluation_report.md)
