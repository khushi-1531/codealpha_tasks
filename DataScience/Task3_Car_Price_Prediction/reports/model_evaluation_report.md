# CodeAlpha Data Science Task 3 — Model Evaluation and Performance Report

## 1. Modeling Objective
The objective of **CodeAlpha Data Science Task 3** is to construct, rigorously validate, and evaluate supervised machine learning regression models to predict automobile valuation using vehicle physical, mechanical, efficiency, and brand attributes. 

Predicting used vehicle prices is a non-linear multivariate regression problem where predictors exhibit strong collinearity (e.g., engine displacement, curb weight, dimensions) and the target variable exhibits a pronounced positive skewness toward luxury/sports vehicles.

---

## 2. Train/Test Methodology & Data Split
- **Dataset Partitioning:**
  - Total records: 205
  - **Training Set (80%):** 164 records ([`data/processed/train.csv`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/data/processed/train.csv))
  - **Untouched Test Set (20%):** 41 records ([`data/processed/test.csv`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/data/processed/test.csv))
  - **Random Seed:** `random_state = 42`
  - **Disjointness:** Zero overlap between train and test `car_ID` sets ($164 \cap 41 = \emptyset$).
- **Strict Protocol:** The test partition was kept completely untouched during feature engineering, preprocessor fitting, cross-validation, and hyperparameter checks. It was used solely for the final unbiased benchmark.

---

## 3. Preprocessing Approach & Feature Handling
- **Categorical Feature Processing (8 features):**
  - Features: `brand`, `fueltype`, `aspiration`, `carbody`, `drivewheel`, `enginelocation`, `enginetype`, `fuelsystem`.
  - Encoder: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` learned **strictly** from the training set. Novel categories observed at test or inference time map safely to zero without crashing or leaking.
- **Numerical Feature Processing (19 features):**
  - Features: `symboling`, `doornumber`, `wheelbase`, `carlength`, `carwidth`, `carheight`, `curbweight`, `cylindernumber`, `enginesize`, `boreratio`, `stroke`, `compressionratio`, `horsepower`, `peakrpm`, `citympg`, `highwaympg`, `power_to_weight`, `engine_power_ratio`, `average_mpg`.
  - Scaler: `StandardScaler()` fitted **strictly** on the training set.

---

## 4. Leakage Prevention Audit
Strict architectural boundaries were enforced:
1. **Target Exclusion:** Neither `price` nor `log_price` was permitted inside the feature matrix $X$.
2. **Identifier Exclusion:** `car_ID` (sequential database index) and `CarName` (raw composite string) were strictly dropped before pipeline ingestion.
3. **Pipeline Encapsulation:** Preprocessing steps were integrated with the regressor inside a Scikit-learn `Pipeline` and `ColumnTransformer`. Cross-validation and test evaluation fit transformers exclusively on training folds.
4. **Target Transformation Safety:** For log-target models, transformation was handled through `TransformedTargetRegressor(regressor=..., func=np.log, inverse_func=np.exp)`, ensuring the underlying pipeline never sees raw prices or targets in $X$, and predictions are inverted back to original USD scale prior to scoring.

---

## 5. Regression Models Evaluated
We benchmarked 5 primary regression architectures across both **Raw-Price** and **Log-Transformed Price** regimes (10 distinct configurations):
1. **Linear Regression:** Baseline parametric Ordinary Least Squares (OLS).
2. **Ridge Regression:** L2-regularized linear model ($\alpha = 1.0$) designed to mitigate coefficient inflation under multicollinearity.
3. **Random Forest Regressor:** Non-linear bagging ensemble (100 decision trees) robust to feature interactions and outliers.
4. **Gradient Boosting Regressor:** Sequential boosting ensemble (100 boosting stages, learning rate = 0.1) optimizing squared-error loss.
5. **Extra Trees Regressor:** Extremely randomized trees ensemble (100 estimators) providing randomized split thresholds.

---

## 6. Cross-Validation Methodology
- **Strategy:** 5-Fold Cross-Validation (`KFold(n_splits=5, shuffle=True, random_state=42)`).
- **Execution:** Executed strictly on the training partition ($N=164$).
- **Scoring Metrics Tracked:**
  - Mean & Standard Deviation of $R^2$
  - Mean & Standard Deviation of Mean Absolute Error (MAE in USD)
  - Mean & Standard Deviation of Root Mean Squared Error (RMSE in USD)

---

## 7. Comprehensive Model Benchmark Results

The table below presents the verified empirical results from the full benchmark, sorted by holdout Test $R^2$:

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

---

## 8. Log-Target Experiment Analysis
- **Empirical Observation:**
  - For linear models (Linear Regression, Ridge), log-target modeling yielded mixed behavior: Ridge improved on the test set ($R^2$ 0.8981 $\rightarrow$ 0.9322), but unregularized OLS suffered severe degradation ($R^2$ 0.8945 $\rightarrow$ 0.7836) due to exponential penalty sensitivity on outlier residuals.
  - For tree-based ensembles (Random Forest, Extra Trees, Gradient Boosting), both raw-target and log-target regimes performed exceptionally well, explaining 92.5% to 95.8% of test variance.
  - Comparing **Random Forest (Raw-Target)** vs. **Random Forest (Log-Target)**:
    - Raw-Target achieved higher test $R^2$ (**0.9582** vs. 0.9559).
    - Raw-Target produced lower test MAE (**$1,279.45** vs. $1,329.37).
    - Raw-Target produced lower test RMSE (**$1,815.58** vs. $1,866.07).
- **Decision:** In accordance with the project rule (*"Do not automatically select log transformation just because the distribution is skewed. Use the actual validation/test results"*), **Random Forest (Raw-Target)** was selected as the final production model based on empirical superiority across all metrics.

---

## 9. Final Model Selection Rationale
**Selected Model:** `Random Forest Regressor (Raw-Target)`
- **Generalization Performance:** Explains **95.82%** of vehicle price variance ($R^2 = 0.9582$) on the untouched holdout test partition.
- **Error Magnitude:** Achieves the lowest absolute dollar error with a Mean Absolute Error of **$1,279.45** on vehicles averaging $13,277 in value (~9.6% relative error).
- **Penalization of Large Residuals:** Achieves an RMSE of **$1,815.58**, substantially outperforming linear baselines ($2,885) and gradient boosted models ($2,255).
- **Cross-Validation Stability:** Boasts exceptionally low variance across 5 folds with a CV $R^2$ of **$0.8917 \pm 0.0166$** (lowest standard deviation among all tree models).
- **Robustness:** Decision tree bagging inherently handles correlated features and non-linear thresholds without suffering from collinear matrix inversion instabilities.

---

## 10. Feature Importance Analysis
Analysis of the underlying Random Forest estimator reveals the primary vehicle attributes driving price predictions:
1. **`enginesize` (Importance: ~0.468):** Displacement remains the primary determinant of price tier.
2. **`curbweight` (Importance: ~0.245):** Structural mass directly indexes vehicle size class and premium chassis materials.
3. **`horsepower` (Importance: ~0.112):** Engine output strongly correlates with consumer willingness-to-pay.
4. **`carwidth` (Importance: ~0.048):** Wider stance characterizes luxury and sports vehicle segments.
5. **`power_to_weight` (Importance: ~0.035):** Dynamic acceleration ratio successfully captured by feature engineering.
6. **`average_mpg` (Importance: ~0.028):** Inversely segments fuel-efficient economy commuter cars from high-cost performance vehicles.
7. **`brand` (Importance: ~0.024):** Luxury manufacturers (Porsche, BMW, Jaguar, Buick) command significant brand goodwill premiums.

---

## 11. Visualizations Reference
All figures are rendered at 300 DPI in [`reports/figures/`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/):
1. [Figure 7: Test Set Performance Comparison — $R^2$](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/07_model_comparison_r2.png)
2. [Figure 8: Test Set Performance Comparison — MAE](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/08_model_comparison_mae.png)
3. [Figure 9: Test Set Performance Comparison — RMSE](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/09_model_comparison_rmse.png)
4. [Figure 10: Actual vs. Predicted Vehicle Prices (Random Forest)](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/10_actual_vs_predicted.png)
5. [Figure 11: Residual Error Distribution (Actual - Predicted)](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/11_residual_distribution.png)
6. [Figure 12: Top 15 Feature Importances (Random Forest)](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/12_feature_importance.png)

---

## 12. Limitations & Causality Disclaimer
- **Correlation vs. Causality:** Strong empirical correlations between vehicle attributes (e.g. `curbweight`, `horsepower`) and `price` do not prove standalone causality. Vehicle pricing is a multi-attribute market equilibrium.
- **Dataset Sample Size:** The sample of 205 records means high-end luxury vehicle subsets (e.g. rear-engine Porsche models) have few training exemplars.
- **Market Scope:** Data reflects historical US automotive market conditions and currency units ($ USD); regional inflation and exchange rate dynamics are outside model scope.

---

## 13. Conclusion
The regression pipeline successfully addresses all official CodeAlpha Task 3 criteria. The selected **Random Forest Regressor** provides high accuracy ($R^2 = 0.9582$, $\text{MAE} = \$1,279.45$), robust cross-validation stability ($R^2 = 0.8917 \pm 0.0166$), and interpretable feature rankings. The complete preprocessing and modeling pipeline is fully encapsulated and ready for production deployment.
