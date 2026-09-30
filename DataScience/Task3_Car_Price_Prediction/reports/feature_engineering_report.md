# CodeAlpha Data Science Task 3 — Feature Engineering Report

## 1. Executive Summary
This report details the domain-grounded feature engineering and train/test dataset preparation performed for **CodeAlpha Task 3: Car Price Prediction with Machine Learning**.

Automotive valuation is heavily influenced by non-linear physical interactions (e.g. power relative to weight, engine displacement efficiency, overall fuel economy) and market perception (brand equity/goodwill). By deriving physically meaningful ratios and normalizing right-skewed price distributions, we enrich the feature space while eliminating risks of data leakage.

---

## 2. Engineered Features & Technical Rationale

| Feature Name | Formula / Derivation | Data Type | Physical / Domain Interpretation | Correlation with Price ($r$) | Correlation with $\log(\text{price})$ ($r$) |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **`power_to_weight`** | $\frac{\text{horsepower}}{\text{curbweight}}$ | `float64` | Power-to-weight ratio (bhp/lb). Fundamental metric of vehicle acceleration, agility, and performance tuning. | **+0.5338** | **+0.5393** |
| **`engine_power_ratio`** | $\frac{\text{horsepower}}{\text{enginesize}}$ | `float64` | Specific power output per cubic inch of displacement. Measures thermodynamic and mechanical engine tuning efficiency. | **+0.1720** | **+0.2617** |
| **`average_mpg`** | $\frac{\text{citympg} + \text{highwaympg}}{2.0}$ | `float64` | Combined fuel economy benchmark. High fuel economy characterizes economy commuter vehicles; low fuel economy characterizes high-displacement performance and luxury vehicles. | **-0.6968** | **-0.7791** |
| **`log_price`** | $\ln(\text{price})$ | `float64` | Natural logarithm of target price. Compresses long right-tail variance to yield an approximately Gaussian target distribution. | **+0.9580** (vs price) | **1.0000** |

---

## 3. Explicit Clarification on Temporal Features (`vehicle_age`)
- **Inspection Finding:** The dataset does not contain a vehicle manufacturing year, purchase date, or model release year column.
- **Rule Adherence:** In strict accordance with the project guidelines (*"Do NOT invent a vehicle age if the required year information is unavailable"*), **no artificial vehicle age or year feature was created**.
- Fabricating a synthetic year would inject noise, confound depreciation calculations, and violate rigorous data science practices.

---

## 4. Target Distribution Analysis & Log-Transformation Justification

### 4.1 Raw Target Distribution (`price`)
- **Min:** $5,118.00
- **25th Percentile:** $7,788.00
- **Median:** $10,295.00
- **Mean:** $13,276.71
- **75th Percentile:** $16,503.00
- **Max:** $45,400.00
- **Standard Deviation:** $7,988.85
- **Skewness:** **+1.7777** (Substantial positive/right skew)

### 4.2 Transformed Target Distribution (`log_price`)
- **Formula:** $\log\_price = \ln(price)$
- **Min:** 8.5405
- **Median:** 9.2394
- **Mean:** 9.3547
- **Max:** 10.7233
- **Standard Deviation:** 0.5049
- **Skewness:** **+0.5005** (Approaching symmetric / normal distribution)

### 4.3 Methodological Decision
1. In standard Ordinary Least Squares (OLS) and regularized linear regression, modeling raw skewed prices induces severe heteroscedasticity (larger prediction errors on higher-priced vehicles).
2. The log-transformation $\log(\text{price})$ stabilizes error variance and aligns with proportional economic pricing effects.
3. **Guardrail:** When evaluating downstream models in Step 4, predictions will be exponentiated back to the original USD currency scale ($\hat{y} = \exp(\widehat{\log\_y})$) so that all primary regression metrics ($R^2$, MAE, RMSE, MAPE) are reported and compared directly on actual dollar values.

---

## 5. Multicollinearity & Feature Correlation Analysis

### 5.1 Primary Predictor Correlations with Target

```text
Feature             r with price    r with log_price
curbweight             +0.8353          +0.8912
enginesize             +0.8741          +0.8320
horsepower             +0.8081          +0.8258
carwidth               +0.7593          +0.8121
carlength              +0.6829          +0.7764
wheelbase              +0.5778          +0.6343
boreratio              +0.5532          +0.5832
power_to_weight        +0.5338          +0.5393
average_mpg            -0.6968          -0.7791
highwaympg             -0.6976          -0.7752
citympg                -0.6858          -0.7716
```

### 5.2 Multicollinearity Observations
- `citympg` and `highwaympg` share an extremely high pairwise correlation of **+0.9712**. By introducing `average_mpg`, we synthesize overall fuel economy into a unified representation.
- Structural dimensions (`carlength`, `carwidth`, `wheelbase`) exhibit strong mutual correlation ($r > 0.80$) with `curbweight`. Heavier vehicles consistently feature larger dimensions.
- **Handling Strategy:**
  - For linear models (Ridge, Lasso), regularization will shrink redundant coefficients and mitigate variance inflation.
  - For tree-based models (Random Forest, Gradient Boosting), collinearity does not degrade predictive accuracy.

---

## 6. Train / Test Split & Leakage-Free Pipeline Architecture

### 6.1 Split Configuration
- **Total Records:** 205
- **Train Set (80%):** 164 records, saved at `data/processed/train.csv`
- **Test Set (20%):** 41 records, saved at `data/processed/test.csv`
- **Random Seed:** `random_state = 42` (Deterministic reproducibility)
- **Disjointness:** Zero overlap between train and test `car_ID` sets ($164 \cap 41 = \emptyset$).

### 6.2 Preprocessor Pipeline Architecture
To strictly eliminate data leakage, the feature transformation pipeline is encapsulated inside a Scikit-learn `ColumnTransformer`:

1. **Numerical Transformer (`num`):**
   - Applies `StandardScaler()` to all 19 numerical predictors (`horsepower`, `enginesize`, `curbweight`, `power_to_weight`, `average_mpg`, etc.).
   - Fitted **exclusively** on `X_train` ($N=164$). `X_test` is transformed using the training mean and standard deviation.
2. **Categorical Transformer (`cat`):**
   - Applies `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` across all 8 categorical features (`brand`, `fueltype`, `aspiration`, `carbody`, `drivewheel`, `enginelocation`, `enginetype`, `fuelsystem`).
   - Learns category mappings **strictly** from `X_train`. Any novel category appearing in test evaluation is gracefully mapped to all zeros without throwing errors or causing leakage.
3. **Excluded Columns:**
   - `car_ID` and `CarName` are dropped prior to transformation.
   - `price` and `log_price` are reserved exclusively as supervised regression targets.

---

## 7. Generated Visualizations Reference
All figures have been rendered at 300 DPI and stored in [`reports/figures/`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/):
1. [Figure 1: Vehicle Price Distribution & Log Transformation Normalization](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/01_price_distribution.png)
2. [Figure 2: Vehicle Price vs. Horsepower](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/02_price_vs_horsepower.png)
3. [Figure 3: Vehicle Price vs. Engine Displacement Size](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/03_price_vs_enginesize.png)
4. [Figure 4: Vehicle Price vs. Curb Weight](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/04_price_vs_curbweight.png)
5. [Figure 5: Vehicle Price vs. Average Fuel Economy (Mileage)](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/05_price_vs_average_mpg.png)
6. [Figure 6: Correlation Heatmap of Primary Vehicle Attributes & Price](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/reports/figures/06_correlation_heatmap.png)

---

## 8. Dataset Limitations
1. **Sample Size:** 205 total records is relatively modest for training deep models, prioritizing cross-validated regularized linear models and shallow ensemble methods over high-parameter architectures.
2. **Absence of Year / Odometer Mileage:** The dataset captures fuel efficiency (`citympg`, `highwaympg`) rather than odometer distance driven or vehicle production year.
3. **Geographic Scope:** Derived from the US automotive market survey, reflecting US Dollar pricing and automotive segmentation.
