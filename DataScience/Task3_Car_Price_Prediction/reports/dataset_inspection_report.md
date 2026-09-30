# CodeAlpha Data Science Task 3 — Dataset Inspection Report

## 1. Executive Summary
This report provides a formal data audit and technical inspection of candidate public datasets for **CodeAlpha Data Science Task 3: Car Price Prediction with Machine Learning**. The primary objective of this task is to train and evaluate a regression model capable of predicting vehicle valuation using multi-dimensional physical, mechanical, efficiency, and brand goodwill attributes.

Following evaluation of public candidate datasets on Kaggle, the **Car Price Prediction Dataset (`CarPrice_Assignment.csv`)** by user *hellbuoy* on Kaggle (derived from the classic automobile pricing study) was selected as the primary dataset. It maps directly to every explicit requirement specified in the official CodeAlpha task prompt—specifically containing **brand goodwill** (via brand extraction from `CarName`), **horsepower** (`horsepower`), **mileage** (`citympg` and `highwaympg`), extensive engine and body attributes, and a continuous numeric target (`price`).

---

## 2. Candidate Datasets Evaluated

| Dataset Name | Kaggle Source Identifier | Rows | Columns | Target Variable | Key Features Available | Evaluation / Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CarPrice Assignment** *(Selected Primary)* | [`hellbuoy/car-price-prediction`](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction) | 205 | 26 | `price` (USD) | Brand (`CarName`), `horsepower`, `citympg`, `highwaympg`, `enginesize`, `curbweight`, dimensions | **Selected:** 100% feature alignment with CodeAlpha prompt wording; complete data with zero missing values; high feature density. |
| **Car details v3** *(Alternative Candidate)* | [`nehalbirla/vehicle-dataset-from-cardekho`](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) | 8,128 | 13 | `selling_price` (INR) | `name`, `year`, `km_driven`, `mileage`, `engine`, `max_power`, `fuel`, `transmission` | Secondary option: larger sample size, but contains 1,100 missing values, 1,202 duplicate rows, and requires unparsed string-unit extraction (`bhp`, `kmpl`, `CC`). |
| **Car Data** *(Small Alternative)* | [`nehalbirla/vehicle-dataset-from-cardekho`](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) | 301 | 9 | `Selling_Price` (Lakhs) | `Car_Name`, `Year`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Transmission` | Rejected: lacks explicit horsepower and mechanical engine specifications. |

---

## 3. Dataset Identity & Acquisition Details
- **Selected Dataset:** Car Price Prediction Dataset (`CarPrice_Assignment.csv`)
- **Primary Source:** Kaggle API (`hellbuoy/car-price-prediction`)
- **Public URL:** [https://www.kaggle.com/datasets/hellbuoy/car-price-prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)
- **Local Storage Path:** `data/raw/CarPrice_Assignment.csv`
- **Companion File:** `data/raw/Data Dictionary - carprices.xlsx` (Official feature dictionary)
- **File Size:** 26,717 bytes (CSV)
- **Ingestion Mode:** Direct download preserved in original unedited raw format. No rows dropped, no values altered.

---

## 4. Dataset Dimensions & Structural Overview
- **Total Records (Rows):** 205
- **Total Variables (Columns):** 26
- **Total Missing / Null Values:** 0 (0.00%)
- **Total Duplicate Rows:** 0 (0.00%)
- **Target Variable:** `price` (Continuous numeric, float64)
- **Feature Types Breakdown:**
  - **Integer (`int64`):** 8 columns (`car_ID`, `symboling`, `curbweight`, `enginesize`, `horsepower`, `peakrpm`, `citympg`, `highwaympg`)
  - **Floating Point (`float64`):** 8 columns (`wheelbase`, `carlength`, `carwidth`, `carheight`, `boreratio`, `stroke`, `compressionratio`, `price`)
  - **Categorical String (`object`):** 10 columns (`CarName`, `fueltype`, `aspiration`, `doornumber`, `carbody`, `drivewheel`, `enginelocation`, `enginetype`, `cylindernumber`, `fuelsystem`)

---

## 5. Comprehensive Column Inventory & Feature Audit

| # | Column Name | Data Type | Null Count | Unique Count | Observed Values / Range | Semantic Role |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 1 | `car_ID` | `int64` | 0 | 205 | 1 to 205 | Identifier (Drop before modeling to prevent index leakage) |
| 2 | `symboling` | `int64` | 0 | 6 | -2, -1, 0, 1, 2, 3 | Insurance actuarial risk rating |
| 3 | `CarName` | `object` | 0 | 147 | e.g. "toyota corolla", "bmw x1" | Vehicle make and model (Extract Brand Goodwill) |
| 4 | `fueltype` | `object` | 0 | 2 | `gas` (185), `diesel` (20) | Engine fuel mechanism |
| 5 | `aspiration` | `object` | 0 | 2 | `std` (168), `turbo` (37) | Naturally aspirated vs. turbocharged |
| 6 | `doornumber` | `object` | 0 | 2 | `four` (115), `two` (90) | Number of doors (Convertible to integer) |
| 7 | `carbody` | `object` | 0 | 5 | `sedan` (96), `hatchback` (70), `wagon` (25), `hardtop` (8), `convertible` (6) | Chassis body construction |
| 8 | `drivewheel` | `object` | 0 | 3 | `fwd` (120), `rwd` (76), `4wd` (9) | Drivetrain layout |
| 9 | `enginelocation` | `object` | 0 | 2 | `front` (202), `rear` (3) | Engine placement (rear engines are rare sports cars) |
| 10 | `wheelbase` | `float64` | 0 | 53 | 86.60 to 120.90 (in) | Distance between front and rear axles |
| 11 | `carlength` | `float64` | 0 | 75 | 141.10 to 208.10 (in) | Exterior length of the vehicle |
| 12 | `carwidth` | `float64` | 0 | 44 | 60.30 to 72.30 (in) | Exterior width of the vehicle |
| 13 | `carheight` | `float64` | 0 | 49 | 47.80 to 59.80 (in) | Exterior height of the vehicle |
| 14 | `curbweight` | `int64` | 0 | 171 | 1,488 to 4,066 (lbs) | Unladen vehicle curb weight |
| 15 | `enginetype` | `object` | 0 | 7 | `ohc`, `ohcv`, `ohcf`, `l`, `dohc`, `rotor`, `dohcv` | Internal combustion valvetrain architecture |
| 16 | `cylindernumber` | `object` | 0 | 7 | `four`, `six`, `five`, `eight`, `two`, `three`, `twelve` | Cylinder count (Convertible to integer) |
| 17 | `enginesize` | `int64` | 0 | 44 | 61 to 326 (cu in) | Engine displacement volume |
| 18 | `fuelsystem` | `object` | 0 | 8 | `mpfi`, `2bbl`, `idi`, `1bbl`, `spdi`, `4bbl`, `mfi`, `spfi` | Fuel injection and metering architecture |
| 19 | `boreratio` | `float64` | 0 | 38 | 2.54 to 3.94 | Cylinder bore diameter to stroke ratio |
| 20 | `stroke` | `float64` | 0 | 37 | 2.07 to 4.17 | Piston displacement distance |
| 21 | `compressionratio`| `float64` | 0 | 32 | 7.00 to 23.00 | Engine cylinder volume compression ratio |
| 22 | `horsepower` | `int64` | 0 | 59 | 48 to 288 (bhp) | Peak engine power output |
| 23 | `peakrpm` | `int64` | 0 | 23 | 4,150 to 6,600 (RPM) | Engine rotational speed at peak power |
| 24 | `citympg` | `int64` | 0 | 29 | 13 to 49 (MPG) | City fuel economy / mileage |
| 25 | `highwaympg` | `int64` | 0 | 30 | 16 to 54 (MPG) | Highway fuel economy / mileage |
| 26 | `price` | `float64` | 0 | 189 | $5,118.00 to $45,400.00 | **Target variable (Supervised regression)** |

---

## 6. Numerical Feature Distribution Summary

| Feature | Min | 25% | Median | Mean | 75% | Max | Std Dev | Skewness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `wheelbase` | 86.60 | 94.50 | 97.00 | 98.76 | 102.40 | 120.90 | 6.02 | 1.05 |
| `carlength` | 141.10 | 166.30 | 173.20 | 174.05 | 183.10 | 208.10 | 12.34 | 0.16 |
| `carwidth` | 60.30 | 64.10 | 65.50 | 65.91 | 66.90 | 72.30 | 2.15 | 0.90 |
| `carheight` | 47.80 | 52.00 | 54.10 | 53.72 | 55.50 | 59.80 | 2.44 | -0.24 |
| `curbweight` | 1,488.00 | 2,148.00 | 2,414.00 | 2,555.57 | 2,935.00 | 4,066.00 | 520.68 | 0.68 |
| `enginesize` | 61.00 | 97.00 | 120.00 | 126.91 | 141.00 | 326.00 | 41.64 | 1.95 |
| `boreratio` | 2.54 | 3.15 | 3.31 | 3.33 | 3.58 | 3.94 | 0.27 | 0.02 |
| `stroke` | 2.07 | 3.11 | 3.29 | 3.26 | 3.41 | 4.17 | 0.31 | -0.69 |
| `compressionratio`| 7.00 | 8.60 | 9.00 | 10.14 | 9.40 | 23.00 | 3.97 | 2.61 |
| `horsepower` | 48.00 | 70.00 | 95.00 | 104.12 | 116.00 | 288.00 | 39.54 | 1.41 |
| `peakrpm` | 4,150.00 | 4,800.00 | 5,200.00 | 5,125.12 | 5,500.00 | 6,600.00 | 476.99 | 0.08 |
| `citympg` | 13.00 | 19.00 | 24.00 | 25.22 | 30.00 | 49.00 | 6.54 | 0.66 |
| `highwaympg` | 16.00 | 25.00 | 30.00 | 30.75 | 34.00 | 54.00 | 6.89 | 0.54 |
| **`price` (Target)** | **$5,118.00** | **$7,788.00** | **$10,295.00** | **$13,276.71** | **$16,503.00** | **$45,400.00** | **$7,988.85** | **1.78** |

---

## 7. Target Variable Analysis (`price`)
- **Range:** $5,118.00 to $45,400.00 (No negative, zero, or missing values).
- **Mean vs. Median:** Mean ($13,276.71) significantly exceeds the Median ($10,295.00), demonstrating a pronounced right-skewed distribution.
- **Skewness:** +1.778.
- **Interquartile Range (IQR):** $8,715.00 (Q1 = $7,788.00, Q3 = $16,503.00). High-end luxury/sports models (e.g. Porsche, Jaguar, high-end BMW/Mercedes) form a long right-hand tail.
- **Modeling Implications:**
  - Standard linear regression with raw prices may suffer from heteroscedasticity due to the right-tail spread.
  - A logarithmic target transformation ($\log(\text{price})$ or $\log1p(\text{price})$) will normalize the target distribution and stabilize residual variance.

---

## 8. Correlation Analysis with Target (`price`)

| Feature | Pearson Correlation ($r$) | Direction & Strength | Physical / Domain Interpretation |
| :--- | :---: | :---: | :--- |
| `enginesize` | **+0.8741** | Very Strong Positive | Larger displacement directly dictates higher manufacturing tier and premium pricing. |
| `curbweight` | **+0.8353** | Very Strong Positive | Heavier structural mass correlates with vehicle size class and premium materials. |
| `horsepower` | **+0.8081** | Very Strong Positive | Directly drives vehicle performance, sports pedigree, and consumer price willingness. |
| `carwidth` | **+0.7593** | Strong Positive | Wider vehicles denote luxury, executive, or sports car body frames. |
| `carlength` | **+0.6829** | Moderate-Strong Positive | Longer wheelbases and chassis length map to higher vehicle segments. |
| `wheelbase` | **+0.5778** | Moderate Positive | Interior cabin roominess and vehicle footprint. |
| `boreratio` | **+0.5532** | Moderate Positive | Engine cylinder architecture. |
| `carheight` | **+0.1193** | Weak Positive | Minimal standalone price impact. |
| `stroke` | **+0.0794** | Negligible Positive | Engine stroke length. |
| `compressionratio`| **+0.0680** | Negligible Positive | Diesel engines have high compression ratios (21-23) but span both budget and premium tiers. |
| `symboling` | **-0.0800** | Negligible Negative | Actuarial risk score (-2 to +3). |
| `peakrpm` | **-0.0853** | Negligible Negative | High-revving smaller engines (e.g., Honda VTEC) vs. lower-revving large displacement V8s. |
| `car_ID` | **-0.1091** | Spurious / Non-Causal | Arbitrary dataset sequence index; must be dropped. |
| `citympg` | **-0.6858** | Strong Negative | Economy/budget commuter vehicles boast high MPG; luxury and sports cars exhibit lower MPG. |
| `highwaympg` | **-0.6976** | Strong Negative | Aligns with `citympg`; high highway mileage is inversely associated with vehicle price. |

---

## 9. Critical Data-Quality Observations

### 9.1 Brand Name Spelling Inconsistencies (`CarName`)
The raw `CarName` column combines brand make and specific model (e.g., `"alfa-romero giulia"`, `"toyota corona"`). When parsing the primary brand (the first whitespace-delimited token), 27 distinct tokens are found. However, domain audit reveals **4 spelling anomalies**:
1. `maxda` (2 occurrences) $\rightarrow$ Typo for `mazda` (which has 15 occurrences).
2. `toyouta` (1 occurrence) $\rightarrow$ Typo for `toyota` (which has 31 occurrences).
3. `vokswagen` (1 occurrence) and `vw` (2 occurrences) $\rightarrow$ Inconsistencies for `volkswagen` (which has 9 occurrences).
4. `porcshce` (1 occurrence) $\rightarrow$ Typo for `porsche` (which has 4 occurrences).

*Pre-processing Recommendation:* Normalize these typos during data cleaning. After correction, the dataset resolves to exactly 22 legitimate automobile manufacturers.

### 9.2 Word-Based Number Representation
- `doornumber` is encoded as strings: `['two', 'four']`.
- `cylindernumber` is encoded as strings: `['four', 'six', 'five', 'three', 'twelve', 'two', 'eight']`.

*Pre-processing Recommendation:* Both features represent ordinal discrete quantities. Converting them to integer values (e.g. `2, 4, 6, 8, 12`) creates continuous numerical features that linear and tree-based regression models can exploit directly.

### 9.3 Spurious Identifier Feature
- `car_ID` is a monotonically increasing integer index (1 to 205).
- It displays a weak spurious correlation of -0.109 with price due to grouping of manufacturer batches.

*Recommendation:* Exclude `car_ID` entirely from feature sets to prevent model reliance on row ordering.

### 9.4 Extreme Multicollinearity
- High pairwise correlations exist among dimensional and engine variables:
  - `curbweight` vs. `carlength`: $r = +0.878$
  - `curbweight` vs. `carwidth`: $r = +0.867$
  - `citympg` vs. `highwaympg`: $r = +0.971$
  - `horsepower` vs. `enginesize`: $r = +0.810$

*Recommendation:* While tree-based ensembles (Random Forest, Gradient Boosting) handle multicollinearity robustly, linear regression will benefit from regularized estimators (Ridge, Lasso) and combined ratio features (e.g., average MPG or power-to-weight ratio).

---

## 10. Suitability Assessment for CodeAlpha Task 3

| CodeAlpha Requirement | Dataset Support & Evidence | Status |
| :--- | :--- | :---: |
| **Car price prediction** | Explicit continuous numerical target column `price` ($5,118 to $45,400). | **Full Match** |
| **Regression task** | Supervised regression setup with multiple continuous and categorical predictors. | **Full Match** |
| **Brand goodwill** | 22 car brands extractable from `CarName`. Permits grouping into Budget, Mid-Tier, and Luxury brand equity clusters. | **Full Match** |
| **Horsepower** | Dedicated numerical feature `horsepower` ($r = +0.808$ with price). | **Full Match** |
| **Mileage** | Two distinct fuel efficiency features: `citympg` ($r = -0.686$) and `highwaympg` ($r = -0.698$). | **Full Match** |
| **Engine & Car Attributes** | 12 mechanical/physical attributes (`enginesize`, `curbweight`, `wheelbase`, `carbody`, `fueltype`, etc.). | **Full Match** |
| **Preprocessing Required** | Brand typo cleaning, word-to-number parsing, categorical encoding, scaling. | **Full Match** |
| **Feature Engineering** | Power-to-weight ratio, average MPG, brand tier grouping, log target transformation. | **Full Match** |
| **Evaluation Framework** | Standard regression metrics ($R^2$, MAE, RMSE, MAPE) directly computable. | **Full Match** |

---

## 11. Recommended Role for Important Features

1. **`CarName` $\rightarrow$ `brand` / `brand_tier` (Feature Engineering):**
   - Extract brand name, clean typos, and cluster into Market Goodwill Tiers:
     - *Luxury / Premium:* Porsche, Jaguar, BMW, Mercedes, Audi, Volvo.
     - *Mid-Tier:* Peugeot, Saab, Volkswagen, Mazda, Alfa-Romeo.
     - *Budget / Economy:* Toyota, Honda, Nissan, Mitsubishi, Dodge, Plymouth, Chevrolet, Isuzu, Subaru.
2. **`horsepower` (Predictor):**
   - Retain as core power metric. Combine with `curbweight` to generate `power_to_weight_ratio`.
3. **`citympg` & `highwaympg` (Predictor / Engineering):**
   - Synthesize an overall average fuel efficiency metric: $\text{avg\_mpg} = 0.55 \times \text{citympg} + 0.45 \times \text{highwaympg}$.
4. **`curbweight`, `enginesize`, `carwidth`, `carlength` (Predictors):**
   - Core physical determinants of vehicle class and manufacturing value.
5. **`cylindernumber` & `doornumber` (Preprocessing):**
   - Map from word representation to numeric values (`2, 3, 4, 5, 6, 8, 12`).
6. **`fueltype`, `aspiration`, `carbody`, `drivewheel`, `enginelocation` (Categorical Encoding):**
   - One-hot encode using Scikit-learn `OneHotEncoder(drop='first', handle_unknown='ignore')`.
7. **`price` (Target Transformation):**
   - Model both raw price and log-transformed price $\log(\text{price})$ to benchmark performance and residual normality.

---

## 12. Verification & Guardrail Audit
- [x] **No data cleaning executed:** The raw dataset in `data/raw/CarPrice_Assignment.csv` remains unedited and identical to the Kaggle source.
- [x] **No rows dropped:** All 205 raw rows preserved.
- [x] **No missing values imputed:** All raw columns retained without mutation.
- [x] **No machine learning models trained:** No baseline models or estimators instantiated or saved.
- [x] **No performance metrics fabricated:** No $R^2$, MAE, or RMSE scores claimed.
- [x] **Tasks 1 and 2 isolated:** Directories `Task1_Iris_Flower_Classification` and `Task2_Unemployment_Analysis` were not modified.
- [x] **Git integrity preserved:** No commits or pushes performed.
