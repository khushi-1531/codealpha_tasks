# CodeAlpha Data Science Task 3 — Data Cleaning Report

## 1. Executive Summary
This report documents the rigorous data cleaning protocol executed on the raw automobile pricing dataset (`CarPrice_Assignment.csv`) for **CodeAlpha Task 3: Car Price Prediction with Machine Learning**.

The raw dataset contains 205 records and 26 features. Although the dataset possesses zero missing values and zero duplicate records in its unedited state, several structural data-quality challenges required standardized intervention:
1. Whitespace padding across column names and string observations.
2. Embedded manufacturer names inside composite model strings (`CarName`) containing typographical inconsistencies.
3. String-based word representations for ordinal discrete numeric variables (`doornumber`, `cylindernumber`).
4. Monotonic identification numbers (`car_ID`) posing data leakage risks if left in feature subsets.

The cleaning pipeline transformed the raw data into a validated, standardized structure stored at `data/processed/car_price_cleaned.csv` (205 rows, 27 columns), while maintaining strict immutability of the original raw file.

---

## 2. Dataset Immutability Verification
- **Raw File Location:** `data/raw/CarPrice_Assignment.csv`
- **Integrity Verification Method:** Cryptographic SHA-256 hash.
- **SHA-256 Fingerprint:** `2c78d99359a34cb6c64a97f276c1b6ea0532197b9b950b4521f65c0d9efcbc2b`
- **Status:** Unchanged. The raw dataset was loaded in read-only mode and not modified in place.

---

## 3. Cleaning Operations & Methodological Rules

### 3.1 Column Name Standardization
- Column headers were inspected and stripped of extraneous whitespace characters.
- All 26 original column names were preserved to maintain direct traceability against the companion data dictionary (`Data Dictionary - carprices.xlsx`).
- One derived column (`brand`) was appended, expanding the column count from 26 to 27.

### 3.2 String Whitespace Normalization
- All object/string columns (`CarName`, `fueltype`, `aspiration`, `doornumber`, `carbody`, `drivewheel`, `enginelocation`, `enginetype`, `cylindernumber`, `fuelsystem`) were trimmed of leading, trailing, and redundant internal whitespace using vectorised `.str.strip()` operations.

### 3.3 Manufacturer Extraction & Typo Corrections
The `CarName` column contains composite strings combining the manufacturer brand and model designation (e.g. `"alfa-romero giulia"`, `"toyota corolla"`). 
The primary manufacturer token was extracted by taking the first whitespace-delimited word in lowercase.

An initial audit of extracted brand tokens revealed 27 unique values containing **5 verified typographical anomalies across 7 records**:

| Observed Erroneous Token | Record Frequency | Verified Canonical Manufacturer | Rationale & Evidence |
| :--- | :---: | :--- | :--- |
| `maxda` | 2 | `mazda` | Common phonetic typo; model lines correspond to Mazda GLC / 626. |
| `toyouta` | 1 | `toyota` | Typographical keystroke error; model line is Toyota Celica. |
| `vokswagen` | 1 | `volkswagen` | Missing 'l' keystroke error; model is Volkswagen Dasher. |
| `vw` | 2 | `volkswagen` | Colloquial abbreviation for Volkswagen; models are Rabbit and Type 3. |
| `porcshce` | 1 | `porsche` | Transposed letter typo; model is Porsche 911. |

**Result:** After normalization, the brand inventory consolidated from 27 messy tokens into exactly **22 legitimate automobile manufacturers**:
*alfa-romero, audi, bmw, buick, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercury, mitsubishi, nissan, peugeot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo.*

### 3.4 Word-Based Numeric Category Conversion
Two categorical features stored ordinal quantities as English words rather than integer counts:
1. **`doornumber`:**
   - Raw values: `'two'` (90 records), `'four'` (115 records).
   - Mapping: `{'two': 2, 'four': 4}`.
   - Result: Encoded as integer `int64`.
2. **`cylindernumber`:**
   - Raw values: `'two'` (4), `'three'` (1), `'four'` (159), `'five'` (11), `'six'` (24), `'eight'` (5), `'twelve'` (1).
   - Mapping: `{'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}`.
   - Result: Encoded as integer `int64`.

Converting these variables into numeric quantities allows linear and tree-based regression models to capture monotonic mechanical scaling (e.g. higher cylinder count directly driving displacement and engine power).

### 3.5 Identifier Handling & Leakage Prevention
- **`car_ID`:** Represents a sequential row index (1 to 205) assigned during dataset compilation.
- In the raw data, `car_ID` displays a weak spurious correlation of -0.109 with `price` solely because manufacturer batches were entered sequentially.
- **Handling:** `car_ID` is preserved in `data/processed/car_price_cleaned.csv` to allow row-level tracking and auditing, but is explicitly categorized in `IDENTIFIER_COLUMNS` and excluded from model feature sets.

---

## 4. Integrity Validations & Data Quality Audit

| Integrity Check | Validation Rule | Observed Result | Status |
| :--- | :--- | :--- | :---: |
| Missing Values | Sum of nulls across all cells == 0 | 0 nulls detected (0.00%) | **Passed** |
| Duplicate Records | Duplicate rows excluding `car_ID` == 0 | 0 duplicates found | **Passed** |
| Price Positivity | `price > 0` | Min: $5,118.00, Max: $45,400.00 | **Passed** |
| Horsepower Range | `horsepower > 0` | Min: 48 bhp, Max: 288 bhp | **Passed** |
| Engine Dimensions | `wheelbase, carlength, enginesize > 0` | All positive physical quantities | **Passed** |
| Brand Cardinality | Canonical manufacturer count == 22 | Exactly 22 unique brands | **Passed** |
| Target Type | `price` is float64 numeric | Valid continuous float64 | **Passed** |

---

## 5. Outlier Observations & Retention Decision
- **High-End Luxury / High-Horsepower Vehicles:**
  - Porsche models (e.g., Porsche 911, horsepower up to 288, prices reaching $45,400).
  - Jaguar, BMW, and Buick models with prices exceeding $30,000.
- **Decision:** **Retained.** These observations are genuine, legitimate market vehicles representing the premium automotive segment. Blindly truncating or removing high-priced vehicles would destroy the model's ability to generalize to luxury vehicle valuation.

---

## 6. Output Artifacts
- **Cleaned Dataset:** [`data/processed/car_price_cleaned.csv`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/data/processed/car_price_cleaned.csv)
- **Cleaning Script:** [`src/data_cleaning.py`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/src/data_cleaning.py)
- **Verification Tests:** [`tests/test_data_pipeline.py`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task3_Car_Price_Prediction/tests/test_data_pipeline.py)
