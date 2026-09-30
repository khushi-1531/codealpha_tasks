# Task 2 — Unemployment Analysis: Data Cleaning & Preprocessing Report

**Project:** CodeAlpha Data Science Task 2: Unemployment Analysis with Python  
**Working Directory:** `C:\Users\khushi\OneDrive\Desktop\codealpha interniship\codealpha_tasks\DataScience\Task2_Unemployment_Analysis`  
**Date:** 2026-09-30  
**Status:** Completed & Validated  

---

## 1. Executive Summary & Dataset Selection Rationale

### 1.1 Dataset Candidates
Two datasets were obtained from the official Kaggle *Unemployment in India* repository:
1. `Unemployment in India.csv` (768 raw rows, 7 raw columns)
2. `Unemployment_Rate_upto_11_2020.csv` (267 raw rows, 9 raw columns)

### 1.2 Primary Dataset Selection: `Unemployment in India.csv`
**Decision:** `Unemployment in India.csv` was selected as the **PRIMARY dataset** for this task.

**Technical Rationale:**
- **Robust Pre-COVID Baseline:** COVID-19 impact analysis requires a solid pre-pandemic baseline. `Unemployment in India.csv` spans from **May 31, 2019 to June 30, 2020** (14 consecutive months), providing **10 months of pre-COVID baseline data** (May 2019 to February 2020) and **4 months of COVID pandemic data** (March 2020 to June 2020). Conversely, `Unemployment_Rate_upto_11_2020.csv` starts only in January 2020, offering just 2 months of pre-lockdown data.
- **Granular Rural vs. Urban Segmentation:** Only `Unemployment in India.csv` contains the `Area` feature (`Rural` vs. `Urban`), directly fulfilling the official requirement to analyze rural vs. urban employment dynamics.
- **State-Level Granularity:** Covers 28 Indian States and Union Territories (including Chandigarh).

### 1.3 Role of the Secondary Dataset: `Unemployment_Rate_upto_11_2020.csv`
- Serves as a **secondary complementary source** for examining the extended economic recovery during the phased "Unlock" period into late 2020 (July 2020 to October 2020).
- Provides zonal geographic groupings (`Region.1` / `zone`: East, North, Northeast, South, West) and geographic coordinates for mapping.
- Not merged blindly into the primary dataset to avoid schema corruption, duplicated observation units, and artificial structural distortion.

---

## 2. Raw Data Characteristics & Issues Detected

| Metric / Issue | `Unemployment in India.csv` (Raw) | `Unemployment_Rate_upto_11_2020.csv` (Raw) |
|---|---|---|
| **Raw File Path** | `data/raw/Unemployment in India.csv` | `data/raw/Unemployment_Rate_upto_11_2020.csv` |
| **Raw Line Count** | 769 lines (1 header + 768 rows) | 268 lines (1 header + 267 rows) |
| **All-NaN Rows** | 28 rows (rows 359–372 and 754–767) | 0 rows |
| **Header Formatting** | Leading whitespace (`' Date'`, `' Frequency'`, etc.) | Leading whitespace (`' Date'`, `' Frequency'`, etc.) |
| **String Formatting** | Leading whitespace in date (`' 31-05-2019'`) and frequency (`' Monthly'`) | Leading whitespace in date (`' 31-01-2020'`) and frequency (`' M'`) |
| **Data Types** | `estimated_employed` loaded as `float64` due to NaNs | Correct numeric types |

---

## 3. Data Cleaning Pipeline Steps

The programmatic cleaning pipeline was implemented in [`src/data_cleaning.py`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/src/data_cleaning.py).

### Step 1: Preservation of Raw Data
The original raw files in `data/raw/` were read in read-only fashion and preserved bit-for-bit without in-place modification.

### Step 2: Blank Row Elimination
In `Unemployment in India.csv`, exactly 28 rows contained `NaN` across all columns:
- 14 rows separated the Rural block from the Urban block (rows 359 to 372).
- 14 trailing rows existed at the bottom of the CSV (rows 754 to 767).
- Action: Dropped all-NaN rows using `df.dropna(how='all')`, leaving **exactly 740 valid observation records**.

### Step 3: Column Header Standardization
All column headers were stripped of whitespace, converted to lowercase snake_case, and standardized:
- `'Region'` &rarr; `'region'`
- `' Date'` &rarr; `'date'`
- `' Frequency'` &rarr; `'frequency'`
- `' Estimated Unemployment Rate (%)'` &rarr; `'estimated_unemployment_rate_pct'`
- `' Estimated Employed'` &rarr; `'estimated_employed'`
- `' Estimated Labour Participation Rate (%)'` &rarr; `'estimated_labour_participation_rate_pct'`
- `'Area'` &rarr; `'area'`

### Step 4: String Normalization
- Categorical columns (`region`, `frequency`, `area`) were stripped of leading/trailing whitespace.
- Frequency entries (`' Monthly'` and `'Monthly'`) were standardized to `'Monthly'`.
- Area confirmed to contain strictly two categories: `'Rural'` (359 records) and `'Urban'` (381 records).

### Step 5: Strict Datetime Parsing
- Date strings were trimmed of whitespace and parsed using `pd.to_datetime(format='%d-%m-%Y')`.
- Confirmed zero `NaT` (Not a Time) values.
- Spanned strictly from `2019-05-31` to `2020-06-30` (14 monthly observations).

### Step 6: Numeric Type Casting
- `estimated_employed` was converted from `float64` to strict 64-bit integer (`int64`).
- `estimated_unemployment_rate_pct` and `estimated_labour_participation_rate_pct` validated as `float64`.

### Step 7: Domain-Specific Feature Engineering
To support granular and structured EDA, the following deterministic features were derived:
- `year`: Integer calendar year (`2019`, `2020`).
- `month`: Integer calendar month (`1` to `12`).
- `month_name`: 3-letter month abbreviation (`'May'`, `'Jun'`, etc.).
- `year_month`: Standardized temporal label (`'2019-05'` to `'2020-06'`).
- `period`: Categorical macroeconomic regime:
  - `'Pre-COVID'`: Observation date < `2020-03-01` (10 months: May 2019 to Feb 2020).
  - `'COVID-Period'`: Observation date &ge; `2020-03-01` (4 months: Mar 2020 to Jun 2020).
- `covid_phase`: Granular event-based timeline segmentation:
  - `'Pre-COVID Baseline'` (May 2019 – Feb 2020)
  - `'Early Lockdown (Mar 2020)'` (National lockdown declared March 24, 2020)
  - `'Peak Lockdown (Apr-May 2020)'` (Strictest nationwide economic restriction)
  - `'Early Unlock (Jun 2020)'` (Unlock Phase 1 reopening)

---

## 4. Final Cleaned Dataset Verification & Schema

The processed dataset was written to [`data/processed/unemployment_india_cleaned.csv`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/data/processed/unemployment_india_cleaned.csv).

### 4.1 Schema Overview

| # | Column Name | Cleaned Data Type | Null Count | Unique Values | Description |
|---|---|---|---|---|---|
| 0 | `region` | `object` (str) | 0 | 28 | Indian State / Union Territory name |
| 1 | `date` | `datetime64[ns]` | 0 | 14 | Monthly reporting date |
| 2 | `frequency` | `object` (str) | 0 | 1 | Survey frequency (`'Monthly'`) |
| 3 | `estimated_unemployment_rate_pct` | `float64` | 0 | 624 | Estimated unemployment rate (%) |
| 4 | `estimated_employed` | `int64` | 0 | 740 | Estimated count of employed persons |
| 5 | `estimated_labour_participation_rate_pct` | `float64` | 0 | 626 | Labour force participation rate (%) |
| 6 | `area` | `object` (str) | 0 | 2 | Geographical classification (`'Rural'`, `'Urban'`) |
| 7 | `year` | `int32` | 0 | 2 | Calendar year (`2019`, `2020`) |
| 8 | `month` | `int32` | 0 | 12 | Calendar month index (1–12) |
| 9 | `month_name` | `object` (str) | 0 | 12 | Short month name (`'Jan'` to `'Dec'`) |
| 10 | `year_month` | `object` (str) | 0 | 14 | Temporal key (`'2019-05'` to `'2020-06'`) |
| 11 | `period` | `object` (str) | 0 | 2 | Macro regime (`'Pre-COVID'`, `'COVID-Period'`) |
| 12 | `covid_phase` | `object` (str) | 0 | 4 | Granular pandemic phase |

### 4.2 Cleaned Dataset Dimensions
- **Rows:** 740
- **Columns:** 13
- **Missing Values:** 0
- **Duplicates:** 0

---

## 5. Automated Integrity Testing
All cleaning operations and integrity invariants are continuously verified by automated tests in [`tests/test_data_cleaning.py`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/tests/test_data_cleaning.py).
- Total tests executed: 10
- Test results: **10 passed, 0 failed, 0 errors** (100% pass rate).
