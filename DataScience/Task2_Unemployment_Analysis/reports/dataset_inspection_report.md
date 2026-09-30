# Task 2 — Unemployment Analysis: Dataset Acquisition & Initial Inspection Report

**Project:** CodeAlpha Data Science Task 2: Unemployment Analysis with Python  
**Working Directory:** `C:\Users\khushi\OneDrive\Desktop\codealpha interniship\codealpha_tasks\DataScience\Task2_Unemployment_Analysis`  
**Date of Inspection:** 2026-09-30  
**Data Source:** Kaggle — *Unemployment in India* (Public Dataset)  

---

## 1. Dataset Acquisition Overview

The official Kaggle dataset **"Unemployment in India"** was acquired directly and saved in the raw data directory without modifications:
- Raw Data Location: `data/raw/`
- Downloaded Files:
  1. `Unemployment_Rate_upto_11_2020.csv` *(Primary expected dataset file)*
  2. `Unemployment in India.csv` *(Complementary file in the same Kaggle dataset containing Rural/Urban breakdown)*

Both raw files are preserved in their original, unmodified form.

---

## 2. Detailed Inspection: Primary Dataset (`Unemployment_Rate_upto_11_2020.csv`)

### 2.1 File & Dimensional Summary
- **Exact Filename:** `Unemployment_Rate_upto_11_2020.csv`
- **File Location:** `data/raw/Unemployment_Rate_upto_11_2020.csv`
- **Total Rows:** 267
- **Total Columns:** 9
- **Total Elements:** 2,403
- **Duplicate Rows:** 0 exact duplicate rows

### 2.2 Column Schema, Data Types & Missing Values

| # | Raw Column Name | Cleaned Column Name | Pandas Data Type | Non-Null Count | Missing Values |
|---|-----------------|---------------------|------------------|----------------|----------------|
| 0 | `Region` | `Region` | `object` (string) | 267 | 0 (0.0%) |
| 1 | ` Date` | `Date` | `object` (string) | 267 | 0 (0.0%) |
| 2 | ` Frequency` | `Frequency` | `object` (string) | 267 | 0 (0.0%) |
| 3 | ` Estimated Unemployment Rate (%)` | `Estimated Unemployment Rate (%)` | `float64` | 267 | 0 (0.0%) |
| 4 | ` Estimated Employed` | `Estimated Employed` | `int64` | 267 | 0 (0.0%) |
| 5 | ` Estimated Labour Participation Rate (%)` | `Estimated Labour Participation Rate (%)` | `float64` | 267 | 0 (0.0%) |
| 6 | `Region.1` | `Region.1` (Zone/Macro-region) | `object` (string) | 267 | 0 (0.0%) |
| 7 | `longitude` | `longitude` | `float64` | 267 | 0 (0.0%) |
| 8 | `latitude` | `latitude` | `float64` | 267 | 0 (0.0%) |

*Note on Raw Headers:* Columns 1 to 5 contain leading whitespace in their raw header names (e.g., `' Date'`, `' Frequency'`).

### 2.3 Date Range & Temporal Attributes
- **Minimum Date:** `31-01-2020` (January 31, 2020)
- **Maximum Date:** `31-10-2020` (October 31, 2020)
- **Observation Count:** 10 unique monthly reporting dates (Jan 2020 to Oct 2020).
- **Date Values:**
  - `31-01-2020`
  - `29-02-2020`
  - `31-03-2020`
  - `30-04-2020`
  - `31-05-2020`
  - `30-06-2020`
  - `31-07-2020`
  - `31-08-2020`
  - `30-09-2020`
  - `31-10-2020`
- **Unique Frequency:** `['M']` (Monthly reporting)

### 2.4 Geographic Entities & Attributes
- **Unique Regions / States Count:** 27
- **Unique Regions / States:**
  1. Andhra Pradesh
  2. Assam
  3. Bihar
  4. Chhattisgarh
  5. Delhi
  6. Goa
  7. Gujarat
  8. Haryana
  9. Himachal Pradesh
  10. Jammu & Kashmir
  11. Jharkhand
  12. Karnataka
  13. Kerala
  14. Madhya Pradesh
  15. Maharashtra
  16. Meghalaya
  17. Odisha
  18. Puducherry
  19. Punjab
  20. Rajasthan
  21. Sikkim
  22. Tamil Nadu
  23. Telangana
  24. Tripura
  25. Uttar Pradesh
  26. Uttarakhand
  27. West Bengal

- **Macro-Regions / Zones (`Region.1`):** 5 zones
  - `East`
  - `North`
  - `Northeast`
  - `South`
  - `West`

- **Area Column (`Rural` vs `Urban`):** 
  - Not present in `Unemployment_Rate_upto_11_2020.csv`. This file provides statewide aggregates along with macro-regional zones.

### 2.5 Summary Statistics of Numerical Columns

| Metric | Estimated Unemployment Rate (%) | Estimated Employed | Estimated Labour Participation Rate (%) | Longitude | Latitude |
|---|---|---|---|---|---|
| **Count** | 267 | 267 | 267 | 267 | 267 |
| **Mean** | 12.24% | 13,962,106 | 41.68% | 22.83 | 80.53 |
| **Std Dev** | 10.80% | 13,366,318 | 7.85% | 6.27 | 5.83 |
| **Min** | 0.50% | 117,542 | 16.77% | 10.8505 | 71.1924 |
| **25%** | 4.85% | 2,838,930 | 37.27% | 18.1124 | 76.0856 |
| **50% (Median)** | 9.65% | 9,732,417 | 40.39% | 23.6102 | 79.0193 |
| **75%** | 16.76% | 21,878,686 | 44.06% | 27.2784 | 85.2799 |
| **Max** | 75.85% | 59,433,759 | 69.69% | 33.7782 | 92.9376 |

---

## 3. Detailed Inspection: Complementary Dataset (`Unemployment in India.csv`)

Because the user requested inspection of `Area` (`Rural` vs `Urban`), the second file bundled in the Kaggle dataset was also inspected:

### 3.1 File & Dimensional Summary
- **Exact Filename:** `Unemployment in India.csv`
- **File Location:** `data/raw/Unemployment in India.csv`
- **Total Rows in File:** 768 (740 valid data rows + 28 trailing completely blank rows)
- **Total Columns:** 7
- **Raw Columns:**
  `['Region', ' Date', ' Frequency', ' Estimated Unemployment Rate (%)', ' Estimated Employed', ' Estimated Labour Participation Rate (%)', 'Area']`
- **Duplicate Rows:** 27 duplicate rows (all occurring within the 28 empty rows at rows 740–767)
- **Missing Values:** Exactly 28 null entries in every single column (corresponding to the trailing empty lines)

### 3.2 Key Attributes of `Unemployment in India.csv`
- **Date Range:** `31-05-2019` to `30-06-2020` (14 consecutive months, covering pre-COVID 2019 and early COVID 2020)
- **Unique States/Regions:** 28 (the 27 states above plus `Chandigarh`)
- **Area Feature:** Present with 2 unique values:
  - `Rural` (359 non-null records)
  - `Urban` (381 non-null records)
- **Frequency:** `Monthly` (recorded as `' Monthly'` and `'Monthly'`)

---

## 4. Analytical Readiness Assessment for CodeAlpha Task 2

| Required Analysis Type | Supported by Primary File (`Unemployment_Rate_upto_11_2020.csv`) | Supported by Complementary File (`Unemployment in India.csv`) | Assessment & Analytical Remarks |
|---|---|---|---|
| **Unemployment Trend Analysis** | **Yes** | **Yes** | Continuous monthly temporal sequence tracking unemployment rate, workforce employment, and labour participation across 2020. |
| **COVID-19 Impact Analysis** | **Yes** | **Yes** | Clear pre-lockdown (Jan–Feb 2020), peak lockdown disruption (April–May 2020), and post-lockdown reopening phases (June–Oct 2020). Unemployment spiked dramatically in April–May 2020 (reaching up to 75.85% in peak states). |
| **Monthly Pattern Analysis** | **Yes** | **Yes** | Consistent monthly intervals suitable for examining month-over-month rate variations and macro changes. |
| **State / Regional Comparison** | **Yes** | **Yes** | 27 states/UTs in the primary dataset with zonal aggregation (`Region.1`: East, North, Northeast, South, West) and geographic coordinates. |
| **Rural vs Urban Comparison** | **No** (aggregated at state level) | **Yes** (has explicit `Area` column: `Rural` vs `Urban`) | The primary file is state-level aggregate; however, the bundled `Unemployment in India.csv` specifically enables granular Rural vs Urban contrast analysis. |

---

## 5. Scope & Boundary Compliance Verification

To strictly honor all constraints specified for Step 1:
- [x] **No data cleaning or transformation applied:** Raw data remains bit-for-bit identical to the downloaded source.
- [x] **No duplicate removal applied:** Unmodified.
- [x] **No missing-value imputation applied:** Unmodified.
- [x] **No visualization generated:** No plot files or charts created.
- [x] **No machine-learning models trained:** Inspection only.
- [x] **No final conclusions or inferences formulated:** Only factual dataset characteristics documented.
- [x] **Work restricted strictly to Task 2 workspace:** No other tasks or external repositories touched.
