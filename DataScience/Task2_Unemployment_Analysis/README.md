# CodeAlpha Data Science Task 2 — Unemployment Analysis with Python

An exploratory data science project investigating unemployment dynamics, workforce employment contraction, and labour force participation rates in India before and during the COVID-19 pandemic.

---

## 1. Project Overview

This repository contains the end-to-end data analysis workflow for **CodeAlpha Data Science Task 2: Unemployment Analysis with Python**. The project delivers rigorous data cleaning, descriptive statistical profiling, event-based pandemic phase modeling, and high-resolution visualizations based on survey data collected by the Centre for Monitoring Indian Economy (CMIE).

---

## 2. Official Task Objective

- **Task Name:** Task 2 — Unemployment Analysis with Python
- **Core Objectives:**
  - Perform professional data acquisition, cleaning, and data integrity validation.
  - Analyze long-term unemployment rate trends across Indian states.
  - Assess the macroeconomic impact of the COVID-19 pandemic and national lockdowns.
  - Contrast rural versus urban employment dynamics.
  - Track changes in active employment workforce volume and labour force participation.
  - Evaluate whether seasonality can be meaningfully assessed from the available observations.
  - Provide modular, production-standard Python source code, automated test verification, and documentation.

---

## 3. Dataset Information

The data originates from the public Kaggle dataset **"Unemployment in India"**:

### 3.1 Primary Dataset: `Unemployment in India.csv`
- **Location:** `data/raw/Unemployment in India.csv` &rarr; Cleaned: `data/processed/unemployment_india_cleaned.csv`
- **Temporal Coverage:** May 31, 2019 to June 30, 2020 (14 consecutive months).
- **Baseline Structure:** Provides a robust **10-month Pre-COVID baseline** (May 2019 – Feb 2020) and **4 months of COVID pandemic disruption** (Mar 2020 – Jun 2020).
- **Geographic Granularity:** 28 States and Union Territories with explicit **Rural vs. Urban** segmentation.
- **Cleaned Dimensions:** 740 observations × 13 features (0 missing values, 0 duplicate rows).
- **Core Features:**
  - `region`: Name of Indian State / Union Territory.
  - `date`: Monthly survey reporting date (parsed as datetime).
  - `frequency`: Survey sampling frequency (`Monthly`).
  - `estimated_unemployment_rate_pct`: Estimated unemployment rate (%).
  - `estimated_employed`: Count of actively employed individuals (int64).
  - `estimated_labour_participation_rate_pct`: Labour force participation rate (%).
  - `area`: Geographic classification (`Rural` vs. `Urban`).

### 3.2 Secondary Dataset: `Unemployment_Rate_upto_11_2020.csv`
- **Location:** `data/raw/Unemployment_Rate_upto_11_2020.csv` &rarr; Cleaned: `data/processed/unemployment_rate_upto_11_2020_cleaned.csv`
- **Temporal Coverage:** January 31, 2020 to October 31, 2020 (10 months).
- **Role:** Serves as a complementary reference source for tracking macro-regional zones (`zone`: East, North, Northeast, South, West) and observing the extended unlock phase through late 2020 at the aggregate state level.
- **Architectural Decision:** Maintained as a separate validated dataset rather than merged into the primary file, preventing schema conflicts and double-counting of state-level aggregates.

---

## 4. Methodology

1. **Data Acquisition:** Obtained official public Kaggle datasets without manual tampering; original raw files preserved read-only in `data/raw/`.
2. **Data Cleaning & Validation:**
   - Stripped leading/trailing whitespaces across column names and string values.
   - Identified and eliminated 28 all-NaN rows (14 separating Rural from Urban blocks and 14 trailing rows).
   - Cast `estimated_employed` to integer and parsed `date` using `%d-%m-%Y`.
   - Engineered deterministic analytical labels (`year_month`, `period`, `covid_phase`).
3. **Exploratory Data Analysis (EDA):**
   - Descriptive statistical analysis across unemployment, employment volume, and labour participation.
   - Group-based comparisons: Pre-COVID vs. COVID period, pandemic phases, rural vs. urban areas, and interstate rankings.
4. **COVID-19 Disruption Analysis:**
   - Evaluated changes between pre-COVID baseline (May 2019 – Feb 2020) and COVID period (Mar 2020 – Jun 2020).
   - Granular breakdown across Early Lockdown (Mar 2020), Peak Lockdown (Apr–May 2020), and Early Unlock (Jun 2020).
5. **Workforce Volume & Participation Analysis:**
   - Summed state employment to assess total active workforce contraction.
   - Assessed decline in Labour Force Participation Rate (LFPR) indicative of discouraged workers.
6. **Visualization:**
   - Generated 8 publication-ready 300 DPI figures using Matplotlib and Seaborn with custom color palettes and annotations.

---

## 5. Key Verified Findings

The analysis indicates the following factual observations from the cleaned dataset:

### 5.1 COVID-19 Impact & Lockdown Shock
- **Pre-COVID Baseline Mean Unemployment Rate (May 2019 – Feb 2020):** **9.51%** (Median: 7.12%).
- **COVID-Period Mean Unemployment Rate (Mar 2020 – Jun 2020):** **17.77%** (Median: 14.52%).
- **Absolute Surge:** **+8.26 percentage points**.
- **Relative Increase:** **+86.9%** increase over baseline.
- **Peak Lockdown Mean (Apr–May 2020):** **24.26%** (Median: 19.96%), with May 2020 reaching a national monthly average peak of **24.88%**.
- **Early Unlock (Jun 2020):** Rebounded to **11.90%** following Unlock 1.0 guidelines.

### 5.2 National Workforce Employment Contraction
- The data indicates total active workforce employment dropped from **403.01 Million** in February 2020 down to **269.45 Million** in April 2020.
- This represents an immediate contraction of **-133.56 Million employed persons (-33.1%)** during the height of national lockdown measures.
- By June 2020, employment partially rebounded to **369.35 Million**.

### 5.3 Labour Force Participation Rate (LFPR)
- Mean LFPR declined from **43.89%** in the pre-COVID baseline to **35.79%** in April 2020 and **36.82%** during the April–May 2020 peak lockdown.
- The parallel decline in both employment and participation highlights the exit of discouraged job-seekers unable to seek employment during travel restrictions.

### 5.4 Rural vs. Urban Dynamics
- **Pre-COVID Baseline:** Urban unemployment (10.84%) exceeded rural unemployment (8.09%) by a structural premium of +2.75 percentage points.
- **COVID-Period:** Rural unemployment surged to **16.18%** (+100.0% relative increase, doubling its baseline rate) due to non-farm enterprise closures and reverse migration. Urban unemployment rose to **19.28%** (+77.9% relative increase) reflecting disruptions in service, trade, and construction sectors.

### 5.5 State-Level Disparities & Peak Impact
- **States with Highest Overall Average Rates:** Tripura (28.35%), Haryana (26.28%), Jharkhand (20.58%), Bihar (18.92%), Himachal Pradesh (18.54%).
- **States with Lowest Overall Average Rates:** Meghalaya (4.80%), Odisha (5.66%), Assam (6.43%), Uttarakhand (6.58%), Gujarat (6.66%).
- **Hardest Hit During Peak Lockdown (April–May 2020 Mean):**
  - Puducherry: 75.42% (up from 2.05% pre-COVID)
  - Jharkhand: 57.12% (up from 13.91% pre-COVID)
  - Bihar: 47.25% (up from 13.25% pre-COVID)
  - Tamil Nadu: 40.86% (up from 3.51% pre-COVID)
  - Haryana: 40.30% (up from 23.48% pre-COVID)

---

## 6. Seasonality Assessment & Analytical Limitations

- **Verdict on Seasonality:** **Annual seasonality cannot be meaningfully assessed on this dataset.**
- **Technical Grounds:**
  1. *Insufficient Time Horizon:* The dataset covers only 14 continuous months (May 2019 to June 2020). Isolating recurring calendar seasonality requires at least 24 to 36 months (2–3 complete annual cycles).
  2. *Single Pair of Overlapping Months:* Only May and June appear twice (2019 and 2020); the other 10 calendar months appear exactly once.
  3. *Confounded by Pandemic Shock:* The divergence between May 2019 (8.87%) and May 2020 (24.88%) was driven by nationwide lockdown measures, not natural annual calendar variation. Decomposing seasonality from this time span would misattribute the pandemic shock to seasonal effects.
- **Causality Caveat:** All findings describe observed associations within the CMIE survey data. The analysis does not claim causal proof, recognizing that external economic factors and state-level policy responses co-occurred with national containment measures.

---

## 7. Project Directory Structure

```text
Task2_Unemployment_Analysis/
├── .gitignore                          # Git exclusion rules for clean version control
├── README.md                           # Comprehensive project overview and documentation
├── requirements.txt                    # Minimal, production dependencies
├── data/
│   ├── raw/                            # Original, unedited Kaggle CSV datasets
│   │   ├── Unemployment in India.csv
│   │   └── Unemployment_Rate_upto_11_2020.csv
│   └── processed/                      # Cleaned and validated datasets
│       ├── unemployment_india_cleaned.csv
│       └── unemployment_rate_upto_11_2020_cleaned.csv
├── reports/
│   ├── dataset_inspection_report.md    # Initial raw data audit report
│   ├── data_cleaning_report.md         # Comprehensive cleaning pipeline report
│   ├── eda_report.md                   # Full exploratory analysis & findings report
│   └── figures/                        # High-resolution (300 DPI) analysis figures
│       ├── 01_overall_unemployment_trend.png
│       ├── 02_covid_impact_unemployment_trend.png
│       ├── 03_state_unemployment_comparison.png
│       ├── 04_rural_vs_urban_comparison.png
│       ├── 05_monthly_unemployment_patterns.png
│       ├── 06_employment_trend.png
│       ├── 07_labour_participation_trend.png
│       └── 08_covid_phase_comparison.png
├── src/                                # Modular, reusable Python source code
│   ├── __init__.py
│   ├── data_cleaning.py                # Data loading, cleaning, and transformation
│   ├── exploratory_analysis.py         # Statistical computations and metric aggregations
│   └── visualization.py                # Matplotlib/Seaborn visualization pipeline
└── tests/                              # Automated test suite
    ├── __init__.py
    └── test_data_cleaning.py           # Integrity, type, and schema test cases
```

---

## 8. Technologies Used

- **Python 3.13**
- **Pandas:** Data manipulation, cleaning, and aggregation
- **NumPy:** Numeric operations and vectorization
- **Matplotlib & Seaborn:** Publication-quality visual generation
- **Pytest:** Automated testing and data validation

---

## 9. Installation & Usage Guide

### 9.1 Environment Setup
Clone the repository and navigate to the Task 2 workspace:
```bash
cd "DataScience/Task2_Unemployment_Analysis"
pip install -r requirements.txt
```

### 9.2 Running the Pipeline
Run data cleaning:
```bash
python src/data_cleaning.py
```

Run exploratory analysis:
```bash
python src/exploratory_analysis.py
```

Generate visualizations:
```bash
python src/visualization.py
```

### 9.3 Running Automated Tests
Execute the full test suite via pytest:
```bash
python -m pytest -v
```

---

## 10. Reports & Figures Reference

- **Initial Inspection:** [`reports/dataset_inspection_report.md`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/dataset_inspection_report.md)
- **Cleaning Documentation:** [`reports/data_cleaning_report.md`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/data_cleaning_report.md)
- **EDA & Findings:** [`reports/eda_report.md`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/eda_report.md)
- **Generated Figures:** [`reports/figures/`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/)
