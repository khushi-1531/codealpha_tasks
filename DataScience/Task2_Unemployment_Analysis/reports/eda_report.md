# Task 2 — Unemployment Analysis: Exploratory Data Analysis (EDA) Report

**Project:** CodeAlpha Data Science Task 2: Unemployment Analysis with Python  
**Working Directory:** `C:\Users\khushi\OneDrive\Desktop\codealpha interniship\codealpha_tasks\DataScience\Task2_Unemployment_Analysis`  
**Dataset Analyzed:** Cleaned Primary Dataset (`data/processed/unemployment_india_cleaned.csv`)  
**Observations:** 740 records across 28 Indian States & Union Territories (May 2019 – June 2020)  
**Date:** 2026-09-30  

---

## 1. Executive Summary

This report delivers a thorough exploratory analysis of India's unemployment dynamics, workforce employment, and labour participation rates over a 14-month window spanning from **May 2019 to June 2020**. 

The dataset captures two distinct economic environments:
1. **Pre-COVID Baseline (May 2019 – Feb 2020):** 10 months of standard economic conditions.
2. **COVID-19 Disruption Period (Mar 2020 – Jun 2020):** 4 months encompassing the initial lockdown shock, peak nationwide economic shutdown, and initial unlock phase.

All visualizations referenced in this report are saved at high resolution (300 DPI) under [`reports/figures/`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/).

---

## 2. Overall Descriptive Statistics

Across the entire 740 observations from May 2019 to June 2020:

| Metric | Estimated Unemployment Rate (%) | Estimated Employed (Persons) | Estimated Labour Participation Rate (%) |
|---|---|---|---|
| **Count** | 740 | 740 | 740 |
| **Mean** | 11.79% | 7,204,460 | 42.63% |
| **Std Dev** | 10.72% | 8,087,988 | 8.11% |
| **Minimum** | 0.00% | 49,420 | 13.33% |
| **25th Percentile (Q1)** | 4.66% | 1,190,404 | 38.06% |
| **50th Percentile (Median)** | 8.35% | 4,744,178 | 41.16% |
| **75th Percentile (Q3)** | 15.89% | 11,275,490 | 45.50% |
| **Maximum** | 76.74% | 45,777,509 | 72.57% |
| **Interquartile Range (IQR)** | 11.23% | 10,085,086 | 7.44% |

**Key Distribution Characteristics:**
- The unemployment rate distribution exhibits significant right skewness (Mean = 11.79% vs Median = 8.35%), driven by acute spikes during the April–May 2020 lockdowns where multiple states experienced unemployment rates exceeding 40% to 75%.
- Employment volume reflects wide state population disparities, ranging from small Union Territories (e.g., Puducherry, Chandigarh) to heavily populated states (e.g., Uttar Pradesh, Maharashtra).

---

## 3. Overall Unemployment Rate Trend Over Time

- **Visual Reference:** [01_overall_unemployment_trend.png](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/01_overall_unemployment_trend.png)

```
Monthly Average Unemployment Rate Trajectory (India):
May 2019:  8.87%
Jun 2019:  9.30%
Jul 2019:  9.03%
Aug 2019:  9.64%
Sep 2019:  9.05%
Oct 2019:  9.90%
Nov 2019:  9.87%
Dec 2019:  9.50%
Jan 2020:  9.95%
Feb 2020:  9.96%
Mar 2020: 10.70% (Lockdown declared Mar 24, 2020)
Apr 2020: 23.64% (Peak lockdown disruption)
May 2020: 24.88% (All-time peak national monthly average)
Jun 2020: 11.90% (Early unlock recovery)
```

**Observations:**
1. **Pre-COVID Stability:** Throughout May 2019 to February 2020, national monthly average unemployment hovered consistently in a narrow band between **8.87% and 9.96%**.
2. **Lockdown Shock:** Following the nationwide lockdown announced on March 24, 2020, average unemployment jumped to **10.70% in March**, then surged violently to **23.64% in April** and peaked at **24.88% in May 2020**.
3. **Phased Recovery:** With the introduction of Unlock 1.0 guidelines in June 2020, the rate rapidly corrected to **11.90%**, demonstrating prompt resumption of informal and agricultural labor, although remaining ~2.4 percentage points above the pre-COVID baseline.

---

## 4. COVID-19 Impact Analysis

- **Visual Reference:** [02_covid_impact_unemployment_trend.png](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/02_covid_impact_unemployment_trend.png)
- **Visual Reference:** [08_covid_phase_comparison.png](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/08_covid_phase_comparison.png)

### 4.1 Pre-COVID Baseline vs. COVID-Period Summary

| Period | Record Count | Mean Unemployment (%) | Median Unemployment (%) | Std Dev (%) | Min (%) | Max (%) |
|---|---|---|---|---|---|---|
| **Pre-COVID Baseline** (May 2019 – Feb 2020) | 536 | 9.51% | 7.12% | 7.36% | 0.00% | 34.69% |
| **COVID-Period** (Mar 2020 – Jun 2020) | 204 | 17.77% | 14.52% | 15.03% | 0.00% | 76.74% |
| **Absolute Change** | — | **+8.26 pp** | **+7.40 pp** | +7.67 pp | 0.00 pp | +42.05 pp |
| **Relative Percentage Change** | — | **+86.9%** | **+103.9%** | — | — | — |

### 4.2 Breakdown by Pandemic Phases

| Pandemic Phase | Observation Months | Records | Mean Unemp. (%) | Median Unemp. (%) | Mean Employed | Mean Labour Part. (%) |
|---|---|---|---|---|---|---|
| **Pre-COVID Baseline** | May 2019 – Feb 2020 | 536 | **9.51%** | 7.12% | 7,466,028 | **43.89%** |
| **Early Lockdown** | March 2020 | 52 | **10.70%** | 8.53% | 7,516,581 | **43.08%** |
| **Peak Lockdown** | April – May 2020 | 102 | **24.26%** | 19.96% | 5,581,341 | **36.82%** |
| **Early Unlock** | June 2020 | 50 | **11.90%** | 10.35% | 7,387,009 | **40.55%** |

**Statistical Findings:**
- During the **Peak Lockdown** (April–May 2020), the mean unemployment rate rose by **+14.75 percentage points** (+155.1% relative surge) over the baseline.
- Concurrently, mean labour force participation fell by **7.07 percentage points** (from 43.89% to 36.82%), confirming an acute "discouraged worker" phenomenon where millions exited active job-seeking due to strict immobility constraints.

---

## 5. Rural vs. Urban Comparative Analysis

- **Visual Reference:** [04_rural_vs_urban_comparison.png](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/04_rural_vs_urban_comparison.png)

### 5.1 Overall Area Comparison

| Area | Record Count | Mean Unemployment (%) | Median Unemployment (%) | Std Dev (%) | Min (%) | Max (%) |
|---|---|---|---|---|---|---|
| **Rural** | 359 | 10.32% | 6.76% | 10.04% | 0.00% | 74.51% |
| **Urban** | 381 | 13.17% | 9.97% | 11.17% | 0.00% | 76.74% |

### 5.2 Rural vs. Urban Disparity Across Periods

| Period | Area | Mean Unemployment (%) | Median Unemployment (%) | Std Dev (%) |
|---|---|---|---|---|
| **Pre-COVID Baseline** | Rural | 8.09% | 5.79% | 5.92% |
|  | Urban | 10.84% | 7.88% | 8.32% |
|  | *Urban Premium (Urban - Rural)* | *+2.75 pp* | *+2.09 pp* | — |
| **COVID-Period** | Rural | 16.18% | 12.50% | 15.01% |
|  | Urban | 19.28% | 15.22% | 14.93% |
|  | *Urban Premium (Urban - Rural)* | *+3.10 pp* | *+2.72 pp* | — |

**Comparative Insights:**
1. **Structural Urban Premium:** Urban unemployment rates consistently exceeded rural rates across both normal and crisis months (pre-COVID urban mean was 10.84% vs 8.09% rural).
2. **Acute Rural Shock:** Rural unemployment virtually doubled during the pandemic (+8.09 pp, or +100.0% relative increase), driven by reverse migration of urban daily wage laborers returning to rural home states and rural non-farm enterprise closures.
3. **Severe Urban Disruption:** Urban unemployment peaked at 19.28% during the COVID period, reflecting the complete vulnerability of service-sector, retail, and construction workers under urban containment zones.

---

## 6. National Workforce Employment Volume & Labour Participation Trends

- **Visual Reference:** [06_employment_trend.png](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/06_employment_trend.png)
- **Visual Reference:** [07_labour_participation_trend.png](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/07_labour_participation_trend.png)

### 6.1 Total National Employed Workforce Volume (Sum across reporting states)
- **Pre-COVID Level (Feb 2020):** **403.01 Million** employed persons.
- **Peak Lockdown Shock (April 2020):** **269.45 Million** employed persons.
- **Immediate Workforce Contraction:** **-133.56 Million employed individuals** (-33.1% contraction in active employment).
- **Subsequent Rebound:**
  - May 2020: 299.85 Million
  - June 2020: 369.35 Million (8.3% below February 2020 baseline)

### 6.2 Labour Force Participation Rate (LFPR) Trajectory
- Pre-COVID Mean: **43.89%** (Median: 42.47%)
- April 2020 Mean: **35.79%** (sharpest dip)
- May 2020 Mean: **37.85%**
- June 2020 Mean: **40.55%**

---

## 7. State-Level Disparities & Regional Vulnerability

- **Visual Reference:** [03_state_unemployment_comparison.png](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/03_state_unemployment_comparison.png)

### 7.1 States with Highest Overall Average Unemployment Rate (14-Month Mean)
1. **Tripura:** 28.35% (Median: 27.31%, Max: 43.64%)
2. **Haryana:** 26.28% (Median: 25.06%, Max: 46.89%)
3. **Jharkhand:** 20.58% (Median: 17.29%, Max: 70.17%)
4. **Bihar:** 18.92% (Median: 15.01%, Max: 58.77%)
5. **Himachal Pradesh:** 18.54% (Median: 18.36%, Max: 50.00%)

### 7.2 States with Lowest Overall Average Unemployment Rate
1. **Meghalaya:** 4.80% (Median: 3.73%, Max: 17.39%)
2. **Odisha:** 5.66% (Median: 3.87%, Max: 24.48%)
3. **Assam:** 6.43% (Median: 5.44%, Max: 11.17%)
4. **Uttarakhand:** 6.58% (Median: 5.56%, Max: 17.36%)
5. **Gujarat:** 6.66% (Median: 5.42%, Max: 25.94%)

### 7.3 Top 5 Hardest Hit States During Peak Lockdown (April–May 2020 Average)

| State / UT | Pre-COVID Mean (%) | Peak Lockdown Mean (%) | Absolute Surge (pp) | Relative Increase (%) |
|---|---|---|---|---|
| **Puducherry** | 2.05% | **75.42%** | **+73.37 pp** | +3579.0% |
| **Jharkhand** | 13.91% | **57.12%** | **+43.21 pp** | +310.6% |
| **Bihar** | 13.25% | **47.25%** | **+34.00 pp** | +256.6% |
| **Tamil Nadu** | 3.51% | **40.86%** | **+37.35 pp** | +1064.1% |
| **Haryana** | 23.48% | **40.30%** | **+16.82 pp** | +71.6% |

**Key State Findings:**
- Puducherry and Tamil Nadu, despite having very low pre-COVID unemployment (<4%), experienced near-total cessation of industrial and service employment during lockdown, surging past 40%–75%.
- Eastern migrant-sending states (Bihar and Jharkhand) experienced acute dual crises: severe local shutdowns compounded by sudden return of hundreds of thousands of unabsorbed migrant laborers.

---

## 8. Monthly Patterns & Seasonality Assessment

- **Visual Reference:** [05_monthly_unemployment_patterns.png](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/05_monthly_unemployment_patterns.png)

### 8.1 Seasonality Feasibility Evaluation
A critical question in time-series analysis is whether repeating annual seasonality can be identified.

**Analytical Verdict:** **Annual seasonality CANNOT be meaningfully or reliably assessed on this dataset.**

**Technical Grounds:**
1. **Insufficient Time Horizon:** The primary dataset spans strictly 14 continuous months (May 2019 to June 2020). Isolating true seasonal periodicities requires a minimum of 24–36 continuous months (2–3 complete annual cycles).
2. **Lack of Repeating Month Pairs:** Only two calendar months appear more than once (May 2019 vs. May 2020, and June 2019 vs. June 2020). The remaining 10 calendar months appear exactly once.
3. **Severe Exogenous Shock Confounding:** Comparing May 2019 (8.87%) with May 2020 (24.88%) does not reflect natural crop/calendar seasonality; it reflects a once-in-a-century pandemic lockdown that shut down economic activity. Attempting to fit a seasonal decomposition model (e.g., STL or ARIMA seasonality) to this data would misattribute pandemic shocks to calendar effects.

---

## 9. Comprehensive Figure Index

| Figure File | Description | Key Insight Visualized |
|---|---|---|
| [`01_overall_unemployment_trend.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/01_overall_unemployment_trend.png) | Overall Monthly Trend | National trajectory with 25th–75th percentile cross-state confidence band and lockdown annotation. |
| [`02_covid_impact_unemployment_trend.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/02_covid_impact_unemployment_trend.png) | COVID-19 Period Impact | Contrast between 10-month baseline (9.51%) and lockdown surge (peaking at 24.88%). |
| [`03_state_unemployment_comparison.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/03_state_unemployment_comparison.png) | State-Level Disparities | Paired horizontal bar chart comparing Pre-COVID vs. Peak Lockdown across 28 states. |
| [`04_rural_vs_urban_comparison.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/04_rural_vs_urban_comparison.png) | Rural vs. Urban Dynamics | Dual panel: monthly timeline trajectory + boxplot distributions by period. |
| [`05_monthly_unemployment_patterns.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/05_monthly_unemployment_patterns.png) | Monthly Distributions | Boxplot distribution across all 14 months highlighting cross-state variance. |
| [`06_employment_trend.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/06_employment_trend.png) | National Workforce Volume | Drop from 403M to 269M employed individuals (-33.1%) during April 2020. |
| [`07_labour_participation_trend.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/07_labour_participation_trend.png) | Labour Participation Trajectory | Trajectory of labour force participation showing drop during April–May 2020. |
| [`08_covid_phase_comparison.png`](file:///c:/Users/khushi/OneDrive/Desktop/codealpha%20interniship/codealpha_tasks/DataScience/Task2_Unemployment_Analysis/reports/figures/08_covid_phase_comparison.png) | Macroeconomic Phase Dashboard | Side-by-side bar plots of Unemployment, Employment, and Participation across 4 pandemic phases. |

---

## 10. Analytical Limitations & Methodological Caveats

1. **Observational Sample:** The dataset derives from CMIE household survey estimates. While robust, sampling variations and non-response during strict mobility lockdowns may slightly skew extreme tails.
2. **No Causal Exaggeration:** Observed changes in April–May 2020 align with nationwide lockdown restrictions, but individual state variations are also influenced by state-specific economic structures (manufacturing vs. agriculture, migrant labor dependence, informal economy share).
3. **No Machine Learning Modeling:** Per task objectives, no predictive ML modeling or synthetic imputation was performed. The analysis is strictly exploratory and factual.
