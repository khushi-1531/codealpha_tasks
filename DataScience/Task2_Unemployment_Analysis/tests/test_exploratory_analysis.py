"""
Unit tests for Task 2 Exploratory Analysis functions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import clean_primary_dataset
from src.exploratory_analysis import (
    compute_overall_summary,
    analyze_covid_impact,
    analyze_rural_urban,
    analyze_state_disparities,
    evaluate_seasonality_feasibility,
)


@pytest.fixture(scope="module")
def cleaned_df():
    return clean_primary_dataset(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "Unemployment in India.csv"),
        output_path=None
    )


def test_compute_overall_summary(cleaned_df):
    summary = compute_overall_summary(cleaned_df)
    assert isinstance(summary, pd.DataFrame)
    assert "mean" in summary.columns
    assert "median" in summary.columns
    assert "iqr" in summary.columns
    # Check that mean unemployment rate is ~11.79
    unemp_mean = summary.loc["estimated_unemployment_rate_pct", "mean"]
    assert pytest.approx(unemp_mean, 0.01) == 11.79


def test_analyze_covid_impact(cleaned_df):
    res = analyze_covid_impact(cleaned_df)
    assert "period_comparison" in res
    assert "phase_comparison" in res
    assert "monthly_aggregates" in res

    period_df = res["period_comparison"]
    assert "Pre-COVID" in period_df.index
    assert "COVID-Period" in period_df.index

    # Pre-COVID mean should be ~9.51%
    pre_mean = period_df.loc["Pre-COVID", ("estimated_unemployment_rate_pct", "mean")]
    assert pytest.approx(pre_mean, 0.01) == 9.51

    # COVID mean should be ~17.77%
    covid_mean = period_df.loc["COVID-Period", ("estimated_unemployment_rate_pct", "mean")]
    assert pytest.approx(covid_mean, 0.01) == 17.77


def test_analyze_rural_urban(cleaned_df):
    res = analyze_rural_urban(cleaned_df)
    assert "area_overall" in res
    assert "area_by_period" in res
    assert "monthly_area" in res

    area_df = res["area_overall"]
    assert "Rural" in area_df.index
    assert "Urban" in area_df.index

    rural_mean = area_df.loc["Rural", ("estimated_unemployment_rate_pct", "mean")]
    urban_mean = area_df.loc["Urban", ("estimated_unemployment_rate_pct", "mean")]
    assert pytest.approx(rural_mean, 0.01) == 10.32
    assert pytest.approx(urban_mean, 0.01) == 13.17


def test_analyze_state_disparities(cleaned_df):
    res = analyze_state_disparities(cleaned_df)
    assert "state_overall" in res
    assert "state_lockdown_impact" in res

    state_comp = res["state_lockdown_impact"]
    assert "absolute_surge_pp" in state_comp.columns
    assert "peak_lockdown_mean" in state_comp.columns
    assert len(state_comp) == 28


def test_evaluate_seasonality_feasibility(cleaned_df):
    res = evaluate_seasonality_feasibility(cleaned_df)
    assert res["supports_seasonality"] is False
    assert res["unique_months_count"] == 14
    assert "insufficient" in res["rationale"].lower() or "not feasible" in res["rationale"].lower()
