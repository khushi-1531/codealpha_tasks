"""
Exploratory Data Analysis (EDA) Module for Task 2: Unemployment Analysis.

This module provides functions to:
1. Compute overall descriptive statistics for unemployment, employment, and labour participation.
2. Analyze Pre-COVID vs COVID-period metrics and phases.
3. Compare Rural vs Urban employment and unemployment dynamics.
4. Assess state-level variations and identify top/bottom impacted states.
5. Evaluate monthly patterns and rigorously test the feasibility of seasonality analysis.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def compute_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes distribution metrics (mean, std, median, min, max, IQR)
    for numeric features.
    """
    numeric_cols = [
        "estimated_unemployment_rate_pct",
        "estimated_employed",
        "estimated_labour_participation_rate_pct",
    ]
    summary = df[numeric_cols].describe().T
    summary["median"] = df[numeric_cols].median()
    summary["iqr"] = summary["75%"] - summary["25%"]
    return summary[["count", "mean", "std", "min", "25%", "median", "75%", "max", "iqr"]]


def analyze_covid_impact(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyzes the impact of COVID-19 by comparing Pre-COVID baseline with COVID period,
    as well as granular COVID phases.
    """
    # 1. Period comparison (Pre-COVID vs COVID-Period)
    period_metrics = df.groupby("period")[
        ["estimated_unemployment_rate_pct", "estimated_employed", "estimated_labour_participation_rate_pct"]
    ].agg(["mean", "median", "std", "min", "max"])

    # 2. Phase-level comparison
    phase_order = [
        "Pre-COVID Baseline",
        "Early Lockdown (Mar 2020)",
        "Peak Lockdown (Apr-May 2020)",
        "Early Unlock (Jun 2020)",
    ]
    phase_metrics = df.groupby("covid_phase")[
        ["estimated_unemployment_rate_pct", "estimated_employed", "estimated_labour_participation_rate_pct"]
    ].agg(["mean", "median", "std"]).reindex(phase_order)

    # 3. Monthly aggregate totals (National Level Estimates)
    monthly_agg = df.groupby(["year_month", "date"]).agg(
        mean_unemployment_rate=("estimated_unemployment_rate_pct", "mean"),
        median_unemployment_rate=("estimated_unemployment_rate_pct", "median"),
        total_estimated_employed=("estimated_employed", "sum"),
        mean_labour_participation=("estimated_labour_participation_rate_pct", "mean"),
    ).reset_index().sort_values(by="date")

    return {
        "period_comparison": period_metrics,
        "phase_comparison": phase_metrics,
        "monthly_aggregates": monthly_agg,
    }


def analyze_rural_urban(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compares rural and urban employment dynamics across baseline and COVID periods.
    """
    area_overall = df.groupby("area")[
        ["estimated_unemployment_rate_pct", "estimated_employed", "estimated_labour_participation_rate_pct"]
    ].agg(["mean", "median", "std", "min", "max"])

    area_by_period = df.groupby(["period", "area"])[
        ["estimated_unemployment_rate_pct", "estimated_employed", "estimated_labour_participation_rate_pct"]
    ].agg(["mean", "median", "std"])

    monthly_area = df.pivot_table(
        index="year_month",
        columns="area",
        values="estimated_unemployment_rate_pct",
        aggfunc="mean"
    )

    return {
        "area_overall": area_overall,
        "area_by_period": area_by_period,
        "monthly_area": monthly_area,
    }


def analyze_state_disparities(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculates state-level performance, rankings, and peak-lockdown vulnerability.
    """
    state_overall = df.groupby("region").agg(
        mean_unemployment=("estimated_unemployment_rate_pct", "mean"),
        median_unemployment=("estimated_unemployment_rate_pct", "median"),
        max_unemployment=("estimated_unemployment_rate_pct", "max"),
        mean_employed=("estimated_employed", "mean"),
        mean_participation=("estimated_labour_participation_rate_pct", "mean"),
    ).sort_values(by="mean_unemployment", ascending=False)

    # Pre-COVID vs Peak Lockdown at State level
    pre_covid_states = df[df["period"] == "Pre-COVID"].groupby("region")["estimated_unemployment_rate_pct"].mean()
    peak_states = df[df["covid_phase"] == "Peak Lockdown (Apr-May 2020)"].groupby("region")["estimated_unemployment_rate_pct"].mean()

    state_comparison = pd.DataFrame({
        "pre_covid_mean": pre_covid_states,
        "peak_lockdown_mean": peak_states,
    })
    state_comparison["absolute_surge_pp"] = state_comparison["peak_lockdown_mean"] - state_comparison["pre_covid_mean"]
    state_comparison["relative_increase_pct"] = (
        (state_comparison["peak_lockdown_mean"] - state_comparison["pre_covid_mean"])
        / state_comparison["pre_covid_mean"]
    ) * 100
    state_comparison.sort_values(by="peak_lockdown_mean", ascending=False, inplace=True)

    return {
        "state_overall": state_overall,
        "state_lockdown_impact": state_comparison,
    }


def evaluate_seasonality_feasibility(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates whether the dataset supports meaningful seasonality modeling.
    """
    unique_months = df["year_month"].nunique()
    date_min = df["date"].min().strftime("%Y-%m-%d")
    date_max = df["date"].max().strftime("%Y-%m-%d")
    calendar_months = df["month_name"].value_counts().to_dict()

    # Feasibility decision:
    # Requires at least 2 full years (24+ months) of unconfounded data.
    supports_seasonality = False
    rationale = (
        f"The dataset covers only {unique_months} continuous months ({date_min} to {date_max}). "
        "Only two calendar months (May and June) appear twice (2019 and 2020), while all other 10 calendar "
        "months appear exactly once. Crucially, the 2020 observations occurred during historic nationwide "
        "COVID-19 lockdowns, creating an extreme exogenous shock rather than natural annual seasonality. "
        "Consequently, robust statistical seasonal decomposition or seasonality claims are not feasible or valid."
    )

    return {
        "unique_months_count": unique_months,
        "date_range": f"{date_min} to {date_max}",
        "calendar_month_counts": calendar_months,
        "supports_seasonality": supports_seasonality,
        "rationale": rationale,
    }


if __name__ == "__main__":
    from src.data_cleaning import clean_primary_dataset
    df = clean_primary_dataset()
    print("--- Overall Summary ---")
    print(compute_overall_summary(df))
    print("\n--- COVID Feasibility ---")
    print(evaluate_seasonality_feasibility(df))
