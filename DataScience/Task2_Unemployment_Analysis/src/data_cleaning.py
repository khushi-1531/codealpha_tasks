"""
Data Cleaning Module for CodeAlpha Task 2: Unemployment Analysis with Python.

This module handles:
1. Loading raw CSV datasets.
2. Cleaning header names, removing leading/trailing spaces.
3. Handling trailing/separator blank rows.
4. Parsing and standardizing date formats.
5. Converting appropriate columns to strict numeric formats.
6. Value range validation and integrity verification.
7. Deriving structured temporal and COVID-phase analytical labels.
8. Exporting clean, production-ready datasets to data/processed/.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np


def clean_primary_dataset(
    raw_path: str = "data/raw/Unemployment in India.csv",
    output_path: str = "data/processed/unemployment_india_cleaned.csv"
) -> pd.DataFrame:
    """
    Cleans the primary dataset 'Unemployment in India.csv'.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV file.
    output_path : str, optional
        Path where the cleaned CSV will be written.

    Returns
    -------
    pd.DataFrame
        The cleaned and validated DataFrame (740 rows, 11 columns).
    """
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    # 1. Load raw data
    raw_df = pd.read_csv(raw_file)

    # 2. Identify and remove all-NaN rows (14 blank rows between sections + 14 at EOF = 28 total)
    blank_mask = raw_df.isnull().all(axis=1)
    dropped_blank_count = int(blank_mask.sum())
    cleaned_df = raw_df[~blank_mask].copy()

    # 3. Clean and standardize column names (lowercase, stripped, snake_case)
    col_mapping = {
        col: col.strip()
        .lower()
        .replace(" (%)", "_pct")
        .replace(" ", "_")
        for col in cleaned_df.columns
    }
    cleaned_df.rename(columns=col_mapping, inplace=True)

    # Expected column mapping:
    # 'region', 'date', 'frequency', 'estimated_unemployment_rate_pct',
    # 'estimated_employed', 'estimated_labour_participation_rate_pct', 'area'

    # 4. Clean string categorical columns
    string_cols = ["region", "frequency", "area"]
    for col in string_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    # Normalize frequency values (' Monthly' / 'Monthly' -> 'Monthly')
    cleaned_df["frequency"] = cleaned_df["frequency"].replace({"Monthly": "Monthly"})

    # 5. Parse dates strictly using '%d-%m-%Y'
    cleaned_df["date"] = cleaned_df["date"].astype(str).str.strip()
    cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], format="%d-%m-%Y")

    # 6. Type cast numeric columns
    cleaned_df["estimated_unemployment_rate_pct"] = cleaned_df[
        "estimated_unemployment_rate_pct"
    ].astype(float)
    cleaned_df["estimated_employed"] = cleaned_df["estimated_employed"].round().astype(np.int64)
    cleaned_df["estimated_labour_participation_rate_pct"] = cleaned_df[
        "estimated_labour_participation_rate_pct"
    ].astype(float)

    # 7. Add derived analytical columns (temporal & COVID phase)
    cleaned_df["year"] = cleaned_df["date"].dt.year
    cleaned_df["month"] = cleaned_df["date"].dt.month
    cleaned_df["month_name"] = cleaned_df["date"].dt.strftime("%b")
    cleaned_df["year_month"] = cleaned_df["date"].dt.strftime("%Y-%m")

    # COVID period classification:
    # Pre-COVID: May 2019 - Feb 2020
    # COVID Period: Mar 2020 - Jun 2020 (Lockdown announced Mar 24, 2020)
    cleaned_df["period"] = np.where(
        cleaned_df["date"] < pd.Timestamp("2020-03-01"),
        "Pre-COVID",
        "COVID-Period"
    )

    def assign_covid_phase(dt: pd.Timestamp) -> str:
        if dt < pd.Timestamp("2020-03-01"):
            return "Pre-COVID Baseline"
        elif dt == pd.Timestamp("2020-03-31"):
            return "Early Lockdown (Mar 2020)"
        elif dt in [pd.Timestamp("2020-04-30"), pd.Timestamp("2020-05-31")]:
            return "Peak Lockdown (Apr-May 2020)"
        else:
            return "Early Unlock (Jun 2020)"

    cleaned_df["covid_phase"] = cleaned_df["date"].apply(assign_covid_phase)

    # 8. Sort values logically (chronological and region)
    cleaned_df.sort_values(by=["date", "region", "area"], inplace=True)
    cleaned_df.reset_index(drop=True, inplace=True)

    # 9. Rigorous Integrity Checks
    assert len(cleaned_df) == 740, f"Expected 740 rows, got {len(cleaned_df)}"
    assert cleaned_df.isnull().sum().sum() == 0, "Cleaned dataset contains unexpected nulls!"
    assert cleaned_df.duplicated().sum() == 0, "Cleaned dataset contains duplicate rows!"
    assert (cleaned_df["estimated_unemployment_rate_pct"] >= 0).all(), "Negative unemployment rates found!"
    assert (cleaned_df["estimated_unemployment_rate_pct"] <= 100).all(), "Unemployment rate > 100% found!"
    assert (cleaned_df["estimated_employed"] > 0).all(), "Non-positive employment count found!"
    assert (cleaned_df["estimated_labour_participation_rate_pct"] >= 0).all(), "Negative participation rate found!"
    assert (cleaned_df["estimated_labour_participation_rate_pct"] <= 100).all(), "Participation rate > 100% found!"

    # 10. Save to processed folder
    if output_path:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)

    return cleaned_df


def clean_secondary_dataset(
    raw_path: str = "data/raw/Unemployment_Rate_upto_11_2020.csv",
    output_path: str = "data/processed/unemployment_rate_upto_11_2020_cleaned.csv"
) -> pd.DataFrame:
    """
    Cleans the secondary dataset 'Unemployment_Rate_upto_11_2020.csv'.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV file.
    output_path : str, optional
        Path where the cleaned CSV will be written.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame (267 rows, 12 columns).
    """
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    raw_df = pd.read_csv(raw_file)

    # Column mapping
    col_mapping = {
        col: col.strip()
        .lower()
        .replace(" (%)", "_pct")
        .replace("region.1", "zone")
        .replace(" ", "_")
        for col in raw_df.columns
    }
    cleaned_df = raw_df.rename(columns=col_mapping).copy()

    # Strip string columns
    for col in ["region", "frequency", "zone"]:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    # Parse dates
    cleaned_df["date"] = cleaned_df["date"].astype(str).str.strip()
    cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], format="%d-%m-%Y")

    # Type casting
    cleaned_df["estimated_unemployment_rate_pct"] = cleaned_df[
        "estimated_unemployment_rate_pct"
    ].astype(float)
    cleaned_df["estimated_employed"] = cleaned_df["estimated_employed"].astype(np.int64)
    cleaned_df["estimated_labour_participation_rate_pct"] = cleaned_df[
        "estimated_labour_participation_rate_pct"
    ].astype(float)
    cleaned_df["longitude"] = cleaned_df["longitude"].astype(float)
    cleaned_df["latitude"] = cleaned_df["latitude"].astype(float)

    # Temporal features
    cleaned_df["year"] = cleaned_df["date"].dt.year
    cleaned_df["month"] = cleaned_df["date"].dt.month
    cleaned_df["month_name"] = cleaned_df["date"].dt.strftime("%b")
    cleaned_df["year_month"] = cleaned_df["date"].dt.strftime("%Y-%m")

    # Period
    cleaned_df["period"] = np.where(
        cleaned_df["date"] < pd.Timestamp("2020-03-01"),
        "Pre-COVID (Jan-Feb 2020)",
        "COVID-Period (Mar-Oct 2020)"
    )

    cleaned_df.sort_values(by=["date", "region"], inplace=True)
    cleaned_df.reset_index(drop=True, inplace=True)

    # Integrity checks
    assert len(cleaned_df) == 267, f"Expected 267 rows, got {len(cleaned_df)}"
    assert cleaned_df.isnull().sum().sum() == 0, "Unexpected nulls in secondary dataset!"

    if output_path:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)

    return cleaned_df


if __name__ == "__main__":
    df_primary = clean_primary_dataset()
    print(f"Cleaned Primary Dataset: {df_primary.shape}")
    print(df_primary.head(2))

    df_secondary = clean_secondary_dataset()
    print(f"Cleaned Secondary Dataset: {df_secondary.shape}")
    print(df_secondary.head(2))
