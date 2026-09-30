"""
Unit and Integration Tests for Task 2 Data Cleaning & Integrity.

Tests verify:
1. Raw dataset existence and preservation.
2. Cleaned dataset loading and dimension consistency.
3. Column schema and naming standards.
4. Strict datetime parsing and temporal integrity.
5. Numeric type conversions and realistic value bounds.
6. Absolute absence of unexpected null/missing values.
7. Zero duplicate records.
8. Categorical features integrity (area, frequency, period).
9. Secondary dataset cleaning integrity.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import clean_primary_dataset, clean_secondary_dataset


def test_raw_files_exist_and_unmodified():
    raw_primary = PROJECT_ROOT / "data" / "raw" / "Unemployment in India.csv"
    raw_secondary = PROJECT_ROOT / "data" / "raw" / "Unemployment_Rate_upto_11_2020.csv"

    assert raw_primary.exists(), "Primary raw dataset missing!"
    assert raw_secondary.exists(), "Secondary raw dataset missing!"

    # Ensure raw files are not empty and contain original row volume
    with open(raw_primary, "r", encoding="utf-8") as f:
        lines_p = f.readlines()
    with open(raw_secondary, "r", encoding="utf-8") as f:
        lines_s = f.readlines()

    assert len(lines_p) == 769, f"Raw primary line count changed: {len(lines_p)} (expected 769)"
    assert len(lines_s) == 268, f"Raw secondary line count changed: {len(lines_s)} (expected 268)"


def test_clean_primary_dataset_loading_and_dimensions():
    df = clean_primary_dataset(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "Unemployment in India.csv"),
        output_path=None
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 740, f"Expected exactly 740 cleaned rows, got {df.shape[0]}"
    assert df.shape[1] >= 11, f"Expected at least 11 columns, got {df.shape[1]}"


def test_expected_important_columns():
    df = clean_primary_dataset(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "Unemployment in India.csv"),
        output_path=None
    )
    required_cols = [
        "region",
        "date",
        "frequency",
        "estimated_unemployment_rate_pct",
        "estimated_employed",
        "estimated_labour_participation_rate_pct",
        "area",
        "year",
        "month",
        "period",
        "covid_phase",
    ]
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' missing from cleaned dataset!"


def test_date_conversion():
    df = clean_primary_dataset(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "Unemployment in India.csv"),
        output_path=None
    )
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "Column 'date' must be datetime dtype"
    assert df["date"].isnull().sum() == 0, "No NaT permitted in date column"
    assert df["date"].min() == pd.Timestamp("2019-05-31"), "Unexpected min date"
    assert df["date"].max() == pd.Timestamp("2020-06-30"), "Unexpected max date"
    assert df["date"].nunique() == 14, f"Expected 14 unique months, got {df['date'].nunique()}"


def test_numeric_conversion_and_types():
    df = clean_primary_dataset(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "Unemployment in India.csv"),
        output_path=None
    )
    assert pd.api.types.is_float_dtype(df["estimated_unemployment_rate_pct"]), "Unemployment rate must be float"
    assert pd.api.types.is_integer_dtype(df["estimated_employed"]), "Employed count must be integer"
    assert pd.api.types.is_float_dtype(df["estimated_labour_participation_rate_pct"]), "Participation rate must be float"


def test_absence_of_missing_values():
    df = clean_primary_dataset(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "Unemployment in India.csv"),
        output_path=None
    )
    null_counts = df.isnull().sum()
    assert null_counts.sum() == 0, f"Cleaned dataset has unexpected missing values:\n{null_counts[null_counts > 0]}"


def test_absence_of_duplicate_rows():
    df = clean_primary_dataset(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "Unemployment in India.csv"),
        output_path=None
    )
    duplicates = df.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate rows in cleaned dataset!"


def test_value_bounds_and_validity():
    df = clean_primary_dataset(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "Unemployment in India.csv"),
        output_path=None
    )
    assert (df["estimated_unemployment_rate_pct"] >= 0.0).all(), "Negative unemployment rates detected"
    assert (df["estimated_unemployment_rate_pct"] <= 100.0).all(), "Unemployment rate exceeds 100%"
    assert (df["estimated_labour_participation_rate_pct"] >= 0.0).all(), "Negative participation rates detected"
    assert (df["estimated_labour_participation_rate_pct"] <= 100.0).all(), "Participation rate exceeds 100%"
    assert (df["estimated_employed"] > 0).all(), "Non-positive employment count detected"


def test_categorical_integrity():
    df = clean_primary_dataset(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "Unemployment in India.csv"),
        output_path=None
    )
    assert set(df["area"].unique()) == {"Rural", "Urban"}, "Unexpected values in area column"
    assert set(df["frequency"].unique()) == {"Monthly"}, "Unexpected values in frequency column"
    assert set(df["period"].unique()) == {"Pre-COVID", "COVID-Period"}, "Unexpected period categories"
    assert len(df["region"].unique()) == 28, f"Expected 28 unique regions, found {len(df['region'].unique())}"


def test_clean_secondary_dataset_integrity():
    df_sec = clean_secondary_dataset(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "Unemployment_Rate_upto_11_2020.csv"),
        output_path=None
    )
    assert df_sec.shape[0] == 267, f"Expected 267 rows, got {df_sec.shape[0]}"
    assert df_sec.isnull().sum().sum() == 0, "Secondary dataset has nulls!"
    assert pd.api.types.is_datetime64_any_dtype(df_sec["date"]), "Secondary date is not datetime"
    assert set(df_sec["zone"].unique()) == {"East", "North", "Northeast", "South", "West"}
