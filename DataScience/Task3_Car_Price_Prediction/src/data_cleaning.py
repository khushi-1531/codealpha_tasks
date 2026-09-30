"""
Data Cleaning Module for CodeAlpha Task 3: Car Price Prediction with Machine Learning.

This module loads the raw automobile dataset, normalizes text fields,
resolves known manufacturer typographical errors, maps word-based numbers
to integer values, performs rigorous integrity validations, and saves the
cleaned dataset to data/processed/.
"""

import os
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np

# Verified spelling corrections for manufacturer names
MANUFACTURER_CORRECTIONS: Dict[str, str] = {
    "maxda": "mazda",
    "toyouta": "toyota",
    "vokswagen": "volkswagen",
    "vw": "volkswagen",
    "porcshce": "porsche",
}

# Word-to-numeric mappings
DOOR_MAP: Dict[str, int] = {
    "two": 2,
    "four": 4,
}

CYLINDER_MAP: Dict[str, int] = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "eight": 8,
    "twelve": 12,
}


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw car price CSV dataset without mutating the original file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset file not found at: {path}")
    df = pd.read_csv(path)
    return df


def clean_car_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform thorough data cleaning on the automobile dataset:
    1. Strip leading/trailing whitespace across column names and string cells.
    2. Extract manufacturer/brand token from CarName.
    3. Correct verified typographical errors in brand names.
    4. Convert word representations of doornumber and cylindernumber to integers.
    5. Ensure proper numeric dtypes.
    """
    cleaned = df.copy()

    # 1. Clean column names
    cleaned.columns = [c.strip() for c in cleaned.columns]

    # 2. Strip whitespace in string columns
    str_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in str_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # 3 & 4. Extract and normalize manufacturer/brand
    cleaned["brand"] = (
        cleaned["CarName"]
        .apply(lambda name: str(name).split(" ")[0].strip().lower())
        .replace(MANUFACTURER_CORRECTIONS)
    )

    # 5. Convert word numbers to numeric integers
    if "doornumber" in cleaned.columns:
        cleaned["doornumber"] = cleaned["doornumber"].str.lower().map(DOOR_MAP)
        if cleaned["doornumber"].isnull().any():
            raise ValueError("Unmapped values found in 'doornumber' column.")
        cleaned["doornumber"] = cleaned["doornumber"].astype(int)

    if "cylindernumber" in cleaned.columns:
        cleaned["cylindernumber"] = cleaned["cylindernumber"].str.lower().map(CYLINDER_MAP)
        if cleaned["cylindernumber"].isnull().any():
            raise ValueError("Unmapped values found in 'cylindernumber' column.")
        cleaned["cylindernumber"] = cleaned["cylindernumber"].astype(int)

    return cleaned


def validate_cleaned_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
    """
    Validate dataset integrity:
    - Missing value audit
    - Duplicate row audit
    - Physical and economic validity checks (positive price, dimensions, horsepower)
    - Legitimate brand inventory verification
    """
    validation_results = {}

    # Check nulls
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    validation_results["total_nulls"] = total_nulls
    if total_nulls > 0:
        raise ValueError(f"Cleaned dataset contains unexpected null values:\n{null_counts[null_counts > 0]}")

    # Check duplicates (excluding car_ID)
    cols_to_check = [c for c in df.columns if c != "car_ID"]
    dup_count = int(df.duplicated(subset=cols_to_check).sum())
    validation_results["duplicate_count"] = dup_count

    # Check price validity
    if (df["price"] <= 0).any():
        raise ValueError("Non-positive price values detected.")
    validation_results["min_price"] = float(df["price"].min())
    validation_results["max_price"] = float(df["price"].max())

    # Check physical dimensions and mechanical metrics
    for col in ["wheelbase", "carlength", "carwidth", "carheight", "curbweight", "enginesize", "horsepower"]:
        if (df[col] <= 0).any():
            raise ValueError(f"Non-positive values detected in required column: {col}")

    # Verify unique brand count (expected 22 legitimate brands)
    unique_brands = sorted(df["brand"].unique())
    validation_results["unique_brands_count"] = len(unique_brands)
    validation_results["unique_brands"] = unique_brands

    return True, validation_results


def save_cleaned_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save cleaned dataframe to designated CSV path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def run_data_cleaning_pipeline(
    raw_path: str | Path = "data/raw/CarPrice_Assignment.csv",
    output_path: str | Path = "data/processed/car_price_cleaned.csv",
) -> pd.DataFrame:
    """Execute end-to-end data cleaning and validation pipeline."""
    df_raw = load_raw_data(raw_path)
    df_cleaned = clean_car_data(df_raw)
    _, results = validate_cleaned_data(df_cleaned)
    save_cleaned_data(df_cleaned, output_path)
    print(f"Data cleaning pipeline successfully completed.")
    print(f"Cleaned dataset saved to: {output_path} ({df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns)")
    print(f"Verified brands ({results['unique_brands_count']}): {', '.join(results['unique_brands'])}")
    return df_cleaned


if __name__ == "__main__":
    run_data_cleaning_pipeline()
