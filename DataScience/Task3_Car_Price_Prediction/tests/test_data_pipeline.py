"""
Unit and Integration Test Suite for CodeAlpha Task 3 Data Pipeline.

Verifies:
1. Raw dataset existence.
2. Raw dataset immutability (SHA-256 integrity).
3. Cleaned dataset existence.
4. Essential columns presence.
5. Numeric target validity.
6. Zero missing values post-cleaning.
7. Zero duplicates post-cleaning.
8. Manufacturer typo correction accuracy.
9. Word-based numeric mapping accuracy.
10. Identification column (car_ID) exclusion from model feature subsets.
11. Numerical validity of engineered features.
12. Absence of data leakage in train/test splits.
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    CYLINDER_MAP,
    DOOR_MAP,
    MANUFACTURER_CORRECTIONS,
    clean_car_data,
    load_raw_data,
)
from src.feature_engineering import (
    IDENTIFIER_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMNS,
    build_preprocessor,
    engineer_features,
    split_data,
)

RAW_CSV_PATH = Path("data/raw/CarPrice_Assignment.csv")
CLEANED_CSV_PATH = Path("data/processed/car_price_cleaned.csv")
ENGINEERED_CSV_PATH = Path("data/processed/car_price_engineered.csv")
TRAIN_CSV_PATH = Path("data/processed/train.csv")
TEST_CSV_PATH = Path("data/processed/test.csv")

# Expected SHA256 hash of original Kaggle raw CSV
EXPECTED_RAW_SHA256 = "2c78d99359a34cb6c64a97f276c1b6ea0532197b9b950b4521f65c0d9efcbc2b"


def test_01_raw_dataset_exists():
    """Verify that the raw CSV dataset exists in data/raw/."""
    assert RAW_CSV_PATH.is_file(), f"Raw dataset missing at {RAW_CSV_PATH}"


def test_02_raw_dataset_unchanged():
    """Verify raw dataset immutability by cryptographic SHA-256 hash."""
    assert RAW_CSV_PATH.is_file()
    with open(RAW_CSV_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == EXPECTED_RAW_SHA256, (
        f"Raw dataset has been mutated! Current SHA256: {file_hash}, Expected: {EXPECTED_RAW_SHA256}"
    )


def test_03_cleaned_dataset_exists():
    """Verify that processed cleaned dataset exists."""
    assert CLEANED_CSV_PATH.is_file(), f"Cleaned dataset missing at {CLEANED_CSV_PATH}"


def test_04_expected_important_columns_exist():
    """Verify that core expected columns and engineered features exist."""
    df = pd.read_csv(CLEANED_CSV_PATH)
    expected_cols = [
        "brand",
        "price",
        "horsepower",
        "enginesize",
        "curbweight",
        "citympg",
        "highwaympg",
        "doornumber",
        "cylindernumber",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Essential column '{col}' missing from cleaned dataset."


def test_05_price_is_numeric():
    """Verify that the target price column is numeric and strictly positive."""
    df = pd.read_csv(CLEANED_CSV_PATH)
    assert pd.api.types.is_numeric_dtype(df["price"]), "Price column is not numeric."
    assert (df["price"] > 0).all(), "Detected non-positive car prices."


def test_06_no_unexpected_missing_values():
    """Verify that cleaned and engineered datasets have zero null entries."""
    df_clean = pd.read_csv(CLEANED_CSV_PATH)
    df_eng = pd.read_csv(ENGINEERED_CSV_PATH)
    assert df_clean.isnull().sum().sum() == 0, "Unexpected null values in cleaned dataset."
    assert df_eng.isnull().sum().sum() == 0, "Unexpected null values in engineered dataset."


def test_07_no_duplicate_records_after_cleaning():
    """Verify that no duplicate vehicle records exist (excluding car_ID)."""
    df = pd.read_csv(CLEANED_CSV_PATH)
    feature_cols = [c for c in df.columns if c != "car_ID"]
    assert df.duplicated(subset=feature_cols).sum() == 0, "Duplicate rows found post-cleaning."


def test_08_manufacturer_normalization_works():
    """Verify that known typographical anomalies are resolved into 22 valid brands."""
    df = pd.read_csv(CLEANED_CSV_PATH)
    unique_brands = set(df["brand"].unique())
    # Ensure erroneous tokens are absent
    for typo in MANUFACTURER_CORRECTIONS.keys():
        assert typo not in unique_brands, f"Typo '{typo}' remains uncorrected in brand column."
    # Ensure corrected forms are present
    assert "mazda" in unique_brands
    assert "toyota" in unique_brands
    assert "volkswagen" in unique_brands
    assert "porsche" in unique_brands
    # Exactly 22 canonical manufacturers
    assert len(unique_brands) == 22, f"Expected 22 unique brands, found {len(unique_brands)}."


def test_09_word_based_numeric_conversion_works():
    """Verify doornumber and cylindernumber are converted to positive integers."""
    df = pd.read_csv(CLEANED_CSV_PATH)
    assert pd.api.types.is_integer_dtype(df["doornumber"]), "doornumber must be integer."
    assert set(df["doornumber"].unique()).issubset({2, 4})

    assert pd.api.types.is_integer_dtype(df["cylindernumber"]), "cylindernumber must be integer."
    assert set(df["cylindernumber"].unique()).issubset({2, 3, 4, 5, 6, 8, 12})


def test_10_car_id_excluded_from_model_features():
    """Verify that car_ID is classified strictly as an identifier and absent from feature sets."""
    assert "car_ID" in IDENTIFIER_COLUMNS
    assert "car_ID" not in NUMERICAL_FEATURES
    assert "car_ID" not in CATEGORICAL_FEATURES
    assert "car_ID" not in TARGET_COLUMNS


def test_11_engineered_features_contain_valid_values():
    """Verify mathematical validity of power_to_weight, engine_power_ratio, average_mpg, and log_price."""
    df = pd.read_csv(ENGINEERED_CSV_PATH)
    for col in ["power_to_weight", "engine_power_ratio", "average_mpg", "log_price"]:
        assert col in df.columns, f"Engineered column '{col}' missing."
        assert not df[col].isnull().any(), f"NaNs detected in '{col}'."
        assert not np.isinf(df[col]).any(), f"Infinities detected in '{col}'."
        assert (df[col] > 0).all(), f"Non-positive values found in '{col}'."

    # Validate exact calculations
    np.testing.assert_allclose(df["power_to_weight"], df["horsepower"] / df["curbweight"])
    np.testing.assert_allclose(df["engine_power_ratio"], df["horsepower"] / df["enginesize"])
    np.testing.assert_allclose(df["average_mpg"], (df["citympg"] + df["highwaympg"]) / 2.0)
    np.testing.assert_allclose(df["log_price"], np.log(df["price"]))


def test_12_train_test_split_prevents_leakage():
    """Verify train/test split size, disjointness, and leak-free preprocessor fitting."""
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    # 1. Row counts (80% / 20% of 205 rows)
    assert len(train_df) == 164, f"Expected 164 train rows, got {len(train_df)}"
    assert len(test_df) == 41, f"Expected 41 test rows, got {len(test_df)}"
    assert len(train_df) + len(test_df) == 205

    # 2. Strict disjointness on car_ID
    train_ids = set(train_df["car_ID"])
    test_ids = set(test_df["car_ID"])
    assert train_ids.isdisjoint(test_ids), "Train and test splits overlap in car_ID!"

    # 3. Preprocessor leak-free fitting
    preprocessor = build_preprocessor()
    X_train = train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    X_test = test_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    # Preprocessor must fit ONLY on X_train
    preprocessor.fit(X_train)
    train_transformed = preprocessor.transform(X_train)
    test_transformed = preprocessor.transform(X_test)

    assert train_transformed.shape[0] == 164
    assert test_transformed.shape[0] == 41
    assert not np.isnan(train_transformed).any()
    assert not np.isnan(test_transformed).any()
