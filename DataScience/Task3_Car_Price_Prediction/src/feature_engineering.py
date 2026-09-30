"""
Feature Engineering and Train/Test Preparation Module for CodeAlpha Task 3.

This module derives domain-grounded vehicle performance and efficiency metrics,
structures feature subsets to prevent data leakage, and establishes reproducible
train/test splits for downstream regression modeling.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Non-predictive identification columns that must be excluded from models
IDENTIFIER_COLUMNS: List[str] = ["car_ID", "CarName"]

# Target variables (raw price and log-transformed price)
TARGET_COLUMNS: List[str] = ["price", "log_price"]

# Core categorical features requiring encoding
CATEGORICAL_FEATURES: List[str] = [
    "brand",
    "fueltype",
    "aspiration",
    "carbody",
    "drivewheel",
    "enginelocation",
    "enginetype",
    "fuelsystem",
]

# Core numerical features
NUMERICAL_FEATURES: List[str] = [
    "symboling",
    "doornumber",
    "wheelbase",
    "carlength",
    "carwidth",
    "carheight",
    "curbweight",
    "cylindernumber",
    "enginesize",
    "boreratio",
    "stroke",
    "compressionratio",
    "horsepower",
    "peakrpm",
    "citympg",
    "highwaympg",
    "power_to_weight",
    "engine_power_ratio",
    "average_mpg",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific engineered features:
    1. power_to_weight: Horsepower normalized by vehicle curb weight (bhp/lb).
    2. engine_power_ratio: Horsepower normalized by engine displacement volume (bhp / cu in).
    3. average_mpg: Arithmetic mean of city and highway mileage (MPG).
    4. log_price: Natural log transformation of vehicle price to mitigate right-skewness.

    Note: vehicle_age is explicitly omitted because the dataset lacks a vehicle
    manufacture/model year field; no synthetic temporal data is fabricated.
    """
    df_feat = df.copy()

    # 1. Power-to-weight ratio
    df_feat["power_to_weight"] = df_feat["horsepower"] / df_feat["curbweight"]

    # 2. Engine power ratio (power output per unit displacement)
    df_feat["engine_power_ratio"] = df_feat["horsepower"] / df_feat["enginesize"]

    # 3. Average fuel economy (combined city + highway mileage)
    df_feat["average_mpg"] = (df_feat["citympg"] + df_feat["highwaympg"]) / 2.0

    # 4. Target log transformation
    df_feat["log_price"] = np.log(df_feat["price"])

    return df_feat


def get_feature_definitions() -> Dict[str, List[str]]:
    """Return explicit mapping of column groupings."""
    return {
        "identifiers": IDENTIFIER_COLUMNS,
        "targets": TARGET_COLUMNS,
        "categorical": CATEGORICAL_FEATURES,
        "numerical": NUMERICAL_FEATURES,
    }


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a reproducible train/test split.
    Uses random_state to ensure deterministic reproducibility.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_df.copy(), test_df.copy()


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a Scikit-learn ColumnTransformer that:
    - Scales numerical features using StandardScaler.
    - One-hot encodes categorical features with handle_unknown='ignore'.
    To prevent data leakage, this preprocessor must be fitted solely on training data.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def run_feature_engineering_pipeline(
    cleaned_data_path: str | Path = "data/processed/car_price_cleaned.csv",
    output_engineered_path: str | Path = "data/processed/car_price_engineered.csv",
    train_path: str | Path = "data/processed/train.csv",
    test_path: str | Path = "data/processed/test.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute feature engineering and train/test data preparation."""
    df_cleaned = pd.read_csv(cleaned_data_path)
    df_engineered = engineer_features(df_cleaned)

    # Save full engineered dataset
    out_eng = Path(output_engineered_path)
    out_eng.parent.mkdir(parents=True, exist_ok=True)
    df_engineered.to_csv(out_eng, index=False)

    # Split into train and test sets
    train_df, test_df = split_data(df_engineered, test_size=0.2, random_state=42)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Feature engineering pipeline completed successfully.")
    print(f"Engineered dataset saved: {output_engineered_path} ({df_engineered.shape})")
    print(f"Train split saved: {train_path} ({train_df.shape})")
    print(f"Test split saved: {test_path} ({test_df.shape})")
    print(f"Engineered features added: power_to_weight, engine_power_ratio, average_mpg, log_price")

    return df_engineered, train_df, test_df


if __name__ == "__main__":
    run_feature_engineering_pipeline()
