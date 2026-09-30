"""
Inference and Prediction Module for CodeAlpha Task 3: Car Price Prediction.

Loads the serialized Scikit-learn model pipeline and generates vehicle price
valuations on the original price scale ($ USD).
"""

import sys
from pathlib import Path
from typing import Any, Dict, Union
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

DEFAULT_MODEL_PATH = Path("models/car_price_model.joblib")


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    """Load serialized model pipeline from disk."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at: {path}")
    model = joblib.load(path)
    return model


def prepare_input_dataframe(input_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """
    Format and validate raw vehicle features into the required model schema.
    Computes engineered features (power_to_weight, engine_power_ratio, average_mpg)
    if not already present in the input.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError(f"Expected dict or pd.DataFrame, got {type(input_data)}")

    # Derive engineered ratios if not provided
    if "power_to_weight" not in df.columns:
        df["power_to_weight"] = df["horsepower"] / df["curbweight"]

    if "engine_power_ratio" not in df.columns:
        df["engine_power_ratio"] = df["horsepower"] / df["enginesize"]

    if "average_mpg" not in df.columns:
        df["average_mpg"] = (df["citympg"] + df["highwaympg"]) / 2.0

    # Ensure required features are present
    required_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required features: {missing}")

    return df[required_features]


def predict_car_price(
    input_data: Union[Dict[str, Any], pd.DataFrame],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> Union[float, np.ndarray]:
    """
    Generate vehicle valuation prediction(s) in original USD ($) scale.
    """
    model = load_model(model_path)
    X = prepare_input_dataframe(input_data)
    predictions = model.predict(X)

    if isinstance(input_data, dict) or (isinstance(input_data, pd.DataFrame) and len(input_data) == 1):
        return float(predictions[0])
    return predictions


def run_prediction_demo():
    """Demonstrate inference using a sample from the test set."""
    test_path = Path("data/processed/test.csv")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    test_df = pd.read_csv(test_path)
    sample = test_df.iloc[0].to_dict()
    actual_price = sample.get("price")

    print("\n" + "=" * 60)
    print("CAR PRICE PREDICTION INFERENCE DEMO")
    print("=" * 60)
    print(f"Vehicle: {sample.get('brand', 'Unknown').title()} (ID: {sample.get('car_ID')})")
    print(f"Horsepower: {sample.get('horsepower')} bhp | Engine Size: {sample.get('enginesize')} cu in")
    print(f"Curb Weight: {sample.get('curbweight')} lbs | Fuel: {sample.get('fueltype')} | Body: {sample.get('carbody')}")
    print(f"City / Highway Mileage: {sample.get('citympg')} / {sample.get('highwaympg')} MPG")

    predicted_price = predict_car_price(sample)
    print(f"\nPredicted Valuation: ${predicted_price:,.2f}")
    if actual_price is not None:
        error = predicted_price - actual_price
        pct_error = (error / actual_price) * 100.0
        print(f"Actual Market Price: ${actual_price:,.2f}")
        print(f"Valuation Difference: ${error:+,.2f} ({pct_error:+.2f}%)")
    print("=" * 60)
    return predicted_price


if __name__ == "__main__":
    run_prediction_demo()
