"""
Unit and Integration Test Suite for Regression Modeling and Inference Pipeline.

Verifies:
1. Model pipeline can be instantiated with preprocessor and regressor.
2. Target 'price' is not present in feature matrix X.
3. 'log_price' target is not present in feature matrix X.
4. 'car_ID' identifier is not present in feature matrix X.
5. Train/test dimensions are valid (164 train rows, 41 test rows, 27 features).
6. Preprocessor fits without throwing errors or exceptions.
7. Model can train cleanly on training partition.
8. Prediction output has the correct length matching input records.
9. Predictions are strictly numeric floats.
10. Saved Joblib model artifact exists and can be loaded.
11. Loaded model can make a valid single-record valuation prediction.
12. No NaN or infinite prediction outputs are produced.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMNS,
    build_preprocessor,
)
from src.predict import load_model, predict_car_price, prepare_input_dataframe

MODEL_PATH = Path("models/car_price_model.joblib")
METADATA_PATH = Path("models/model_metadata.json")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")


@pytest.fixture(scope="module")
def train_test_data():
    """Load train and test datasets and construct feature matrices."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    X_train = train_df[features].copy()
    y_train = train_df["price"].copy()
    X_test = test_df[features].copy()
    y_test = test_df["price"].copy()

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "features": features,
    }


def test_01_model_pipeline_creation():
    """Verify that a model pipeline with preprocessor and regressor instantiates correctly."""
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(n_estimators=10, random_state=42)),
    ])
    assert pipe is not None
    assert "preprocessor" in pipe.named_steps
    assert "regressor" in pipe.named_steps


def test_02_target_price_not_in_x(train_test_data):
    """Verify that 'price' is strictly absent from feature inputs X."""
    assert "price" not in train_test_data["X_train"].columns
    assert "price" not in train_test_data["X_test"].columns


def test_03_log_price_not_in_x(train_test_data):
    """Verify that 'log_price' is strictly absent from feature inputs X."""
    assert "log_price" not in train_test_data["X_train"].columns
    assert "log_price" not in train_test_data["X_test"].columns


def test_04_car_id_not_in_x(train_test_data):
    """Verify that 'car_ID' identifier is strictly absent from feature inputs X."""
    assert "car_ID" not in train_test_data["X_train"].columns
    assert "car_ID" not in train_test_data["X_test"].columns
    assert "CarName" not in train_test_data["X_train"].columns
    assert "CarName" not in train_test_data["X_test"].columns


def test_05_train_test_dimensions_valid(train_test_data):
    """Verify train and test record counts and feature column count."""
    assert len(train_test_data["X_train"]) == 164
    assert len(train_test_data["X_test"]) == 41
    assert train_test_data["X_train"].shape[1] == 27
    assert train_test_data["X_test"].shape[1] == 27


def test_06_preprocessing_fits_without_errors(train_test_data):
    """Verify preprocessor fits on training features without errors."""
    preprocessor = build_preprocessor()
    preprocessor.fit(train_test_data["X_train"])
    transformed = preprocessor.transform(train_test_data["X_test"])
    assert transformed.shape[0] == 41
    assert not np.isnan(transformed).any()


def test_07_model_trains_on_training_data(train_test_data):
    """Verify model pipeline trains on training set and produces an estimator."""
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(n_estimators=10, random_state=42)),
    ])
    pipe.fit(train_test_data["X_train"], train_test_data["y_train"])
    assert hasattr(pipe.named_steps["regressor"], "estimators_")


def test_08_prediction_output_has_correct_length(train_test_data):
    """Verify predictions array length equals input row count."""
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(n_estimators=10, random_state=42)),
    ])
    pipe.fit(train_test_data["X_train"], train_test_data["y_train"])
    preds = pipe.predict(train_test_data["X_test"])
    assert len(preds) == len(train_test_data["X_test"])


def test_09_predictions_are_numeric(train_test_data):
    """Verify predictions are numeric floating point values."""
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(n_estimators=10, random_state=42)),
    ])
    pipe.fit(train_test_data["X_train"], train_test_data["y_train"])
    preds = pipe.predict(train_test_data["X_test"])
    assert issubclass(preds.dtype.type, np.floating)
    assert (preds > 0).all()


def test_10_saved_joblib_model_exists_and_loads():
    """Verify saved car_price_model.joblib exists and loads cleanly."""
    assert MODEL_PATH.is_file(), f"Missing model artifact at {MODEL_PATH}"
    model = load_model(MODEL_PATH)
    assert model is not None
    assert hasattr(model, "predict")


def test_11_loaded_model_can_make_prediction():
    """Verify that loaded model can make a prediction on a sample dictionary."""
    test_df = pd.read_csv(TEST_PATH)
    sample_dict = test_df.iloc[0].to_dict()
    pred = predict_car_price(sample_dict, model_path=MODEL_PATH)
    assert isinstance(pred, float)
    assert pred > 0


def test_12_no_nan_or_inf_predictions(train_test_data):
    """Verify loaded model produces zero NaNs or infinities across all test records."""
    model = load_model(MODEL_PATH)
    preds = model.predict(train_test_data["X_test"])
    assert not np.isnan(preds).any(), "NaN detected in test predictions."
    assert not np.isinf(preds).any(), "Infinity detected in test predictions."
    assert len(preds) == 41
