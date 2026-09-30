"""Unit tests for the preprocessing module.

CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
"""

from pathlib import Path
import unittest
import pandas as pd
import numpy as np

from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LABEL_MAPPING,
    INVERSE_LABEL_MAPPING,
    load_raw_data,
    validate_raw_data,
    separate_features_and_target,
    encode_target,
    split_data,
    validate_splits,
    prepare_dataset,
)


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.raw_csv_path = cls.project_root / "data" / "raw" / "iris.csv"

    def test_raw_csv_exists(self):
        """Verify the raw CSV exists at the expected path."""
        self.assertTrue(self.raw_csv_path.exists(), "data/raw/iris.csv must exist.")

    def test_load_and_validate_raw_data(self):
        """Verify raw data loads and passes structural validation."""
        df = load_raw_data(self.raw_csv_path)
        self.assertEqual(len(df), 150)
        self.assertEqual(len(df.columns), 5)
        # Should not raise
        validate_raw_data(df)

    def test_validation_fails_on_missing_column(self):
        """Validation must raise ValueError if required feature is missing."""
        bad_df = pd.DataFrame({
            "sepal_length": [5.1],
            "sepal_width": [3.5],
            "species": ["setosa"],
        })
        with self.assertRaises(ValueError):
            validate_raw_data(bad_df)

    def test_validation_fails_on_null_values(self):
        """Validation must raise ValueError if any nulls are present."""
        df = load_raw_data(self.raw_csv_path).copy()
        df.loc[0, "sepal_length"] = np.nan
        with self.assertRaises(ValueError):
            validate_raw_data(df)

    def test_feature_target_separation(self):
        """Verify separation into X (4 features) and y (target)."""
        df = load_raw_data(self.raw_csv_path)
        X, y = separate_features_and_target(df)
        self.assertEqual(list(X.columns), FEATURE_COLUMNS)
        self.assertEqual(X.shape, (150, 4))
        self.assertEqual(y.name, TARGET_COLUMN)
        self.assertEqual(len(y), 150)

    def test_target_encoding(self):
        """Verify deterministic target encoding and reversible mapping."""
        df = load_raw_data(self.raw_csv_path)
        _, y = separate_features_and_target(df)
        y_encoded, mapping = encode_target(y)

        self.assertEqual(mapping, LABEL_MAPPING)
        self.assertEqual(set(y_encoded.unique()), {0, 1, 2})
        # Check mapping correctness
        for name, code in LABEL_MAPPING.items():
            self.assertEqual(y_encoded[y == name].unique()[0], code)

    def test_train_test_split_ratio_and_stratification(self):
        """Verify 80/20 train/test split size and stratified class distribution."""
        df = load_raw_data(self.raw_csv_path)
        X, y = separate_features_and_target(df)
        y_encoded, _ = encode_target(y)

        X_train, X_test, y_train, y_test = split_data(
            X, y_encoded, test_size=0.20, random_state=42, stratify=True
        )

        self.assertEqual(len(X_train), 120)
        self.assertEqual(len(X_test), 30)
        self.assertEqual(len(y_train), 120)
        self.assertEqual(len(y_test), 30)

        # Check stratification: exactly 40 per class in train (120/3) and 10 per class in test (30/3)
        for class_id in [0, 1, 2]:
            self.assertEqual(sum(y_train == class_id), 40)
            self.assertEqual(sum(y_test == class_id), 10)

    def test_no_data_leakage(self):
        """Verify no sample index overlap exists between training and test sets."""
        df = load_raw_data(self.raw_csv_path)
        X, y = separate_features_and_target(df)
        y_encoded, _ = encode_target(y)

        X_train, X_test, _, _ = split_data(X, y_encoded, test_size=0.20, random_state=42)
        overlap = set(X_train.index).intersection(set(X_test.index))
        self.assertEqual(len(overlap), 0, "Train and test index sets must have zero overlap.")

    def test_prepare_dataset_pipeline(self):
        """Verify the full prepare_dataset pipeline output structure."""
        data = prepare_dataset(self.raw_csv_path, test_size=0.20, random_state=42)
        self.assertEqual(data.X_train.shape, (120, 4))
        self.assertEqual(data.X_test.shape, (30, 4))
        self.assertEqual(len(data.y_train), 120)
        self.assertEqual(len(data.y_test), 30)
        self.assertFalse(data.metadata["split_validation"]["index_leakage"])
        self.assertEqual(data.metadata["duplicate_count"], 1)


if __name__ == "__main__":
    unittest.main()
