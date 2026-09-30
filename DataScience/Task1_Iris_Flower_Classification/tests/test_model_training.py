"""Automated tests for model training, evaluation, predictions, and model persistence.

CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
"""

from pathlib import Path
import sys
import unittest
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import prepare_dataset
from src.model_training import (
    get_candidate_models,
    evaluate_candidates_cv,
    select_best_model,
    train_final_model,
    evaluate_on_test_set,
)


class TestModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.raw_csv_path = cls.project_root / "data" / "raw" / "iris.csv"
        cls.model_path = cls.project_root / "models" / "iris_classifier.joblib"
        cls.metadata_path = cls.project_root / "models" / "model_metadata.json"
        cls.preprocessed = prepare_dataset(cls.raw_csv_path, test_size=0.20, random_state=42)

    def test_candidate_models_definition(self):
        """Verify candidate models dictionary contains all required algorithms."""
        candidates = get_candidate_models(random_state=42)
        expected_names = {"Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest"}
        self.assertEqual(set(candidates.keys()), expected_names)

    def test_cv_evaluation_on_training_data_only(self):
        """Verify cross-validation runs on training data and computes mean/std accuracy."""
        candidates = get_candidate_models(random_state=42)
        cv_results = evaluate_candidates_cv(
            candidates,
            self.preprocessed.X_train,
            self.preprocessed.y_train,
            n_splits=5,
            random_state=42,
        )
        self.assertEqual(len(cv_results), 4)
        for name, res in cv_results.items():
            self.assertIn("mean_cv_accuracy", res)
            self.assertIn("std_cv_accuracy", res)
            self.assertEqual(len(res["fold_scores"]), 5)
            self.assertTrue(0.80 <= res["mean_cv_accuracy"] <= 1.0)

    def test_model_training_and_predictions(self):
        """Verify model trains, produces predictions of expected length and valid class values."""
        candidates = get_candidate_models(random_state=42)
        model = candidates["Logistic Regression"]
        fitted_model = train_final_model(model, self.preprocessed.X_train, self.preprocessed.y_train)

        # Make predictions on test set
        predictions = fitted_model.predict(self.preprocessed.X_test)
        self.assertEqual(len(predictions), 30, "Predictions count must equal test set size (30).")

        # Predictions must only contain valid encoded target classes {0, 1, 2}
        unique_preds = set(predictions)
        self.assertTrue(unique_preds.issubset({0, 1, 2}), f"Invalid predictions found: {unique_preds}")

    def test_evaluation_metrics_calculation(self):
        """Verify test evaluation metrics can be computed accurately."""
        candidates = get_candidate_models(random_state=42)
        model = train_final_model(candidates["Logistic Regression"], self.preprocessed.X_train, self.preprocessed.y_train)
        target_names = ["setosa", "versicolor", "virginica"]

        metrics = evaluate_on_test_set(
            model,
            self.preprocessed.X_test,
            self.preprocessed.y_test,
            target_names=target_names,
        )

        self.assertIn("test_accuracy", metrics)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("classification_report", metrics)

        # Validate range
        self.assertTrue(0.0 <= metrics["test_accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["precision_macro"] <= 1.0)
        self.assertTrue(0.0 <= metrics["recall_macro"] <= 1.0)
        self.assertTrue(0.0 <= metrics["f1_macro"] <= 1.0)

        # Confusion matrix shape (3x3)
        cm = np.array(metrics["confusion_matrix"])
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(int(cm.sum()), 30)

    def test_saved_model_artifact_loading_and_prediction(self):
        """Verify the serialized joblib model exists and can be loaded to make valid predictions."""
        self.assertTrue(self.model_path.exists(), f"Model artifact missing at {self.model_path}")
        self.assertTrue(self.metadata_path.exists(), f"Model metadata missing at {self.metadata_path}")

        loaded_model = joblib.load(self.model_path)
        sample_input = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=self.preprocessed.X_test.columns)
        pred = loaded_model.predict(sample_input)

        self.assertEqual(len(pred), 1)
        self.assertIn(int(pred[0]), [0, 1, 2])
        # A 5.1, 3.5, 1.4, 0.2 sample is canonically Setosa (class 0)
        self.assertEqual(int(pred[0]), 0)

        # Also verify loaded model can predict on entire test set with identical length
        batch_preds = loaded_model.predict(self.preprocessed.X_test)
        self.assertEqual(len(batch_preds), len(self.preprocessed.X_test))
        self.assertTrue(set(batch_preds).issubset({0, 1, 2}))

    def test_generated_figures_and_reports_exist(self):
        """Verify that evaluation figures and report exist on disk."""
        comparison_plot = self.project_root / "reports" / "figures" / "model_comparison.png"
        confusion_plot = self.project_root / "reports" / "figures" / "confusion_matrix.png"
        eval_report = self.project_root / "reports" / "model_evaluation_report.md"

        self.assertTrue(comparison_plot.exists(), f"Missing figure: {comparison_plot}")
        self.assertTrue(confusion_plot.exists(), f"Missing figure: {confusion_plot}")
        self.assertTrue(eval_report.exists(), f"Missing report: {eval_report}")


if __name__ == "__main__":
    unittest.main()
