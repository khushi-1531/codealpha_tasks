"""Unit tests for the prediction module and visualization pipeline.

CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
"""

from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import IrisPredictor, get_predictor, predict_species
from src.visualization import (
    load_dataset,
    plot_class_distribution,
    plot_feature_distributions,
    plot_pairwise_relationships,
)


class TestPredictionModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.predictor = get_predictor()

    def test_model_and_metadata_load_successfully(self):
        """Verify model artifact and metadata file load without errors."""
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.metadata)
        self.assertIn("features", self.predictor.metadata)
        self.assertIn("inverse_label_mapping", self.predictor.metadata)

    def test_valid_numeric_input_produces_valid_species(self):
        """Verify standard physical measurements return a recognized Iris class."""
        result = predict_species(5.1, 3.5, 1.4, 0.2, predictor=self.predictor)
        self.assertIn("predicted_species", result)
        self.assertIn(result["predicted_species"], ["setosa", "versicolor", "virginica"])
        self.assertEqual(result["predicted_species"], "setosa")

    def test_predictions_contain_valid_classes(self):
        """Verify different measurement profiles return recognized Iris class names."""
        test_cases = [
            (5.0, 3.4, 1.5, 0.2),  # Archetype Setosa
            (6.0, 2.8, 4.5, 1.3),  # Archetype Versicolor
            (6.5, 3.0, 5.5, 2.0),  # Archetype Virginica
        ]
        for sl, sw, pl, pw in test_cases:
            res = predict_species(sl, sw, pl, pw, predictor=self.predictor)
            self.assertIn(res["predicted_species"], {"setosa", "versicolor", "virginica"})

    def test_probability_output_structure(self):
        """Verify probabilities output contains all 3 classes and sums approximately to 1.0."""
        result = predict_species(5.8, 2.7, 4.1, 1.0, predictor=self.predictor)
        probs = result.get("probabilities")
        self.assertIsNotNone(probs, "Model must provide prediction probabilities.")
        self.assertEqual(set(probs.keys()), {"setosa", "versicolor", "virginica"})

        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)
        for class_name, prob in probs.items():
            self.assertTrue(0.0 <= prob <= 1.0, f"Probability {prob} out of bounds for {class_name}")

    def test_invalid_non_numeric_input_handled(self):
        """Verify non-numeric values trigger TypeError or ValueError."""
        with self.assertRaises(TypeError):
            predict_species("five", 3.5, 1.4, 0.2, predictor=self.predictor)
        with self.assertRaises(TypeError):
            predict_species(5.1, None, 1.4, 0.2, predictor=self.predictor)
        with self.assertRaises(TypeError):
            predict_species(5.1, 3.5, True, 0.2, predictor=self.predictor)

    def test_infinite_and_negative_input_rejected(self):
        """Verify infinite or negative measurements are rejected."""
        with self.assertRaises(ValueError):
            predict_species(float("inf"), 3.5, 1.4, 0.2, predictor=self.predictor)
        with self.assertRaises(ValueError):
            predict_species(float("nan"), 3.5, 1.4, 0.2, predictor=self.predictor)
        with self.assertRaises(ValueError):
            predict_species(-5.1, 3.5, 1.4, 0.2, predictor=self.predictor)
        with self.assertRaises(ValueError):
            predict_species(0.0, 3.5, 1.4, 0.2, predictor=self.predictor)

    def test_wrong_number_of_features_rejected(self):
        """Verify providing wrong number of positional arguments raises TypeError."""
        with self.assertRaises(TypeError):
            predict_species(5.1, 3.5, 1.4)  # Missing 4th feature
        with self.assertRaises(TypeError):
            predict_species(5.1, 3.5, 1.4, 0.2, 1.0)  # Too many features


class TestVisualizations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.df = load_dataset()
        cls.figures_dir = cls.project_root / "reports" / "figures"

    def test_class_distribution_plot_generation(self):
        """Verify class distribution plot generates and exists."""
        target_path = self.figures_dir / "class_distribution.png"
        out = plot_class_distribution(self.df, target_path)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1000)

    def test_feature_distributions_plot_generation(self):
        """Verify feature distributions boxplot generates and exists."""
        target_path = self.figures_dir / "feature_distributions.png"
        out = plot_feature_distributions(self.df, target_path)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1000)

    def test_pairwise_relationships_plot_generation(self):
        """Verify pairwise feature relationships pairplot generates and exists."""
        target_path = self.figures_dir / "pairwise_feature_relationships.png"
        out = plot_pairwise_relationships(self.df, target_path)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
