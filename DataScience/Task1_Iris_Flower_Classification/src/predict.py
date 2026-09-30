"""Inference module for Iris flower species prediction.

CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
"""

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "iris_classifier.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


class IrisPredictor:
    """Production predictor wrapper for the trained Iris classification pipeline."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized model and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")

        self.model = joblib.load(self.model_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata: Dict[str, Any] = json.load(f)

        self.features: List[str] = self.metadata.get(
            "features",
            ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )
        # Reconstruct inverse label mapping {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        raw_inv = self.metadata.get("inverse_label_mapping", {"0": "setosa", "1": "versicolor", "2": "virginica"})
        self.inverse_label_mapping: Dict[int, str] = {int(k): v for k, v in raw_inv.items()}

    def validate_input(
        self,
        sepal_length: Any,
        sepal_width: Any,
        petal_length: Any,
        petal_width: Any,
    ) -> List[float]:
        """Validate that all 4 measurements are numeric, finite, and physically plausible (> 0)."""
        raw_inputs = [
            ("sepal_length", sepal_length),
            ("sepal_width", sepal_width),
            ("petal_length", petal_length),
            ("petal_width", petal_width),
        ]

        validated: List[float] = []
        for name, val in raw_inputs:
            # Check boolean explicitly because bool is subclass of int in Python
            if isinstance(val, bool):
                raise TypeError(f"Measurement '{name}' cannot be boolean. Received: {val}")

            try:
                float_val = float(val)
            except (ValueError, TypeError) as exc:
                raise TypeError(f"Measurement '{name}' must be a valid number. Received: {val}") from exc

            if not math.isfinite(float_val):
                raise ValueError(f"Measurement '{name}' must be finite. Received: {float_val}")

            if float_val <= 0:
                raise ValueError(f"Measurement '{name}' must be strictly positive (> 0 cm). Received: {float_val}")

            validated.append(float_val)

        return validated

    def predict(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> Dict[str, Any]:
        """Generate prediction and class probabilities for a single Iris flower measurement."""
        validated_values = self.validate_input(sepal_length, sepal_width, petal_length, petal_width)
        input_df = pd.DataFrame([validated_values], columns=self.features)

        encoded_pred = int(self.model.predict(input_df)[0])
        predicted_species = self.inverse_label_mapping.get(encoded_pred, "unknown")

        probabilities: Optional[Dict[str, float]] = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(input_df)[0]
            # Map probabilities to species names
            probabilities = {}
            for class_idx, prob in enumerate(probs):
                species_name = self.inverse_label_mapping.get(class_idx, f"class_{class_idx}")
                probabilities[species_name] = round(float(prob), 4)

        return {
            "predicted_species": predicted_species,
            "encoded_prediction": encoded_pred,
            "probabilities": probabilities,
            "input_features": {
                "sepal_length": validated_values[0],
                "sepal_width": validated_values[1],
                "petal_length": validated_values[2],
                "petal_width": validated_values[3],
            },
        }


# Global singleton predictor for lightweight reuse
_GLOBAL_PREDICTOR: Optional[IrisPredictor] = None


def get_predictor() -> IrisPredictor:
    """Return or initialize global IrisPredictor instance."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = IrisPredictor()
    return _GLOBAL_PREDICTOR


def predict_species(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    *,
    predictor: Optional[IrisPredictor] = None,
) -> Dict[str, Any]:
    """Top-level convenience function for predicting Iris flower species.
    
    Parameters:
        sepal_length (float): Sepal length in centimeters.
        sepal_width (float): Sepal width in centimeters.
        petal_length (float): Petal length in centimeters.
        petal_width (float): Petal width in centimeters.
        predictor (IrisPredictor, optional, keyword-only): Custom predictor instance.

    Returns:
        dict: Containing 'predicted_species', 'encoded_prediction',
              'probabilities', and 'input_features'.
    """
    if predictor is not None and not isinstance(predictor, IrisPredictor):
        raise TypeError(f"'predictor' must be an IrisPredictor instance. Got: {type(predictor)}")
    pred_engine = predictor or get_predictor()
    return pred_engine.predict(sepal_length, sepal_width, petal_length, petal_width)
