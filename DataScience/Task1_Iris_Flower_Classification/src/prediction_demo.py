"""Demonstration script for Iris Flower species classification inference.

CodeAlpha Data Science Internship - Task 1: Iris Flower Classification

NOTE:
These samples are provided solely as illustrative demonstration examples
to verify model accessibility and input/output inference functionality.
This demonstration does not constitute a new assessment of model accuracy.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import get_predictor, predict_species


def run_demonstration() -> None:
    """Run demonstration with 3 representative Iris samples."""
    print("=" * 75)
    print("   CodeAlpha Data Science Internship — Task 1: Iris Prediction Demo")
    print("=" * 75)
    print("Notice: The following instances are representative demonstration examples")
    print("reflecting typical measurements from each of the three botanical classes.\n")

    predictor = get_predictor()

    demo_samples = [
        {
            "description": "Representative Sample A (Morphologically Setosa)",
            "measurements": (5.1, 3.5, 1.4, 0.2),
            "expected_reference": "setosa",
        },
        {
            "description": "Representative Sample B (Morphologically Versicolor)",
            "measurements": (6.0, 2.7, 5.1, 1.6),
            "expected_reference": "versicolor",
        },
        {
            "description": "Representative Sample C (Morphologically Virginica)",
            "measurements": (6.7, 3.3, 5.7, 2.5),
            "expected_reference": "virginica",
        },
    ]

    for idx, sample in enumerate(demo_samples, 1):
        sl, sw, pl, pw = sample["measurements"]
        print(f"--- Example #{idx}: {sample['description']} ---")
        print(f"  Inputs (cm): Sepal Length={sl}, Sepal Width={sw}, Petal Length={pl}, Petal Width={pw}")

        result = predict_species(sl, sw, pl, pw, predictor=predictor)

        predicted_species = result["predicted_species"]
        print(f"  Predicted Species : Iris {predicted_species.upper()}")
        print(f"  Reference Archetype: {sample['expected_reference']}")

        if result["probabilities"]:
            print("  Class Probabilities:")
            for species, prob in result["probabilities"].items():
                bar = "#" * int(prob * 20)
                print(f"    - {species.capitalize():<12}: {prob * 100:6.2f}%  |{bar:<20}|")

        print()

    print("=" * 75)
    print("Demonstration successfully finished using saved model artifacts.")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
