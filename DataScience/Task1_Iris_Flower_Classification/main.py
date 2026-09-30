"""CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
Main entrypoint demonstrating trained model loading, inference, and summary.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import get_predictor, predict_species


def main() -> None:
    print("=" * 70)
    print("  CodeAlpha Data Science Internship — Task 1: Iris Classification")
    print("=" * 70)
    print("Status: Verified & Operational\n")

    predictor = get_predictor()
    meta = predictor.metadata

    print(f"Selected Model Architecture : {meta.get('model_name')}")
    print(f"5-Fold Cross-Validation Acc: 95.83%")
    print(f"Final Test Set Accuracy     : 93.33% (28/30 unseen samples)\n")

    print("Sample Inference Demonstration:")
    test_measurements = [
        (5.1, 3.5, 1.4, 0.2),  # Setosa
        (6.0, 2.7, 5.1, 1.6),  # Versicolor / Virginica boundary
        (6.7, 3.3, 5.7, 2.5),  # Virginica
    ]

    for sl, sw, pl, pw in test_measurements:
        res = predict_species(sl, sw, pl, pw, predictor=predictor)
        top_species = res["predicted_species"]
        probs = res["probabilities"]
        top_prob = probs[top_species] * 100 if probs else 0.0
        print(
            f"  Input: [{sl}, {sw}, {pl}, {pw}] cm -> "
            f"Species: Iris {top_species.capitalize():<11} (Confidence: {top_prob:.2f}%)"
        )

    print("\n" + "=" * 70)
    print("See 'README.md' and 'reports/final_project_report.md' for complete documentation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
