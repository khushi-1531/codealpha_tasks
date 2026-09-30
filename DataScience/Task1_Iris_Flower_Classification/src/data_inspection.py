"""Dataset acquisition, inspection, and verification module for Iris Flower Classification.

CodeAlpha Data Science Internship - Task 1
"""

import os
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from sklearn.datasets import load_iris


def acquire_dataset(csv_path: Path) -> Path:
    """Ensure raw dataset exists in the expected location.
    
    If no dataset file exists at `csv_path`, it loads the standard Iris dataset
    via scikit-learn's official loader and serializes it to CSV.
    """
    if csv_path.exists():
        print(f"[INFO] Found existing dataset at: {csv_path}")
        return csv_path

    print(f"[INFO] Dataset not found at {csv_path}. Acquiring via sklearn.datasets.load_iris...")
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    # Map target integers (0, 1, 2) to class names (setosa, versicolor, virginica)
    target_names = iris.target_names
    df["species"] = df["target"].map(lambda idx: target_names[idx])
    df.drop(columns=["target"], inplace=True)

    # Standardize column names for ease of use and professional consistency
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Dataset successfully saved to: {csv_path}")
    return csv_path


def inspect_dataset(csv_path: Path) -> Dict[str, Any]:
    """Perform comprehensive data inspection on the Iris dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    expected_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    expected_classes = sorted(["setosa", "versicolor", "virginica"])

    features_present = all(feat in df.columns for feat in expected_features)
    unique_classes = sorted(df["species"].dropna().unique().tolist())
    classes_match = unique_classes == expected_classes

    inspection = {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(count) for col, count in df.isnull().sum().items()},
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": {k: int(v) for k, v in df["species"].value_counts().to_dict().items()},
        "unique_classes": unique_classes,
        "descriptive_stats": df.describe().to_dict(),
        "verification": {
            "features_present": features_present,
            "classes_match": classes_match,
            "no_missing_values": int(df.isnull().sum().sum()) == 0,
            "is_valid": features_present and classes_match and int(df.isnull().sum().sum()) == 0,
        },
    }

    return inspection


def generate_report(inspection: Dict[str, Any], report_path: Path) -> str:
    """Generate a clean markdown inspection report and save to reports/."""
    lines = [
        "# Iris Dataset Inspection Report",
        "**Project:** CodeAlpha Data Science Internship — Task 1 (Iris Flower Classification)  ",
        f"**Inspection Status:** {'PASSED' if inspection['verification']['is_valid'] else 'FAILED'}\n",
        "---",
        "## 1. Dataset Dimensions & Schema",
        f"- **Total Samples (Rows):** {inspection['num_rows']}",
        f"- **Total Attributes (Columns):** {inspection['num_columns']}",
        f"- **Column Names:** `{', '.join(inspection['columns'])}`\n",
        "### Data Types",
        "| Column | Data Type | Missing Count |",
        "| :--- | :--- | :--- |",
    ]

    for col in inspection["columns"]:
        dtype = inspection["data_types"][col]
        missing = inspection["missing_values"][col]
        lines.append(f"| `{col}` | `{dtype}` | {missing} |")

    lines.extend([
        "",
        "## 2. Data Integrity",
        f"- **Total Missing Values:** {inspection['total_missing']}",
        f"- **Duplicate Rows Detected:** {inspection['duplicate_rows']}",
        "  *(Note: 1 duplicate row is expected in the canonical 150-sample Iris dataset at index 142)*\n",
        "## 3. Target Distribution",
        "| Species | Sample Count | Balance Ratio |",
        "| :--- | :--- | :--- |",
    ])

    total = inspection["num_rows"]
    for species, count in inspection["class_distribution"].items():
        ratio = f"{(count / total) * 100:.1f}%"
        lines.append(f"| *Iris {species}* | {count} | {ratio} |")

    lines.extend([
        "",
        "## 4. Descriptive Statistics",
        "| Metric | sepal_length | sepal_width | petal_length | petal_width |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    stats = inspection["descriptive_stats"]
    for metric in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        sl = f"{stats['sepal_length'][metric]:.3f}"
        sw = f"{stats['sepal_width'][metric]:.3f}"
        pl = f"{stats['petal_length'][metric]:.3f}"
        pw = f"{stats['petal_width'][metric]:.3f}"
        lines.append(f"| **{metric}** | {sl} | {sw} | {pl} | {pw} |")

    lines.extend([
        "",
        "## 5. Verification Checklist",
        f"- [x] 4 Iris physical measurements present: `{inspection['verification']['features_present']}`",
        f"- [x] 3 Species represented (setosa, versicolor, virginica): `{inspection['verification']['classes_match']}`",
        f"- [x] Zero missing values: `{inspection['verification']['no_missing_values']}`",
        f"- [x] Overall structure valid: `{inspection['verification']['is_valid']}`",
    ])

    content = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return content


def run() -> None:
    """Execution runner for data inspection."""
    base_dir = Path(__file__).resolve().parent.parent
    raw_data_dir = base_dir / "data" / "raw"
    csv_file = raw_data_dir / "iris.csv"
    reports_dir = base_dir / "reports"
    report_file = reports_dir / "dataset_inspection_report.md"

    # Step 1: Ensure dataset is available
    acquire_dataset(csv_file)

    # Step 2: Inspect dataset
    results = inspect_dataset(csv_file)

    # Step 3: Write report
    report_md = generate_report(results, report_file)
    print("\n" + "=" * 60)
    print(report_md)
    print("=" * 60)
    print(f"[SUCCESS] Report saved to: {report_file}")


if __name__ == "__main__":
    run()
