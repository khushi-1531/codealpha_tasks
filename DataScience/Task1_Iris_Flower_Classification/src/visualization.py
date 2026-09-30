"""Exploratory data visualization module for the Iris Flower dataset.

CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
"""

import os
# Configure OpenBLAS and multi-threading for stable Windows execution
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "iris.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Professional color palette for Iris species
SPECIES_PALETTE = {
    "setosa": "#1f77b4",       # Blue
    "versicolor": "#ff7f0e",   # Orange
    "virginica": "#2ca02c",    # Green
}


def load_dataset(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load the Iris dataset for visualization."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    return pd.read_csv(csv_path)


def plot_class_distribution(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate and save class frequency distribution bar plot."""
    if output_path is None:
        output_path = DEFAULT_FIGURES_DIR / "class_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = df["species"].value_counts().reset_index()
    counts.columns = ["species", "count"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [s.capitalize() for s in counts["species"]],
        counts["count"],
        color=[SPECIES_PALETTE[s] for s in counts["species"]],
        edgecolor="#1D3557",
        alpha=0.85,
        width=0.45,
    )
    ax.set_ylim(0, 62)
    ax.set_ylabel("Sample Count", fontsize=11, fontweight="bold")
    ax.set_xlabel("Iris Species", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Sample Distribution by Iris Species (N=150)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    total = len(df)
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.5,
            f"{int(yval)} ({pct:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_feature_distributions(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate and save 2x2 boxplots comparing feature distributions across species."""
    if output_path is None:
        output_path = DEFAULT_FIGURES_DIR / "feature_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    plot_df["species"] = plot_df["species"].str.capitalize()
    cap_palette = {k.capitalize(): v for k, v in SPECIES_PALETTE.items()}

    features = [
        ("sepal_length", "Sepal Length (cm)"),
        ("sepal_width", "Sepal Width (cm)"),
        ("petal_length", "Petal Length (cm)"),
        ("petal_width", "Petal Width (cm)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(features):
        ax = axes[idx]
        sns.boxplot(
            x="species",
            y=col,
            hue="species",
            data=plot_df,
            palette=cap_palette,
            legend=False,
            ax=ax,
            width=0.45,
            boxprops=dict(alpha=0.85),
            fliersize=3,
        )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("cm", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Morphological Measurement Distributions by Iris Species",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_pairwise_relationships(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate and save pairwise feature scatterplot matrix (pairplot) colored by species."""
    if output_path is None:
        output_path = DEFAULT_FIGURES_DIR / "pairwise_feature_relationships.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    plot_df["species"] = plot_df["species"].str.capitalize()
    capitalized_palette = {k.capitalize(): v for k, v in SPECIES_PALETTE.items()}

    pair_grid = sns.pairplot(
        plot_df,
        hue="species",
        palette=capitalized_palette,
        markers=["o", "s", "D"],
        diag_kind="kde",
        plot_kws={"alpha": 0.8, "s": 25},
        diag_kws={"fill": True, "alpha": 0.3},
        height=1.8,
    )

    pair_grid.fig.subplots_adjust(top=0.93)
    pair_grid.fig.suptitle(
        "Pairwise Feature Relationships Across Iris Species",
        fontsize=12,
        fontweight="bold",
    )

    pair_grid.savefig(output_path, dpi=150)
    plt.close("all")
    return output_path


def run() -> None:
    """Generate all exploratory data visualizations."""
    print("[INFO] Generating exploratory data visualizations...")
    df = load_dataset()

    path_dist = plot_class_distribution(df)
    print(f"[SUCCESS] Saved class distribution plot: {path_dist}")

    path_feat = plot_feature_distributions(df)
    print(f"[SUCCESS] Saved feature distributions plot: {path_feat}")

    path_pair = plot_pairwise_relationships(df)
    print(f"[SUCCESS] Saved pairwise feature relationships plot: {path_pair}")


if __name__ == "__main__":
    run()
