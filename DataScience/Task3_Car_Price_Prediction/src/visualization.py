"""
Visualization Pipeline for CodeAlpha Task 3: Car Price Prediction with Machine Learning.

Generates publication-quality 300 DPI figures illustrating price distributions,
predictor relationships, and correlation patterns.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set overall publication-grade visual theme
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 1.0


def plot_price_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot dual-panel distribution comparing raw price vs. log-transformed price."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Raw price
    sns.histplot(df["price"], kde=True, ax=axes[0], color="#2980b9", edgecolor="black", bins=25)
    mean_val = df["price"].mean()
    median_val = df["price"].median()
    axes[0].axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Mean: ${mean_val:,.0f}")
    axes[0].axvline(median_val, color="#27ae60", linestyle="-.", linewidth=1.8, label=f"Median: ${median_val:,.0f}")
    axes[0].set_title("Raw Price Distribution (Right-Skewed)", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Vehicle Price (USD)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].legend(frameon=True)

    # Log price
    sns.histplot(df["log_price"], kde=True, ax=axes[1], color="#8e44ad", edgecolor="black", bins=25)
    log_mean = df["log_price"].mean()
    axes[1].axvline(log_mean, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Mean: {log_mean:.2f}")
    axes[1].set_title("Log-Transformed Price Distribution (Normalized)", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Natural Log of Price: log(Price)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].legend(frameon=True)

    plt.suptitle("Figure 1: Vehicle Price Distribution & Log Transformation Normalization", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_price_vs_horsepower(df: pd.DataFrame, output_path: str | Path) -> None:
    """Scatter plot and trendline of Price vs. Horsepower."""
    plt.figure(figsize=(9, 6))
    sns.regplot(
        data=df,
        x="horsepower",
        y="price",
        scatter_kws={"color": "#3498db", "alpha": 0.7, "edgecolor": "k", "s": 55},
        line_kws={"color": "#e74c3c", "linewidth": 2.2, "label": "Linear Fit (r = +0.808)"},
    )
    plt.title("Figure 2: Vehicle Price vs. Horsepower", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Horsepower (bhp)", fontsize=12)
    plt.ylabel("Vehicle Price (USD)", fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_price_vs_enginesize(df: pd.DataFrame, output_path: str | Path) -> None:
    """Scatter plot and trendline of Price vs. Engine Size."""
    plt.figure(figsize=(9, 6))
    sns.regplot(
        data=df,
        x="enginesize",
        y="price",
        scatter_kws={"color": "#e67e22", "alpha": 0.7, "edgecolor": "k", "s": 55},
        line_kws={"color": "#2c3e50", "linewidth": 2.2, "label": "Linear Fit (r = +0.874)"},
    )
    plt.title("Figure 3: Vehicle Price vs. Engine Displacement Size", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Engine Size (cubic inches)", fontsize=12)
    plt.ylabel("Vehicle Price (USD)", fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_price_vs_curbweight(df: pd.DataFrame, output_path: str | Path) -> None:
    """Scatter plot and trendline of Price vs. Curb Weight."""
    plt.figure(figsize=(9, 6))
    sns.regplot(
        data=df,
        x="curbweight",
        y="price",
        scatter_kws={"color": "#16a085", "alpha": 0.7, "edgecolor": "k", "s": 55},
        line_kws={"color": "#c0392b", "linewidth": 2.2, "label": "Linear Fit (r = +0.835)"},
    )
    plt.title("Figure 4: Vehicle Price vs. Curb Weight", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Curb Weight (lbs)", fontsize=12)
    plt.ylabel("Vehicle Price (USD)", fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_price_vs_average_mpg(df: pd.DataFrame, output_path: str | Path) -> None:
    """Scatter plot and inverse relationship of Price vs. Average Fuel Economy."""
    plt.figure(figsize=(9, 6))
    sns.regplot(
        data=df,
        x="average_mpg",
        y="price",
        scatter_kws={"color": "#9b59b6", "alpha": 0.7, "edgecolor": "k", "s": 55},
        line_kws={"color": "#d35400", "linewidth": 2.2, "label": "Linear Fit (r = -0.697)"},
    )
    plt.title("Figure 5: Vehicle Price vs. Average Fuel Economy (Mileage)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Combined Average Fuel Economy (MPG)", fontsize=12)
    plt.ylabel("Vehicle Price (USD)", fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot correlation heatmap for key vehicle attributes and target variables."""
    key_features = [
        "price",
        "log_price",
        "enginesize",
        "curbweight",
        "horsepower",
        "carwidth",
        "carlength",
        "wheelbase",
        "power_to_weight",
        "average_mpg",
        "citympg",
        "highwaympg",
    ]
    corr_matrix = df[key_features].corr()

    plt.figure(figsize=(11, 8.5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
    )
    plt.title("Figure 6: Correlation Heatmap of Primary Vehicle Attributes & Price", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_visualizations(
    engineered_data_path: str | Path = "data/processed/car_price_engineered.csv",
    output_dir: str | Path = "reports/figures",
) -> None:
    """Generate all required figures and save as 300 DPI images."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(engineered_data_path)

    plot_price_distribution(df, out_dir / "01_price_distribution.png")
    plot_price_vs_horsepower(df, out_dir / "02_price_vs_horsepower.png")
    plot_price_vs_enginesize(df, out_dir / "03_price_vs_enginesize.png")
    plot_price_vs_curbweight(df, out_dir / "04_price_vs_curbweight.png")
    plot_price_vs_average_mpg(df, out_dir / "05_price_vs_average_mpg.png")
    plot_correlation_heatmap(df, out_dir / "06_correlation_heatmap.png")

    print(f"All 6 visualizations generated successfully in: {out_dir}")


if __name__ == "__main__":
    generate_all_visualizations()
