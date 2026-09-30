"""
Visualization Module for Task 2: Unemployment Analysis with Python.

Generates high-resolution, professional figures for:
1. Overall Unemployment Trend Over Time
2. COVID-19 Period Impact & Phase Disruption
3. State-Level Unemployment Disparity & Rankings
4. Rural vs. Urban Comparative Dynamics
5. Monthly Unemployment Patterns & Distributions
6. National Employment Workforce Volume Trend
7. Labour Participation Rate Trajectory
8. Multi-Phase COVID Metric Comparison
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np


# Setup consistent styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_overall_unemployment_trend(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """1. Overall monthly unemployment rate trend across India."""
    monthly_stats = df.groupby("date")["estimated_unemployment_rate_pct"].agg(
        mean="mean",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Shaded interquartile range across states
    ax.fill_between(
        monthly_stats["date"],
        monthly_stats["q25"],
        monthly_stats["q75"],
        color="#2b5c8f",
        alpha=0.2,
        label="Interquartile Range (25th–75th Percentile Across States)",
    )

    # Mean and Median lines
    ax.plot(
        monthly_stats["date"],
        monthly_stats["mean"],
        color="#1f4e79",
        linewidth=2.5,
        marker="o",
        label="National Mean Unemployment Rate (%)",
    )
    ax.plot(
        monthly_stats["date"],
        monthly_stats["median"],
        color="#d9534f",
        linewidth=2,
        linestyle="--",
        marker="s",
        label="National Median Unemployment Rate (%)",
    )

    # Lockdown annotation
    ax.axvline(pd.Timestamp("2020-03-24"), color="#b22222", linestyle=":", linewidth=1.8, label="Lockdown Imposed (Mar 24, 2020)")
    ax.annotate(
        "Nationwide Lockdown\nImposed (Mar 24, 2020)",
        xy=(pd.Timestamp("2020-03-24"), 20),
        xytext=(pd.Timestamp("2019-11-01"), 22),
        arrowprops=dict(facecolor="#b22222", shrink=0.05, width=1.5, headwidth=8),
        fontsize=10,
        fontweight="bold",
        color="#800000",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2f2", edgecolor="#b22222", alpha=0.9),
    )

    ax.set_title("Overall Monthly Unemployment Rate Trend in India (May 2019 – June 2020)", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Observation Month", fontsize=11, labelpad=10)
    ax.set_ylabel("Unemployment Rate (%)", fontsize=11, labelpad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha="right")
    ax.set_ylim(0, 32)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "01_overall_unemployment_trend.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_covid_impact_trend(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """2. COVID-19 Period Impact Highlighting Pre-COVID Baseline vs Lockdown Surge."""
    monthly_stats = df.groupby(["date", "period"])["estimated_unemployment_rate_pct"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Highlight Pre-COVID zone vs COVID zone
    ax.axvspan(pd.Timestamp("2019-05-15"), pd.Timestamp("2020-02-29"), color="#e8f4f8", alpha=0.7, label="Pre-COVID Baseline (Mean: 9.51%)")
    ax.axvspan(pd.Timestamp("2020-02-29"), pd.Timestamp("2020-06-30"), color="#ffebee", alpha=0.7, label="COVID Shock & Lockdown (Mean: 17.77%)")

    # Plot lines with color based on period
    pre_data = monthly_stats[monthly_stats["period"] == "Pre-COVID"]
    covid_data = monthly_stats[monthly_stats["period"] == "COVID-Period"]

    ax.plot(pre_data["date"], pre_data["estimated_unemployment_rate_pct"], color="#0d47a1", marker="o", linewidth=2.5, label="Pre-COVID Mean Rate")
    
    # Bridge point between Feb and Mar
    bridge_dates = [pre_data["date"].iloc[-1], covid_data["date"].iloc[0]]
    bridge_vals = [pre_data["estimated_unemployment_rate_pct"].iloc[-1], covid_data["estimated_unemployment_rate_pct"].iloc[0]]
    ax.plot(bridge_dates, bridge_vals, color="#c62828", linestyle="--", linewidth=2)

    ax.plot(covid_data["date"], covid_data["estimated_unemployment_rate_pct"], color="#c62828", marker="D", linewidth=2.8, label="COVID-Period Mean Rate")

    # Annotate peak
    peak_row = covid_data.loc[covid_data["estimated_unemployment_rate_pct"].idxmax()]
    ax.annotate(
        f"Peak Lockdown: {peak_row['estimated_unemployment_rate_pct']:.2f}%\n(May 2020)",
        xy=(peak_row["date"], peak_row["estimated_unemployment_rate_pct"]),
        xytext=(pd.Timestamp("2020-02-15"), 26),
        arrowprops=dict(facecolor="#c62828", shrink=0.08, width=1.5, headwidth=8),
        fontsize=10,
        fontweight="bold",
        color="#8e0000",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffebee", edgecolor="#c62828", alpha=0.9),
    )

    ax.set_title("COVID-19 Disruption on Unemployment Rate: Pre-COVID vs. Pandemic Months", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Month", fontsize=11, labelpad=10)
    ax.set_ylabel("Average Unemployment Rate (%)", fontsize=11, labelpad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha="right")
    ax.set_ylim(0, 30)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "02_covid_impact_unemployment_trend.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_state_comparison(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """3. State/Region comparison: Pre-COVID vs Peak Lockdown."""
    pre_covid = df[df["period"] == "Pre-COVID"].groupby("region")["estimated_unemployment_rate_pct"].mean()
    peak_lockdown = df[df["covid_phase"] == "Peak Lockdown (Apr-May 2020)"].groupby("region")["estimated_unemployment_rate_pct"].mean()

    comp_df = pd.DataFrame({"Pre-COVID Mean": pre_covid, "Peak Lockdown Mean": peak_lockdown}).dropna()
    comp_df.sort_values(by="Peak Lockdown Mean", ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 12), dpi=300)
    y_indices = np.arange(len(comp_df))
    height = 0.38

    ax.barh(y_indices - height/2, comp_df["Pre-COVID Mean"], height=height, color="#1976d2", alpha=0.85, label="Pre-COVID Mean (%)")
    ax.barh(y_indices + height/2, comp_df["Peak Lockdown Mean"], height=height, color="#d32f2f", alpha=0.85, label="Peak Lockdown Mean (Apr–May 2020) (%)")

    ax.set_yticks(y_indices)
    ax.set_yticklabels(comp_df.index, fontsize=9.5)
    ax.set_xlabel("Unemployment Rate (%)", fontsize=11, labelpad=10)
    ax.set_title("State-Level Unemployment Rate Comparison: Pre-COVID Baseline vs. Peak Lockdown", fontsize=13, pad=15, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.set_xlim(0, 85)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "03_state_unemployment_comparison.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_rural_vs_urban(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """4. Rural vs Urban comparative monthly trajectory and distribution."""
    area_monthly = df.groupby(["date", "area"])["estimated_unemployment_rate_pct"].mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300, gridspec_kw={"width_ratios": [2.2, 1]})

    # Trajectory plot
    rural = area_monthly[area_monthly["area"] == "Rural"]
    urban = area_monthly[area_monthly["area"] == "Urban"]

    ax1.plot(rural["date"], rural["estimated_unemployment_rate_pct"], color="#2e7d32", marker="o", linewidth=2.5, label="Rural Areas")
    ax1.plot(urban["date"], urban["estimated_unemployment_rate_pct"], color="#e65100", marker="s", linewidth=2.5, label="Urban Areas")
    ax1.axvline(pd.Timestamp("2020-03-24"), color="#b22222", linestyle=":", linewidth=1.5, label="Lockdown (Mar 24, 2020)")

    ax1.set_title("Monthly Unemployment Rate: Rural vs. Urban (May 2019 – June 2020)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Month", fontsize=10)
    ax1.set_ylabel("Unemployment Rate (%)", fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.set_ylim(0, 32)
    ax1.legend(loc="upper left", frameon=True)

    # Boxplot comparison by period
    sns.boxplot(
        data=df,
        x="area",
        y="estimated_unemployment_rate_pct",
        hue="period",
        palette=["#64b5f6", "#e57373"],
        ax=ax2,
        boxprops=dict(alpha=0.85),
    )
    ax2.set_title("Unemployment Distribution by Area & Period", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Geographical Area", fontsize=10)
    ax2.set_ylabel("Unemployment Rate (%)", fontsize=10)
    ax2.legend(title="Period", loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "04_rural_vs_urban_comparison.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_monthly_patterns(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """5. Monthly unemployment distribution boxplots across all months."""
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)

    # Order chronologically
    chronological_months = sorted(df["year_month"].unique())

    palette = [
        "#90caf9" if m < "2020-03" else "#ef9a9a" if m in ["2020-04", "2020-05"] else "#ffe082"
        for m in chronological_months
    ]

    sns.boxplot(
        data=df,
        x="year_month",
        y="estimated_unemployment_rate_pct",
        hue="year_month",
        palette=palette,
        legend=False,
        ax=ax,
        fliersize=3,
    )

    ax.set_title("Cross-State Unemployment Rate Distribution by Month (May 2019 – June 2020)", fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Observation Month (YYYY-MM)", fontsize=11, labelpad=10)
    ax.set_ylabel("Unemployment Rate (%)", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    ax.set_ylim(-1, 80)

    # Custom legend for color codes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#90caf9", label="Pre-COVID Baseline Months"),
        Patch(facecolor="#ef9a9a", label="Peak Lockdown Shock (Apr–May 2020)"),
        Patch(facecolor="#ffe082", label="Transition Months (Mar 2020 / Jun 2020)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "05_monthly_unemployment_patterns.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_employment_trend(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """6. Total estimated national employment workforce trend."""
    emp_monthly = df.groupby(["date", "year_month"])["estimated_employed"].sum().reset_index()
    emp_monthly["employed_millions"] = emp_monthly["estimated_employed"] / 1e6

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    ax.plot(
        emp_monthly["date"],
        emp_monthly["employed_millions"],
        color="#00695c",
        linewidth=2.8,
        marker="o",
        label="Total Active Workforce (Millions)",
    )

    ax.axvline(pd.Timestamp("2020-03-24"), color="#b22222", linestyle=":", linewidth=1.5, label="Lockdown (Mar 24, 2020)")

    # Highlight drop
    pre_drop = emp_monthly.loc[emp_monthly["year_month"] == "2020-02", "employed_millions"].values[0]
    trough = emp_monthly.loc[emp_monthly["year_month"] == "2020-04", "employed_millions"].values[0]
    drop_pct = ((trough - pre_drop) / pre_drop) * 100

    ax.annotate(
        f"Workforce contraction:\n{pre_drop:.1f}M -> {trough:.1f}M ({drop_pct:.1f}%)\n(Apr 2020)",
        xy=(pd.Timestamp("2020-04-30"), trough),
        xytext=(pd.Timestamp("2019-10-01"), 290),
        arrowprops=dict(facecolor="#00695c", shrink=0.08, width=1.5, headwidth=8),
        fontsize=10,
        fontweight="bold",
        color="#004d40",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e0f2f1", edgecolor="#00695c", alpha=0.9),
    )

    ax.set_title("Total National Workforce Employment Volume Over Time (May 2019 – June 2020)", fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Observation Month", fontsize=11, labelpad=10)
    ax.set_ylabel("Employed Individuals (Millions)", fontsize=11, labelpad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha="right")
    ax.set_ylim(240, 430)
    ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "06_employment_trend.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_labour_participation_trend(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """7. Labour participation rate trajectory over time."""
    lpr_monthly = df.groupby("date")["estimated_labour_participation_rate_pct"].agg(["mean", "median"]).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    ax.plot(
        lpr_monthly["date"],
        lpr_monthly["mean"],
        color="#6a1b9a",
        linewidth=2.5,
        marker="o",
        label="Mean Labour Participation Rate (%)",
    )
    ax.plot(
        lpr_monthly["date"],
        lpr_monthly["median"],
        color="#ab47bc",
        linewidth=2,
        linestyle="--",
        marker="^",
        label="Median Labour Participation Rate (%)",
    )

    ax.axvline(pd.Timestamp("2020-03-24"), color="#b22222", linestyle=":", linewidth=1.5, label="Lockdown (Mar 24, 2020)")

    ax.set_title("Labour Force Participation Rate Trajectory (May 2019 – June 2020)", fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Observation Month", fontsize=11, labelpad=10)
    ax.set_ylabel("Labour Participation Rate (%)", fontsize=11, labelpad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha="right")
    ax.set_ylim(32, 48)
    ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "07_labour_participation_trend.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_covid_phase_comparison(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """8. Multi-Phase COVID Metric Comparison: Unemployment, Employment, Participation."""
    phase_order = [
        "Pre-COVID Baseline",
        "Early Lockdown (Mar 2020)",
        "Peak Lockdown (Apr-May 2020)",
        "Early Unlock (Jun 2020)",
    ]

    phase_stats = df.groupby("covid_phase").agg(
        unemployment=("estimated_unemployment_rate_pct", "mean"),
        employed_millions=("estimated_employed", lambda x: x.mean() / 1e6),
        participation=("estimated_labour_participation_rate_pct", "mean"),
    ).reindex(phase_order)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 1. Unemployment
    axes[0].bar(phase_stats.index, phase_stats["unemployment"], color=["#1976d2", "#ffb300", "#d32f2f", "#43a047"], alpha=0.85)
    axes[0].set_title("Mean Unemployment Rate (%)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(range(len(phase_stats)))
    axes[0].set_xticklabels(["Pre-COVID", "Early Lockdown", "Peak Lockdown", "Early Unlock"], rotation=25, ha="right")
    for i, v in enumerate(phase_stats["unemployment"]):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9.5)
    axes[0].set_ylim(0, 30)

    # 2. Employment
    axes[1].bar(phase_stats.index, phase_stats["employed_millions"], color=["#1976d2", "#ffb300", "#d32f2f", "#43a047"], alpha=0.85)
    axes[1].set_title("Mean State Employment (Millions)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Employed (Millions)")
    axes[1].set_xticks(range(len(phase_stats)))
    axes[1].set_xticklabels(["Pre-COVID", "Early Lockdown", "Peak Lockdown", "Early Unlock"], rotation=25, ha="right")
    for i, v in enumerate(phase_stats["employed_millions"]):
        axes[1].text(i, v + 0.15, f"{v:.2f}M", ha="center", fontweight="bold", fontsize=9.5)
    axes[1].set_ylim(0, 9)

    # 3. Participation
    axes[2].bar(phase_stats.index, phase_stats["participation"], color=["#1976d2", "#ffb300", "#d32f2f", "#43a047"], alpha=0.85)
    axes[2].set_title("Mean Labour Participation (%)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Rate (%)")
    axes[2].set_xticks(range(len(phase_stats)))
    axes[2].set_xticklabels(["Pre-COVID", "Early Lockdown", "Peak Lockdown", "Early Unlock"], rotation=25, ha="right")
    for i, v in enumerate(phase_stats["participation"]):
        axes[2].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9.5)
    axes[2].set_ylim(0, 50)

    fig.suptitle("Key Macroeconomic Labour Metrics Across COVID-19 Pandemic Phases in India", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "08_covid_phase_comparison.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def generate_all_visualizations(df: pd.DataFrame, output_dir: str = "reports/figures") -> list:
    """Generates all required visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    generated.append(plot_overall_unemployment_trend(df, output_dir))
    generated.append(plot_covid_impact_trend(df, output_dir))
    generated.append(plot_state_comparison(df, output_dir))
    generated.append(plot_rural_vs_urban(df, output_dir))
    generated.append(plot_monthly_patterns(df, output_dir))
    generated.append(plot_employment_trend(df, output_dir))
    generated.append(plot_labour_participation_trend(df, output_dir))
    generated.append(plot_covid_phase_comparison(df, output_dir))
    return generated


if __name__ == "__main__":
    from src.data_cleaning import clean_primary_dataset
    df = clean_primary_dataset()
    created_plots = generate_all_visualizations(df)
    print(f"Generated {len(created_plots)} figures in reports/figures/:")
    for p in created_plots:
        print(f" - {p}")
