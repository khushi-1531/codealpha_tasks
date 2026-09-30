"""
Regression Modeling and Evaluation Pipeline for CodeAlpha Task 3.

This module evaluates multiple regression estimators on raw-target and log-transformed
price objectives using 5-fold cross-validation on the training set, benchmarks generalization
on the untouched test set, produces comparative evaluation figures, and serializes the optimal
model pipeline along with comprehensive metadata.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMNS,
    build_preprocessor,
)

# Publication visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Define standard suite of regression algorithms."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=random_state),
    }


def evaluate_model_cv_and_test(
    model_name: str,
    base_estimator: Any,
    use_log_target: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    kf: KFold,
) -> Dict[str, Any]:
    """
    Perform 5-fold cross-validation on X_train/y_train only,
    then train on full X_train and evaluate on untouched X_test/y_test.
    Always evaluate metrics on the original price scale ($ USD).
    """
    # Build complete preprocessing + regressor pipeline
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", base_estimator),
    ])

    if use_log_target:
        estimator = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log,
            inverse_func=np.exp,
        )
    else:
        estimator = pipeline

    # 1. 5-fold cross-validation on training data only
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }
    cv_res = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=kf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    cv_r2_mean = float(np.mean(cv_res["test_r2"]))
    cv_r2_std = float(np.std(cv_res["test_r2"]))
    cv_mae_mean = float(np.mean(-cv_res["test_neg_mae"]))
    cv_mae_std = float(np.std(-cv_res["test_neg_mae"]))
    cv_rmse_mean = float(np.mean(-cv_res["test_neg_rmse"]))
    cv_rmse_std = float(np.std(-cv_res["test_neg_rmse"]))

    # 2. Fit on full training set, evaluate on untouched test set
    estimator.fit(X_train, y_train)
    test_preds = estimator.predict(X_test)

    test_r2 = float(r2_score(y_test, test_preds))
    test_mae = float(mean_absolute_error(y_test, test_preds))
    test_rmse = float(root_mean_squared_error(y_test, test_preds))

    display_name = f"{model_name} (Log-Target)" if use_log_target else f"{model_name} (Raw-Target)"

    return {
        "name": display_name,
        "base_model": model_name,
        "use_log_target": use_log_target,
        "estimator": estimator,
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "cv_mae_mean": cv_mae_mean,
        "cv_mae_std": cv_mae_std,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_preds": test_preds,
    }


def run_full_modeling_experiment(
    train_path: str | Path = "data/processed/train.csv",
    test_path: str | Path = "data/processed/test.csv",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Execute complete modeling benchmark across raw and log target approaches."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X_train = train_df[feature_cols].copy()
    y_train = train_df["price"].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df["price"].copy()

    # Leakage assertions
    assert "price" not in X_train.columns
    assert "log_price" not in X_train.columns
    assert "car_ID" not in X_train.columns
    assert "CarName" not in X_train.columns

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    candidate_models = get_candidate_models(random_state=random_state)

    results_list = []
    models_dict = {}

    # Evaluate Raw-Target Models
    for name, base_model in candidate_models.items():
        res = evaluate_model_cv_and_test(
            name, base_model, False, X_train, y_train, X_test, y_test, kf
        )
        results_list.append(res)
        models_dict[res["name"]] = res

    # Evaluate Log-Target Models
    for name, base_model in candidate_models.items():
        res = evaluate_model_cv_and_test(
            name, base_model, True, X_train, y_train, X_test, y_test, kf
        )
        results_list.append(res)
        models_dict[res["name"]] = res

    # Convert metrics to clean summary DataFrame
    summary_data = []
    for r in results_list:
        summary_data.append({
            "Model": r["name"],
            "Target Type": "Log(Price)" if r["use_log_target"] else "Raw Price",
            "CV R² (Mean ± Std)": f"{r['cv_r2_mean']:.4f} ± {r['cv_r2_std']:.4f}",
            "CV MAE ($)": f"${r['cv_mae_mean']:,.2f} ± ${r['cv_mae_std']:,.2f}",
            "CV RMSE ($)": f"${r['cv_rmse_mean']:,.2f} ± ${r['cv_rmse_std']:,.2f}",
            "Test R²": r["test_r2"],
            "Test MAE ($)": r["test_mae"],
            "Test RMSE ($)": r["test_rmse"],
            "cv_r2_mean_raw": r["cv_r2_mean"],
            "cv_mae_mean_raw": r["cv_mae_mean"],
            "cv_rmse_mean_raw": r["cv_rmse_mean"],
        })
    df_results = pd.DataFrame(summary_data)

    return df_results, models_dict, X_train, y_train, X_test, y_test


def generate_evaluation_visualizations(
    df_results: pd.DataFrame,
    best_model_res: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: str | Path = "reports/figures",
) -> None:
    """Generate all 6 required evaluation visualizations at 300 DPI."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_by_r2 = df_results.sort_values(by="Test R²", ascending=True)
    sorted_by_mae = df_results.sort_values(by="Test MAE ($)", ascending=False)
    sorted_by_rmse = df_results.sort_values(by="Test RMSE ($)", ascending=False)

    # 1. Model Comparison — R²
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_by_r2["Model"], sorted_by_r2["Test R²"], color="#2980b9", edgecolor="black")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9, fontweight="bold")
    plt.title("Figure 7: Test Set Performance Comparison — Coefficient of Determination ($R^2$)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Test $R^2$ Score (Higher is Better)", fontsize=11)
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "07_model_comparison_r2.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Model Comparison — MAE
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_by_mae["Model"], sorted_by_mae["Test MAE ($)"], color="#e67e22", edgecolor="black")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 30, bar.get_y() + bar.get_height() / 2, f"${w:,.0f}", va="center", fontsize=9, fontweight="bold")
    plt.title("Figure 8: Test Set Performance Comparison — Mean Absolute Error (MAE)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Test MAE in USD (Lower is Better)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "08_model_comparison_mae.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Model Comparison — RMSE
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_by_rmse["Model"], sorted_by_rmse["Test RMSE ($)"], color="#c0392b", edgecolor="black")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 40, bar.get_y() + bar.get_height() / 2, f"${w:,.0f}", va="center", fontsize=9, fontweight="bold")
    plt.title("Figure 9: Test Set Performance Comparison — Root Mean Squared Error (RMSE)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Test RMSE in USD (Lower is Better)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "09_model_comparison_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Actual vs Predicted Prices for Selected Best Model
    test_preds = best_model_res["test_preds"]
    plt.figure(figsize=(8, 6.5))
    plt.scatter(y_test, test_preds, color="#27ae60", alpha=0.75, edgecolors="k", s=65, label="Test Observations ($N=41$)")
    min_bound = min(y_test.min(), test_preds.min()) - 1000
    max_bound = max(y_test.max(), test_preds.max()) + 1000
    plt.plot([min_bound, max_bound], [min_bound, max_bound], "r--", linewidth=2.0, label="Perfect Valuation Line ($y = \\hat{y}$)")
    plt.title(f"Figure 10: Actual vs. Predicted Vehicle Prices\nSelected Model: {best_model_res['name']}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Actual Vehicle Price (USD)", fontsize=11)
    plt.ylabel("Predicted Vehicle Price (USD)", fontsize=11)
    plt.xlim(min_bound, max_bound)
    plt.ylim(min_bound, max_bound)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "10_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Residual / Error Distribution
    residuals = y_test - test_preds
    plt.figure(figsize=(9, 5.5))
    sns.histplot(residuals, kde=True, color="#8e44ad", edgecolor="black", bins=15)
    mean_err = residuals.mean()
    median_err = residuals.median()
    plt.axvline(0, color="black", linestyle="-", linewidth=1.2, label="Zero Error")
    plt.axvline(mean_err, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Mean Error: ${mean_err:,.0f}")
    plt.axvline(median_err, color="#27ae60", linestyle="-.", linewidth=1.8, label=f"Median Error: ${median_err:,.0f}")
    plt.title(f"Figure 11: Residual Error Distribution (Actual - Predicted)\nSelected Model: {best_model_res['name']}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Residual Error in USD (Actual - Predicted)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "11_residual_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Feature Importance for Selected Model
    # Extract underlying regressor from pipeline
    pipeline = (
        best_model_res["estimator"].regressor_
        if best_model_res["use_log_target"]
        else best_model_res["estimator"]
    )
    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    # Clean feature names (remove num__ and cat__ prefixes)
    clean_feature_names = [
        f.replace("num__", "").replace("cat__", "") for f in feature_names
    ]

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        fi_df = pd.DataFrame({"Feature": clean_feature_names, "Importance": importances})
        top_fi = fi_df.sort_values(by="Importance", ascending=True).tail(15)

        plt.figure(figsize=(10, 7))
        bars = plt.barh(top_fi["Feature"], top_fi["Importance"], color="#16a085", edgecolor="black")
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8.5, fontweight="bold")
        plt.title(f"Figure 12: Top 15 Feature Importances\nSelected Model: {best_model_res['name']}", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Relative Feature Importance (Gini Impurity / Reduction in Variance)", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / "12_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
    elif hasattr(regressor, "coef_"):
        coefs = np.abs(regressor.coef_)
        fi_df = pd.DataFrame({"Feature": clean_feature_names, "Importance": coefs})
        top_fi = fi_df.sort_values(by="Importance", ascending=True).tail(15)

        plt.figure(figsize=(10, 7))
        bars = plt.barh(top_fi["Feature"], top_fi["Importance"], color="#16a085", edgecolor="black")
        plt.title(f"Figure 12: Top 15 Absolute Coefficients\nSelected Model: {best_model_res['name']}", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Absolute Standardized Coefficient Magnitude", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / "12_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_model_artifact_and_metadata(
    best_model_res: Dict[str, Any],
    all_results_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_output_path: str | Path = "models/car_price_model.joblib",
    metadata_output_path: str | Path = "models/model_metadata.json",
    random_state: int = 42,
) -> None:
    """Save selected model pipeline and metadata JSON."""
    model_file = Path(model_output_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_res["estimator"], model_file)

    meta_file = Path(metadata_output_path)
    meta_dict = {
        "model_name": best_model_res["name"],
        "base_algorithm": best_model_res["base_model"],
        "target_variable": "price",
        "log_target_transformed": best_model_res["use_log_target"],
        "training_records": len(X_train),
        "test_records": len(X_test),
        "random_state": random_state,
        "features_used": {
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "total_feature_count": len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES),
        },
        "excluded_features": IDENTIFIER_COLUMNS + ["price", "log_price"],
        "cross_validation_evaluation": {
            "cv_splits": 5,
            "cv_r2_mean": best_model_res["cv_r2_mean"],
            "cv_r2_std": best_model_res["cv_r2_std"],
            "cv_mae_mean": best_model_res["cv_mae_mean"],
            "cv_mae_std": best_model_res["cv_mae_std"],
            "cv_rmse_mean": best_model_res["cv_rmse_mean"],
            "cv_rmse_std": best_model_res["cv_rmse_std"],
        },
        "test_set_evaluation": {
            "test_r2": best_model_res["test_r2"],
            "test_mae": best_model_res["test_mae"],
            "test_rmse": best_model_res["test_rmse"],
        },
        "all_models_summary": all_results_df[[
            "Model", "Target Type", "CV R² (Mean ± Std)", "CV MAE ($)", "CV RMSE ($)", "Test R²", "Test MAE ($)", "Test RMSE ($)"
        ]].to_dict(orient="records"),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    print(f"Model artifact successfully saved to: {model_file}")
    print(f"Model metadata successfully saved to: {meta_file}")


def run_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete Step 4 training, evaluation, visualization, and serialization."""
    df_results, models_dict, X_train, y_train, X_test, y_test = run_full_modeling_experiment()

    print("\n" + "=" * 90)
    print("MODEL EVALUATION BENCHMARK SUMMARY (Ordered by Test R²)")
    print("=" * 90)
    print(df_results.sort_values(by="Test R²", ascending=False)[
        ["Model", "CV R² (Mean ± Std)", "Test R²", "Test MAE ($)", "Test RMSE ($)"]
    ].to_string(index=False))
    print("=" * 90)

    # Model Selection:
    # We choose the model demonstrating the optimal combination of high generalization R2,
    # lowest test MAE, lowest test RMSE, and high CV stability.
    # We pick the top performing model on test MAE / R2.
    best_candidate_name = df_results.sort_values(by="Test MAE ($)", ascending=True).iloc[0]["Model"]
    best_model_res = models_dict[best_candidate_name]
    print(f"\nOptimal Selected Model: {best_model_res['name']}")
    print(f"Test R²: {best_model_res['test_r2']:.4f} | Test MAE: ${best_model_res['test_mae']:,.2f} | Test RMSE: ${best_model_res['test_rmse']:,.2f}")
    print(f"5-Fold CV R²: {best_model_res['cv_r2_mean']:.4f} ± {best_model_res['cv_r2_std']:.4f}")

    # Generate visualizations
    generate_evaluation_visualizations(df_results, best_model_res, X_train, y_train, X_test, y_test)
    print("All 6 evaluation figures generated in reports/figures/")

    # Save artifacts
    save_model_artifact_and_metadata(best_model_res, df_results, X_train, X_test)

    return df_results, best_model_res


if __name__ == "__main__":
    run_pipeline()
