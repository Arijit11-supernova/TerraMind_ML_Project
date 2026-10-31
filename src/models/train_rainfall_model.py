"""Train and compare baseline models for next-day heavy rainfall."""

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delhi_rainfall_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "evaluation"

MLFLOW_DIR = PROJECT_ROOT / "artifacts" / "mlflow"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "TerraMind-Heavy-Rainfall-Prediction"

TARGET = "target_heavy_rainfall"

FEATURES = [
    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_mean_3d",
    "rainfall_mean_7d",
    "rainfall_max_7d",
    "latitude",
    "longitude",
    "target_month",
    "target_day_sin",
    "target_day_cos",
    "regional_mean_rainfall_mm",
    "regional_max_rainfall_mm",
    "cells_over_10mm",
    "cells_over_35mm",
]


def choose_threshold(y_true, probabilities):
    """Choose a probability threshold using validation F1 only."""

    best_threshold = 0.50
    best_f1 = -1.0

    # Search thresholds using validation data only.
    thresholds = np.arange(0.05, 0.951, 0.01)

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def evaluate_predictions(y_true, probabilities, threshold):
    """Calculate metrics for a selected classification threshold."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def print_metrics(name, metrics):
    """Print evaluation results in a readable format."""

    print(f"\n{name}")
    print("-" * 60)
    print(f"Decision threshold: {metrics['threshold']:.2f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1-score:           {metrics['f1']:.4f}")
    print(f"PR-AUC:             {metrics['pr_auc']:.4f}")

    print("\nConfusion matrix counts:")
    print(f"True negatives:  {metrics['true_negatives']:,}")
    print(f"False positives: {metrics['false_positives']:,}")
    print(f"False negatives: {metrics['false_negatives']:,}")
    print(f"True positives:  {metrics['true_positives']:,}")


def main():
    print("=" * 75)
    print("TERRAMIND ML: MODEL TRAINING AND EVALUATION")
    print("=" * 75)

    # 1. Configure MLflow.
    MLFLOW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")

    # 2. Load the engineered dataset.
    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["date", "target_date"],
    )

    if data[FEATURES + [TARGET]].isna().any().any():
        raise ValueError(
            "Missing feature or target values detected."
        )

    # 3. Separate data by the date being predicted.
    train = data[data["period"] == "train"].copy()
    validation = data[data["period"] == "validation"].copy()
    test = data[data["period"] == "test"].copy()

    for name, subset in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        if subset.empty:
            raise ValueError(f"{name} dataset is empty.")

        print(
            f"{name:12} rows={len(subset):,}, "
            f"positives={int(subset[TARGET].sum()):,}, "
            f"positive_rate={subset[TARGET].mean():.4%}"
        )

    # Confirm that the chronological split is correct.
    assert train["target_date"].dt.year.max() <= 2018
    assert validation["target_date"].dt.year.min() >= 2019
    assert validation["target_date"].dt.year.max() <= 2021
    assert test["target_date"].dt.year.min() >= 2022

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val = validation[FEATURES]
    y_val = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # 4. Evaluate the majority-class baseline.
    # It predicts no positive cases.
    majority_probabilities = np.zeros(len(y_test))
    majority_predictions = np.zeros(
        len(y_test),
        dtype=int,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        majority_predictions,
        labels=[0, 1],
    ).ravel()

    baseline_metrics = {
        "threshold": 0.50,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "pr_auc": average_precision_score(
            y_test,
            majority_probabilities,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    print_metrics(
        "MAJORITY-CLASS BASELINE (TEST)",
        baseline_metrics,
    )

    # Log the baseline as a separate MLflow run.
    with mlflow.start_run(run_name="majority_class_baseline"):
        mlflow.log_param("model_name", "majority_class_baseline")
        mlflow.log_param("target", TARGET)
        mlflow.log_param("rainfall_threshold_mm", 35.0)
        mlflow.log_param("feature_count", len(FEATURES))
        mlflow.log_param("test_period", "2022-2024")

        for metric_name, metric_value in baseline_metrics.items():
            mlflow.log_metric(
                f"test_{metric_name}",
                float(metric_value),
            )

        print(
            "MLflow baseline run ID: "
            f"{mlflow.active_run().info.run_id}"
        )

    # 5. Define the candidate models.
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    # 6. Train, validate, evaluate, save, and track each model.
    for model_name, model in models.items():
        print("\n" + "=" * 75)
        print(f"TRAINING: {model_name}")
        print("=" * 75)

        with mlflow.start_run(run_name=model_name):

            # Log experiment configuration.
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("target", TARGET)
            mlflow.log_param("rainfall_threshold_mm", 35.0)
            mlflow.log_param("feature_count", len(FEATURES))
            mlflow.log_param("training_period", "2000-2018")
            mlflow.log_param("validation_period", "2019-2021")
            mlflow.log_param("test_period", "2022-2024")
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("validation_rows", len(X_val))
            mlflow.log_param("test_rows", len(X_test))

            # Log model hyperparameters.
            mlflow.log_params(model.get_params())

            # Fit on training data only.
            model.fit(X_train, y_train)

            # Choose the decision threshold using validation data.
            val_probabilities = model.predict_proba(X_val)[:, 1]

            threshold, validation_f1 = choose_threshold(
                y_val,
                val_probabilities,
            )

            # Calculate validation PR-AUC without using the test split.
            validation_pr_auc = average_precision_score(
                y_val,
                val_probabilities,
            )

            mlflow.log_metric(
                "validation_selected_threshold",
                threshold,
            )
            mlflow.log_metric(
                "validation_f1",
                validation_f1,
            )
            mlflow.log_metric(
                "validation_pr_auc",
                float(validation_pr_auc),
            )

            print(
                f"\nValidation-selected threshold: {threshold:.2f}"
            )
            print(
                "Validation F1 at selected threshold: "
                f"{validation_f1:.4f}"
            )
            print(
                f"Validation PR-AUC: {validation_pr_auc:.4f}"
            )

            # Evaluate once on the untouched test split.
            test_probabilities = model.predict_proba(X_test)[:, 1]

            metrics = evaluate_predictions(
                y_test,
                test_probabilities,
                threshold,
            )

            print_metrics(
                f"{model_name.upper()} (TEST)",
                metrics,
            )

            # Log test metrics to MLflow.
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(
                    f"test_{metric_name}",
                    float(metric_value),
                )

            # Add exactly one comparison row per candidate model.
            all_results.append(
                {
                    "model": model_name,
                    **metrics,
                    "validation_f1_at_selected_threshold": validation_f1,
                }
            )

            # Save the fitted model and its decision threshold.
            model_artifact = {
                "model": model,
                "features": FEATURES,
                "target": TARGET,
                "rainfall_threshold_mm": 35.0,
                "decision_threshold": threshold,
                "training_period": "2000-2018",
                "validation_period": "2019-2021",
                "test_period": "2022-2024",
            }

            model_path = MODEL_DIR / f"{model_name}.joblib"

            joblib.dump(
                model_artifact,
                model_path,
            )

            print(f"\nSaved model: {model_path}")

            # Record the saved artifact in this MLflow run.
            mlflow.log_artifact(
                str(model_path),
                artifact_path="model_artifacts",
            )

            print(
                "MLflow run ID: "
                f"{mlflow.active_run().info.run_id}"
            )

    # 7. Save a machine-readable model comparison.
    results = pd.DataFrame(all_results)

    results_path = (
        RESULTS_DIR
        / "rainfall_model_comparison.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 75)
    print("MODEL COMPARISON SUMMARY (TEST SET)")
    print("=" * 75)

    print(
        results[
            [
                "model",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "threshold",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSaved comparison: {results_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()