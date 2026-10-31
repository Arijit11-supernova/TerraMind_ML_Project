"""Analyze classification thresholds using validation data only."""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delhi_rainfall_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluation"
    / "validation_threshold_analysis.csv"
)

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]


def main():
    print("=" * 75)
    print("TERRAMIND ML: VALIDATION THRESHOLD ANALYSIS")
    print("=" * 75)

    data = pd.read_csv(DATA_FILE)

    validation = data[
        data["period"] == "validation"
    ].copy()

    if validation.empty:
        raise ValueError("Validation dataset is empty.")

    y_true = validation["target_heavy_rainfall"].astype(int)

    all_results = []

    for model_name in [
        "logistic_regression",
        "random_forest",
    ]:
        model_path = MODEL_DIR / f"{model_name}.joblib"

        artifact = joblib.load(model_path)

        model = artifact["model"]
        features = artifact["features"]

        X_val = validation[features]

        probabilities = model.predict_proba(X_val)[:, 1]

        print(f"\n{'-' * 75}")
        print(f"MODEL: {model_name}")
        print(f"Validation observations: {len(validation):,}")
        print(f"Actual positive cases: {int(y_true.sum()):,}")
        print(
            "Validation PR-AUC: "
            f"{average_precision_score(y_true, probabilities):.4f}"
        )
        print("-" * 75)

        print(
            f"{'Threshold':>10} "
            f"{'Precision':>11} "
            f"{'Recall':>10} "
            f"{'F1':>10} "
            f"{'Predicted+':>12} "
            f"{'TP':>6} "
            f"{'FP':>6} "
            f"{'FN':>6}"
        )

        for threshold in THRESHOLDS:
            predictions = (
                probabilities >= threshold
            ).astype(int)

            tp = int(
                ((predictions == 1) & (y_true == 1)).sum()
            )
            fp = int(
                ((predictions == 1) & (y_true == 0)).sum()
            )
            fn = int(
                ((predictions == 0) & (y_true == 1)).sum()
            )

            precision = precision_score(
                y_true,
                predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_true,
                predictions,
                zero_division=0,
            )

            f1 = f1_score(
                y_true,
                predictions,
                zero_division=0,
            )

            print(
                f"{threshold:10.2f} "
                f"{precision:11.4f} "
                f"{recall:10.4f} "
                f"{f1:10.4f} "
                f"{int(predictions.sum()):12,} "
                f"{tp:6,} "
                f"{fp:6,} "
                f"{fn:6,}"
            )

            all_results.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "predicted_positive": int(predictions.sum()),
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "validation_pr_auc": average_precision_score(
                        y_true,
                        probabilities,
                    ),
                }
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(all_results).to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSaved validation analysis:")
    print(OUTPUT_FILE)
    print("\nNo test-set observations were used in threshold selection.")
    print("=" * 75)


if __name__ == "__main__":
    main()