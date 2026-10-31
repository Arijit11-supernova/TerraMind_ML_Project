from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = Path("data/processed/delhi_rainfall_features.csv")
OUTPUT_DIR = Path("artifacts/evaluation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_heavy_rainfall"

# Group A: Historical rainfall only.
HISTORICAL_FEATURES = [
    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_mean_3d",
    "rainfall_mean_7d",
    "rainfall_max_7d",
]

# Group B: Historical rainfall + location + seasonality.
CONTEXT_FEATURES = HISTORICAL_FEATURES + [
    "latitude",
    "longitude",
    "target_month",
    "target_day_sin",
    "target_day_cos",
]

# Group C: All current features, including regional rainfall.
FULL_FEATURES = CONTEXT_FEATURES + [
    "regional_mean_rainfall_mm",
    "regional_max_rainfall_mm",
    "cells_over_10mm",
    "cells_over_35mm",
]

FEATURE_GROUPS = {
    "A_Historical_Rainfall": HISTORICAL_FEATURES,
    "B_Rainfall_Plus_Context": CONTEXT_FEATURES,
    "C_All_Features": FULL_FEATURES,
}


# --------------------------------------------------
# 2. Load chronological train/validation splits
# --------------------------------------------------

print("=" * 70)
print("TERRAMIND ML: FEATURE-ABLATION EXPERIMENT")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

train_df = df[df["period"] == "train"].copy()
validation_df = df[df["period"] == "validation"].copy()

if train_df.empty or validation_df.empty:
    raise ValueError("Training or validation data is empty.")

if train_df[TARGET].isna().any() or validation_df[TARGET].isna().any():
    raise ValueError("Missing target values detected.")

y_train = train_df[TARGET].astype(int)
y_val = validation_df[TARGET].astype(int)

print(f"\nTraining rows:   {len(train_df):,}")
print(f"Training events: {int(y_train.sum()):,}")
print(f"Validation rows: {len(validation_df):,}")
print(f"Validation events: {int(y_val.sum()):,}")

print("\nFeature groups:")
for group_name, features in FEATURE_GROUPS.items():
    print(f"{group_name}: {len(features)} features")


# --------------------------------------------------
# 3. Train and evaluate each feature group
# --------------------------------------------------

results = []

# Threshold selection is for diagnostic comparison only.
# The validation PR-AUC does not depend on this threshold.
thresholds = np.arange(0.05, 0.951, 0.01)

for group_name, features in FEATURE_GROUPS.items():

    print("\n" + "-" * 70)
    print(f"Experiment: {group_name}")
    print("-" * 70)

    X_train = train_df[features]
    X_val = validation_df[features]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ])

    # Fit only on the training period.
    model.fit(X_train, y_train)

    # Validation probabilities are used for evaluation.
    probabilities = model.predict_proba(X_val)[:, 1]

    pr_auc = average_precision_score(y_val, probabilities)

    # Find the threshold with the highest validation F1.
    # This is a diagnostic result, not an unbiased final score.
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        score = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    final_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        final_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        final_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        final_predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        final_predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "feature_group": group_name,
        "number_of_features": len(features),
        "validation_pr_auc": pr_auc,
        "selected_validation_threshold": best_threshold,
        "validation_precision": precision,
        "validation_recall": recall,
        "validation_f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "predicted_positive": int(final_predictions.sum()),
    }

    results.append(result)

    print(f"Features:       {len(features)}")
    print(f"Validation PR-AUC: {pr_auc:.4f}")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 score:       {f1:.4f}")
    print(f"True positives: {tp}")
    print(f"False positives:{fp}")
    print(f"False negatives:{fn}")


# --------------------------------------------------
# 4. Save and display the comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "validation_pr_auc",
    ascending=False,
)

output_path = OUTPUT_DIR / "feature_ablation_comparison.csv"

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print("FEATURE-ABLATION RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

print(f"\nResults saved to: {output_path}")

print("\nIMPORTANT:")
print("- No test data was used.")
print("- PR-AUC compares ranking quality on validation data.")
print("- Thresholds and F1 scores were also selected/evaluated")
print("  on validation data, so they are diagnostic, not final estimates.")
print("- These results do not establish performance on other regions.")
