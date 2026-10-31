from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

DATA_PATH = Path("data/processed/delhi_rainfall_features.csv")
MODEL_DIR = Path("artifacts/models")
OUTPUT_DIR = Path("artifacts/evaluation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_heavy_rainfall"
PERIOD = "validation"

MODELS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}


# --------------------------------------------------
# 2. Load and prepare validation data
# --------------------------------------------------

print("=" * 65)
print("TERRAMIND ML: VALIDATION ERROR ANALYSIS")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["target_date"] = pd.to_datetime(df["target_date"])

validation_df = (
    df.loc[df["period"] == PERIOD]
    .copy()
    .sort_values(["target_date", "latitude", "longitude"])
    .reset_index(drop=True)
)

if validation_df.empty:
    raise ValueError("No validation rows found.")

if validation_df[TARGET].isna().any():
    raise ValueError("The validation target contains missing values.")

print(f"\nValidation rows: {len(validation_df):,}")
print(f"Validation period: {validation_df['target_date'].min().date()}"
      f" to {validation_df['target_date'].max().date()}")
print(f"Positive cases: {int(validation_df[TARGET].sum()):,}")
print(f"Positive rate: {validation_df[TARGET].mean():.4%}")


# --------------------------------------------------
# 3. Generate predictions using saved models
# --------------------------------------------------

y_true = validation_df[TARGET].astype(int).to_numpy()

prediction_results = {}
overall_results = []

for model_name, model_path in MODELS.items():
    artifact = joblib.load(model_path)

    model = artifact["model"]
    features = artifact["features"]
    threshold = float(artifact["decision_threshold"])

    missing_features = [
        feature for feature in features
        if feature not in validation_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"{model_name} is missing features: {missing_features}"
        )

    X = validation_df[features]

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    result = {
        "model": model_name,
        "threshold": threshold,
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "predicted_positive": int(predictions.sum()),
    }

    overall_results.append(result)

    prediction_results[model_name] = {
        "probabilities": probabilities,
        "predictions": predictions,
        "threshold": threshold,
    }

    print(f"\n--- {model_name} ---")
    print(f"Threshold:          {threshold:.2f}")
    print(f"Precision:          {result['precision']:.4f}")
    print(f"Recall:             {result['recall']:.4f}")
    print(f"F1 score:           {result['f1']:.4f}")
    print(f"PR-AUC:             {result['pr_auc']:.4f}")
    print(f"True positives:     {tp}")
    print(f"False positives:    {fp}")
    print(f"True negatives:     {tn}")
    print(f"False negatives:    {fn}")

overall_df = pd.DataFrame(overall_results)
overall_path = OUTPUT_DIR / "validation_error_overall.csv"
overall_df.to_csv(overall_path, index=False)


# --------------------------------------------------
# 4. Seasonal analysis
# --------------------------------------------------

print("\n" + "=" * 65)
print("SEASONAL ANALYSIS")
print("=" * 65)

# Indian meteorological seasons, used here as analytical groups:
# Jan-Feb: winter
# Mar-May: pre-monsoon
# Jun-Sep: monsoon
# Oct-Dec: post-monsoon

def assign_season(month):
    if month in (1, 2):
        return "Winter (Jan-Feb)"
    elif month in (3, 4, 5):
        return "Pre-monsoon (Mar-May)"
    elif month in (6, 7, 8, 9):
        return "Monsoon (Jun-Sep)"
    else:
        return "Post-monsoon (Oct-Dec)"


validation_df["season"] = (
    validation_df["target_date"].dt.month.map(assign_season)
)

seasonal_results = []

for model_name, result in prediction_results.items():
    probabilities = result["probabilities"]
    predictions = result["predictions"]

    for season, indices in validation_df.groupby("season").groups.items():
        idx = np.asarray(list(indices), dtype=int)

        actual = y_true[idx]
        predicted = predictions[idx]
        scores = probabilities[idx]

        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        seasonal_results.append({
            "model": model_name,
            "season": season,
            "observations": len(idx),
            "actual_positive": int(actual.sum()),
            "predicted_positive": int(predicted.sum()),
            "precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "f1": f1_score(
                actual, predicted, zero_division=0
            ),
            "pr_auc": (
                average_precision_score(actual, scores)
                if actual.sum() > 0
                else np.nan
            ),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        })

seasonal_df = pd.DataFrame(seasonal_results)
seasonal_path = OUTPUT_DIR / "validation_error_by_season.csv"
seasonal_df.to_csv(seasonal_path, index=False)

print(seasonal_df.to_string(index=False))


# --------------------------------------------------
# 5. Grid-cell geographical analysis
# --------------------------------------------------

print("\n" + "=" * 65)
print("GRID-CELL ANALYSIS")
print("=" * 65)

location_results = []

for model_name, result in prediction_results.items():
    probabilities = result["probabilities"]
    predictions = result["predictions"]

    for (latitude, longitude), group in validation_df.groupby(
        ["latitude", "longitude"]
    ):
        idx = group.index.to_numpy()

        actual = y_true[idx]
        predicted = predictions[idx]
        scores = probabilities[idx]

        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        location_results.append({
            "model": model_name,
            "latitude": latitude,
            "longitude": longitude,
            "observations": len(idx),
            "actual_positive": int(actual.sum()),
            "predicted_positive": int(predicted.sum()),
            "precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "f1": f1_score(
                actual, predicted, zero_division=0
            ),
            "pr_auc": (
                average_precision_score(actual, scores)
                if actual.sum() > 0
                else np.nan
            ),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        })

location_df = pd.DataFrame(location_results)
location_path = OUTPUT_DIR / "validation_error_by_grid_cell.csv"
location_df.to_csv(location_path, index=False)

print(
    location_df.sort_values(
        ["model", "recall"], ascending=[True, True]
    ).to_string(index=False)
)


# --------------------------------------------------
# 6. Error analysis by antecedent rainfall
# --------------------------------------------------

print("\n" + "=" * 65)
print("ERROR ANALYSIS BY RECENT RAINFALL")
print("=" * 65)

# These are descriptive groups based on the preceding
# 7-day rainfall maximum, not official weather categories.

rainfall_bins = [-np.inf, 0, 1, 10, 35, np.inf]
rainfall_labels = [
    "No rain in previous 7 days",
    ">0 to 1 mm",
    ">1 to 10 mm",
    ">10 to 35 mm",
    ">35 mm",
]

validation_df["antecedent_rainfall_group"] = pd.cut(
    validation_df["rainfall_max_7d"],
    bins=rainfall_bins,
    labels=rainfall_labels,
    include_lowest=True,
    right=True,
)

antecedent_results = []

for model_name, result in prediction_results.items():
    probabilities = result["probabilities"]
    predictions = result["predictions"]

    for group_name, group in validation_df.groupby(
        "antecedent_rainfall_group", observed=False
    ):
        if group.empty:
            continue

        idx = group.index.to_numpy()
        actual = y_true[idx]
        predicted = predictions[idx]
        scores = probabilities[idx]

        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        antecedent_results.append({
            "model": model_name,
            "antecedent_rainfall_group": str(group_name),
            "observations": len(idx),
            "actual_positive": int(actual.sum()),
            "predicted_positive": int(predicted.sum()),
            "precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "f1": f1_score(
                actual, predicted, zero_division=0
            ),
            "pr_auc": (
                average_precision_score(actual, scores)
                if actual.sum() > 0
                else np.nan
            ),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        })

antecedent_df = pd.DataFrame(antecedent_results)
antecedent_path = (
    OUTPUT_DIR / "validation_error_by_antecedent_rainfall.csv"
)
antecedent_df.to_csv(antecedent_path, index=False)

print(antecedent_df.to_string(index=False))


# --------------------------------------------------
# 7. Save individual false positives and false negatives
# --------------------------------------------------

print("\n" + "=" * 65)
print("SAVING INDIVIDUAL MODEL ERRORS")
print("=" * 65)

error_frames = []

for model_name, result in prediction_results.items():
    errors = validation_df[
        [
            "date",
            "target_date",
            "latitude",
            "longitude",
            "rainfall_mm",
            "rainfall_lag_1",
            "rainfall_lag_2",
            "rainfall_lag_3",
            "rainfall_mean_7d",
            "rainfall_max_7d",
            "regional_mean_rainfall_mm",
            "regional_max_rainfall_mm",
            "target_month",
            TARGET,
        ]
    ].copy()

    errors["model"] = model_name
    errors["predicted_probability"] = result["probabilities"]
    errors["decision_threshold"] = result["threshold"]
    errors["predicted_label"] = result["predictions"]

    errors["error_type"] = np.select(
        [
            (errors[TARGET] == 1) & (errors["predicted_label"] == 1),
            (errors[TARGET] == 0) & (errors["predicted_label"] == 1),
            (errors[TARGET] == 1) & (errors["predicted_label"] == 0),
        ],
        [
            "True Positive",
            "False Positive",
            "False Negative",
        ],
        default="True Negative",
    )

    error_frames.append(errors)

all_errors = pd.concat(error_frames, ignore_index=True)

error_path = OUTPUT_DIR / "validation_individual_predictions.csv"
all_errors.to_csv(error_path, index=False)

mistakes = all_errors[
    all_errors["error_type"].isin(
        ["False Positive", "False Negative"]
    )
].copy()

mistakes_path = OUTPUT_DIR / "validation_misclassified_cases.csv"
mistakes.to_csv(mistakes_path, index=False)

print(f"Individual predictions saved: {error_path}")
print(f"Misclassified cases saved:    {mistakes_path}")

print("\nMisclassification counts:")
print(
    mistakes.groupby(["model", "error_type"])
    .size()
    .rename("count")
    .to_string()
)


# --------------------------------------------------
# 8. Final report
# --------------------------------------------------

print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)

for path in [
    overall_path,
    seasonal_path,
    location_path,
    antecedent_path,
    error_path,
    mistakes_path,
]:
    print(path)

print("\nImportant: These are validation results, not test results.")
print("Use the findings to guide improvements before final testing.")