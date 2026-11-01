"""
Evaluate MLflow model candidates and update Champion selection.

This script records promotion decisions on MLflow runs
and persists the selected Champion model reference.
It does not deploy or serve predictions.
"""

import json

import mlflow
from mlflow.tracking import MlflowClient

from src.models.champion_manager import ChampionManager
from src.models.model_promotion import evaluate_candidate


TRACKING_URI = "sqlite:///artifacts/mlflow/mlflow.db"
EXPERIMENT_NAME = "TerraMind-Heavy-Rainfall-Prediction"

MODEL_NAMES = {
    "logistic_regression",
    "random_forest",
}


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    champion_manager = ChampionManager()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: {EXPERIMENT_NAME}"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1000,
    )

    # Select the most recent run for each model.
    latest_runs = {}

    for run in runs:
        model_name = run.data.params.get("model_name")

        if model_name not in MODEL_NAMES:
            continue

        if model_name not in latest_runs:
            latest_runs[model_name] = run

    for model_name in sorted(MODEL_NAMES):
        run = latest_runs.get(model_name)

        if run is None:
            print(f"\n{model_name}: no MLflow run found.")
            continue

        metrics = run.data.metrics

        required_metrics = (
            "validation_pr_auc",
            "validation_f1",
        )

        missing = [
            name for name in required_metrics
            if name not in metrics
        ]

        if missing:
            print(f"\n{model_name}: SKIPPED")
            print(f"Run ID: {run.info.run_id}")
            print(
                "Missing validation metrics: "
                + ", ".join(missing)
            )
            continue

        decision = evaluate_candidate(
            validation_pr_auc=metrics["validation_pr_auc"],
            validation_f1=metrics["validation_f1"],
        )

        decision_text = (
            "approved" if decision.approved else "rejected"
        )

        client.set_tag(
            run.info.run_id,
            "promotion_decision",
            decision_text,
        )

        client.set_tag(
            run.info.run_id,
            "promotion_reasons",
            json.dumps(decision.reasons),
        )

        print(f"\nModel: {model_name}")
        print(f"Run ID: {run.info.run_id}")
        print(
            f"Validation PR-AUC: "
            f"{metrics['validation_pr_auc']:.4f}"
        )
        print(
            f"Validation F1: "
            f"{metrics['validation_f1']:.4f}"
        )
        print(f"Promotion decision: {decision_text.upper()}")

        for reason in decision.reasons:
            print(f"- {reason}")

        result = champion_manager.promote(
            model_name=model_name,
            run_id=run.info.run_id,
            validation_pr_auc=metrics["validation_pr_auc"],
            validation_f1=metrics["validation_f1"],
        )

        print(
            "Champion update: "
            + ("PROMOTED" if result["promoted"] else "NOT PROMOTED")
        )

    print("\nCurrent Champion:")
    print(champion_manager.champion)

    print("\nPromotion evaluation finished.")


if __name__ == "__main__":
    main()
