"""
Load the current Champion and predict next-day heavy rainfall.
"""
import joblib
import pandas as pd

from pathlib import Path

from src.models.champion_manager import ChampionManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"


def load_champion_model():
    """Load the model artifact belonging to the current Champion."""

    champion = ChampionManager().champion

    if champion is None:
        raise RuntimeError("No Champion model is currently available.")

    model_name = champion["model_name"]
    model_path = MODEL_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Champion model artifact not found: {model_path}"
        )

    artifact = joblib.load(model_path)

    return artifact


def predict_rainfall(input_data):
    """
    Predict whether a grid cell will receive at least
    the configured heavy-rainfall threshold on the next day.
    """

    artifact = load_champion_model()

    features = artifact["features"]
    model = artifact["model"]
    decision_threshold = artifact["decision_threshold"]

    missing_features = [
        feature for feature in features
        if feature not in input_data
    ]

    if missing_features:
        raise ValueError(
            "Missing input features: "
            + ", ".join(missing_features)
        )

    input_df = pd.DataFrame(
        [[input_data[feature] for feature in features]],
        columns=features,
    )

    probability = float(model.predict_proba(input_df)[0][1])

    prediction = int(probability >= decision_threshold)

    return {
        "model_name": ChampionManager().champion["model_name"],
        "probability": probability,
        "decision_threshold": decision_threshold,
        "predicted_heavy_rainfall": prediction,
        "rainfall_threshold_mm": artifact["rainfall_threshold_mm"],
    }
