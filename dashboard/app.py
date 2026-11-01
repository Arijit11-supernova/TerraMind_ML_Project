
import streamlit as st
import pandas as pd
import sys

from pathlib import Path


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="TerraMind ML",
    page_icon="🌧️",
    layout="wide",
)


# --------------------------------------------------
# Project Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delhi_rainfall_2000_2024.csv"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delhi_rainfall_features.csv"
)


# --------------------------------------------------
# Application Header
# --------------------------------------------------

st.title("🌧️ TerraMind ML")

st.subheader("Delhi Heavy Rainfall Risk Prediction")

st.markdown(
    """
    An ML and MLOps prototype for analysing historical
    rainfall patterns and predicting next-day heavy-rainfall
    risk across Delhi's rainfall grid cells.
    """
)

st.info(
    "Research prototype: predictions demonstrate model "
    "behaviour using historical engineered feature records. "
    "This is not a live weather forecast or flood-warning system."
)

st.divider()

st.markdown("### Welcome to TerraMind ML")

st.write(
    "Explore historical rainfall data, examine the selected "
    "machine-learning model, and understand the model-promotion "
    "workflow."
)


# --------------------------------------------------
# Historical Rainfall Analytics
# --------------------------------------------------

st.divider()

st.header("📊 Historical Rainfall Analytics")


@st.cache_data
def load_rainfall_data():

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    daily_data = (
        df.groupby("date")["rainfall_mm"]
        .agg(
            mean_rainfall="mean",
            max_rainfall="max",
        )
        .reset_index()
    )

    return daily_data


try:

    rainfall_df = load_rainfall_data()

    # --------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Historical Period",
            "2000–2024",
        )

    with col2:
        st.metric(
            "Days Available",
            f"{len(rainfall_df):,}",
        )

    with col3:
        st.metric(
            "Grid Cells",
            "12",
        )

    # --------------------------------------------------
    # Date Range Selection
    # --------------------------------------------------

    st.subheader("Select Analysis Period")

    date_range = st.date_input(
        "Choose start and end dates",
        value=(
            rainfall_df["date"].min().date(),
            rainfall_df["date"].max().date(),
        ),
        min_value=rainfall_df["date"].min().date(),
        max_value=rainfall_df["date"].max().date(),
        format="DD/MM/YYYY",
    )

    if len(date_range) == 2:

        start_date, end_date = date_range

        filtered_df = rainfall_df[
            (rainfall_df["date"] >= pd.Timestamp(start_date))
            &
            (rainfall_df["date"] <= pd.Timestamp(end_date))
        ]

    else:

        filtered_df = rainfall_df.iloc[0:0]

    # --------------------------------------------------
    # Filtered Dataset Overview
    # --------------------------------------------------

    if filtered_df.empty:

        st.warning(
            "No rainfall records found for the selected date range."
        )

    else:

        st.metric(
            "Selected Days",
            f"{len(filtered_df):,}",
        )

        # --------------------------------------------------
        # Average Rainfall Chart
        # --------------------------------------------------

        st.subheader(
            "Daily Average Rainfall Across Delhi Grid"
        )

        st.line_chart(
            filtered_df.set_index("date")["mean_rainfall"],
            y_label="Rainfall (mm)",
        )

        # --------------------------------------------------
        # Maximum Rainfall Chart
        # --------------------------------------------------

        st.subheader(
            "Daily Maximum Grid-Cell Rainfall"
        )

        st.line_chart(
            filtered_df.set_index("date")["max_rainfall"],
            y_label="Rainfall (mm)",
        )


except Exception as e:

    st.error(
        f"Unable to load rainfall data: {e}"
    )


# --------------------------------------------------
# Champion Model Prediction
# --------------------------------------------------

from src.models.predict_rainfall import predict_rainfall

st.divider()

st.header("🤖 Heavy Rainfall Prediction")

st.write(
    """
    Select a historical date and Delhi grid cell to replay the
    Champion model's next-day heavy-rainfall prediction.
    """
)

st.info(
    "Historical case replay: the model uses only features available "
    "for the selected date. Actual next-day rainfall is displayed "
    "separately for retrospective comparison."
)


# --------------------------------------------------
# Load Engineered Features
# --------------------------------------------------

@st.cache_data
def load_feature_data():

    df = pd.read_csv(FEATURES_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df["target_date"] = pd.to_datetime(df["target_date"])

    return df


try:

    features_df = load_feature_data()

    # --------------------------------------------------
    # Historical Date Selection
    # --------------------------------------------------

    available_dates = sorted(
        features_df["date"].dropna().unique()
    )

    selected_date = st.selectbox(
        "Select historical date",
        available_dates,
        format_func=lambda x: pd.Timestamp(x).strftime("%d/%m/%Y"),
    )

    # --------------------------------------------------
    # Grid Cell Selection
    # --------------------------------------------------

    date_records = features_df[
        features_df["date"] == selected_date
    ]

    grid_cells = sorted(
        zip(
            date_records["latitude"],
            date_records["longitude"],
        )
    )

    selected_cell = st.selectbox(
        "Select Delhi grid cell (Latitude, Longitude)",
        grid_cells,
        format_func=lambda x: f"{x[0]}, {x[1]}",
    )

    # --------------------------------------------------
    # Retrieve Selected Historical Record
    # --------------------------------------------------

    selected_record = date_records[
        (date_records["latitude"] == selected_cell[0])
        &
        (date_records["longitude"] == selected_cell[1])
    ].iloc[0]

    # --------------------------------------------------
    # Generate Champion Prediction
    # --------------------------------------------------

    if st.button("Run Champion Prediction", type="primary"):

        try:

            input_data = selected_record.to_dict()

            result = predict_rainfall(input_data)

            st.subheader("Prediction Result")

            prediction_col1, prediction_col2 = st.columns(2)

            with prediction_col1:

                st.metric(
                    "Champion Model",
                    result["model_name"],
                )

            with prediction_col2:

                st.metric(
                    "Predicted Probability",
                    f"{result['probability'] * 100:.2f}%",
                )

            st.metric(
                "Decision Threshold",
                f"{result['decision_threshold'] * 100:.2f}%",
            )

            if result["predicted_heavy_rainfall"] == 1:

                st.error(
                    "Heavy Rainfall Predicted for the Next Day"
                )

            else:

                st.success(
                    "Heavy Rainfall Not Predicted for the Next Day"
                )

            st.write(
                f"**Rainfall threshold:** "
                f"{result['rainfall_threshold_mm']} mm"
            )

            st.write(
                f"**Prediction date:** "
                f"{selected_record['target_date'].strftime('%d/%m/%Y')}"
            )

            # --------------------------------------------------
            # Actual Historical Outcome
            # --------------------------------------------------

            st.divider()

            st.subheader("Actual Historical Outcome")

            actual_col1, actual_col2 = st.columns(2)

            with actual_col1:

                st.metric(
                    "Observed Next-Day Rainfall",
                    f"{selected_record['next_day_rainfall_mm']:.2f} mm",
                )

            with actual_col2:

                actual_class = int(
                    selected_record["target_heavy_rainfall"]
                )

                st.metric(
                    "Actual Heavy Rainfall",
                    "Yes" if actual_class == 1 else "No",
                )

            st.caption(
                "The actual outcome is historical ground truth. "
                "It was not provided to the prediction model."
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


except Exception as e:

    st.error(
        f"Unable to load engineered feature data: {e}"
    )


# --------------------------------------------------
# MLOps Overview
# --------------------------------------------------

from src.models.champion_manager import ChampionManager
from src.models.model_promotion import PromotionPolicy


st.divider()

st.header("⚙️ MLOps Overview")

st.write(
    """
    Monitor the current Champion model, review model-promotion
    decisions, and inspect the configured validation policy.
    """
)


# --------------------------------------------------
# Load Persistent Champion State
# --------------------------------------------------

champion_manager = ChampionManager()

champion = champion_manager.champion
history = champion_manager.history

policy = PromotionPolicy()


# --------------------------------------------------
# Current Champion
# --------------------------------------------------

st.subheader("🏆 Current Champion")

if champion is None:

    st.warning("No Champion model has been selected yet.")

else:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Champion Model",
            champion["model_name"],
        )

    with col2:

        st.metric(
            "Validation PR-AUC",
            f"{champion['validation_pr_auc']:.4f}",
        )

    with col3:

        st.metric(
            "Validation F1",
            f"{champion['validation_f1']:.4f}",
        )

    st.caption(
        f"MLflow Run ID: {champion['run_id']}"
    )


# --------------------------------------------------
# Promotion Policy
# --------------------------------------------------

st.divider()

st.subheader("📋 Model Promotion Policy")

policy_col1, policy_col2 = st.columns(2)

with policy_col1:

    st.metric(
        "Minimum Validation PR-AUC",
        f"{policy.min_validation_pr_auc:.2f}",
    )

with policy_col2:

    st.metric(
        "Minimum Validation F1",
        f"{policy.min_validation_f1:.2f}",
    )

st.caption(
    "A candidate must satisfy both validation requirements "
    "to be approved for promotion."
)


# --------------------------------------------------
# Promotion History
# --------------------------------------------------

st.divider()

st.subheader("📜 Model Promotion History")

if not history:

    st.info("No model-promotion events are available.")

else:

    history_rows = []

    for event in reversed(history):

        new_champion = event.get("new_champion") or {}

        history_rows.append({

            "Timestamp": event.get("timestamp", "N/A"),

            "Action": event.get("action", "N/A"),

            "Model": event.get(
                "model_name",
                new_champion.get("model_name", "N/A"),
            ),

            "Status": event.get(
                "status",
                "N/A",
            ).upper(),

            "Run ID": event.get(
                "run_id",
                new_champion.get("run_id", "N/A"),
            ),

        })

    history_df = pd.DataFrame(history_rows)

    if not history_df.empty:

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info("No promotion history records are available.")


# --------------------------------------------------
# Promotion Summary
# --------------------------------------------------

promotion_events = [
    event for event in history
    if event.get("action") == "promotion"
]

approved_count = sum(
    event.get("status") == "promoted"
    for event in promotion_events
)

rejected_count = sum(
    event.get("status") == "rejected"
    for event in promotion_events
)

rollback_count = sum(
    event.get("action") == "rollback"
    for event in history
)


st.divider()

st.subheader("📊 Promotion Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:

    st.metric(
        "Approved Promotions",
        approved_count,
    )

with summary_col2:

    st.metric(
        "Rejected Candidates",
        rejected_count,
    )

with summary_col3:

    st.metric(
        "Rollback Events",
        rollback_count,
    )
