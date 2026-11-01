import streamlit as st
import pandas as pd
import altair as alt
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

# Allow Altair to plot the full 2000-2024 daily series (>5000 rows)
alt.data_transformers.disable_max_rows()


# --------------------------------------------------
# TerraMind ML — Mission-Control Theme
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #050d18;
    --panel: rgba(12, 30, 48, 0.72);
    --panel-solid: #0c1e30;
    --line: rgba(61, 232, 255, 0.18);
    --line-strong: rgba(61, 232, 255, 0.45);
    --cyan: #3de8ff;
    --teal: #2fe0c0;
    --amber: #ffb547;
    --red: #ff5d73;
    --text: #dbeaf7;
    --muted: #7f9bb5;
}

/* ---------- Base canvas: deep navy + faint engineering grid ---------- */
.stApp {
    background:
        radial-gradient(1100px 600px at 85% -10%, rgba(61, 232, 255, 0.10), transparent 60%),
        radial-gradient(900px 500px at -10% 110%, rgba(47, 224, 192, 0.08), transparent 60%),
        var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(61, 232, 255, 0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(61, 232, 255, 0.045) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse at 50% 30%, black 20%, transparent 75%);
    -webkit-mask-image: radial-gradient(ellipse at 50% 30%, black 20%, transparent 75%);
    z-index: 0;
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 1400px;
    position: relative;
    z-index: 1;
}

/* ---------- Typography ---------- */
h1, h2, h3, h4 {
    font-family: 'Chakra Petch', sans-serif !important;
    color: #eaf8ff !important;
    letter-spacing: -0.3px;
}
h1 { font-size: 3rem !important; font-weight: 700 !important; }
h3 { font-size: 1.25rem !important; font-weight: 600 !important; color: #bfeeff !important; }

p, label, .stMarkdown, li, span { color: var(--text); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }

hr { border-color: var(--line) !important; margin: 2rem 0 !important; }

/* ---------- Hero ---------- */
.tm-hero {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    padding: 2.2rem 2.4rem;
    border: 1px solid var(--line);
    border-radius: 6px;
    background:
        linear-gradient(120deg, rgba(12, 30, 48, 0.9), rgba(8, 20, 34, 0.75));
    position: relative;
    overflow: hidden;
}
.tm-hero::before, .tm-hero::after {
    content: "";
    position: absolute;
    width: 22px; height: 22px;
    border: 2px solid var(--cyan);
}
.tm-hero::before { top: -1px; left: -1px; border-right: none; border-bottom: none; }
.tm-hero::after { bottom: -1px; right: -1px; border-left: none; border-top: none; }

.tm-hero-text { flex: 1 1 420px; min-width: 0; }
.tm-title {
    font-family: 'Chakra Petch', sans-serif;
    font-weight: 700;
    font-size: clamp(2.4rem, 5vw, 4rem);
    line-height: 1.02;
    letter-spacing: -1.5px;
    margin: 0 0 0.6rem 0;
    background: linear-gradient(90deg, #ffffff 0%, #9cf3ff 55%, #3de8ff 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.tm-sub {
    font-family: 'Chakra Petch', sans-serif;
    font-size: 1.3rem;
    color: #bfeeff;
    margin: 0 0 0.9rem 0;
}
.tm-desc {
    max-width: 58ch;
    color: var(--muted);
    line-height: 1.6;
    margin: 0 0 1.3rem 0;
}
.tm-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.tm-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    padding: 0.28rem 0.7rem;
    border: 1px solid var(--line-strong);
    border-radius: 3px;
    color: var(--cyan);
    background: rgba(61, 232, 255, 0.06);
}
.tm-chip.warn { color: var(--amber); border-color: rgba(255, 181, 71, 0.5); background: rgba(255, 181, 71, 0.07); }

/* Radar: the one memorable element */
.tm-radar { flex: 0 0 auto; width: 250px; height: 250px; }
.tm-radar svg { width: 100%; height: 100%; overflow: visible; }
.tm-sweep { transform-origin: 125px 125px; animation: tm-spin 6s linear infinite; }
@keyframes tm-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .tm-sweep { animation: none; } }

/* ---------- Section headers ---------- */
.tm-section {
    display: flex;
    align-items: baseline;
    gap: 0.9rem;
    margin: 0.4rem 0 1.1rem 0;
    padding-left: 0.9rem;
    border-left: 3px solid var(--cyan);
}
.tm-section h2 {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
}
.tm-section span {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
}

/* ---------- Metric cards ---------- */
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-top: 2px solid var(--cyan);
    border-radius: 4px;
    padding: 16px 18px;
    backdrop-filter: blur(6px);
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: 0.82rem;
}
[data-testid="stMetricValue"] {
    color: var(--teal) !important;
    font-family: 'Chakra Petch', sans-serif !important;
    font-weight: 600 !important;
    text-shadow: 0 0 18px rgba(47, 224, 192, 0.35);
    font-size: 1.7rem !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: transparent;
    color: var(--cyan);
    border: 1px solid var(--cyan);
    border-radius: 3px;
    padding: 0.7rem 1.6rem;
    font-family: 'Chakra Petch', sans-serif;
    font-weight: 600;
    letter-spacing: 0.4px;
    box-shadow: 0 0 0 rgba(61, 232, 255, 0);
    transition: background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:hover {
    background: var(--cyan);
    color: #031018;
    border-color: var(--cyan);
    box-shadow: 0 0 24px rgba(61, 232, 255, 0.45);
}
.stButton > button:focus-visible { outline: 2px solid var(--amber); outline-offset: 2px; }

/* ---------- Inputs ---------- */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: rgba(12, 30, 48, 0.85);
    border: 1px solid var(--line-strong);
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
}
div[data-baseweb="popover"] ul { background: var(--panel-solid); }

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 3px solid var(--cyan);
    border-radius: 3px;
}

/* ---------- Dataframe ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 2px;
    background: var(--panel);
}

/* ---------- Chart frame ---------- */
[data-testid="stArrowVegaLiteChart"], .stVegaLiteChart {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 10px;
}

/* ---------- Probability gauge ---------- */
.tm-gauge {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1.1rem 1.3rem 1.6rem 1.3rem;
    margin: 0.8rem 0 1rem 0;
}
.tm-gauge-head {
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; color: var(--muted);
    margin-bottom: 0.7rem;
}
.tm-track {
    position: relative; height: 16px; border-radius: 2px;
    background: repeating-linear-gradient(90deg, rgba(61,232,255,0.10) 0 2px, transparent 2px 10%);
    border: 1px solid var(--line);
}
.tm-fill { height: 100%; border-radius: 1px; }
.tm-tick {
    position: absolute; top: -7px; bottom: -7px; width: 2px; background: var(--amber);
}
.tm-tick span {
    position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
    margin-top: 4px; white-space: nowrap;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--amber);
}

/* ---------- Verdict banner ---------- */
.tm-verdict {
    border: 1px solid; border-radius: 4px;
    padding: 1rem 1.3rem; margin: 0.6rem 0 1rem 0;
    font-family: 'Chakra Petch', sans-serif; font-size: 1.25rem; font-weight: 600;
}
.tm-verdict.hit { color: var(--red); border-color: rgba(255, 93, 115, 0.6); background: rgba(255, 93, 115, 0.08); }
.tm-verdict.clear { color: var(--teal); border-color: rgba(47, 224, 192, 0.55); background: rgba(47, 224, 192, 0.07); }

/* ---------- Footer ---------- */
.tm-foot {
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line);
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--muted);
    display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# UI Helpers
# --------------------------------------------------

def section(title: str, tag: str) -> None:
    st.markdown(
        f'<div class="tm-section"><h2>{title}</h2><span>{tag}</span></div>',
        unsafe_allow_html=True,
    )


def area_chart(df: pd.DataFrame, column: str, color: str) -> alt.Chart:
    gradient = alt.Gradient(
        gradient="linear",
        stops=[
            alt.GradientStop(color=color + "00", offset=0),
            alt.GradientStop(color=color + "88", offset=1),
        ],
        x1=1, x2=1, y1=1, y2=0,
    )

    chart = (
        alt.Chart(df)
        .mark_area(
            line={"color": color, "strokeWidth": 1.6},
            color=gradient,
            interpolate="monotone",
        )
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y(f"{column}:Q", title="Rainfall (mm)"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d/%m/%Y"),
                alt.Tooltip(f"{column}:Q", title="Rainfall (mm)", format=".2f"),
            ],
        )
        .properties(height=270, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#14304a",
            domainColor="#28516a",
            tickColor="#28516a",
            labelColor="#7f9bb5",
            titleColor="#7f9bb5",
            labelFont="IBM Plex Mono",
            titleFont="IBM Plex Mono",
        )
        .interactive(bind_y=False)
    )
    return chart


def probability_gauge(probability: float, threshold: float, flagged: bool) -> None:
    p = max(0.0, min(1.0, probability)) * 100
    t = max(0.0, min(1.0, threshold)) * 100
    fill = (
        "linear-gradient(90deg, #ff9f43, #ff5d73)"
        if flagged
        else "linear-gradient(90deg, #2fe0c0, #3de8ff)"
    )
    st.markdown(
        f'<div class="tm-gauge">'
        f'<div class="tm-gauge-head"><span>predicted probability</span><span>{p:.2f}%</span></div>'
        f'<div class="tm-track">'
        f'<div class="tm-fill" style="width:{p:.2f}%;background:{fill};"></div>'
        f'<div class="tm-tick" style="left:{t:.2f}%;"><span>threshold {t:.1f}%</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
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

# 12 Delhi grid cells drawn as a 4 x 3 radar plot
_radar_dots = "".join(
    f'<circle cx="{70 + c * 38}" cy="{80 + r * 45}" r="3.2" fill="#3de8ff" opacity="0.85"/>'
    for r in range(3)
    for c in range(4)
)

st.markdown(
    f"""
<div class="tm-hero">
<div class="tm-hero-text">
<div class="tm-title">TerraMind ML</div>
<div class="tm-sub">Delhi heavy rainfall risk prediction</div>
<p class="tm-desc">An ML and MLOps prototype that combines live rainfall data with 25 years of IMD history to predict next-day heavy-rainfall risk across Delhi's rainfall grid cells, and an agent that turns the prediction into an advisory.</p>
<div class="tm-chips">
<span class="tm-chip">2000–2024 archive</span>
<span class="tm-chip">12 grid cells</span>
<span class="tm-chip">live Open-Meteo feed</span>
<span class="tm-chip">champion / challenger</span>
<span class="tm-chip warn">research prototype</span>
</div>
</div>
<div class="tm-radar">
<svg viewBox="0 0 250 250" xmlns="http://www.w3.org/2000/svg">
<defs>
<radialGradient id="rg" cx="50%" cy="50%" r="50%">
<stop offset="0%" stop-color="#3de8ff" stop-opacity="0.10"/>
<stop offset="100%" stop-color="#3de8ff" stop-opacity="0"/>
</radialGradient>
<linearGradient id="sw" x1="0" y1="0" x2="1" y2="0">
<stop offset="0%" stop-color="#3de8ff" stop-opacity="0"/>
<stop offset="100%" stop-color="#3de8ff" stop-opacity="0.55"/>
</linearGradient>
</defs>
<circle cx="125" cy="125" r="120" fill="url(#rg)" stroke="#3de8ff" stroke-opacity="0.45"/>
<circle cx="125" cy="125" r="85" fill="none" stroke="#3de8ff" stroke-opacity="0.22"/>
<circle cx="125" cy="125" r="50" fill="none" stroke="#3de8ff" stroke-opacity="0.22"/>
<line x1="5" y1="125" x2="245" y2="125" stroke="#3de8ff" stroke-opacity="0.2"/>
<line x1="125" y1="5" x2="125" y2="245" stroke="#3de8ff" stroke-opacity="0.2"/>
{_radar_dots}
<circle cx="146" cy="125" r="6" fill="none" stroke="#ffb547" stroke-width="1.5"/>
<g class="tm-sweep">
<path d="M125 125 L245 125 A120 120 0 0 0 208 40 Z" fill="url(#sw)" transform="rotate(-48 125 125)"/>
<line x1="125" y1="125" x2="245" y2="125" stroke="#3de8ff" stroke-width="1.5"/>
</g>
</svg>
</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

st.info(
    "Research prototype: the Live Risk Monitor uses current Open-Meteo rainfall, "
    "and the Backtest replays historical IMD records. "
    "This is not an official weather forecast or flood-warning system."
)

st.write(
    "Run the live risk agent, explore historical rainfall data, replay the "
    "selected machine-learning model on past cases, and review the "
    "model-promotion workflow."
)


# --------------------------------------------------
# Live Risk Monitor (live data + agent)
# --------------------------------------------------

try:
    try:
        from dashboard.live_section import render_live_section
    except ImportError:
        from live_section import render_live_section

    st.divider()
    render_live_section()

except Exception as e:

    st.error(f"Live Risk Monitor could not be loaded: {e}")


# --------------------------------------------------
# Historical Rainfall Analytics
# --------------------------------------------------

st.divider()

section("📊 Historical Rainfall Analytics", "source: processed daily grid data")


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

        stat1, stat2, stat3 = st.columns(3)

        peak_row = filtered_df.loc[filtered_df["max_rainfall"].idxmax()]

        with stat1:
            st.metric(
                "Selected Days",
                f"{len(filtered_df):,}",
            )

        with stat2:
            st.metric(
                "Peak Grid-Cell Rainfall",
                f"{peak_row['max_rainfall']:.1f} mm",
            )

        with stat3:
            st.metric(
                "Date of Peak",
                peak_row["date"].strftime("%d/%m/%Y"),
            )

        # --------------------------------------------------
        # Average Rainfall Chart
        # --------------------------------------------------

        st.subheader(
            "Daily Average Rainfall Across Delhi Grid"
        )

        st.altair_chart(
            area_chart(filtered_df, "mean_rainfall", "#3de8ff"),
            use_container_width=True,
        )

        # --------------------------------------------------
        # Maximum Rainfall Chart
        # --------------------------------------------------

        st.subheader(
            "Daily Maximum Grid-Cell Rainfall"
        )

        st.altair_chart(
            area_chart(filtered_df, "max_rainfall", "#ffb547"),
            use_container_width=True,
        )

        st.caption("Scroll to zoom, drag to pan, double-click to reset.")


except Exception as e:

    st.error(
        f"Unable to load rainfall data: {e}"
    )


# --------------------------------------------------
# Champion Model Prediction
# --------------------------------------------------

from src.models.predict_rainfall import predict_rainfall

st.divider()

section("🧪 Backtest: Historical Case Replay", "champion model on IMD history")

st.write(
    """
    Select a historical date and Delhi grid cell to replay the
    Champion model's next-day heavy-rainfall prediction and compare
    it with what actually happened.
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

    select_col1, select_col2 = st.columns(2)

    with select_col1:

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

    with select_col2:

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

            prediction_col1, prediction_col2, prediction_col3 = st.columns(3)

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

            with prediction_col3:

                st.metric(
                    "Decision Threshold",
                    f"{result['decision_threshold'] * 100:.2f}%",
                )

            is_flagged = result["predicted_heavy_rainfall"] == 1

            probability_gauge(
                result["probability"],
                result["decision_threshold"],
                is_flagged,
            )

            if is_flagged:

                st.markdown(
                    '<div class="tm-verdict hit">⚠️ Heavy rainfall predicted for the next day</div>',
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    '<div class="tm-verdict clear">✅ Heavy rainfall not predicted for the next day</div>',
                    unsafe_allow_html=True,
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

section("⚙️ MLOps Overview", "lifecycle: champion, policy, history")

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

    STATUS_ICONS = {
        "PROMOTED": "🟢 PROMOTED",
        "REJECTED": "🔴 REJECTED",
    }

    history_rows = []

    for event in reversed(history):

        new_champion = event.get("new_champion") or {}

        status_label = event.get(
            "status",
            "N/A",
        ).upper()

        history_rows.append({

            "Timestamp": event.get("timestamp", "N/A"),

            "Action": event.get("action", "N/A"),

            "Model": event.get(
                "model_name",
                new_champion.get("model_name", "N/A"),
            ),

            "Status": STATUS_ICONS.get(status_label, status_label),

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


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    '<div class="tm-foot"><span>TerraMind ML · research prototype</span>'
    '<span>live Open-Meteo + historical IMD · not a flood-warning system</span></div>',
    unsafe_allow_html=True,
)