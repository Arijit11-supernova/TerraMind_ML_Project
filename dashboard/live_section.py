"""Live risk monitor: runs the TerraMind agent on real, current rainfall data."""
import json
import os

import pandas as pd
import streamlit as st

from src.agent.risk_agent import DEFAULT_MODEL, run_agent
from src.ingestion.live_weather import LiveDataError


def _secret(name: str):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


def _section_header(title: str, tag: str) -> None:
    st.markdown(
        f'<div class="tm-section"><h2>{title}</h2><span>{tag}</span></div>',
        unsafe_allow_html=True,
    )


def _render_steps(result: dict) -> None:
    mode_label = "Claude plans the tool calls" if result["mode"] == "llm" else "Rule-based planner"
    with st.expander(f"🧭 Agent trace · {mode_label}", expanded=True):
        for index, step in enumerate(result["steps"], start=1):
            if step["kind"] == "reasoning":
                st.markdown(f"**Step {index} · Reasoning**")
                st.write(step["text"])
                continue
            icon = "✅" if step["ok"] else "⚠️"
            st.markdown(f"**Step {index} · {icon} `{step['tool']}`**")
            st.code(json.dumps(step["output"], indent=2, default=str), language="json")
        for note in result["notes"]:
            st.caption(f"Resilience: {note}")


def _render_result(result: dict) -> None:
    risk, weather, fetch = result["risk"], result["weather"], result["fetch"]
    level = risk["level"]

    if fetch.from_cache:
        st.warning(
            f"Live API unreachable. Showing the last successful real fetch from {fetch.fetched_at} UTC."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk level", level)
    c2.metric("Peak model score", f"{risk['peak_score_pct']:.2f}%")
    c3.metric("Cells flagged", f"{risk['cells_flagged']} / {risk['cells_total']}")
    c4.metric("Prediction for", weather["target_date"])

    if level == "HIGH":
        st.markdown(
            '<div class="tm-verdict hit">⚠️ High heavy-rainfall risk for the next day</div>',
            unsafe_allow_html=True,
        )
    elif level == "ELEVATED":
        st.warning("Elevated risk: the score is approaching the alert threshold or the forecast model shows heavy rain.")
    else:
        st.markdown(
            '<div class="tm-verdict clear">✅ Low heavy-rainfall risk for the next day</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"Source: {result['source']} · fetched {fetch.fetched_at} UTC · inputs as of {weather['as_of']} · "
        f"model: {risk['model_name']} · alert threshold {risk['decision_threshold_pct']}% · "
        f"heavy rain = {risk['rainfall_threshold_mm']:.0f} mm or more in a cell."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Advisory (English)")
        with st.container(border=True):
            st.write(result["advisory"]["english"])
    with right:
        st.subheader("सलाह (हिन्दी)")
        with st.container(border=True):
            st.write(result["advisory"]["hindi"])

    _render_steps(result)

    st.subheader("Grid-cell detail")
    table = result["scores"].copy()
    table["score"] = (table["score"] * 100).round(2)
    table = table.rename(columns={
        "latitude": "Lat",
        "longitude": "Lon",
        "rainfall_yesterday_mm": "Rain on input day (mm)",
        "rainfall_max_prev7d_mm": "Max rain, prior 7 days (mm)",
        "score": "Model score (%)",
        "flagged": "Flagged",
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    if result["history"]:
        hist = result["history"]
        st.caption(
            f"Seasonal context ({hist['comparison_window']}): "
            f"{hist['share_of_comparable_days_at_least_this_wet_pct']}% of comparable days were at least this wet; "
            f"{hist['historical_share_of_days_followed_by_35mm_cell_pct']}% were followed by a 35 mm+ cell "
            f"({hist['historical_heavy_days_count']} of {hist['days_compared']} days)."
        )

    st.info(
        "Data note: the model was trained on IMD gridded rainfall. Live inputs come from Open-Meteo for the "
        "same 12 cell centres, so a small source shift is possible. Heavy-rain days are rare (about 1% of "
        "cell-days), so a low score on most days is the expected, correct behaviour."
    )


def render_live_section() -> None:
    _section_header("🛰️ Live Risk Monitor", "live data · agent workflow")

    st.write(
        "The agent fetches real current rainfall for Delhi's 12 grid cells, runs the deployed champion model, "
        "compares against 25 years of IMD history, and writes an advisory in English and Hindi."
    )

    api_key = _secret("ANTHROPIC_API_KEY")
    model = _secret("ANTHROPIC_MODEL") or DEFAULT_MODEL
    st.caption(
        "Planner: Claude tool-calling" if api_key else "Planner: rule-based (add ANTHROPIC_API_KEY in Streamlit secrets to enable Claude)"
    )

    if st.button("Run live risk agent", type="primary", key="run_live_agent"):
        with st.spinner("Fetching live rainfall, running model, writing advisory..."):
            try:
                st.session_state["live_result"] = run_agent(api_key=api_key, model=model)
                st.session_state.pop("live_error", None)
            except LiveDataError as exc:
                st.session_state.pop("live_result", None)
                st.session_state["live_error"] = str(exc)
            except Exception as exc:  # keep the demo alive and explain what failed
                st.session_state.pop("live_result", None)
                st.session_state["live_error"] = f"{type(exc).__name__}: {exc}"

    if "live_error" in st.session_state:
        st.error(
            "Live prediction is unavailable right now, so no risk is shown (TerraMind never substitutes made-up "
            f"data). Reason: {st.session_state['live_error']}"
        )
    elif "live_result" in st.session_state:
        _render_result(st.session_state["live_result"])