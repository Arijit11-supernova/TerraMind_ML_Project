"""
TerraMind risk agent.

Plans and executes a multi-step tool workflow on LIVE data:

  get_live_weather -> predict_risk -> check_history -> advisory (English + Hindi)

Two planners share the same tools:
  * LLM planner  - Claude chooses which tools to call (needs an API key).
  * Rule planner - fixed order with the same error handling (no key needed).
If the LLM path fails for any reason the agent falls back to the rule planner.

Guardrails: the advisory may only use numbers returned by the tools, every
advisory is accompanied by the evidence it was built from, and the output is
labelled a research prototype that defers to official IMD warnings.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.ingestion.live_weather import (
    SOURCE_NAME,
    LiveDataError,
    build_live_features,
    fetch_live_rainfall,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "delhi_rainfall_features.csv"

MAX_TURNS = 8
HISTORY_WINDOW_DAYS = 15
DEFAULT_MODEL = "claude-sonnet-5-5"


# --------------------------------------------------
# Tools
# --------------------------------------------------

def tool_get_live_weather(ctx: dict, **_) -> dict:
    fetch = fetch_live_rainfall()
    features, meta = build_live_features(fetch.frame)
    ctx.update(fetch=fetch, features=features, weather=meta)
    return {
        "source": fetch.source,
        "fetched_at_utc": fetch.fetched_at,
        "served_from_cache": fetch.from_cache,
        "as_of_date": meta["as_of"],
        "prediction_target_date": meta["target_date"],
        "regional_mean_rainfall_mm_on_as_of_date": round(meta["regional_mean_mm"], 2),
        "regional_max_rainfall_mm_on_as_of_date": round(meta["regional_max_mm"], 2),
        "forecast_model_regional_max_mm_for_target_date": (
            None if meta["nwp_target_regional_max_mm"] is None
            else round(meta["nwp_target_regional_max_mm"], 2)
        ),
    }


def assess_level(flagged: int, max_score: float, threshold: float, nwp_max) -> str:
    if flagged > 0:
        return "HIGH"
    if max_score >= 0.5 * threshold or (nwp_max is not None and nwp_max >= 35):
        return "ELEVATED"
    return "LOW"


def tool_predict_risk(ctx: dict, **_) -> dict:
    if "features" not in ctx:
        raise RuntimeError("No live weather loaded yet. Call get_live_weather first.")

    from src.models.predict_rainfall import predict_rainfall  # lazy: needs model artifacts

    rows = []
    for _, row in ctx["features"].iterrows():
        result = predict_rainfall(row.to_dict())
        rows.append({
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "rainfall_yesterday_mm": round(row["rainfall_mm"], 2),
            "rainfall_max_prev7d_mm": round(row["rainfall_max_7d"], 2),
            "score": result["probability"],
            "flagged": bool(result["predicted_heavy_rainfall"]),
        })
    scores = pd.DataFrame(rows)
    top = scores.loc[scores["score"].idxmax()]
    flagged = int(scores["flagged"].sum())
    nwp_max = ctx["weather"]["nwp_target_regional_max_mm"]
    level = assess_level(flagged, float(top["score"]), result["decision_threshold"], nwp_max)

    ctx["scores"] = scores
    ctx["risk"] = {
        "level": level,
        "model_name": result["model_name"],
        "rainfall_threshold_mm": result["rainfall_threshold_mm"],
        "decision_threshold_pct": round(result["decision_threshold"] * 100, 2),
        "peak_score_pct": round(float(top["score"]) * 100, 2),
        "peak_cell": f"{top['latitude']}N, {top['longitude']}E",
        "cells_flagged": flagged,
        "cells_total": len(scores),
    }
    return ctx["risk"]


@lru_cache(maxsize=1)
def _history() -> pd.DataFrame:
    df = pd.read_csv(
        FEATURES_PATH,
        usecols=["date", "regional_mean_rainfall_mm", "regional_max_rainfall_mm", "target_heavy_rainfall"],
        parse_dates=["date"],
    )
    daily = df.groupby("date").agg(
        regional_max=("regional_max_rainfall_mm", "first"),
        any_heavy_next_day=("target_heavy_rainfall", "max"),
    )
    daily["doy"] = daily.index.day_of_year
    return daily


def tool_check_history(ctx: dict, **_) -> dict:
    if "weather" not in ctx:
        raise RuntimeError("No live weather loaded yet. Call get_live_weather first.")

    daily = _history()
    as_of = pd.Timestamp(ctx["weather"]["as_of"])
    gap = (daily["doy"] - as_of.day_of_year + 183) % 366 - 183
    season = daily[gap.abs() <= HISTORY_WINDOW_DAYS]
    current = ctx["weather"]["regional_max_mm"]

    result = {
        "comparison_window": f"within {HISTORY_WINDOW_DAYS} days of this date in the year, "
                             f"{daily.index.min().year}-{daily.index.max().year} (IMD record)",
        "days_compared": int(len(season)),
        "share_of_comparable_days_at_least_this_wet_pct": round(float((season["regional_max"] >= current).mean() * 100), 1),
        "historical_share_of_days_followed_by_35mm_cell_pct": round(float(season["any_heavy_next_day"].mean() * 100), 2),
        "historical_heavy_days_count": int(season["any_heavy_next_day"].sum()),
    }
    ctx["history"] = result
    return result


TOOL_FUNCTIONS = {
    "get_live_weather": tool_get_live_weather,
    "predict_risk": tool_predict_risk,
    "check_history": tool_check_history,
}

TOOL_SCHEMAS = [
    {
        "name": "get_live_weather",
        "description": (
            "Fetch real, current daily rainfall for the 12 Delhi grid cells from Open-Meteo and build "
            "the model's input features. Must be called first. May fail if the API is down; if it fails, "
            "you may retry once."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "predict_risk",
        "description": (
            "Run the deployed champion ML model on the live features for all 12 cells and return the "
            "peak next-day heavy-rainfall score, flagged cell count, and overall risk level."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_history",
        "description": (
            "Compare the current regional rainfall with the same season in the 2000-2024 IMD record, "
            "including how often heavy-rain days followed. Use it to put the prediction in context."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _execute(name: str, args: dict, ctx: dict):
    """Run a tool; never raise. Returns (output, ok)."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool '{name}'."}, False
    try:
        return func(ctx, **(args or {})), True
    except (LiveDataError, RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}, False


# --------------------------------------------------
# Advisory (rule-based writer, also the fallback for the LLM path)
# --------------------------------------------------

_EN_ACTIONS = {
    "LOW": "No special precautions are needed. Keep following official IMD updates.",
    "ELEVATED": "Stay alert: avoid waterlogging-prone underpasses and low-lying roads where possible, "
                "and check official IMD and Delhi authority updates.",
    "HIGH": "Avoid low-lying roads and underpasses, postpone non-essential travel, keep phones charged, "
            "and follow official IMD and Delhi authority warnings.",
}
_HI_LEVEL = {"LOW": "कम", "ELEVATED": "बढ़ा हुआ", "HIGH": "उच्च"}
_HI_ACTIONS = {
    "LOW": "किसी विशेष सावधानी की आवश्यकता नहीं है। IMD के आधिकारिक अपडेट देखते रहें।",
    "ELEVATED": "सतर्क रहें: जलभराव वाले अंडरपास और निचले इलाकों की सड़कों से यथासंभव बचें, "
                "और IMD तथा दिल्ली प्रशासन के आधिकारिक अपडेट देखें।",
    "HIGH": "निचले इलाकों की सड़कों और अंडरपास से बचें, गैर-ज़रूरी यात्रा टालें, फ़ोन चार्ज रखें, "
            "और IMD तथा दिल्ली प्रशासन की आधिकारिक चेतावनियों का पालन करें।",
}


def rule_based_advisory(ctx: dict) -> dict:
    risk, weather = ctx["risk"], ctx["weather"]
    level = risk["level"]
    nwp = weather["nwp_target_regional_max_mm"]
    nwp_en = "" if nwp is None else f" The weather forecast model shows a regional maximum of {nwp:.1f} mm for that day."
    nwp_hi = "" if nwp is None else f" मौसम पूर्वानुमान मॉडल उस दिन के लिए क्षेत्रीय अधिकतम {nwp:.1f} मिमी दर्शाता है।"
    hist = ctx.get("history")
    hist_en = "" if not hist else (
        f" Historically, {hist['historical_share_of_days_followed_by_35mm_cell_pct']}% of days in this season "
        f"were followed by a 35 mm+ cell."
    )
    hist_hi = "" if not hist else (
        f" ऐतिहासिक रूप से इस मौसम के {hist['historical_share_of_days_followed_by_35mm_cell_pct']}% दिनों के बाद "
        f"किसी ग्रिड सेल में 35 मिमी से अधिक बारिश हुई।"
    )
    english = (
        f"Delhi heavy-rainfall risk for {weather['target_date']}: {level}. "
        f"The model's peak score is {risk['peak_score_pct']}% (alert threshold {risk['decision_threshold_pct']}%), "
        f"with {risk['cells_flagged']} of {risk['cells_total']} grid cells flagged for {risk['rainfall_threshold_mm']} mm or more."
        f"{nwp_en}{hist_en} {_EN_ACTIONS[level]} "
        "This is a research prototype, not an official warning."
    )
    hindi = (
        f"{weather['target_date']} के लिए दिल्ली में भारी बारिश का जोखिम: {_HI_LEVEL[level]}। "
        f"मॉडल का सर्वोच्च स्कोर {risk['peak_score_pct']}% है (चेतावनी सीमा {risk['decision_threshold_pct']}%), "
        f"और {risk['cells_total']} में से {risk['cells_flagged']} ग्रिड सेल में {risk['rainfall_threshold_mm']} मिमी या अधिक बारिश का संकेत है।"
        f"{nwp_hi}{hist_hi} {_HI_ACTIONS[level]} "
        "यह एक शोध प्रोटोटाइप है, आधिकारिक चेतावनी नहीं।"
    )
    return {"english": english, "hindi": hindi}


# --------------------------------------------------
# Planners
# --------------------------------------------------

def _run_rules(ctx: dict, steps: list, notes: list) -> None:
    out, ok = _execute("get_live_weather", {}, ctx)
    steps.append({"kind": "tool", "tool": "get_live_weather", "ok": ok, "output": out})
    if not ok:  # one retry, then stop: no data means no honest prediction
        notes.append("Live weather fetch failed; retrying once.")
        out, ok = _execute("get_live_weather", {}, ctx)
        steps.append({"kind": "tool", "tool": "get_live_weather", "ok": ok, "output": out})
        if not ok:
            raise LiveDataError(out["error"])

    out, ok = _execute("predict_risk", {}, ctx)
    steps.append({"kind": "tool", "tool": "predict_risk", "ok": ok, "output": out})
    if not ok:
        raise RuntimeError(out["error"])

    out, ok = _execute("check_history", {}, ctx)
    steps.append({"kind": "tool", "tool": "check_history", "ok": ok, "output": out})
    if not ok:
        notes.append("Historical comparison unavailable; advisory written without it.")


SYSTEM_PROMPT = """You are TerraMind's climate-risk agent for Delhi. Work only from tool results.
Plan: (1) call get_live_weather, (2) call predict_risk, (3) call check_history, then write the advisory.
If a tool returns an error, retry it once; if it still fails, continue with what you have and say what is missing.
If get_live_weather cannot succeed, do not invent any numbers: say live data is unavailable.
Final answer: ONLY a JSON object {"english": "...", "hindi": "..."} (Hindi in Devanagari).
Each field: at most 90 words, plain language for a non-expert, state the risk level and the target date,
quote only numbers that appeared in tool results, name the data source and whether it came from cache,
give one or two practical precautions matching the level, and say this is a research prototype that does
not replace official IMD warnings."""

USER_PROMPT = "Assess next-day heavy-rainfall risk for Delhi now and write the advisory."


def _run_llm(api_key: str, model: str, ctx: dict, steps: list, notes: list) -> dict | None:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": USER_PROMPT}]
    final_text = ""

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=model, max_tokens=1500, system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason != "tool_use":
            final_text = text
            break
        if text:
            steps.append({"kind": "reasoning", "text": text})

        results = []
        for block in response.content:
            if block.type == "tool_use":
                out, ok = _execute(block.name, dict(block.input or {}), ctx)
                steps.append({"kind": "tool", "tool": block.name, "ok": ok, "output": out})
                results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps(out), "is_error": not ok,
                })
        messages.append({"role": "user", "content": results})

    match = re.search(r"\{.*\}", final_text, re.DOTALL)
    if not match:
        notes.append("LLM did not return a valid advisory; used the rule-based writer.")
        return None
    try:
        parsed = json.loads(match.group(0))
        if parsed.get("english") and parsed.get("hindi"):
            return {"english": parsed["english"], "hindi": parsed["hindi"]}
    except json.JSONDecodeError:
        pass
    notes.append("LLM advisory could not be parsed; used the rule-based writer.")
    return None


# --------------------------------------------------
# Public entry point
# --------------------------------------------------

def run_agent(api_key: str | None = None, model: str | None = None) -> dict:
    """
    Run the full workflow. Returns:
      {mode, steps, notes, advisory{english,hindi}, risk, weather, history, scores, evidence_ok}
    Raises LiveDataError if live data is unavailable (no honest prediction possible).
    """
    ctx: dict = {}
    steps: list = []
    notes: list = []
    advisory = None
    mode = "rules"

    if api_key:
        try:
            advisory = _run_llm(api_key, model or DEFAULT_MODEL, ctx, steps, notes)
            mode = "llm" if advisory else "rules"
        except Exception as exc:  # network, auth, model-id errors: degrade, don't crash
            notes.append(f"LLM planner unavailable ({type(exc).__name__}); switched to the rule-based planner.")
            ctx.clear()
            steps.clear()

    if "risk" not in ctx:  # LLM path not used, failed, or skipped a tool: run the rule planner
        for key in ("fetch", "features", "weather", "scores", "risk", "history"):
            ctx.pop(key, None)
        advisory = None
        mode = "rules"
        _run_rules(ctx, steps, notes)

    if advisory is None:
        advisory = rule_based_advisory(ctx)

    return {
        "mode": mode,
        "steps": steps,
        "notes": notes,
        "advisory": advisory,
        "risk": ctx["risk"],
        "weather": ctx["weather"],
        "history": ctx.get("history"),
        "scores": ctx["scores"],
        "fetch": ctx["fetch"],
        "source": SOURCE_NAME,
    }