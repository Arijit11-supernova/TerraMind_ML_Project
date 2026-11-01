import pandas as pd
from src.models.predict_rainfall import load_champion_model

art = load_champion_model()
d = pd.read_csv("data/processed/delhi_rainfall_features.csv", parse_dates=["date"])
d["score"] = art["model"].predict_proba(d[art["features"]])[:, 1]

rows = []
for period in ["validation", "test"]:
    p = d[d.period == period]
    events = int(p.target_heavy_rainfall.sum())
    for t in [0.94, 0.80, 0.60, 0.47, 0.30]:
        flagged = p.score >= t
        hits = int((flagged & (p.target_heavy_rainfall == 1)).sum())
        rows.append({
            "period": period,
            "threshold_pct": int(t * 100),
            "alert_rate_pct": round(flagged.mean() * 100, 2),
            "events_caught": f"{hits}/{events}",
            "recall_pct": round(hits / events * 100, 1),
            "precision_pct": round(hits / max(int(flagged.sum()), 1) * 100, 1),
        })
print(pd.DataFrame(rows).to_string(index=False))