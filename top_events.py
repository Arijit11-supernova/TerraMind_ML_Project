import pandas as pd
from src.models.predict_rainfall import predict_rainfall

d = pd.read_csv("data/processed/delhi_rainfall_features.csv")
top = d[d.target_heavy_rainfall == 1].sort_values("next_day_rainfall_mm", ascending=False).head(20)

rows = []
for _, r in top.iterrows():
    res = predict_rainfall(r.to_dict())
    rows.append((r["date"], r["latitude"], r["longitude"], round(r["next_day_rainfall_mm"]),
                 round(res["probability"] * 100, 1), res["predicted_heavy_rainfall"]))

out = pd.DataFrame(rows, columns=["input_date", "lat", "lon", "next_day_mm", "score_pct", "flagged"])
print(out.to_string(index=False))
print("flagged:", out.flagged.sum(), "of", len(out))