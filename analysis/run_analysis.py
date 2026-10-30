"""Re-check the CLV model and the segments, and export docs/analysis.json.

Run from the repo root:  python analysis/run_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "data" / "shopping_behavior_updated.csv")
AMT, PREV = "Purchase Amount (USD)", "Previous Purchases"
d["CLV"] = d[AMT] * d[PREV]
cv = KFold(5, shuffle=True, random_state=42)

X_all = pd.get_dummies(d.drop(columns=["Customer ID", "CLV"]), drop_first=True, dtype=float)
X_fair = X_all.drop(columns=[AMT, PREV])


def r2(model, X):
    return round(float(cross_val_score(model, X, d.CLV, cv=cv, scoring="r2").mean()), 4)


models = {
    "with_formula_inputs": {"label": "Gradient boosting, all columns (as in the notebook)",
                            "r2": r2(HistGradientBoostingRegressor(random_state=0), X_all)},
    "without_formula_inputs": {"label": "Gradient boosting, without purchase amount and previous purchases",
                               "r2": r2(HistGradientBoostingRegressor(random_state=0), X_fair)},
    "linear_without": {"label": "Linear regression, without those two columns",
                       "r2": r2(LinearRegression(), X_fair)},
}

NUM = ["Age", AMT, "Review Rating", PREV]
corr = d[NUM].corr().round(3)
max_abs_corr = float(corr.where(~np.eye(len(NUM), dtype=bool)).abs().max().max())

scaler = StandardScaler().fit(d[NUM])
Z = scaler.transform(d[NUM])
sil = {k: round(float(silhouette_score(Z, KMeans(k, n_init=10, random_state=42).fit_predict(Z))), 3) for k in range(2, 7)}
km = KMeans(3, n_init=10, random_state=42).fit(Z)
d["segment"] = km.labels_
prof = d.groupby("segment").agg(n=("CLV", "size"), age=("Age", "mean"), amount=(AMT, "mean"),
                                 rating=("Review Rating", "mean"), prev=(PREV, "mean"), clv=("CLV", "mean"))
order = prof.sort_values("clv", ascending=False).index.tolist()
names = {order[0]: "Older, higher spend per purchase", order[1]: "Younger, moderate spend", order[2]: "Older, lower spend per purchase"}

out = {
    "rows": int(len(d)),
    "clv_definition": "Purchase amount (USD) x previous purchases",
    "clv_mean": round(float(d.CLV.mean()), 2),
    "clv_median": round(float(d.CLV.median()), 2),
    "models": models,
    "max_abs_correlation": round(max_abs_corr, 3),
    "silhouette": sil,
    "scaler": {"features": NUM, "mean": [round(float(x), 5) for x in scaler.mean_], "scale": [round(float(x), 5) for x in scaler.scale_]},
    "segments": [
        {"id": int(k), "name": names[k], "centroid": [round(float(x), 5) for x in km.cluster_centers_[k]],
         "n": int(prof.loc[k, "n"]), "age": round(float(prof.loc[k, "age"]), 1), "amount": round(float(prof.loc[k, "amount"]), 1),
         "rating": round(float(prof.loc[k, "rating"]), 2), "prev": round(float(prof.loc[k, "prev"]), 1), "clv": round(float(prof.loc[k, "clv"]), 0)}
        for k in order
    ],
    "notebook_claims": {"r2": 0.9996, "roi": 29.36, "cac": 50, "budget": 250000, "customers": 5000},
}
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "analysis.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("scaler", "segments")}, indent=1))
for s in out["segments"]:
    print(s)
