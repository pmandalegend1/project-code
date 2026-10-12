"""
fit_stability_critic.py

Fit the logistic stability critic used by run_planner.load_stability_critic:
    P(concede) = sigmoid(a + b * delta_struct)

Produces six files in results/:
    stability_logistic.json              (fit on all labels)
    stability_logistic_excl0.json        (fit without fold 0's matches)
    stability_logistic_excl1.json
    stability_logistic_excl2.json
    stability_logistic_excl3.json
    stability_logistic_excl4.json

Same coefficients model_1's train_stability.py would produce: same labels,
same structural_distance feature, same 5-fold match-grouped split.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from critics import structural_distance  # noqa: E402

LABELS = os.path.join(HERE, "data", "processed", "stability_labels.parquet")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet(LABELS)
print(f"columns: {list(df.columns)}")
print(f"rows:    {len(df)}")

for col in ("f_src", "f_tgt", "conceded", "match_id"):
    if col not in df.columns:
        raise SystemExit(f"stability_labels.parquet is missing column: {col}")

delta = df.apply(
    lambda r: structural_distance(int(r["f_src"]), int(r["f_tgt"])),
    axis=1,
).to_numpy(dtype=float)
X = delta.reshape(-1, 1)
y = df["conceded"].to_numpy(dtype=int)
print(f"conceded rate: {y.mean():.4f}")
print(f"delta_struct:  mean {delta.mean():.4f}  sd {delta.std():.4f}")

def fit_one(X_, y_, tag):
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_, y_)
    coef = {"intercept": float(lr.intercept_[0]), "coef": float(lr.coef_[0, 0])}
    with open(os.path.join(OUT, f"stability_logistic{tag}.json"), "w") as f:
        json.dump(coef, f, indent=2)
    print(f"  {tag or 'main':<10} intercept {coef['intercept']:+.4f}  coef {coef['coef']:+.4f}")
    return coef

print("\nfitting...")
fit_one(X, y, "")

groups = df["match_id"].to_numpy()
gkf = GroupKFold(n_splits=5)
for k, (tr_idx, _) in enumerate(gkf.split(X, y, groups)):
    fit_one(X[tr_idx], y[tr_idx], f"_excl{k}")

print("\nwrote 6 files to", OUT)