"""
tabular_baselines.py

Adds XGBoost + LightGBM rows to Table 1 (Formation ranking).
Uses the same cached cross-fit features and the same fold splits as
learn_ranker.py --crossfit.
"""
import os, json
from collections import defaultdict
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(HERE, "results")
OUT = os.path.join(FEAT_DIR, "tabular_baselines.json")

FEAT_COLS = ["prior", "xG", "stability", "counter_resistance"]

def rank_metrics(scores_per_dec, truth_idx_per_dec):
    t1 = t3 = 0; mrr = 0.0; n = len(scores_per_dec)
    for sc, tp in zip(scores_per_dec, truth_idx_per_dec):
        order = np.argsort(-sc)
        rank = int(np.where(order == tp)[0][0]) + 1
        if rank == 1: t1 += 1
        if rank <= 3: t3 += 1
        mrr += 1.0 / rank
    return t1/n, t3/n, mrr/n


def prep_fold(df):
    """Return per-decision groups."""
    groups = []
    for _, g in df.groupby("decision", sort=False):
        g = g.reset_index(drop=True)
        ti = g.index[g["f_tgt"] == g["truth"]].tolist()
        if not ti: continue
        groups.append((g, ti[0]))
    return groups


def evaluate_sklearn_like(model_fn, feats_by_fold, fold_k):
    """Train on folds != k, test on fold k. Lambda-rank style via pairwise."""
    tr = pd.concat([feats_by_fold[j] for j in range(5) if j != fold_k], ignore_index=True)
    te = feats_by_fold[fold_k].copy()

    # Build pairwise training data: for each decision, truth (rank higher) vs others
    X_rows, y_rows, group_sizes = [], [], []
    for _, g in tr.groupby("decision", sort=False):
        g = g.reset_index(drop=True)
        ti = g.index[g["f_tgt"] == g["truth"]].tolist()
        if not ti: continue
        truth_row = g.iloc[ti[0]]
        # score-like label: 1 for truth, 0 for others
        for i in range(len(g)):
            X_rows.append(g.loc[i, FEAT_COLS].fillna(0).values)
            y_rows.append(1 if i == ti[0] else 0)
        group_sizes.append(len(g))

    X = np.array(X_rows)
    y = np.array(y_rows)
    # LightGBM/XGBoost ranker: use group info
    model = model_fn()
    model.fit(X, y, group=group_sizes)

    # Predict on test fold
    scores_per_dec, truths = [], []
    for _, g in te.groupby("decision", sort=False):
        g = g.reset_index(drop=True)
        ti = g.index[g["f_tgt"] == g["truth"]].tolist()
        if not ti: continue
        sc = model.predict(g[FEAT_COLS].fillna(0).values)
        scores_per_dec.append(sc)
        truths.append(ti[0])
    return rank_metrics(scores_per_dec, truths)


def main():
    # Load cached features
    feats_by_fold = {}
    for k in range(5):
        p = os.path.join(FEAT_DIR, f"ranker_features_crossfit_fold{k}.parquet")
        d = pd.read_parquet(p)
        d["_fold"] = k
        feats_by_fold[k] = d

    results = {}

    # XGBoost ranker
    print("XGBoost (rank:pairwise)...")
    def xgb_model():
        return xgb.XGBRanker(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            objective="rank:pairwise", random_state=42, n_jobs=4)
    t1s, t3s, mrrs = [], [], []
    for k in range(5):
        r = evaluate_sklearn_like(xgb_model, feats_by_fold, k)
        t1s.append(r[0]); t3s.append(r[1]); mrrs.append(r[2])
        print(f"  fold {k}: top-1 {r[0]:.4f}  top-3 {r[1]:.4f}")
    results["XGBoost (rank:pairwise)"] = {
        "top1": float(np.mean(t1s)), "top3": float(np.mean(t3s)),
        "mrr": float(np.mean(mrrs)), "per_fold_top1": [float(x) for x in t1s]}
    print(f"  AVG top-1 {np.mean(t1s):.4f}  top-3 {np.mean(t3s):.4f}")

    # LightGBM ranker
    print("\nLightGBM (lambdarank)...")
    def lgb_model():
        return lgb.LGBMRanker(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            objective="lambdarank", random_state=42, n_jobs=4, verbose=-1)
    t1s, t3s, mrrs = [], [], []
    for k in range(5):
        r = evaluate_sklearn_like(lgb_model, feats_by_fold, k)
        t1s.append(r[0]); t3s.append(r[1]); mrrs.append(r[2])
        print(f"  fold {k}: top-1 {r[0]:.4f}  top-3 {r[1]:.4f}")
    results["LightGBM (lambdarank)"] = {
        "top1": float(np.mean(t1s)), "top3": float(np.mean(t3s)),
        "mrr": float(np.mean(mrrs)), "per_fold_top1": [float(x) for x in t1s]}
    print(f"  AVG top-1 {np.mean(t1s):.4f}  top-3 {np.mean(t3s):.4f}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()