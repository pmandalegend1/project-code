#!/usr/bin/env python3
"""
baseline_world_model.py -- what the bipartite GNN has to beat.

A validation correlation is uninterpretable on its own. r = 0.46 for
possession share sounds like the graph learned something about formations;
it may equally be that possession share is largely predictable from the
scoreline and the clock, which any model would pick up, or from the two
formation codes treated as plain categories, with nothing graph-shaped about
it. Without these baselines the world model's numbers cannot be read.

Three references, all on exactly the split train_world_model.py uses:

  mean       predict the training mean of each target. r = 0 by construction;
             its MSE is the variance to beat.
  ridge      one-hot home formation, one-hot away formation, scoreline,
             minute, period, both teams' style clusters. Linear, no
             interaction between the two formations.
  boosting   the same features, gradient boosted. Learns formation-by-
             formation interactions, which is the thing the bipartite graph
             is supposed to be for -- just without the graph.

If the GNN does not beat boosting, the graph structure is not earning its
place and the paper should say so.

    python baseline_world_model.py --config configs/world_model.yaml
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

TARGETS = ["xg_delta", "possession_pct", "ppda", "shots_on_target"]


def temporal_split(w, cfg):
    """Same rule as train_world_model.temporal_split, kept in step by hand."""
    meta = w.dropna(subset=["competition_id", "season_id"])
    pairs = set(map(tuple, meta[["competition_id", "season_id"]].drop_duplicates().values))
    test_keys, train_keys = set(), set()
    for cid in sorted({c for c, _ in pairs}):
        seasons = sorted(s for c, s in pairs if c == cid)
        if len(seasons) > 1:
            test_keys.add((cid, seasons[-1]))
            train_keys |= {(cid, s) for s in seasons[:-1]}
        else:
            train_keys.add((cid, seasons[0]))
    key = list(zip(w["competition_id"], w["season_id"]))
    test_mask = np.array([k in test_keys for k in key])
    return w[~test_mask].reset_index(drop=True), w[test_mask].reset_index(drop=True)


def featurise(df, vocab, style_map):
    n = len(df)
    idx = {c: i for i, c in enumerate(vocab)}
    home = np.zeros((n, len(vocab)), dtype=np.float32)
    away = np.zeros((n, len(vocab)), dtype=np.float32)
    for r, (hf, af) in enumerate(zip(df["home_formation"], df["away_formation"])):
        if int(hf) in idx:
            home[r, idx[int(hf)]] = 1.0
        if int(af) in idx:
            away[r, idx[int(af)]] = 1.0

    ctx = np.column_stack([
        df["scoreline_home"].astype(float) - df["scoreline_away"].astype(float),
        df["minute_bucket"].astype(float),
        df["period"].astype(float) if "period" in df.columns else np.ones(n),
        df["home_team_id"].map(style_map).fillna(0).astype(float),
        df["away_team_id"].map(style_map).fillna(0).astype(float),
    ]).astype(np.float32)
    return np.hstack([home, away, ctx])


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/world_model.yaml")
    ap.add_argument("--out", default="results/world_model_baselines.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    w = pd.read_parquet(cfg["stable_windows_path"])
    vocab = json.load(open(cfg["formation_vocab_path"]))["vocab"]
    sty = pd.read_parquet(cfg["style_clusters_path"])
    style_map = dict(zip(sty["team_id"].astype(float), sty["style_cluster"].astype(int)))

    train, test = temporal_split(w, cfg)
    ppda_mean = train["ppda"].mean()
    for d in (train, test):
        d["ppda"] = d["ppda"].fillna(ppda_mean)
    print(f"train {len(train):,} | test {len(test):,} windows "
          f"({train['match_id'].nunique():,} / {test['match_id'].nunique():,} matches)")

    Xtr = featurise(train, vocab, style_map)
    Xte = featurise(test, vocab, style_map)

    results = {}
    print(f"\n{'target':<17}{'model':<11}{'test MSE':>11}{'test r':>9}")
    for t in TARGETS:
        ytr = train[t].astype(float).values
        yte = test[t].astype(float).values
        results[t] = {}

        pred = np.full_like(yte, ytr.mean())
        results[t]["mean"] = {"mse": float(mean_squared_error(yte, pred)), "r": 0.0}

        ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
        pr = ridge.predict(Xte)
        results[t]["ridge"] = {"mse": float(mean_squared_error(yte, pr)), "r": pearson(yte, pr)}

        gb = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, ytr)
        pg = gb.predict(Xte)
        results[t]["boosting"] = {"mse": float(mean_squared_error(yte, pg)), "r": pearson(yte, pg)}

        for name in ("mean", "ridge", "boosting"):
            m = results[t][name]
            print(f"{t if name == 'mean' else '':<17}{name:<11}{m['mse']:>11.4f}{m['r']:>9.3f}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("\nRead the GNN's per-target val_mse and val_r against the 'boosting' row.\n"
          "Beating 'mean' only says the target is not constant.")


if __name__ == "__main__":
    main()
