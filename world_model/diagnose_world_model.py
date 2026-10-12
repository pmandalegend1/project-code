#!/usr/bin/env python3
"""
diagnose_world_model.py -- does the graph learn the formation matchup, or the
game state?

Why this is the question
------------------------
The world model beats a linear baseline on ppda (0.493 vs 0.257) and edges it
on xg_delta and possession. Read alone, that says the bipartite matchup graph
is doing its job. But every model here also sees scoreline, minute and PPDA
bucket, and those are shared by all 22 nodes -- they say nothing about which
formation is playing which. If the predictions come from those, the graph is
decoration and the headline correlations are about game state, not tactics.

A swap-consistency check first raised this: possession(A,B) + possession(B,A)
should be 100 and came out near 140, which cannot happen if the two formations
are driving the answer.

The test
--------
Three predictions for each target, on the same trained model:

  real          real windows, real contexts. This is the number the paper
                would otherwise report.
  formations    context pinned to one fixed value, formations varied over every
                pair in the vocabulary. The spread here is ALL the model gets
                from the matchup.
  context       formations pinned to one pair, context varied over real windows.
                The spread here is all the model gets from game state.

If sd(formations) is near zero while sd(context) is large, the graph is not
being used, whatever the headline correlation says. The same decomposition is
run on the ridge baseline, so the comparison is like for like.

    python diagnose_world_model.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge

from models.world_model.graph_builder import build_bipartite_graph
from world_model_adapter import TrainedWorldModel

TARGETS = ["xg_delta", "possession_pct", "ppda", "shots_on_target"]


def predict_rows(wm, rows, rms, sty):
    out = []
    with torch.no_grad():
        for _, r in rows.iterrows():
            ctx = {
                "minute": r.get("minute_bucket", 45),
                "scoreline_home": r.get("scoreline_home", 0),
                "scoreline_away": r.get("scoreline_away", 0),
                "ppda": r.get("ppda", None),
                "home_team_id": r.get("home_team_id"),
                "away_team_id": r.get("away_team_id"),
            }
            g = build_bipartite_graph(int(r.home_formation), int(r.away_formation),
                                      ctx, rms, sty, wm.vocab)
            o = wm.model(g.to(wm.device)).reshape(-1).cpu().numpy()
            out.append([float(o[i]) * wm.norm[t]["std"] + wm.norm[t]["mean"]
                        for i, t in enumerate(TARGETS)])
    return np.array(out)


def ridge_models(train, vocab):
    """Same features as baseline_world_model.py, refit here for the decomposition."""
    idx = {c: i for i, c in enumerate(vocab)}

    def feats(df):
        n = len(df)
        h = np.zeros((n, len(vocab)), np.float32)
        a = np.zeros((n, len(vocab)), np.float32)
        for r, (hf, af) in enumerate(zip(df["home_formation"], df["away_formation"])):
            if int(hf) in idx:
                h[r, idx[int(hf)]] = 1
            if int(af) in idx:
                a[r, idx[int(af)]] = 1
        ctx = np.column_stack([
            df["scoreline_home"].astype(float) - df["scoreline_away"].astype(float),
            df["minute_bucket"].astype(float),
            df["period"].astype(float) if "period" in df.columns else np.ones(n),
        ]).astype(np.float32)
        return np.hstack([h, a, ctx])

    models = {}
    X = feats(train)
    for t in TARGETS:
        y = train[t].astype(float).values
        # ppda is NaN where a window had no defensive action; training imputes
        # with the train mean, so the baseline must do the same or it cannot
        # be fitted at all.
        ok = np.isfinite(y)
        models[t] = Ridge(alpha=1.0).fit(X[ok], y[ok])
    return models, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/world_model.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/world_model_best.pt")
    ap.add_argument("--n-rows", type=int, default=300)
    ap.add_argument("--out", default="results/world_model_diagnosis.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    wm = TrainedWorldModel(checkpoint_path=args.checkpoint)
    w = pd.read_parquet(cfg["stable_windows_path"])
    rms = pd.read_parquet(cfg["role_matchup_stats_path"])
    sty = pd.read_parquet(cfg["style_clusters_path"])
    vocab = sorted(wm.vocab)

    w["ppda"] = w["ppda"].fillna(w["ppda"].mean())
    rows = w.sample(min(args.n_rows, len(w)), random_state=0).reset_index(drop=True)

    # --- 1. real windows -----------------------------------------------------
    real = predict_rows(wm, rows, rms, sty)

    # --- 2. formations varied, context pinned --------------------------------
    pinned = rows.iloc[[0]].copy()
    pairs = [(a, b) for a in vocab for b in vocab if a != b]
    form_rows = pd.concat([pinned] * len(pairs), ignore_index=True)
    form_rows["home_formation"] = [a for a, _ in pairs]
    form_rows["away_formation"] = [b for _, b in pairs]
    form = predict_rows(wm, form_rows, rms, sty)

    # --- 3. context varied, formations pinned --------------------------------
    ctx_rows = rows.copy()
    ctx_rows["home_formation"] = int(pinned["home_formation"].iloc[0])
    ctx_rows["away_formation"] = int(pinned["away_formation"].iloc[0])
    ctx = predict_rows(wm, ctx_rows, rms, sty)

    # --- ridge, same decomposition ------------------------------------------
    rmods, feats = ridge_models(w, vocab)
    r_real = np.column_stack([rmods[t].predict(feats(rows)) for t in TARGETS])
    r_form = np.column_stack([rmods[t].predict(feats(form_rows)) for t in TARGETS])
    r_ctx = np.column_stack([rmods[t].predict(feats(ctx_rows)) for t in TARGETS])

    print(f"{len(rows)} real windows | {len(pairs)} formation pairs "
          f"| context pinned at minute {int(pinned['minute_bucket'].iloc[0])}, "
          f"score {int(pinned['scoreline_home'].iloc[0])}-{int(pinned['scoreline_away'].iloc[0])}\n")

    summary = {}
    header = f"{'target':<17}{'model':<8}{'sd real':>10}{'sd formations':>16}{'sd context':>13}{'form share':>13}"
    print(header)
    print("-" * len(header))
    for i, t in enumerate(TARGETS):
        summary[t] = {}
        for name, R, F, C in (("gnn", real, form, ctx), ("ridge", r_real, r_form, r_ctx)):
            sd_r, sd_f, sd_c = R[:, i].std(), F[:, i].std(), C[:, i].std()
            share = sd_f / (sd_f + sd_c) if (sd_f + sd_c) > 0 else float("nan")
            summary[t][name] = {"sd_real": float(sd_r), "sd_formations": float(sd_f),
                                "sd_context": float(sd_c), "formation_share": float(share)}
            print(f"{t if name == 'gnn' else '':<17}{name:<8}{sd_r:>10.4f}{sd_f:>16.4f}"
                  f"{sd_c:>13.4f}{share:>12.1%}")
        actual_sd = float(rows[t].astype(float).std())
        summary[t]["actual_sd"] = actual_sd
        print(f"{'':<17}{'(data)':<8}{actual_sd:>10.4f}\n")

    print("'form share' is sd(formations) / (sd(formations) + sd(context)).")
    print("Near 0 means the model ignores the matchup and predicts from game state.")

    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
